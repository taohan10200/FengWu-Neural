"""
时间积分器
"""

import torch
import torch.nn as nn
from typing import Literal
from configs.model_config import ModelConfig
from core.dynamic_core import DynamicCore
from core.state import AtmosphericState

class TimeIntegrator(nn.Module):
    """时间积分模块（支持多种方案）"""
    
    def __init__(self, 
                 config: ModelConfig, 
                 dynamic_core: DynamicCore,
                 scheme: Literal['euler', 'rk4', 'leapfrog'] = 'rk4'):
        """
        Args:
            config: 模型配置
            dynamic_core:  动力学核心
            scheme: 时间积分方案
        """
        super().__init__()
        self.config = config
        self.core = dynamic_core
        self.scheme = scheme
        
        if scheme == 'leapfrog':
            self.state_past = None
            self.alpha_filter = 0.05  # Robert-Asselin filter
    
    def forward(self, state: AtmosphericState, nsteps: int = 1, return_trajectory: bool = False):
        """
        时间积分

        Args:
            state: 初始状态
            nsteps: 积分步数
            return_trajectory: 如果为 True，返回每一步的状态列表

        Returns:
            如果 return_trajectory 为 False，返回单个 AtmosphericState（最终状态）；
            否则返回 List[AtmosphericState]（按步的状态快照）
        """
        current_state = state
        trajectory = []

        for step_idx in range(nsteps):
            if self.scheme == 'euler':
                current_state = self._step_euler(current_state)
            elif self.scheme == 'rk4':
                current_state = self._step_rk4(current_state)
            elif self.scheme == 'leapfrog':
                current_state = self._step_leapfrog(current_state)
            else:
                raise ValueError(f"Unknown scheme: {self.scheme}")

            # 每步后打印风速诊断
            wind_speed = torch.sqrt(current_state.u**2 + current_state.v**2)
            print(f"Step {step_idx+1}/{nsteps}: wind min={wind_speed.min().item():.2f}, max={wind_speed.max().item():.2f}, mean={wind_speed.mean().item():.2f} m/s")

            if return_trajectory:
                trajectory.append(current_state.clone())

            # NaN/Inf 检查：若出现则抛出带诊断信息的异常
            def has_bad(x: torch.Tensor):
                return torch.isnan(x).any() or torch.isinf(x).any()

            bad_fields = []
            for name in ['u', 'v', 'T', 'q', 'ps']:
                val = getattr(current_state, name)
                if has_bad(val):
                    bad_fields.append(name)

            if bad_fields:
                stats = {}
                for name in ['u', 'v', 'T', 'q', 'ps']:
                    val = getattr(current_state, name)
                    # mask NaN/Inf
                    mask = torch.isnan(val) | torch.isinf(val)
                    if mask.all():
                        stats[name] = {'min': float('nan'), 'max': float('nan'), 'mean': float('nan'), 'std': float('nan')}
                    else:
                        good = val[~mask]
                        # move to CPU for safe python float extraction
                        g = good.detach().cpu()
                        stats[name] = {
                            'min': float(torch.min(g).item()),
                            'max': float(torch.max(g).item()),
                            'mean': float(torch.mean(g).item()),
                            'std': float(torch.std(g).item()),
                        }

                # 如果存在最近的倾向项，附加它们的量级供调试
                tend_info = {}
                if hasattr(self, '_last_tendencies') and self._last_tendencies is not None:
                    for kname, tend in self._last_tendencies.items():
                        tinfo = {}
                        for tvar in ['dudt', 'dvdt', 'dTdt', 'dqdt', 'dpsdt']:
                            if tvar in tend:
                                tv = tend[tvar]
                                mask = torch.isnan(tv) | torch.isinf(tv)
                                if mask.all():
                                    tinfo[tvar] = {'max_abs': float('nan')}
                                else:
                                    g = tv[~mask].detach().cpu()
                                    tinfo[tvar] = {'max_abs': float(torch.max(torch.abs(g)).item())}
                        # core may also return diagnostic scalars under 'diag'
                        if 'diag' in tend and isinstance(tend['diag'], dict):
                            tinfo['diag'] = {}
                            for dname, dval in tend['diag'].items():
                                try:
                                    tinfo['diag'][dname] = float(dval)
                                except Exception:
                                    tinfo['diag'][dname] = None

                        tend_info[kname] = tinfo

                if tend_info:
                    raise RuntimeError(f"Numerical instability detected at integrator step {step_idx+1}. Bad fields: {bad_fields}. Stats: {stats}. Tendencies: {tend_info}")
                else:
                    raise RuntimeError(f"Numerical instability detected at integrator step {step_idx+1}. Bad fields: {bad_fields}. Stats: {stats}")

        if return_trajectory:
            return trajectory
        return current_state
    def _step_euler(self, state: AtmosphericState) -> AtmosphericState:
        """前向 Euler"""
        dt = self.config.dt
        tend = self.core(state)
        
        new_state = state.clone()
        new_state.u = state.u + dt * tend['dudt']
        new_state.v = state.v + dt * tend['dvdt']
        new_state.T = state.T + dt * tend['dTdt']
        new_state.q = state.q + dt * tend['dqdt']
        new_state.ps = state.ps + dt * tend['dpsdt']
        
        return new_state
    
    def _step_rk4(self, state: AtmosphericState) -> AtmosphericState:
        """4阶 Runge-Kutta"""
        dt = self.config.dt
        
        # 保存初始状态
        u0 = state.u.clone()
        v0 = state.v.clone()
        T0 = state.T.clone()
        q0 = state.q.clone()
        ps0 = state.ps.clone()
        
        # k1: evaluate at original state (use clone to avoid side-effects)
        k1 = self.core(state.clone())

        # k2: evaluate at state + 0.5*dt*k1
        temp = state.clone()
        temp.u = u0 + 0.5 * dt * k1['dudt']
        temp.v = v0 + 0.5 * dt * k1['dvdt']
        temp.T = T0 + 0.5 * dt * k1['dTdt']
        temp.q = q0 + 0.5 * dt * k1['dqdt']
        temp.ps = ps0 + 0.5 * dt * k1['dpsdt']
        k2 = self.core(temp)

        # k3: evaluate at state + 0.5*dt*k2
        temp = state.clone()
        temp.u = u0 + 0.5 * dt * k2['dudt']
        temp.v = v0 + 0.5 * dt * k2['dvdt']
        temp.T = T0 + 0.5 * dt * k2['dTdt']
        temp.q = q0 + 0.5 * dt * k2['dqdt']
        temp.ps = ps0 + 0.5 * dt * k2['dpsdt']
        k3 = self.core(temp)

        # k4: evaluate at state + dt*k3
        temp = state.clone()
        temp.u = u0 + dt * k3['dudt']
        temp.v = v0 + dt * k3['dvdt']
        temp.T = T0 + dt * k3['dTdt']
        temp.q = q0 + dt * k3['dqdt']
        temp.ps = ps0 + dt * k3['dpsdt']
        k4 = self.core(temp)

        # store last tendencies for diagnostics
        try:
            self._last_tendencies = {'k1': k1, 'k2': k2, 'k3': k3, 'k4': k4}
        except Exception:
            self._last_tendencies = None
        
        # 最终更新
        state.u = u0 + dt/6 * (k1['dudt'] + 2*k2['dudt'] + 2*k3['dudt'] + k4['dudt'])
        state.v = v0 + dt/6 * (k1['dvdt'] + 2*k2['dvdt'] + 2*k3['dvdt'] + k4['dvdt'])
        state.T = T0 + dt/6 * (k1['dTdt'] + 2*k2['dTdt'] + 2*k3['dTdt'] + k4['dTdt'])
        state.q = q0 + dt/6 * (k1['dqdt'] + 2*k2['dqdt'] + 2*k3['dqdt'] + k4['dqdt'])
        state.ps = ps0 + dt/6 * (k1['dpsdt'] + 2*k2['dpsdt'] + 2*k3['dpsdt'] + k4['dpsdt'])
        
        # 物理约束（放宽范围，只防止完全非物理的值）
        state.T = torch.clamp(state.T, min=100.0, max=400.0)
        state.q = torch.clamp(state.q, min=0.0, max=0.1)
        state.ps = torch.clamp(state.ps, min=20000.0, max=110000.0)
        
        # 风速硬限制（防止爆炸）
        state.u = torch.clamp(state.u, min=-200.0, max=200.0)
        state.v = torch.clamp(state.v, min=-200.0, max=200.0)
        
        return state
    
    def _step_leapfrog(self, state: AtmosphericState) -> AtmosphericState:
        """Leap-Frog + Robert-Asselin 滤波"""
        dt = self.config.dt
        
        if self.state_past is None:
            # 第一步使用 Euler
            self.state_past = state.clone()
            return self._step_euler(state)
        
        # Leap-Frog
        tend = self.core(state)
        
        state_future = state.clone()
        state_future.u = self.state_past.u + 2 * dt * tend['dudt']
        state_future.v = self.state_past.v + 2 * dt * tend['dvdt']
        state_future.T = self.state_past.T + 2 * dt * tend['dTdt']
        state_future.q = self.state_past.q + 2 * dt * tend['dqdt']
        state_future.ps = self.state_past.ps + 2 * dt * tend['dpsdt']
        
        # Robert-Asselin 滤波
        alpha = self.alpha_filter
        state.u += alpha * (self.state_past.u - 2*state.u + state_future.u)
        state.v += alpha * (self.state_past.v - 2*state.v + state_future.v)
        state.T += alpha * (self.state_past.T - 2*state.T + state_future.T)
        state.q += alpha * (self.state_past.q - 2*state.q + state_future.q)
        state.ps += alpha * (self.state_past.ps - 2*state.ps + state_future.ps)
        
        # 更新
        self.state_past = state.clone()
        return state_future