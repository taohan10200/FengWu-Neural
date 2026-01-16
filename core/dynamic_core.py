"""
动力学核心
"""

import torch
import torch.nn as nn
from typing import Dict
from configs.model_config import ModelConfig
from core.grid import GridGeometry
from core.vertical_coordinate import VerticalCoordinate
from core.state import AtmosphericState
from core.derivatives import SpatialDerivatives
from core.physics import DiagnosticPhysics

class DynamicCore(nn.Module):
    """3D 原始方程动力学核心（PyTorch版）"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 子模块
        self.vertical_coord = VerticalCoordinate(config)
        self.grid = GridGeometry(config)
        self.derivatives = SpatialDerivatives(config, self.grid)
        self.diagnostics = DiagnosticPhysics(config, self.vertical_coord)
        
        # 可学习的物理参数
        if config.learnable_physics:
            self.g = nn.Parameter(torch.tensor(config.g))
            self.R = nn.Parameter(torch.tensor(config.R))
        else:
            self.register_buffer('g', torch.tensor(config.g))
            self.register_buffer('R', torch.tensor(config.R))
    
    def forward(self, state: AtmosphericState) -> Dict[str, torch.Tensor]: 
        """
        计算倾向项 dX/dt (完善的 3D 原始方程)
        """
        # 记录原始维度
        is_batched = state.u.ndim == 4
        
        # 1. 更新诊断量
        state.p = self.diagnostics.compute_pressure(state.ps)
        state.phi = self.diagnostics.compute_geopotential(state.T, state.p)
        
        # 2. 计算空间导数
        _, b_k = self.vertical_coord()
        dsigma = self.grid.get_dsigma(b_k)
        
        # 3. 计算水平导数
        dudx, dudy = self.derivatives.horizontal_gradient(state.u)
        dvdx, dvdy = self.derivatives.horizontal_gradient(state.v)
        dTdx, dTdy = self.derivatives.horizontal_gradient(state.T)
        dphidx, dphidy = self.derivatives.horizontal_gradient(state.phi)
        dpsdx, dpsdy = self.derivatives.horizontal_gradient(state.ps)
        
        # 4. 连续方程与垂直运动
        div_h = dudx + dvdy
        
        # 统一处理 B_k 和 dsigma 到 state 的维度 (兼容 batch)
        # 如果 is_batched 为 True, shape 是 (B, Y, X, Z), 否则是 (Y, X, Z)
        if is_batched:
            view_shape = (1, 1, 1, -1)
            sum_dim = -1
        else:
            view_shape = (1, 1, -1)
            sum_dim = -1
            
        B_k = b_k.view(*view_shape)
        DS = dsigma.view(*view_shape)
        
        # dpsdt = -integral[ div_h*ps + V*grad(ps) ] dsigma
        # 注意: ps 需要 unsqueeze(-1) 以便与 nz 维度相乘
        mass_div = (div_h * state.ps.unsqueeze(-1) + 
                    state.u * dpsdx.unsqueeze(-1) + 
                    state.v * dpsdy.unsqueeze(-1)) * DS
        dpsdt = -torch.sum(mass_div, dim=sum_dim)
        
        # sigma_dot (dsigma/dt)
        sigma_dot = self.diagnostics.compute_vertical_velocity(
            state.u, state.v, state.ps, div_h, dsigma
        )

        # 5. 动量方程 (u, v)
        f_val = self.grid.f.unsqueeze(-1)
        coriolis_u = f_val * state.v
        coriolis_v = -f_val * state.u
        
        # PGF: -grad(phi) - (RT/p) * B * grad(ps)
        # grad(p) = B * grad(ps)
        pgf_u = -dphidx - (self.R * state.T * B_k / (state.p + 1e-8)) * dpsdx.unsqueeze(-1)
        pgf_v = -dphidy - (self.R * state.T * B_k / (state.p + 1e-8)) * dpsdy.unsqueeze(-1)

        # 平流 (水平 + 垂直)
        h_adv_u = -(state.u * dudx + state.v * dudy)
        h_adv_v = -(state.u * dvdx + state.v * dvdy)
        
        dudsigma = self.derivatives.vertical_gradient(state.u, dsigma)
        dvdsigma = self.derivatives.vertical_gradient(state.v, dsigma)
        v_adv_u = -sigma_dot * dudsigma
        v_adv_v = -sigma_dot * dvdsigma
        
        # 6. 能量方程 (T)
        h_adv_T = -(state.u * dTdx + state.v * dTdy)
        dTdsigma = self.derivatives.vertical_gradient(state.T, dsigma)
        v_adv_T = -sigma_dot * dTdsigma
        
        # 绝热项
        cp = 1004.0
        adiabatic_T = (self.R * state.T / (state.p * cp + 1e-8)) * (sigma_dot * state.ps.unsqueeze(-1))

        # 7. 数值稳定项 (扩散 + 摩擦)
        # 针对大步长 (dt=900s)，需要强效稳定性措施
        
        # 纬度因子 (用于调节极地附近的扩散系数，防止违背 CFL 条件)
        cos_lat = torch.cos(self.grid.lat_rad).view(1, -1, 1, 1)
        cos_factor = torch.clamp(cos_lat**2, min=0.01) # 允许在极地降得更低

        # 7.1 散度阻尼 (Divergence Damping)
        nu_div_base = 5.0e5
        nu_div = nu_div_base * cos_factor # 同样需要随纬度减小
        
        grad_div_u, grad_div_v = self.derivatives.horizontal_gradient(div_h)
        div_damp_u = nu_div * grad_div_u
        div_damp_v = nu_div * grad_div_v

        # 7.2 普通扩散 (Horizontal Diffusion)
        nu_h_base = 1.0e5
        nu_h_eff = nu_h_base * cos_factor
        
        diffusion_u = nu_h_eff * self.derivatives.laplacian(state.u)
        diffusion_v = nu_h_eff * self.derivatives.laplacian(state.v)
        diffusion_T = nu_h_eff * self.derivatives.laplacian(state.T)
        
        r_fric = 1e-5 # 稍微增加摩擦
        friction_u = -r_fric * state.u
        friction_v = -r_fric * state.v

        # 组合
        dudt = h_adv_u + v_adv_u + coriolis_u + pgf_u + diffusion_u + friction_u + div_damp_u
        dvdt = h_adv_v + v_adv_v + coriolis_v + pgf_v + diffusion_v + friction_v + div_damp_v
        dTdt = h_adv_T + v_adv_T + adiabatic_T + diffusion_T
        
        # 7.3 极地滤波 (Polar Filtering) - 关键！
        # 必须对倾向项进行纬向傅里叶滤波，消除极地高频不稳定波
        dudt = self._apply_polar_filter(dudt)
        dvdt = self._apply_polar_filter(dvdt)
        dTdt = self._apply_polar_filter(dTdt)
        dpsdt = self._apply_polar_filter(dpsdt)

        # 8. 辅助限制 (防止发散)
        # 适当放宽限制，让滤波起作用
        # 但针对 dt=900s，必须严格限制单步变化量
        # 0.02 m/s^2 * 900s = 18 m/s (每步允许的最大风速变化)
        dudt = torch.clamp(dudt, -0.02, 0.02)
        dvdt = torch.clamp(dvdt, -0.02, 0.02)
        # 0.01 K/s * 900s = 9 K
        dTdt = torch.clamp(dTdt, -0.01, 0.01)
        # 0.5 Pa/s * 900s = 450 Pa = 4.5 hPa
        dpsdt = torch.clamp(dpsdt, -0.5, 0.5)
        
        dqdt = torch.zeros_like(state.q)

        return {
            'dudt': dudt, 'dvdt': dvdt, 'dTdt': dTdt, 
            'dqdt': dqdt, 'dpsdt': dpsdt,
            'diag': {'sigdot_max': float(torch.max(torch.abs(sigma_dot)))}
        }

    def _apply_polar_filter(self, field: torch.Tensor) -> torch.Tensor:
        """
        对场进行纬向(Zonal) FFT 滤波，根据纬度切断高波数
        """
        orig_shape = field.shape
        ndim = field.ndim
        ny = self.grid.lat.shape[0]
        nx = self.grid.lon.shape[0]

        # 统一 Reshape 到 (B, Y, X, Z)
        # 如果 Z 不存在，则设为 1
        if ndim == 2: 
            # (Y, X) -> (1, Y, X, 1)
            # check dimensions
            if field.shape[0] == ny and field.shape[1] == nx:
                field_reshaped = field.view(1, ny, nx, 1)
            else:
                 # Fallback or error? Assuming (Y, X)
                 # Might be (B, N) or something else? But in dynamic core it's predictable.
                 field_reshaped = field.view(1, field.shape[0], field.shape[1], 1)
        elif ndim == 3:
            # Could be (Y, X, Z) or (B, Y, X)
            if field.shape[0] == ny and field.shape[1] == nx:
                 # (Y, X, Z)
                 field_reshaped = field.unsqueeze(0) # (1, Y, X, Z)
            else:
                 # (B, Y, X)
                 field_reshaped = field.unsqueeze(-1) # (B, Y, X, 1)
        elif ndim == 4:
            # (B, Y, X, Z)
             field_reshaped = field
        else:
            return field
        
        B, Y, X, Z = field_reshaped.shape
        device = field.device
        
        # 转换到频域: result shape (B, Y, X//2 + 1, Z)
        # dim=2 corresponds to X
        fft_coeffs = torch.fft.rfft(field_reshaped, dim=2)
        
        # ... rest of the logic ...
        lat_rad = self.grid.lat_rad.to(device) # (Y,)
        K_max = X // 2
        
        k_indices = torch.arange(fft_coeffs.shape[2], device=device).view(1, 1, -1, 1)
        
        cutoffs = (K_max * torch.cos(lat_rad)).view(1, Y, 1, 1)
        # 允许极地保留的最小波数 (避免完全平滑)
        cutoffs = torch.clamp(cutoffs, min=1.0)
        
        mask = (k_indices <= cutoffs).float()
        
        fft_filtered = fft_coeffs * mask
        
        filtered_field_reshaped = torch.fft.irfft(fft_filtered, n=X, dim=2)
        
        return filtered_field_reshaped.view(orig_shape)
        diag = {}
        try:
            diag['dudx_max'] = float(torch.max(torch.abs(dudx)).detach().cpu())
            diag['dudy_max'] = float(torch.max(torch.abs(dudy)).detach().cpu())
            diag['dvdx_max'] = float(torch.max(torch.abs(dvdx)).detach().cpu())
            diag['dvdy_max'] = float(torch.max(torch.abs(dvdy)).detach().cpu())
            diag['dphidx_max'] = float(torch.max(torch.abs(dphidx)).detach().cpu())
            diag['dphidy_max'] = float(torch.max(torch.abs(dphidy)).detach().cpu())

            # per-term magnitudes
            diag['adv_u_max'] = float(torch.max(torch.abs(advection_u)).detach().cpu())
            diag['coriolis_u_max'] = float(torch.max(torch.abs(coriolis_u)).detach().cpu())
            diag['pgf_u_max'] = float(torch.max(torch.abs(pgf_u)).detach().cpu())

            diag['adv_v_max'] = float(torch.max(torch.abs(advection_v)).detach().cpu())
            diag['coriolis_v_max'] = float(torch.max(torch.abs(coriolis_v)).detach().cpu())
            diag['pgf_v_max'] = float(torch.max(torch.abs(pgf_v)).detach().cpu())

            diag['adv_T_max'] = float(torch.max(torch.abs(advection_T)).detach().cpu())
        except Exception:
            # 如果计算过程中出错（如包含 NaN/Inf），用 NaN 占位
            for k in ['dudx_max','dudy_max','dvdx_max','dvdy_max','dphidx_max','dphidy_max',
                      'adv_u_max','coriolis_u_max','pgf_u_max','adv_v_max','coriolis_v_max','pgf_v_max',
                      'adv_T_max']:
                diag[k] = float('nan')

        return {
            'dudt': dudt,
            'dvdt': dvdt,
            'dTdt': dTdt,
            'dqdt': dqdt,
            'dpsdt':  dpsdt,
            'diag': diag
        }