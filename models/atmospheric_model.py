"""
完整的大气模型
"""

import torch
import torch.nn as nn
from typing import List, Literal

from configs.model_config import ModelConfig
from core.dynamic_core import DynamicCore
from core.state import AtmosphericState
from integrators.time_integrator import TimeIntegrator

class AtmosphericModel(nn.Module):
    """完整的大气模型（端到端）"""
    
    def __init__(self, 
                 config: ModelConfig,
                 use_hybrid:  bool = False,
                 integrator_scheme: Literal['euler', 'rk4', 'leapfrog'] = 'rk4'):
        """
        Args:
            config: 模型配置
            use_hybrid: 是否使用混合模型（物理+AI）
            integrator_scheme:  时间积分方案
        """
        super().__init__()
        self.config = config
        
        # 动力学核心
        if use_hybrid:
            from models.hybrid_model import HybridDynamicCore
            config.use_hybrid_model = True
            self.core = HybridDynamicCore(config)
        else:
            self.core = DynamicCore(config)
        
        # 时间积分器
        self.integrator = TimeIntegrator(config, self.core, scheme=integrator_scheme)
    
    def forward(self, initial_state: AtmosphericState, 
                forecast_steps: int) -> List[AtmosphericState]: 
        """
        执行预报
        
        Args:
            initial_state: 初始状态
            forecast_steps: 预报步数
        
        Returns:
            trajectory: 状态序列
        """
        trajectory = [initial_state]
        current_state = initial_state
        
        for _ in range(forecast_steps):
            current_state = self.integrator(current_state, nsteps=1)
            trajectory.append(current_state.clone())
        
        return trajectory
    
    def rollout(self, 
                initial_state: AtmosphericState,
                forecast_hours: float,
                save_interval_hours: float = 6.0) -> List[AtmosphericState]:
        """
        滚动预报
        
        Args:
            initial_state:  初始状态
            forecast_hours: 预报时长（小时）
            save_interval_hours: 保存间隔（小时）
        
        Returns:
            saved_states: 保存的状态列表
        """
        dt_seconds = self.config.dt
        total_steps = int(forecast_hours * 3600 / dt_seconds)
        save_interval = int(save_interval_hours * 3600 / dt_seconds)
        
        # 克隆初始状态以避免被修改
        saved_states = [initial_state.clone()]
        current_state = initial_state.clone()

        # 一次性调用积分器完成所有内部步（integrator 会在内部循环并可返回轨迹）
        if total_steps <= 0:
            return saved_states

        traj = self.integrator(current_state, nsteps=total_steps, return_trajectory=True)

        # traj 是按内部步顺序的状态列表（长度 == total_steps）
        for i, st in enumerate(traj, start=1):
            if i % 100 == 0:
                print(f"Step {i}/{total_steps}...", flush=True)
            if i % save_interval == 0:
                saved_states.append(st.clone())

        return saved_states
    
    def save_checkpoint(self, filepath: str, **kwargs):
        """保存模型检查点"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            **kwargs
        }
        torch.save(checkpoint, filepath)
        print(f"模型已保存到:  {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """加载模型检查点"""
        checkpoint = torch.load(filepath, map_location=self.config.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已从 {filepath} 加载")
        return checkpoint