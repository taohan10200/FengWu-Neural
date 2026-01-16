"""
大气状态变量容器
"""

import torch
from typing import Dict
from configs.model_config import ModelConfig

class AtmosphericState:
    """大气状态变量容器"""
    
    def __init__(self, config: ModelConfig, requires_grad: bool = False):
        """
        Args:
            config: 模型配置
            requires_grad:  是否需要梯度
        """
        self.config = config
        shape3d = (config.ny, config.nx, config.nz)
        shape2d = (config.ny, config.nx)
        
        device = config.device
        dtype = config.dtype
        
        # 动力学变量
        self.u = torch.zeros(shape3d, dtype=dtype, device=device, 
                           requires_grad=requires_grad)
        self.v = torch.zeros(shape3d, dtype=dtype, device=device,
                           requires_grad=requires_grad)
        self.w = torch.zeros(shape3d, dtype=dtype, device=device,
                           requires_grad=requires_grad)
        self.T = torch.zeros(shape3d, dtype=dtype, device=device,
                           requires_grad=requires_grad)
        self.q = torch.zeros(shape3d, dtype=dtype, device=device,
                           requires_grad=requires_grad)
        
        # 地面气压
        self.ps = torch.ones(shape2d, dtype=dtype, device=device,
                           requires_grad=requires_grad) * config.p0
        
        # 诊断量
        self.p = torch.zeros(shape3d, dtype=dtype, device=device)
        self.phi = torch.zeros(shape3d, dtype=dtype, device=device)
    
    def to(self, device: torch.device):
        """移动到指定设备"""
        self.u = self.u.to(device)
        self.v = self.v.to(device)
        self.w = self.w.to(device)
        self.T = self.T.to(device)
        self.q = self.q.to(device)
        self.ps = self.ps.to(device)
        self.p = self.p.to(device)
        self.phi = self.phi.to(device)
        return self
    
    def clone(self) -> 'AtmosphericState':
        """深拷贝"""
        new_state = AtmosphericState(self.config)
        new_state.u = self.u.clone()
        new_state.v = self.v.clone()
        new_state.w = self.w.clone()
        new_state.T = self.T.clone()
        new_state.q = self.q.clone()
        new_state.ps = self.ps.clone()
        new_state.p = self.p.clone()
        new_state.phi = self.phi.clone()
        return new_state
    
    def detach(self) -> 'AtmosphericState':
        """分离计算图"""
        new_state = AtmosphericState(self.config)
        new_state.u = self.u.detach()
        new_state.v = self.v.detach()
        new_state.w = self.w.detach()
        new_state.T = self.T.detach()
        new_state.q = self.q.detach()
        new_state.ps = self.ps.detach()
        new_state.p = self.p.detach()
        new_state.phi = self.phi.detach()
        return new_state
    
    def as_dict(self) -> Dict[str, torch.Tensor]:
        """转换为字典"""
        return {
            'u': self.u,
            'v': self.v,
            'w': self.w,
            'T': self.T,
            'q':  self.q,
            'ps': self.ps,
            'p': self.p,
            'phi': self.phi
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, torch.Tensor], 
                  config: ModelConfig) -> 'AtmosphericState':
        """从字典创建"""
        state = cls(config)
        for key, value in data.items():
            if hasattr(state, key):
                setattr(state, key, value)
        return state