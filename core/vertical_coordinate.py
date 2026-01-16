"""
垂直坐标系统
"""

import torch
import torch.nn as nn
from typing import Tuple
from configs.model_config import ModelConfig

class VerticalCoordinate(nn.Module):
    """垂直坐标系统（可学习）"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.nz = config.nz
        
        # 可学习的垂直坐标参数
        if config.learnable_physics:
            self.power = nn.Parameter(torch.tensor(config.sigma_power))
            self.a_scale = nn.Parameter(torch.tensor(config.a_scale))
        else:
            self.register_buffer('power', torch.tensor(config.sigma_power))
            self.register_buffer('a_scale', torch.tensor(config.a_scale))
        
        # 初始化 eta 层
        k = torch.arange(self.nz + 1, dtype=config.dtype, device=config.device)
        eta = k / self.nz
        self.register_buffer('eta_full', eta)
    
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算垂直坐标系数
        
        Returns:
            a_half: (nz,) - 固定气压部分（半层）
            b_half: (nz,) - 地面气压比例部分（半层）
        """
        eta = self.eta_full
        
        # b_k:  σ系数（可微分）
        b_k = eta ** self.power.abs()  # abs确保正值
        
        # a_k: 固定气压系数
        a_k = (1 - eta) * self.a_scale.abs()
        
        # 半层（用于标量）
        b_half = (b_k[:-1] + b_k[1:]) / 2
        a_half = (a_k[:-1] + a_k[1:]) / 2
        
        return a_half, b_half