"""
混合物理-AI模型
"""

import torch
import torch.nn as nn
from typing import Dict

from configs.model_config import ModelConfig
from core.dynamic_core import DynamicCore
from core.state import AtmosphericState
from models.neural_parameterization import NeuralParameterization

class HybridDynamicCore(nn.Module):
    """
    混合动力学核心（物理 + AI）
    总倾向 = 物理倾向 + AI修正
    """
    
    def __init__(self, config:  ModelConfig):
        super().__init__()
        self.config = config
        
        # 物理核心
        self.physics_core = DynamicCore(config)
        
        # AI 修正模块
        self.neural_correction = NeuralParameterization(config)
        
        # 混合权重（可学习）
        if config.learnable_physics: 
            self.alpha = nn.Parameter(torch.tensor(0.1))  # AI修正的权重
        else:
            self.register_buffer('alpha', torch.tensor(0.1))
    
    def forward(self, state: AtmosphericState) -> Dict[str, torch.Tensor]: 
        """
        计算总倾向 = 物理倾向 + AI修正
        
        Args:
            state: 当前大气状态
        
        Returns:
            total_tendencies: 总倾向
        """
        # 1. 计算物理倾向
        physics_tend = self.physics_core(state)
        
        # 2. 计算 AI 修正
        ai_corrections = self.neural_correction(state, physics_tend)
        
        # 3. 组合（加权）
        total_tend = {}
        for key in physics_tend.keys():
            if key in ai_corrections:
                # 混合：物理 + α * AI修正
                total_tend[key] = physics_tend[key] + self.alpha.abs() * ai_corrections[key]
            else:
                total_tend[key] = physics_tend[key]
        
        # 4. 记录用于分析
        if self.training:
            self.last_physics_tend = physics_tend
            self.last_ai_corrections = ai_corrections
        
        return total_tend
    
    def get_diagnostics(self) -> Dict[str, torch.Tensor]:
        """获取诊断信息（用于分析AI的贡献）"""
        if hasattr(self, 'last_physics_tend'):
            return {
                'physics_magnitude': torch.stack([
                    self.last_physics_tend['dudt'].abs().mean(),
                    self.last_physics_tend['dvdt'].abs().mean(),
                    self.last_physics_tend['dTdt'].abs().mean(),
                ]),
                'ai_magnitude': torch.stack([
                    self.last_ai_corrections['dudt'].abs().mean(),
                    self.last_ai_corrections['dvdt'].abs().mean(),
                    self.last_ai_corrections['dTdt'].abs().mean(),
                ]),
                'mixing_weight': self.alpha.abs()
            }
        return {}