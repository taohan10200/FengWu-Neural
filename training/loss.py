"""
损失函数
"""

import torch
import torch.nn as nn
from core.state import AtmosphericState
from configs.model_config import ModelConfig

class PhysicsLoss(nn.Module):
    """物理约束损失"""
    
    def __init__(self, config:  ModelConfig, 
                 variable_weights: dict = None):
        """
        Args:
            config: 模型配置
            variable_weights: 各变量的权重
        """
        super().__init__()
        self.config = config
        
        # 默认权重
        self.weights = variable_weights or {
            'u': 1.0,
            'v': 1.0,
            'T': 2.0,  # 温度更重要
            'q': 0.5,
            'ps': 1.0
        }
        
        self.mse = nn.MSELoss()
    
    def forward(self, pred_state: AtmosphericState, 
                target_state: AtmosphericState) -> torch.Tensor:
        """
        计算多变量加权损失
        
        Args:
            pred_state: 预测状态
            target_state: 目标状态
        
        Returns:
            total_loss: 总损失
        """
        total_loss = 0.0
        
        # 各变量的 MSE 损失
        for var_name, weight in self.weights.items():
            if hasattr(pred_state, var_name) and hasattr(target_state, var_name):
                pred_var = getattr(pred_state, var_name)
                target_var = getattr(target_state, var_name)
                
                loss = self.mse(pred_var, target_var)
                total_loss += weight * loss
        
        return total_loss


class WeightedSpectralLoss(nn.Module):
    """
    谱空间损失（更关注大尺度特征）
    使用 FFT 在频谱域计算损失
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.mse = nn.MSELoss()
    
    def forward(self, pred_state:  AtmosphericState,
                target_state: AtmosphericState) -> torch.Tensor:
        """
        计算谱空间损失
        """
        total_loss = 0.0
        
        for var_name in ['u', 'v', 'T']: 
            pred_var = getattr(pred_state, var_name)
            target_var = getattr(target_state, var_name)
            
            # 对每个垂直层计算 2D FFT
            for k in range(self.config.nz):
                pred_fft = torch.fft.rfft2(pred_var[:, :, k])
                target_fft = torch.fft.rfft2(target_var[: , :, k])
                
                # 谱空间 MSE
                loss = self.mse(pred_fft.real, target_fft.real) + \
                       self.mse(pred_fft.imag, target_fft.imag)
                
                total_loss += loss
        
        return total_loss / self.config.nz


class ConservationLoss(nn.Module):
    """
    物理守恒约束损失
    - 质量守恒
    - 能量守恒
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
    
    def forward(self, pred_state: AtmosphericState,
                target_state: AtmosphericState) -> torch.Tensor:
        """
        计算守恒约束损失
        """
        # 1. 质量守恒：总质量应该相同
        pred_mass = pred_state.ps.sum()
        target_mass = target_state.ps.sum()
        mass_loss = torch.abs(pred_mass - target_mass) / target_mass
        
        # 2. 能量守恒：总能量应该近似相同
        # 动能
        pred_ke = 0.5 * (pred_state.u**2 + pred_state.v**2).sum()
        target_ke = 0.5 * (target_state.u**2 + target_state.v**2).sum()
        
        # 位能（简化）
        pred_pe = (self.config.g * pred_state.T).sum()
        target_pe = (self.config.g * target_state.T).sum()
        
        energy_loss = torch.abs((pred_ke + pred_pe) - (target_ke + target_pe)) / \
                     (target_ke + target_pe + 1e-8)
        
        return mass_loss + energy_loss


class CombinedLoss(nn.Module):
    """组合损失"""
    
    def __init__(self, config: ModelConfig,
                 mse_weight: float = 1.0,
                 spectral_weight: float = 0.1,
                 conservation_weight: float = 0.01):
        super().__init__()
        
        self.mse_loss = PhysicsLoss(config)
        self.spectral_loss = WeightedSpectralLoss(config)
        self.conservation_loss = ConservationLoss(config)
        
        self.mse_weight = mse_weight
        self.spectral_weight = spectral_weight
        self.conservation_weight = conservation_weight
    
    def forward(self, pred_state:  AtmosphericState,
                target_state: AtmosphericState) -> torch.Tensor:
        """
        计算组合损失
        """
        loss_mse = self.mse_loss(pred_state, target_state)
        loss_spectral = self.spectral_loss(pred_state, target_state)
        loss_conservation = self.conservation_loss(pred_state, target_state)
        
        total_loss = (self.mse_weight * loss_mse +
                     self.spectral_weight * loss_spectral +
                     self.conservation_weight * loss_conservation)
        
        return total_loss