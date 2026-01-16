"""
神经网络参数化模块
"""

import torch
import torch.nn as nn
from typing import Dict

from configs.model_config import ModelConfig
from core.state import AtmosphericState

class NeuralParameterization(nn.Module):
    """
    神经网络参数化（用于修正未解析的物理过程）
    
    架构选择：
    - 简单版：MLP（逐格点）
    - 中等版：CNN（考虑局部空间相关）
    - 高级版：U-Net 或 Transformer（全局依赖）
    """
    
    def __init__(self, config: ModelConfig, 
                 architecture: str = 'mlp',
                 hidden_dim: int = 128):
        """
        Args:
            config: 模型配置
            architecture: 'mlp', 'cnn', 'unet'
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.config = config
        self.architecture = architecture
        
        if architecture == 'mlp': 
            self.net = self._build_mlp(hidden_dim)
        elif architecture == 'cnn':
            self.net = self._build_cnn(hidden_dim)
        elif architecture == 'unet': 
            self.net = self._build_unet(hidden_dim)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def _build_mlp(self, hidden_dim: int) -> nn.Module:
        """构建简单的 MLP（逐格点处理）"""
        # 输入：u, v, T, q, 物理倾向 (dudt, dvdt, dTdt)
        input_dim = 7
        # 输出：修正的倾向 (dudt, dvdt, dTdt)
        output_dim = 3
        
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # 限制输出范围
        )
    
    def _build_cnn(self, hidden_dim: int) -> nn.Module:
        """构建 CNN（考虑空间相关性）"""
        input_channels = 7  # u, v, T, q, dudt, dvdt, dTdt
        output_channels = 3  # 修正的 dudt, dvdt, dTdt
        
        return nn.Sequential(
            # 编码器
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            
            # 解码器
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, output_channels, kernel_size=1),
            nn.Tanh()
        )
    
    def _build_unet(self, hidden_dim: int) -> nn.Module:
        """构建 U-Net（更强的空间建模能力）"""
        from models.unet_blocks import UNet
        return UNet(
            in_channels=7,
            out_channels=3,
            features=[hidden_dim, hidden_dim * 2, hidden_dim * 4]
        )
    
    def forward(self, state: AtmosphericState, 
                physics_tendency: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算 AI 修正项
        
        Args:
            state: 当前状态
            physics_tendency: 物理倾向
        
        Returns:
            corrections: AI修正的倾向
        """
        if self.architecture == 'mlp':
            return self._forward_mlp(state, physics_tendency)
        elif self.architecture in ['cnn', 'unet']:
            return self._forward_cnn(state, physics_tendency)
    
    def _forward_mlp(self, state: AtmosphericState,
                    physics_tendency: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """MLP 前向传播（逐格点）"""
        # 拼接输入特征 (ny, nx, nz, 7)
        features = torch.stack([
            state.u,
            state.v,
            state.T,
            state.q,
            physics_tendency['dudt'],
            physics_tendency['dvdt'],
            physics_tendency['dTdt']
        ], dim=-1)
        
        # 通过 MLP
        corrections_raw = self.net(features)  # (ny, nx, nz, 3)
        
        # 缩放到合理范围（避免修正过大）
        scale = 1e-5  # 修正量级约为物理倾向的 1/10000
        
        return {
            'dudt': corrections_raw[.. ., 0] * scale,
            'dvdt': corrections_raw[..., 1] * scale,
            'dTdt': corrections_raw[..., 2] * scale
        }
    
    def _forward_cnn(self, state: AtmosphericState,
                    physics_tendency: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """CNN/U-Net 前向传播（考虑空间相关）"""
        ny, nx, nz = state.u.shape
        
        # 对每个垂直层独立处理（或者可以改为3D CNN）
        corrections_u = torch.zeros_like(state.u)
        corrections_v = torch.zeros_like(state.v)
        corrections_T = torch.zeros_like(state.T)
        
        for k in range(nz):
            # 准备输入 (1, 7, ny, nx)
            features = torch.stack([
                state.u[: , :, k],
                state.v[:, :, k],
                state.T[: , :, k],
                state.q[:, :, k],
                physics_tendency['dudt'][:, :, k],
                physics_tendency['dvdt'][:, :, k],
                physics_tendency['dTdt'][:, :, k]
            ], dim=0).unsqueeze(0)
            
            # 通过网络
            output = self.net(features)  # (1, 3, ny, nx)
            
            corrections_u[:, :, k] = output[0, 0]
            corrections_v[:, : , k] = output[0, 1]
            corrections_T[:, :, k] = output[0, 2]
        
        scale = 1e-5
        
        return {
            'dudt': corrections_u * scale,
            'dvdt': corrections_v * scale,
            'dTdt': corrections_T * scale
        }