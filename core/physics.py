"""
诊断物理过程
"""

import torch
import torch.nn as nn
from configs.model_config import ModelConfig
from core.vertical_coordinate import VerticalCoordinate

class DiagnosticPhysics(nn.Module):
    """诊断物理量计算"""
    
    def __init__(self, config: ModelConfig, vertical_coord: VerticalCoordinate):
        super().__init__()
        self.config = config
        self.vertical_coord = vertical_coord
    
    def compute_pressure(self, ps: torch.Tensor) -> torch.Tensor:
        """
        计算3D气压场
        """
        a_k, b_k = self.vertical_coord()
        
        # 统一处理维度
        is_3d = ps.ndim == 3
        if not is_3d: 
            ps_batch = ps.unsqueeze(0)
        else:
            ps_batch = ps
            
        batch, ny, nx = ps_batch.shape
        nz = self.config.nz
        
        p = torch.zeros(batch, ny, nx, nz, dtype=ps.dtype, device=ps.device)
        for k in range(nz):
            p[:, :, :, k] = a_k[k] + b_k[k] * ps_batch
            
        if not is_3d:
            p = p.squeeze(0)
            
        return p
    
    def compute_geopotential(self, T: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        计算位势（静力平衡）
        """
        R = self.config.R
        
        # 统一处理为 (batch, ny, nx, nz)
        is_4d = T.ndim == 4
        if not is_4d: 
            T_batch = T.unsqueeze(0)
            p_batch = p.unsqueeze(0)
        else:
            T_batch = T
            p_batch = p
            
        batch, ny, nx, nz = T_batch.shape
        phi = torch.zeros_like(T_batch)
        phi[..., -1] = 0.0
        
        for k in range(nz - 2, -1, -1):
            T_layer = (T_batch[..., k] + T_batch[..., k+1]) * 0.5
            p_ratio = (p_batch[..., k+1] + 1e-8) / (p_batch[..., k] + 1e-8)
            dphi = R * T_layer * torch.log(p_ratio)
            phi[..., k] = phi[..., k+1] + dphi
            
        if not is_4d: 
            phi = phi.squeeze(0)
            
        return phi
    
    def compute_vertical_velocity(self, u: torch.Tensor, v: torch.Tensor,
                                  ps: torch.Tensor, divergence: torch.Tensor,
                                  dsigma: torch.Tensor) -> torch.Tensor:
        """
        计算垂直速度 sigma_dot = dsigma/dt
        注意：输入divergence应为水平散度
        """
        _, b_k = self.vertical_coord()
        
        has_batch = divergence.ndim == 4
        if not has_batch: 
            divergence = divergence.unsqueeze(0)
            ps = ps.unsqueeze(0)
        
        # 计算每一层的质量散度项: ps * div_h * dsigma
        # divergence shape: (batch, ny, nx, nz)
        # ps shape: (batch, ny, nx) -> (batch, ny, nx, 1)
        # dsigma shape: (nz,) -> (1, 1, 1, nz)
        
        term = ps.unsqueeze(-1) * divergence * dsigma.view(1, 1, 1, -1)
        
        # 柱积分 (近似 -dps/dt)
        col_integral = term.sum(dim=-1, keepdim=True)
        
        # 从顶向下的累积积分 (int_0^sigma)
        partial_sum = torch.cumsum(term, dim=-1)
        
        # sigma_dot * ps = sigma * col_integral - partial_sum
        # b_k 是层中心的sigma值
        sigma_profile = b_k.view(1, 1, 1, -1)
        
        sigma_dot_ps = sigma_profile * col_integral - partial_sum
        
        # 计算 sigma_dot
        omega = sigma_dot_ps / (ps.unsqueeze(-1) + 1e-8)
        
        if not has_batch: 
            omega = omega.squeeze(0)
        
        return omega