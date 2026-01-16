"""
空间微分算子
"""

import torch
import torch.nn as nn
from typing import Tuple
from configs.model_config import ModelConfig
from core.grid import GridGeometry

class SpatialDerivatives(nn.Module):
    """空间微分算子（支持自动微分）"""
    
    def __init__(self, config: ModelConfig, grid: GridGeometry):
        super().__init__()
        self.config = config
        self.grid = grid
    
    def horizontal_gradient(self, field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算水平梯度
        
        Args:
            field: [..., ny, nx, nz] 或 [..., ny, nx]
        
        Returns:
            dfdx, dfdy: 水平导数
        """
        # 统一处理维度：确保至少有 3 维 (batch/dummy, ny, nx) 或 4 维 (batch/dummy, ny, nx, nz)
        orig_ndim = field.ndim
        if orig_ndim == 2: # (ny, nx) -> (1, ny, nx)
            field = field.unsqueeze(0)
        elif orig_ndim == 3: # (ny, nx, nz) 或 (batch, ny, nx)
            # 这里默认 (ny, nx, nz) -> (1, ny, nx, nz)
            # 因为动力核中 ps 通常是 (ny, nx)，而 u,v,T 是 (ny, nx, nz)
            # 如果是批量 ps (batch, ny, nx)，这里会变成 (1, batch, ny, nx)，后面 unpack 会出问题
            # 兼容性方案：检查 self.grid.ny 是否匹配
            if field.shape[0] == self.grid.lat.shape[0]: # 可能没有 batch
                field = field.unsqueeze(0)
        
        shape = field.shape
        if len(shape) == 4:
            batch, ny, nx, nz = shape
            has_nz = True
        else:
            batch, ny, nx = shape
            has_nz = False
        
        # x方向（经度）- 周期边界
        dfdx = torch.zeros_like(field)
        if has_nz:
            dfdx[:, :, 1:-1, :] = (field[:, :, 2:, :] - field[:, :, :-2, :]) / 2
            dfdx[:, :, 0, :] = (field[:, :, 1, :] - field[:, :, -1, :]) / 2
            dfdx[:, :, -1, :] = (field[:, :, 0, :] - field[:, :, -2, :]) / 2
        else:
            dfdx[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / 2
            dfdx[:, :, 0] = (field[:, :, 1] - field[:, :, -1]) / 2
            dfdx[:, :, -1] = (field[:, :, 0] - field[:, :, -2]) / 2
        
        dx = self.grid.dx
        if has_nz:
            dx = dx.unsqueeze(0).unsqueeze(-1)
        else:
            dx = dx.unsqueeze(0)
        dfdx = dfdx / (dx + 1e-8)
        
        # y方向（纬度）
        dfdy = torch.zeros_like(field)
        if has_nz:
            dfdy[:, 1:-1, :, :] = (field[:, 2:, :, :] - field[:, :-2, :, :]) / 2
            dfdy[:, 0, :, :] = (field[:, 1, :, :] - field[:, 0, :, :])
            dfdy[:, -1, :, :] = (field[:, -1, :, :] - field[:, -2, :, :])
        else:
            dfdy[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) / 2
            dfdy[:, 0, :] = (field[:, 1, :] - field[:, 0, :])
            dfdy[:, -1, :] = (field[:, -1, :] - field[:, -2, :])
        
        dy = self.grid.dy
        if has_nz:
            dy = dy.unsqueeze(0).unsqueeze(-1)
        else:
            dy = dy.unsqueeze(0)
        dfdy = dfdy / (dy + 1e-8)
        
        # 还原维度
        if orig_ndim == 2:
            dfdx = dfdx.squeeze(0)
            dfdy = dfdy.squeeze(0)
        elif orig_ndim == 3 and field.shape[0] == 1:
            dfdx = dfdx.squeeze(0)
            dfdy = dfdy.squeeze(0)
        
        return dfdx, dfdy
    
    def vertical_gradient(self, field: torch.Tensor, dsigma: torch.Tensor) -> torch.Tensor:
        """
        计算垂直梯度
        
        Args: 
            field: (ny, nx, nz) 或 (batch, ny, nx, nz)
            dsigma: (nz,) σ层厚度
        
        Returns:
            dfds: 垂直导数
        """
        has_batch = field.ndim == 4
        if not has_batch: 
            field = field.unsqueeze(0)
        
        batch, ny, nx, nz = field.shape
        dfds = torch.zeros_like(field)
        
        # 顶层：向前差分
        dfds[:, :, : , 0] = (field[:, :, :, 1] - field[:, :, : , 0]) / (dsigma[0] + 1e-8)
        
        # 中间层：中心差分
        for k in range(1, nz - 1):
            ds = dsigma[k-1] + dsigma[k]
            dfds[:, :, :, k] = (field[:, :, :, k+1] - field[:, :, :, k-1]) / (ds + 1e-8)
        
        # 底层：向后差分
        dfds[:, : , :, -1] = (field[:, : , :, -1] - field[:, :, : , -2]) / (dsigma[-1] + 1e-8)
        
        if not has_batch:
            dfds = dfds.squeeze(0)
        
        return dfds
    
    def divergence(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor,
                   dsigma: torch.Tensor) -> torch.Tensor:
        """计算3D散度"""
        dudx, _ = self.horizontal_gradient(u)
        _, dvdy = self.horizontal_gradient(v)
        dwds = self.vertical_gradient(w, dsigma)
        return dudx + dvdy + dwds

    def laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """
        计算水平拉普拉斯算子 (d2/dx2 + d2/dy2)
        用于数值扩散
        """
        has_batch = field.ndim == 4
        if not has_batch: 
            field = field.unsqueeze(0)
            
        # 一阶导
        dfdx, dfdy = self.horizontal_gradient(field)
        
        # 二阶导
        d2fdx2, _ = self.horizontal_gradient(dfdx)
        _, d2fdy2 = self.horizontal_gradient(dfdy)
        
        lap = d2fdx2 + d2fdy2
        
        if not has_batch:
            lap = lap.squeeze(0)
            
        return lap
    
    def vorticity(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """计算垂直涡度"""
        _, dudy = self.horizontal_gradient(u)
        dvdx, _ = self.horizontal_gradient(v)
        return dvdx - dudy