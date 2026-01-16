"""
网格几何模块
"""

import torch
import torch.nn as nn
from configs.model_config import ModelConfig

class GridGeometry(nn.Module):
    """网格几何（支持球面坐标）"""
    
    def __init__(self, config:  ModelConfig):
        super().__init__()
        self.config = config
        
        # 经纬度网格 (degrees)
        lon = torch.linspace(config.lon_min, config.lon_max, config.nx,
                           dtype=config.dtype, device=config.device)
        lat = torch.linspace(config.lat_min, config.lat_max, config.ny,
                           dtype=config.dtype, device=config.device)
        
        # 转弧度
        lon_rad = torch.deg2rad(lon)
        lat_rad = torch.deg2rad(lat)
        
        # 2D 网格
        LAT, LON = torch.meshgrid(lat_rad, lon_rad, indexing='ij')
        
        # 注册为 buffer
        self.register_buffer('lon', lon)
        self.register_buffer('lat', lat)
        self.register_buffer('lon_rad', lon_rad)
        self.register_buffer('lat_rad', lat_rad)
        self.register_buffer('LAT', LAT)
        self.register_buffer('LON', LON)
        
        # 网格间距
        dlon = (config.lon_max - config.lon_min) / config.nx
        dlat = (config.lat_max - config.lat_min) / config.ny
        dlon_rad = torch.deg2rad(torch.tensor(dlon, device=config.device))
        dlat_rad = torch.deg2rad(torch.tensor(dlat, device=config.device))
        
        # 度量因子
        # 避免极点 dx 为 0 导致的数值不稳定性
        cos_lat = torch.cos(LAT)
        # 限制 cos_lat 的最小值为一个较大的正数以满足 CFL 条件
        # Clamp 0.1 对应约 84 度。意味着极地区域 (84-90) 的 dx 被人为放大，
        # 从而允许较大的 dt 而不违反 CFL 稳定性条件 (Gravity wave stability)
        cos_lat_clamped = torch.clamp(cos_lat, min=0.1)
        
        self.register_buffer('dx', config.a * dlon_rad * cos_lat_clamped)
        self.register_buffer('dy', config.a * dlat_rad * torch.ones_like(LAT))
        
        # 科氏参数
        self.register_buffer('f', 2 * config.omega * torch.sin(LAT))
        
        # 地图因子
        self.register_buffer('map_factor', torch.cos(LAT))
    
    def get_dsigma(self, sigma_levels: torch.Tensor) -> torch.Tensor:
        """计算σ层厚度"""
        dsigma = torch.zeros(self.config.nz, device=self.config.device)
        dsigma[0] = sigma_levels[0]
        dsigma[1:] = sigma_levels[1:] - sigma_levels[:-1]
        return dsigma