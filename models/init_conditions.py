"""
初始化条件
"""

import torch
import torch.nn as nn
import math

from configs.model_config import ModelConfig
from core.grid import GridGeometry
from core.vertical_coordinate import VerticalCoordinate
from core.state import AtmosphericState

class InitializationModule(nn.Module):
    """初始化模块"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.grid = GridGeometry(config)
        self.vertical_coord = VerticalCoordinate(config)
    
    def baroclinic_wave(self) -> AtmosphericState: 
        """
        斜压波初始化（Jablonowski & Williamson 2006）
        标准动力学核心测试案例
        """
        state = AtmosphericState(self.config)
        
        # 物理参数
        u0 = 35.0          # 最大风速 m/s
        T0 = 288.0         # 参考温度 K
        gamma = 0.005      # 温度递减率 K/m
        
        lat = self.grid.LAT
        lon = self.grid.LON
        
        _, b_k = self.vertical_coord()
        
        for k in range(self.config.nz):
            sigma = b_k[k]
            
            # 纬向风（急流）
            eta = (sigma - 0.252) * math.pi / 2
            eta_t = eta if isinstance(eta, torch.Tensor) else torch.tensor(eta, device=self.config.device)
            state.u[:, :, k] = u0 * torch.cos(eta_t)**1.5 * \
                              torch.sin(2 * lat)**2
            
            # 经向风（初始为0）
            state.v[: , :, k] = 0.0
            
            # 温度分布
            height = -self.config.R * T0 / self.config.g * torch.log(sigma + 1e-8)
            T_mean = T0 - gamma * height
            
            # 纬度依赖
            T_eq = T_mean + 2.0 / 3.0 * (self.config.omega * self.config.a / self.config.R) * \
                   u0 * torch.cos(eta_t)**1.5 * (
                       -2 * torch.sin(lat)**6 * (torch.cos(lat)**2 + 1/3) + 10/63
                   )
            
            state.T[:, :, k] = T_eq
            
            # 添加扰动（激发不稳定）
            perturbation = 1.0 * torch.exp(
                -((lon - math.pi) / (math.pi/9))**2 -
                ((lat - math.pi/4) / (2*math.pi/9))**2
            ) * math.sin(math.pi * float(sigma))
            
            state.T[:, :, k] += perturbation
        
        # 比湿（简单初始化）
        for k in range(self.config.nz):
            state.q[:, :, k] = 0.01 * math.exp(-float(b_k[k]) / 0.8)
        
        # 地面气压
        state.ps[: ] = self.config.p0
        
        return state
    
    def rest_atmosphere(self) -> AtmosphericState:
        """静止大气（用于调试）"""
        state = AtmosphericState(self.config)
        
        # 温度：标准大气
        _, b_k = self.vertical_coord()
        for k in range(self.config.nz):
            sigma = b_k[k]
            T = 288.0 - 6.5 * 1000 * (1 - sigma)  # 简化的温度递减
            state.T[:, :, k] = torch.clamp(torch.tensor(T), min=180.0)
        
        # 风速为0
        state.u[:] = 0.0
        state.v[:] = 0.0
        
        # 比湿
        state.q[:] = 0.001
        
        # 地面气压
        state.ps[:] = self.config.p0
        
        return state