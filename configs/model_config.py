"""
模型配置文件
"""

from dataclasses import dataclass, field
import torch
from typing import Optional

@dataclass
class ModelConfig: 
    """模型配置参数"""
    
    # ========== 网格设置 ==========
    nx:  int = 256                    # 经向网格数
    ny: int = 128                    # 纬向网格数
    nz: int = 40                     # 垂直层数
    
    # ========== 物理域 ==========
    lon_min: float = 0.0
    lon_max: float = 360.0
    lat_min: float = -90.0
    lat_max: float = 90.0
    
    # ========== 时间设置 ==========
    dt: float = 300.0                # 时间步长（秒）
    
    # ========== 物理常数 ==========
    g:  float = 9.80665               # 重力加速度 m/s²
    R:  float = 287.05                # 干空气气体常数 J/(kg·K)
    cp: float = 1004.0               # 定压比热 J/(kg·K)
    omega: float = 7.292e-5          # 地球自转角速度 rad/s
    a:  float = 6.371e6               # 地球半径 m
    p0: float = 1e5                  # 参考气压 Pa
    
    # ========== 垂直坐标参数 ==========
    sigma_power: float = 2.5         # σ坐标幂指数
    a_scale: float = 1000.0          # 混合坐标 a 系数 (Pa)
    
    # ========== 训练相关 ==========
    learnable_physics: bool = False  # 是否学习物理参数
    use_hybrid_model: bool = False   # 是否使用混合模型
    
    # ========== 设备设置 ==========
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    dtype: torch.dtype = torch.float32
    
    # ========== ERA5 相关 ==========
    use_era5_init: bool = False      # 是否使用 ERA5 初始化
    era5_source_levels: int = 37     # ERA5 源数据层数
    
    def __post_init__(self):
        """后处理"""
        if isinstance(self.device, str):
            self.device = torch.device(self.device)