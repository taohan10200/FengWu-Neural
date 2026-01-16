"""
ERA5 数据配置
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Dict, Optional

@dataclass
class ERA5Config:
    """ERA5 数据配置"""
    
    # ERA5 标准气压层 (hPa)
    pressure_levels: np.ndarray = field(default_factory=lambda: np.array([
        1000., 975., 950., 925., 900., 875., 850., 825., 800.,
        775., 750., 700., 650., 600., 550., 500., 450., 400.,
        350., 300., 250., 225., 200., 175., 150., 125., 100.,
        70., 50., 30., 20., 10., 7., 5., 3., 2., 1.
    ]))
    
    n_pressure_levels: int = 37
    
    # 变量名映射 (模型变量名 -> ERA5变量名)
    var_names: Dict[str, str] = field(default_factory=lambda:  {
        'u': 'u',      # 纬向风
        'v':  'v',      # 经向风
        'w': 'w',      # 垂直速度
        'T': 't',      # 温度
        'q': 'q',      # 比湿
        'ps': 'sp'     # 地面气压
    })
    
    # 数据范围检查
    valid_ranges: Dict[str, tuple] = field(default_factory=lambda:  {
        'u': (-150, 150),      # m/s
        'v': (-150, 150),      # m/s
        'w': (-10, 10),        # Pa/s
        'T': (150, 350),       # K
        'q': (0, 0.05),        # kg/kg
        'ps': (30000, 110000)  # Pa
    })