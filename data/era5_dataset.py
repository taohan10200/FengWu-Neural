"""
PyTorch Dataset for ERA5
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from pathlib import Path

from configs.model_config import ModelConfig
from configs.era5_config import ERA5Config
from core.state import AtmosphericState
from data.era5_loader import ERA5Data
from data.era5_interpolator import ERA5ToModelConverter

class ERA5Dataset(Dataset):
    """ERA5 数据集（用于训练）"""
    
    def __init__(self, 
                 data_dir: str,
                 config: ModelConfig,
                 era5_config: ERA5Config,
                 forecast_hours: int = 24,
                 file_pattern: str = "era5_*.npz"):
        """
        Args:
            data_dir: 数据目录
            config:  模型配置
            era5_config: ERA5配置
            forecast_hours: 预报时长（用于生成训练对）
            file_pattern: 文件匹配模式
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.era5_config = era5_config
        self.forecast_hours = forecast_hours
        
        # 转换器
        self.converter = ERA5ToModelConverter(config, era5_config)
        
        # 构建数据索引
        self.data_files = sorted(self.data_dir.glob(file_pattern))
        
        if len(self.data_files) == 0:
            raise ValueError(f"在 {data_dir} 中找不到匹配 {file_pattern} 的文件")
        
        print(f"找到 {len(self.data_files)} 个数据文件")
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx) -> Tuple[AtmosphericState, AtmosphericState]:
        """
        Returns:
            initial_state: 初始状态
            target_state: 目标状态（用于训练）
        """
        # 加载初始状态
        file_path = self.data_files[idx]
        era5_data = ERA5Data.from_numpy(str(file_path))
        initial_state = self.converter(era5_data, horizontal_interp=True)
        
        # 加载目标状态（简化：假设文件按时间顺序排列）
        target_idx = min(idx + self.forecast_hours // 6, len(self.data_files) - 1)
        target_file = self.data_files[target_idx]
        era5_target = ERA5Data.from_numpy(str(target_file))
        target_state = self.converter(era5_target, horizontal_interp=True)
        
        return initial_state, target_state