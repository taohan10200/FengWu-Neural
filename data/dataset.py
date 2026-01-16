"""
Data Processing Module for Neural NWP
气象数据处理模块
"""

import torch
import numpy as np
import torch.nn.functional as F
import xarray as xr
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import os
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data.distributed import DistributedSampler
from ghr.utils.s3_client import s3_client

class ERA5Dataset(Dataset):
    """
    ERA5数据集加载器
    支持从本地文件或对象存储加载ERA5再分析数据
    参考FengWu-GHR的数据加载方式
    """
    
    def __init__(self,
                 cfg: dict,
                 mode: str = 'train',
                 ):
        """
        Args:
            cfg: 配置字典，包含所有参数
            data_root: ERA5数据根目录
        """
        # 优先使用cfg中的配置
        self.cfg = cfg
        self.vnames = cfg.vnames or {
                'pressure': ['t', 'u', 'v', 'q' ],
                'single': ['v10', 'u10', 'v100', 'u100', 't2m', 'tcc', 'sp', 'msl', 'tp6h', 'ssr6h']
            }
        
        self.pressure_levels = cfg.pressure_levels or [
                1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50
            ]
        self.input_shape = (128, 256)  # 目标分辨率
        
        # 构建时间戳列表
        self.timestamps = []
        
        # 支持ERA5时间范围
        if hasattr(cfg, 'era5_time_range') and cfg.era5_time_range:
            if mode == 'train':
                cfg.era5_time_range.st = cfg.era5_time_range.get('train_st', cfg.era5_time_range.train_st)
                cfg.era5_time_range.et = cfg.era5_time_range.get('train_et', cfg.era5_time_range.train_et)
            elif mode == 'val':
                cfg.era5_time_range.st = cfg.era5_time_range.get('val_st', cfg.era5_time_range.val_st)
                cfg.era5_time_range.et = cfg.era5_time_range.get('val_et', cfg.era5_time_range.val_et)
            era5_timestamps = pd.date_range(
                start=cfg.era5_time_range.st,
                end=cfg.era5_time_range.et,
                freq=cfg.era5_time_range.get('interval', '6H')
            )
            for ts in era5_timestamps.strftime("%Y-%m-%d %H:%M:%S").tolist():
                self.timestamps.append({
                    'timestamp': ts,
                    'source': 'ERA5',
                    'data_root': cfg.era5_time_range.data_root
                })
        
        # 支持Analysis时间范围
        if hasattr(cfg, 'analysis_time_range') and cfg.analysis_time_range:
            
            analysis_timestamps = pd.date_range(
                start=cfg.analysis_time_range.st,
                end=cfg.analysis_time_range.et,
                freq=cfg.analysis_time_range.get('interval', '6H')
            )
            for ts in analysis_timestamps.strftime("%Y-%m-%d %H:%M:%S").tolist():
                self.timestamps.append({
                    'timestamp': ts,
                    'source': 'Analysis',
                    'data_root': cfg.analysis_time_range.data_root
                })
                
        # S3/Ceph客户端
        self.client = s3_client(**self.cfg.bucket) if s3_client else None
        
        # 线程池用于并行加载
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # 加载归一化统计信息
        self.mean, self.std = self._load_mean_std()
        
    def _load_mean_std(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载变量的均值和标准差
        从预计算的统计文件中读取
        """
        mean_std_path = './ghr/models/mean_std.json'
        mean_std_single_path = './ghr/models/mean_std_single.json'
        
        mean_list, std_list = [], []
        
        try:
            # 加载气压层变量统计信息
            if os.path.exists(mean_std_path):
                with open(mean_std_path, 'r') as f:
                    mean_std = json.load(f)
                
                for vname in self.vnames.get('pressure', []):
                    for level in self.pressure_levels:
                        level_idx = mean_std.get('levels', []).index(level) if level in mean_std.get('levels', []) else 0
                        mean_list.append(mean_std['mean'].get(vname, [0])[level_idx] if len(mean_std['mean'].get(vname, [0])) > level_idx else 0)
                        std_list.append(mean_std['std'].get(vname, [1])[level_idx] if len(mean_std['std'].get(vname, [1])) > level_idx else 1)
            
            # 加载单层变量统计信息
            if os.path.exists(mean_std_single_path):
                with open(mean_std_single_path, 'r') as f:
                    mean_std_single = json.load(f)
                
                for vname in self.vnames.get('single', []):
                    mean_list.append(mean_std_single['mean'].get(vname, 0))
                    std_list.append(mean_std_single['std'].get(vname, 1))
        
        except Exception as e:
            print(f"Warning: Could not load mean_std files, using default values.Error: {e}")
            # 使用默认值
            num_pressure_vars = len(self.vnames.get('pressure', [])) * len(self.pressure_levels)
            num_single_vars = len(self.vnames.get('single', []))
            mean_list = [0.0] * (num_pressure_vars + num_single_vars)
            std_list = [1.0] * (num_pressure_vars + num_single_vars)
        
        return np.array(mean_list, dtype=np.float32), np.array(std_list, dtype=np.float32)
    
    def __len__(self) -> int:
        # 减1以确保有GT数据
        return len(self.timestamps) - 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        获取一个训练样本
        
        Returns:
            input_data: 初始场 [1, lat, lon, channels]
            target_data: 序列真值 [rollout_steps, lat, lon, channels]
            ts_info: 时间戳信息
        """
        # 当前时间戳
        ts_info = self.timestamps[idx]
        init_timestamp = ts_info['timestamp']
        source = ts_info['source']
        data_root = ts_info['data_root']
        
        # 加载初始输入
        input_data = self._load_data(init_timestamp, data_root)
        
        # 加载未来 rollout_steps 个时间步的数据作为 target
        # 从 dataset_cfg 中获取 rollout_steps，默认为6
        rollout_steps = self.cfg.get('rollout_steps', 6)
        interval_hours = 1 # 数据间隔1小时
        
        targets = []
        curr_time = datetime.strptime(init_timestamp, "%Y-%m-%d %H:%M:%S")
        
        for i in range(rollout_steps):
            future_time = curr_time + timedelta(hours=(i+1)*interval_hours)
            future_timestamp = future_time.strftime("%Y-%m-%d %H:%M:%S")
            # 加载每一步的真值
            try:
                step_target = self._load_data(future_timestamp, data_root)
                targets.append(step_target)
            except Exception as e:
                # 如果某个时间步加载失败（例如超出文件范围），这里简单处理为复制上一步或者报错
                print(f"Error loading target for {future_timestamp}: {e}")
                if targets:
                    targets.append(targets[-1])
                else:
                    targets.append(input_data)

        # 调整维度以匹配模型输入预期
        # _load_data 返回 [channels, lat, lon]
        # output expected: [time_steps, lat, lon, channels]
        
        # input: [channels, lat, lon] -> [1, lat, lon, channels]
        input_data = input_data.permute(1, 2, 0).unsqueeze(0)
        
        # targets: [rollout_steps, channels, lat, lon] -> [rollout_steps, lat, lon, channels]
        if targets:
            target_data = torch.stack(targets).permute(0, 2, 3, 1)
        else:
            # Fallback
            target_data = input_data.repeat(rollout_steps, 1, 1, 1)
        
        return input_data, target_data, ts_info
    
    def _load_data(self, timestamp: str, data_root: str) -> torch.Tensor:
        """
        加载单个时间点的数据
        
        Args:
            timestamp: 时间戳字符串 "YYYY-MM-DD HH:MM:SS"
            data_root: 数据根目录
            
        Returns:
            data: [channels, lat, lon]
        """
        file_paths = []
        
        # 生成文件路径（气压层变量）
        for vname in self.vnames.get('pressure', []):
            for height in self.pressure_levels:
                if self.client:
                    # S3路径格式
                    file_path = f'{data_root}/{timestamp[:4]}/{timestamp[:10]}/{timestamp[-8:]}-{vname}-{height}.npy'
                else:
                    # 本地路径格式
                    file_path = os.path.join(
                        data_root, 
                        timestamp[:4], 
                        timestamp[:10], 
                        f'{timestamp[-8:]}-{vname}-{height}.npy'
                    )
                file_paths.append((file_path, 'pressure', vname))
        
        # 生成文件路径（单层变量）
        for vname in self.vnames.get('single', []):
            if self.client:
                file_path = f'{data_root}/single/{timestamp[:4]}/{timestamp[:10]}/{timestamp[-8:]}-{vname}.npy'
            else:
                file_path = os.path.join(
                    data_root, 'single',
                    timestamp[:4],
                    timestamp[:10],
                    f'{timestamp[-8:]}-{vname}.npy'
                )
            file_paths.append((file_path, 'single', vname))
        
        # 并行加载函数
        def load_file(file_info):
            file_path, var_type, vname = file_info
            
            if self.client:
                # 从S3加载
                vdata = self.client.read_npy_from_BytesIO(objectName=file_path)
            else:
                # 从本地加载
                vdata = np.load(file_path)
            
            vdata = torch.tensor(vdata, dtype=torch.float32)
            vdata = self._check_input(vdata)
            
            # 特殊处理降水（转换为mm）
            if var_type == 'single' and 'tp' in vname:
                vdata = vdata * 1000
            
            return vdata
            
        
        # 并行执行加载
        results = list(self.executor.map(load_file, file_paths))
        
        # 拼接所有变量
        input_data = torch.stack(results, dim=0)  # [channels, lat, lon]
        
        return input_data
    
    def _check_input(self, vdata: torch.Tensor) -> torch.Tensor:
        """
        调整数据到目标分辨率
        
        Args:
            vdata: [1, 1, H, W]
        Returns:
            resized: [1, target_H, target_W]
        """
        if vdata.shape != self.input_shape:
            vdata = F.interpolate(
                vdata[None, None, :, :],
                size=self.input_shape,
                mode='bicubic',
                align_corners=False
            )
            vdata = vdata.squeeze(0).squeeze(0)
        
        return vdata
    
    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        归一化数据
        
        Args:
            data: [channels, lat, lon]
        Returns:
            normalized: [channels, lat, lon]
        """
        mean = torch.tensor(self.mean[:, None, None], dtype=data.dtype, device=data.device)
        std = torch.tensor(self.std[:, None, None], dtype=data.dtype, device=data.device)
        
        return (data - mean) / std
    
    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        反归一化数据
        
        Args:
            data: [channels, lat, lon]
        Returns:
            denormalized: [channels, lat, lon]
        """
        mean = torch.tensor(self.mean[:, None, None], dtype=data.dtype, device=data.device)
        std = torch.tensor(self.std[:, None, None], dtype=data.dtype, device=data.device)
        
        return data * std + mean

class InfiniteDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

    def __iter__(self):
        while True:
            # Use the parent's __iter__ method to get the indices for this process
            indices = list(super().__iter__())
            # Yield the indices in an infinite loop
            yield from indices
            
def create_dataloaders(
    dataset_cfg: dict,
    batch_size: int = 4,
    num_workers: int = 4,
    world_size: int = 1,
    rank: int = 0,
    
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批大小
        num_workers: 数据加载线程数
        train_split: 训练集比例
        **dataset_kwargs: 传递给Dataset的其他参数
        
    Returns:
        train_loader, val_loader
    """
    # 创建数据集
    train_dataset = ERA5Dataset(dataset_cfg, mode='train')
    val_dataset = ERA5Dataset(dataset_cfg, mode='val')


    if world_size is not None and world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return train_loader, val_loader


class DataNormalizer:
    """
    数据归一化工具类
    支持按变量和气压层分别归一化
    """
    
    def __init__(self, 
                 num_vars: int = 5,
                 num_levels: int = 37,
                 var_names: List[str] = None):
        """
        Args:
            num_vars: 变量数量
            num_levels: 垂直层数
            var_names: 变量名列表
        """
        self.num_vars = num_vars
        self.num_levels = num_levels
        self.var_names = var_names or ['T', 'u', 'v', 'q', 'sp']
        
        # 存储统计信息：{var_name: {'mean': [...], 'std': [...]}}
        self.stats = {}
        
        # 每个变量的均值和标准差 [num_vars * num_levels]
        self.means = None
        self.stds = None
    
    def fit(self, data: torch.Tensor):
        """
        从数据中计算均值和标准差
        
        Args:
            data: [N, lat, lon, num_vars * num_levels] 或 [N, num_vars * num_levels, lat, lon]
        """
        if data.ndim == 4:
            # 重排为 [N, num_vars * num_levels, lat, lon]
            if data.shape[-1] == self.num_vars * self.num_levels:
                data = data.permute(0, 3, 1, 2)
        
        # 按变量计算统计信息
        data_reshaped = data.reshape(-1, self.num_vars, self.num_levels, 
                                     data.shape[-2], data.shape[-1])
        
        means_list = []
        stds_list = []
        
        for i, var_name in enumerate(self.var_names):
            var_data = data_reshaped[:, i, :, :, :]  # [N, num_levels, lat, lon]
            
            # 按层计算均值和标准差
            var_means = var_data.mean(dim=(0, 2, 3))  # [num_levels]
            var_stds = var_data.std(dim=(0, 2, 3))    # [num_levels]
            
            self.stats[var_name] = {
                'mean': var_means.cpu().numpy(),
                'std': var_stds.cpu().numpy()
            }
            
            means_list.append(var_means)
            stds_list.append(var_stds)
        
        # 拼接为单一数组
        self.means = torch.cat(means_list).cpu().numpy()  # [num_vars * num_levels]
        self.stds = torch.cat(stds_list).cpu().numpy()
        
        print("Normalization statistics computed:")
        for var in self.var_names:
            print(f"  {var}: mean={self.stats[var]['mean'].mean():.4f}, "
                  f"std={self.stats[var]['std'].mean():.4f}")
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        归一化：(x - μ) / σ
        
        Args:
            data: [..., num_vars * num_levels, ...] or [..., lat, lon, num_vars * num_levels]
        Returns:
            normalized: 同样形状
        """
        if self.means is None or self.stds is None:
            raise ValueError("Normalizer not fitted.Call fit() first.")
        
        original_shape = data.shape
        device = data.device
        
        # 转换为 [batch, channels, lat, lon] 格式
        if data.shape[-1] == len(self.means):  # [..., lat, lon, channels]
            data = data.permute(0, 3, 1, 2) if data.ndim == 4 else data
        
        mean = torch.tensor(self.means[:, None, None], 
                          dtype=data.dtype, device=device)
        std = torch.tensor(self.stds[:, None, None], 
                         dtype=data.dtype, device=device)
        
        normalized = (data - mean) / (std + 1e-8)
        
        # 恢复原始形状
        if original_shape[-1] == len(self.means):
            normalized = normalized.permute(0, 2, 3, 1) if normalized.ndim == 4 else normalized
        
        return normalized
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        反归一化：x = normalized * σ + μ
        
        Args:
            data: 归一化后的数据
        Returns:
            denormalized: 原始尺度的数据
        """
        if self.means is None or self.stds is None:
            raise ValueError("Normalizer not fitted.Call fit() first.")
        
        original_shape = data.shape
        device = data.device
        
        # 转换为 [batch, channels, lat, lon] 格式
        if data.shape[-1] == len(self.means):
            data = data.permute(0, 3, 1, 2) if data.ndim == 4 else data
        
        mean = torch.tensor(self.means[:, None, None],
                          dtype=data.dtype, device=device)
        std = torch.tensor(self.stds[:, None, None],
                         dtype=data.dtype, device=device)
        
        denormalized = data * std + mean
        
        # 恢复原始形状
        if original_shape[-1] == len(self.means):
            denormalized = denormalized.permute(0, 2, 3, 1) if denormalized.ndim == 4 else denormalized
        
        return denormalized
    
    def save(self, path: str):
        """保存统计信息"""
        save_dict = {
            'means': self.means,
            'stds': self.stds,
            'stats': {k: {'mean': v['mean'].tolist() if isinstance(v['mean'], np.ndarray) else v['mean'],
                         'std': v['std'].tolist() if isinstance(v['std'], np.ndarray) else v['std']}
                     for k, v in self.stats.items()},
            'num_vars': self.num_vars,
            'num_levels': self.num_levels,
            'var_names': self.var_names
        }
        
        # 保存为JSON或PyTorch格式
        if path.endswith('.json'):
            with open(path, 'w') as f:
                json.dump(save_dict, f, indent=2)
        else:
            torch.save(save_dict, path)
        
        print(f"Normalizer saved to {path}")
    
    def load(self, path: str):
        """加载统计信息"""
        if path.endswith('.json'):
            with open(path, 'r') as f:
                save_dict = json.load(f)
            self.means = np.array(save_dict['means'])
            self.stds = np.array(save_dict['stds'])
        else:
            save_dict = torch.load(path)
            self.means = save_dict['means']
            self.stds = save_dict['stds']
        
        self.stats = save_dict.get('stats', {})
        self.num_vars = save_dict.get('num_vars', self.num_vars)
        self.num_levels = save_dict.get('num_levels', self.num_levels)
        self.var_names = save_dict.get('var_names', self.var_names)
        
        print(f"Normalizer loaded from {path}")
        print(f"  Variables: {self.var_names}")
        print(f"  Num levels: {self.num_levels}")


if __name__ == '__main__':
    # 测试数据加载
    print("Testing WeatherDataset...")
    
    dataset = WeatherDataset(
        data_path='dummy_path',
        variables=['t', 'u', 'v', 'q', 'sp'],
        num_input_steps=1,
        num_output_steps=1
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 获取一个样本
    input_data, target_data = dataset[0]
    print(f"Input shape: {input_data.shape}")
    print(f"Target shape: {target_data.shape}")
    
    # 测试数据加载器
    print("\nTesting DataLoader...")
    train_loader, val_loader = create_dataloaders(
        data_dir='dummy_path',
        batch_size=2,
        num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # 获取一个批次
    for batch_input, batch_target in train_loader:
        print(f"Batch input shape: {batch_input.shape}")
        print(f"Batch target shape: {batch_target.shape}")
        break
    
    print("\n✓ Data processing test passed!")
