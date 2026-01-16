"""
ERA5 数据加载器
"""

import numpy as np
import xarray as xr
import os
import torch
import torch.nn.functional as F
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ERA5Data:
    """ERA5 数据容器"""
    
    u: np.ndarray      # (nlat, nlon, nz)
    v: np.ndarray
    w: np.ndarray
    T: np.ndarray
    q: np.ndarray
    ps: np.ndarray     # (nlat, nlon)
    lat: np.ndarray
    lon: np.ndarray
    pressure_levels: np.ndarray
    
    def __post_init__(self):
        """验证数据"""
        # 放宽验证，适应加载过程中的数据
        pass
        
    def validate(self):
        assert self.u.shape == self.v.shape == self.T.shape == self.q.shape
        if self.w is not None:
             assert self.u.shape == self.w.shape
        assert self.u.shape[-1] == len(self.pressure_levels)
        assert self.ps.shape[:2] == self.u.shape[:2]

    
    @property
    def shape(self):
        return self.u.shape
    
    @property
    def nlat(self):
        return self.u.shape[0]
    
    @property
    def nlon(self):
        return self.u.shape[1]
    
    @property
    def nlevels(self):
        return self.u.shape[2]
    
    def interpolate_to_sigma(self, target_nz: int, sigma_power: float = 2.5, a_scale: float = 1000.0) -> 'ERA5Data':
        """
        将ERA5气压层数据插值到sigma混合坐标层
        
        Args:
            target_nz: 目标sigma层数
            sigma_power: sigma坐标幂指数
            a_scale: 固定气压部分的尺度（Pa）
        
        Returns:
            插值后的ERA5Data
        """
        nlat, nlon, _ = self.u.shape
        
        # 计算目标混合坐标层（p = a + b*ps）
        # 使用与VerticalCoordinate相同的公式
        k = np.arange(target_nz + 1)
        eta = k / target_nz
        b_k = eta ** sigma_power
        a_k = (1 - eta) * a_scale  # 高层有固定气压，避免p趋近0
        
        b_half = (b_k[:-1] + b_k[1:]) / 2  # (target_nz,)
        a_half = (a_k[:-1] + a_k[1:]) / 2  # (target_nz,)
        
        # 检测ERA5 pressure_levels的单位（hPa或Pa）
        # ERA5标准层通常是1000, 925, 850, ..., 1 hPa
        if np.max(self.pressure_levels) > 2000:
            # 可能已经是Pa
            source_p = self.pressure_levels
            print(f"  检测到气压单位为Pa")
        else:
            # 应该是hPa，转换为Pa
            source_p = self.pressure_levels * 100.0
            print(f"  检测到气压单位为hPa，转换为Pa")
        
        # 确保source_p是递减的（从高空到地面）
        if source_p[0] > source_p[-1]:
            source_p = source_p[::-1]
            u_src = self.u[:, :, ::-1]
            v_src = self.v[:, :, ::-1]
            T_src = self.T[:, :, ::-1]
            q_src = self.q[:, :, ::-1]
            w_src = self.w[:, :, ::-1] if self.w is not None else None
        else:
            u_src = self.u
            v_src = self.v
            T_src = self.T
            q_src = self.q
            w_src = self.w
        
        # 初始化输出数组
        u_interp = np.zeros((nlat, nlon, target_nz), dtype=np.float32)
        v_interp = np.zeros((nlat, nlon, target_nz), dtype=np.float32)
        T_interp = np.zeros((nlat, nlon, target_nz), dtype=np.float32)
        q_interp = np.zeros((nlat, nlon, target_nz), dtype=np.float32)
        w_interp = np.zeros((nlat, nlon, target_nz), dtype=np.float32)
        
        # 对每个水平格点进行垂直插值
        for i in range(nlat):
            for j in range(nlon):
                ps_ij = self.ps[i, j]  # Pa
                
                # 目标混合坐标层的气压 (Pa): p = a + b * ps
                target_p = a_half + b_half * ps_ij
                
                # 确保target_p在合理范围内
                # ERA5顶层通常是1 hPa = 100 Pa，底层不超过ps
                target_p = np.clip(target_p, source_p[0], ps_ij)
                
                # 对每个变量进行线性插值
                u_interp[i, j, :] = np.interp(target_p, source_p, u_src[i, j, :])
                v_interp[i, j, :] = np.interp(target_p, source_p, v_src[i, j, :])
                T_interp[i, j, :] = np.interp(target_p, source_p, T_src[i, j, :])
                q_interp[i, j, :] = np.interp(target_p, source_p, q_src[i, j, :])
                if w_src is not None:
                    w_interp[i, j, :] = np.interp(target_p, source_p, w_src[i, j, :])
        
        # 创建新的sigma层数组（用b_half表示sigma值）
        sigma_levels = b_half
        
        # 验证插值后的数据质量
        print(f"  插值后数据统计:")
        print(f"    u: [{np.min(u_interp):.1f}, {np.max(u_interp):.1f}] m/s")
        print(f"    T: [{np.min(T_interp):.1f}, {np.max(T_interp):.1f}] K")
        
        # 检查水平梯度（相邻格点之间的最大差异）
        u_hdiff_x = np.max(np.abs(np.diff(u_interp, axis=1)))
        u_hdiff_y = np.max(np.abs(np.diff(u_interp, axis=0)))
        print(f"    水平梯度: du_x_max={u_hdiff_x:.1f} m/s, du_y_max={u_hdiff_y:.1f} m/s")
        
        # 如果水平梯度过大，应用轻度平滑
        if u_hdiff_x > 50 or u_hdiff_y > 50:  # 相邻格点差异>50m/s时平滑
            print(f"  检测到大梯度，应用水平平滑...")
            from scipy.ndimage import gaussian_filter
            
            # 对每一层分别平滑（保持垂直结构）
            for k in range(target_nz):
                u_interp[:, :, k] = gaussian_filter(u_interp[:, :, k], sigma=0.5)
                v_interp[:, :, k] = gaussian_filter(v_interp[:, :, k], sigma=0.5)
                T_interp[:, :, k] = gaussian_filter(T_interp[:, :, k], sigma=0.5)
            
            # 重新检查
            u_hdiff_x = np.max(np.abs(np.diff(u_interp, axis=1)))
            u_hdiff_y = np.max(np.abs(np.diff(u_interp, axis=0)))
            print(f"    平滑后: du_x_max={u_hdiff_x:.1f} m/s, du_y_max={u_hdiff_y:.1f} m/s")
        
        # 检查垂直梯度（相邻层之间的最大差异）
        if target_nz > 1:
            u_vdiff = np.max(np.abs(np.diff(u_interp, axis=2)))
            T_vdiff = np.max(np.abs(np.diff(T_interp, axis=2)))
            print(f"    垂直梯度: du_max={u_vdiff:.1f} m/s, dT_max={T_vdiff:.1f} K")
        
        return ERA5Data(
            u=u_interp,
            v=v_interp,
            w=w_interp,
            T=T_interp,
            q=q_interp,
            ps=self.ps.copy(),
            lat=self.lat.copy(),
            lon=self.lon.copy(),
            pressure_levels=sigma_levels  # 现在存储sigma值而非气压
        )
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """转换为字典"""
        return {
            'u': self.u,
            'v': self.v,
            'w': self.w,
            'T': self.T,
            'q':  self.q,
            'ps': self.ps,
            'lat': self.lat,
            'lon': self.lon,
            'pressure_levels': self.pressure_levels
        }
    
    @classmethod
    def from_dict(cls, data:  Dict[str, np.ndarray]) -> 'ERA5Data':
        """从字典创建"""
        return cls(
            u=data['u'],
            v=data['v'],
            w=data['w'],
            T=data['T'],
            q=data['q'],
            ps=data['ps'],
            lat=data['lat'],
            lon=data['lon'],
            pressure_levels=data['pressure_levels']
        )
    
    @classmethod
    def from_real_data(cls, 
                       timestamp: str, 
                       config_path: str = None, 
                       data_root: str = None) -> 'ERA5Data':
        """
        从使用 dataset.py 中的 ERA5Dataset 读取实际数据
        
        Args:
            timestamp: 时间戳字符串 "YYYY-MM-DD HH:MM:SS"
            config_path: 配置文件路径
        """
        import sys
        
        # 确保 dataset.py (在同级目录 data/ 中) 可以被导入
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)

        try:
            from dataset import ERA5Dataset
        except ImportError:
            try:
                from .dataset import ERA5Dataset
            except ImportError:
                from data.dataset import ERA5Dataset

        from mmengine.config import Config
        
        # 确保项目根目录在 path 中
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)

        # 加载配置
        if config_path is None:
            # 默认尝试上级目录的 config.py
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(os.path.dirname(current_dir), 'config.py')
            
        cfg = Config.fromfile(config_path)
        
        # 修改 dataset 配置以只加载特定时间
        if data_root is not None:
             cfg.data.era5_time_range.data_root = data_root

        # 将时间戳格式转换为配置所需的格式 (YYYY-MM-DDTHH:MM:SS)
        ts_obj = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        ts_str = ts_obj.strftime("%Y-%m-%dT%H:%M:%S")

        cfg.data.era5_time_range.train_st = ts_str
        cfg.data.era5_time_range.train_et = ts_str
        cfg.data.era5_time_range.val_st = ts_str
        cfg.data.era5_time_range.val_et = ts_str
        
        # 初始化数据集
        # 由于我们只需要一个样本，且不需要 GT (虽然 dataset 会尝试加载)
        # 我们这里 trick 一下：
        # dataset._load_data 需要 timestamp 格式为 YYYY-MM-DD HH:MM:SS
        
        dataset = ERA5Dataset(cfg.data, mode='train')
        
        # 直接调用内部加载函数，避免 __getitem__ 中加载 target 的逻辑
        # dataset.timestamps 应该包含我们构造的那个时间点
        
        if not dataset.timestamps:
             raise ValueError(f"No data found for timestamp {timestamp} in {cfg.data.era5_time_range.data_root}")

        ts_info = dataset.timestamps[0]
        data_root = ts_info['data_root']
        
        # 加载 [channels, lat, lon]
        data_tensor = dataset._load_data(timestamp, data_root)
        
        # 反归一化 (如果 dataset 已经加载了 mean/std)
        # ERA5Dataset 默认加载是做了归一化的吗？看代码 _load_data 好像没有，
        # 只有在 pipeline 或者 __getitem__ 里可能并未显式调用 normalize (看 dataset.py 实现)
        # Wait, dataset.py 里的 __getitem__ 调用了 _load_data 
        # 而 _load_data 只是加载原始值然后转 tensor，并没有调用 dataset.normalize
        # 所以加载出来的是原始物理量
        
        # 解析数据张量到各个变量
        # 变量顺序由 dataset.vnames 决定
        pressure_vars = dataset.vnames.get('pressure', [])
        single_vars = dataset.vnames.get('single', [])
        pressure_levels = dataset.pressure_levels
        
        n_pressure = len(pressure_vars)
        n_single = len(single_vars)
        n_levels = len(pressure_levels)
        
        # channel索引映射
        # 假设数据排列是: 
        # v1_l1, v1_l2... v1_ln, v2_l1...
        # 还是 v1_l1, v2_l1..., v1_l2... ?
        # 查看 dataset._load_data 实现：
        # 它按 pressure_vars 循环，内部按 pressure_levels 循环
        # 所以顺序是: [Var1_L1, Var1_L2, ..., Var1_Ln, Var2_L1, ...]
        
        # 转换为 (lat, lon, nz) 格式供模型使用
        # 我们的模型通常需要 (lat, lon, nz) 但是这里的 nz 是所有变量层数之和？
        # 不，AtmosphericState 需要特定的变量：u, v, w, T, q, ps
        
        # 构造 numpy 数组
        # 原始：(C, H, W) -> (H, W, C)
        data_np = data_tensor.numpy().transpose(1, 2, 0)
        h, w, c = data_np.shape
        
        u = np.zeros((h, w, n_levels), dtype=np.float32)
        v = np.zeros((h, w, n_levels), dtype=np.float32)
        T = np.zeros((h, w, n_levels), dtype=np.float32)
        q = np.zeros((h, w, n_levels), dtype=np.float32)
        w_vel = np.zeros((h, w, n_levels), dtype=np.float32) # dataset可能没有w
        ps = np.zeros((h, w), dtype=np.float32)
        
        idx = 0
        for vname in pressure_vars:
            # 提取该变量的所有层
            # 它们是连续的吗？是的，内层循环是 levels
            var_data = data_np[:, :, idx : idx + n_levels]
            idx += n_levels
            
            # 映射到标准变量名
            # dataset config: ['t', 'u', 'v', 'q']
            if vname.lower() in ['u', 'u_component_of_wind']:
                u = var_data
            elif vname.lower() in ['v', 'v_component_of_wind']:
                v = var_data
            elif vname.lower() in ['t', 'temperature']:
                T = var_data
            elif vname.lower() in ['q', 'specific_humidity']:
                q = var_data
            elif vname.lower() in ['w', 'vertical_velocity']:
                w_vel = var_data
                
        # 单层变量
        for vname in single_vars:
            var_data = data_np[:, :, idx]
            idx += 1
            
            # ps 通常对应 sp (Surface Pressure) 或 msl (Mean Sea Level Pressure)
            # 这里优先用 sp
            if vname.lower() in ['sp', 'surface_pressure']:
                ps = var_data
            elif vname.lower() in ['msl', 'mean_sea_level_pressure'] and np.all(ps == 0):
                ps = var_data
                
        # 构造网格坐标 (简化的经纬度)
        # ERA5 通常是 Global 
        lat = np.linspace(90, -90, h)
        lon = np.linspace(0, 360 - 360/w, w)
        
        return cls(
            u=u,
            v=v,
            w=w_vel,
            T=T,
            q=q,
            ps=ps,
            lat=lat,
            lon=lon,
            pressure_levels=np.array(pressure_levels)
        )
    
    @classmethod
    def from_numpy(cls, filepath: str) -> 'ERA5Data': 
        """从 . npz 文件加载"""
        data = np.load(filepath)
        return cls.from_dict(data)
    
    def save_numpy(self, filepath: str):
        """保存为 .npz 文件"""
        np.savez_compressed(filepath, **self.to_dict())
        print(f"数据已保存到:  {filepath}")