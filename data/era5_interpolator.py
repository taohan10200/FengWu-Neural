"""
ERA5 数据插值器：气压层 → σ坐标
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple

from configs.model_config import ModelConfig
from configs.era5_config import ERA5Config
from core.vertical_coordinate import VerticalCoordinate
from core.grid import GridGeometry
from core.state import AtmosphericState
from data.era5_loader import ERA5Data

class PressureToSigmaInterpolator(nn.Module):
    """气压层 → σ坐标插值器"""
    
    def __init__(self, config: ModelConfig, era5_config: ERA5Config):
        super().__init__()
        self.config = config
        self.era5_config = era5_config
        
        # 创建垂直坐标
        self.vertical_coord = VerticalCoordinate(config)
        
        # 注册 ERA5 气压层
        self.register_buffer(
            'era5_pressure_levels',
            torch.tensor(era5_config.pressure_levels * 100,  # hPa → Pa
                        dtype=config.dtype)
        )
    
    def forward(self, era5_data: ERA5Data) -> AtmosphericState:
        """
        将 ERA5 数据插值到模型坐标
        
        Args:
            era5_data: ERA5 原始数据 (气压层)
        
        Returns:
            state: 模型状态 (σ坐标)
        """
        device = self.config.device
        dtype = self.config.dtype
        
        # 1. 转换为 Tensor
        ps_tensor = torch.tensor(era5_data.ps, dtype=dtype, device=device)
        
        # 2. 计算模型的 3D 气压场（σ坐标）
        a_k, b_k = self.vertical_coord()
        
        nlat, nlon = era5_data.nlat, era5_data.nlon
        nz_model = self.config.nz
        
        p_model = torch.zeros(nlat, nlon, nz_model, dtype=dtype, device=device)
        for k in range(nz_model):
            p_model[:, : , k] = a_k[k] + b_k[k] * ps_tensor
        
        # 3. 对每个变量进行插值
        state = AtmosphericState(self.config)
        state.ps = ps_tensor
        
        state.u = self._interpolate_variable(era5_data.u, era5_data.ps, p_model)
        state.v = self._interpolate_variable(era5_data.v, era5_data.ps, p_model)
        state.w = self._interpolate_variable(era5_data.w, era5_data.ps, p_model)
        state.T = self._interpolate_variable(era5_data.T, era5_data.ps, p_model)
        state.q = self._interpolate_variable(era5_data.q, era5_data.ps, p_model)
        
        # 4. 计算诊断量
        state.p = p_model
        
        return state
    
    def _interpolate_variable(self, 
                             var_era5: np.ndarray,
                             ps_era5: np.ndarray,
                             p_model: torch.Tensor) -> torch.Tensor:
        """
        单个变量的垂直插值
        """
        nlat, nlon, _ = var_era5.shape
        nz_model = p_model.shape[2]
        device = p_model.device
        dtype = p_model.dtype
        
        var_model = torch.zeros(nlat, nlon, nz_model, dtype=dtype, device=device)
        
        p_era5_levels = self.era5_pressure_levels.cpu().numpy()
        
        # 对每个格点进行插值
        for i in range(nlat):
            for j in range(nlon):
                ps_ij = ps_era5[i, j]
                
                # ERA5 有效气压层（过滤高于地面的层）
                valid_mask = p_era5_levels <= ps_ij
                p_era5_valid = p_era5_levels[valid_mask]
                var_era5_valid = var_era5[i, j, valid_mask]
                
                if len(p_era5_valid) < 2:
                    continue
                
                # 模型气压层
                p_model_ij = p_model[i, j, : ].cpu().numpy()
                
                # 对数压强插值
                log_p_era5 = np.log(p_era5_valid + 1e-8)
                log_p_model = np.log(p_model_ij + 1e-8)
                
                # 线性插值
                interp_func = interp1d(
                    log_p_era5, 
                    var_era5_valid,
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                
                var_interp = interp_func(log_p_model)
                var_model[i, j, :] = torch.tensor(var_interp, dtype=dtype, device=device)
        
        return var_model


class ERA5ToModelConverter(nn.Module):
    """
    完整的 ERA5 → 模型转换器
    包括：垂直插值 + 水平插值 + 单位转换
    """
    
    def __init__(self, config: ModelConfig, era5_config: ERA5Config):
        super().__init__()
        self.config = config
        self.era5_config = era5_config
        
        # 垂直插值器
        self.vertical_interpolator = PressureToSigmaInterpolator(config, era5_config)
        
        # 模型网格
        self.grid = GridGeometry(config)
    
    def forward(self, era5_data: ERA5Data, horizontal_interp: bool = True) -> AtmosphericState:
        """
        完整转换流程
        
        Args: 
            era5_data: ERA5 原始数据
            horizontal_interp: 是否进行水平插值
        
        Returns:
            state: 模型状态
        """
        # 1. 水平插值（如果需要）
        if horizontal_interp:
            era5_data = self._horizontal_interpolation(era5_data)
        
        # 2. 垂直插值
        state = self.vertical_interpolator(era5_data)
        
        # 3. 单位转换和质量检查
        state = self._unit_conversion(state)
        self._quality_check(state)
        
        return state
    
    def _horizontal_interpolation(self, era5_data: ERA5Data) -> ERA5Data:
        """水平插值：ERA5 网格 → 模型网格"""
        from scipy.interpolate import RegularGridInterpolator
        
        era5_lat = era5_data.lat
        era5_lon = era5_data.lon
        
        model_lat = self.grid.lat.cpu().numpy()
        model_lon = self.grid.lon.cpu().numpy()
        
        # 创建目标网格点
        model_lat_2d, model_lon_2d = np.meshgrid(model_lat, model_lon, indexing='ij')
        points = np.stack([model_lat_2d.ravel(), model_lon_2d.ravel()], axis=-1)
        
        def interp_3d(data_3d):
            """插值 (nlat, nlon, nlev) 数据"""
            nlev = data_3d.shape[2]
            ny_model, nx_model = len(model_lat), len(model_lon)
            result = np.zeros((ny_model, nx_model, nlev))
            
            for k in range(nlev):
                interpolator = RegularGridInterpolator(
                    (era5_lat, era5_lon),
                    data_3d[: , :, k],
                    method='linear',
                    bounds_error=False,
                    fill_value=None
                )
                result[:, :, k] = interpolator(points).reshape(ny_model, nx_model)
            
            return result
        
        def interp_2d(data_2d):
            """插值 (nlat, nlon) 数据"""
            interpolator = RegularGridInterpolator(
                (era5_lat, era5_lon),
                data_2d,
                method='linear',
                bounds_error=False,
                fill_value=None
            )
            ny_model, nx_model = len(model_lat), len(model_lon)
            return interpolator(points).reshape(ny_model, nx_model)
        
        # 插值所有变量
        u_interp = interp_3d(era5_data.u)
        v_interp = interp_3d(era5_data.v)
        w_interp = interp_3d(era5_data.w)
        T_interp = interp_3d(era5_data.T)
        q_interp = interp_3d(era5_data.q)
        ps_interp = interp_2d(era5_data.ps)
        
        return ERA5Data(
            u=u_interp,
            v=v_interp,
            w=w_interp,
            T=T_interp,
            q=q_interp,
            ps=ps_interp,
            lat=model_lat,
            lon=model_lon,
            pressure_levels=era5_data.pressure_levels
        )
    
    def _unit_conversion(self, state: AtmosphericState) -> AtmosphericState: 
        """单位转换和范围检查"""
        # ERA5 单位通常是正确的，只需范围检查
        state.q = torch.clamp(state.q, min=0.0, max=0.05)
        state.T = torch.clamp(state.T, min=150.0, max=350.0)
        state.ps = torch.clamp(state.ps, min=30000.0, max=110000.0)
        
        return state
    
    def _quality_check(self, state:  AtmosphericState):
        """数据质量检查"""
        for var_name in ['u', 'v', 'T', 'q', 'ps']:
            var = getattr(state, var_name)
            if torch.isnan(var).any():
                raise ValueError(f"变量 {var_name} 包含 NaN 值")
            if torch.isinf(var).any():
                raise ValueError(f"变量 {var_name} 包含 Inf 值")
        
        # 警告检查
        if (state.ps < 50000).any() or (state.ps > 110000).any():
            print("⚠️  警告：地面气压超出正常范围 [500-1100 hPa]")
        
        if (state.T < 150).any() or (state.T > 350).any():
            print("⚠️  警告：温度超出正常范围 [150-350 K]")