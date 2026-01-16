"""
诊断初始场数据
"""

import sys
import os
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.era5_loader import ERA5Data
from configs.model_config import ModelConfig
from core.state import AtmosphericState

def diagnose():
    print("="*60)
    print("诊断初始场数据")
    print("="*60)
    
    timestamp = "2020-01-01 00:00:00"
    
    # 1. 加载ERA5数据
    print(f"\n1. 加载ERA5数据: {timestamp}")
    era5_data = ERA5Data.from_real_data(timestamp)
    
    print(f"   Shape: {era5_data.shape}")
    print(f"   U: [{era5_data.u.min():.2f}, {era5_data.u.max():.2f}] m/s")
    print(f"   V: [{era5_data.v.min():.2f}, {era5_data.v.max():.2f}] m/s")
    print(f"   T: [{era5_data.T.min():.2f}, {era5_data.T.max():.2f}] K")
    print(f"   Q: [{era5_data.q.min():.2e}, {era5_data.q.max():.2e}] kg/kg")
    print(f"   PS: [{era5_data.ps.min():.2f}, {era5_data.ps.max():.2f}] Pa")
    
    # 检查NaN
    has_nan = (np.isnan(era5_data.u).any() or 
               np.isnan(era5_data.v).any() or
               np.isnan(era5_data.T).any() or
               np.isnan(era5_data.q).any() or
               np.isnan(era5_data.ps).any())
    print(f"   含有NaN: {has_nan}")
    
    # 2. 创建模型状态
    print("\n2. 创建模型状态")
    config = ModelConfig(
        nx=era5_data.nlon,
        ny=era5_data.nlat,
        nz=era5_data.nlevels,
        dt=30.0,
        device='cpu'  # 使用CPU便于调试
    )
    
    state = AtmosphericState(config)
    
    # 转换并清理数据
    def clean_data(data, name, default_val=0.0):
        if data is None: return None
        if np.isnan(data).any():
            nan_count = np.isnan(data).sum()
            print(f"  警告: {name} 中含有 {nan_count} 个 NaN")
            if name in ['T', 'ps']:
                mean_val = np.nanmean(data)
                return np.nan_to_num(data, nan=mean_val)
            return np.nan_to_num(data, nan=default_val)
        return data
    
    era5_data.u = clean_data(era5_data.u, 'u', 0.0)
    era5_data.v = clean_data(era5_data.v, 'v', 0.0)
    era5_data.T = clean_data(era5_data.T, 'T', 288.0)
    era5_data.q = clean_data(era5_data.q, 'q', 0.0)
    era5_data.ps = clean_data(era5_data.ps, 'ps', 101325.0)
    
    state.u.data = torch.from_numpy(era5_data.u).float()
    state.v.data = torch.from_numpy(era5_data.v).float()
    state.T.data = torch.from_numpy(era5_data.T).float()
    state.q.data = torch.from_numpy(era5_data.q).float()
    state.ps.data = torch.from_numpy(era5_data.ps).float()
    
    print(f"   状态创建完成")
    print(f"   U: [{state.u.min():.2f}, {state.u.max():.2f}] m/s")
    print(f"   V: [{state.v.min():.2f}, {state.v.max():.2f}] m/s")
    print(f"   T: [{state.T.min():.2f}, {state.T.max():.2f}] K")
    
    # 3. 测试物理诊断
    print("\n3. 测试物理诊断")
    from core.physics import DiagnosticPhysics
    from core.vertical_coordinate import VerticalCoordinate
    
    vertical_coord = VerticalCoordinate(config)
    diagnostics = DiagnosticPhysics(config, vertical_coord)
    
    # 计算气压
    state.p = diagnostics.compute_pressure(state.ps)
    print(f"   压力场: [{state.p.min():.2f}, {state.p.max():.2f}] Pa")
    print(f"   含有NaN: {torch.isnan(state.p).any()}")
    
    # 计算位势
    state.phi = diagnostics.compute_geopotential(state.T, state.p)
    print(f"   位势场: [{state.phi.min():.2f}, {state.phi.max():.2f}] m^2/s^2")
    print(f"   含有NaN: {torch.isnan(state.phi).any()}")
    
    # 检查位势梯度（这决定了气压梯度力）
    from core.derivatives import SpatialDerivatives
    from core.grid import GridGeometry
    
    grid = GridGeometry(config)
    derivatives = SpatialDerivatives(config, grid)
    
    dphidx, dphidy = derivatives.horizontal_gradient(state.phi)
    print(f"   dΦ/dx: [{dphidx.min():.2e}, {dphidx.max():.2e}] m/s^2")
    print(f"   dΦ/dy: [{dphidy.min():.2e}, {dphidy.max():.2e}] m/s^2")
    print(f"   PGF量级: dΦ/dx std={dphidx.std():.2e}, abs_max={dphidx.abs().max():.2e}")
    print(f"   PGF量级: dΦ/dy std={dphidy.std():.2e}, abs_max={dphidy.abs().max():.2e}")
    
    # 4. 测试一步时间积分
    print("\n4. 测试一步时间积分")
    from core.dynamic_core import DynamicCore
    
    core = DynamicCore(config)
    tendencies = core(state)
    
    print(f"   du/dt: [{tendencies['dudt'].min():.2e}, {tendencies['dudt'].max():.2e}] m/s^2")
    print(f"   dv/dt: [{tendencies['dvdt'].min():.2e}, {tendencies['dvdt'].max():.2e}] m/s^2")
    print(f"   dT/dt: [{tendencies['dTdt'].min():.2e}, {tendencies['dTdt'].max():.2e}] K/s")
    print(f"   dps/dt: [{tendencies['dpsdt'].min():.2e}, {tendencies['dpsdt'].max():.2e}] Pa/s")
    
    # 检查NaN
    for key, val in tendencies.items():
        if torch.isnan(val).any():
            print(f"   警告: {key} 包含 NaN!")
    
    print("\n✓ 诊断完成")

if __name__ == "__main__":
    diagnose()
