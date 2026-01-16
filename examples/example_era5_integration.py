"""
示例：ERA5 数据集成
"""

import torch
import numpy as np
import sys
sys.path.append('..')

from configs.model_config import ModelConfig
from configs.era5_config import ERA5Config
from models.atmospheric_model import AtmosphericModel
from data.era5_loader import ERA5Data
from data.era5_interpolator import ERA5ToModelConverter
from visualization.plotter import Plotter
from utils.io_utils import save_forecast_results

def create_mock_era5_data():
    """创建模拟的 ERA5 数据（用于演示）"""
    print("���建模拟 ERA5 数据...")
    
    nlat, nlon = 181, 360  # 1° 分辨率
    nlevels = 37
    
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    
    # 模拟真实的大气结构
    LAT, LON = np.meshgrid(lat, lon, indexing='ij')
    
    # 气压层
    era5_config = ERA5Config()
    p_levels = era5_config.pressure_levels * 100  # hPa -> Pa
    
    u = np.zeros((nlat, nlon, nlevels))
    v = np.zeros((nlat, nlon, nlevels))
    w = np.zeros((nlat, nlon, nlevels))
    T = np.zeros((nlat, nlon, nlevels))
    q = np.zeros((nlat, nlon, nlevels))
    
    for k, p in enumerate(p_levels):
        # 温度：标准大气 + 纬度依赖
        T[:, : , k] = 288 - 6.5 * (1 - p/101325) * 10000 - 30 * np.abs(LAT) / 90
        
        # 纬向风：急流结构
        u[:, :, k] = 30 * np.exp(-((p - 25000) / 15000)**2) * np.sin(2 * np.deg2rad(LAT))**2
        
        # 经向风：小扰动
        v[:, : , k] = 5 * np.sin(4 * np.deg2rad(LON)) * np.cos(np.deg2rad(LAT))
        
        # 垂直速度
        w[:, :, k] = 0.01 * np.sin(2 * np.deg2rad(LAT))
        
        # 比湿：指数衰减
        q[: , :, k] = 0.01 * np.exp(-p / 50000) * np.maximum(0, np.cos(np.deg2rad(LAT)))
    
    # 地面气压
    ps = 101325 - 500 * np.sin(2 * np.deg2rad(LAT))
    
    return ERA5Data(
        u=u, v=v, w=w, T=T, q=q, ps=ps,
        lat=lat, lon=lon,
        pressure_levels=era5_config.pressure_levels
    )

def main():
    print("="*60)
    print("示例：ERA5 数据集成")
    print("="*60)
    
    # 1. 准备 ERA5 数据
    era5_data = create_mock_era5_data()
    
    print(f"\nERA5 数据:")
    print(f"  形��: {era5_data.shape}")
    print(f"  纬度: [{era5_data.lat.min()}, {era5_data.lat.max()}]")
    print(f"  经度: [{era5_data.lon.min()}, {era5_data.lon.max()}]")
    print(f"  气压层数: {era5_data.nlevels}")
    
    # 保存为 . npz（可选）
    era5_data.save_numpy('outputs/era5_sample.npz')
    
    # 2. 配置模型
    model_config = ModelConfig(
        nx=180,  # 2° 分辨率
        ny=90,
        nz=30,   # 30 层（不同于 ERA5 的 37 层）
        dt=300.0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    era5_config = ERA5Config()
    
    print(f"\n模型配置:")
    print(f"  目标网格: {model_config.nx} x {model_config.ny} x {model_config.nz}")
    print(f"  垂直插值: {era5_config.n_pressure_levels} 层 -> {model_config.nz} 层")
    
    # 3. 转换 ERA5 数据
    print("\n转换 ERA5 -> 模型坐标...")
    print("  步骤 1: 水平插值（ERA5 网格 -> 模型网格）")
    print("  步骤 2: 垂直插值（气压层 -> σ坐标）")
    
    import time
    start = time.time()
    
    converter = ERA5ToModelConverter(model_config, era5_config)
    model_state = converter(era5_data, horizontal_interp=True)
    
    elapsed = time.time() - start
    print(f"  完成！用时: {elapsed:. 2f}秒")
    
    # 验证
    print(f"\n转换后的模型状态:")
    print(f"  u:  shape={model_state.u.shape}, range=[{model_state.u.min():.2f}, {model_state.u.max():.2f}] m/s")
    print(f"  T: shape={model_state.T.shape}, range=[{model_state.T.min():.2f}, {model_state.T.max():.2f}] K")
    print(f"  ps: shape={model_state.ps.shape}, range=[{model_state.ps.min():.2f}, {model_state.ps.max():.2f}] Pa")
    
    # 4. 运行预报
    print("\n运行模型预报（120小时）...")
    model = AtmosphericModel(model_config, use_hybrid=False)
    
    start = time.time()
    trajectory = model.rollout(
        model_state,
        forecast_hours=120,
        save_interval_hours=12
    )
    elapsed = time.time() - start
    
    print(f"  完成！用时: {elapsed:.2f}秒")
    print(f"  生成 {len(trajectory)} ��时间步")
    
    # 5. 可视化
    print("\n生成可视化...")
    plotter = Plotter()
    
    # 初始状态
    plotter.plot_horizontal_slice(
        trajectory[0],
        model.core.grid,
        level=15,
        variables=['T', 'wind', 'ps'],
        save_path='outputs/era5_initial.png'
    )
    
    # 120小时预报
    plotter.plot_horizontal_slice(
        trajectory[-1],
        model.core.grid,
        level=15,
        variables=['T', 'wind', 'ps'],
        save_path='outputs/era5_forecast_120h.png'
    )
    
    # 6. 保存结果
    print("\n保存结果...")
    save_forecast_results(
        trajectory,
        model_config,
        'outputs/era5_forecast.pt',
        metadata={
            'source': 'ERA5 (simulated)',
            'forecast_hours': 120,
            'vertical_levels': model_config.nz
        }
    )
    
    print("\n✓ 完成！结果保存在 outputs/ 目录")
    print("\n说明:")
    print("  - ERA5 37层气压数据 -> 模型30层σ坐标")
    print("  - 水平分辨率: 1° -> 2°")
    print("  - 可替换 create_mock_era5_data() 为真实 ERA5 文件加载")

if __name__ == "__main__":
    main()