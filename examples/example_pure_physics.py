"""
示例：纯物理模型预报
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.model_config import ModelConfig
from models.atmospheric_model import AtmosphericModel
from models.init_conditions import InitializationModule
from visualization.plotter import Plotter
from data.era5_loader import ERA5Data

def main():
    print("="*60)
    print("示例：纯物理模型预报 (真实 ERA5 初始化)")
    print("="*60)

    # 1. 读取真实数据
    timestamp = "2020-01-01 00:00:00"
    print(f"\n正在读取 ERA5 数据: {timestamp}...")
    try:
        era5_data = ERA5Data.from_real_data(timestamp)
        print("  数据读取成功！")
    except Exception as e:
        print(f"  数据读取失败: {e}")
        return

    # 2. 插值到sigma层（关键！ERA5是气压层，模型用sigma层）
    print("\n插值ERA5气压层数据到sigma坐标...")
    target_nz = 37  # 使用相同的层数但转换坐标系统
    era5_data = era5_data.interpolate_to_sigma(target_nz=target_nz, sigma_power=2.5)
    print(f"  插值完成：{target_nz}层sigma坐标")
    
    # 3. 根据数据自动配置模型
    print("\n配置模型参数...")
    config = ModelConfig(
        nx=era5_data.nlon,
        ny=era5_data.nlat,
        nz=target_nz,
        dt=900.0,       # 15分钟步长
        lat_min=-90.0,  # ERA5 是全球覆盖
        lat_max=90.0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"  网格:  {config.nx} x {config.ny} x {config.nz}")
    print(f"  设备: {config.device}")
    print(f"  时间步长: {config.dt}秒")
    
    # 4. 创建模型
    print("\n创建模型...")
    model = AtmosphericModel(
        config,
        use_hybrid=False,
        integrator_scheme='rk4'
    )
    
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. 初始化状态 (使用真实数据)
    print("\n初始化状态（从 ERA5 加载）...")
    from core.state import AtmosphericState
    initial_state = AtmosphericState(config)
    
    # 将 numpy 数据转换为 tensor 并移动到设备
    device = torch.device(config.device)
    import numpy as np 

    # 数据清洗：处理 NaN
    def clean_data(data, name, default_val=0.0):
        if data is None: return None
        if np.isnan(data).any():
            nan_count = np.isnan(data).sum()
            print(f"  警告: {name} 中含有 {nan_count} 个 NaN, 使用 {default_val:.2f} 填充")
            # 如果全是 NaN, 就用 default_val 填充
            if nan_count == data.size:
                 return np.full_like(data, default_val)
            
            # 使用现有数据的均值填充 (对于 T 和 PS 更合理)
            if name in ['T', 'ps']:
                 mean_val = np.nanmean(data)
                 print(f"    (使用均值 {mean_val:.2f} 替代默认值)")
                 return np.nan_to_num(data, nan=mean_val)
            
            return np.nan_to_num(data, nan=default_val)
        return data

    era5_data.u = clean_data(era5_data.u, 'u', 0.0)
    era5_data.v = clean_data(era5_data.v, 'v', 0.0)
    era5_data.T = clean_data(era5_data.T, 'T', 288.0)
    era5_data.q = clean_data(era5_data.q, 'q', 0.0)
    era5_data.ps = clean_data(era5_data.ps, 'ps', 101325.0)
    era5_data.w = clean_data(era5_data.w, 'w', 0.0)

    initial_state.u.data = torch.from_numpy(era5_data.u).float().to(device)
    initial_state.v.data = torch.from_numpy(era5_data.v).float().to(device)
    initial_state.T.data = torch.from_numpy(era5_data.T).float().to(device)
    initial_state.q.data = torch.from_numpy(era5_data.q).float().to(device)
    initial_state.ps.data = torch.from_numpy(era5_data.ps).float().to(device)
    
    if era5_data.w is not None:
        initial_state.w.data = torch.from_numpy(era5_data.w).float().to(device)
    
    print(f"  温度范围: [{initial_state.T.min():.1f}, {initial_state.T.max():.1f}] K")
    print(f"  风速范围: [{initial_state.u.min():.1f}, {initial_state.u.max():.1f}] m/s")
    
    # 6. 运行预报
    print("\n运行预报（1小时，分段积分）...")
    import time
    start = time.time()
    
    # 将1小时积分拆分为4个15分钟的片段
    forecast_hours = 1.0
    segment_minutes = 15
    total_segments = int(forecast_hours * 60 / segment_minutes) # 4 segments
    steps_per_segment = int(segment_minutes * 60 / config.dt)   # 3 steps (if dt=300)
    
    current_state = initial_state
    full_trajectory = [initial_state]
    
    for seg in range(total_segments):
        print(f"\n片段 {seg+1}/{total_segments} ({(seg)*segment_minutes}-{(seg+1)*segment_minutes} 分钟)...")
        # 15分钟一段，刚好是 forecast_hours = 0.25 或者使用 dt * steps
        segment_traj = model.rollout(
            current_state, 
            forecast_hours=segment_minutes / 60.0,
            save_interval_hours=segment_minutes / 60.0 # 只保存最后一步
        )
        # segment_traj[0] is current_state, so we skip it to avoid duplication
        full_trajectory.extend(segment_traj[1:])
        current_state = segment_traj[-1]
        
        # 简单诊断
        u_max = current_state.u.max().item()
        v_max = current_state.v.max().item()
        w_spd = torch.sqrt(current_state.u**2 + current_state.v**2).max().item()
        print(f"  当前最大风速: {w_spd:.2f} m/s (u={u_max:.2f}, v={v_max:.2f})")

    trajectory = full_trajectory
    elapsed = time.time() - start
    print(f"  完成！用时: {elapsed:.2f}秒")
    print(f"  模拟速度: {1 * 3600 / elapsed:.2f}x 实时")
    
    # 7. 可视化
    print("\n生成可视化...")
    plotter = Plotter()
    
    # 初始状态
    plotter.plot_horizontal_slice(
        trajectory[0],
        model.core.grid,
        level=10,
        variables=['T', 'wind'],
        save_path='outputs/initial_state.png'
    )
    
    # 最终状态
    plotter.plot_horizontal_slice(
        trajectory[-1],
        model.core.grid,
        level=10,
        variables=['T', 'wind'],
        save_path='outputs/final_state_48h.png'
    )
    
    # 垂直剖面
    plotter.plot_vertical_cross_section(
        trajectory[-1],
        model.core.grid,
        lat_idx=32,
        save_path='outputs/vertical_section.png'
    )
    
    print("\n✓ 完成！结果保存在 outputs/ 目录")

if __name__ == "__main__":
    main()