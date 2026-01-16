"""
运行预报脚本
"""

import torch
import argparse
from pathlib import Path
import time

from configs.model_config import ModelConfig
from configs.era5_config import ERA5Config
from models.atmospheric_model import AtmosphericModel
from data.era5_loader import ERA5Data
from data.era5_interpolator import ERA5ToModelConverter
from initialization.init_conditions import InitializationModule
from utils.io_utils import save_forecast_results

def main():
    parser = argparse.ArgumentParser(description='运行大气模型预报')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='配置文件路径')
    parser.add_argument('--era5_file', type=str, default=None,
                       help='ERA5 初始化文件（. nc 或 .npz）')
    parser.add_argument('--forecast_hours', type=float, default=240,
                       help='预报时长（小时）')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备 (cuda/cpu)')
    parser.add_argument('--use_hybrid', action='store_true',
                       help='使用混合模型')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("大气模型预报系统")
    print("="*60)
    
    # 1. 配置模型
    print("\n1. 初始化模型...")
    config = ModelConfig(
        nx=256,
        ny=128,
        nz=40,
        dt=300. 0,
        device=args.device
    )
    
    print(f"   网格:  {config.nx} x {config.ny} x {config.nz}")
    print(f"   设备: {config.device}")
    print(f"   时间步长: {config.dt}秒")
    
    # 2. 创建模型
    model = AtmosphericModel(
        config,
        use_hybrid=args.use_hybrid,
        integrator_scheme='rk4'
    ).to(config.device)
    
    print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 初始化状态
    print("\n2. 初始化状态...")
    
    if args.era5_file:
        # 从 ERA5 数据初始化
        print(f"   从 ERA5 文件加载:  {args.era5_file}")
        era5_config = ERA5Config()
        converter = ERA5ToModelConverter(config, era5_config)
        
        if args.era5_file.endswith('.nc'):
            era5_data = ERA5Data.from_netcdf(args.era5_file)
        else:
            era5_data = ERA5Data.from_numpy(args.era5_file)
        
        initial_state = converter(era5_data, horizontal_interp=True)
    else:
        # 使用理想初始条件
        print("   使用斜压波初始条件")
        init_module = InitializationModule(config)
        initial_state = init_module.baroclinic_wave()
    
    initial_state = initial_state.to(config.device)
    
    # 4. 运行预报
    print(f"\n3. 开始预报 ({args.forecast_hours}小时)...")
    
    start_time = time.time()
    
    trajectory = model.rollout(
        initial_state,
        forecast_hours=args.forecast_hours,
        save_interval_hours=6.0
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n预报完成！")
    print(f"   用时: {elapsed:.2f}秒")
    print(f"   模拟速度: {args.forecast_hours * 3600 / elapsed:.2f}x 实时")
    print(f"   生成 {len(trajectory)} 个时间步")
    
    # 5. 输出诊断
    print("\n4. 诊断信息:")
    for i, state in enumerate(trajectory):
        hours = i * 6
        u_max = state.u.abs().max().item()
        T_min = state.T.min().item()
        T_max = state.T.max().item()
        ps_mean = state.ps.mean().item()
        
        print(f"   +{hours: 3d}h: |u|_max={u_max: 6.2f} m/s, "
              f"T=[{T_min:.1f}, {T_max:.1f}] K, "
              f"ps_mean={ps_mean/100:.1f} hPa")
    
    # 6. 保存结果
    print("\n5. 保存结果...")
    output_file = output_dir / f"forecast_{int(args.forecast_hours)}h.pt"
    
    save_forecast_results(
        trajectory,
        config,
        output_file,
        metadata={
            'forecast_hours':  args.forecast_hours,
            'elapsed_time': elapsed,
            'era5_file': args.era5_file,
            'use_hybrid': args.use_hybrid
        }
    )
    
    print(f"   结果已保存到:  {output_file}")
    print("\n✓ 完成！")

if __name__ == "__main__":
    main()