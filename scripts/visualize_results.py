"""
可视化结果脚本
"""

import torch
import argparse
from pathlib import Path

from utils.io_utils import load_forecast_results
from visualization.plotter import Plotter
from visualization.animator import Animator

def main():
    parser = argparse.ArgumentParser(description='可视化预报结果')
    parser.add_argument('--forecast', type=str, required=True,
                       help='预报结果文件 (. pt)')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='输出目录')
    parser.add_argument('--create_animation', action='store_true',
                       help='创建动画')
    parser.add_argument('--levels', type=int, nargs='+', default=[10, 20],
                       help='要可视化的垂直层')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("可视化预报结果")
    print("="*60)
    
    # 加载结果
    print(f"\n加载预报结果: {args.forecast}")
    trajectory, config, metadata = load_forecast_results(args.forecast)
    
    print(f"  时间步数: {len(trajectory)}")
    print(f"  预报时长: {metadata.get('forecast_hours', 'N/A')} 小时")
    print(f"  网格: {config.nx}x{config.ny}x{config.nz}")
    
    # 重建网格（用于可视化）
    from core.grid import GridGeometry
    grid = GridGeometry(config)
    
    plotter = Plotter()
    
    # 1. 水平切片（不同时间）
    print("\n生成水平切片图...")
    for t_idx in [0, len(trajectory)//2, -1]:
        hours = t_idx * metadata.get('save_interval_hours', 6) if t_idx >= 0 else (len(trajectory)-1) * 6
        
        for level in args.levels:
            plotter.plot_horizontal_slice(
                trajectory[t_idx],
                grid,
                level=level,
                variables=['T', 'wind'],
                save_path=output_dir / f'horizontal_t{hours: 03d}h_lev{level}.png'
            )
    
    # 2. 垂直剖面
    print("\n生成垂直剖面图...")
    plotter.plot_vertical_cross_section(
        trajectory[-1],
        grid,
        lat_idx=config.ny//2,
        save_path=output_dir / 'vertical_section_final.png'
    )
    
    # 3. 时间序列（如果有多个时间步）
    if len(trajectory) > 1:
        print("\n生成时间序列图...")
        import matplotlib.pyplot as plt
        import numpy as np
        
        times = np.arange(len(trajectory)) * 6  # 假设6小时间隔
        T_mean = [state.T.mean().item() for state in trajectory]
        u_max = [state.u.abs().max().item() for state in trajectory]
        ps_mean = [state.ps.mean().item() / 100 for state in trajectory]
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        
        axes[0].plot(times, T_mean, 'o-')
        axes[0].set_ylabel('Mean Temperature (K)')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(times, u_max, 'o-', color='orange')
        axes[1].set_ylabel('Max Wind Speed (m/s)')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(times, ps_mean, 'o-', color='green')
        axes[2].set_ylabel('Mean Surface Pressure (hPa)')
        axes[2].set_xlabel('Forecast Hour')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'time_series.png', dpi=150)
        plt.close()
        print(f"  保存:  time_series.png")
    
    # 4. 动画
    if args.create_animation:
        print("\n创建动画...")
        animator = Animator()
        
        for level in args.levels:
            animator.create_forecast_animation(
                trajectory,
                grid,
                variable='T',
                level=level,
                output_file=output_dir / f'animation_T_lev{level}.mp4',
                fps=5
            )
    
    print(f"\n✓ 完成！所有可视化保存在: {output_dir}")

if __name__ == "__main__":
    main()