"""
模型评估脚本
"""

import torch
import argparse
from pathlib import Path
import json

from configs.model_config import ModelConfig
from models.atmospheric_model import AtmosphericModel
from data.era5_loader import ERA5Data
from data.era5_interpolator import ERA5ToModelConverter
from configs.era5_config import ERA5Config
from training.metrics import get_all_metrics
from visualization.plotter import Plotter

def main():
    parser = argparse.ArgumentParser(description='评估大气模型')
    parser.add_argument('--model', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--test_data', type=str, required=True,
                       help='测试数据目录')
    parser.add_argument('--output_dir', type=str, default='evaluation',
                       help='输出目录')
    parser.add_argument('--forecast_hours', type=float, default=120,
                       help='预报时长（小时）')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("模型评估")
    print("="*60)
    
    # 加载模型
    print(f"\n加载模型:  {args.model}")
    checkpoint = torch.load(args.model, map_location=args.device)
    config = checkpoint['config']
    
    model = AtmosphericModel(config, use_hybrid=config.use_hybrid_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  模型配置:  {config.nx}x{config.ny}x{config.nz}")
    
    # 加载测试数据
    print(f"\n加载测试数据: {args.test_data}")
    test_files = sorted(Path(args.test_data).glob('era5_*.npz'))
    
    if len(test_files) == 0:
        raise ValueError(f"在 {args.test_data} 中找不到测试文件")
    
    print(f"  找到 {len(test_files)} 个测试样本")
    
    # 转换器
    era5_config = ERA5Config()
    converter = ERA5ToModelConverter(config, era5_config)
    
    # 评估指标
    metrics = get_all_metrics()
    all_results = []
    
    # 对每个测试样本进行评估
    print("\n开始评估...")
    
    for i, test_file in enumerate(test_files[: 10]):  # 限制前10个
        print(f"\n测试样本 {i+1}/{min(10, len(test_files))}: {test_file.name}")
        
        # 加载初始状态
        era5_data = ERA5Data.from_numpy(str(test_file))
        initial_state = converter(era5_data, horizontal_interp=True)
        
        # 运行预报
        with torch.no_grad():
            trajectory = model.rollout(
                initial_state,
                forecast_hours=args.forecast_hours,
                save_interval_hours=24
            )
        
        print(f"  生成 {len(trajectory)} 个预报时间步")
        
        # 可视化
        plotter = Plotter()
        plotter.plot_horizontal_slice(
            trajectory[-1],
            model.core.grid,
            level=10,
            variables=['T', 'wind'],
            save_path=output_dir / f'forecast_{i: 03d}.png'
        )
        
        # 记录结果
        result = {
            'file': test_file.name,
            'forecast_hours': args.forecast_hours,
            'n_timesteps': len(trajectory),
            'final_state': {
                'T_mean': trajectory[-1].T.mean().item(),
                'u_max': trajectory[-1].u.abs().max().item(),
                'ps_mean': trajectory[-1].ps.mean().item()
            }
        }
        all_results.append(result)
    
    # 保存结果
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ 评估完成！")
    print(f"  结果保存在: {output_dir}")
    print(f"  评估报告: {results_file}")

if __name__ == "__main__":
    main()