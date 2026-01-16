
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到 path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.era5_loader import ERA5Data
from configs.model_config import ModelConfig

def test_real_data_loading():
    print("="*60)
    print("测试：读取真实 ERA5 数据")
    print("="*60)
    
    # 时间戳 (确保在你的数据范围内)
    timestamp = "2020-01-01 00:00:00" 
    
    try:
        print(f"正在读取 {timestamp} 的数据...")
        
        # 使用 from_real_data 读取
        # 如果需要指定 data_root, 可以在这里传入
        # data_root = "/path/to/data" 
        # era5_data = ERA5Data.from_real_data(timestamp, data_root=data_root)
        
        era5_data = ERA5Data.from_real_data(timestamp)
        
        print("\n数据读取成功！")
        print(f"维度信息:")
        print(f"  Shape (u): {era5_data.shape}")
        print(f"  Lat: {era5_data.nlat}")
        print(f"  Lon: {era5_data.nlon}")
        print(f"  Levels: {era5_data.nlevels}")
        
        print(f"\n变量范围:")
        print(f"  U: [{era5_data.u.min():.2f}, {era5_data.u.max():.2f}]")
        print(f"  V: [{era5_data.v.min():.2f}, {era5_data.v.max():.2f}]")
        print(f"  T: [{era5_data.T.min():.2f}, {era5_data.T.max():.2f}]")
        print(f"  Q: [{era5_data.q.min():.2e}, {era5_data.q.max():.2e}]")
        print(f"  PS: [{era5_data.ps.min():.2f}, {era5_data.ps.max():.2f}]")
        
        # 简单可视化验证
        print("\n生成可视化验证...")
        plt.figure(figsize=(15, 10))
        
        # U wind at level 20 (approx 500hPa if levels are sorted high to low pressure)
        level_idx = 20 if era5_data.nlevels > 20 else 0
        plt.subplot(2, 2, 1)
        plt.imshow(era5_data.u[:, :, level_idx], cmap='RdBu_r')
        plt.colorbar(label='m/s')
        plt.title(f'U Wind (Level {level_idx})')
        
        # Temperature
        plt.subplot(2, 2, 2)
        plt.imshow(era5_data.T[:, :, level_idx], cmap='viridis')
        plt.colorbar(label='K')
        plt.title(f'Temperature (Level {level_idx})')
        
        # Surface Pressure
        plt.subplot(2, 2, 3)
        plt.imshow(era5_data.ps, cmap='rainbow')
        plt.colorbar(label='Pa')
        plt.title('Surface Pressure')
        
        # Q
        plt.subplot(2, 2, 4)
        plt.imshow(era5_data.q[:, :, level_idx], cmap='Blues')
        plt.colorbar(label='kg/kg')
        plt.title(f'Specific Humidity (Level {level_idx})')
        
        save_path = 'outputs/test_real_data.png'
        if not os.path.exists('outputs'):
            os.makedirs('outputs')
        plt.savefig(save_path)
        print(f"可视化已保存至: {save_path}")
        
    except Exception as e:
        print(f"\n错误: 读取失败")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_data_loading()
