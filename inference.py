"""
Inference Script for Neural NWP Model
神经网络数值气象预测模型推理脚本
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation
import argparse
import os
from typing import List, Optional

from neural_nwp.models.model import NeuralNWP, create_model


class NWPPredictor:
    """气象预测器"""
    
    def __init__(self, 
                 model: NeuralNWP,
                 checkpoint_path: str,
                 device: torch.device):
        
        self.model = model.to(device)
        self.device = device
        
        # 加载检查点
        self.load_checkpoint(checkpoint_path)
        self.model.eval()
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载模型权重"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
    
    @torch.no_grad()
    def predict(self, 
                initial_state: torch.Tensor,
                num_steps: int,
                dt: float = 3600.0) -> torch.Tensor:
        """
        多步预测
        
        Args:
            initial_state: [batch, lat, lon, channels] 初始状态
            num_steps: 预测步数
            dt: 时间步长（秒）
            
        Returns:
            predictions: [batch, num_steps, lat, lon, channels]
        """
        initial_state = initial_state.to(self.device)
        
        predictions = self.model.rollout(
            initial_state=initial_state,
            num_steps=num_steps,
            dt=dt
        )
        
        return predictions
    
    def predict_single_step(self,
                           initial_state: torch.Tensor,
                           dt: float = 3600.0) -> torch.Tensor:
        """
        单步预测
        
        Args:
            initial_state: [batch, lat, lon, channels]
            dt: 时间步长（秒）
            
        Returns:
            prediction: [batch, lat, lon, channels]
        """
        initial_state = initial_state.to(self.device)
        prediction = self.model(initial_state, dt=dt)
        return prediction


def visualize_prediction(
    prediction: np.ndarray,
    variable_idx: int = 0,
    level_idx: int = 0,
    num_vars: int = 5,
    num_levels: int = 37,
    save_path: Optional[str] = None
):
    """
    可视化预测结果
    
    Args:
        prediction: [num_steps, lat, lon, channels] 预测结果
        variable_idx: 变量索引 (0=T, 1=u, 2=v, 3=q, 4=sp)
        level_idx: 层次索引
        num_vars: 变量数量
        num_levels: 层次数量
        save_path: 保存路径
    """
    num_steps, lat, lon, channels = prediction.shape
    
    # 重塑为 [num_steps, lat, lon, num_vars, num_levels]
    pred_reshaped = prediction.reshape(num_steps, lat, lon, num_vars, num_levels)
    
    # 提取特定变量和层次
    data = pred_reshaped[:, :, :, variable_idx, level_idx]
    
    # 变量名称
    var_names = ['Temperature', 'U-wind', 'V-wind', 'Specific Humidity', 'Surface Pressure']
    var_name = var_names[variable_idx]
    
    # 创建图形
    fig = plt.figure(figsize=(15, 10))
    
    # 绘制多个时间步
    plot_steps = min(6, num_steps)
    for i in range(plot_steps):
        ax = fig.add_subplot(2, 3, i + 1, projection=ccrs.PlateCarree())
        
        # 绘制数据
        im = ax.contourf(data[i], levels=20, cmap='RdBu_r', transform=ccrs.PlateCarree())
        
        # 添加地图特征
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        
        ax.set_title(f'{var_name} - Step {i+1}')
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_animation(
    prediction: np.ndarray,
    variable_idx: int = 0,
    level_idx: int = 0,
    num_vars: int = 5,
    num_levels: int = 37,
    save_path: str = 'forecast_animation.gif',
    fps: int = 2
):
    """
    创建预测动画
    
    Args:
        prediction: [num_steps, lat, lon, channels]
        variable_idx: 变量索引
        level_idx: 层次索引
        num_vars: 变量数量
        num_levels: 层次数量
        save_path: 保存路径
        fps: 帧率
    """
    num_steps, lat, lon, channels = prediction.shape
    
    # 重塑数据
    pred_reshaped = prediction.reshape(num_steps, lat, lon, num_vars, num_levels)
    data = pred_reshaped[:, :, :, variable_idx, level_idx]
    
    var_names = ['Temperature', 'U-wind', 'V-wind', 'Specific Humidity', 'Surface Pressure']
    var_name = var_names[variable_idx]
    
    # 创建图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # 确定颜色范围
    vmin, vmax = data.min(), data.max()
    
    def update(frame):
        ax.clear()
        
        # 绘制数据
        im = ax.contourf(data[frame], levels=20, cmap='RdBu_r', 
                        vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        
        # 添加地图特征
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.gridlines(draw_labels=True)
        
        ax.set_title(f'{var_name} - Forecast Step {frame+1}/{num_steps}')
        
        return im,
    
    # 创建动画
    anim = FuncAnimation(fig, update, frames=num_steps, interval=1000//fps, blit=False)
    
    # 保存
    anim.save(save_path, writer='pillow', fps=fps)
    print(f"✓ Saved animation to {save_path}")
    
    plt.close()


def compare_with_ground_truth(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    variable_idx: int = 0,
    level_idx: int = 0,
    num_vars: int = 5,
    num_levels: int = 37,
    save_path: Optional[str] = None
):
    """
    对比预测和真值
    
    Args:
        prediction: [num_steps, lat, lon, channels]
        ground_truth: [num_steps, lat, lon, channels]
        variable_idx: 变量索引
        level_idx: 层次索引
        num_vars: 变量数量
        num_levels: 层次数量
        save_path: 保存路径
    """
    num_steps = min(prediction.shape[0], ground_truth.shape[0])
    
    # 重塑数据
    pred_reshaped = prediction.reshape(num_steps, -1, num_vars, num_levels)
    gt_reshaped = ground_truth.reshape(num_steps, -1, num_vars, num_levels)
    
    pred_data = pred_reshaped[:, :, :, variable_idx, level_idx]
    gt_data = gt_reshaped[:, :, :, variable_idx, level_idx]
    
    var_names = ['Temperature', 'U-wind', 'V-wind', 'Specific Humidity', 'Surface Pressure']
    var_name = var_names[variable_idx]
    
    # 创建图形
    fig, axes = plt.subplots(num_steps, 3, figsize=(18, 5*num_steps))
    
    if num_steps == 1:
        axes = axes.reshape(1, -1)
    
    for t in range(num_steps):
        # 预测
        im1 = axes[t, 0].imshow(pred_data[t], cmap='RdBu_r')
        axes[t, 0].set_title(f'Prediction - Step {t+1}')
        plt.colorbar(im1, ax=axes[t, 0])
        
        # 真值
        im2 = axes[t, 1].imshow(gt_data[t], cmap='RdBu_r')
        axes[t, 1].set_title(f'Ground Truth - Step {t+1}')
        plt.colorbar(im2, ax=axes[t, 1])
        
        # 误差
        error = pred_data[t] - gt_data[t]
        im3 = axes[t, 2].imshow(error, cmap='seismic')
        axes[t, 2].set_title(f'Error - Step {t+1}')
        plt.colorbar(im3, ax=axes[t, 2])
    
    fig.suptitle(f'{var_name} Comparison', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Neural NWP Inference')
    
    # 模型参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--img_size', type=int, nargs=2, default=[128, 256],
                        help='Image size (lat, lon)')
    parser.add_argument('--num_vars', type=int, default=5,
                        help='Number of variables')
    parser.add_argument('--num_levels', type=int, default=37,
                        help='Number of vertical levels')
    
    # 推理参数
    parser.add_argument('--num_steps', type=int, default=10,
                        help='Number of forecast steps')
    parser.add_argument('--dt', type=float, default=3600.0,
                        help='Time step in seconds')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    
    # 可视化参数
    parser.add_argument('--variable', type=int, default=0,
                        help='Variable to visualize (0=T, 1=u, 2=v, 3=q, 4=sp)')
    parser.add_argument('--level', type=int, default=0,
                        help='Vertical level to visualize')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--create_animation', action='store_true',
                        help='Create animation')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    print("Creating model...")
    model_config = {
        'img_size': tuple(args.img_size),
        'num_vars': args.num_vars,
        'num_levels': args.num_levels
    }
    model = create_model(model_config)
    
    # 创建预测器
    predictor = NWPPredictor(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device
    )
    
    # 创建随机初始状态（实际应用中应该从真实数据加载）
    print("Creating initial state...")
    lat, lon = args.img_size
    channels = args.num_vars * args.num_levels
    initial_state = torch.randn(args.batch_size, lat, lon, channels)
    
    # 预测
    print(f"Running forecast for {args.num_steps} steps...")
    predictions = predictor.predict(
        initial_state=initial_state,
        num_steps=args.num_steps,
        dt=args.dt
    )
    
    print(f"Prediction shape: {predictions.shape}")
    
    # 转为numpy
    pred_np = predictions[0].cpu().numpy()  # 取第一个batch
    
    # 可视化
    print("Creating visualizations...")
    viz_path = os.path.join(args.output_dir, 'forecast_steps.png')
    visualize_prediction(
        prediction=pred_np,
        variable_idx=args.variable,
        level_idx=args.level,
        num_vars=args.num_vars,
        num_levels=args.num_levels,
        save_path=viz_path
    )
    
    # 创建动画
    if args.create_animation:
        print("Creating animation...")
        anim_path = os.path.join(args.output_dir, 'forecast_animation.gif')
        create_animation(
            prediction=pred_np,
            variable_idx=args.variable,
            level_idx=args.level,
            num_vars=args.num_vars,
            num_levels=args.num_levels,
            save_path=anim_path
        )
    
    print("\n✓ Inference completed!")


if __name__ == '__main__':
    main()
