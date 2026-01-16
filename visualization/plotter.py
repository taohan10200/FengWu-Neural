"""
可视化绘图工具
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from typing import Optional

from core.state import AtmosphericState
from core.grid import GridGeometry

class Plotter:
    """可视化绘图器"""
    
    @staticmethod
    def plot_horizontal_slice(state: AtmosphericState,
                             grid: GridGeometry,
                             level: int = 20,
                             variables: list = ['T', 'wind'],
                             save_path: str = 'horizontal_slice.png'):
        """
        绘制水平切片
        
        Args:
            state: 大气状态
            grid: 网格几何
            level: 垂直层索引
            variables: 要绘制的变量
            save_path: 保存路径
        """
        # 移到 CPU
        lon = grid.lon.cpu().numpy()
        lat = grid.lat.cpu().numpy()
        LON, LAT = np.meshgrid(lon, lat)
        
        n_vars = len(variables)
        fig, axes = plt.subplots(n_vars, 1, figsize=(14, 5 * n_vars),
                                subplot_kw={'projection': ccrs.PlateCarree()})
        
        if n_vars == 1:
            axes = [axes]
        
        for ax, var in zip(axes, variables):
            if var == 'T':
                # Use ellipsis to slice the last dimension (level) regardless of batch dim
                data = state.T[..., level].cpu().numpy()
                if data.ndim > 2: data = data.squeeze()
                levels = np.linspace(data.min(), data.max(), 20)
                contour = ax.contourf(LON, LAT, data, levels=levels, 
                                     cmap='RdYlBu_r', transform=ccrs.PlateCarree())
                plt.colorbar(contour, ax=ax, label='Temperature (K)')
                ax.set_title(f'Temperature at Level {level}')
                
            elif var == 'wind': 
                u = state.u[..., level].cpu().numpy()
                v = state.v[..., level].cpu().numpy()
                if u.ndim > 2: u = u.squeeze()
                if v.ndim > 2: v = v.squeeze()
                speed = np.sqrt(u**2 + v**2)
                
                contour = ax.contourf(LON, LAT, speed, levels=20, 
                                     cmap='viridis', transform=ccrs.PlateCarree())
                
                # 风矢量（降采样）
                skip = 8
                ax.quiver(LON[::skip, ::skip], LAT[::skip, ::skip],
                         u[::skip, ::skip], v[::skip, ::skip],
                         transform=ccrs.PlateCarree(), alpha=0.7)
                
                plt.colorbar(contour, ax=ax, label='Wind Speed (m/s)')
                ax.set_title(f'Wind Field at Level {level}')
            
            elif var == 'ps':
                data = state.ps.cpu().numpy() / 100  # Pa -> hPa
                levels = np.linspace(data.min(), data.max(), 20)
                contour = ax.contourf(LON, LAT, data, levels=levels,
                                     cmap='RdBu_r', transform=ccrs.PlateCarree())
                plt.colorbar(contour, ax=ax, label='Surface Pressure (hPa)')
                ax.set_title('Surface Pressure')
            
            # 添加地理特征
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.gridlines(draw_labels=True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 图像已保存: {save_path}")
    
    @staticmethod
    def plot_vertical_cross_section(state: AtmosphericState,
                                   grid:  GridGeometry,
                                   lat_idx: int = None,
                                   lon_idx: int = None,
                                   save_path: str = 'cross_section.png'):
        """
        绘制垂直剖面
        
        Args:
            state: 大气状态
            grid: 网格几何
            lat_idx: 纬度索引（纬向剖面）
            lon_idx: 经度索引（经向剖面）
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        if lat_idx is not None: 
            # 纬向剖面
            u = state.u[lat_idx, :, : ].cpu().numpy()
            T = state.T[lat_idx, :, :].cpu().numpy()
            lon = grid.lon.cpu().numpy()
            
            x_label = 'Longitude (°)'
            x_data = lon
            title_suffix = f'at Lat={grid.lat[lat_idx]:. 1f}°'
            
        elif lon_idx is not None: 
            # 经向剖面
            u = state.u[:, lon_idx, :].cpu().numpy()
            T = state.T[: , lon_idx, :].cpu().numpy()
            lat = grid.lat.cpu().numpy()
            
            x_label = 'Latitude (°)'
            x_data = lat
            title_suffix = f'at Lon={grid.lon[lon_idx]:.1f}°'
        else: 
            raise ValueError("必须指定 lat_idx 或 lon_idx")
        
        # 创建网格
        nz = state.u.shape[2]
        levels = np.arange(nz)
        X, Z = np.meshgrid(x_data, levels)
        
        # 纬向风
        c1 = axes[0].contourf(X, Z, u.T, levels=20, cmap='RdBu_r')
        axes[0].set_ylabel('Level Index')
        axes[0].set_xlabel(x_label)
        axes[0].invert_yaxis()
        plt.colorbar(c1, ax=axes[0], label='Zonal Wind (m/s)')
        axes[0].set_title(f'Zonal Wind - Vertical Cross-Section {title_suffix}')
        axes[0].grid(True, alpha=0.3)
        
        # 温度
        c2 = axes[1].contourf(X, Z, T.T, levels=20, cmap='RdYlBu_r')
        axes[1].set_ylabel('Level Index')
        axes[1].set_xlabel(x_label)
        axes[1].invert_yaxis()
        plt.colorbar(c2, ax=axes[1], label='Temperature (K)')
        axes[1].set_title(f'Temperature - Vertical Cross-Section {title_suffix}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 图像已保存: {save_path}")
    
    @staticmethod
    def plot_forecast_comparison(pred_states: list,
                                target_states: list,
                                grid: GridGeometry,
                                variable: str = 'T',
                                level: int = 20,
                                save_path: str = 'comparison.png'):
        """
        对比预报与真实值
        
        Args:
            pred_states: 预测状态列表
            target_states: 目标状态列表
            grid: 网格几何
            variable: 变量名
            level: 垂直层
            save_path: 保存路径
        """
        n_times = len(pred_states)
        fig, axes = plt.subplots(n_times, 3, figsize=(18, 5 * n_times),
                                subplot_kw={'projection': ccrs.PlateCarree()})
        
        if n_times == 1:
            axes = axes.reshape(1, -1)
        
        lon = grid.lon.cpu().numpy()
        lat = grid.lat.cpu().numpy()
        LON, LAT = np.meshgrid(lon, lat)
        
        for t in range(n_times):
            pred_data = getattr(pred_states[t], variable)[:, :, level].cpu().numpy()
            target_data = getattr(target_states[t], variable)[:, :, level].cpu().numpy()
            error = pred_data - target_data
            
            # 预测
            im1 = axes[t, 0].contourf(LON, LAT, pred_data, levels=20,
                                     cmap='RdYlBu_r', transform=ccrs.PlateCarree())
            axes[t, 0].coastlines()
            axes[t, 0].set_title(f'Prediction +{t*6}h')
            plt.colorbar(im1, ax=axes[t, 0])
            
            # 真实
            im2 = axes[t, 1].contourf(LON, LAT, target_data, levels=20,
                                     cmap='RdYlBu_r', transform=ccrs.PlateCarree())
            axes[t, 1].coastlines()
            axes[t, 1].set_title(f'Target +{t*6}h')
            plt.colorbar(im2, ax=axes[t, 1])
            
            # 误差
            im3 = axes[t, 2].contourf(LON, LAT, error, levels=20,
                                     cmap='seismic', transform=ccrs.PlateCarree())
            axes[t, 2].coastlines()
            axes[t, 2].set_title(f'Error +{t*6}h (RMSE={np.sqrt(np.mean(error**2)):.2f})')
            plt.colorbar(im3, ax=axes[t, 2])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 对比图已保存: {save_path}")