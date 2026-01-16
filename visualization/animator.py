"""
动画生成工具
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
from typing import List

from core.state import AtmosphericState
from core.grid import GridGeometry

class Animator: 
    """动画生成器"""
    
    @staticmethod
    def create_forecast_animation(trajectory: List[AtmosphericState],
                                 grid: GridGeometry,
                                 variable: str = 'T',
                                 level: int = 20,
                                 output_file: str = 'forecast.mp4',
                                 fps: int = 5):
        """
        创建预报动画
        
        Args:
            trajectory: 状态轨迹
            grid: 网格几何
            variable: 变量名
            level: 垂直层
            output_file: 输出文件
            fps: 帧率
        """
        fig, ax = plt.subplots(figsize=(12, 8),
                              subplot_kw={'projection': ccrs.PlateCarree()})
        
        lon = grid.lon.cpu().numpy()
        lat = grid.lat.cpu().numpy()
        LON, LAT = np.meshgrid(lon, lat)
        
        # 计算全局范围（保持色标一致）
        all_data = [getattr(state, variable)[:, :, level].cpu().numpy() 
                   for state in trajectory]
        vmin = min(d.min() for d in all_data)
        vmax = max(d.max() for d in all_data)
        
        def update(frame):
            ax.clear()
            data = all_data[frame]
            
            im = ax.contourf(LON, LAT, data, levels=20, 
                           cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree())
            
            ax.coastlines()
            ax.gridlines(draw_labels=True, alpha=0.3)
            ax.set_title(f'{variable} at Level {level} - Forecast Hour {frame * 6}')
            
            return [im]
        
        anim = animation.FuncAnimation(fig, update, frames=len(trajectory),
                                      interval=1000//fps, blit=False)
        
        # 保存
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Atmospheric Model'), bitrate=1800)
        anim.save(output_file, writer=writer)
        
        plt.close()
        print(f"✓ 动画已保存: {output_file}")