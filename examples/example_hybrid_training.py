"""
示例：混合模型训练
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
sys.path.append('..')

from configs.model_config import ModelConfig
from configs.era5_config import ERA5Config
from models.atmospheric_model import AtmosphericModel
from training.trainer import ModelTrainer
from training.loss import CombinedLoss
from training.metrics import get_all_metrics
from data.era5_dataset import ERA5Dataset

def create_dummy_dataset():
    """创建虚拟数据集用于演示"""
    import numpy as np
    from pathlib import Path
    from data.era5_loader import ERA5Data
    
    data_dir = Path('dummy_data')
    data_dir.mkdir(exist_ok=True)
    
    print("创建虚拟训练数据...")
    
    era5_config = ERA5Config()
    
    for i in range(20):  # 20个样本
        nlat, nlon = 91, 180
        nlevels = 37
        
        # 随机但合理的大气数据
        u = np.random.randn(nlat, nlon, nlevels) * 10 + 5
        v = np.random.randn(nlat, nlon, nlevels) * 10
        w = np.random.randn(nlat, nlon, nlevels) * 0.01
        T = np.random.randn(nlat, nlon, nlevels) * 10 + 280
        q = np.random.rand(nlat, nlon, nlevels) * 0.01
        ps = np.random.randn(nlat, nlon) * 1000 + 101325
        
        lat = np.linspace(-90, 90, nlat)
        lon = np.linspace(0, 360, nlon, endpoint=False)
        
        era5_data = ERA5Data(
            u=u, v=v, w=w, T=T, q=q, ps=ps,
            lat=lat, lon=lon,
            pressure_levels=era5_config.pressure_levels
        )
        
        era5_data.save_numpy(f'dummy_data/era5_{i: 04d}.npz')
    
    print(f"  创建了 20 个样本在 {data_dir}/")
    return str(data_dir)

def main():
    print("="*60)
    print("示例：混合模型训练（物理 + AI）")
    print("="*60)
    
    # 1. 配置
    config = ModelConfig(
        nx=90,
        ny=45,
        nz=15,
        dt=300.0,
        use_hybrid_model=True,
        learnable_physics=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\n配置:")
    print(f"  网格: {config.nx} x {config.ny} x {config.nz}")
    print(f"  设备: {config.device}")
    print(f"  混合模型: {config.use_hybrid_model}")
    
    # 2. 创建数据集
    data_dir = create_dummy_dataset()
    
    era5_config = ERA5Config()
    
    # 注意：实际使用时应该有训练/验证分割
    dataset = ERA5Dataset(
        data_dir=data_dir,
        config=config,
        era5_config=era5_config,
        forecast_hours=24,
        file_pattern="era5_*.npz"
    )
    
    # 简单分割
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    print(f"\n数据集:")
    print(f"  训练样本: {train_size}")
    print(f"  验证样本: {val_size}")
    
    # 3. 创建混合模型
    print("\n创建混合模型...")
    model = AtmosphericModel(
        config,
        use_hybrid=True,
        integrator_scheme='rk4'
    )
    
    print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  可训练参数:  {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 4. 设置训练
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    loss_fn = CombinedLoss(
        config,
        mse_weight=1.0,
        spectral_weight=0.1,
        conservation_weight=0.01
    )
    
    metrics = get_all_metrics()
    
    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        device=config.device,
        gradient_clip=1.0
    )
    
    print("\n训练配置:")
    print(f"  优化器: Adam (lr=1e-4)")
    print(f"  损失函数: Combined (MSE + Spectral + Conservation)")
    print(f"  评估指标: {list(metrics.keys())}")
    
    # 5. 训练循环
    print("\n开始训练...")
    n_epochs = 5
    
    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch {epoch}/{n_epochs}")
        print("-" * 60)
        
        # 训练
        train_loss = trainer.train_epoch(train_loader, loss_fn, epoch*
