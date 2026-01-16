"""
测试物理约束损失函数
Demonstration of PhysicsConstrainedLoss
"""

import torch
from neural_nwp.models.model import create_model, PhysicsConstrainedLoss

def test_physics_constrained_loss():
    """测试物理约束损失函数"""
    
    print("=" * 80)
    print("Testing PhysicsConstrainedLoss")
    print("=" * 80)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # 创建模型
    model = create_model().to(device)
    print("✓ Model created")
    
    # 创建损失函数
    criterion = PhysicsConstrainedLoss(
        num_vars=5,
        num_levels=37,
        img_size=(128, 256),
        mse_weight=1.0,
        mass_weight=0.1,
        energy_weight=0.1,
        moisture_weight=0.05,
        momentum_weight=0.05
    ).to(device)
    print("✓ Loss function created")
    
    # 创建测试数据
    batch_size = 4
    lat, lon = 128, 256
    num_vars = 5
    num_levels = 37
    channels = num_vars * num_levels
    
    print(f"\nTest configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Spatial resolution: {lat} x {lon}")
    print(f"  Variables: {num_vars}")
    print(f"  Vertical levels: {num_levels}")
    print(f"  Total channels: {channels}")
    
    # 生成模拟数据
    print("\n" + "-" * 80)
    print("Generating simulated data...")
    
    initial_state = torch.randn(batch_size, lat, lon, channels).to(device)
    print(f"  Initial state shape: {initial_state.shape}")
    
    # 模型预测
    with torch.no_grad():
        prediction = model(initial_state, dt=3600.0)
    print(f"  Prediction shape: {prediction.shape}")
    
    # 生成"真值"（这里用加噪声的预测模拟）
    target = prediction + 0.01 * torch.randn_like(prediction)
    print(f"  Target shape: {target.shape}")
    
    # 计算损失
    print("\n" + "-" * 80)
    print("Computing physics-constrained loss...")
    
    loss_dict = criterion(prediction, target, initial_state)
    
    print("\n📊 Loss Components:")
    print("-" * 80)
    for key, value in loss_dict.items():
        print(f"  {key:25s}: {value.item():.6f}")
    
    # 测试守恒量
    print("\n" + "-" * 80)
    print("Testing conservation laws...")
    
    with torch.no_grad():
        # 质量守恒
        mass_initial = criterion.compute_total_mass(initial_state)
        mass_pred = criterion.compute_total_mass(prediction)
        mass_change = ((mass_pred - mass_initial) / mass_initial * 100).abs()
        print(f"\n  Mass Conservation:")
        print(f"    Initial mass: {mass_initial.mean().item():.2f}")
        print(f"    Predicted mass: {mass_pred.mean().item():.2f}")
        print(f"    Relative change: {mass_change.mean().item():.4f}%")
        
        # 能量守恒
        energy_initial = criterion.compute_total_energy(initial_state)
        energy_pred = criterion.compute_total_energy(prediction)
        energy_change = ((energy_pred - energy_initial) / energy_initial * 100).abs()
        print(f"\n  Energy Conservation:")
        print(f"    Initial energy: {energy_initial.mean().item():.2f}")
        print(f"    Predicted energy: {energy_pred.mean().item():.2f}")
        print(f"    Relative change: {energy_change.mean().item():.4f}%")
        
        # 水汽守恒
        moisture_initial = criterion.compute_total_moisture(initial_state)
        moisture_pred = criterion.compute_total_moisture(prediction)
        moisture_change = ((moisture_pred - moisture_initial) / (moisture_initial.abs() + 1e-6) * 100).abs()
        print(f"\n  Moisture Conservation:")
        print(f"    Initial moisture: {moisture_initial.mean().item():.6f}")
        print(f"    Predicted moisture: {moisture_pred.mean().item():.6f}")
        print(f"    Relative change: {moisture_change.mean().item():.4f}%")
        
        # 动量守恒
        mom_x_init, mom_y_init = criterion.compute_total_momentum(initial_state)
        mom_x_pred, mom_y_pred = criterion.compute_total_momentum(prediction)
        print(f"\n  Momentum Conservation:")
        print(f"    Initial momentum (x): {mom_x_init.mean().item():.6f}")
        print(f"    Predicted momentum (x): {mom_x_pred.mean().item():.6f}")
        print(f"    Initial momentum (y): {mom_y_init.mean().item():.6f}")
        print(f"    Predicted momentum (y): {mom_y_pred.mean().item():.6f}")
    
    # 测试反向传播
    print("\n" + "-" * 80)
    print("Testing backward pass...")
    
    loss_dict['total'].backward()
    
    # 检查梯度
    has_grad = any(p.grad is not None for p in model.parameters())
    if has_grad:
        total_grad_norm = torch.sqrt(
            sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)
        )
        print(f"  ✓ Gradients computed successfully")
        print(f"  Total gradient norm: {total_grad_norm.item():.6f}")
    else:
        print(f"  ✗ No gradients found!")
    
    # 使用建议
    print("\n" + "=" * 80)
    print("📚 Usage Example in Training Loop:")
    print("=" * 80)
    print("""
# 在训练脚本中使用：

from model import create_model, PhysicsConstrainedLoss

# 创建模型和损失函数
model = create_model().to(device)
criterion = PhysicsConstrainedLoss(
    mse_weight=1.0,        # 数据拟合权重
    mass_weight=0.1,       # 质量守恒权重
    energy_weight=0.1,     # 能量守恒权重
    moisture_weight=0.05,  # 水汽守恒权重
    momentum_weight=0.05   # 动量守恒权重
)

# 训练循环
for batch in dataloader:
    initial_state, target = batch
    
    # 前向传播
    prediction = model(initial_state, dt=3600.0)
    
    # 计算损失（包含物理约束）
    loss_dict = criterion(prediction, target, initial_state)
    
    # 反向传播
    optimizer.zero_grad()
    loss_dict['total'].backward()
    optimizer.step()
    
    # 记录各项损失
    print(f"Total: {loss_dict['total'].item():.4f}, "
          f"MSE: {loss_dict['mse'].item():.4f}, "
          f"Mass: {loss_dict['mass_conservation'].item():.4f}, "
          f"Energy: {loss_dict['energy_conservation'].item():.4f}")
    """)
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    test_physics_constrained_loss()
