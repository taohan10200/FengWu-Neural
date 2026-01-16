"""
Quick Start Example
快速开始示例
"""

import torch
from neural_nwp.models.model import create_model

def main():
    print("=" * 60)
    print("Neural NWP Quick Start Example")
    print("=" * 60)
    
    # 检查GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    
    if torch.cuda.is_available():
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 创建模型
    print("\n" + "=" * 60)
    print("Creating Neural NWP Model...")
    print("=" * 60)
    
    config = {
        'img_size': (128, 256),
        'num_vars': 5,
        'num_levels': 37,
        'hidden_dim': 512,
        'parameterization_dim': 256
    }
    
    model = create_model(config).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n✓ Model created successfully!")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1e6:.2f} MB (FP32)")
    
    # 创建测试输入
    print("\n" + "=" * 60)
    print("Testing Forward Pass...")
    print("=" * 60)
    
    batch_size = 2
    lat, lon = config['img_size']
    channels = config['num_vars'] * config['num_levels']
    
    print(f"\nInput configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Spatial resolution: {lat} x {lon}")
    print(f"  Number of variables: {config['num_vars']}")
    print(f"  Vertical levels: {config['num_levels']}")
    print(f"  Total channels: {channels}")
    
    # 创建随机输入
    x = torch.randn(batch_size, lat, lon, channels).to(device)
    input_size_mb = x.element_size() * x.nelement() / 1e6
    
    print(f"\n✓ Input tensor created")
    print(f"  Shape: {tuple(x.shape)}")
    print(f"  Memory: {input_size_mb:.2f} MB")
    
    # 单步预测
    print("\n" + "=" * 60)
    print("Running Single-Step Prediction...")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        import time
        
        # 预热
        _ = model(x, dt=3600.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 计时
        start_time = time.time()
        output = model(x, dt=3600.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
    
    print(f"\n✓ Single-step prediction completed")
    print(f"  Output shape: {tuple(output.shape)}")
    print(f"  Time: {elapsed_time*1000:.2f} ms")
    print(f"  Throughput: {batch_size / elapsed_time:.2f} samples/sec")
    
    # 多步预测
    print("\n" + "=" * 60)
    print("Running Multi-Step Forecast...")
    print("=" * 60)
    
    num_steps = 10
    print(f"\nForecast configuration:")
    print(f"  Number of steps: {num_steps}")
    print(f"  Time step: 1 hour")
    print(f"  Total forecast: {num_steps} hours")
    
    with torch.no_grad():
        start_time = time.time()
        trajectory = model.rollout(x, num_steps=num_steps, dt=3600.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
    
    print(f"\n✓ Multi-step forecast completed")
    print(f"  Output shape: {tuple(trajectory.shape)}")
    print(f"  Time: {elapsed_time:.2f} s")
    print(f"  Average per step: {elapsed_time/num_steps*1000:.2f} ms")
    
    # GPU显存使用情况
    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("GPU Memory Usage")
        print("=" * 60)
        
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        
        print(f"\n  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
    
    # 总结
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\n✅ All tests passed successfully!")
    print("\nNext steps:")
    print("  1.Prepare your weather data in the required format")
    print("  2.Train the model: python train.py --data_dir ./data")
    print("  3.Run inference: python inference.py --checkpoint ./checkpoints/best.pth")
    print("\nFor more information, see README.md")
    print("=" * 60)


if __name__ == '__main__':
    main()
