# 物理约束损失函数使用指南

## 📚 概述

`PhysicsConstrainedLoss` 是一个结合数据拟合和物理定律约束的损失函数，用于确保神经网络预测遵循基本的物理守恒律。

## 🎯 物理约束类型

### 1. **质量守恒** (Mass Conservation)
```python
# 物理定律：∂ρ/∂t + ∇·(ρV) = 0
# 简化为：大气总质量应保持不变
mass_initial = ∫∫ p_s dA
mass_predicted = ∫∫ p_s dA  
loss_mass = MSE(mass_predicted, mass_initial)
```

**意义**：
- 大气总质量不会凭空消失或产生
- 保证预报的地表气压场物理合理

### 2. **能量守恒** (Energy Conservation)
```python
# 物理定律：dE/dt = Q - W (第一热力学定律)
# E = 动能 + 内能 + 位能

E_kinetic = 0.5 * (u² + v²)
E_internal = cp * T
E_potential = g * z

loss_energy = MSE(E_predicted, E_target)
```

**意义**：
- 能量可以转化但总量守恒（忽略辐射等源汇）
- 防止模型产生非物理的能量爆炸或衰减

### 3. **水汽守恒** (Moisture Conservation)
```python
# 物理定律：∂q/∂t + ∇·(qV) = E - P
# 在没有降水和蒸发时：q应守恒

total_moisture = ∫∫∫ q dV
loss_moisture = MSE(moisture_predicted, moisture_target)
```

**意义**：
- 水汽不会凭空消失（除非有降水）
- 确保湿度场的物理合理性

### 4. **动量守恒** (Momentum Conservation)
```python
# 物理定律：全球积分动量守恒（无外力）
Px = ∫∫∫ ρu dV
Py = ∫∫∫ ρv dV

loss_momentum = MSE(Px_pred, Px_target) + MSE(Py_pred, Py_target)
```

**意义**：
- 全球风场的总动量守恒
- 防止系统性的风速偏差

## 🔧 使用方法

### 基本用法

```python
from model import create_model, PhysicsConstrainedLoss

# 创建模型
model = create_model()

# 创建物理约束损失函数
criterion = PhysicsConstrainedLoss(
    num_vars=5,           # T, u, v, q, sp
    num_levels=37,        # 垂直层数
    img_size=(128, 256),  # (lat, lon)
    mse_weight=1.0,       # 数据拟合权重
    mass_weight=0.1,      # 质量守恒权重
    energy_weight=0.1,    # 能量守恒权重
    moisture_weight=0.05, # 水汽守恒权重
    momentum_weight=0.05  # 动量守恒权重
)

# 训练循环
for initial_state, target in dataloader:
    # 模型预测
    prediction = model(initial_state, dt=3600.0)
    
    # 计算损失（包含所有物理约束）
    loss_dict = criterion(prediction, target, initial_state)
    
    # 反向传播
    optimizer.zero_grad()
    loss_dict['total'].backward()
    optimizer.step()
    
    # 记录损失
    print(f"Total: {loss_dict['total'].item():.4f}")
    print(f"MSE: {loss_dict['mse'].item():.4f}")
    print(f"Mass: {loss_dict.get('mass_conservation', 0):.4f}")
    print(f"Energy: {loss_dict.get('energy_conservation', 0):.4f}")
```

### 在配置文件中设置

```yaml
# config.yaml
training:
  # 使用物理约束损失
  use_physics_constrained_loss: true
  
  # 各项权重
  mse_weight: 1.0
  mass_conservation_weight: 0.1
  energy_conservation_weight: 0.1
  moisture_conservation_weight: 0.05
  momentum_conservation_weight: 0.05
```

### 权重调优建议

| 训练阶段 | MSE权重 | 物理约束权重 | 说明 |
|---------|---------|------------|------|
| **初期** (Epoch 1-20) | 1.0 | 0.01 | 先学习基本预测 |
| **中期** (Epoch 21-50) | 1.0 | 0.1 | 增加物理约束 |
| **后期** (Epoch 51+) | 1.0 | 0.2-0.5 | 强化物理一致性 |

```python
# 动态调整权重
def adjust_physics_weights(epoch):
    if epoch < 20:
        return 0.01
    elif epoch < 50:
        return 0.1
    else:
        return 0.2

criterion.mass_weight = adjust_physics_weights(current_epoch)
criterion.energy_weight = adjust_physics_weights(current_epoch)
```

## 📊 监控和诊断

### 1. 检查守恒律违反程度

```python
with torch.no_grad():
    # 质量守恒
    mass_initial = criterion.compute_total_mass(initial_state)
    mass_pred = criterion.compute_total_mass(prediction)
    mass_violation = abs(mass_pred - mass_initial) / mass_initial * 100
    print(f"Mass violation: {mass_violation.mean().item():.2f}%")
    
    # 能量守恒
    energy_initial = criterion.compute_total_energy(initial_state)
    energy_pred = criterion.compute_total_energy(prediction)
    energy_violation = abs(energy_pred - energy_initial) / energy_initial * 100
    print(f"Energy violation: {energy_violation.mean().item():.2f}%")
```

### 2. TensorBoard可视化

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

for epoch in range(num_epochs):
    loss_dict = train_epoch()
    
    # 记录各项损失
    writer.add_scalar('Loss/total', loss_dict['total'], epoch)
    writer.add_scalar('Loss/mse', loss_dict['mse'], epoch)
    writer.add_scalar('Loss/mass_conservation', 
                     loss_dict.get('mass_conservation', 0), epoch)
    writer.add_scalar('Loss/energy_conservation', 
                     loss_dict.get('energy_conservation', 0), epoch)
```

## ⚙️ 高级配置

### 自定义物理约束

```python
class CustomPhysicsLoss(PhysicsConstrainedLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vorticity_weight = 0.05
    
    def compute_vorticity_conservation(self, state):
        """计算位涡守恒"""
        variables = self.extract_variables(state)
        u = variables['u']
        v = variables['v']
        
        # 计算相对涡度: ζ = ∂v/∂x - ∂u/∂y
        # 简化实现...
        vorticity = self.finite_difference_vorticity(u, v)
        return vorticity.mean()
    
    def forward(self, prediction, target, initial_state=None):
        # 调用父类方法
        loss_dict = super().forward(prediction, target, initial_state)
        
        # 添加位涡守恒约束
        if initial_state is not None:
            vort_init = self.compute_vorticity_conservation(initial_state)
            vort_pred = self.compute_vorticity_conservation(prediction)
            vort_loss = F.mse_loss(vort_pred, vort_init)
            
            loss_dict['vorticity_conservation'] = vort_loss
            loss_dict['total'] = loss_dict['total'] + self.vorticity_weight * vort_loss
        
        return loss_dict
```

### 条件性物理约束

```python
def forward(self, prediction, target, initial_state=None):
    loss_dict = super().forward(prediction, target, initial_state)
    
    # 只在特定条件下应用约束
    if self.training:  # 只在训练时应用
        # 物理约束...
        pass
    
    return loss_dict
```

## 🔬 实验结果预期

### 典型的损失曲线

```
Epoch  | Total Loss | MSE    | Mass   | Energy | Moisture
-------|------------|--------|--------|--------|----------
1      | 0.1250     | 0.1200 | 0.0030 | 0.0015 | 0.0005
10     | 0.0580     | 0.0500 | 0.0050 | 0.0020 | 0.0010
50     | 0.0245     | 0.0200 | 0.0025 | 0.0015 | 0.0005
100    | 0.0185     | 0.0150 | 0.0020 | 0.0010 | 0.0005
```

### 守恒律违反度

**良好的模型应该达到**：
- 质量守恒违反 < 1%
- 能量守恒违反 < 5%
- 水汽守恒违反 < 2%
- 动量守恒违反 < 5%

## 🐛 常见问题

### Q1: 物理约束损失过大，导致训练不稳定？

**解决方案**：
1. 降低物理约束权重（从0.01开始）
2. 使用梯度裁剪
3. 先不加约束训练几个epoch

```python
# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 逐步增加约束权重
criterion.mass_weight = min(0.1, epoch * 0.001)
```

### Q2: 某些守恒律很难满足？

**分析**：
- 能量守恒：最难，因为包含多个项的复杂相互作用
- 质量守恒：相对容易，主要依赖地表气压
- 水汽守恒：中等难度，需要准确的湿度预测

**建议**：
- 针对性地调整权重
- 能量约束可以设置更小的权重
- 可以只约束相对变化而非绝对值

### Q3: 如何验证物理约束是否有效？

**方法**：
1. 对比训练中有/无物理约束的模型
2. 检查长期预报的守恒律违反度
3. 观察极端天气事件的预报质量

```python
# A/B测试
model_with_physics = train_model(use_physics_loss=True)
model_without_physics = train_model(use_physics_loss=False)

# 评估
evaluate_conservation_laws(model_with_physics, test_data)
evaluate_conservation_laws(model_without_physics, test_data)
```

## 📖 参考文献

1. **物理知识引导的机器学习**
   - Raissi et al. (2019) "Physics-informed neural networks"
   - Beucler et al. (2021) "Enforcing analytic constraints in neural networks"

2. **NWP中的守恒律**
   - Arakawa & Lamb (1977) "Conservation properties of numerical schemes"
   - Lin (2004) "A finite-volume integration method"

3. **神经网络天气预报**
   - Pathak et al. (2022) "FourCastNet"
   - Bi et al. (2023) "Pangu-Weather"

## 🎓 最佳实践总结

1. ✅ **渐进式训练**：从低权重开始，逐步增加物理约束
2. ✅ **监控所有分量**：不仅看总损失，更要看各项约束
3. ✅ **验证守恒律**：定期检查模型是否真的满足物理定律
4. ✅ **权衡取舍**：物理约束vs预测精度需要平衡
5. ✅ **特定场景调优**：不同应用可能需要不同的权重配置

---

**联系方式**：如有问题请查看 [OPTIMIZATION_ROADMAP.md](OPTIMIZATION_ROADMAP.md) 或提交 Issue。
