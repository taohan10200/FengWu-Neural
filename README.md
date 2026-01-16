# 3D Primitive Equations Dynamic Core (PyTorch)

基于 PyTorch 的 3D 原始方程大气动力学核心，支持 GPU 加速和 ERA5 数据集成。

## 📋 特性

- ✅ **完整的物理动力学**：3D 原始方程组（动量、热力学、连续方程）
- ✅ **GPU 加速**：基于 PyTorch，自动微分支持
- ✅ **ERA5 集成**：37层气压数据 → σ坐标自动转换
- ✅ **混合建模**：物理模型 + AI 修正
- ✅ **多种时间积分**：Euler、RK4、Leap-Frog
- ✅ **可视化工具**：水平切片、垂直剖面、动画生成
- ✅ **端到端训练**：支持数据同化和参数优化

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/your_username/atmospheric_model_pytorch.git
cd atmospheric_model_pytorch

# 创建环境
conda create -n atmos python=3.9
conda activate atmos

# 安装依赖
pip install -r requirements. txt

# 安装本项目
pip install -e . 