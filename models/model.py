"""
Neural Numerical Weather Prediction Model with Learnable Parameterizations
基于神经网络参数化的数值气象预测模型

This model combines traditional NWP dynamics with learnable neural network 
parameterizations for physical processes like radiation, convection, and boundary layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class PhysicsParameterization(nn.Module):
    """神经网络物理参数化基类"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RadiationParameterization(PhysicsParameterization):
    """辐射过程参数化 - 计算短波和长波辐射的加热率"""
    
    def __init__(self, num_levels: int = 37, hidden_dim: int = 256):
        # 输入: 温度、比湿、云量、太阳天顶角等
        input_dim = num_levels * 3 + 2  # T, q, cloud per level + zenith angle + surface albedo
        output_dim = num_levels  # 每层的辐射加热率
        super().__init__(input_dim, hidden_dim, output_dim, num_layers=4)


class ConvectionParameterization(PhysicsParameterization):
    """对流过程参数化 - 计算对流引起的温度和湿度变化"""
    
    def __init__(self, num_levels: int = 37, hidden_dim: int = 256):
        # 输入: 温度、比湿、垂直速度
        input_dim = num_levels * 3
        # 输出: 温度倾向、湿度倾向
        output_dim = num_levels * 2
        super().__init__(input_dim, hidden_dim, output_dim, num_layers=4)


class BoundaryLayerParameterization(PhysicsParameterization):
    """边界层过程参数化 - 计算湍流混合效应"""
    
    def __init__(self, num_levels: int = 37, hidden_dim: int = 256):
        # 输入: 温度、比湿、风速、表面通量
        input_dim = num_levels * 4 + 3  # T, q, u, v per level + surface fluxes
        # 输出: 温度倾向、湿度倾向、动量倾向
        output_dim = num_levels * 4
        super().__init__(input_dim, hidden_dim, output_dim, num_layers=4)


class DynamicCore(nn.Module):
    """动力核心 - 使用深度学习模拟大气动力学方程"""
    
    def __init__(self, 
                 num_vars: int = 5,  # T, u, v, q, sp
                 num_levels: int = 37,
                 num_single_vars: int = 7,
                 hidden_dim: int = 512,
                 num_heads: int = 8):
        super().__init__()
        
        self.num_vars = num_vars
        self.num_levels = num_levels
        self.num_single_vars = num_single_vars
        
        input_dim = num_vars * num_levels + num_single_vars
        
        # 使用Transformer编码器处理空间-垂直结构
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, lat, lon, num_vars * num_levels]
        Returns:
            tendency: [batch, lat, lon, num_vars * num_levels]
        """
        batch, lat, lon, channels = x.shape
        
        # Reshape for transformer: [batch * lat * lon, 1, channels]
        x_flat = x.reshape(batch, lat * lon, channels)
        # Process through transformer
        x_embed = self.input_proj(x_flat)
        x_transformed = self.transformer(x_embed)
        tendency = self.output_proj(x_transformed)
        
        # Reshape back
        tendency = tendency.reshape(batch, lat, lon, channels)
        
        return tendency


class NeuralNWP(nn.Module):
    """
    神经网络数值气象预测模型
    结合动力核心和神经网络参数化方案
    
    支持气压层变量和单层变量的混合输入：
    - 气压层变量(pressure vars): [num_pressure_vars × num_levels] channels
    - 单层变量(single vars): [num_single_vars] channels
    - 总输入: [batch, num_pressure_vars × num_levels + num_single_vars, H, W]
    """
    
    def __init__(self,
                 img_size: Tuple[int, int] = (128, 256),  # (lat, lon)
                 pressure_vars: list = None,  # ['z', 'q', 'u', 'v', 't', 'w']
                 single_vars: list = None,  # ['v10', 'u10', 't2m', 'sp', ...]
                 pressure_levels: list = None,  # [1000, 925, 850, ...]
                 hidden_dim: int = 512,
                 parameterization_dim: int = 256,
                 num_heads: int = 8):
        super().__init__()
        
        self.img_size = img_size
        self.pressure_vars = pressure_vars or ['t', 'u', 'v', 'q']
        self.single_vars = single_vars or ['v10', 'u10', 'v100', 'u100', 't2m', 'sp', 'msl']
        self.pressure_levels = pressure_levels or [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
        
        self.num_pressure_vars = len(self.pressure_vars)
        self.num_single_vars = len(self.single_vars)
        self.num_levels = len(self.pressure_levels)
        
        # 计算总通道数
        self.total_channels = self.num_pressure_vars * self.num_levels + self.num_single_vars
        
        # 为了兼容旧接口
        self.num_vars = self.num_pressure_vars
        
        # 动力核心 - 使用总通道数
        self.dynamic_core = DynamicCore(
            num_vars=self.num_pressure_vars,
            num_levels=self.num_levels,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )
        # 注意：DynamicCore的输入会是total_channels，需要在forward中处理
        
        # 物理过程参数化
        self.radiation = RadiationParameterization(
            num_levels=self.num_levels,
            hidden_dim=parameterization_dim
        )
        
        self.convection = ConvectionParameterization(
            num_levels=self.num_levels,
            hidden_dim=parameterization_dim
        )
        
        self.boundary_layer = BoundaryLayerParameterization(
            num_levels=self.num_levels,
            hidden_dim=parameterization_dim
        )
        
    def extract_variables(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """从状态向量中提取各个气象变量
        
        输入格式: [batch, lat, lon, channels]
        其中 channels = num_pressure_vars * num_levels + num_single_vars
        
        前 num_pressure_vars * num_levels 个channels是气压层变量
        后 num_single_vars 个channels是单层变量
        """
        batch, lat, lon, channels = state.shape
        
        # 分离气压层变量和单层变量
        pressure_channels = self.num_pressure_vars * self.num_levels
        pressure_data = state[..., :pressure_channels]  # [batch, lat, lon, num_pressure_vars * num_levels]
        single_data = state[..., pressure_channels:]     # [batch, lat, lon, num_single_vars]
        
        # Reshape气压层数据: [batch, lat, lon, num_pressure_vars, num_levels]
        pressure_reshaped = pressure_data.reshape(batch, lat, lon, self.num_pressure_vars, self.num_levels)
        
        # 构建变量字典
        variables = {}
        
        # 气压层变量
        for i, var_name in enumerate(self.pressure_vars):
            variables[var_name] = pressure_reshaped[..., i, :]  # [batch, lat, lon, num_levels]
        
        # 单层变量
        for i, var_name in enumerate(self.single_vars):
            variables[var_name] = single_data[..., i:i+1]  # [batch, lat, lon, 1]
        
        return variables
    
    def apply_radiation(self, variables: Dict[str, torch.Tensor]) -> torch.Tensor:
        """应用辐射参数化"""
        # 使用't'和'q'变量（如果存在）
        if 't' not in variables or 'q' not in variables:
            # 如果没有必需的变量，返回零
            batch, lat, lon, num_levels = list(variables.values())[0].shape
            return torch.zeros(batch, lat, lon, num_levels, device=list(variables.values())[0].device)
        
        batch, lat, lon, num_levels = variables['t'].shape
        
        # 准备辐射输入
        # 简化版本：只使用温度和湿度
        rad_input = torch.cat([
            variables['t'].reshape(batch * lat * lon, num_levels),
            variables['q'].reshape(batch * lat * lon, num_levels),
            torch.zeros(batch * lat * lon, num_levels, device=variables['t'].device),  # 云量占位
            torch.zeros(batch * lat * lon, 2, device=variables['t'].device)  # 天顶角和反照率占位
        ], dim=-1)
        
        # 计算辐射加热率
        heating = self.radiation(rad_input)
        heating = heating.reshape(batch, lat, lon, num_levels)
        
        return heating
    
    def apply_convection(self, variables: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用对流参数化"""
        if 't' not in variables or 'q' not in variables:
            # 如果没有必需的变量，返回零
            batch, lat, lon, num_levels = list(variables.values())[0].shape
            zeros = torch.zeros(batch, lat, lon, num_levels, device=list(variables.values())[0].device)
            return zeros, zeros
        
        batch, lat, lon, num_levels = variables['t'].shape
        
        # 准备对流输入
        conv_input = torch.cat([
            variables['t'].reshape(batch * lat * lon, num_levels),
            variables['q'].reshape(batch * lat * lon, num_levels),
            torch.zeros(batch * lat * lon, num_levels, device=variables['t'].device)  # 垂直速度占位
        ], dim=-1)
        
        # 计算对流倾向
        tendency = self.convection(conv_input)
        tendency = tendency.reshape(batch, lat, lon, num_levels * 2)
        
        temp_tendency = tendency[..., :num_levels]
        humid_tendency = tendency[..., num_levels:]
        
        return temp_tendency, humid_tendency
    
    def apply_boundary_layer(self, variables: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """应用边界层参数化"""
        if 't' not in variables or 'q' not in variables or 'u' not in variables or 'v' not in variables:
            # 如果没有必需的变量，返回零
            batch, lat, lon, num_levels = list(variables.values())[0].shape
            zeros = torch.zeros(batch, lat, lon, num_levels, device=list(variables.values())[0].device)
            return zeros, zeros, zeros, zeros
        
        batch, lat, lon, num_levels = variables['t'].shape
        
        # 准备边界层输入
        bl_input = torch.cat([
            variables['t'].reshape(batch * lat * lon, num_levels),
            variables['q'].reshape(batch * lat * lon, num_levels),
            variables['u'].reshape(batch * lat * lon, num_levels),
            variables['v'].reshape(batch * lat * lon, num_levels),
            torch.zeros(batch * lat * lon, 3, device=variables['t'].device)  # 表面通量占位
        ], dim=-1)
        
        # 计算边界层倾向
        tendency = self.boundary_layer(bl_input)
        tendency = tendency.reshape(batch, lat, lon, num_levels * 4)
        
        temp_tendency = tendency[..., :num_levels]
        humid_tendency = tendency[..., num_levels:num_levels*2]
        u_tendency = tendency[..., num_levels*2:num_levels*3]
        v_tendency = tendency[..., num_levels*3:]
        
        return temp_tendency, humid_tendency, u_tendency, v_tendency
    
    def forward(self, 
                state: torch.Tensor, 
                dt: float = 3600.0) -> torch.Tensor:
        """
        前向传播 - 预测下一时刻的大气状态
        
        Args:
            state: [batch, lat, lon, total_channels] 当前大气状态
                   total_channels = num_pressure_vars * num_levels + num_single_vars
            dt: 时间步长（秒）
            
        Returns:
            next_state: [batch, lat, lon, total_channels] 下一时刻的大气状态
        """
        batch, lat, lon, channels = state.shape
        
        # 1. 动力核心计算
        dynamic_tendency = self.dynamic_core(state)
        
        # 2. 提取变量用于物理参数化
        variables = self.extract_variables(state)
        
        # 3. 应用物理参数化（简化版本，只处理气压层变量）
        # 注意：这里需要根据实际的pressure_vars和single_vars来调整
        # 为了简化，这里假设有't', 'u', 'v', 'q'等变量
        physics_tendency = torch.zeros_like(state)
        
        # 如果有对应的变量，则应用物理参数化
        if 't' in variables and 'q' in variables:
            rad_heating = self.apply_radiation(variables)
            conv_temp_tend, conv_humid_tend = self.apply_convection(variables)
            bl_temp_tend, bl_humid_tend, bl_u_tend, bl_v_tend = self.apply_boundary_layer(variables)
            
            # 重新组织倾向项（只处理气压层部分）
            pressure_channels = self.num_pressure_vars * self.num_levels
            
            # 创建一个新的张量来存储物理倾向，避免in-place操作问题
            physics_pressure_reshaped = torch.zeros(
                batch, lat, lon, self.num_pressure_vars, self.num_levels,
                device=state.device, dtype=state.dtype
            )
            
            # 根据变量名找到对应的索引
            if 't' in self.pressure_vars:
                t_idx = self.pressure_vars.index('t')
                physics_pressure_reshaped[..., t_idx, :] = rad_heating + conv_temp_tend + bl_temp_tend
            
            if 'u' in self.pressure_vars:
                u_idx = self.pressure_vars.index('u')
                physics_pressure_reshaped[..., u_idx, :] = bl_u_tend
            
            if 'v' in self.pressure_vars:
                v_idx = self.pressure_vars.index('v')
                physics_pressure_reshaped[..., v_idx, :] = bl_v_tend
            
            if 'q' in self.pressure_vars:
                q_idx = self.pressure_vars.index('q')
                physics_pressure_reshaped[..., q_idx, :] = conv_humid_tend + bl_humid_tend
            
            # 使用 clone() 避免 in-place 操作错误
            physics_tendency_pressure = physics_pressure_reshaped.reshape(
                batch, lat, lon, pressure_channels
            )
            physics_tendency[..., :pressure_channels] = physics_tendency_pressure
        
        # 5. 时间积分（Euler前向格式）
        total_tendency = dynamic_tendency + physics_tendency
        next_state = state + total_tendency * dt
        
        return next_state
    
    def rollout(self, 
                initial_state: torch.Tensor,
                num_steps: int,
                dt: float = 3600.0) -> torch.Tensor:
        """
        多步预测
        
        Args:
            initial_state: [batch, lat, lon, num_vars * num_levels]
            num_steps: 预测步数
            dt: 时间步长（秒）
            
        Returns:
            trajectory: [batch, num_steps, lat, lon, num_vars * num_levels]
        """
        batch, lat, lon, channels = initial_state.shape
        trajectory = torch.zeros(batch, num_steps, lat, lon, channels, 
                                device=initial_state.device, dtype=initial_state.dtype)
        
        state = initial_state
        for t in range(num_steps):
            state = self.forward(state, dt)
            trajectory[:, t] = state
        
        return trajectory


def create_model(config: Optional[Dict] = None) -> NeuralNWP:
    """
    创建模型的工厂函数
    
    Args:
        config: 配置字典，包含模型超参数
        支持的字段：
        - img_size: tuple, 图像尺寸 (lat, lon)
        - pressure_vars: list, 气压层变量名称
        - single_vars: list, 单层变量名称
        - pressure_levels: list, 气压层列表
        - hidden_dim: int, 隐藏层维度
        - parameterization_dim: int, 参数化网络维度
        
    Returns:
        model: NeuralNWP模型实例
    """
    if config is None:
        config = {}
    
    # 提取参数，使用默认值
    model_config = {
        'img_size': config.get('img_size', (128, 256)),
        'pressure_vars': config.get('pressure_vars', ['t', 'u', 'v', 'q']),
        'single_vars': config.get('single_vars', ['sp', 't2m', 'u10', 'v10']),
        'pressure_levels': config.get('pressure_levels', 
            [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]),
        'hidden_dim': config.get('hidden_dim', 512),
        'parameterization_dim': config.get('parameterization_dim', 256),
        'num_heads': config.get('num_heads', 8)
    }
    
    model = NeuralNWP(**model_config)
    
    return model


class PhysicsConstrainedLoss(nn.Module):
    """
    物理约束损失函数
    结合数据拟合和物理定律约束（质量守恒、能量守恒等）
    """
    
    def __init__(self,
                 pressure_vars: list = None,
                 single_vars: list = None,
                 pressure_levels: list = None,
                 img_size: Tuple[int, int] = (128, 256),
                 mean: Optional[torch.Tensor] = None,
                 std: Optional[torch.Tensor] = None,
                 mse_weight: float = 1.0,
                 mass_weight: float = 0.1,
                 energy_weight: float = 0.1,
                 moisture_weight: float = 0.05,
                 momentum_weight: float = 0.05):
        """
        Args:
            pressure_vars: 气压层变量列表
            single_vars: 单层变量列表
            pressure_levels: 气压层高度列表
            img_size: 空间分辨率 (lat, lon)
            mean: 归一化均值 [channels]
            std: 归一化标准差 [channels]
            mse_weight: MSE损失权重
            mass_weight: 质量守恒约束权重
            energy_weight: 能量守恒约束权重
            moisture_weight: 水汽守恒约束权重
            momentum_weight: 动量守恒约束权重
        """
        super().__init__()
        
        self.pressure_vars = pressure_vars or ['t', 'u', 'v', 'q']
        self.single_vars = single_vars or ['v10', 'u10', 'v100', 'u100', 't2m', 'sp', 'msl']
        self.pressure_levels = pressure_levels or [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
        
        self.num_pressure_vars = len(self.pressure_vars)
        self.num_single_vars = len(self.single_vars)
        self.num_levels = len(self.pressure_levels)
        self.img_size = img_size
        
        # 注册归一化参数
        if mean is not None:
            self.register_buffer('mean', mean)
        else:
            self.mean = None
            
        if std is not None:
            self.register_buffer('std', std)
        else:
            self.std = None
        
        self.mse_weight = mse_weight
        self.mass_weight = mass_weight
        self.energy_weight = energy_weight
        self.moisture_weight = moisture_weight
        self.momentum_weight = momentum_weight
        
        # 物理常数
        self.g = 9.81           # 重力加速度 [m/s^2]
        self.cp = 1004.0        # 定压比热 [J/(kg·K)]
        self.Lv = 2.5e6         # 凝结潜热 [J/kg]
        self.R_dry = 287.0      # 干空气气体常数 [J/(kg·K)]
        
    def extract_variables(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """从状态向量中提取各个气象变量"""
        # 反归一化
        if self.mean is not None and self.std is not None:
            # state: [batch, lat, lon, channels]
            # mean/std: [channels] -> [1, 1, 1, channels]
            # 确保维度匹配
            if self.mean.dim() == 1:
                mean = self.mean.view(1, 1, 1, -1)
                std = self.std.view(1, 1, 1, -1)
            else:
                mean = self.mean
                std = self.std
                
            state = state * std + mean
            
        batch, lat, lon, channels = state.shape
        
        # 分离气压层变量和单层变量
        pressure_channels = self.num_pressure_vars * self.num_levels
        pressure_data = state[..., :pressure_channels]
        single_data = state[..., pressure_channels:]
        
        # Reshape气压层数据: [batch, lat, lon, num_pressure_vars, num_levels]
        pressure_reshaped = pressure_data.reshape(batch, lat, lon, self.num_pressure_vars, self.num_levels)
        
        variables = {}
        
        # 气压层变量
        for i, var_name in enumerate(self.pressure_vars):
            variables[var_name] = pressure_reshaped[..., i, :]
            
        # 单层变量
        for i, var_name in enumerate(self.single_vars):
            variables[var_name] = single_data[..., i]  # [batch, lat, lon]
            
        return variables
    
    def compute_total_mass(self, variables: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        计算大气总质量
        质量 ≈ ∫∫ p_s dA (简化，假设均匀重力场)
        """
        if 'sp' not in variables:
            return None
            
        sp = variables['sp']  # [batch, lat, lon]
        
        # 地表气压积分 → 大气柱总质量
        # 真实情况需要考虑纬度权重: cos(lat) * dlon * dlat
        total_mass = sp.mean(dim=(1, 2))  # 简化：取平均
        
        return total_mass
    
    def compute_total_energy(self, variables: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        计算大气总能量
        E_total = E_kinetic + E_potential + E_internal
        返回单位: kJ/kg (为了数值稳定性，将J/kg除以1000)
        """
        # 检查必要变量
        if not all(v in variables for v in ['t', 'u', 'v']):
            return None
            
        T = variables['t']   # [batch, lat, lon, num_levels]
        u = variables['u']
        v = variables['v']
        
        # 1. 动能: KE = 0.5 * m * (u^2 + v^2)
        kinetic_energy = 0.5 * (u**2 + v**2)  # [batch, lat, lon, num_levels]
        kinetic_energy = kinetic_energy.mean(dim=(1, 2, 3))  # [batch]
        
        # 2. 内能: IE = m * cp * T
        internal_energy = self.cp * T  # [batch, lat, lon, num_levels]
        internal_energy = internal_energy.mean(dim=(1, 2, 3))  # [batch]
        
        # 3. 位能: PE = m * g * z
        # 简化：假设等间距垂直层，位能正比于层数
        height_levels = torch.arange(self.num_levels, 
                                     device=T.device, 
                                     dtype=T.dtype)
        height_levels = height_levels[None, None, None, :]  # [1, 1, 1, num_levels]
        
        potential_energy = self.g * T * height_levels  # 简化的位能估算
        potential_energy = potential_energy.mean(dim=(1, 2, 3))  # [batch]
        
        # 总能量
        total_energy = kinetic_energy + internal_energy + potential_energy
        
        # 缩放: J/kg -> kJ/kg，避免数值过大导致Loss不稳定
        return total_energy / 1000.0
    
    def compute_total_moisture(self, variables: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        计算大气总水汽量
        Total_moisture = ∫∫∫ q * ρ dV
        """
        if 'q' not in variables:
            return None
            
        q = variables['q']  # [batch, lat, lon, num_levels]
        
        # 积分水汽含量
        total_moisture = q.mean(dim=(1, 2, 3))  # [batch]
        
        return total_moisture
    
    def compute_total_momentum(self, variables: Dict[str, torch.Tensor]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        计算总动量
        P_x = ∫∫∫ ρ * u dV
        P_y = ∫∫∫ ρ * v dV
        """
        if 'u' not in variables or 'v' not in variables:
            return None, None
            
        u = variables['u']  # [batch, lat, lon, num_levels]
        v = variables['v']
        
        momentum_x = u.mean(dim=(1, 2, 3))  # [batch]
        momentum_y = v.mean(dim=(1, 2, 3))  # [batch]
        
        return momentum_x, momentum_y
    
    def forward(self, 
                prediction: torch.Tensor, 
                target: torch.Tensor,
                initial_state: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算物理约束损失
        
        Args:
            prediction: 模型预测 [batch, lat, lon, channels]
            target: 真值 [batch, lat, lon, channels]
            initial_state: 初始状态（用于守恒律检查）
            
        Returns:
            loss_dict: 包含各项损失的字典
        """
        # 1. 数据拟合损失 (MSE)
        mse_loss = F.mse_loss(prediction, target)
        
        # 初始化损失字典
        loss_dict = {
            'mse': mse_loss,
            'total': self.mse_weight * mse_loss
        }
        
        # 如果有初始状态，计算守恒律约束
        if initial_state is not None:
            # 预先提取变量，避免重复计算
            pred_vars = self.extract_variables(prediction)
            target_vars = self.extract_variables(target)
            init_vars = self.extract_variables(initial_state)
            
            # 2. 质量守恒约束
            if self.mass_weight > 0:
                mass_initial = self.compute_total_mass(init_vars)
                mass_pred = self.compute_total_mass(pred_vars)
                
                if mass_initial is not None and mass_pred is not None:
                    # 预测值应该保持与初始状态相同的质量
                    mass_loss = F.mse_loss(mass_pred, mass_initial)
                    loss_dict['mass_conservation'] = mass_loss
                    loss_dict['total'] = loss_dict['total'] + self.mass_weight * mass_loss
            
            # 3. 能量守恒约束
            if self.energy_weight > 0:
                energy_initial = self.compute_total_energy(init_vars)
                energy_pred = self.compute_total_energy(pred_vars)
                energy_target = self.compute_total_energy(target_vars)
                
                if energy_pred is not None and energy_target is not None:
                    # 能量可能因物理过程改变，但预测应该接近目标的能量
                    energy_loss = F.mse_loss(energy_pred, energy_target)
                    loss_dict['energy_conservation'] = energy_loss
                    loss_dict['total'] = loss_dict['total'] + self.energy_weight * energy_loss
            
            # 4. 水汽守恒约束（在没有降水的情况下）
            if self.moisture_weight > 0:
                moisture_initial = self.compute_total_moisture(init_vars)
                moisture_pred = self.compute_total_moisture(pred_vars)
                moisture_target = self.compute_total_moisture(target_vars)
                
                if moisture_pred is not None and moisture_target is not None:
                    # 水汽应该守恒（忽略降水损失）
                    moisture_loss = F.mse_loss(moisture_pred, moisture_target)
                    loss_dict['moisture_conservation'] = moisture_loss
                    loss_dict['total'] = loss_dict['total'] + self.moisture_weight * moisture_loss
            
            # 5. 动量守恒约束（全球积分动量应守恒）
            if self.momentum_weight > 0:
                mom_x_pred, mom_y_pred = self.compute_total_momentum(pred_vars)
                mom_x_target, mom_y_target = self.compute_total_momentum(target_vars)
                
                if mom_x_pred is not None and mom_x_target is not None:
                    # 动量守恒
                    momentum_loss = F.mse_loss(mom_x_pred, mom_x_target) + \
                                   F.mse_loss(mom_y_pred, mom_y_target)
                    loss_dict['momentum_conservation'] = momentum_loss
                    loss_dict['total'] = loss_dict['total'] + self.momentum_weight * momentum_loss
        
        return loss_dict


if __name__ == '__main__':
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model = create_model().to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 测试前向传播
    batch_size = 2
    lat, lon = 16, 32
    
    # 获取模型期望的通道数
    total_channels = model.total_channels
    print(f"Model total channels: {total_channels}")
    
    # 创建随机输入
    x = torch.randn(batch_size, lat, lon, total_channels).to(device)
    
    print(f"\nInput shape: {x.shape}")
    
    # 单步预测
    with torch.no_grad():
        output = model(x, dt=1.0)
    print(f"Output shape: {output.shape}")
    
    # 多步预测
    with torch.no_grad():
        trajectory = model.rollout(x, num_steps=4, dt=1.0)
    print(f"Trajectory shape: {trajectory.shape}")
    
    # 保存测试数据
    save_path = 'test_results.pt'
    torch.save({
        'input': x.cpu(),
        'trajectory': trajectory.cpu(),
        'config': {
            'total_channels': total_channels,
            'lat': lat,
            'lon': lon
        }
    }, save_path)
    print(f"Test results saved to {save_path}")
    
    # 测试损失函数
    print("\nTesting PhysicsConstrainedLoss...")
    loss_fn = PhysicsConstrainedLoss(
        pressure_vars=model.pressure_vars,
        single_vars=model.single_vars,
        pressure_levels=model.pressure_levels,
        img_size=(lat, lon)
    )
    
    # 模拟预测和目标
    pred = output
    target = torch.randn_like(pred)
    # import pdb
    # pdb.set_trace()
    loss_dict = loss_fn(pred, target, initial_state=x)
    print(loss_dict)
    print("Loss dict keys:", loss_dict.keys())
    print("Total loss:", loss_dict['total'].item())
    
    print("\n✓ Model test passed!")
