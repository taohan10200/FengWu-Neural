# Neural NWP 优化完善路线图

## 📋 项目现状评估

### ✅ 已实现的物理过程

当前项目实现了基础的NWP框架：

1. **动力核心 (Dynamic Core)** - 基于Transformer的大气动力方程求解
2. **辐射参数化 (Radiation)** - 短波/长波辐射加热率计算
3. **对流参数化 (Convection)** - 积云对流的温度和湿度倾向
4. **边界层参数化 (Boundary Layer/PBL)** - 湍流混合效应

### ❌ 关键缺失

目前是一个**概念验证原型 (Proof of Concept)**，距离完整的NWP系统还有显著差距。

---

## 🎯 优化路线图

### Phase 1: 核心物理过程补充 (1-3个月) 🔴 **关键优先级**

#### 1.1 云微物理参数化 (Cloud Microphysics) 
**重要性：★★★★★ | 预计工作量：2-3周**

**缺失影响：**
- 无法预测降水
- 温度场预测不准确（缺少潜热反馈）
- 辐射计算不完整（需要云的分布）

**实现方案：**

```python
class MicrophysicsParameterization(PhysicsParameterization):
    """
    云微物理参数化 - 水汽相变和降水形成
    
    模拟过程：
    1. 云水凝结/蒸发
    2. 云冰形成/升华
    3. 雨水自动转化和碰并增长
    4. 冰晶聚合和淞附
    5. 融化和冻结
    6. 降水率计算
    7. 潜热释放/吸收
    """
    
    def __init__(self, num_levels: int = 37, hidden_dim: int = 512):
        # 输入变量：
        # - 温度 (T): num_levels
        # - 比湿 (q): num_levels
        # - 云水 (qc): num_levels
        # - 云冰 (qi): num_levels
        # - 雨水 (qr): num_levels
        # - 雪 (qs): num_levels
        # - 霰/冰雹 (qg): num_levels
        # - 气压: num_levels
        input_dim = num_levels * 8
        
        # 输出变量：
        # - 各水凝物倾向: num_levels * 6 (qc, qi, qr, qs, qg的变化率)
        # - 温度倾向（潜热）: num_levels
        # - 地表降水率: 1
        output_dim = num_levels * 7 + 1
        
        super().__init__(input_dim, hidden_dim, output_dim, num_layers=6)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict containing:
            - qc_tend: 云水倾向
            - qi_tend: 云冰倾向
            - qr_tend: 雨水倾向
            - qs_tend: 雪倾向
            - qg_tend: 霰倾向
            - temp_tend: 温度倾向（潜热）
            - precip_rate: 地表降水率 [mm/hr]
        """
        output = self.net(x)
        # 解析输出...
        return parsed_output
```

**新增变量：**
需要在模型状态向量中增加水凝物变量：
- `num_vars`: 5 → 10 (T, u, v, q, sp, qc, qi, qr, qs, qg)

**训练数据需求：**
- ERA5变量：云水含量、云冰含量
- 降水观测数据用于监督学习
- IMERG或GPM降水产品

---

#### 1.2 陆面过程模型 (Land Surface Model)
**重要性：★★★★★ | 预计工作量：3-4周**

**缺失影响：**
- 下边界条件不完整
- 地表-大气相互作用缺失
- 温度和湿度预测误差大
- 日变化模拟不准确

**实现方案：**

```python
class LandSurfaceModel(nn.Module):
    """
    陆面过程参数化
    
    模拟过程：
    1. 土壤热传导和温度演变
    2. 土壤水分扩散和根系吸水
    3. 植被蒸腾和截留
    4. 地表能量平衡
    5. 感热和潜热通量
    6. 雪盖累积和融化
    7. 径流产生
    """
    
    def __init__(self, 
                 num_soil_levels: int = 4,
                 hidden_dim: int = 512):
        super().__init__()
        
        self.num_soil_levels = num_soil_levels
        
        # 输入：
        # - 地表温度 (Ts): 1
        # - 土壤温度 (T_soil): num_soil_levels
        # - 土壤湿度 (SM): num_soil_levels
        # - 下行短波辐射: 1
        # - 下行长波辐射: 1
        # - 降水率: 1
        # - 近地面风速: 1
        # - 近地面气温: 1
        # - 近地面比湿: 1
        # - 地表气压: 1
        # - 植被类型: 1 (encoded)
        # - 土壤类型: 1 (encoded)
        # - 雪水当量 (SWE): 1
        input_dim = 3 + num_soil_levels * 2 + 10
        
        # 输出：
        # - 感热通量 (SH): 1
        # - 潜热通量 (LH): 1
        # - 地表温度倾向: 1
        # - 土壤温度倾向: num_soil_levels
        # - 土壤湿度倾向: num_soil_levels
        # - 雪水当量倾向: 1
        # - 地表径流: 1
        output_dim = 5 + num_soil_levels * 2
        
        # 网络结构
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.energy_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2)  # SH, LH
        )
        
        self.soil_temp_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_soil_levels + 1)  # Ts + T_soil
        )
        
        self.soil_moisture_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_soil_levels)
        )
        
        self.snow_runoff_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2)  # SWE, runoff
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict containing:
            - sensible_heat: 感热通量 [W/m^2]
            - latent_heat: 潜热通量 [W/m^2]
            - surface_temp_tend: 地表温度倾向 [K/s]
            - soil_temp_tend: 土壤温度倾向 [K/s]
            - soil_moisture_tend: 土壤湿度倾向 [m^3/m^3/s]
            - swe_tend: 雪水当量倾向 [kg/m^2/s]
            - runoff: 径流 [kg/m^2/s]
        """
        h = self.encoder(x)
        
        fluxes = self.energy_branch(h)
        temp_tends = self.soil_temp_branch(h)
        moisture_tends = self.soil_moisture_branch(h)
        snow_runoff = self.snow_runoff_branch(h)
        
        return {
            'sensible_heat': fluxes[..., 0],
            'latent_heat': fluxes[..., 1],
            'surface_temp_tend': temp_tends[..., 0],
            'soil_temp_tend': temp_tends[..., 1:],
            'soil_moisture_tend': moisture_tends,
            'swe_tend': snow_runoff[..., 0],
            'runoff': snow_runoff[..., 1]
        }
```

**新增状态变量：**
需要增加陆面状态变量（二维场）：
- 地表温度 (Ts)
- 土壤温度 (T_soil, 4层)
- 土壤湿度 (SM, 4层)
- 雪水当量 (SWE)

**训练数据需求：**
- ERA5-Land数据集
- 土壤温湿度观测
- 地表通量观测（FLUXNET）

---

#### 1.3 大尺度云和层云降水
**重要性：★★★★☆ | 预计工作量：1-2周**

**实现方案：**

```python
class LargeScaleCloudParameterization(PhysicsParameterization):
    """
    大尺度云和层状降水参数化
    
    模拟过程：
    1. 格点尺度凝结/蒸发
    2. 层状云形成条件判断
    3. 云量诊断
    4. 层状降水形成
    5. 云-辐射相互作用输入
    """
    
    def __init__(self, num_levels: int = 37, hidden_dim: int = 256):
        # 输入：T, q, w(垂直速度), RH(相对湿度)
        input_dim = num_levels * 4
        
        # 输出：云水倾向、云量、降水率
        output_dim = num_levels * 2 + 1
        
        super().__init__(input_dim, hidden_dim, output_dim, num_layers=4)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns:
            - cloud_water_tend: 云水倾向
            - cloud_fraction: 云量 (0-1)
            - precip_rate: 降水率
        """
        pass
```

---

### Phase 2: 技术架构优化 (1-2个月) 🟡 **重要优先级**

#### 2.1 改进时间积分方案
**当前问题：**
- 使用Euler前向格式，数值不稳定
- 时间步长受限（CFL条件）

**优化方案：**

```python
class TimeIntegrator(nn.Module):
    """时间积分器 - 支持多种数值格式"""
    
    def __init__(self, method: str = 'rk4'):
        super().__init__()
        self.method = method
    
    def forward(self, 
                state: torch.Tensor,
                tendency_func: callable,
                dt: float) -> torch.Tensor:
        """
        Args:
            state: 当前状态
            tendency_func: 计算倾向的函数 f(state) -> tendency
            dt: 时间步长
        """
        if self.method == 'euler':
            # 前向Euler
            tendency = tendency_func(state)
            return state + dt * tendency
            
        elif self.method == 'rk4':
            # 四阶Runge-Kutta
            k1 = tendency_func(state)
            k2 = tendency_func(state + 0.5 * dt * k1)
            k3 = tendency_func(state + 0.5 * dt * k2)
            k4 = tendency_func(state + dt * k3)
            return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
        elif self.method == 'leapfrog':
            # Leapfrog with Robert-Asselin filter
            # 需要维护两个时间层
            pass
            
        elif self.method == 'semi_implicit':
            # Semi-implicit方案（适合重力波）
            # 分离快波和慢波
            pass
```

**建议：**
- 短期：实现RK3或RK4
- 长期：实现semi-implicit方案处理声波和重力波

---

#### 2.2 增加必需的预报变量

**当前变量 (5个)：**
```python
num_vars = 5  # T, u, v, q, sp
```

**扩展后变量 (至少14个)：**
```python
# 大气变量 (11个 * 37层)
- T:  温度
- u:  纬向风
- v:  经向风
- w:  垂直速度 (新增)
- q:  水汽比湿
- qc: 云水 (新增)
- qi: 云冰 (新增)
- qr: 雨水 (新增)
- qs: 雪 (新增)
- qg: 霰/雹 (新增)
- tke: 湍流动能 (新增，可选)

# 地表变量 (1层)
- sp: 地表气压

# 陆面变量 (二维场，空间分布)
- ts: 地表温度 (新增)
- swe: 雪水当量 (新增)

# 土壤变量 (4层)
- t_soil: 土壤温度 (新增)
- sm: 土壤湿度 (新增)
```

**数据结构重构：**
```python
class StateVector:
    """大气状态向量的数据结构"""
    
    def __init__(self, batch, lat, lon, num_levels):
        # 三维大气场 [batch, lat, lon, var, levels]
        self.atmos_3d = {
            'T': torch.zeros(batch, lat, lon, num_levels),
            'u': torch.zeros(batch, lat, lon, num_levels),
            'v': torch.zeros(batch, lat, lon, num_levels),
            'w': torch.zeros(batch, lat, lon, num_levels),
            'q': torch.zeros(batch, lat, lon, num_levels),
            'qc': torch.zeros(batch, lat, lon, num_levels),
            'qi': torch.zeros(batch, lat, lon, num_levels),
            'qr': torch.zeros(batch, lat, lon, num_levels),
            'qs': torch.zeros(batch, lat, lon, num_levels),
            'qg': torch.zeros(batch, lat, lon, num_levels),
        }
        
        # 二维地表场 [batch, lat, lon]
        self.surface_2d = {
            'sp': torch.zeros(batch, lat, lon),
            'ts': torch.zeros(batch, lat, lon),
            'swe': torch.zeros(batch, lat, lon),
        }
        
        # 土壤场 [batch, lat, lon, soil_levels]
        self.soil = {
            't_soil': torch.zeros(batch, lat, lon, 4),
            'sm': torch.zeros(batch, lat, lon, 4),
        }
    
    def to_tensor(self) -> torch.Tensor:
        """转换为单一tensor用于网络计算"""
        pass
    
    def from_tensor(self, tensor: torch.Tensor):
        """从tensor恢复状态变量"""
        pass
```

---

#### 2.3 垂直坐标系统

**当前问题：**
- 只有`num_levels=37`，但没有定义垂直坐标
- 没有地形高度处理

**实现方案：**

```python
class VerticalCoordinate:
    """垂直坐标系统"""
    
    def __init__(self, 
                 num_levels: int = 37,
                 coord_type: str = 'hybrid',
                 p_top: float = 1.0):  # hPa
        """
        Args:
            coord_type: 'sigma' | 'pressure' | 'hybrid'
            p_top: 模式顶气压
        """
        self.num_levels = num_levels
        self.coord_type = coord_type
        self.p_top = p_top
        
        # 定义标准气压层 (hPa)
        self.pressure_levels = np.array([
            1000, 975, 950, 925, 900, 875, 850, 825, 800,
            775, 750, 700, 650, 600, 550, 500, 450, 400,
            350, 300, 250, 225, 200, 175, 150, 125, 100,
            70, 50, 30, 20, 10, 7, 5, 3, 2, 1
        ])
        
        if coord_type == 'hybrid':
            # Hybrid sigma-pressure坐标
            # p(k) = A(k) + B(k) * ps
            self.A_coef, self.B_coef = self._init_hybrid_coords()
    
    def _init_hybrid_coords(self):
        """初始化hybrid坐标系数"""
        # ECMWF L137方案简化版
        A = np.zeros(self.num_levels)
        B = np.linspace(0.0, 1.0, self.num_levels)
        
        # 近地层使用sigma坐标 (B~1)
        # 高层使用pressure坐标 (B~0, A>0)
        for k in range(self.num_levels):
            eta = k / (self.num_levels - 1)
            if eta < 0.2:  # 高层
                B[k] = 0.0
                A[k] = self.p_top + eta * 200  # 线性过渡
            elif eta < 0.8:  # 中层
                B[k] = (eta - 0.2) / 0.6
                A[k] = (1 - B[k]) * 200
            else:  # 近地层
                B[k] = 1.0
                A[k] = 0.0
        
        return A, B
    
    def compute_pressure(self, 
                        surface_pressure: torch.Tensor) -> torch.Tensor:
        """
        计算各层气压
        
        Args:
            surface_pressure: [batch, lat, lon]
        Returns:
            pressure: [batch, lat, lon, num_levels]
        """
        batch, lat, lon = surface_pressure.shape
        A = torch.tensor(self.A_coef, device=surface_pressure.device)
        B = torch.tensor(self.B_coef, device=surface_pressure.device)
        
        # p(k) = A(k) + B(k) * ps
        pressure = A[None, None, None, :] + \
                   B[None, None, None, :] * surface_pressure[..., None]
        
        return pressure
    
    def compute_layer_thickness(self, pressure: torch.Tensor) -> torch.Tensor:
        """计算层厚 dp"""
        # dp(k) = p(k+1) - p(k)
        dp = pressure[..., 1:] - pressure[..., :-1]
        return dp
```

**集成到模型：**
```python
class NeuralNWP(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 添加垂直坐标
        self.vertical_coord = VerticalCoordinate(
            num_levels=num_levels,
            coord_type='hybrid'
        )
        
        # 地形高度 [lat, lon]
        self.register_buffer('terrain_height', torch.zeros(img_size))
```

---

#### 2.4 边界条件处理

**实现方案：**

```python
class BoundaryConditions(nn.Module):
    """边界条件管理"""
    
    def __init__(self, 
                 model_type: str = 'global',  # 'global' or 'regional'
                 lat_size: int = 128,
                 lon_size: int = 256):
        super().__init__()
        self.model_type = model_type
        
        if model_type == 'global':
            # 全球模式：周期性边界（经度）+ 极点处理（纬度）
            self.lateral_bc = 'periodic'
        else:
            # 区域模式：需要大尺度强迫
            self.lateral_bc = 'relaxation'
            # 缓冲区宽度
            self.buffer_width = 5
    
    def apply_lateral_bc(self, state: torch.Tensor) -> torch.Tensor:
        """应用侧边界条件"""
        if self.lateral_bc == 'periodic':
            # 经度方向周期性
            return self._apply_periodic(state)
        elif self.lateral_bc == 'relaxation':
            # 松弛边界条件（区域模式）
            return self._apply_relaxation(state)
    
    def apply_top_bc(self, state: torch.Tensor) -> torch.Tensor:
        """上边界条件 - 海绵层"""
        # 在模式顶部添加Rayleigh阻尼
        pass
    
    def apply_surface_bc(self, 
                        state: torch.Tensor,
                        surface_fluxes: Dict) -> torch.Tensor:
        """下边界条件 - 来自陆面模式的通量"""
        pass
```

---

### Phase 3: 高级功能扩展 (2-3个月) 🟢 **增强优先级**

#### 3.1 重力波拖曳参数化

```python
class GravityWaveDrag(PhysicsParameterization):
    """重力波拖曳参数化"""
    
    def __init__(self, num_levels: int = 37, hidden_dim: int = 256):
        # 输入：u, v, T, 地形高度
        input_dim = num_levels * 3 + 1
        # 输出：u_tend, v_tend
        output_dim = num_levels * 2
        super().__init__(input_dim, hidden_dim, output_dim, num_layers=3)
```

#### 3.2 数据同化模块

```python
class DataAssimilation:
    """简单的数据同化模块"""
    
    def __init__(self, method: str = '3dvar'):
        self.method = method
    
    def assimilate(self,
                   background: torch.Tensor,
                   observations: Dict,
                   obs_operators: Dict) -> torch.Tensor:
        """
        Args:
            background: 背景场（短期预报）
            observations: 观测数据
            obs_operators: 观测算子
        Returns:
            analysis: 分析场
        """
        if self.method == '3dvar':
            return self._3dvar(background, observations, obs_operators)
        elif self.method == 'enkf':
            return self._enkf(background, observations, obs_operators)
```

#### 3.3 海洋耦合模块

```python
class OceanCoupler(nn.Module):
    """海洋-大气耦合器"""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        # 海温预测网络
        self.sst_predictor = nn.Sequential(...)
    
    def forward(self,
                sst: torch.Tensor,
                atmos_forcing: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            sst: 当前海温
            atmos_forcing: 大气强迫（风应力、热通量等）
        Returns:
            new_sst: 更新后的海温
            ocean_fluxes: 海洋向大气的通量
        """
        pass
```

#### 3.4 集成预报系统

```python
class EnsembleForecast:
    """集成预报系统"""
    
    def __init__(self, 
                 model: NeuralNWP,
                 num_members: int = 20):
        self.model = model
        self.num_members = num_members
    
    def generate_perturbations(self, 
                              initial_state: torch.Tensor) -> torch.Tensor:
        """生成初值扰动"""
        # 方法1: 基于历史误差统计
        # 方法2: 奇异向量
        # 方法3: 集合变换
        pass
    
    def run_ensemble(self,
                    initial_state: torch.Tensor,
                    num_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        运行集成预报
        
        Returns:
            ensemble_mean: 集合平均
            ensemble_spread: 集合离散度
        """
        # 生成扰动初值
        perturbed_states = self.generate_perturbations(initial_state)
        
        # 运行多个成员
        trajectories = []
        for i in range(self.num_members):
            traj = self.model.rollout(perturbed_states[i], num_steps)
            trajectories.append(traj)
        
        trajectories = torch.stack(trajectories)
        
        # 计算集合统计量
        ensemble_mean = trajectories.mean(dim=0)
        ensemble_spread = trajectories.std(dim=0)
        
        return ensemble_mean, ensemble_spread
```

---

## 📊 数据需求清单

### 训练数据

| 数据集 | 用途 | 时间分辨率 | 空间分辨率 | 变量 |
|--------|------|-----------|-----------|------|
| ERA5 | 大气状态 | 1小时 | 0.25° | T, u, v, q, w, sp, 云水 |
| ERA5-Land | 陆面状态 | 1小时 | 0.1° | 土壤T/湿度, 雪深, 地表通量 |
| IMERG/GPM | 降水 | 30分钟 | 0.1° | 降水率 |
| CERES | 辐射 | 1小时 | 1° | TOA辐射通量 |
| FLUXNET | 地表通量 | 30分钟 | 站点 | 感热、潜热通量 |

### 静态数据

- 地形高度 (GTOPO30)
- 土地利用类型 (MODIS)
- 土壤类型 (FAO)
- 植被类型和参数 (MODIS)
- 反照率气候态

---

## 🔧 技术架构改进

### 代码重构建议

```
neural_nwp/
├── model/
│   ├── __init__.py
│   ├── dynamic_core.py          # 动力核心
│   ├── parameterizations/       # 物理参数化
│   │   ├── __init__.py
│   │   ├── base.py             # 基类
│   │   ├── radiation.py
│   │   ├── convection.py
│   │   ├── boundary_layer.py
│   │   ├── microphysics.py     # 新增
│   │   ├── land_surface.py     # 新增
│   │   ├── large_scale_cloud.py # 新增
│   │   └── gravity_wave.py     # 新增
│   ├── coordinates.py           # 垂直坐标系统
│   ├── time_integrator.py       # 时间积分
│   ├── boundary_conditions.py   # 边界条件
│   └── neural_nwp.py           # 主模型
│
├── data/
│   ├── __init__.py
│   ├── era5_loader.py
│   ├── era5_land_loader.py
│   ├── preprocessor.py
│   ├── normalizer.py
│   └── state_vector.py          # 状态向量管理
│
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   ├── losses.py
│   ├── metrics.py
│   └── callbacks.py
│
├── inference/
│   ├── __init__.py
│   ├── predictor.py
│   ├── ensemble.py              # 集成预报
│   └── postprocessor.py
│
├── assimilation/
│   ├── __init__.py
│   ├── variational.py           # 变分同化
│   └── ensemble_kalman.py       # 集合卡尔曼滤波
│
├── evaluation/
│   ├── __init__.py
│   ├── deterministic.py         # 确定性预报评估
│   ├── probabilistic.py         # 概率预报评估
│   └── verification.py
│
├── visualization/
│   ├── __init__.py
│   ├── field_plots.py
│   ├── animation.py
│   └── diagnostics.py
│
├── config/
│   ├── model_config.yaml
│   ├── data_config.yaml
│   └── train_config.yaml
│
└── tests/
    ├── test_model.py
    ├── test_parameterizations.py
    └── test_integration.py
```

---

## 📈 性能优化

### 计算效率

1. **混合精度训练** (已支持)
   ```python
   # 使用Automatic Mixed Precision
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

2. **分布式训练**
   ```python
   # 使用torch.distributed
   import torch.distributed as dist
   from torch.nn.parallel import DistributedDataParallel as DDP
   ```

3. **模型优化**
   - Flash Attention 2.0
   - 梯度检查点 (Gradient Checkpointing)
   - 算子融合 (Operator Fusion)

### 显存优化

```python
# 梯度累积
accumulation_steps = 4

# 激活值检查点
from torch.utils.checkpoint import checkpoint
```

---

## 🎯 里程碑规划

### Milestone 1: 核心物理完善 (Week 1-8)
- [ ] 实现云微物理参数化
- [ ] 实现陆面过程模型
- [ ] 增加水凝物预报变量
- [ ] 重构模型以支持新变量
- [ ] 收集和预处理训练数据

### Milestone 2: 技术架构优化 (Week 9-12)
- [ ] 实现RK4时间积分
- [ ] 定义垂直坐标系统
- [ ] 实现边界条件处理
- [ ] 代码重构和模块化
- [ ] 完善单元测试

### Milestone 3: 训练和验证 (Week 13-16)
- [ ] 完整数据集准备
- [ ] 模型训练（24小时预报）
- [ ] 预报技巧评估
- [ ] 与ERA5对比验证
- [ ] 降水预报验证

### Milestone 4: 高级功能 (Week 17-20)
- [ ] 集成预报系统
- [ ] 数据同化接口
- [ ] 重力波拖曳
- [ ] 72小时预报能力
- [ ] 性能优化

---

## 📚 参考文献和资源

### 传统NWP系统

1. **WRF (Weather Research and Forecasting Model)**
   - 完整的中尺度NWP系统
   - 参考物理参数化方案设计

2. **ECMWF IFS**
   - 世界领先的全球预报系统
   - Hybrid垂直坐标系统

3. **NCEP GFS**
   - 美国全球预报系统
   - 物理参数化方案文档

### 机器学习NWP

1. **FourCastNet** (NVIDIA)
   - 纯数据驱动的全球预报
   - Adaptive Fourier Neural Operator

2. **Pangu-Weather** (华为)
   - 3D Earth-Specific Transformer
   - 高分辨率全球预报

3. **GraphCast** (Google DeepMind)
   - 图神经网络
   - 10天预报

4. **FengWu-GHR** (上海AI Lab)
   - 高分辨率预报
   - 集合预报

### 物理参数化

1. **云微物理**
   - Thompson方案
   - Morrison 2-moment方案
   - WSM6方案

2. **陆面过程**
   - Noah-MP LSM
   - CLM (Community Land Model)
   - JULES

3. **边界层**
   - YSU方案
   - MYNN方案

---

## ⚠️ 已知问题修复

### 立即修复

1. **移除调试代码**
   ```python
   # model.py line 119
   import pdb; pdb.set_trace()  # ← 删除这行
   ```

2. **辐射参数化占位变量**
   - 当前使用`torch.zeros`占位云量、天顶角
   - 需要实际计算这些变量

3. **地表气压处理**
   - 目前地表气压没有倾向项
   - 需要从质量守恒诊断

---

## 💼 人力和资源需求

### 研发团队配置建议

- **算法工程师** × 2: 模型开发和训练
- **气象专家** × 1: 物理参数化设计和验证
- **数据工程师** × 1: 数据处理和管道
- **系统工程师** × 1: 计算资源和部署

### 计算资源

- **训练**: 8×A100 (80GB) 或等效
- **存储**: 至少100TB (ERA5全球1小时数据)
- **推理**: 单卡V100/A100即可

### 时间估算

- **Phase 1**: 2-3个月（2人全职）
- **Phase 2**: 1-2个月（2人全职）
- **Phase 3**: 2-3个月（需要扩充团队）
- **总计**: 6-8个月达到可用水平

---

## 📞 下一步行动

### 立即开始（Week 1-2）

1. **设计扩展的状态向量**
   - 定义所有需要的预报变量
   - 设计数据结构

2. **实现云微物理框架**
   - 先实现简化版本
   - 建立训练和测试流程

3. **准备训练数据**
   - 下载ERA5和ERA5-Land
   - 设计数据预处理管道

4. **代码重构**
   - 按新架构重组代码
   - 添加单元测试

### 本月目标（Week 3-4）

5. **完成Phase 1的1-2项**
   - 云微物理和陆面模式
   - 集成到主模型

6. **初步训练**
   - 小规模数据测试
   - 验证梯度流动和收敛性

---

## 📝 总结

当前的Neural NWP项目是一个**很好的起点**，但要成为完整可用的NWP系统，需要：

1. **核心缺失**：补充云微物理和陆面过程（最关键）
2. **架构优化**：改进时间积分、垂直坐标、边界条件
3. **变量扩展**：从5个变量扩展到14+个变量
4. **高级功能**：集成预报、数据同化、耦合系统

**预计6-8个月**可以达到：
- ✅ 24-72小时确定性预报
- ✅ 准确的降水预报
- ✅ 合理的地表过程模拟
- ✅ 基本的集成预报能力

**长期目标（12个月+）**：
- 与业务NWP系统性能相当
- 支持数据同化
- 实时业务化运行
