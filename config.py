# Neural NWP 配置文件
# 使用 mmengine Config 格式

# 模型架构配置

total_levels= [1000.,  975.,  950.,  925.,  900.,  875.,  850.,  825.,  800.,
 775.,  750.,  700.,  650.,  600.,  550.,  500.,  450.,  400.,
 350.,  300.,  250.,  225.,  200.,  175.,  150.,  125.,  100.,
 70.,   50.,   30.,   20.,   10.,    7.,    5.,    3.,    2.,
 1.]

pressure_levels = [1000.,  975.,  950.,  925.,  900.,  875.,  850.,  825.,  800.,
 775.,  750.,  700.,  650.,  600.,  550.,  500.,  450.,  400.,
 350.,  300.,  250.,  225.,  200.,  175.,  150.,  125.,  100.,
 70.,   50.,   30.,   20.,   10.,    7.,    5.,    3.,    2.,
 1.]

pressure_vars=['t', 'u', 'v', 'q']  # 气压层变量
single_vars=['v10', 'u10', 'v100', 'u100', 't2m', 'sp', 'msl']  # 单层变量
    
model = dict(
    img_size=(128, 256),  # (lat, lon)
    pressure_vars=pressure_vars,  # 气压层变量
    single_vars=single_vars,  # 单层变量
    pressure_levels=pressure_levels,
    hidden_dim=512,       # Transformer隐藏维度
    parameterization_dim=256,  # 参数化网络维度
    num_heads=8,          # Transformer注意力头数
)



# 训练配置
training = dict(
    epochs=100,
    batch_size=1,
    learning_rate=1.0e-4,
    weight_decay=1.0e-5,
    min_lr=1.0e-6,
    
    # 损失函数权重
    mse_weight=1.0,
    conservation_weight=0.1,
    
    # 混合精度训练
    use_amp=True,
    
    # 时间步长(hour)
    dt=1.0, # 模型内部step delta t
    rollout_steps=6, # 训练时向前rollout多少步
    supervision_steps=[1, 3, 6], # 需要监督的时间步 (1-based index)
    
    # 梯度裁剪
    max_grad_norm=1.0,
    
)

# 数据配置
data = dict(
    num_workers=4,
    # 数据归一化
    normalize=True,
    # ERA5变量配置 - 与model配置保持一致
    vnames=dict(
        pressure=pressure_vars ,
        single=single_vars,
    ),
    
    # 对象存储配置
    bucket=dict(
        ak_sk="ai4earth",
        endpoint='http://10.140.27.254',
    ),
    
    # ERA5时间范围配置
    era5_time_range=dict(
        data_root='era5_np_float32',
        train_st='1979-01-01T00:00:00',
        train_et='2023-12-31T00:00:00',
        val_st='2024-01-01T00:00:00',
        val_et='2024-12-31T00:00:00',
        interval='1h',
    ),
    
    # 分析场时间范围配置
    # analysis_time_range=dict(
    #     data_root='nwp_initial_fileds/analysis_MIR/np2001x4000',
    #     st='2016-03-08T00:00:00',
    #     et='2023-12-31T00:00:00',
    #     iterval='6h',
    # ),
    
    # 垂直层次 (hPa)
    pressure_levels=pressure_levels,
)

# 推理配置
inference = dict(
    num_steps=10,
    dt=3600.0,
    batch_size=1,
    
    # 可视化配置
    visualization=dict(
        create_animation=True,
        fps=2,
        default_variable=0,  # 温度
        default_level=0,
    ),
)

# 检查点配置
checkpoint = dict(
    save_dir="./checkpoints",
    save_interval=1,  # 每多少个epoch保存一次
    keep_last_n=5,    # 保留最近N个检查点
)

# 日志配置
logging = dict(
    log_dir="./logs",
    log_interval=10,  # 每多少个batch打印一次
    use_tensorboard=True,
)
