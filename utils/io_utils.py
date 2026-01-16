"""
输入输出工具
"""

import torch
from pathlib import Path
from typing import List, Dict, Any

from configs.model_config import ModelConfig
from core.state import AtmosphericState

def save_forecast_results(trajectory: List[AtmosphericState],
                         config: ModelConfig,
                         filepath: str,
                         metadata:  Dict[str, Any] = None):
    """
    保存预报结果
    
    Args:
        trajectory: 状态序列
        config: 模型配置
        filepath: 输出文件路径
        metadata: 元数据
    """
    data = {
        'config': config,
        'trajectory': [state.as_dict() for state in trajectory],
        'n_timesteps': len(trajectory),
    }
    
    if metadata: 
        data['metadata'] = metadata
    
    torch.save(data, filepath)

def load_forecast_results(filepath: str) -> tuple: 
    """
    加载预报结果
    
    Returns:
        trajectory, config, metadata
    """
    data = torch.load(filepath)
    
    config = data['config']
    trajectory = [AtmosphericState.from_dict(state_dict, config) 
                 for state_dict in data['trajectory']]
    metadata = data.get('metadata', {})
    
    return trajectory, config, metadata