"""
评估指标
"""

import torch
import numpy as np
from core.state import AtmosphericState

def rmse(pred_state: AtmosphericState, 
         target_state: AtmosphericState,
         variable:  str = 'T') -> float:
    """
    均方根误差 (RMSE)
    
    Args:
        pred_state: 预测状态
        target_state: 目标状态
        variable: 变量名
    
    Returns:
        rmse:  RMSE 值
    """
    pred = getattr(pred_state, variable)
    target = getattr(target_state, variable)
    
    mse = torch.mean((pred - target) ** 2)
    return torch.sqrt(mse).item()


def acc(pred_state: AtmosphericState,
        target_state: AtmosphericState,
        variable:  str = 'T') -> float:
    """
    距平相关系数 (ACC)
    
    Args:
        pred_state: 预测状态
        target_state: 目标状态
        variable: 变量名
    
    Returns:
        acc: ACC 值 [0, 1]
    """
    pred = getattr(pred_state, variable).flatten()
    target = getattr(target_state, variable).flatten()
    
    # 去均值
    pred_anom = pred - pred.mean()
    target_anom = target - target.mean()
    
    # 相关系数
    numerator = (pred_anom * target_anom).sum()
    denominator = torch.sqrt((pred_anom ** 2).sum() * (target_anom ** 2).sum())
    
    return (numerator / (denominator + 1e-8)).item()


def bias(pred_state: AtmosphericState,
         target_state: AtmosphericState,
         variable: str = 'T') -> float:
    """
    平均偏差
    
    Args:
        pred_state: 预测状态
        target_state: 目标状态
        variable: 变量名
    
    Returns:
        bias: 偏差值
    """
    pred = getattr(pred_state, variable)
    target = getattr(target_state, variable)
    
    return (pred.mean() - target.mean()).item()


def wind_speed_rmse(pred_state: AtmosphericState,
                   target_state: AtmosphericState) -> float:
    """
    风速 RMSE
    
    Returns:
        rmse: 风速 RMSE
    """
    pred_speed = torch.sqrt(pred_state.u ** 2 + pred_state.v ** 2)
    target_speed = torch.sqrt(target_state.u ** 2 + target_state.v ** 2)
    
    mse = torch.mean((pred_speed - target_speed) ** 2)
    return torch.sqrt(mse).item()


def get_all_metrics():
    """返回所有可用的指标"""
    return {
        'rmse_T': lambda p, t: rmse(p, t, 'T'),
        'rmse_u': lambda p, t: rmse(p, t, 'u'),
        'rmse_v': lambda p, t: rmse(p, t, 'v'),
        'rmse_ps': lambda p, t: rmse(p, t, 'ps'),
        'acc_T': lambda p, t:  acc(p, t, 'T'),
        'acc_u': lambda p, t: acc(p, t, 'u'),
        'bias_T': lambda p, t:  bias(p, t, 'T'),
        'wind_speed_rmse': wind_speed_rmse,
    }