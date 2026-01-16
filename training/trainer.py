"""
模型训练器
"""

import torch
import torch.nn as nn
from typing import Dict, Callable, List
from tqdm import tqdm

from configs.model_config import ModelConfig
from models.atmospheric_model import AtmosphericModel
from core.state import AtmosphericState

class ModelTrainer:
    """模型训练器（用于混合模型）"""
    
    def __init__(self, 
                 model: AtmosphericModel,
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cuda',
                 gradient_clip: float = 1.0):
        """
        Args: 
            model: 大气模型
            optimizer: 优化器
            device: 设备
            gradient_clip:  梯度裁剪阈值
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.gradient_clip = gradient_clip
        
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, 
                   initial_state: AtmosphericState,
                   target_states: List[AtmosphericState],
                   loss_fn: nn.Module) -> float:
        """
        单步训练
        
        Args:
            initial_state: 初始状态
            target_states: 目标状态序列
            loss_fn: 损失函数
        
        Returns:
            loss: 损失值
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        predicted_states = self.model(initial_state, len(target_states) - 1)
        
        # 计算损失（所有时间步）
        total_loss = 0.0
        for t, (pred, target) in enumerate(zip(predicted_states[1:], target_states[1:])):
            # 时间加权：早期预报权重更高
            weight = 1.0 / (1.0 + 0.1 * t)
            loss = loss_fn(pred, target) * weight
            total_loss += loss
        
        # 反向传播
        total_loss.backward()
        
        # 梯度裁剪
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        
        # 更新参数
        self.optimizer.step()
        
        return total_loss.item()
    
    def train_epoch(self, 
                   train_loader: torch.utils.data.DataLoader,
                   loss_fn: nn.Module,
                   epoch: int) -> float:
        """
        训练一个 epoch
        
        Args: 
            train_loader: 训练数据加载器
            loss_fn:  损失函数
            epoch: 当前 epoch
        
        Returns: 
            avg_loss: 平均损失
        """
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (initial_states, target_states) in enumerate(pbar):
            # 移动到设备
            initial_states = initial_states.to(self.device)
            target_states = target_states.to(self.device)
            
            # 训练步
            loss = self.train_step(initial_states, [target_states], loss_fn)
            epoch_loss += loss
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss:.6f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def evaluate(self, 
                val_loader: torch.utils.data.DataLoader,
                loss_fn: nn.Module,
                metrics: Dict[str, Callable] = None) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            val_loader: 验证数据加载器
            loss_fn: 损失函数
            metrics: 额外的评估指标
        
        Returns:
            results: 评估结果
        """
        self.model.eval()
        
        total_loss = 0.0
        metric_values = {name: 0.0 for name in (metrics or {}).keys()}
        
        with torch.no_grad():
            for initial_states, target_states in tqdm(val_loader, desc="Evaluating"):
                initial_states = initial_states.to(self.device)
                target_states = target_states.to(self.device)
                
                # 前向传播
                predicted_states = self.model(initial_states, 1)
                
                # 计算损失
                loss = loss_fn(predicted_states[-1], target_states)
                total_loss += loss.item()
                
                # 计算其他指标
                if metrics:
                    for name, metric_fn in metrics.items():
                        metric_values[name] += metric_fn(predicted_states[-1], target_states)
        
        # 平均
        n_batches = len(val_loader)
        avg_loss = total_loss / n_batches
        
        results = {'loss': avg_loss}
        for name in metric_values: 
            results[name] = metric_values[name] / n_batches
        
        self.val_losses.append(avg_loss)
        
        return results
    
    def save_checkpoint(self, filepath: str, epoch: int, **kwargs):
        """保存训练检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses':  self.val_losses,
            **kwargs
        }
        torch.save(checkpoint, filepath)
        print(f"✓ 检查点已保存:  {filepath}")
    
    def load_checkpoint(self, filepath:  str):
        """加载训练检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"✓ 检查点已加载: {filepath}")
        return checkpoint['epoch']