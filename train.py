"""
Training Script for Neural NWP Model
神经网络数值气象预测模型训练脚本
"""
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
import argparse
import json
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Optional
import matplotlib.pyplot as plt
from mmengine.config import Config, DictAction

from models.model import NeuralNWP, create_model, PhysicsConstrainedLoss
from dataset import  create_dataloaders

class Trainer:
    """训练器类"""
    
    def __init__(self,
                 cfg: Config,
                 model: NeuralNWP,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict,
                 device: torch.device):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.cfg = cfg
        self.device = device
        
        self.level_mapping =  [cfg.total_levels.index(val) for val in cfg.pressure_levels if val in cfg.total_levels ]
        print( self.level_mapping)
        self.mean, self.std = self.get_mean_std() #read the channel-wise mean and std according to the defined variable in configuration.self.mean = torch.tensor(self.mean[None,None,None,:]).to(self.device)
        self.std =  torch.tensor(self.std[None,None,None,:]).to(self.device)
        
                
        # 损失函数 - 使用物理约束损失
        use_physics_loss = config.get('use_physics_constrained_loss', True)
        
        if use_physics_loss:
            # 准备反归一化参数 [channels]
            loss_mean = self.mean.squeeze() if self.mean is not None else None
            loss_std = self.std.squeeze() if self.std is not None else None

            self.criterion = PhysicsConstrainedLoss(
                pressure_vars=config.get('vnames', {}).get('pressure', ['t', 'u', 'v', 'q']),
                single_vars=config.get('vnames', {}).get('single', ['v10', 'u10', 't2m', 'sp', 'msl']),
                pressure_levels=config.get('pressure_level', [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]),
                img_size=config.get('img_size', (128, 256)),
                mean=loss_mean,
                std=loss_std,
                mse_weight=config.get('mse_weight', 1.0),
                mass_weight=config.get('mass_conservation_weight', 0.1),
                energy_weight=config.get('energy_conservation_weight', 0.1),
                moisture_weight=config.get('moisture_conservation_weight', 0.05),
                momentum_weight=config.get('momentum_conservation_weight', 0.05)
            )
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('max_iters', 100000),
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # 混合精度训练
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # 训练状态
        self.current_iter = 0
        self.best_val_loss = float('inf')
        
        # 保存目录
        self.save_dir = config.get('save_dir', './checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)

    def normalization(self, data):
        data -= self.mean
        data /= self.std
        return data
    
    def de_normalization(self, data):
        data *= self.std
        data += self.mean
        return data   
                    
    def get_mean_std(self):
        with open('./models/mean_std.json',mode='r') as f:
            mean_std = json.load(f)
            f.close()
        with open('./models/mean_std_single.json',mode='r') as f:
            mean_std_single = json.load(f)
            f.close()
        mean_list, std_list = [],[]
        for  vname in self.config['vnames'].get('pressure'):
            mean_list += [mean_std['mean'][vname][idx] for idx in self.level_mapping]
            std_list += [mean_std['std'][vname][idx] for idx in self.level_mapping]

        for vname in self.config['vnames'].get('single'):
            mean_list.append(mean_std_single['mean'][vname])
            std_list.append(mean_std_single['std'][vname])
            
        return np.array(mean_list, dtype=np.float32), np.array(std_list, dtype=np.float32)

    def visualize(self, inputs, targets, preds_seq, iteration):
        """可视化预测结果 (支持多步，单图汇总)"""
        vis_dir = os.path.join(self.save_dir, 'vis')
        os.makedirs(vis_dir, exist_ok=True)

        # Handle inputs (T=0)
        if inputs.dim() == 5:
            inputs = inputs.squeeze(1)
        # inputs: [Batch, Lat, Lon, Channels] -> take first sample
        inp_np = self.de_normalization(inputs.clone()).detach().cpu().numpy()[0] # [Lat, Lon, Chan]
        
        # Handle targets
        # targets: [Batch, Time, Lat, Lon, Channels] -> take first sample
        targets_np = self.de_normalization(targets.clone()).detach().cpu().numpy()[0] # [Time, Lat, Lon, Chan]
        
        # Handle preds_seq
        if isinstance(preds_seq, torch.Tensor):
            preds_seq = {1: preds_seq}
        
        # Sort steps
        steps = sorted(preds_seq.keys())
        num_steps = len(steps)
        
        # 获取配置
        pressure_vars = self.cfg.get('pressure_vars')
        single_vars = self.cfg.get('single_vars')
        pressure_levels = self.cfg.get('pressure_levels')
        
        num_levels = len(pressure_levels)
        num_pressure_vars = len(pressure_vars)
        
        # 选择要画的变量
        plot_configs = []
        
        # 1.T at 500hPa
        if 't' in pressure_vars and 500 in pressure_levels:
            var_idx = pressure_vars.index('t')
            level_idx = pressure_levels.index(500)
            ch_idx = var_idx * num_levels + level_idx
            plot_configs.append({'name': 'T_500hPa', 'idx': ch_idx, 'cmap': 'coolwarm'})
            
        # 2.Z at 500hPa (if available)
        if 'z' in pressure_vars and 500 in pressure_levels:
            var_idx = pressure_vars.index('z')
            level_idx = pressure_levels.index(500)

            ch_idx = var_idx * num_levels + level_idx
            plot_configs.append({'name': 'Z_500hPa', 'idx': ch_idx, 'cmap': 'viridis'})
            
        # 3.T2M (if available)
        if 't2m' in single_vars:
            var_idx = single_vars.index('t2m')
            ch_idx = num_pressure_vars * num_levels + var_idx
            plot_configs.append({'name': 'T2M', 'idx': ch_idx, 'cmap': 'coolwarm'})
            
        # 4.MSL or SP
        if 'msl' in single_vars:
            var_idx = single_vars.index('msl')
            ch_idx = num_pressure_vars * num_levels + var_idx
            plot_configs.append({'name': 'MSL', 'idx': ch_idx, 'cmap': 'viridis'})
        elif 'sp' in single_vars:
            var_idx = single_vars.index('sp')
            ch_idx = num_pressure_vars * num_levels + var_idx
            plot_configs.append({'name': 'SP', 'idx': ch_idx, 'cmap': 'viridis'})
        

        if not plot_configs:
            return

        num_vars = len(plot_configs)
        
        # Grid: Rows = Var * 3 (Target, Pred, Error), Cols = 1 (Input) + Num Steps
        nrows = num_vars * 3
        ncols = 1 + num_steps # T=0, T=1...T=6
        
        # Adjust figure size based on grid
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), constrained_layout=True)
        # Ensure axes is always valid 2D array even for small grids
        if nrows == 1 and ncols == 1: axes = np.array([[axes]])
        elif nrows == 1: axes = axes[None, :]
        elif ncols == 1: axes = axes[:, None]
        
        for v_idx, cfg in enumerate(plot_configs):
            ch_idx = cfg['idx']
            name = cfg['name']
            cmap = cfg['cmap']
            
            # Row indices for this variable
            row_tgt = v_idx * 3
            row_pred = v_idx * 3 + 1
            row_err = v_idx * 3 + 2
            
            # --- Column 0: Input (T=0) ---
            img_inp = inp_np[..., ch_idx]
            
            # Plot Input in Target Row (as GT sequence start)
            ax = axes[row_tgt, 0]
            im = ax.imshow(img_inp, cmap=cmap)
            ax.set_title(f'{name} T=0 (Inp)')
            # Only turn off x/y ticks but keep frame? or clear all.ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Plot Input in Pred Row (as Pred sequence start - known)
            ax = axes[row_pred, 0]
            im = ax.imshow(img_inp, cmap=cmap)
            ax.set_title(f'{name} T=0 (Inp)')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Error Row (T=0) - Empty/Text
            ax = axes[row_err, 0]
            ax.axis('off')
            ax.text(0.5, 0.5, "No Error\nat T=0", ha='center', va='center')
            
            # --- Columns 1..N: Steps ---
            for s_i, step in enumerate(steps):
                col = s_i + 1
                
                # Get Target (step-1 for index)
                if step-1 < targets_np.shape[0]:
                    tgt_img = targets_np[step-1, ..., ch_idx]
                else:
                    tgt_img = np.zeros_like(img_inp)
                
                # Get Pred
                pred_tensor = preds_seq[step]
                pred_img_full = self.de_normalization(pred_tensor.clone()).detach().cpu().numpy()[0]
                pred_img = pred_img_full[..., ch_idx]
                
                # Error
                err_img = pred_img - tgt_img
                
                # Metrics
                bias = np.mean(err_img)
                rmse = np.sqrt(np.mean(err_img**2))
                
                # Plot Target
                ax = axes[row_tgt, col]
                im = ax.imshow(tgt_img, cmap=cmap)
                ax.set_title(f'T={step} GT')
                ax.set_xticks([])
                ax.set_yticks([])
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # Plot Pred
                ax = axes[row_pred, col]
                im = ax.imshow(pred_img, cmap=cmap)
                ax.set_title(f'T={step} Pred')
                ax.set_xticks([])
                ax.set_yticks([])
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # Plot Error
                ax = axes[row_err, col]
                im = ax.imshow(err_img, cmap='bwr') # Blue-White-Red for error
                
                # Find max error for scale? Or auto.Auto is usually fine for bwr but 0 should be center.
                # Let's enforce symmetric CLim for error if possible, or just let it be.
                # For quick viz, auto is fine.ax.set_title(f'Err B:{bias:.2f} R:{rmse:.2f}')
                ax.set_xticks([])
                ax.set_yticks([])
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Labels for rows
        # We can add text to the left of the subplot grid
        # or use set_ylabel on the first column axes (we turned off ticks, but labels show? ticks=[] hides ticks)
        for v_idx, cfg in enumerate(plot_configs):
            axes[v_idx*3, 0].set_ylabel("Ground Truth", fontsize=10)
            axes[v_idx*3+1, 0].set_ylabel("Prediction", fontsize=10)
            axes[v_idx*3+2, 0].set_ylabel("Error", fontsize=10)

        plt.suptitle(f"Iteration {iteration}", fontsize=16)
        
        # Save
        plt.savefig(os.path.join(vis_dir, f'iter_{iteration}_rollout.png'))
        plt.close()

    def train_step(self, batch, iteration) -> Dict[str, float]:
        """训练一个step (Autoregressive Rollout)"""
        self.model.train()
        inputs, targets, ts_info = batch
        
        # Determine input shape and permute to [Batch, ..., Channel]
        # Common scenario: DataLoader returns [B, T, H, W, C] for inputs/targets
        # 移到GPU并归一化
        inputs = self.normalization(inputs.to(self.device))
        targets = self.normalization(targets.to(self.device))
        
        # 初始状态 [batch, lat, lon, channels]
        # If input has time dim 1, squeeze it.state = inputs.squeeze(1) if inputs.dim() == 5 else inputs
        
        # 获取训练配置
        dt = self.config.get('dt', 3600.0)
        rollout_steps = self.config.get('rollout_steps', 6)
        supervision_steps = self.config.get('supervision_steps', [1, 3, 6])
        
        # 前向传播
        self.optimizer.zero_grad()
        
        loss = 0.0
        mse_total = 0.0
        loss_count = 0
        step_losses = {}
        
        # 保存预测结果用于可视化
        collected_preds = {}
        
        if self.use_amp:
            with autocast():
                # 自回归滚动预测
                for step in range(1, rollout_steps + 1):
                    # 前向推演一步
                    # state: [batch, lat, lon, channels]
                    next_state = self.model(state, dt=dt)
                    
                    # 获取当前步的真值 (targets索引从0开始，对应step 1)
                    target_step = targets[:, step-1]
                    
                    # 如果当前步需要监督
                    if step in supervision_steps:
                        loss_dict = self.criterion(next_state, target_step, state) # state作为initial_state用于守恒律
                        loss += loss_dict['total']
                        mse_total += loss_dict['mse']
                        loss_count += 1
                        step_losses[f'loss_{step}'] = loss_dict['total'].item()
                        step_losses[f'mse_{step}'] = loss_dict['mse'].item()
                        
                    # 收集预测结果 (仅保存需要的步骤或者全部)
                    collected_preds[step] = next_state
                        
                    # 更新状态用于下一步输入
                    # 训练时通常使用预测值继续滚动 (Autoregressive)
                    state = next_state
                    
                # 平均损失
                if loss_count > 0:
                    loss = loss / loss_count
                    mse_total = mse_total / loss_count
            
            # 反向传播
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.get('max_grad_norm', 1.0))
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 标准精度训练逻辑 (同上)
            for step in range(1, rollout_steps + 1):
                next_state = self.model(state, dt=dt)
                target_step = targets[:, step-1]
                
                if step in supervision_steps:
                    loss_dict = self.criterion(next_state, target_step, state)
                    loss += loss_dict['total']
                    mse_total += loss_dict['mse']
                    loss_count += 1
                    step_losses[f'loss_{step}'] = loss_dict['total'].item()
                    step_losses[f'mse_{step}'] = loss_dict['mse'].item()
                
                collected_preds[step] = next_state
                    
                state = next_state
            
            if loss_count > 0:
                loss = loss / loss_count
                mse_total = mse_total / loss_count
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.get('max_grad_norm', 1.0))
            self.optimizer.step()
            
        # 可视化 (绘制所有收集到的步骤)
        if iteration % 1000 == 0:
            self.visualize(inputs.squeeze(1), targets, collected_preds, iteration)
            
        metrics = {
            'loss': loss.item(),
            'mse': mse_total.item()
        }
        metrics.update(step_losses)
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        total_loss = 0.0
        total_mse = 0.0
        num_batches = 0
        
        for inputs, targets in tqdm(self.val_loader, desc='Validation'):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            initial_state = inputs[:, -1]
            target_state = targets[:, 0]

            # 前向传播
            pred = self.model(initial_state, dt=self.config.get('dt', 3600.0))
            loss_dict = self.criterion(pred, target_state, initial_state)
            
            total_loss += loss_dict['total'].item()
            total_mse += loss_dict['mse'].item()
            num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_mse': total_mse / num_batches
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'iteration': self.current_iter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # 保存最新的检查点
        torch.save(checkpoint, os.path.join(self.save_dir, 'latest.pth'))
        
        # 保存最好的检查点
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best.pth'))
            print(f"✓ Saved best model with val_loss: {self.best_val_loss:.6f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 兼容旧的epoch格式
        if 'iteration' in checkpoint:
            self.current_iter = checkpoint['iteration']
        elif 'epoch' in checkpoint:
            self.current_iter = checkpoint['epoch'] * 1000 # 粗略估计
            
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"✓ Loaded checkpoint from iteration {self.current_iter}")
    
    def train(self):
        """完整训练流程"""
        max_iters = self.config.get('max_iters', 100000)
        val_interval = self.config.get('val_interval', 1000)
        save_interval = self.config.get('save_interval', 5000)
        
        print(f"Starting training for {max_iters} iterations")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        train_iterator = iter(self.train_loader)
        
        pbar = tqdm(range(self.current_iter, max_iters), initial=self.current_iter, total=max_iters)
        
        history = {}
        
        for iteration in pbar:
            self.current_iter = iteration
            
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_loader)
                batch = next(train_iterator)
                
            # 训练一步
            metrics = self.train_step(batch, iteration)
            
            # 更新历史记录
            for k, v in metrics.items():
                if k not in history:
                    history[k] = []
                history[k].append(v)
                if len(history[k]) > 1000:
                    history[k].pop(0)
            
            # 更新进度条
            postfix_dict = {}
            for k, v in history.items():
                if 'loss' in k or 'mse' in k:
                    postfix_dict[k] = sum(v) / len(v)
            postfix_dict['lr'] = self.optimizer.param_groups[0]['lr']
            
            pbar.set_postfix(postfix_dict)
            
            # 验证
            if (iteration + 1) % val_interval == 0:
                val_metrics = self.validate()
                print(f"\nIteration {iteration + 1}:")
                print(f"  Train Loss: {metrics['loss']:.6f}")
                print(f"  Val Loss: {val_metrics['val_loss']:.6f}")
                
                # 保存最佳模型
                is_best = val_metrics['val_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(is_best=is_best)
                
                self.model.train() # Switch back to train mode
            
            # 定期保存
            if (iteration + 1) % save_interval == 0:
                self.save_checkpoint(is_best=False)
            
            # 更新学习率
            self.scheduler.step()
        
        print("\n✓ Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train Neural NWP Model')
    
    # 配置文件
    parser.add_argument('--config', type=str, default='config.py',
                        help='Config file path')
    parser.add_argument('--cfg-options',
                        nargs='+',
                        action=DictAction,
                        help='Override config options')
    
    # 其他参数
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # 加载配置文件
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 保存配置
    save_dir = cfg.checkpoint.get('save_dir', './checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    cfg.dump(os.path.join(save_dir, 'config.py'))
    
    # 创建数据加载器
    print("Creating data loaders...")
    # detect distributed world size and rank
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0

    train_loader, val_loader = create_dataloaders(
        dataset_cfg=cfg.data,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        world_size=world_size,
        rank=rank,
    )
    # 创建模型
    print("Creating model...")
    model = create_model(cfg.model)
    
    # 创建训练器配置
    trainer_config = {
        'max_iters': cfg.training.get('max_iters', 100000),
        'val_interval': cfg.training.get('val_interval', 100000),
        'save_interval': cfg.training.get('save_interval', 5000),
        'epochs': cfg.training.epochs,
        'learning_rate': cfg.training.learning_rate,
        'weight_decay': cfg.training.weight_decay,
        'dt': cfg.training.dt,
        'use_amp': cfg.training.use_amp,
        'mse_weight': cfg.training.mse_weight,
        'conservation_weight': cfg.training.conservation_weight,
        'save_dir': save_dir,
        'num_vars': len(cfg.model.pressure_vars) + len(cfg.model.single_vars),
        'num_levels': len(cfg.model.pressure_levels),
        'img_size': cfg.model.img_size,
        'pressure_levels': cfg.data.get('pressure_level', []),
        'vnames': cfg.data.get('vnames', [])
    
    }
    
    # 创建训练器
    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        device=device
    )
    
    # 从检查点恢复
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
