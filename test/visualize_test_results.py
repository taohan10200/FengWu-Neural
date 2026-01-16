import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_comparison(results_path='test_results.pt', save_dir='test_viz'):
    """
    Visualize comparison between input and predicted trajectory
    """
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.Please run model.py first.")
        return

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load results
    print(f"Loading results from {results_path}...")
    data = torch.load(results_path)
    
    input_state = data['input']  # [batch, lat, lon, channels]
    trajectory = data['trajectory']  # [batch, steps, lat, lon, channels]
    config = data['config']
    
    num_vars = config['num_vars']
    num_levels = config['num_levels']
    
    # Select the first sample in batch
    batch_idx = 0
    
    # We will visualize Temperature (variable index 0) at a specific level (e.g., surface/bottom level)
    # Note: The channel layout is [num_vars * num_levels]
    # Assuming the layout is [var1_l1, var1_l2..., var2_l1...] or [var1_l1, var2_l1...]
    # Based on model.py extract_variables:
    # state_reshaped = state.reshape(batch, lat, lon, self.num_vars, self.num_levels)
    # So we can reshape to get variables easily
    
    input_reshaped = input_state.reshape(input_state.shape[0], input_state.shape[1], input_state.shape[2], num_vars, num_levels)
    traj_reshaped = trajectory.reshape(trajectory.shape[0], trajectory.shape[1], trajectory.shape[2], trajectory.shape[3], num_vars, num_levels)
    
    # Get Temperature (index 0) at surface (index 0)
    var_idx = 0 # Temperature
    level_idx = 0 # Surface
    var_name = "Temperature"
    
    input_field = input_reshaped[batch_idx, :, :, var_idx, level_idx].numpy()
    
    num_steps = trajectory.shape[1]
    
    # Create a figure with 1 row for input + num_steps for trajectory
    fig, axes = plt.subplots(1, num_steps + 1, figsize=(4 * (num_steps + 1), 4))
    
    # Plot Input
    im0 = axes[0].imshow(input_field, cmap='coolwarm')
    axes[0].set_title(f"Input (t=0)\n{var_name} Level {level_idx}")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot Trajectory Steps
    for t in range(num_steps):
        step_field = traj_reshaped[batch_idx, t, :, :, var_idx, level_idx].numpy()
        
        # Calculate difference from input to see the change
        diff = step_field - input_field
        
        im = axes[t+1].imshow(step_field, cmap='coolwarm')
        axes[t+1].set_title(f"Pred Step {t+1}\n{var_name} Level {level_idx}")
        plt.colorbar(im, ax=axes[t+1], fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'prediction_sequence.png')
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    
    # Also plot the difference to see what changed
    fig_diff, axes_diff = plt.subplots(1, num_steps, figsize=(4 * num_steps, 4))
    if num_steps == 1:
        axes_diff = [axes_diff]
        
    for t in range(num_steps):
        step_field = traj_reshaped[batch_idx, t, :, :, var_idx, level_idx].numpy()
        diff = step_field - input_field
        
        im = axes_diff[t].imshow(diff, cmap='RdBu_r')
        axes_diff[t].set_title(f"Diff (Step {t+1} - Input)")
        plt.colorbar(im, ax=axes_diff[t], fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    save_path_diff = os.path.join(save_dir, 'prediction_difference.png')
    plt.savefig(save_path_diff)
    print(f"Saved difference visualization to {save_path_diff}")

if __name__ == "__main__":
    visualize_comparison()
