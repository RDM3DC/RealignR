def log_spinor_memory(writer, spinor_layer, global_step):
    """
    Log the rotor memory of a SpinorLayer for each class as a histogram.
    Each class's memory is a list of rotor lists (per step).
    """
    import numpy as np
    for class_idx, memlist in spinor_layer.memory.items():
        if memlist:
            # memlist[-1] is a list of rotors for this class at the latest step
            arr = np.stack([
                np.array([float(r.value[0]) for r in rotors])
                for rotors in memlist
            ])
            writer.add_histogram(f"SpinorMemory/class_{class_idx}", arr, global_step)
"""
realignr_logging.py
-------------------
Meta-logging, slope tracking, memory heatmap visualization, and RealignR diagnostic tools.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os

def log_arp_metrics(writer, optimizer, global_step, tag_prefix="ARP"):
    """
    Log G_mean, alpha, mu, and loss slope for each ARP param group.
    """
    for grp_idx, group in enumerate(optimizer.param_groups):
        Gs = [optimizer.state[p]['G'].detach().cpu().mean().item()
              for p in group['params'] if 'G' in optimizer.state[p]]
        if Gs:
            writer.add_scalar(f'{tag_prefix}/G_mean_group{grp_idx}', np.mean(Gs), global_step)
        writer.add_scalar(f'{tag_prefix}/alpha_group{grp_idx}', group.get('alpha', float('nan')), global_step)
        writer.add_scalar(f'{tag_prefix}/mu_group{grp_idx}', group.get('mu', float('nan')), global_step)

def log_loss_slope(writer, loss, prev_loss, global_step, tag="ARP/loss_slope"):
    """
    Log the difference in loss (slope) for ARP plateau analysis.
    """
    slope = loss - prev_loss
    writer.add_scalar(tag, slope, global_step)
    return loss

def plot_rotor_memory_heatmap(spinor_layer, epoch, save_dir):
    """
    Visualize SpinorLayer's rotor memory as a heatmap.
    """
    mem_matrix = []
    for class_idx, memlist in spinor_layer.memory.items():
        if memlist:  # If memory is not empty
            # memlist[-1] is a list of rotors for this class at the latest step
            rotor_vals = [float(r.value[0]) for r in memlist[-1]]
            mem_matrix.append(rotor_vals)
    if mem_matrix:
        arr = np.array(mem_matrix)
        plt.figure(figsize=(12, 4))
        sns.heatmap(arr, cmap='coolwarm', cbar=True)
        plt.title(f'SpinorLayer Rotor Memory (epoch {epoch})')
        plt.xlabel('Rotor Index')
        plt.ylabel('Class')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"rotor_memory_heatmap_epoch_{epoch}.png"))
        plt.close()
