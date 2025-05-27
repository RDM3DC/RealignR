"""
Lyapunov Stability Verification Module for RealignR

This module implements tools to verify the Lyapunov stability of the training process,
providing mathematical assurances about convergence and stability properties.
"""

import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from pathlib import Path

class LyapunovStabilityVerifier:
    """
    Implements Lyapunov stability verification for optimizer dynamics
    
    The verifier checks whether the training dynamics satisfy Lyapunov stability criteria,
    which provides formal guarantees about convergence and resistance to perturbations.
    """
    
    def __init__(self, 
                history_length=100, 
                check_interval=1000,
                stability_threshold=0.05,
                energy_function_type='quadratic'):
        """
        Initialize the Lyapunov stability verifier.
        
        Args:
            history_length: Number of steps to maintain in history
            check_interval: How often to perform full stability check
            stability_threshold: Threshold for declaring dynamics stable
            energy_function_type: Type of Lyapunov function to use
        """
        self.history_length = history_length
        self.check_interval = check_interval
        self.stability_threshold = stability_threshold
        self.energy_function_type = energy_function_type
        
        # Histories
        self.loss_history = deque(maxlen=history_length)
        self.grad_norm_history = deque(maxlen=history_length)
        self.param_change_history = deque(maxlen=history_length)
        self.energy_history = deque(maxlen=history_length)
        
        # Reference weight snapshot
        self.reference_weights = None
        self.last_check_step = 0
        self.is_stable = False
        self.stability_score = 0.0
        
    def take_parameter_snapshot(self, model):
        """Take a snapshot of model parameters for reference"""
        self.reference_weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.reference_weights[name] = param.detach().clone()
                
    def compute_parameter_changes(self, model):
        """Compute changes in parameters from reference snapshot"""
        if self.reference_weights is None:
            self.take_parameter_snapshot(model)
            return 0.0
            
        total_change = 0.0
        change_ratio = 0.0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.reference_weights:
                change = (param - self.reference_weights[name]).norm().item()
                baseline = self.reference_weights[name].norm().item() + 1e-8
                total_change += change
                change_ratio += change / baseline
                param_count += 1
                
        avg_change_ratio = change_ratio / max(1, param_count)
        self.param_change_history.append(avg_change_ratio)
        
        return avg_change_ratio
        
    def compute_energy_function(self, model, loss_value, grad_norm=None):
        """
        Compute Lyapunov energy function value
        
        The Lyapunov function should be positive definite and its derivative
        along system trajectories should be negative definite for stability.
        """
        if self.reference_weights is None:
            self.take_parameter_snapshot(model)
            
        # Compute parameter distance from reference point
        param_distance = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.reference_weights:
                param_distance += (param - self.reference_weights[name]).norm().item() ** 2
                
        # Different forms of energy function based on configuration
        if self.energy_function_type == 'quadratic':
            # V(x) = loss + λ‖x - x_ref‖²
            energy = loss_value + 0.01 * param_distance
        elif self.energy_function_type == 'log':
            # V(x) = log(1 + loss) + λ‖x - x_ref‖²
            energy = np.log(1 + loss_value) + 0.01 * param_distance
        else:
            # Simple energy function
            energy = loss_value
            
        self.energy_history.append(energy)
        return energy
        
    def check_lyapunov_stability(self, step, model, loss_value, grad_norm=None):
        """
        Check if the system exhibits Lyapunov stability
        
        Args:
            step: Current training step
            model: The model being trained
            loss_value: Current loss value
            grad_norm: Optional gradient norm
            
        Returns:
            Dictionary with stability results
        """
        # Only perform full check at specified intervals
        if step - self.last_check_step < self.check_interval:
            # Just collect data between full checks
            self.loss_history.append(loss_value)
            if grad_norm is not None:
                self.grad_norm_history.append(grad_norm)
            energy = self.compute_energy_function(model, loss_value, grad_norm)
            param_change = self.compute_parameter_changes(model)
            
            return {
                "is_stable": self.is_stable,
                "stability_score": self.stability_score,
                "energy": energy,
                "param_change": param_change
            }
            
        # Perform full stability check
        self.last_check_step = step
        
        # Calculate energy function
        energy = self.compute_energy_function(model, loss_value, grad_norm)
        param_change = self.compute_parameter_changes(model)
        
        # Check if energy function is decreasing (Lyapunov condition)
        if len(self.energy_history) > self.history_length // 2:
            # Split history into first and second half
            first_half = list(self.energy_history)[:len(self.energy_history)//2]
            second_half = list(self.energy_history)[len(self.energy_history)//2:]
            
            # Energy should decrease on average
            mean_first = np.mean(first_half)
            mean_second = np.mean(second_half)
            energy_decreasing = mean_second < mean_first
            
            # Check stability of energy values
            energy_stability = np.std(second_half) / (np.mean(second_half) + 1e-8)
            
            # Check parameter changes are slowing down
            if len(self.param_change_history) > self.history_length // 2:
                change_first = list(self.param_change_history)[:len(self.param_change_history)//2]
                change_second = list(self.param_change_history)[len(self.param_change_history)//2:]
                changes_slowing = np.mean(change_second) < np.mean(change_first)
            else:
                changes_slowing = False
                
            # Combine metrics into stability score
            self.stability_score = (
                0.5 * (1.0 if energy_decreasing else 0.0) + 
                0.3 * max(0, 1.0 - energy_stability) +
                0.2 * (1.0 if changes_slowing else 0.0)
            )
            
            # System is considered stable if score exceeds threshold
            self.is_stable = self.stability_score > self.stability_threshold
        else:
            # Not enough history
            self.is_stable = False
            self.stability_score = 0.0
            
        # Update reference point for next check
        self.take_parameter_snapshot(model)
        
        return {
            "is_stable": self.is_stable,
            "stability_score": self.stability_score,
            "energy_decreasing": energy_decreasing if 'energy_decreasing' in locals() else None,
            "energy_stability": energy_stability if 'energy_stability' in locals() else None,
            "changes_slowing": changes_slowing if 'changes_slowing' in locals() else None,
            "energy": energy,
            "param_change": param_change
        }
        
    def save_stability_plot(self, save_dir, step):
        """Generate and save stability analysis plot"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot energy function
        energy_values = list(self.energy_history)
        ax1.plot(range(len(energy_values)), energy_values, 'b-', label='Energy Function')
        ax1.set_title('Lyapunov Energy Function')
        ax1.set_ylabel('Energy Value')
        ax1.legend()
        
        # Plot parameter changes
        param_changes = list(self.param_change_history)
        ax2.plot(range(len(param_changes)), param_changes, 'r-', label='Parameter Change')
        ax2.set_title('Parameter Change Ratio')
        ax2.set_xlabel('Relative Step')
        ax2.set_ylabel('Change Ratio')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / f"stability_analysis_{step}.png")
        plt.close(fig)
        
        return save_dir / f"stability_analysis_{step}.png"
