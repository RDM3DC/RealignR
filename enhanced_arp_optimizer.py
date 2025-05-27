import torch
import math

class EnhancedARPOptimizer(torch.optim.Optimizer):
    r"""
    EnhancedARPOptimizer implements an improved version of RealignR:
        G_{t+1} = (1 - mu) * G_t + alpha * |grad|
        p_{t+1} = p_t - lr * G_{t+1} * sign(grad)

    With additional features:
    - Cosine learning rate decay support
    - G-statistic diagnostics
    - Multiple hyperparameter presets

    Args:
        params (iterable): iterable of parameters to optimize
        lr (float): initial step size (default: 1e-3)
        alpha (float): conduction growth rate (default: 1e-2)
        mu (float): conduction decay rate (default: 1e-3)
        weight_decay (float): weight decay for parameters (default: 0)
        variant (str): optimization variant - options:
          - 'default': Original α=0.01, μ=0.001
          - 'aggressive': Higher α=0.05, μ=0.005
          - 'balanced': Mid-range α=0.02, μ=0.002
        enable_g_diagnostics (bool): whether to track G-statistics (default: False)
    """

    def __init__(self, params,
                 lr=1e-3,
                 alpha=None,  # Will be set based on variant
                 mu=None,     # Will be set based on variant
                 weight_decay=0.0,
                 clamp_G_min=1e-4,
                 clamp_G_max=10.0,
                 variant='default',
                 enable_g_diagnostics=False):
        
        # Set hyperparameters based on variant
        if variant == 'default':
            alpha = 1e-2 if alpha is None else alpha
            mu = 1e-3 if mu is None else mu
        elif variant == 'aggressive':
            alpha = 5e-2 if alpha is None else alpha
            mu = 5e-3 if mu is None else mu
        elif variant == 'balanced':
            alpha = 2e-2 if alpha is None else alpha
            mu = 2e-3 if mu is None else mu
        else:
            raise ValueError(f"Unknown variant: {variant}. Choose from 'default', 'aggressive', 'balanced'")
            
        defaults = dict(lr=lr, alpha=alpha, mu=mu,
                        weight_decay=weight_decay,
                        clamp_G_min=clamp_G_min,
                        clamp_G_max=clamp_G_max,
                        enable_g_diagnostics=enable_g_diagnostics)
        super().__init__(params, defaults)
        
        # Initialize diagnostic data if enabled
        self.g_stats = {
            'mean': [],
            'min': [],
            'max': [],
            'std': [],
            'steps': 0
        }

    def get_g_statistics(self):
        """Return G-statistics for diagnostics"""
        return self.g_stats
    
    def print_g_statistics(self):
        """Print G-statistics for the current step"""
        if not self.g_stats['mean']:
            print("No G-statistics available yet")
            return
            
        idx = -1  # Get the latest statistics
        print(f"G-statistics (step {self.g_stats['steps']}):")
        print(f"  Mean: {self.g_stats['mean'][idx]:.6f}")
        print(f"  Min:  {self.g_stats['min'][idx]:.6f}")
        print(f"  Max:  {self.g_stats['max'][idx]:.6f}")
        print(f"  Std:  {self.g_stats['std'][idx]:.6f}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        all_g_values = []  # For collecting G values across parameters
        
        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            mu = group['mu']
            weight_decay = group['weight_decay']
            clamp_min = group['clamp_G_min']
            clamp_max = group['clamp_G_max']
            enable_g_diagnostics = group['enable_g_diagnostics']
            
            # --- Warm-start G from initial gradient ---
            if 'warm_started' not in self.state:
                for p in group['params']:
                    if p.grad is not None:
                        grad = p.grad.detach().abs()
                        state = self.state[p]
                        if 'G' not in state:
                            # Initialize G based on the first gradient
                            safe_alpha = max(alpha, 1e-6)
                            safe_mu = max(mu, 1e-6)
                            state['G'] = (safe_alpha / safe_mu) * grad.clamp(min=1e-5, max=10.0)
                self.state['warm_started'] = True
                print(f"🔥 Warm-started G with α={alpha:.4f}, μ={mu:.4f}")

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("EnhancedARPOptimizer does not support sparse gradients")

                if weight_decay != 0:
                    p.data.add_(-weight_decay * p.data)

                state = self.state[p]
                if 'G' not in state:
                    state['G'] = torch.zeros_like(p.data)

                G = state['G']
                G.mul_(1 - mu).add_(alpha * grad.abs())

                if clamp_min is not None or clamp_max is not None:
                    G.clamp_(min=clamp_min, max=clamp_max)
                    
                # Collect G statistics if diagnostics are enabled
                if enable_g_diagnostics:
                    all_g_values.append(G.flatten())

                p.data.add_(G * grad.sign(), alpha=-lr)
        
        # Update G-statistics if enabled and we have G values
        if all_g_values and enable_g_diagnostics:
            all_g = torch.cat(all_g_values)
            self.g_stats['mean'].append(all_g.mean().item())
            self.g_stats['min'].append(all_g.min().item())
            self.g_stats['max'].append(all_g.max().item())
            self.g_stats['std'].append(all_g.std().item())
            self.g_stats['steps'] += 1

        return loss


class CosineARPScheduler:
    """
    Implements cosine annealing schedule for the EnhancedARPOptimizer.
    
    Args:
        optimizer (EnhancedARPOptimizer): Optimizer
        T_max (int): Maximum number of iterations
        eta_min (float): Minimum learning rate. Default: 0
        last_epoch (int): The index of the last epoch. Default: -1
    """
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step()
        
    def step(self):
        self.last_epoch += 1
        for i, param_group in enumerate(self.optimizer.param_groups):
            lr = self._compute_lr(self.base_lrs[i])
            param_group['lr'] = lr
            
    def _compute_lr(self, base_lr):
        # Cosine annealing formula
        return self.eta_min + (base_lr - self.eta_min) * (
            1 + math.cos(math.pi * self.last_epoch / self.T_max)
        ) / 2
