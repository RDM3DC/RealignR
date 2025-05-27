import torch

class UnifiedARPOptimizer(torch.optim.Optimizer):
    r"""
    Unified ARPOptimizer implements various ARP-style optimizers with flexible options.
    
    This optimizer combines features from:
    - Basic ARPOptimizer
    - MLARPOptimizer (multi-layer adaptation)
    - NARPOptimizer (noise threshold)
    - RobustMLARP (gradient clipping)
    - ARPAdamW (hybrid with AdamW)
    
    Args:
        params (iterable): iterable of parameters to optimize
        lr (float): step size applied to the update (default: 1e-3)
        alpha (float): conduction growth rate from |grad| (default: 1e-2)
        mu (float): conduction decay rate (default: 5.21e-3)
        weight_decay (float): weight decay (L2 penalty) (default: 0)
        mode (str): optimization mode - 'standard', 'multilayer', 'noise_adaptive', 'robust', 'hybrid_adamw'
        depth_factor (float): scaling factor for multi-layer mode (default: 0.9)
        noise_threshold (float): gradient threshold for noise adaptive mode (default: 0.05)
        grad_clip (float): gradient clipping value for robust mode (default: 1.0)
        clamp_G_min (float, optional): minimum value for G state
        clamp_G_max (float, optional): maximum value for G state
        betas (tuple of floats): AdamW momentum parameters (default: (0.9, 0.999))
        
    Example:
        >>> # Standard ARP
        >>> optimizer = UnifiedARPOptimizer(model.parameters(), lr=1e-3, mode='standard')
        >>> # Multi-layer ARP
        >>> optimizer = UnifiedARPOptimizer(model.parameters(), lr=1e-3, mode='multilayer', depth_factor=0.9)
    """

    def __init__(self, params,
                 lr=1e-3,
                 alpha=1e-2,
                 mu=5.21e-3,
                 weight_decay=0.0,
                 mode='standard',
                 depth_factor=0.9,
                 noise_threshold=0.05,
                 grad_clip=1.0,
                 clamp_G_min=None,
                 clamp_G_max=10.0,
                 betas=(0.9, 0.999)):
        
        self.hyperparameters = {
            'lr': lr,
            'alpha': alpha,
            'mu': mu,
            'weight_decay': weight_decay,
            'mode': mode,
            'depth_factor': depth_factor,
            'noise_threshold': noise_threshold,
            'grad_clip': grad_clip,
            'clamp_G_min': clamp_G_min,
            'clamp_G_max': clamp_G_max,
            'betas': betas
        }
        
        # Initialize AdamW if in hybrid mode
        self.hybrid_mode = (mode == 'hybrid_adamw')
        if self.hybrid_mode:
            self.adamw = torch.optim.AdamW(
                params, 
                lr=lr, 
                betas=betas, 
                weight_decay=weight_decay
            )
            
        super(UnifiedARPOptimizer, self).__init__(params, self.hyperparameters)
        
        print(f"Initialized UnifiedARPOptimizer in {mode} mode")

    @torch.no_grad()
    def step(self, closure=None):
        r"""
        Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        # For hybrid mode, first take an AdamW step
        if self.hybrid_mode:
            loss = self.adamw.step(closure)
                
        layer_index = 0
        for group in self.param_groups:
            mode = group['mode']
            lr = group['lr']
            mu = group['mu']
            grad_clip = group['grad_clip']
            weight_decay = group['weight_decay']
            clamp_min = group['clamp_G_min']
            clamp_max = group['clamp_G_max']
            noise_threshold = group['noise_threshold']
            
            # Adjust alpha based on layer depth for multi-layer mode
            alpha = group['alpha']
            if mode == 'multilayer':
                alpha = alpha * (group['depth_factor'] ** layer_index)
                layer_index += 1

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("UnifiedARPOptimizer does not support sparse gradients")

                # Apply weight decay if needed (except in hybrid mode, where AdamW handles it)
                if weight_decay != 0 and not self.hybrid_mode:
                    p.data.mul_(1 - lr * weight_decay)

                # Apply gradient clipping for robust mode
                if mode == 'robust':
                    grad = grad.clamp_(-grad_clip, grad_clip)

                # State initialization
                state = self.state[p]
                if 'G' not in state:
                    state['G'] = torch.zeros_like(p.data)

                G = state['G']
                I = grad.abs()

                # For noise adaptive mode, only update G where gradient exceeds threshold
                if mode == 'noise_adaptive':
                    mask = (I > noise_threshold).float()
                    G.mul_(1 - mu).add_(alpha * I * mask)
                else:
                    G.mul_(1 - mu).add_(alpha * I)

                # Optional clamping of G
                if clamp_min is not None or clamp_max is not None:
                    G.clamp_(min=clamp_min, max=clamp_max)

                # For hybrid_adamw mode, apply adaptive scaling
                if mode == 'hybrid_adamw':
                    adaptive_scale = G / (G.mean() + 1e-8)
                    p.data.mul_(adaptive_scale)
                else:
                    # Standard parameter update for other modes
                    update = G * grad.sign()
                    p.data.add_(update, alpha=-lr)

        return loss
        
    def get_mode(self):
        """Returns the current optimization mode."""
        return self.param_groups[0]['mode']
        
    def set_mode(self, mode):
        """Change the optimization mode during training."""
        valid_modes = ['standard', 'multilayer', 'noise_adaptive', 'robust', 'hybrid_adamw']
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}")
            
        for group in self.param_groups:
            group['mode'] = mode
            
        # Initialize AdamW if switching to hybrid mode
        if mode == 'hybrid_adamw' and not self.hybrid_mode:
            self.adamw = torch.optim.AdamW(
                self.param_groups,
                lr=self.param_groups[0]['lr'],
                betas=self.param_groups[0]['betas'],
                weight_decay=self.param_groups[0]['weight_decay']
            )
            self.hybrid_mode = True
            
        print(f"Switched to {mode} mode")
    
    def set_hyperparameters(self, **kwargs):
        """Update hyperparameters dynamically."""
        for key, value in kwargs.items():
            if key in self.hyperparameters:
                self.hyperparameters[key] = value
                for group in self.param_groups:
                    group[key] = value
            else:
                raise ValueError(f"Invalid hyperparameter: {key}")
        print(f"Updated hyperparameters: {kwargs}")
