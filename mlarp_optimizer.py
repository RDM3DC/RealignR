import torch

class MLARPOptimizer(torch.optim.Optimizer):
    def __init__(self, params, alpha=0.01, mu=0.00521, depth_factor=0.9):
        defaults = dict(alpha=alpha, mu=mu, depth_factor=depth_factor)
        super(MLARPOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        
        layer_index = 0
        for group in self.param_groups:
            alpha = group['alpha'] * (group['depth_factor'] ** layer_index)
            mu = group['mu']
            layer_index += 1
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                if 'G' not in state:
                    state['G'] = torch.zeros_like(p)  # Initialize G to zeros for stability

                G = state['G']
                I = grad.abs()

                # Update G: G += alpha * I - mu * G
                G += alpha * I - mu * G

                # Optional clamping of G to avoid instability
                G.clamp_(min=0, max=10)  # Adjust max value as needed

                # Parameter update with clamping
                update = -G * grad.sign()
                p.add_(update.clamp_(-1, 1))  # Clamp updates to avoid large steps

        return loss
