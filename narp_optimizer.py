import torch

class NARPOptimizer(torch.optim.Optimizer):
    def __init__(self, params, alpha=0.01, mu=0.00521, noise_threshold=0.1):
        defaults = dict(alpha=alpha, mu=mu, noise_threshold=noise_threshold)
        super(NARPOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            alpha = group['alpha']
            mu = group['mu']
            threshold = group['noise_threshold']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if 'G' not in state:
                    state['G'] = torch.ones_like(p)

                G = state['G']
                I = grad.abs()

                mask = (I > threshold).float()
                G += alpha * I * mask - mu * G
                update = -G * grad.sign()

                p.add_(update)

        return loss
