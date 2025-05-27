import torch

class RobustMLARP(torch.optim.Optimizer):
    def __init__(self, params, alpha=0.01, mu=0.00521, grad_clip=1.0):
        defaults = dict(alpha=alpha, mu=mu, grad_clip=grad_clip)
        super(RobustMLARP, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            alpha = group['alpha']
            mu = group['mu']
            grad_clip = group['grad_clip']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.clamp_(-grad_clip, grad_clip)
                state = self.state[p]

                if 'G' not in state:
                    state['G'] = torch.ones_like(p)

                G = state['G']
                I = grad.abs()

                G += alpha * I - mu * G
                update = -G * grad.sign()

                p.add_(update)

        return loss
