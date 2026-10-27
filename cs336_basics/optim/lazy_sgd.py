import torch
from collections.abc import Callable

class DecayingSGD(torch.optim.Optimizer):
    def __init__(self, params, lr):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get('t', 0)
                p.data -= lr / (t + 1) ** 0.5 * p.grad.data
                state['t'] = t + 1
        return loss
