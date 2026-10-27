import torch
from collections.abc import Callable

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas = (0.9, 0.999), weight_decay = 0.01, eps = 1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr=}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                # step count
                t = state.get('t', 1)
                # first momentum
                m = state.get('m', p.data.new_zeros(p.data.shape))
                # second momentum
                v = state.get('v', p.data.new_zeros(p.data.shape))

                grad = p.grad.data
                adjusted_lr = lr * (1 - b2 ** t) ** 0.5 / (1 - b1 ** t)
                p.data *= 1 - lr * weight_decay

                m = b1 * m + (1 - b1) * grad
                v = b2 * v + (1 - b2) * grad ** 2
                p.data -= adjusted_lr * m / (v ** 0.5 + eps)

                state['t'] = t + 1
                state['m'] = m
                state['v'] = v
        return loss
