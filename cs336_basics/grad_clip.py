import torch
from collections.abc import Iterator

def clip_grad(params: Iterator[torch.nn.Parameter], max_l2_norm: float):
    if not isinstance(params, (list, tuple)):
        params = [param for param in params]

    eps = 1e-6
    l2_norm = 0.0
    for param in params:
        if param.grad is None:
            continue
        l2_norm += (param.grad.to(torch.float32) ** 2).sum()
    l2_norm **= 0.5
    if l2_norm < max_l2_norm:
        return
    for param in params:
        if param.grad is None:
            continue
        param.grad *= max_l2_norm / (l2_norm + eps)
