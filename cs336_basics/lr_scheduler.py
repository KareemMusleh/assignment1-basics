from math import cos, pi

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def cosine_lr(it: int, max_lr: float, min_lr: float, warmup_iters: int, cycle_iters: int) -> float:
    if not 0 <= warmup_iters < cycle_iters:
        raise ValueError("Expected 0 <= warmup_iters < cycle_iters")
    if not 0 <= min_lr <= max_lr:
        raise ValueError("Expected 0 <= min_lr <= max_lr")
    if it < warmup_iters:
        return max_lr * (it / warmup_iters)
    if it <= cycle_iters:
        angle = pi * ((it - warmup_iters) / (cycle_iters - warmup_iters))
        return min_lr + 0.5 * (1 + cos(angle)) * (max_lr - min_lr)
    return min_lr


class LRCosineScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, max_lr: float, min_lr: float, warmup_iters: int, cycle_iters: int):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters
        self.cycle_iters = cycle_iters
        # Initializes the optimizer's LR for iteration zero.
        super().__init__(optimizer)

    def lr_at(self, it: int) -> float:
        return cosine_lr(it, self.max_lr, self.min_lr, self.warmup_iters, self.cycle_iters)

    def get_lr(self) -> list[float]:
        lr = self.lr_at(self.last_epoch)
        return [lr for _ in self.optimizer.param_groups]
