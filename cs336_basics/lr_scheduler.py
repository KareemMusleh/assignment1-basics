from math import cos, pi

class LRCosineScheduler:
    def __init__(self, max_lr: float, min_lr: float, warmup_iters: int, cycle_iters: int):
        if warmup_iters > cycle_iters:
            raise ValueError("warmup_iters must be less than cycle_iters")
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters
        self.cycle_iters = cycle_iters
    def __call__(self, it: int):
        if it < self.warmup_iters:
            return self.max_lr * (it / self.warmup_iters)
        if self.warmup_iters <= it <= self.cycle_iters:
            angle = pi * ((it - self.warmup_iters) / (self.cycle_iters - self.warmup_iters))
            return self.min_lr + 0.5 * (1 + cos(angle)) * (self.max_lr - self.min_lr)
        return self.min_lr
