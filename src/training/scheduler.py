"""
Learning rate scheduler with linear warmup + cosine decay.
"""

import math


class WarmupCosineScheduler:
    """
    Linear warmup for `warmup_steps`, then cosine decay from peak_lr to min_lr.
    """

    def __init__(
        self,
        optimizers: list,
        warmup_steps: int,
        total_steps: int,
        peak_lr: float,
        min_lr: float,
    ):
        self.optimizers = optimizers
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.current_step = 0

    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.peak_lr * (self.current_step / max(1, self.warmup_steps))
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            progress = min(progress, 1.0)
            return self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )

    def step(self):
        lr = self.get_lr()
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        self.current_step += 1
        return lr

    def state_dict(self):
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict):
        self.current_step = state_dict["current_step"]
