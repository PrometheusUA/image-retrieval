import torch 
import math


class CosineAnnealingLRWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, warmup_steps=0):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        self.last_epoch = last_epoch
        super(CosineAnnealingLRWithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / (self.T_max - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]
