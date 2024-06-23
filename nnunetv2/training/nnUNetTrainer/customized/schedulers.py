from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

class DALRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, lr_alpha: float = 10., exponent: float = 1., current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.lr_alpha = lr_alpha
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr / (1 + self.lr_alpha * current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

class GRLAlphaScheduler():
    def __init__(self, model, max_steps: int, alpha_max: float = 3, p1: float = 0.2, p2: float = 0.7, current_step: int = None):
        self.model = model
        self.max_steps = max_steps
        self.alpha_max = alpha_max
        self.p1 = p1
        self.p2 = p2
        self.ctr = 0
        self.current_step = current_step if current_step is not None else -1
        # super().__init__(model, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_alpha = self.alpha_max * np.clip((current_step / self.max_steps - self.p1)/(self.p2 - self.p1), 0., 1.)
        self.model.domain_classifier.set_alpha(new_alpha)
