import math

import numpy as np
from torch.optim.lr_scheduler import LRScheduler


class CosineHashCurriculumScheduler(LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0.0, frac_start=0.0, frac_end=1.0, n_restarts=1, last_epoch=-1,
                 verbose=False):
        self.T_max = T_max
        self.frac_start = frac_start
        self.frac_end = frac_end
        self.n_restarts = n_restarts
        self.eta_min = eta_min
        self.n_grids = len(optimizer.param_groups) - 1
        super(CosineHashCurriculumScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [(self.eta_min + (base_lr - self.eta_min) * (1 + math.cos((math.pi * ((self.n_restarts * self.last_epoch)
                                                                                     % self.T_max)) / self.T_max)) / 2)
                * self.get_curriculum_value(idx)
                for base_lr, group, idx in zip(self.base_lrs, self.optimizer.param_groups, range(len(self.base_lrs)))]

    def get_curriculum_value(self, param_group_id):
        if param_group_id < 2:
            return 1
        else:
            a = (self.n_grids - 1) * ((self.last_epoch / self.T_max - self.frac_start) /
                                      (self.frac_end - self.frac_start))
            if a <= param_group_id - 2:
                return 0
            elif a - (param_group_id - 2) < 1:
                return (1 - math.cos((a - (param_group_id - 2)) * math.pi)) / 2
            else:
                return 1


class GSScheduler(LRScheduler):
    def __init__(self, optimizer, final_lr, n_steps=30000, names_to_schedule=None, last_epoch=-1, verbose=False):
        if names_to_schedule is None:
            names_to_schedule = ["means"]
        self.names_to_schedule = names_to_schedule
        self.final_lr = final_lr
        self.n_steps = n_steps
        super(GSScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch > self.n_steps:
            return self.final_lr

        new_lrs = []
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):

            if (self.last_epoch < 0 or (base_lr <= 0.0 and self.final_lr <= 0.0)
                    or group["name"] not in self.names_to_schedule):
                # Disable this parameter
                new_lrs.append(base_lr)
                continue

            t = np.clip(self.last_epoch / self.n_steps, 0, 1)
            new_lr = base_lr * (self.final_lr / base_lr) ** t

            new_lrs.append(new_lr)

        return new_lrs


class ExpCosScheduler(LRScheduler):
    def __init__(self, optimizer, final_lr, n_steps=30000, cos_after_n_steps=7500, n_warmup_steps=0,
                 names_to_schedule=None, last_epoch=-1, verbose=False):
        self.names_to_schedule = names_to_schedule
        self.final_lr = final_lr
        self.n_steps = n_steps
        self.cos_after_n_steps = cos_after_n_steps
        self.n_warmup_steps = n_warmup_steps
        super(ExpCosScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch > self.n_steps:
            return [self.final_lr] * len(self.base_lrs)

        new_lrs = []
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            if (self.last_epoch < 0 or (base_lr <= 0.0 and self.final_lr <= 0.0)
                    or (self.names_to_schedule is not None and group["name"] not in self.names_to_schedule)):
                # Disable this parameter
                new_lrs.append(base_lr)
                continue

            if self.last_epoch < self.cos_after_n_steps:
                t = np.clip(self.last_epoch / self.n_steps, 0, 1)
                new_lr = base_lr * (self.final_lr / base_lr) ** t
            else:
                t_step = np.clip(self.cos_after_n_steps / self.n_steps, 0, 1)
                step_lr = base_lr * (self.final_lr / base_lr) ** t_step
                t_2 = np.clip((self.last_epoch - self.cos_after_n_steps) /
                              (self.n_steps - self.cos_after_n_steps), 0, 1)
                new_lr = self.final_lr + 1/2 * (step_lr - self.final_lr) * (1 + np.cos(t_2 * np.pi))

            if self.last_epoch < self.n_warmup_steps:
                new_lr = new_lr * (1 - 1/2 * (1 + np.cos(self.last_epoch / self.n_warmup_steps * np.pi)))
            new_lrs.append(new_lr)

        return new_lrs
