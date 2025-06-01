import torch
import math

class AGSGDOptimizer(torch.optim.Optimizer):
    def __init__(self, params, s=1.0, d=0.5, iter_freq=5):
        defaults = dict(s=s, d=d, iter_freq=iter_freq, step_count=0)
        super().__init__(params, defaults)
        self.prev_grads = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        for group in self.param_groups:
            s, d, iter_freq = group['s'], group['d'], group['iter_freq']
            group['step_count'] += 1
            step_count = group['step_count']

            new_grads, param_pairs = [], []
            for p in group['params']:
                if p.grad is None: continue
                new_grads.append(p.grad.detach().clone())
                param_pairs.append(p)

            if self.prev_grads is None:
                self.prev_grads = new_grads
                for p, g in zip(param_pairs, new_grads):
                    p.add_(g, alpha=-s)
                return loss

            angle_vals = []
            for pg, cg in zip(self.prev_grads, new_grads):
                dot = (pg * cg).sum()
                angle_vals.append(dot / (pg.norm() * cg.norm() + 1e-8))
            angle = torch.acos(torch.mean(torch.tensor(angle_vals)).clamp(-1, 1)) / math.pi

            coef_pg = -s * (2 * angle - 1)
            coef_cg = s - coef_pg

            for p, pg, cg in zip(param_pairs, self.prev_grads, new_grads):
                delta = coef_pg * pg + coef_cg * cg
                p.add_(delta, alpha=-1)

            if step_count % iter_freq == 0:
                group['s'] *= d

            self.prev_grads = new_grads

        return loss
