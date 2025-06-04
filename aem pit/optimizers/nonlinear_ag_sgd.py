import torch
import math
from .ag_sgd import AGSGDOptimizer

class NonLinearAGSGDOptimizer(AGSGDOptimizer):
    def __init__(self, params, s=1.0, d=0.5, iter_freq=5, k=5.0, min_s=1e-4, max_s=5.0):
        if not isinstance(params, list):
            params = [{'params': params}]

        defaults = dict(s=s, d=d, iter_freq=iter_freq, k=k, min_s=min_s, max_s=max_s)
        for param_group in params:
            for key in defaults:
                if key not in param_group:
                    param_group[key] = defaults[key]

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        for group in self.param_groups:
            s, d, iter_freq, k = group['s'], group['d'], group['iter_freq'], group['k']
            min_s, max_s = group['min_s'], group['max_s']

            if 'step_count' not in group:
                group['step_count'] = 0
            group['step_count'] += 1
            step_count = group['step_count']

            new_grads, param_pairs = [], []
            for p in group['params']:
                if p.grad is None:
                    continue
                new_grads.append(p.grad.detach().clone())
                param_pairs.append(p)

            if self.prev_grads is None:
                if not hasattr(self, '_prev_grads'):
                    self._prev_grads = {}
                self._prev_grads[id(group)] = new_grads
                for p, g in zip(param_pairs, new_grads):
                    p.add_(g, alpha=-s)
                return loss

            prev_grads = self._prev_grads.get(id(group))
            if prev_grads is None:
                self._prev_grads[id(group)] = new_grads
                for p, g in zip(param_pairs, new_grads):
                    p.add_(g, alpha=-s)
                return loss

            if step_count < 5:
                for p, g in zip(param_pairs, new_grads):
                    p.add_(g, alpha=-s * 0.5)
                self._prev_grads[id(group)] = new_grads
                return loss

            # Compute cosine similarities with stability checks
            angle_vals = []
            for pg, cg in zip(prev_grads, new_grads):
                pg_norm = pg.norm()
                cg_norm = cg.norm()
                if pg_norm < 1e-8 or cg_norm < 1e-8:
                    angle_vals.append(torch.tensor(1.0, device=pg.device))  # treat as aligned
                else:
                    dot = (pg * cg).sum()
                    cos_sim = dot / (pg_norm * cg_norm + 1e-8)
                    angle_vals.append(cos_sim.clamp(-1.0, 1.0))

            mean_cos_raw = torch.mean(torch.stack(angle_vals))
            mean_cos = mean_cos_raw.clamp(-1, 1)
            angle_raw = torch.acos(mean_cos) / math.pi

            # Exponential moving average smoothing of angle
            angle_ema = group.get('angle_ema', angle_raw)
            angle = 0.9 * angle_ema + 0.1 * angle_raw
            group['angle_ema'] = angle

            coef_pg = s * (1 + (-2 / (1 + torch.exp(-k * (2 * angle - 1)))))
            coef_cg = s - coef_pg

            for p, pg, cg in zip(param_pairs, prev_grads, new_grads):
                delta = coef_pg * pg + coef_cg * cg
                p.add_(delta, alpha=-1)

            if step_count % iter_freq == 0:
                # Adapt step size s based on alignment
                if mean_cos > 0.9:  # well-aligned gradients, increase s slightly
                    group['s'] = min(max_s, s * 1.01)
                else:  # less alignment, decay s exponentially with margin
                    decay_factor = torch.exp(-5 * (1 - mean_cos))
                    group['s'] = max(min_s, s * decay_factor.item())

            self._prev_grads[id(group)] = new_grads

        return loss