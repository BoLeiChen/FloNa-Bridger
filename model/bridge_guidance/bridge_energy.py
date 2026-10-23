"""Bounded clean-prediction guidance for the VE diffusion bridge (not DDPM)."""
import math
import torch
from model.bridge_diffusion import decode_trajectory
from .navigation_v2 import NavigationObjectiveV2


class BridgeEnergyGuidance:
    def __init__(self, local_map, goal, config, action_step_m):
        self.objective = NavigationObjectiveV2(local_map, config['costs'], goal)
        self.scale = float(config.get('scale', 1.0))
        self.sigma_max = float(config.get('sigma_max', .5))
        self.max_update_m = float(config.get('max_update_m', .025))
        self.action_step_m = action_step_m
        self.records = []
        if not all(math.isfinite(x) for x in [self.scale, self.sigma_max, self.max_update_m]):
            raise ValueError('Nonfinite guidance parameter')
        if self.scale < 0 or self.sigma_max <= 0 or self.max_update_m <= 0:
            raise ValueError('Invalid guidance parameter')

    def __call__(self, clean, sigma, stage):
        # D -> D - lambda sigma^2 grad J gives grad0 -> grad0 - lambda grad J.
        # The original endpoint score, churn and Heun integration remain intact.
        # This is a clean-space approximation, not an exact posterior guarantee.
        if self.scale == 0 or float(sigma) > self.sigma_max:
            return clean
        with torch.enable_grad():
            x = clean.detach().requires_grad_(True)
            path = decode_trajectory(x, self.action_step_m)
            self.objective.bind_reference(path)
            cost = self.objective(path)
            grad, = torch.autograd.grad(cost.sum(), x)
        if not torch.isfinite(grad).all():
            raise FloatingPointError('Nonfinite energy gradient')
        change = -self.scale * float(sigma)**2 * grad.detach()
        change[:, 0] = 0  # fixed current-origin token
        reference = path.detach()
        with torch.no_grad():
            proposal = (clean + change).clamp(-1, 1)
            displacement = (decode_trajectory(proposal, self.action_step_m)-reference).norm(dim=-1).amax(-1)
            change *= (self.max_update_m / displacement.clamp_min(1e-12)).clamp(max=1)[:, None, None]
            accepted = torch.zeros(len(clean), dtype=torch.bool, device=clean.device)
            output = clean.clone()
            for _ in range(7):
                proposal = (clean + change).clamp(-1, 1)
                candidate_path = decode_trajectory(proposal, self.action_step_m)
                displacement = (candidate_path-reference).norm(dim=-1).amax(-1)
                good = (self.objective.accept_update(reference, candidate_path)
                        & (displacement <= self.max_update_m + 1e-7) & ~accepted)
                output[good] = proposal[good]
                accepted |= good
                if bool(accepted.all()):
                    break
                change *= .5
            after = decode_trajectory(output, self.action_step_m)
            before_terms = self.objective.terms(reference)
            after_terms = self.objective.terms(after)
            self.records.append(dict(sigma=float(sigma), stage=stage,
                accepted=accepted.cpu().tolist(), gradient_norm=grad.flatten(1).norm(dim=-1).cpu().tolist(),
                max_shift_m=(after-reference).norm(dim=-1).amax(-1).cpu().tolist(),
                before={k:v.cpu().tolist() for k,v in before_terms.items()},
                after={k:v.cpu().tolist() for k,v in after_terms.items()}))
            return output
