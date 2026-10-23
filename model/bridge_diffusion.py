"""VE denoising diffusion bridge, adapted from VGCNav (see THIRD_PARTY_BRIDGE.md).

All trajectories are local metric XY, and include the current origin at index 0.
Only normalized step-unit deltas enter the diffusion process. action_step_m is
checkpoint configuration: 1.0 preserves legacy meter encoding; new runs use 0.1.
Local/global conditions and gate geometry remain in their original metric units.
"""
from dataclasses import dataclass
import math

import torch


ACTION_MIN = (-2.5, -4.0)
ACTION_MAX = (5.0, 4.0)


def _check_action_step(action_step_m):
    if not math.isfinite(action_step_m) or action_step_m <= 0:
        raise ValueError("action_step_m must be finite and positive")


def encode_trajectory(path, action_step_m=1.0):
    """Meters -> normalized step deltas; default preserves old checkpoints."""
    _check_action_step(action_step_m)
    path = path.float() / action_step_m
    delta = torch.diff(path, dim=1, prepend=torch.zeros_like(path[:, :1]))
    lo, hi = path.new_tensor(ACTION_MIN), path.new_tensor(ACTION_MAX)
    return 2 * (delta - lo) / (hi - lo) - 1


def decode_trajectory(value, action_step_m=1.0):
    """Normalized step deltas -> meters with the known current origin fixed."""
    _check_action_step(action_step_m)
    lo, hi = value.new_tensor(ACTION_MIN), value.new_tensor(ACTION_MAX)
    delta = (value + 1) * 0.5 * (hi - lo) + lo
    # Both original FloVerse labels and our A* priors begin at the current pose.
    delta = torch.cat([torch.zeros_like(delta[:, :1]), delta[:, 1:]], dim=1)
    return delta.cumsum(dim=1) * action_step_m


@dataclass(frozen=True)
class BridgeConfig:
    sigma_data: float = 0.5
    sigma_min: float = 0.002
    sigma_max: float = 10.0
    rho: float = 7.0
    steps: int = 10
    guidance: float = 1.0
    churn: float = 0.33
    # Missing in legacy saved configs: keep their meter encoding unchanged.
    action_step_m: float = 1.0

    def __post_init__(self):
        _check_action_step(self.action_step_m)
        if not 0 < self.sigma_min < self.sigma_max or self.sigma_data <= 0:
            raise ValueError('Invalid bridge noise scales')
        if self.steps < 2 or self.rho <= 0 or not 0 <= self.churn < 1:
            raise ValueError('Bridge requires steps >= 2, rho > 0, and 0 <= churn < 1')


class DiffusionBridge:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def expand(x):
        return x[:, None, None]

    def denoise(self, net, x, sigma, local, global_cond):
        c = self.config
        ratio = (sigma / c.sigma_max).square()
        variance = (ratio.square() + (1 - ratio).square()) * c.sigma_data**2
        variance = variance + sigma.square() * (1 - ratio)
        c_in = variance.rsqrt()
        c_skip = (1 - ratio) * c.sigma_data**2 / variance
        c_out = (ratio.square() * c.sigma_data**4
                 + c.sigma_data**2 * sigma.square() * (1 - ratio)).sqrt() * c_in
        prediction = net(sample=self.expand(c_in) * x,
                         timestep=250 * sigma.log(), local_cond=local,
                         global_cond=global_cond).float()
        return self.expand(c_out) * prediction + self.expand(c_skip) * x

    def loss(self, net, target, prior, local, global_cond):
        x0 = encode_trajectory(target, self.config.action_step_m)
        xT = encode_trajectory(prior, self.config.action_step_m)
        c = self.config
        sigma = torch.empty(len(x0), device=x0.device).uniform_(c.sigma_min, c.sigma_max)
        ratio = self.expand((sigma / c.sigma_max).square())
        std = self.expand(sigma) * (1 - ratio).clamp_min(0).sqrt()
        xt = (1 - ratio) * x0 + ratio * xT + std * torch.randn_like(x0)
        clean = self.denoise(net, xt, sigma, local, global_cond)
        weight = self.expand(sigma.reciprocal().square() + c.sigma_data**-2)
        return (weight * (clean - x0).square()).mean(), decode_trajectory(clean.clamp(-1, 1), c.action_step_m)

    def derivative(self, x, clean, xT, sigma, stochastic=False):
        grad0 = (clean - x) / self.expand(sigma.square())
        if stochastic:
            return -2 * self.expand(sigma) * grad0
        endpoint_variance = (self.config.sigma_max**2 - sigma.square()).clamp_min(1e-8)
        gradT = (xT - x) / self.expand(endpoint_variance)
        return -self.expand(sigma) * (grad0 - self.config.guidance * gradT)

    @torch.no_grad()
    def sample(self, net, prior, local, global_cond, energy_guidance=None):
        c = self.config
        xT = encode_trajectory(prior, c.action_step_m)
        x = xT.clone()
        ramp = torch.linspace(0, 1, c.steps, device=x.device)
        hi, lo = (c.sigma_max - 1e-4)**(1 / c.rho), c.sigma_min**(1 / c.rho)
        sigmas = torch.cat([(hi + ramp * (lo - hi))**c.rho, ramp.new_zeros(1)])
        for index in range(c.steps):
            sigma, next_sigma = sigmas[index], sigmas[index + 1]
            sigma_hat = sigma
            if c.churn:
                clean = self.denoise(net, x, sigma.expand(len(x)), local, global_cond).clamp(-1, 1)
                if energy_guidance is not None:
                    clean = energy_guidance(clean, sigma, "churn")
                ds = (next_sigma - sigma) * c.churn
                x = x + self.derivative(x, clean, xT, sigma.expand(len(x)), True) * ds
                x = x + torch.randn_like(x) * ds.abs().sqrt() * (2 * sigma).sqrt()
                sigma_hat = sigma + ds
            clean = self.denoise(net, x, sigma_hat.expand(len(x)), local, global_cond).clamp(-1, 1)
            if energy_guidance is not None:
                clean = energy_guidance(clean, sigma_hat, "predictor")
            derivative = self.derivative(x, clean, xT, sigma_hat.expand(len(x)))
            ds = next_sigma - sigma_hat
            euler = x + derivative * ds
            if index == c.steps - 1:
                x = euler
            else:
                clean_next = self.denoise(net, euler, next_sigma.expand(len(x)), local, global_cond).clamp(-1, 1)
                if energy_guidance is not None:
                    clean_next = energy_guidance(clean_next, next_sigma, "corrector")
                derivative_next = self.derivative(euler, clean_next, xT, next_sigma.expand(len(x)))
                x = x + (derivative + derivative_next) * (0.5 * ds)
        return decode_trajectory(x.clamp(-1, 1), c.action_step_m)
