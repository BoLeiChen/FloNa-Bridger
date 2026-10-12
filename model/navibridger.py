"""Denoising diffusion bridge utilities for A* prior trajectory refinement.

This module adapts the VE DDBM formulation used by NaviBridger to VGCNav's
existing normalized 2-D delta trajectory representation.  The visual encoder
and ConditionalUnet1D remain owned by the surrounding Flona model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class NaviBridgeConfig:
    sigma_data: float = 0.5
    sigma_min: float = 0.002
    sigma_max: float = 10.0
    rho: float = 7.0
    num_inference_steps: int = 10
    guidance: float = 1.0
    clip_denoised: bool = True
    churn_step_ratio: float = 0.33
    use_astar_local_cond: bool = True
    astar_local_cond_dropout_prob: float = 0.0

    def __post_init__(self):
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be at least 1")
        if not 0.0 <= self.churn_step_ratio < 1.0:
            raise ValueError("churn_step_ratio must be in [0, 1)")
        if not 0.0 <= self.astar_local_cond_dropout_prob <= 1.0:
            raise ValueError("astar_local_cond_dropout_prob must be in [0, 1]")


def waypoints_to_normalized_deltas(waypoints: torch.Tensor, action_min, action_max) -> torch.Tensor:
    """Convert local A* waypoints to the exact normalized delta action space."""
    origin = torch.zeros_like(waypoints[:, :1])
    deltas = torch.cat((origin, waypoints), dim=1)[:, 1:] - torch.cat((origin, waypoints), dim=1)[:, :-1]
    action_min = torch.as_tensor(action_min, dtype=waypoints.dtype, device=waypoints.device)
    action_max = torch.as_tensor(action_max, dtype=waypoints.dtype, device=waypoints.device)
    return 2.0 * (deltas - action_min) / (action_max - action_min) - 1.0


class NaviBridgeDenoiser:
    """VE denoising diffusion bridge from an A* trajectory prior to an action trajectory.

    This is a compact, dependency-free adaptation of NaviBridger's KarrasDenoiser.
    The endpoint xT is the encoded A* path, rather than Gaussian noise.
    """

    def __init__(self, config: NaviBridgeConfig):
        self.config = config

    def _prepare_local_cond(
        self,
        local_cond: torch.Tensor,
        *,
        training: bool,
    ) -> Optional[torch.Tensor]:
        """Apply the A* local-condition switch and per-sample train dropout.

        The bridge endpoint ``xT`` is intentionally untouched: disabling or
        dropping this condition only removes the duplicate A* input to the
        ConditionalUnet1D local-condition branch.
        """
        if not self.config.use_astar_local_cond:
            return None
        dropout_prob = self.config.astar_local_cond_dropout_prob
        if training and dropout_prob > 0.0:
            drop = torch.rand(
                local_cond.shape[0], device=local_cond.device
            ) < dropout_prob
            drop = drop.reshape(
                (local_cond.shape[0],) + (1,) * (local_cond.ndim - 1)
            )
            local_cond = local_cond.masked_fill(drop, 0.0)
        return local_cond

    def _scalings(self, sigma: torch.Tensor):
        c = self.config
        sigma2_ratio = (sigma / c.sigma_max).square()
        variance = (
            sigma2_ratio.square() * c.sigma_data**2
            + (1.0 - sigma2_ratio).square() * c.sigma_data**2
            + sigma.square() * (1.0 - sigma2_ratio)
        )
        c_in = variance.rsqrt()
        c_skip = (1.0 - sigma2_ratio) * c.sigma_data**2 / variance
        c_out = (
            sigma2_ratio.square() * c.sigma_data**4
            + c.sigma_data**2 * sigma.square() * (1.0 - sigma2_ratio)
        ).sqrt() * c_in
        return c_skip, c_out, c_in

    def _loss_weights(self, sigma: torch.Tensor) -> torch.Tensor:
        """Return the default ``karras`` weighting used by NaviBridger.

        The bundled ``navibridge.yaml`` selects ``weight_schedule: karras``;
        its VE form is ``sigma^-2 + sigma_data^-2``.  The bridge-specific
        weighting is an optional upstream variant, not the default method.
        """
        return sigma.reciprocal().square() + self.config.sigma_data**-2

    @staticmethod
    def _expand(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return value[(...,) + (None,) * (target.ndim - value.ndim)]

    def denoise(
        self,
        model,
        x_t: torch.Tensor,
        sigma: torch.Tensor,
        *,
        local_cond: Optional[torch.Tensor],
        global_cond: torch.Tensor,
    ):
        c_skip, c_out, c_in = (self._expand(v, x_t) for v in self._scalings(sigma))
        # NaviBridger uses a log-sigma time embedding rescaled to the DDPM range.
        timestep = 250.0 * torch.log(sigma.clamp_min(1e-44))
        prediction = model(
            "noise_pred_net",
            sample=c_in * x_t,
            timestep=timestep,
            local_cond=local_cond,
            global_cond=global_cond,
        )
        return c_out * prediction + c_skip * x_t

    def training_loss(
        self,
        model,
        x0: torch.Tensor,
        xT: torch.Tensor,
        global_cond: torch.Tensor,
        *,
        local_cond: torch.Tensor,
    ) -> torch.Tensor:
        local_cond = self._prepare_local_cond(
            local_cond, training=bool(model.training)
        )
        batch_size = x0.shape[0]
        sigma = torch.empty(batch_size, device=x0.device, dtype=x0.dtype).uniform_(
            self.config.sigma_min, self.config.sigma_max
        )
        ratio = self._expand((sigma / self.config.sigma_max).square(), x0)
        std = self._expand(sigma * (1.0 - (sigma / self.config.sigma_max).square()).clamp_min(0).sqrt(), x0)
        x_t = ratio * xT + (1.0 - ratio) * x0 + std * torch.randn_like(x0)
        denoised = self.denoise(
            model, x_t, sigma, local_cond=local_cond, global_cond=global_cond
        )
        weights = self._expand(self._loss_weights(sigma), x0)
        return (weights * (denoised - x0).square()).mean()

    def _sigmas(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        c = self.config
        ramp = torch.linspace(0, 1, c.num_inference_steps, device=device, dtype=dtype)
        # Match karras_sample(..., sigma_max - 1e-4) from NaviBridger.  The
        # small offset keeps the bridge endpoint variance strictly positive.
        sigma_start = c.sigma_max - 1e-4
        sigmas = (sigma_start ** (1 / c.rho) + ramp * (c.sigma_min ** (1 / c.rho) - sigma_start ** (1 / c.rho))) ** c.rho
        return torch.cat((sigmas, torch.zeros(1, device=device, dtype=dtype)))

    @torch.no_grad()
    def sample(
        self,
        model,
        xT: torch.Tensor,
        global_cond: torch.Tensor,
        *,
        local_cond: torch.Tensor,
        energy_guidance=None,
    ) -> torch.Tensor:
        """Generate a trajectory with NaviBridger's churned Heun solver."""
        local_cond = self._prepare_local_cond(local_cond, training=False)
        x = xT.clone()
        sigmas = self._sigmas(x.device, x.dtype)
        batch = x.shape[0]
        for index in range(len(sigmas) - 1):
            current_sigma = sigmas[index]
            next_sigma = sigmas[index + 1]
            sigma_hat = current_sigma

            if self.config.churn_step_ratio > 0.0:
                current_sigma_batch = current_sigma.expand(batch)
                denoised = self.denoise(
                    model,
                    x,
                    current_sigma_batch,
                    local_cond=local_cond,
                    global_cond=global_cond,
                )
                if self.config.clip_denoised:
                    denoised = denoised.clamp(-1, 1)
                if energy_guidance is not None:
                    denoised = energy_guidance(denoised, current_sigma_batch, "churn")
                stochastic_derivative = self._bridge_derivative(
                    x, denoised, xT, current_sigma_batch, stochastic=True
                )
                churn_delta = (next_sigma - current_sigma) * self.config.churn_step_ratio
                x = (
                    x
                    + stochastic_derivative * churn_delta
                    + torch.randn_like(x)
                    * churn_delta.abs().sqrt()
                    * (2.0 * current_sigma).sqrt()
                )
                sigma_hat = current_sigma + churn_delta

            sigma_hat_batch = sigma_hat.expand(batch)
            denoised = self.denoise(
                model,
                x,
                sigma_hat_batch,
                local_cond=local_cond,
                global_cond=global_cond,
            )
            if self.config.clip_denoised:
                denoised = denoised.clamp(-1, 1)
            if energy_guidance is not None:
                denoised = energy_guidance(denoised, sigma_hat_batch, "predictor")
            derivative = self._bridge_derivative(x, denoised, xT, sigma_hat_batch)
            delta_sigma = next_sigma - sigma_hat
            euler_x = x + derivative * delta_sigma
            if next_sigma == 0:
                x = euler_x
                continue

            next_sigma_batch = next_sigma.expand(batch)
            next_denoised = self.denoise(
                model,
                euler_x,
                next_sigma_batch,
                local_cond=local_cond,
                global_cond=global_cond,
            )
            if self.config.clip_denoised:
                next_denoised = next_denoised.clamp(-1, 1)
            if energy_guidance is not None:
                next_denoised = energy_guidance(next_denoised, next_sigma_batch, "corrector")
            next_derivative = self._bridge_derivative(
                euler_x, next_denoised, xT, next_sigma_batch
            )
            x = x + 0.5 * (derivative + next_derivative) * delta_sigma
        return x.clamp(-1, 1) if self.config.clip_denoised else x

    def _bridge_derivative(
        self,
        x: torch.Tensor,
        denoised: torch.Tensor,
        xT: torch.Tensor,
        sigma: torch.Tensor,
        *,
        stochastic: bool = False,
    ) -> torch.Tensor:
        """Return NaviBridger's VE bridge probability-flow/SDE derivative."""
        grad_x0 = (denoised - x) / self._expand(sigma.square(), x)
        if stochastic:
            return -2.0 * self._expand(sigma, x) * grad_x0
        endpoint_variance = (self.config.sigma_max**2 - sigma.square()).clamp_min(1e-8)
        grad_xT = (xT - x) / self._expand(endpoint_variance, x)
        return -self._expand(sigma, x) * (grad_x0 - self.config.guidance * grad_xT)
