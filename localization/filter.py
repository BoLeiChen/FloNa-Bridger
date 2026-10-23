"""Grid-based SE(2) Bayes filter for CF2Loc diffusion measurements."""

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F

from .geometry import pose_errors, wrap_angle


@dataclass
class LocalizationEstimate:
    pose: np.ndarray
    confidence: float
    candidate_pose: np.ndarray
    candidate_confidence: float
    observation_accepted: bool
    innovation_position_m: float = 0.0
    innovation_yaw_deg: float = 0.0
    stage1_pose: np.ndarray = None
    candidate_count: int = 0


def normalized_entropy_confidence(probabilities, support_size=None):
    """Measure concentration relative to the complete valid SE(2) support."""
    values = torch.as_tensor(probabilities).detach().flatten()
    values = values[values > 0]
    if values.numel() <= 1:
        return 1.0
    values = values / values.sum().clamp_min(1e-30)
    entropy = -(values * torch.log(values.clamp_min(1e-30))).sum()
    support_size = int(support_size or values.numel())
    if support_size <= 1:
        return 1.0
    confidence = 1.0 - entropy / math.log(support_size)
    return float(torch.clamp(confidence, 0.0, 1.0).item())


def _gaussian_kernel1d(sigma, device, dtype, truncate=3.0):
    sigma = float(sigma)
    if sigma <= 1e-6:
        return torch.ones(1, device=device, dtype=dtype)
    radius = max(1, int(math.ceil(truncate * sigma)))
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum().clamp_min(1e-30)


class SequentialPoseFilter:
    """Fuse all diffusion particles in a multi-modal, grid-based SE(2) belief."""

    def __init__(self, config, device="cpu"):
        self.config = config
        self.device = torch.device(device)
        self.resolution = float(config["belief_resolution_m"])
        self.yaw_bins = int(config["belief_yaw_bins"])
        self.yaw_resolution = 2.0 * math.pi / self.yaw_bins
        self.belief = None
        self.free_mask = None
        self.uniform = None
        self.origin_xy = None
        self.map_size_m = None
        self.tracking_ready = False

    def reset(self, map_size_m=None, world_origin_xy=None, free_mask=None):
        self.belief = None
        self.tracking_ready = False
        self.free_mask = None
        self.uniform = None
        self.origin_xy = None
        self.map_size_m = None
        if map_size_m is not None:
            self.configure_grid(map_size_m, world_origin_xy, free_mask)

    def configure_grid(self, map_size_m, world_origin_xy, free_mask=None):
        size = np.asarray(map_size_m, dtype=np.float64)
        origin = np.asarray(world_origin_xy, dtype=np.float64)
        if size.shape != (2,) or origin.shape != (2,) or np.any(size <= 0):
            raise ValueError("map_size_m and world_origin_xy must contain two valid values")
        width = int(math.ceil(size[0] / self.resolution))
        height = int(math.ceil(size[1] / self.resolution))
        self.origin_xy = origin
        self.map_size_m = size

        if free_mask is None:
            spatial_mask = torch.ones(
                (height, width), dtype=torch.bool, device=self.device
            )
        else:
            mask = torch.as_tensor(
                np.asarray(free_mask), dtype=torch.float32, device=self.device
            )
            if mask.ndim != 2:
                raise ValueError("free_mask must be a two-dimensional array")
            spatial_mask = F.interpolate(
                mask[None, None], size=(height, width), mode="nearest"
            )[0, 0] > 0.5
            if int(spatial_mask.sum()) < max(1, spatial_mask.numel() // 100):
                raise ValueError("free_mask contains too few traversable cells")

        self.free_mask = spatial_mask[None].expand(
            self.yaw_bins, -1, -1
        ).clone()
        self.uniform = self.free_mask.to(torch.float32)
        self.uniform /= self.uniform.sum().clamp_min(1.0)

    def _auto_configure(self, candidates):
        padding = float(self.config["belief_auto_padding_m"])
        low = candidates[:, :2].min(axis=0) - padding
        high = candidates[:, :2].max(axis=0) + padding
        self.configure_grid(high - low, low)

    def begin_tracking(self):
        if self.belief is None:
            raise RuntimeError("pose filter must observe at least one frame before tracking")
        self.tracking_ready = True

    def _smooth(self, probability, position_sigma_m, yaw_sigma_deg):
        position_kernel = _gaussian_kernel1d(
            float(position_sigma_m) / self.resolution,
            probability.device,
            probability.dtype,
        )
        if position_kernel.numel() > 1:
            radius = position_kernel.numel() // 2
            planes = probability[:, None]
            planes = F.conv2d(
                F.pad(planes, (radius, radius, 0, 0)),
                position_kernel[None, None, None, :],
            )
            planes = F.conv2d(
                F.pad(planes, (0, 0, radius, radius)),
                position_kernel[None, None, :, None],
            )
            probability = planes[:, 0]

        yaw_kernel = _gaussian_kernel1d(
            math.radians(float(yaw_sigma_deg)) / self.yaw_resolution,
            probability.device,
            probability.dtype,
        )
        if yaw_kernel.numel() > 1:
            radius = yaw_kernel.numel() // 2
            rows = probability.permute(1, 2, 0).reshape(-1, 1, self.yaw_bins)
            rows = F.conv1d(
                F.pad(rows, (radius, radius), mode="circular"),
                yaw_kernel[None, None],
            )
            probability = rows.reshape(
                probability.shape[1], probability.shape[2], self.yaw_bins
            ).permute(2, 0, 1)

        probability = probability * self.free_mask
        total = probability.sum()
        if not torch.isfinite(total) or float(total) <= 0:
            raise RuntimeError("SE(2) smoothing eliminated the complete belief")
        return probability / total

    def _measurement_belief(self, candidates):
        histogram = torch.zeros_like(self.uniform)
        poses = torch.as_tensor(
            candidates, dtype=torch.float32, device=self.device
        )
        gx = torch.floor(
            (poses[:, 0] - self.origin_xy[0]) / self.resolution
        ).long()
        gy = torch.floor(
            (poses[:, 1] - self.origin_xy[1]) / self.resolution
        ).long()
        gt = torch.remainder(
            torch.round(poses[:, 2] / self.yaw_resolution).long(),
            self.yaw_bins,
        )
        inside = (
            (gx >= 0)
            & (gx < histogram.shape[2])
            & (gy >= 0)
            & (gy < histogram.shape[1])
        )
        if not bool(inside.any()):
            raise RuntimeError(
                "all CF2Loc particles fell outside the SE(2) belief grid"
            )
        safe_x = gx.clamp(0, histogram.shape[2] - 1)
        safe_y = gy.clamp(0, histogram.shape[1] - 1)
        valid = inside & self.free_mask[gt, safe_y, safe_x]
        if not bool(valid.any()):
            raise RuntimeError(
                "all CF2Loc particles fell outside traversable map cells"
            )
        flat = (
            (gt[valid] * histogram.shape[1] + gy[valid])
            * histogram.shape[2]
            + gx[valid]
        )
        histogram.view(-1).scatter_add_(
            0, flat, torch.ones_like(flat, dtype=histogram.dtype)
        )
        return self._smooth(
            histogram,
            self.config["measurement_sigma_position_m"],
            self.config["measurement_sigma_yaw_deg"],
        )

    def _propagate(self, probability, odometry):
        if odometry is None:
            return probability
        odometry = torch.as_tensor(
            odometry, dtype=torch.float32, device=self.device
        )
        yaw_idx, row, col = torch.nonzero(probability > 0, as_tuple=True)
        weights = probability[yaw_idx, row, col]
        yaw = yaw_idx.to(torch.float32) * self.yaw_resolution
        global_dx = (
            torch.cos(yaw) * odometry[0] - torch.sin(yaw) * odometry[1]
        )
        global_dy = (
            torch.sin(yaw) * odometry[0] + torch.cos(yaw) * odometry[1]
        )
        x = col.to(torch.float32) + global_dx / self.resolution
        y = row.to(torch.float32) + global_dy / self.resolution
        t = yaw_idx.to(torch.float32) + odometry[2] / self.yaw_resolution
        x0, y0, t0 = torch.floor(x).long(), torch.floor(y).long(), torch.floor(t).long()
        fx, fy, ft = x - x0, y - y0, t - t0
        moved = torch.zeros(
            probability.shape,
            dtype=probability.dtype,
            device=probability.device,
        )

        for dt in (0, 1):
            ti = torch.remainder(t0 + dt, self.yaw_bins)
            wt = ft if dt else 1.0 - ft
            for dy in (0, 1):
                yi = y0 + dy
                wy = fy if dy else 1.0 - fy
                for dx in (0, 1):
                    xi = x0 + dx
                    wx = fx if dx else 1.0 - fx
                    valid = (
                        (xi >= 0)
                        & (xi < moved.shape[2])
                        & (yi >= 0)
                        & (yi < moved.shape[1])
                    )
                    flat = (
                        (ti[valid] * moved.shape[1] + yi[valid])
                        * moved.shape[2]
                        + xi[valid]
                    )
                    moved.view(-1).scatter_add_(
                        0,
                        flat,
                        weights[valid] * wt[valid] * wy[valid] * wx[valid],
                    )
        if float(moved.sum()) <= 0:
            return self.uniform.clone()
        return self._smooth(
            moved,
            self.config["motion_sigma_position_m"],
            self.config["motion_sigma_yaw_deg"],
        )

    def _mode_pose(self, probability):
        flat = int(torch.argmax(probability).item())
        height, width = probability.shape[1:]
        yaw_idx = flat // (height * width)
        remainder = flat % (height * width)
        row, col = divmod(remainder, width)
        radius_cells = int(math.ceil(
            self.config["estimate_mode_radius_m"] / self.resolution
        ))
        yaw_radius = int(math.ceil(
            math.radians(self.config["estimate_mode_radius_yaw_deg"])
            / self.yaw_resolution
        ))
        rows = torch.arange(
            max(0, row - radius_cells),
            min(height, row + radius_cells + 1),
            device=self.device,
        )
        cols = torch.arange(
            max(0, col - radius_cells),
            min(width, col + radius_cells + 1),
            device=self.device,
        )
        yaws = torch.remainder(
            torch.arange(
                yaw_idx - yaw_radius,
                yaw_idx + yaw_radius + 1,
                device=self.device,
            ),
            self.yaw_bins,
        )
        tt, yy, xx = torch.meshgrid(yaws, rows, cols, indexing="ij")
        spatial_distance = torch.sqrt(
            (xx.float() - col) ** 2 + (yy.float() - row) ** 2
        ) * self.resolution
        angular_distance = torch.abs(
            torch.remainder(
                (tt.float() - yaw_idx) * self.yaw_resolution + math.pi,
                2.0 * math.pi,
            )
            - math.pi
        )
        valid = (
            spatial_distance <= self.config["estimate_mode_radius_m"]
        ) & (
            angular_distance
            <= math.radians(self.config["estimate_mode_radius_yaw_deg"])
        )
        weights = probability[tt, yy, xx] * valid
        total = weights.sum().clamp_min(1e-30)
        world_x = self.origin_xy[0] + (
            xx.float() + 0.5
        ) * self.resolution
        world_y = self.origin_xy[1] + (
            yy.float() + 0.5
        ) * self.resolution
        angles = tt.float() * self.yaw_resolution
        x = (weights * world_x).sum() / total
        y = (weights * world_y).sum() / total
        yaw_value = torch.atan2(
            (weights * torch.sin(angles)).sum(),
            (weights * torch.cos(angles)).sum(),
        )
        return np.asarray(
            [x.item(), y.item(), wrap_angle(yaw_value.item())],
            dtype=np.float32,
        )

    def update(
        self,
        candidate_poses,
        visual_logits=None,
        odometry_delta=None,
        stage1_pose=None,
    ):
        candidates = np.asarray(candidate_poses, dtype=np.float64)
        if (
            candidates.ndim != 2
            or candidates.shape[1] != 3
            or len(candidates) == 0
        ):
            raise ValueError(
                "candidate_poses must have shape [N, 3] with N > 0"
            )
        if not np.isfinite(candidates).all():
            raise ValueError("candidate_poses must be finite")

        # Diffusion outputs posterior samples. Equal mass preserves secondary
        # modes; KDE-weighting would count the same concentration twice.
        if self.uniform is None:
            self._auto_configure(candidates)
        measurement = self._measurement_belief(candidates)
        candidate_pose = self._mode_pose(measurement)
        support_size = int(self.free_mask.sum().item())
        candidate_confidence = normalized_entropy_confidence(
            measurement, support_size
        )

        predicted = None
        if self.belief is None:
            posterior = measurement
        else:
            predicted = self._propagate(self.belief, odometry_delta)
            prior_mix = float(self.config["prior_uniform_mix"])
            measurement_mix = float(
                self.config["measurement_uniform_mix"]
            )
            prior = (
                (1.0 - prior_mix) * predicted
                + prior_mix * self.uniform
            )
            likelihood = (
                (1.0 - measurement_mix) * measurement
                + measurement_mix * self.uniform
            )
            log_posterior = (
                float(self.config["motion_log_weight"])
                * torch.log(prior.clamp_min(1e-30))
                + float(self.config["stage1_log_weight"])
                * torch.log(likelihood.clamp_min(1e-30))
            )
            log_posterior = log_posterior.masked_fill(
                ~self.free_mask, -torch.inf
            )
            posterior = torch.softmax(
                log_posterior.flatten(), dim=0
            ).reshape_as(prior)

        self.belief = posterior / posterior.sum().clamp_min(1e-30)
        output_pose = self._mode_pose(self.belief)
        confidence = normalized_entropy_confidence(
            self.belief, support_size
        )
        innovation_position = innovation_yaw = 0.0
        if predicted is not None:
            innovation_position, innovation_yaw = pose_errors(
                candidate_pose, self._mode_pose(predicted)
            )

        return LocalizationEstimate(
            pose=output_pose,
            confidence=confidence,
            candidate_pose=candidate_pose,
            candidate_confidence=candidate_confidence,
            observation_accepted=True,
            innovation_position_m=float(innovation_position),
            innovation_yaw_deg=float(innovation_yaw),
            stage1_pose=(
                None
                if stage1_pose is None
                else np.asarray(stage1_pose, dtype=np.float32)
            ),
            candidate_count=len(candidates),
        )
