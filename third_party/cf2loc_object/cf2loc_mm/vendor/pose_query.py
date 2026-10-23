import math
from typing import Dict, Optional, Sequence, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR





LEGACY_COORDINATE_CONVENTION = "legacy_normalized_v0"
METRIC_COORDINATE_CONVENTION = "metric_cell_center_v1"
SUPPORTED_COORDINATE_CONVENTIONS = {
    LEGACY_COORDINATE_CONVENTION,
    METRIC_COORDINATE_CONVENTION,
}


def map_xy_to_normalized(xy: torch.Tensor, wh: torch.Tensor) -> torch.Tensor:
    """Map continuous pixel coordinates to the normalized map extent.

    Pose coordinates are measured from the top-left map boundary, so 0 and W/H
    map to -1 and +1. Feature tokens use cell centers within the same extent.
    """
    wh = wh.to(device=xy.device, dtype=xy.dtype)
    while wh.ndim < xy.ndim:
        wh = wh.unsqueeze(1)
    return xy / wh.clamp_min(1.0) * 2.0 - 1.0


def normalized_to_map_xy(xy_norm: torch.Tensor, wh: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`map_xy_to_normalized`."""
    wh = wh.to(device=xy_norm.device, dtype=xy_norm.dtype)
    while wh.ndim < xy_norm.ndim:
        wh = wh.unsqueeze(1)
    return (xy_norm + 1.0) * 0.5 * wh


def build_cell_center_coordinates(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return normalized feature-cell centers in row-major order."""
    pos_x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    pos_y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    pos_x = pos_x * 2.0 - 1.0
    pos_y = pos_y * 2.0 - 1.0
    grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)


def cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(num_steps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(
        ((steps / num_steps + s) / (1 + s)) * math.pi * 0.5
    ).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-5, 0.999).float()


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.feature_dim // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(
                half_dim,
                device=timesteps.device,
                dtype=torch.float32,
            )
            / max(half_dim - 1, 1)
        )
        angles = timesteps.float().unsqueeze(-1) * frequencies
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if embedding.shape[-1] < self.feature_dim:
            embedding = F.pad(embedding, (0, self.feature_dim - embedding.shape[-1]))
        return self.mlp(embedding)


class PoseMapCrossAttention(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        dropout: float,
        metric_relative_max_m: float,
        metric_relative_scale_m: float,
        coordinate_convention: str,
    ):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.metric_relative_max_m = float(metric_relative_max_m)
        self.metric_relative_scale_m = float(metric_relative_scale_m)
        self.coordinate_convention = coordinate_convention
        if self.metric_relative_max_m <= 0:
            raise ValueError("metric_relative_max_m must be positive.")
        if self.metric_relative_scale_m <= 0:
            raise ValueError("metric_relative_scale_m must be positive.")
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.output = nn.Linear(feature_dim, feature_dim)
        self.relative_bias = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, num_heads),
        )
        self.dropout = nn.Dropout(dropout)

    def project_map_tokens(
        self,
        map_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_map_tokens, feature_dim = map_tokens.shape
        key = self.key(map_tokens).view(
            batch_size, num_map_tokens, self.num_heads, self.head_dim
        )
        value = self.value(map_tokens).view(
            batch_size, num_map_tokens, self.num_heads, self.head_dim
        )
        if feature_dim != self.num_heads * self.head_dim:
            raise ValueError("Unexpected map-token feature dimension.")
        return key, value

    @staticmethod
    def build_metric_relative_features(
        noisy_pose: torch.Tensor,
        map_coordinates: torch.Tensor,
        wh: torch.Tensor,
        map_res,
        max_distance_m: float,
        scale_m: float,
    ) -> torch.Tensor:
        """Build clipped local-frame metric offsets with a fixed metric scale."""
        batch_size, num_particles = noisy_pose.shape[:2]
        num_map_tokens = map_coordinates.shape[0]
        wh = wh.to(device=noisy_pose.device, dtype=noisy_pose.dtype)
        map_res = torch.as_tensor(
            map_res,
            device=noisy_pose.device,
            dtype=noisy_pose.dtype,
        )
        if map_res.ndim == 0:
            map_res = map_res.expand(batch_size)
        map_res = map_res.reshape(batch_size, 1, 1)

        pose_xy = noisy_pose[..., :2]
        angle_vec = F.normalize(noisy_pose[..., 2:4], dim=-1, eps=1e-6)
        sin_theta = angle_vec[..., 0].unsqueeze(-1)
        cos_theta = angle_vec[..., 1].unsqueeze(-1)
        delta_norm = (
            map_coordinates.view(1, 1, num_map_tokens, 2)
            - pose_xy.unsqueeze(2)
        )
        delta_m = delta_norm * wh.view(batch_size, 1, 1, 2) * 0.5
        delta_m = delta_m * map_res.unsqueeze(-1)
        dx_m = delta_m[..., 0]
        dy_m = delta_m[..., 1]
        local_x_m = cos_theta * dx_m + sin_theta * dy_m
        local_y_m = -sin_theta * dx_m + cos_theta * dy_m
        distance_m = torch.sqrt(dx_m.square() + dy_m.square() + 1e-8)

        clip = float(max_distance_m)
        scale = float(scale_m)
        local_x = local_x_m.clamp(-clip, clip) / scale
        local_y = local_y_m.clamp(-clip, clip) / scale
        distance = distance_m.clamp(0.0, clip) / scale
        return torch.stack(
            [local_x, local_y, distance, distance.square()],
            dim=-1,
        )

    def forward(
        self,
        pose_tokens: torch.Tensor,
        map_tokens: torch.Tensor,
        noisy_pose: torch.Tensor,
        map_coordinates: torch.Tensor,
        wh: torch.Tensor,
        map_res,
        map_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size, num_particles, feature_dim = pose_tokens.shape
        num_map_tokens = map_tokens.shape[1]

        query = self.query(pose_tokens).view(
            batch_size, num_particles, self.num_heads, self.head_dim
        )
        if map_key_value is None:
            key, value = self.project_map_tokens(map_tokens)
        else:
            key, value = map_key_value
            expected_shape = (
                batch_size,
                num_map_tokens,
                self.num_heads,
                self.head_dim,
            )
            if key.shape != expected_shape or value.shape != expected_shape:
                raise ValueError(
                    "Cached map K/V shape does not match the current map tokens."
                )
        attention = torch.einsum("bmhd,bnhd->bhmn", query, key) * self.scale

        if self.coordinate_convention == METRIC_COORDINATE_CONVENTION:
            relative_features = self.build_metric_relative_features(
                noisy_pose,
                map_coordinates,
                wh,
                map_res,
                self.metric_relative_max_m,
                self.metric_relative_scale_m,
            )
        else:
            pose_xy = noisy_pose[..., :2]
            angle_vec = F.normalize(noisy_pose[..., 2:4], dim=-1, eps=1e-6)
            sin_theta = angle_vec[..., 0].unsqueeze(-1)
            cos_theta = angle_vec[..., 1].unsqueeze(-1)
            delta = (
                map_coordinates.view(1, 1, num_map_tokens, 2)
                - pose_xy.unsqueeze(2)
            )
            dx = delta[..., 0]
            dy = delta[..., 1]
            local_x = cos_theta * dx + sin_theta * dy
            local_y = -sin_theta * dx + cos_theta * dy
            distance = torch.sqrt(dx.square() + dy.square() + 1e-8)
            relative_features = torch.stack(
                [local_x, local_y, distance, distance.square()], dim=-1
            )
        relative_bias = self.relative_bias(relative_features).permute(0, 3, 1, 2)
        attention = torch.softmax(attention + relative_bias, dim=-1)
        attention = self.dropout(attention)

        attended = torch.einsum("bhmn,bnhd->bmhd", attention, value)
        attended = attended.reshape(batch_size, num_particles, feature_dim)
        return self.output(attended)


class PoseDenoiserBlock(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        dropout: float,
        metric_relative_max_m: float,
        metric_relative_scale_m: float,
        coordinate_convention: str,
    ):
        super().__init__()
        self.cross_attn = PoseMapCrossAttention(
            feature_dim,
            num_heads,
            dropout,
            metric_relative_max_m,
            metric_relative_scale_m,
            coordinate_convention,
        )
        self.cross_norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        pose_tokens: torch.Tensor,
        map_tokens: torch.Tensor,
        noisy_pose: torch.Tensor,
        map_coordinates: torch.Tensor,
        wh: torch.Tensor,
        map_res,
        map_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        attended = self.cross_attn(
            pose_tokens,
            map_tokens,
            noisy_pose,
            map_coordinates,
            wh,
            map_res,
            map_key_value=map_key_value,
        )
        pose_tokens = self.cross_norm(pose_tokens + self.dropout(attended))
        return self.ffn_norm(pose_tokens + self.dropout(self.ffn(pose_tokens)))


class PoseQueryDenoiser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_dim = int(config.get("diffusion_feature_dim", 128))
        self.num_steps = int(config.get("diffusion_train_steps", 1000))
        self.fourier_bands = int(config.get("diffusion_pose_fourier_bands", 4))
        pose_input_dim = 4 * (1 + 2 * self.fourier_bands)
        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_input_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.time_embedding = SinusoidalTimeEmbedding(self.feature_dim)
        self.image_global_projection = nn.Linear(self.feature_dim, self.feature_dim)
        metric_relative_max_m = float(
            config.get("diffusion_metric_relative_max_m", 20.0)
        )
        metric_relative_scale_m = float(
            config.get("diffusion_metric_relative_scale_m", 5.0)
        )
        coordinate_convention = config.get(
            "diffusion_coordinate_convention",
            LEGACY_COORDINATE_CONVENTION,
        )
        self.blocks = nn.ModuleList(
            [
                PoseDenoiserBlock(
                    feature_dim=self.feature_dim,
                    num_heads=int(config.get("diffusion_num_heads", 4)),
                    dropout=float(config.get("diffusion_dropout", 0.1)),
                    metric_relative_max_m=metric_relative_max_m,
                    metric_relative_scale_m=metric_relative_scale_m,
                    coordinate_convention=coordinate_convention,
                )
                for _ in range(int(config.get("diffusion_denoiser_layers", 2)))
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, 4),
        )

    def fourier_encode(self, pose: torch.Tensor) -> torch.Tensor:
        features = [pose]
        for band in range(self.fourier_bands):
            frequency = math.pi * (2**band)
            features.extend(
                [torch.sin(frequency * pose), torch.cos(frequency * pose)]
            )
        return torch.cat(features, dim=-1)

    def build_map_kv_cache(
        self,
        map_tokens: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple(
            block.cross_attn.project_map_tokens(map_tokens)
            for block in self.blocks
        )

    def forward(
        self,
        noisy_pose: torch.Tensor,
        timesteps: torch.Tensor,
        map_tokens: torch.Tensor,
        map_coordinates: torch.Tensor,
        image_global: torch.Tensor,
        wh: torch.Tensor,
        map_res,
        map_kv_cache: Optional[
            Sequence[Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> torch.Tensor:
        if map_kv_cache is not None and len(map_kv_cache) != len(self.blocks):
            raise ValueError("Map K/V cache must contain one entry per block.")
        pose_tokens = self.pose_mlp(self.fourier_encode(noisy_pose))
        pose_tokens = pose_tokens + self.time_embedding(timesteps)
        pose_tokens = pose_tokens + self.image_global_projection(image_global).unsqueeze(1)
        for block_index, block in enumerate(self.blocks):
            pose_tokens = block(
                pose_tokens,
                map_tokens,
                noisy_pose,
                map_coordinates,
                wh,
                map_res,
                map_key_value=(
                    map_kv_cache[block_index]
                    if map_kv_cache is not None
                    else None
                ),
            )
        return self.output(pose_tokens)
