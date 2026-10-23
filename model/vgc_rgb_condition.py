"""VGCNav condition components; attribution in THIRD_PARTY_BRIDGE.md."""
import math
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F

def compute_path_turn_risk(
    astar_local_cond: torch.Tensor,
    horizon: int = 20,
    stride: int = 3,
    smoothing_window: int = 3,
    waypoint_trust: Optional[torch.Tensor] = None,
    return_path_confidence: bool = False,
):
    """Return a robust [0, 1] turn summary from a clean local A* path.

    The previous implementation consumed adjacent, independently perturbed
    points and mixed the mean with the single largest turn.  That made one
    grid kink (or one noisy point) dominate the visual encoder.  Here the
    path is mask-aware smoothed first and headings span several waypoints.
    When waypoint trust is supplied, each turn uses the minimum trust of its
    three support points and the ungated valid-weight denominator makes both
    turn risk and returned path confidence vanish as trust vanishes.
    """
    if astar_local_cond.ndim != 3 or astar_local_cond.shape[-1] < 5:
        raise ValueError(
            "astar_local_cond must have shape [B, T, >=5] with "
            "[x, y, sin(turn), cos(turn), valid]"
        )
    if waypoint_trust is not None:
        if waypoint_trust.shape != astar_local_cond.shape[:2]:
            raise ValueError("waypoint_trust must have shape [B, T]")
        waypoint_trust = waypoint_trust.to(
            device=astar_local_cond.device, dtype=astar_local_cond.dtype
        ).clamp(0.0, 1.0)
    horizon = min(int(horizon), astar_local_cond.shape[1])
    if horizon < 3:
        zeros = astar_local_cond.new_zeros(astar_local_cond.shape[0])
        return (zeros, zeros) if return_path_confidence else zeros

    local_cond = astar_local_cond[:, :horizon]
    valid = local_cond[..., 4] > 0.5
    points = local_cond[..., :2]

    window = max(1, min(int(smoothing_window), horizon))
    if window % 2 == 0:
        window -= 1
    if window > 1:
        mask = valid.to(points.dtype).unsqueeze(1)
        kernel = points.new_ones(1, 1, window)
        padding = window // 2
        point_sum = F.conv1d(
            (points * valid.unsqueeze(-1)).transpose(1, 2),
            kernel.expand(2, 1, window),
            padding=padding,
            groups=2,
        )
        count = F.conv1d(mask, kernel, padding=padding).clamp_min(1.0)
        smoothed = (point_sum / count).transpose(1, 2)
        points = torch.where(valid.unsqueeze(-1), smoothed, points)

    max_stride = max(1, min(int(stride), (horizon - 1) // 2))
    risk = astar_local_cond.new_zeros(astar_local_cond.shape[0])
    path_confidence = astar_local_cond.new_zeros(astar_local_cond.shape[0])
    unresolved = torch.ones_like(risk, dtype=torch.bool)
    # Prefer the requested wide baseline, but fall back for short valid paths.
    for current_stride in range(max_stride, 0, -1):
        centers = points[:, current_stride : horizon - current_stride]
        incoming = centers - points[:, : horizon - 2 * current_stride]
        outgoing = points[:, 2 * current_stride : horizon] - centers
        usable = (
            valid[:, : horizon - 2 * current_stride]
            & valid[:, current_stride : horizon - current_stride]
            & valid[:, 2 * current_stride : horizon]
            & (torch.linalg.vector_norm(incoming, dim=-1) > 1e-6)
            & (torch.linalg.vector_norm(outgoing, dim=-1) > 1e-6)
        )
        cross = incoming[..., 0] * outgoing[..., 1] - incoming[..., 1] * outgoing[..., 0]
        dot = (incoming * outgoing).sum(dim=-1)
        # Ninety degrees corresponds to full turn risk; larger turns are clipped.
        abs_turn = (torch.atan2(cross, dot).abs() / (math.pi / 2.0)).clamp(0.0, 1.0)
        weights = torch.linspace(
            1.0,
            0.25,
            abs_turn.shape[1],
            device=local_cond.device,
            dtype=local_cond.dtype,
        ).unsqueeze(0)
        weighted_valid = weights * usable.to(local_cond.dtype)
        if waypoint_trust is None:
            turn_trust = torch.ones_like(abs_turn)
        else:
            trust = waypoint_trust[:, :horizon]
            # A turn is trustworthy only when its incoming, center and outgoing
            # support points are all trustworthy.
            turn_trust = torch.minimum(
                torch.minimum(
                    trust[:, : horizon - 2 * current_stride],
                    trust[:, current_stride : horizon - current_stride],
                ),
                trust[:, 2 * current_stride : horizon],
            )
        has_turn = usable.any(dim=1)
        normalizer = weighted_valid.sum(dim=1).clamp_min(1e-6)
        candidate = (
            abs_turn * weighted_valid * turn_trust
        ).sum(dim=1) / normalizer
        confidence_candidate = (
            weighted_valid * turn_trust
        ).sum(dim=1) / normalizer
        take = unresolved & has_turn
        risk = torch.where(take, candidate, risk)
        path_confidence = torch.where(
            take, confidence_candidate, path_confidence
        )
        unresolved = unresolved & ~has_turn
    if return_path_confidence:
        return risk, path_confidence
    return risk

class CurvatureResidualFusion(nn.Module):
    """Add a bounded, spatially attended shallow residual to deep features."""

    def __init__(
        self,
        feature_dim: int,
        shallow_dim: int,
        attention_heads: int = 6,
        alpha_max: float = 0.25,
        initial_alpha_fraction: float = 0.10,
    ) -> None:
        super().__init__()
        if not 0.0 < alpha_max <= 1.0:
            raise ValueError("alpha_max must be in (0, 1]")
        self.alpha_max = float(alpha_max)
        self.query = nn.Linear(feature_dim + 1, shallow_dim)
        self.spatial_attention = nn.MultiheadAttention(
            shallow_dim, attention_heads, batch_first=True
        )
        self.shallow_summary = nn.Linear(shallow_dim, feature_dim)
        self.shallow_adapter = nn.Sequential(
            nn.LayerNorm(shallow_dim),
            nn.Linear(shallow_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(feature_dim * 2 + 1),
            nn.Linear(feature_dim * 2 + 1, max(32, feature_dim // 2)),
            nn.GELU(),
            nn.Linear(max(32, feature_dim // 2), 1),
        )
        # The model starts exactly at the original deep-feature baseline.
        nn.init.zeros_(self.shallow_adapter[-1].weight)
        nn.init.zeros_(self.shallow_adapter[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        fraction = min(max(float(initial_alpha_fraction), 1e-4), 1.0 - 1e-4)
        nn.init.constant_(self.gate[-1].bias, math.log(fraction / (1.0 - fraction)))

    def forward(
        self,
        shallow_patch_tokens: torch.Tensor,
        deep_features: torch.Tensor,
        turn_risk: torch.Tensor,
        path_confidence: Optional[torch.Tensor] = None,
    ):
        if shallow_patch_tokens.ndim != 3 or deep_features.ndim != 3:
            raise ValueError("Expected shallow [B,P,C] and deep [B,T,C] features")
        deep_current = deep_features[:, -1]
        query = self.query(torch.cat([deep_current, turn_risk[:, None]], dim=-1))
        attended, attention_weights = self.spatial_attention(
            query[:, None], shallow_patch_tokens, shallow_patch_tokens
        )
        attended = attended[:, 0]
        correction = self.shallow_adapter(attended)
        gate_input = torch.cat(
            [deep_current, self.shallow_summary(attended), turn_risk[:, None]], dim=-1
        )
        # ``turn_risk`` has already been confidence-weighted waypoint by
        # waypoint. Multiplying by aggregate path confidence here again caused
        # a quadratic attenuation and nearly disabled the shallow branch.
        alpha = self.alpha_max * torch.sigmoid(
            self.gate(gate_input).squeeze(-1)
        )
        fused = deep_features.clone()
        fused[:, -1] = fused[:, -1] + alpha[:, None] * correction
        return fused, alpha, attention_weights[:, 0]

class LocalConditionTrustGate(nn.Module):
    """Gate each five-dimensional A* token using vision and path quality.

    The visual head estimates risk around the queried A* waypoint, while the
    path-only head estimates how reliable the A* geometry is.  Separate logits
    allow the two causes to receive explicit, interpretable supervision.
    """

    def __init__(
        self,
        visual_dim: int,
        local_cond_dim: int = 5,
        hidden_dim: int = 128,
        attention_heads: int = 4,
        coordinate_scale: float = 32.0,
        initial_trust: float = 0.90,
        minimum_trust: float = 0.50,
        risk_residual_scale: float = 0.50,
    ) -> None:
        super().__init__()
        if local_cond_dim != 5:
            raise ValueError("LocalConditionTrustGate expects the five A* features")
        if visual_dim % attention_heads != 0:
            raise ValueError("visual_dim must be divisible by attention_heads")
        if coordinate_scale <= 0.0:
            raise ValueError("coordinate_scale must be positive")
        if not 0.0 < initial_trust < 1.0:
            raise ValueError("initial_trust must be in (0, 1)")
        if not 0.0 <= minimum_trust < 1.0:
            raise ValueError("minimum_trust must be in [0, 1)")
        if risk_residual_scale < 0.0:
            raise ValueError("risk_residual_scale must be non-negative")

        self.coordinate_scale = float(coordinate_scale)
        self.initial_visual_risk = 1.0 - float(initial_trust)
        self.minimum_trust = float(minimum_trust)
        self.risk_residual_scale = float(risk_residual_scale)
        self.path_encoder = nn.Sequential(
            nn.Linear(local_cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.path_context = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.visual_query = nn.Linear(hidden_dim, visual_dim)
        self.visual_attention = nn.MultiheadAttention(
            visual_dim, attention_heads, batch_first=True
        )
        self.visual_risk_head = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.astar_reliability_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Begin near the former full-condition behavior; supervised losses
        # calibrate both factors instead of letting the diffusion loss collapse
        # the gate to zero.
        nn.init.zeros_(self.visual_risk_head[-1].weight)
        nn.init.zeros_(self.visual_risk_head[-1].bias)
        trust_logit = math.log(initial_trust / (1.0 - initial_trust))
        nn.init.zeros_(self.astar_reliability_head[-1].weight)
        nn.init.constant_(self.astar_reliability_head[-1].bias, trust_logit)

    def forward(
        self,
        shallow_patch_tokens: torch.Tensor,
        astar_local_cond: torch.Tensor,
        depth_risk: Optional[torch.Tensor] = None,
    ) -> dict:
        if shallow_patch_tokens.ndim != 3:
            raise ValueError("Expected shallow visual tokens with shape [B, P, C]")
        if astar_local_cond.ndim != 3 or astar_local_cond.shape[-1] != 5:
            raise ValueError(
                "astar_local_cond must be [B, T, 5] with "
                "[x, y, sin(turn), cos(turn), valid]"
            )

        gate_input = astar_local_cond.float().clone()
        gate_input[..., :2] = gate_input[..., :2] / self.coordinate_scale
        path_features = self.path_encoder(gate_input)
        contextual = self.path_context(path_features.transpose(1, 2)).transpose(1, 2)
        path_features = path_features + contextual

        visual_query = self.visual_query(path_features)
        visual_features, spatial_attention = self.visual_attention(
            visual_query, shallow_patch_tokens, shallow_patch_tokens
        )
        if depth_risk is None:
            depth_risk = astar_local_cond.new_full(
                astar_local_cond.shape[:2], self.initial_visual_risk
            )
        if depth_risk.shape != astar_local_cond.shape[:2]:
            raise ValueError("depth_risk must have shape [B, T]")
        depth_risk = depth_risk.detach().clamp(1e-4, 1.0 - 1e-4)
        depth_risk_logits = torch.logit(depth_risk)
        # Optional depth geometry sets a detached base risk. The FloVerse RGB-only
        # caller supplies no depth, so the base is the constant initial_visual_risk.
        # RGB learns a bounded residual; without collision/depth supervision this
        # is a confidence modulation signal, not a calibrated collision probability.
        visual_residual = torch.tanh(
            self.visual_risk_head(visual_features).squeeze(-1)
        )
        visual_risk_logits = (
            depth_risk_logits + self.risk_residual_scale * visual_residual
        )
        astar_reliability_logits = self.astar_reliability_head(
            path_features
        ).squeeze(-1)

        visual_risk = torch.sigmoid(visual_risk_logits)
        astar_reliability = torch.sigmoid(astar_reliability_logits)
        valid = astar_local_cond[..., 4].clamp(0.0, 1.0)
        trust_gate = astar_reliability * (1.0 - visual_risk) * valid
        effective_gate = valid * (
            self.minimum_trust + (1.0 - self.minimum_trust) * trust_gate
        )

        # Preserve x/y and the unit sin/cos geometry.  Confidence replaces the
        # binary validity channel as a soft validity feature; invalid confidence
        # remains zero while terminal padded XY is preserved. The former
        # all-channel multiplication shrank
        # direction vectors off the unit circle and conflicted with the full
        # A* bridge endpoint.
        gated_local_cond = astar_local_cond.clone()
        gated_local_cond[..., 4] = effective_gate
        return {
            "gated_local_cond": gated_local_cond,
            "local_condition_gate": trust_gate,
            "local_condition_effective_gate": effective_gate,
            "visual_risk": visual_risk,
            "visual_risk_logits": visual_risk_logits,
            "visual_risk_residual": visual_residual,
            "depth_risk_target": depth_risk,
            "astar_reliability": astar_reliability,
            "astar_reliability_logits": astar_reliability_logits,
            "local_gate_spatial_attention": spatial_attention,
        }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6):
        super().__init__()

        # Compute the positional encoding once
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pos_enc[:, :x.size(1), :]
        return x
