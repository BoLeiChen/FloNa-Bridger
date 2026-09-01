import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision
import math
from typing import Callable, Optional, Sequence
from efficientnet_pytorch import EfficientNet
from depth_anything_v2.dinov2 import DINOv2


def compute_path_turn_risk(
    astar_local_cond: torch.Tensor,
    horizon: int = 20,
    stride: int = 3,
    smoothing_window: int = 3,
) -> torch.Tensor:
    """Return a robust [0, 1] turn summary from a clean local A* path.

    The previous implementation consumed adjacent, independently perturbed
    points and mixed the mean with the single largest turn.  That made one
    grid kink (or one noisy point) dominate the visual encoder.  Here the
    path is mask-aware smoothed first and headings span several waypoints.
    """
    if astar_local_cond.ndim != 3 or astar_local_cond.shape[-1] < 5:
        raise ValueError(
            "astar_local_cond must have shape [B, T, >=5] with "
            "[x, y, sin(turn), cos(turn), valid]"
        )
    horizon = min(int(horizon), astar_local_cond.shape[1])
    if horizon < 3:
        return astar_local_cond.new_zeros(astar_local_cond.shape[0])

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
        has_turn = usable.any(dim=1)
        candidate = (abs_turn * weighted_valid).sum(dim=1) / weighted_valid.sum(dim=1).clamp_min(1e-6)
        take = unresolved & has_turn
        risk = torch.where(take, candidate, risk)
        unresolved = unresolved & ~has_turn
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
        alpha = self.alpha_max * torch.sigmoid(self.gate(gate_input).squeeze(-1))
        fused = deep_features.clone()
        fused[:, -1] = fused[:, -1] + alpha[:, None] * correction
        return fused, alpha, attention_weights[:, 0]


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
    
class flona_ViNT(nn.Module):
    def __init__(
        self,
        context_size: int = 5,
        obs_encoder: Optional[str] = "dinov2",  # Changed default to dinov2
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
        dinov2_checkpoint: Optional[str] = "checkpoints/depth_anything_v2_vits.pth",
        shallow_feature_layers: Sequence[int] = (2, 5),
        deep_feature_layer: int = 11,
        curvature_gate_horizon: int = 20,
        curvature_stride: int = 3,
        curvature_smoothing_window: int = 3,
        shallow_residual_alpha_max: float = 0.25,
        use_shallow_visual_residual: bool = True,
        obs_already_normalized: bool = True,
    ) -> None:
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.floorplan_encoding_size = obs_encoding_size
        self.context_size = context_size
        self.shallow_feature_layers = tuple(int(i) for i in shallow_feature_layers)
        self.deep_feature_layer = int(deep_feature_layer)
        self.curvature_gate_horizon = int(curvature_gate_horizon)
        self.curvature_stride = int(curvature_stride)
        self.curvature_smoothing_window = int(curvature_smoothing_window)
        self.use_shallow_visual_residual = bool(use_shallow_visual_residual)
        self.obs_already_normalized = bool(obs_already_normalized)

        # self.obs_floorplan_enc = nn.Linear(4, obs_encoding_size,device="cuda")
        # self.obs_floorplan_expand = nn.Linear(6, obs_encoding_size,device="cuda")
        self.obs_goal_pos_ori_enc = nn.Linear(obs_encoding_size + 6, obs_encoding_size)
        
        # Initialize the observation encoder with DINOv2
        self.obs_encoder = DINOv2("vits")  # ViT-Small, embed_dim=384
        self.num_deep_features = self.obs_encoder.embed_dim * 2
        self.obs_encoder_type = "dinov2"
        
        # Load Depth Anything V2 pretrained weights
        if dinov2_checkpoint:
            state_dict = torch.load(dinov2_checkpoint, map_location="cpu")
            # Only load the pretrained (DINOv2) part of the weights
            dinov2_state_dict = {k.replace("pretrained.", ""): v 
                                 for k, v in state_dict.items() 
                                 if k.startswith("pretrained.")}
            self.obs_encoder.load_state_dict(dinov2_state_dict, strict=True)
            print(f"Loaded DINOv2 weights from {dinov2_checkpoint}")
        
        # Freeze DINOv2 parameters
        for param in self.obs_encoder.parameters():
            param.requires_grad = False
        self.obs_encoder.eval()
        
        # Register ImageNet normalization constants as buffers
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
       
        # Initialize the floorplan encoder (keep EfficientNet for 6-channel input)
        self.floorplan_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6)
        self.floorplan_encoder = replace_bn_with_gn(self.floorplan_encoder)
        self.num_floorplan_features = self.floorplan_encoder._fc.in_features

        self.compress_deep_obs_enc = nn.Linear(
            self.num_deep_features, self.obs_encoding_size
        )
        self.curvature_visual_fusion = CurvatureResidualFusion(
            feature_dim=self.obs_encoding_size,
            shallow_dim=self.obs_encoder.embed_dim,
            alpha_max=shallow_residual_alpha_max,
        )

        
        if self.num_floorplan_features != self.floorplan_encoding_size:
            self.compress_floorplan_enc = nn.Linear(self.num_floorplan_features, self.floorplan_encoding_size)
        else:
            self.compress_floorplan_enc = nn.Identity()

        # Initialize positional encoding and self-attention layers
        self.positional_encoding = PositionalEncoding(self.obs_encoding_size, max_seq_len=self.context_size + 2)
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size, 
            nhead=mha_num_attention_heads, 
            dim_feedforward=mha_ff_dim_factor*self.obs_encoding_size, 
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=mha_num_attention_layers)

    def train(self, mode: bool = True):
        super().train(mode)
        # The pretrained DINO backbone is frozen and must remain deterministic.
        self.obs_encoder.eval()
        return self

    @staticmethod
    def _pool_deep_features(patch_tokens, cls_token):
        return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)

    def forward(self, 
                obs_img: torch.tensor, 
                floorplan_img: torch.tensor, 
                obs_pos: torch.tensor, 
                goal_pos: torch.tensor, 
                obs_ori: torch.tensor,
                astar_local_cond: Optional[torch.Tensor] = None,
                astar_global_cond: Optional[torch.Tensor] = None,
                # input_goal_mask: torch.tensor = None
                ) -> torch.Tensor:
        device = obs_img.device
        obs_goal_pos = torch.cat([obs_pos, goal_pos], dim=1)
        obs_goal_pos_ori = torch.cat([obs_goal_pos, obs_ori], dim=1).to(device) 
        
        # Get the current obs and floorplan encoding
        # Extract current observation (last frame in context) and resize to floorplan resolution
        current_obs = obs_img[:, 3*self.context_size:, :, :]  # (B, 3, 224, 224)
        floorplan_h, floorplan_w = floorplan_img.shape[2], floorplan_img.shape[3]
        if current_obs.shape[2] != floorplan_h or current_obs.shape[3] != floorplan_w:
            current_obs_resized = torch.nn.functional.interpolate(
                current_obs, size=(floorplan_h, floorplan_w), mode='bilinear', align_corners=False
            )
        else:
            current_obs_resized = current_obs
        obsfloorplan_img = torch.cat([current_obs_resized, floorplan_img], dim=1)  # (B, 6, 96, 96)
        obsfloorplan_encoding = self.floorplan_encoder.extract_features(obsfloorplan_img) # get encoding of this img
        obsfloorplan_encoding = self.floorplan_encoder._avg_pooling(obsfloorplan_encoding) # avg pooling

        if self.floorplan_encoder._global_params.include_top:
            obsfloorplan_encoding = obsfloorplan_encoding.flatten(start_dim=1)
            obsfloorplan_encoding = self.floorplan_encoder._dropout(obsfloorplan_encoding)
        obsfloorplan_encoding = self.compress_floorplan_enc(obsfloorplan_encoding)

        if len(obsfloorplan_encoding.shape) == 2:
            obsfloorplan_encoding = obsfloorplan_encoding.unsqueeze(1)
        assert obsfloorplan_encoding.shape[2] == self.floorplan_encoding_size
        floorplan_encoding = obsfloorplan_encoding    # b, 1, obs_encoding_size
        
        # Get the observations encoding using DINOv2
        obs_img_split = torch.split(obs_img, 3, dim=1)  # List of (B, 3, H, W)
        obs_img_cat = torch.cat(obs_img_split, dim=0)    # ((context_size+1)*B, 3, H, W)
        
        # Dataset transforms already apply ImageNet normalization by default.
        # Keeping this switch explicit prevents the former double-normalization.
        if self.obs_already_normalized:
            obs_img_normalized = obs_img_cat
        else:
            obs_img_normalized = (
                obs_img_cat - self.imagenet_mean.to(device)
            ) / self.imagenet_std.to(device)
        
        # One DINO pass returns both shallow spatial and deep semantic features.
        with torch.no_grad():
            requested_layers = tuple(sorted(set(
                self.shallow_feature_layers + (self.deep_feature_layer,)
            )))
            intermediate = self.obs_encoder.get_intermediate_layers(
                obs_img_normalized,
                n=requested_layers,
                return_class_token=True,
                norm=True,
            )
        layer_features = dict(zip(requested_layers, intermediate))
        # Preserve the full 7x7 shallow patch grid from the current frame.
        shallow_patch_tokens = torch.stack(
            [layer_features[layer_idx][0] for layer_idx in self.shallow_feature_layers],
            dim=0,
        ).mean(dim=0)
        deep_patch_tokens, deep_cls_token = layer_features[self.deep_feature_layer]
        deep_encoding = self.compress_deep_obs_enc(
            self._pool_deep_features(deep_patch_tokens, deep_cls_token)
        )

        batch_size = obs_img.shape[0]
        num_obs_tokens = self.context_size + 1
        deep_encoding = deep_encoding.reshape(
            num_obs_tokens, batch_size, self.obs_encoding_size
        ).transpose(0, 1)
        shallow_patch_tokens = shallow_patch_tokens.reshape(
            num_obs_tokens, batch_size, shallow_patch_tokens.shape[1],
            shallow_patch_tokens.shape[2],
        )[-1]
        clean_astar_cond = astar_global_cond
        if clean_astar_cond is None:
            # Backward-compatible fallback for non-training callers.
            clean_astar_cond = astar_local_cond
        if clean_astar_cond is None:
            turn_risk = deep_encoding.new_zeros(batch_size)
        else:
            turn_risk = compute_path_turn_risk(
                clean_astar_cond.to(device),
                horizon=self.curvature_gate_horizon,
                stride=self.curvature_stride,
                smoothing_window=self.curvature_smoothing_window,
            )
        if self.use_shallow_visual_residual:
            obs_encoding, shallow_weight, spatial_attention = self.curvature_visual_fusion(
                shallow_patch_tokens, deep_encoding, turn_risk
            )
        else:
            obs_encoding = deep_encoding
            shallow_weight = turn_risk.new_zeros(batch_size)
            spatial_attention = shallow_patch_tokens.new_zeros(
                batch_size, shallow_patch_tokens.shape[1]
            )
        # Detached diagnostics make the learned behavior observable without
        # changing the encoder's public output contract.
        self.last_turn_risk = turn_risk.detach()
        self.last_shallow_weight = shallow_weight.detach()
        self.last_shallow_spatial_attention = spatial_attention.detach()
        obs_encoding = torch.cat((obs_encoding, floorplan_encoding), dim=1) # b, context_size+2, obs_encoding_size
        
        # Apply positional encoding 
        if self.positional_encoding:
            obs_encoding = self.positional_encoding(obs_encoding)
        obs_encoding_tokens = self.sa_encoder(obs_encoding, src_key_padding_mask=None)
        obs_encoding_tokens = torch.mean(obs_encoding_tokens, dim=1)

        obs_encoding_tokens = torch.cat([obs_encoding_tokens, obs_goal_pos_ori], dim=1)
        obs_encoding_tokens = self.obs_goal_pos_ori_enc(obs_encoding_tokens)

        return obs_encoding_tokens


# Utils for Group Norm
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module



    
