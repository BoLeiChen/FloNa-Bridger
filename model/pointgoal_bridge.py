"""RGB+floorplan PointGoal trajectory bridge: calibrated A* prior -> expert path."""
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from depth_anything_v2.dinov2 import DINOv2
from model.bridge_diffusion import BridgeConfig, DiffusionBridge
from model.vgc_rgb_condition import (compute_path_turn_risk, CurvatureResidualFusion,
                                     LocalConditionTrustGate, PositionalEncoding)
from utils.model_utils import replace_bn_with_gn

CHECKPOINT_FORMAT = 'floverse-pointgoal-world-rgb4-planner-v4'


class BridgeUNet(ConditionalUnet1D):
    """Activate both local branches without changing shared diffusion_policy files.

    Upstream's checkpoint-compatible forward never adds its second local branch.
    This new-checkpoint-only forward fixes that otherwise unused trainable branch.
    """
    def forward(self, sample, timestep, local_cond=None, global_cond=None):
        time = torch.as_tensor(timestep, device=sample.device).reshape(-1).expand(len(sample))
        condition = self.diffusion_step_encoder(time)
        if global_cond is not None:
            condition = torch.cat([condition, global_cond], -1)
        local = []
        if local_cond is not None:
            local = [block(local_cond.transpose(1, 2), condition) for block in self.local_cond_encoder]
        x, skips = sample.transpose(1, 2), []
        for index, (first, second, down) in enumerate(self.down_modules):
            x = first(x, condition)
            if index == 0 and local:
                x = x + local[0]
            x = second(x, condition)
            skips.append(x)
            x = down(x)
        for block in self.mid_modules:
            x = block(x, condition)
        for index, (first, second, up) in enumerate(self.up_modules):
            x = first(torch.cat([x, skips.pop()], 1), condition)
            x = up(second(x, condition))
            if index == len(self.up_modules) - 1 and local:
                x = x + F.interpolate(local[1], size=x.shape[-1], mode='nearest')
        return self.final_conv(x).transpose(1, 2)


def path_features(path, valid):
    """Attach the turn at each waypoint to that waypoint, in metric ego XY.

    A turn requires valid incoming/center/outgoing support. Boundary valid points
    carry a straight direction; invalid padding carries zero direction features.
    """
    sine = torch.zeros_like(valid)
    cosine = (valid > 0).to(path.dtype)
    if path.shape[1] >= 3:
        incoming = path[:, 1:-1] - path[:, :-2]
        outgoing = path[:, 2:] - path[:, 1:-1]
        usable = ((valid[:, :-2] > 0) & (valid[:, 1:-1] > 0) & (valid[:, 2:] > 0)
                  & (incoming.norm(dim=-1) > 1e-6) & (outgoing.norm(dim=-1) > 1e-6))
        turn = torch.atan2(outgoing[..., 1], outgoing[..., 0]) - torch.atan2(incoming[..., 1], incoming[..., 0])
        sine[:, 1:-1] = torch.where(usable, turn.sin(), sine[:, 1:-1])
        cosine[:, 1:-1] = torch.where(usable, turn.cos(), cosine[:, 1:-1])
    return torch.cat([path, sine[..., None], cosine[..., None], valid[..., None]], -1)


def reliability_target(path, target, valid, temperature=0.3):
    """Distance to the expert polyline, with heading agreement (metric units)."""
    start, segment = target[:, :-1], torch.diff(target, dim=1)
    relative = path[:, :, None] - start[:, None]
    fraction = ((relative * segment[:, None]).sum(-1)
                / segment.square().sum(-1)[:, None].clamp_min(1e-8)).clamp(0, 1)
    closest = start[:, None] + fraction[..., None] * segment[:, None]
    distance, index = torch.linalg.vector_norm(path[:, :, None] - closest, dim=-1).min(-1)
    tangent = segment.gather(1, index[..., None].expand(-1, -1, 2))
    path_delta = torch.diff(path, dim=1, prepend=path[:, :1])
    heading = (F.cosine_similarity(path_delta, tangent, dim=-1) + 1) * 0.5
    heading = torch.where(path_delta.norm(dim=-1) > 1e-6, heading, torch.ones_like(heading))
    return torch.exp(-distance / temperature) * (0.75 + 0.25 * heading) * valid



class PointBridgePlanner(nn.Module):
    def __init__(self, config, load_pretrained=True):
        super().__init__()
        if [float(config[k]) for k in ['point_goal_train_prob', 'img_goal_train_prob', 'object_goal_train_prob']] != [1., 0., 0.]:
            raise ValueError('This implementation requires PointGoal probabilities 1/0/0')
        if int(config['history_length']) != 3:
            raise ValueError('RGB Planner requires three history frames plus the current frame')
        if config.get('global_context_token_mode', 'floorplan') != 'floorplan':
            raise ValueError('Planner requires floorplan global condition')
        if config.get('global_coordinate_mode') != 'world':
            raise ValueError('This checkpoint format requires world global coordinates')
        self.global_coordinate_step_m = float(config.get('global_coordinate_step_m', 0.1))
        if self.global_coordinate_step_m != 0.1:
            raise ValueError('DDPM-aligned global coordinates require 0.1m units')
        self.config = config
        dim = int(config.get('condition_dim', 256))
        self.rgb_size = int(config.get('obs_image_size', 98))
        if self.rgb_size < 14 or self.rgb_size % 14:
            raise ValueError('obs_image_size must be a positive multiple of DINO patch size 14')
        self.shallow_layers = tuple(config.get('shallow_feature_layers', [2, 5]))
        self.deep_layer = int(config.get('deep_feature_layer', 11))
        if not self.shallow_layers or any(i < 0 or i > 11 for i in self.shallow_layers + (self.deep_layer,)):
            raise ValueError('DINO ViT-S layers must be in [0, 11]')
        self.obs_encoder = DINOv2('vits')
        if load_pretrained:
            checkpoint = Path(config['dinov2_checkpoint']).expanduser()
            if not checkpoint.is_file():
                raise FileNotFoundError('Depth Anything V2 checkpoint required: ' + str(checkpoint))
            weights = torch.load(str(checkpoint), map_location='cpu')
            self.obs_encoder.load_state_dict({k[len('pretrained.'):]: v for k, v in weights.items()
                                             if k.startswith('pretrained.')}, strict=True)
        # Same frozen, deterministic feature backbone as VGCNav. Trained checkpoints
        # package this backbone too; evaluation does not need the original weight file.
        self.obs_encoder.requires_grad_(False).eval()
        self.register_buffer('imagenet_mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        self.compress_deep_obs_enc = nn.Linear(768, dim)
        self.floorplan_encoder = replace_bn_with_gn(EfficientNet.from_name('efficientnet-b0', in_channels=6))
        self.floorplan_encoder._fc = nn.Identity()
        self.compress_floorplan_enc = nn.Linear(1280, dim)
        self.curvature_visual_fusion = CurvatureResidualFusion(dim, 384,
            alpha_max=float(config.get('shallow_residual_alpha_max', .25)))
        self.local_condition_gate = LocalConditionTrustGate(visual_dim=384,
            hidden_dim=int(config.get('local_gate_hidden_dim', 128)),
            coordinate_scale=float(config.get('local_gate_coordinate_scale', 3.2)))
        self.positional_encoding = PositionalEncoding(dim, max_seq_len=5)
        self.context = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            dim, int(config.get('mha_num_attention_heads', 4)), dim * 4,
            dropout=float(config.get('mha_dropout', .1)), activation='gelu', batch_first=True, norm_first=True),
            int(config.get('mha_num_attention_layers', 4)))
        # VGCNav fuses [observation XY, goal XY, heading XY] after token pooling.
        # Global coordinates match DDPM; action/local-condition geometry stays egocentric.
        self.obs_goal_pos_ori_enc = nn.Linear(dim + 6, dim)
        self.net = BridgeUNet(input_dim=2, local_cond_dim=5, global_cond_dim=dim,
                             down_dims=list(config['down_dims']), cond_predict_scale=False)
        self.bridge = DiffusionBridge(BridgeConfig(**dict(config['bridge'])))

    def train(self, mode=True):
        super().train(mode)
        self.obs_encoder.eval()
        return self

    def conditions(self, batch):
        rgb = torch.cat([batch['rgb'], batch['current_rgb'][:, None]], 1)
        if rgb.ndim != 5 or rgb.shape[1:3] != (4, 3):
            raise ValueError('Expected exactly four chronological RGB frames [B,4,3,H,W]')
        b, t = rgb.shape[:2]
        # Existing PNG/LMDB observations are identical 96px RGB in [0,1]. Resize
        # identically online/offline, then apply ImageNet normalization exactly once.
        flat = F.interpolate(rgb.flatten(0, 1), (self.rgb_size, self.rgb_size),
                             mode='bilinear', align_corners=False)
        normalized = (flat - self.imagenet_mean) / self.imagenet_std
        layers = tuple(sorted(set(self.shallow_layers + (self.deep_layer,))))
        with torch.no_grad():
            features = dict(zip(layers, self.obs_encoder.get_intermediate_layers(
                normalized, n=layers, return_class_token=True, norm=True)))
        patches, cls = features[self.deep_layer]
        deep = self.compress_deep_obs_enc(torch.cat([cls, patches.mean(1)], -1)).reshape(b, t, -1)
        shallow = torch.stack([features[i][0] for i in self.shallow_layers]).mean(0)
        shallow = shallow.reshape(b, t, shallow.shape[1], 384)[:, -1]
        clean = path_features(batch['prior'], batch['prior_valid'])
        # Use VGCNav's RGB-only gate fallback; no measured or predicted depth map.
        aux = self.local_condition_gate(shallow, clean)
        turn, confidence = compute_path_turn_risk(clean,
            horizon=int(self.config.get('curvature_gate_horizon', 20)),
            stride=int(self.config.get('curvature_stride', 3)),
            smoothing_window=int(self.config.get('curvature_smoothing_window', 3)),
            waypoint_trust=aux['local_condition_effective_gate'].detach(), return_path_confidence=True)
        deep, alpha, attention = self.curvature_visual_fusion(shallow, deep, turn, confidence)
        # VGCNav floorplan mode: current RGB + unmarked RGB floorplan -> one token.
        floor = batch['floorplan']
        if floor.shape != (b, 3, 96, 96):
            raise ValueError('Expected unmarked floorplan [B,3,96,96] in [0,1]')
        current = F.interpolate(normalized.reshape(b, t, 3, self.rgb_size, self.rgb_size)[:, -1],
                                (96, 96), mode='bilinear', align_corners=False)
        floor = (floor - self.imagenet_mean) / self.imagenet_std
        auxiliary_image = torch.cat([current, floor], dim=1)
        encoded = self.floorplan_encoder.extract_features(auxiliary_image).mean((-2, -1))
        auxiliary = self.compress_floorplan_enc(self.floorplan_encoder._dropout(encoded))[:, None]
        fused = self.context(self.positional_encoding(torch.cat([deep, auxiliary], 1))).mean(1)
        pose = batch['agent_pose']
        goal = batch['goal_world']
        if pose.shape != (b, 3) or goal.shape != (b, 2):
            raise ValueError('Expected world agent_pose [B,3] and goal_world [B,2]')
        position = pose[:, :2]
        heading = position + torch.stack([pose[:, 2].cos(), pose[:, 2].sin()], -1)
        coordinates = torch.cat([position / self.global_coordinate_step_m,
                                 goal / self.global_coordinate_step_m,
                                 heading / self.global_coordinate_step_m], -1)
        global_cond = self.obs_goal_pos_ori_enc(torch.cat([fused, coordinates], -1))
        aux.update(shallow_weight=alpha, turn_risk=turn, shallow_attention=attention)
        return aux['gated_local_cond'], global_cond, aux

    def forward(self, batch, mode='loss', energy_guidance=None):
        local, global_cond, aux = self.conditions(batch)
        if mode == 'sample':
            return self.bridge.sample(self.net, batch['prior'], local, global_cond, energy_guidance=energy_guidance)
        if mode != 'loss':
            raise ValueError(mode)
        bridge_loss, prediction = self.bridge.loss(self.net, batch['gt_pose'], batch['prior'], local, global_cond)
        valid = batch['prior_valid']
        target = reliability_target(batch['prior'], batch['gt_pose'], valid).detach()
        count = valid.sum().clamp_min(1)
        reliability = (F.binary_cross_entropy_with_logits(aux['astar_reliability_logits'], target, reduction='none') * valid).sum() / count
        risk = aux['depth_risk_target']  # Constant RGB-only fallback, not a depth observation.
        consistency = (F.smooth_l1_loss(aux['local_condition_gate'], target * (1 - risk), reduction='none') * valid).sum() / count
        residual_loss = (aux['visual_risk_residual'].square() * valid).sum() / count
        loss = bridge_loss + float(self.config.get('gate_loss_weight', .01)) * (reliability + .1 * consistency + .1 * residual_loss)
        return dict(loss=loss, bridge_loss=bridge_loss.detach(), gate_loss=reliability.detach(),
                    prediction=prediction, shallow_weight=aux['shallow_weight'].mean().detach())
