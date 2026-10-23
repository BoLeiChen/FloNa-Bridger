"""HPR-inspired query-local DisCo ranking; existing CF2Loc image features stay frozen.

No GT is accepted by the inference API. The matcher sees only image tokens and
oriented metric map crops. Coordinate conventions remain those of its parent
CF2Loc checkpoint. This is an adaptation, not a drop-in HPR-Loc checkpoint.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def diverse_modes(samples, density, map_res=0.02, top_k=5,
                  position_radius=1.0, yaw_radius_deg=30.0):
    """HPR elliptical SE(2) NMS; never fill remaining slots with duplicates."""
    if samples.ndim != 2 or samples.shape[1] != 3 or len(samples) != len(density):
        raise ValueError('Expected poses [N,3] and density [N]')
    if not torch.isfinite(samples).all() or not torch.isfinite(density).all():
        raise ValueError('Nonfinite diffusion candidates')
    chosen = []
    for idx in torch.argsort(density, descending=True, stable=True).tolist():
        if chosen:
            other = samples[chosen]
            xy = torch.linalg.vector_norm((other[:, :2] - samples[idx, :2]) * map_res, dim=1)
            yaw = torch.abs(torch.remainder(other[:, 2] - samples[idx, 2] + math.pi, 2*math.pi)-math.pi)
            if torch.any((xy/position_radius)**2 + (yaw/math.radians(yaw_radius_deg))**2 <= 1):
                continue
        chosen.append(idx)
        if len(chosen) == top_k:
            break
    return torch.tensor(chosen, device=samples.device, dtype=torch.long)


def fused_scores(scores, density, valid, weight=0.9):
    """Per-query HPR z-score fusion. Padding has no influence, even for K=1."""
    def standardize(x):
        count = valid.sum(-1, keepdim=True).clamp_min(1)
        x = torch.where(valid, x, torch.zeros_like(x))
        mean = x.sum(-1, keepdim=True)/count
        centered = torch.where(valid, x-mean, torch.zeros_like(x))
        std = (centered.square().sum(-1, keepdim=True)/count).sqrt().clamp_min(1e-6)
        return centered/std
    result = weight*standardize(scores.float()) + (1-weight)*standardize(density.float().clamp_min(1e-12).log())
    return result.masked_fill(~valid, -1e9)


def matching_targets(xy_error, yaw_error, valid):
    """All near-correct candidates are positives, with smoothly graded quality.

    Candidate-local loss avoids treating neighboring training frames as negatives.
    Every training query must include a GT-near positive (never added at validation).
    """
    positive = valid & (xy_error < 1.0) & (yaw_error < math.radians(30))
    weights = torch.exp(-0.5*(xy_error/0.5)**2 - 0.5*(yaw_error/math.radians(20))**2) * positive
    return weights/weights.sum(-1, keepdim=True).clamp_min(1e-12), positive.any(-1)


def ranking_loss(scores, xy_error, yaw_error, valid):
    targets, usable = matching_targets(xy_error, yaw_error, valid)
    if not bool(usable.all()):
        raise ValueError('Training query has no positive candidate')
    logits = scores.float().masked_fill(~valid, -1e9)
    return -(targets * F.log_softmax(logits, dim=-1)).sum(-1).mean()


def crop_maps(raw_map, poses, config):
    from third_party.cf2loc_hm3d.DisCo_model.pose_local_refiner import crop_to_refiner_tensor
    return torch.stack([crop_to_refiner_tensor(
        raw_map, np.asarray(p, dtype=np.float32), config['crop_m'],
        config['map_res'], config['crop_size'], 'gray', oriented=True,
    ) for p in poses])


class DisCoMatcher(nn.Module):
    def __init__(self, config):
        super().__init__()
        from third_party.cf2loc_hm3d.DisCo_model.floorplan_encoder import ResNetFloorplanEncoder
        self.config = dict(config)
        dim = config.get('feature_dim', 128)
        self.local_map_encoder = ResNetFloorplanEncoder(dim, 'gray', 2, False)
        self.map_pos = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.map_norm = nn.LayerNorm(dim)
        self.image_adapter = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
        self.cross_attn = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.token_head = nn.Sequential(nn.Linear(dim, dim//2), nn.GELU(), nn.Linear(dim//2, 1))
        self.score_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def initialize_map_from_refiner(self, checkpoint):
        prefix = 'refiner.local_map_encoder.'
        state = checkpoint['state_dict']
        self.local_map_encoder.load_state_dict({k[len(prefix):]: v for k,v in state.items() if k.startswith(prefix)}, strict=True)

    def train(self, mode=True):
        super().train(mode)
        # Legacy checkpoints used pretrained map features with frozen BN statistics.
        # A newly initialized encoder learns its running statistics during training.
        if self.config.get('map_encoder_initialization', 'refiner') == 'refiner':
            for module in self.local_map_encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, image_tokens, maps):
        # image_tokens [B,256,128]; maps [B,K,1,128,128]. No pose/GT/density input.
        b, k = maps.shape[:2]
        feature = F.adaptive_avg_pool2d(self.local_map_encoder(maps.flatten(0,1)), (8,8))
        tokens = feature.flatten(2).transpose(1,2)
        grid = torch.linspace(-0.5, 0.5, 8, device=tokens.device, dtype=tokens.dtype)
        y,x = torch.meshgrid(grid, grid, indexing='ij')
        tokens = self.map_norm(tokens + self.map_pos(torch.stack((x,y),-1).reshape(1,64,2)))
        query = self.image_adapter(image_tokens).unsqueeze(1).expand(-1,k,-1,-1).flatten(0,1)
        aligned,_ = self.cross_attn(query, tokens, tokens, need_weights=False)
        aligned = self.cross_norm(query+aligned)
        pooled = (aligned * self.token_head(aligned).softmax(dim=1)).sum(dim=1)
        return self.score_head(pooled).view(b,k)


class DisCoRanker:
    def __init__(self, checkpoint, diffusion_sha256, device, map_res=0.02):
        state = torch.load(checkpoint, map_location='cpu')
        if state.get('format') != 'floverse_disco_v1':
            raise ValueError('Unsupported DisCo checkpoint')
        self.config = state['config']
        if self.config['diffusion_sha256'] != diffusion_sha256:
            raise ValueError('DisCo image encoder/diffusion checkpoint mismatch')
        if abs(self.config['map_res']-map_res) > 1e-9:
            raise ValueError('DisCo map resolution mismatch')
        self.weight = float(state['fusion_weight'])
        self.device = device
        self.model = DisCoMatcher(self.config).to(device).eval()
        self.model.load_state_dict(state['model'], strict=True)

    @torch.inference_mode()
    def rank(self, image_tokens, samples, density, raw_map):
        c = self.config
        ids = diverse_modes(samples, density, c['map_res'], c['modes'], c['nms_m'], c['nms_deg'])
        maps = crop_maps(raw_map, samples[ids].cpu().numpy(), c).unsqueeze(0).to(self.device)
        # Match the frozen fp16 feature cache and bf16 validation computation.
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            logits = self.model(image_tokens.half().float(), maps)
        scores = fused_scores(logits, density[ids][None], torch.ones_like(logits, dtype=torch.bool), self.weight)
        order = torch.argsort(scores[0], descending=True, stable=True)
        return samples[ids[order]], [
            {'candidate_count': len(ids), 'candidate_index': int(ids[i]),
             'match_logit': float(logits[0,i]), 'fusion_weight': self.weight,
             'fusion_score': float(scores[0,i])} for i in order.tolist()]

    @torch.inference_mode()
    def select(self, image_tokens, samples, density, raw_map):
        poses, info = self.rank(image_tokens, samples, density, raw_map)
        return poses[0], info[0]
