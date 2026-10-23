"""Category-conditioned DisCo adaptation: metric XY crops, no fabricated yaw.

Frozen stage-one candidates -> KDE-first XY NMS -> learned compatibility ->
validation-selected fusion. Query-local multiple positives and all-negative sets
are supervised without injecting GT candidates. This is not an image DisCo weight.
"""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .metrics import rank_candidates

FORMAT='cf2loc-object-disco-v1'
VERSION='xy-multiscale-crops-v1'

def candidate_seed(seed,key,variant=0):
    text=str(seed)+key if variant==0 else str(seed)+key+':disco:'+str(variant)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'little')%(2**63-1)

def candidates_xy(poses):
    keep,_=rank_candidates(poses,'object',ranking_mode='position')
    xy=np.asarray(poses)[:,:2]
    density=np.exp(-.5*(np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)/.75)**2).mean(1)
    return keep,density

def fused_scores(logits,density,mask,weight):
    def standardize(x):
        n=mask.sum(-1,keepdim=True).clamp_min(1)
        mean=torch.where(mask,x,0.).sum(-1,keepdim=True)/n
        z=torch.where(mask,x-mean,0.)
        return z/(z.square().sum(-1,keepdim=True)/n).sqrt().clamp_min(1e-6)
    return (weight*standardize(logits)+(1-weight)*standardize(density.clamp_min(1e-12).log())).masked_fill(~mask,-1e9)

def ranking_loss(logits,errors,mask,query_weight=None):
    """Balanced candidate BCE plus graded multi-positive ranking; misses remain negative."""
    positive=(errors<1.)&mask
    targets=torch.exp(-.5*(errors/.5).square())*positive
    targets=targets/targets.sum(-1,keepdim=True).clamp_min(1e-12)
    rank=-(targets*F.log_softmax(logits.masked_fill(~mask,-1e9),-1)).sum(-1)
    usable=positive.any(-1)
    weight=torch.ones_like(rank) if query_weight is None else query_weight
    rank=(rank*usable*weight).sum()/(usable*weight).sum().clamp_min(1e-12)
    bce=F.binary_cross_entropy_with_logits(logits,positive.float(),reduction='none')
    negative=mask&~positive
    pos=(bce*positive).sum(-1)/positive.sum(-1).clamp_min(1)
    neg=(bce*negative).sum(-1)/negative.sum(-1).clamp_min(1)
    per_query=(pos+neg)/(positive.any(-1).float()+negative.any(-1).float()).clamp_min(1)
    binary=(per_query*weight).sum()/weight.sum().clamp_min(1e-12)
    return rank+.5*binary

def metric_crops(floor,xy,origin,extent,sizes=(4.,12.),pixels=32):
    """World XY -> image XY. Axis-aligned crops; white padding outside map.
    floor [B,3,H,W], xy [B,K,2], origin/extent [B,2]. align_corners=False
    uses map cell centers exactly; no vertical flip and no yaw assumed.
    """
    b,k=xy.shape[:2];axis=(torch.arange(pixels,device=xy.device,dtype=xy.dtype)+.5)/pixels-.5
    yy,xx=torch.meshgrid(axis,axis,indexing='ij');offset=torch.stack([xx,yy],-1)
    source=floor.mean(1,keepdim=True)-1.;crops=[]
    for size in sizes:
        world=xy[:,:,None,None]+offset[None,None]*size
        grid=2*(world-origin[:,None,None,None])/extent[:,None,None,None]-1
        crop=F.grid_sample(source,grid.reshape(b,k*pixels,pixels,2),align_corners=False,padding_mode='zeros')+1
        crops.append(crop.reshape(b,1,k,pixels,pixels).transpose(1,2))
    return torch.cat(crops,2)

class ObjectDisCo(nn.Module):
    def __init__(self,dim=96):
        super().__init__()
        def encoder(channels):return nn.Sequential(nn.Conv2d(channels,32,5,2,2),nn.GroupNorm(8,32),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.GroupNorm(8,64),nn.GELU(),nn.Conv2d(64,dim,3,2,1),nn.GroupNorm(8,dim),nn.GELU())
        self.local=encoder(2);self.global_map=encoder(1)
        self.text=nn.Sequential(nn.Linear(512,dim),nn.LayerNorm(dim),nn.GELU())
        self.position=nn.Linear(2,dim)
        self.match=nn.MultiheadAttention(dim,4,batch_first=True,dropout=.1)
        self.head=nn.Sequential(nn.LayerNorm(dim*3),nn.Linear(dim*3,dim),nn.GELU(),nn.Dropout(.1),nn.Linear(dim,1))
    def forward(self,batch):
        xy=batch['xy'];b,k=xy.shape[:2]
        crops=metric_crops(batch['floor'],xy,batch['origin'],batch['extent'])
        local=self.local(crops.flatten(0,1)).flatten(2).transpose(1,2)
        side=int(local.shape[1]**.5);a=(torch.arange(side,device=xy.device,dtype=xy.dtype)+.5)/side*2-1
        yy,xx=torch.meshgrid(a,a,indexing='ij');local=local+self.position(torch.stack([xx,yy],-1).reshape(1,-1,2))
        text=self.text(batch['text']);query=text[:,None,None].expand(-1,k,1,-1).reshape(b*k,1,-1)
        matched,_=self.match(query,local,local,need_weights=False)
        global_feature=self.global_map(F.interpolate(batch['floor'].mean(1,keepdim=True),(64,64),mode='bilinear',align_corners=False)).mean((-1,-2))
        joined=torch.cat([matched[:,0],query[:,0],global_feature[:,None].expand(-1,k,-1).reshape(b*k,-1)],-1)
        return self.head(joined).reshape(b,k)

class ObjectDisCoLocalizer:
    """GT-free inference API: category, map and world calibration only."""
    def __init__(self,diffusion_checkpoint,disco_checkpoint,device='cpu',backbone_paths=None):
        from .api import Localizer
        from .encoders import file_hash
        saved=torch.load(disco_checkpoint,map_location='cpu')
        if saved.get('format')!=FORMAT:raise ValueError('Expected object DisCo checkpoint')
        if file_hash(diffusion_checkpoint)!=saved['diffusion_sha256']:raise ValueError('Frozen diffusion checkpoint mismatch')
        self.base=Localizer(diffusion_checkpoint,device,backbone_paths=backbone_paths)
        if self.base.config['setting']!='object_only':raise ValueError('Requires object_only diffusion')
        self.device=torch.device(device);self.weight=float(saved['fusion_weight'])
        self.ranker=ObjectDisCo().to(device).eval();self.ranker.load_state_dict(saved['model'],strict=True)
    @torch.inference_mode()
    def predict(self,floorplan,*,origin_xy,extent_xy,category,top_k=16,seed=0):
        from .encoders import image_tensor
        from .prepare import category_name
        if not 1<=top_k<=64:raise ValueError('Top K must be in [1,64]')
        raw=self.base.predict(floorplan,origin_xy=origin_xy,extent_xy=extent_xy,category=category,num_candidates=64,top_k=64,seed=seed)
        poses=np.column_stack([raw['raw_world_xy'],np.zeros(64)])
        ids,density=candidates_xy(poses);xy=poses[ids,:2].astype('float32')
        def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=self.device)[None]
        batch=dict(floor=image_tensor(floorplan,256)[None].to(self.device),xy=tensor(xy),origin=tensor(origin_xy),extent=tensor(extent_xy),text=self.base.features.text([category_name(category)]))
        logits=self.ranker(batch);den=tensor(density[ids]);mask=torch.ones_like(logits,dtype=torch.bool)
        scores=fused_scores(logits,den,mask,self.weight)[0];order=torch.argsort(scores,descending=True,stable=True)[:top_k].cpu().numpy()
        return dict(world_xy=xy[order].tolist(),world_yaw=None,ids=ids[order].tolist(),scores=scores.cpu().numpy()[order].tolist(),density=density[ids[order]].tolist(),match_logits=logits[0].cpu().numpy()[order].tolist(),requested_top_k=top_k,returned_top_k=len(order),confidence_kind='uncalibrated_object_disco_fusion',requires_reachability_check=True)
