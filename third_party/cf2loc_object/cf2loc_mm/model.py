"""Multimodal extension of CF2Loc's metric pose-query diffusion localizer."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .vendor.floorplan import ResNetFloorplanEncoder
from .vendor.pose_query import PoseQueryDenoiser, cosine_beta_schedule, build_cell_center_coordinates

FORMAT='cf2loc-multimodal-metric-v1'
SETTINGS=('object_only','joint_clip','joint_geometry')

def group_norm(module):
    for name,child in module.named_children():
        if isinstance(child,nn.BatchNorm2d):setattr(module,name,nn.GroupNorm(min(8,child.num_features),child.num_features))
        else:group_norm(child)
    return module

class MultimodalLocalizer(nn.Module):
    def __init__(self, config):
        super().__init__(); self.config=dict(config);self.setting=config['setting']
        if self.setting not in SETTINGS:raise ValueError(self.setting)
        d=int(config.get('dim',128));heads=int(config.get('heads',4));drop=float(config.get('dropout',.1))
        self.map_encoder=group_norm(ResNetFloorplanEncoder(d,'gray',context_blocks=2,pretrained=False))
        self.map_grid=int(config.get('map_grid',16));self.position=nn.Sequential(nn.Linear(2,d),nn.GELU(),nn.Linear(d,d))
        self.text_projection=nn.Sequential(nn.Linear(int(config.get('clip_dim',512)),d),nn.LayerNorm(d))
        self.task_embedding=nn.Embedding(2,d)
        if self.setting!='object_only':
            self.image_projection=nn.Sequential(nn.Linear(int(config.get('clip_dim',512)),d),nn.LayerNorm(d))
            self.image_position=nn.Sequential(nn.Linear(2,d),nn.GELU(),nn.Linear(d,d))
        if self.setting=='joint_geometry':self.geometry_projection=nn.Sequential(nn.Linear(384,d),nn.LayerNorm(d))
        self.token_types=nn.Embedding(3,d)
        self.query_mixer=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,heads,d*4,drop,activation='gelu',batch_first=True,norm_first=True),1)
        self.cross=nn.MultiheadAttention(d,heads,dropout=drop,batch_first=True)
        self.cross_norm=nn.LayerNorm(d);self.ffn=nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d));self.norm=nn.LayerNorm(d)
        cfg=dict(diffusion_feature_dim=d,diffusion_num_heads=heads,diffusion_dropout=drop,diffusion_denoiser_layers=int(config.get('denoiser_layers',2)),diffusion_coordinate_convention='metric_cell_center_v1',diffusion_train_steps=int(config.get('diffusion_steps',1000)))
        self.denoiser=PoseQueryDenoiser(cfg)
        self.register_buffer('alpha_bar',torch.cumprod(1-cosine_beta_schedule(cfg['diffusion_train_steps']),0))

    def context(self,batch):
        task=batch['task'];floor=batch['floorplan'];task_id=0 if task=='object' else 1
        if task=='object':q=self.text_projection(batch['text_features'])[:,None]+self.token_types.weight[0]
        elif task=='image' and self.setting!='object_only':
            q=self.image_projection(batch['image_features'])+self.token_types.weight[1]
            side=math.isqrt(q.shape[1]-1)
            if side*side!=q.shape[1]-1:raise ValueError('Expected CLS plus a square patch grid')
            pos=self.image_position(build_cell_center_coordinates(side,side,q.device,q.dtype))[None]
            q=torch.cat([q[:,:1],q[:,1:]+pos],1)
            if self.setting=='joint_geometry':
                g=self.geometry_projection(batch['geometry_features'])+self.token_types.weight[2]
                if g.shape[1]!=side*side:raise ValueError('Geometry and semantic grids differ')
                q=torch.cat([q,g+pos],1)
        else:raise ValueError('Unsupported query task for this checkpoint: '+task)
        q=q+self.task_embedding.weight[task_id];q=self.query_mixer(q)
        fmap=F.adaptive_avg_pool2d(self.map_encoder(floor),(self.map_grid,self.map_grid))
        coords=build_cell_center_coordinates(self.map_grid,self.map_grid,fmap.device,fmap.dtype)
        tokens=fmap.flatten(2).transpose(1,2)+self.position(coords)[None]
        attended,_=self.cross(tokens,q,q,need_weights=False)
        tokens=self.cross_norm(tokens+attended);tokens=self.norm(tokens+self.ffn(tokens))
        return tokens,coords,q.mean(1)

    def _loss(self,batch):
        context=self.context(batch);targets=batch['targets'];mask=batch['target_mask'];b=targets.shape[0];p=int(self.config.get('train_particles',8))
        if not mask.any(1).all():raise ValueError('Each query needs at least one positive target')
        # Every particle samples a valid positive independently: no mode averaging,
        # no arbitrary first-instance label for class-only queries.
        ids=torch.multinomial(mask.float(),p,replacement=True)
        if 'forced_target' in batch:
            forced=batch['forced_target'];active=forced>=0
            if active.any():
                if (forced[active]>=mask.shape[1]).any() or not mask[active].gather(1,forced[active,None]).all():raise ValueError('Invalid scheduled positive')
                ids[active,0]=forced[active]
        target=targets.gather(1,ids[...,None].expand(-1,-1,3))
        xy=2*(target[...,:2]-batch['origin'][:,None])/batch['extent'][:,None]-1
        if (xy.abs()>1.0001).any():raise ValueError('Target outside calibrated map')
        state=torch.cat([xy,target[...,2].sin()[...,None],target[...,2].cos()[...,None]],-1)
        object_task=batch['task']=='object'
        if object_task:state[...,2]=0;state[...,3]=1
        t=torch.randint(len(self.alpha_bar),(b,p),device=state.device);a=self.alpha_bar[t][...,None]
        noise=torch.randn_like(state);noisy=a.sqrt()*state+(1-a).sqrt()*noise
        if object_task:noisy=torch.cat([noisy[...,:2],state[...,2:]],-1)
        pred=self.denoiser(noisy,t,*context,batch['extent'],1.)
        xy_loss=F.mse_loss(pred[...,:2],noise[...,:2]);yaw=pred[...,2:].sum()*0
        if not object_task:yaw=F.mse_loss(pred[...,2:],noise[...,2:])
        return xy_loss+float(self.config.get('yaw_weight',1.))*yaw

    def forward(self,batch):
        # One DDP forward per update, so both tasks participate in one reduction.
        if isinstance(batch,list):
            losses=[self._loss(x) for x in batch]
            return losses[0]+float(self.config.get('image_loss_weight',1.))*losses[1],torch.stack([x.detach() for x in losses])
        loss=self._loss(batch);return loss,loss.detach()[None]

    @torch.no_grad()
    def sample(self,batch,num_candidates=64,steps=20,generator=None):
        if not 2<=steps<=len(self.alpha_bar) or num_candidates<1:raise ValueError('Invalid sampling settings')
        context=self.context(batch);b=len(batch['extent']);device=batch['extent'].device
        state=torch.randn((b,num_candidates,4),device=device,generator=generator)
        object_task=batch['task']=='object'
        ts=torch.linspace(len(self.alpha_bar)-1,0,steps,device=device).long()
        kv=self.denoiser.build_map_kv_cache(context[0])
        for i,t in enumerate(ts):
            if object_task:state[...,2]=0;state[...,3]=1
            times=t.expand(b,num_candidates);eps=self.denoiser(state,times,*context,batch['extent'],1.,map_kv_cache=kv)
            a=self.alpha_bar[t];x0=(state-(1-a).sqrt()*eps)/a.sqrt().clamp_min(1e-6)
            angle=F.normalize(x0[...,2:],dim=-1,eps=1e-6)
            x0=torch.cat([x0[...,:2].clamp(-1,1),angle],-1)
            if object_task:x0[...,2]=0;x0[...,3]=1
            if i+1==len(ts):state=x0
            else:
                ap=self.alpha_bar[ts[i+1]];state=ap.sqrt()*x0+(1-ap).sqrt()*eps
        xy=(state[...,:2]+1)*.5*batch['extent'][:,None]+batch['origin'][:,None]
        yaw=torch.atan2(state[...,2],state[...,3])
        # Object yaw is deliberately not an inferred arrival orientation.
        if object_task:yaw=torch.full_like(yaw,float('nan'))
        return torch.cat([xy,yaw[...,None]],-1)
