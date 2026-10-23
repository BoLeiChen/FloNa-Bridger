"""Goal-sensitive route policy with a frozen, independently trained PointGoal bridge."""
import torch
from torch import nn
import torch.nn.functional as F
from model.pointgoal_bridge import PointBridgePlanner, CHECKPOINT_FORMAT as CONTROLLER_FORMAT
from model.multimodal_path_bridge import valid_denominator

CHECKPOINT_FORMAT='floverse-imagegoal-path-distribution-v2'

class GoalMatchingRouteScorer(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.map_encoder=nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.GELU())
        self.route_encoder=nn.Sequential(nn.Linear(69,dim),nn.GELU(),nn.Linear(dim,dim))
        self.route_summary=nn.Linear(dim*2,dim)
        self.stats_encoder=nn.Sequential(nn.Linear(4,dim),nn.GELU(),nn.Linear(dim,dim))
        self.visual_projection=nn.Linear(384,dim)
        self.spatial_projection=nn.Linear(2,dim)
        self.frame_embedding=nn.Embedding(4,dim)
        self.obs_goal_attention=nn.MultiheadAttention(dim,4,batch_first=True,dropout=0.)
        self.goal_match_norm=nn.LayerNorm(dim)
        self.route_attention=nn.MultiheadAttention(dim,4,batch_first=True,dropout=0.)
        self.score=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,1))
    def forward(self,batch,observations,goal):
        # Frozen DINO deep patches retain spatial structure; goal and observations
        # share the trainable projection. Explicit obs->goal attention precedes routing.
        b,t,n,_=observations.shape;side=int((n-1)**.5)
        if side*side!=n-1:raise ValueError('Expected class token plus square patch grid')
        axis=torch.linspace(-1,1,side,device=observations.device,dtype=observations.dtype)
        yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        xy=torch.cat([observations.new_zeros(1,2),torch.stack([xx,yy],-1).reshape(-1,2)],0)
        spatial=self.spatial_projection(xy)
        obs=self.visual_projection(observations)+spatial[None,None]+self.frame_embedding.weight[None,:,None]
        obs=obs.reshape(b,t*n,-1)
        goal=self.visual_projection(goal)+(spatial[None] if goal.shape[1]==spatial.shape[0] else spatial.new_zeros(1,1,spatial.shape[-1]))
        matched,_=self.obs_goal_attention(obs,goal,goal,need_weights=False)
        contextual=self.goal_match_norm(obs+matched)
        memory=torch.cat([contextual,goal.mean(1,keepdim=True)],1)
        fmap=self.map_encoder(batch['floorplan'])
        along=F.grid_sample(fmap,batch['candidate_grid'],align_corners=False).permute(0,2,3,1)
        routes=batch['candidate_routes'];order=torch.linspace(0,1,routes.shape[2],device=routes.device)[None,None,:,None].expand(*routes.shape[:-1],1)
        points=self.route_encoder(torch.cat([routes/10,batch['candidate_grid'],along,order],-1))
        tokens=self.route_summary(torch.cat([points.mean(2),points[:,:,-1]],-1))+self.stats_encoder(batch['candidate_stats'])
        attended,_=self.route_attention(tokens,memory,memory,need_weights=False)
        return self.score(tokens+attended).squeeze(-1)

class PathDistributionV2(PointBridgePlanner):
    def __init__(self,config,load_pretrained=True):
        if [float(config[k]) for k in ('point_goal_train_prob','img_goal_train_prob','object_goal_train_prob')]!=[0.,1.,0.]:
            raise ValueError('V2 is ImageGoal-only; ObjectGoal is not implemented')
        parent=dict(config,point_goal_train_prob=1.,img_goal_train_prob=0.)
        super().__init__(parent,load_pretrained=False)
        if load_pretrained:
            saved=torch.load(config['controller_checkpoint'],map_location='cpu')
            if saved['format']!=CONTROLLER_FORMAT:raise ValueError('Expected aligned RGB4 PointGoal controller')
            for key in ('history_length','len_traj_pred','global_coordinate_mode','global_coordinate_step_m'):
                if saved['config'][key]!=config[key]:raise ValueError('Controller mismatch: '+key)
            if saved['config']['bridge']['action_step_m']!=config['bridge']['action_step_m']:raise ValueError('Action scale mismatch')
            self.load_state_dict(saved['planner'],strict=True)
        self.requires_grad_(False)
        self.config=config;self.scorer=GoalMatchingRouteScorer(int(config['condition_dim']));self.epoch=0
        self.train(False)
    def train(self,mode=True):
        # Frozen controller must remain deterministic even while route head trains.
        super().train(False)
        if hasattr(self,'scorer'):self.scorer.train(mode)
        return self
    def visual_features(self,batch,include_alternate=False,include_goal=True):
        rgb=torch.cat([batch['rgb'],batch['current_rgb'][:,None]],1)
        if rgb.shape[1:3]!=(4,3):raise ValueError('Expected four RGB frames')
        if include_alternate and not include_goal:raise ValueError('Alternate goals require goal images')
        goals=[batch['goal_rgb']] if include_goal else []
        if include_alternate:goals.append(batch['alternate_goal_rgb'])
        images=torch.cat([rgb,torch.stack(goals,1)],1) if goals else rgb;b,t=images.shape[:2]
        images=F.interpolate(images.flatten(0,1),(self.rgb_size,self.rgb_size),mode='bilinear',align_corners=False)
        normalized=(images-self.imagenet_mean)/self.imagenet_std
        layers=tuple(sorted(set(self.shallow_layers+(self.deep_layer,))))
        with torch.no_grad():
            features=dict(zip(layers,self.obs_encoder.get_intermediate_layers(normalized,n=layers,return_class_token=True,norm=True)))
            patches,cls=features[self.deep_layer]
            patches=patches.reshape(b,t,patches.shape[1],384);cls=cls.reshape(b,t,384)
            raw=torch.cat([cls[:,:,None],patches],2)
            deep=self.compress_deep_obs_enc(torch.cat([cls[:,:4],patches[:,:4].mean(2)],-1))
            shallow=torch.stack([features[i][0] for i in self.shallow_layers]).mean(0)
            shallow=shallow.reshape(b,t,shallow.shape[1],384)[:,3]
            normalized_obs=normalized.reshape(b,t,3,self.rgb_size,self.rgb_size)[:,:4].reshape(b*4,3,self.rgb_size,self.rgb_size)
        return raw[:,:4],raw[:,4:],(b,4,normalized_obs,deep,shallow)
    def distribution(self,batch,observations,goal):
        logits=self.scorer(batch,observations,goal)
        logits=logits-float(self.config.get('route_revisit_penalty',.25))*batch['candidate_stats'][...,1]-float(self.config.get('route_failure_penalty',1.))*batch['candidate_stats'][...,3]
        mask=batch['candidate_mask'].bool();safe=mask.clone();safe[~mask.any(1),0]=True
        logits=logits.masked_fill(~safe,-1e4)
        return logits.softmax(-1),logits
    def score_paths(self,batch):
        if not (batch['task_type']==1).all():raise ValueError('V2 policy only supports ImageGoal')
        observed,goals,rgb=self.visual_features(batch)
        probability,logits=self.distribution(batch,observed,goals[:,0])
        return probability,logits,None,rgb
    def forward(self,batch,mode='loss',selected_index=None,prepared=None):
        if mode=='loss':
            observations,goals,_=self.visual_features(batch,True)
            probability,logits=self.distribution(batch,observations,goals[:,0])
            alternate,other_logits=self.distribution(batch,observations,goals[:,1])
            target=batch['route_target'];other=batch['alternate_route_target'];valid=target.sum(-1)>0;other_valid=other.sum(-1)>0
            logp=F.log_softmax(logits,-1);logq=F.log_softmax(other_logits,-1)
            ce=(-(target*logp).sum(-1)*valid).sum()/valid_denominator(valid)
            alt_ce=(-(other*logq).sum(-1)*other_valid).sum()/valid_denominator(other_valid)
            pair=batch['goal_pair_valid'].bool()&valid&other_valid
            # Correct goal should improve its own route-label likelihood over the alternate.
            preference=((target-other)*(logp-logq)).sum(-1)
            margin=float(self.config.get('goal_pair_margin',.25))
            contrast=(F.relu(margin-preference)*pair).sum()/valid_denominator(pair)
            route_loss=.5*(ce+alt_ce)+float(self.config.get('goal_pair_loss_weight',.1))*contrast
            rows=torch.arange(len(probability),device=probability.device);index=probability.argmax(-1)
            zero=route_loss.detach()*0
            return dict(loss=route_loss,route_loss=route_loss.detach(),bridge_loss=zero,gate_loss=zero,shallow_weight=zero,
                bridge_coverage=batch['bridge_compatible'].any(-1).float().mean(),label_coverage=valid.float().mean(),
                predicted_compatible=batch['bridge_compatible'][rows,index].float().mean(),
                route_entropy=-(probability*probability.clamp_min(1e-8).log()).sum(-1).mean().detach(),
                goal_pair_rate=pair.float().mean(),goal_pair_switch=((index!=alternate.argmax(-1))&pair).float().sum()/pair.float().sum().clamp_min(1),
                goal_pair_contrast=contrast.detach(),
                goal_pair_advantage=(preference.detach()*pair).sum()/pair.float().sum().clamp_min(1),
                goal_pair_margin_satisfied=((preference.detach()>=margin)&pair).float().sum()/pair.float().sum().clamp_min(1))
        probability,logits,_,rgb=self.score_paths(batch) if prepared is None else prepared
        if mode=='score':return dict(probabilities=probability,logits=logits)
        if mode!='sample':raise ValueError(mode)
        index=probability.argmax(-1) if selected_index is None else selected_index
        if index.shape!=probability.shape[:1] or ((index<0)|(index>=probability.shape[1])).any():raise ValueError('Invalid route selection')
        rows=torch.arange(len(index),device=index.device);available=batch['candidate_mask'].any(-1)
        if (available&~batch['candidate_mask'][rows,index]).any():raise ValueError('Padded selected route')
        chosen=dict(batch,prior=batch['candidate_priors'][rows,index],prior_valid=batch['candidate_valid'][rows,index],goal_world=batch['candidate_goals'][rows,index])
        local,condition,_=super().conditions(chosen,rgb)
        trajectory=self.bridge.sample(self.net,chosen['prior'],local,condition)
        return torch.where(available[:,None,None],trajectory,torch.zeros_like(trajectory))
