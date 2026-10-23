"""Task-conditioned discrete route distribution + shared local denoising bridge."""
import torch
from torch import nn
import torch.nn.functional as F
from model.pointgoal_bridge import PointBridgePlanner, reliability_target
from model.bridge_diffusion import encode_trajectory

CHECKPOINT_FORMAT='floverse-multimodal-path-distribution-v1'


def valid_denominator(mask):
    """DDP averages gradients: normalize over valid examples across all ranks."""
    count=mask.detach().float().sum()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(count)
        return count.clamp_min(1)/torch.distributed.get_world_size()
    return count.clamp_min(1)



class RouteScorer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.map_encoder=nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.GELU())
        self.route_encoder=nn.Sequential(nn.Linear(66,dim),nn.GELU(),nn.Linear(dim,dim))
        self.stats_encoder=nn.Sequential(nn.Linear(4,dim),nn.GELU(),nn.Linear(dim,dim))
        self.attention=nn.MultiheadAttention(dim,4,dropout=0.,batch_first=True)
        self.score=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim//2),nn.GELU(),nn.Linear(dim//2,1))

    def forward(self,batch,observations,task):
        fmap=self.map_encoder(batch['floorplan'])
        along=F.grid_sample(fmap,batch['candidate_grid'],mode='bilinear',padding_mode='zeros',align_corners=False)
        # [B,C,K,L] -> [B,K,L,C]; local geometry and spatial map features remain paired.
        tokens=self.route_encoder(torch.cat([batch['candidate_routes']/10,along.permute(0,2,3,1)],-1)).mean(2)
        tokens=tokens+self.stats_encoder(batch['candidate_stats'])
        memory=torch.cat([observations,task[:,None]],1)
        attended,_=self.attention(tokens,memory,memory,need_weights=False)
        return self.score(tokens+attended).squeeze(-1)


class MultimodalPathBridge(PointBridgePlanner):
    def __init__(self,config,load_pretrained=True):
        if float(config.get('object_goal_train_prob',0))!=0:
            raise ValueError('Stage one has no ObjectGoal training')
        parent=dict(config,point_goal_train_prob=1.,img_goal_train_prob=0.,object_goal_train_prob=0.)
        super().__init__(parent,load_pretrained)
        self.config=config
        dim=int(config['condition_dim'])
        self.goal_image_projection=nn.Linear(768,dim)
        self.goal_point_projection=nn.Sequential(nn.Linear(2,dim),nn.GELU(),nn.Linear(dim,dim))
        self.modality=nn.Embedding(2,dim)
        self.goal_norm=nn.LayerNorm(dim)
        self.scorer=RouteScorer(dim)
        self.task_condition=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,dim))
        nn.init.zeros_(self.task_condition[-1].weight)
        nn.init.zeros_(self.task_condition[-1].bias)
        self.epoch=0

    def task_embedding(self,batch):
        image=F.interpolate(batch['goal_rgb'],(self.rgb_size,self.rgb_size),mode='bilinear',align_corners=False)
        with torch.no_grad():
            feat=self.obs_encoder.forward_features((image-self.imagenet_mean)/self.imagenet_std)
            visual=torch.cat([feat['x_norm_clstoken'],feat['x_norm_patchtokens'].mean(1)],-1)
        image=self.goal_image_projection(visual)
        pose=batch['agent_pose'];delta=batch['task_point_world']-pose[:,:2]
        co,si=pose[:,2].cos(),pose[:,2].sin()
        relative=torch.stack([co*delta[:,0]+si*delta[:,1],-si*delta[:,0]+co*delta[:,1]],-1)/10
        point=self.goal_point_projection(relative)
        task=batch['task_type']
        if not ((task==0)|(task==1)).all():raise ValueError('Unknown goal modality')
        return self.goal_norm(torch.where((task==1)[:,None],image,point)+self.modality(task))

    def score_paths(self,batch):
        rgb_features=self.encode_rgb(batch)
        task=self.task_embedding(batch)
        observed=self.positional_encoding(rgb_features[3])
        logits=self.scorer(batch,observed,task)
        memory_cost=(float(self.config.get('route_revisit_penalty',.25))*batch['candidate_stats'][...,1]
                     +float(self.config.get('route_failure_penalty',1.))*batch['candidate_stats'][...,3])
        logits=logits-(batch['task_type']==1)[:,None]*memory_cost
        mask=batch['candidate_mask'].bool()
        safe=mask.clone();safe[~mask.any(1),0]=True
        logits=logits.masked_fill(~safe,-1e4)
        probability=logits.softmax(-1)
        return probability,logits,task,rgb_features

    def selected_conditions(self,batch,index,task,rgb_features):
        rows=torch.arange(len(index),device=index.device)
        chosen=dict(batch,prior=batch['candidate_priors'][rows,index],
                    prior_valid=batch['candidate_valid'][rows,index],goal_world=batch['candidate_goals'][rows,index])
        local,condition,aux=super().conditions(chosen,rgb_features)
        return chosen,local,condition+self.task_condition(task),aux

    def forward(self,batch,mode='loss',selected_index=None,prepared=None):
        probability,logits,task,rgb_features=self.score_paths(batch) if prepared is None else prepared
        predicted=probability.argmax(-1)
        if mode=='score':return dict(probabilities=probability,logits=logits)
        index=predicted if selected_index is None else selected_index
        if index.shape != predicted.shape or ((index<0)|(index>=probability.shape[1])).any():
            raise ValueError('Invalid selected route index')
        if selected_index is not None:
            rows=torch.arange(len(index),device=index.device)
            if (batch['candidate_mask'].any(1) & ~batch['candidate_mask'][rows,index]).any():
                raise ValueError('Selected route is padded or unreachable')
        if mode=='loss':
            weights=batch['route_target'];covered=weights.sum(-1)>0
            rank_loss=(-(weights*F.log_softmax(logits,-1)).sum(-1)*covered).sum()/valid_denominator(covered)
            compatible=batch['bridge_compatible'].bool() & batch['candidate_mask'].bool()
            # Teacher selects ONLY among compatible naturally generated candidates.
            teacher=weights.masked_fill(~compatible,-1).argmax(-1)
            rows=torch.arange(len(index),device=index.device)
            ratio=float(self.config.get('predicted_pair_probability',.5)) if self.epoch>=int(self.config.get('selector_warmup_epochs',1)) else 0.
            use_predicted=(torch.rand(len(index),device=index.device)<ratio)&compatible[rows,predicted]
            index=torch.where(use_predicted,predicted,teacher)
            pair_ok=compatible.any(-1)
        elif mode!='sample':raise ValueError(mode)
        chosen,local,condition,aux=self.selected_conditions(batch,index,task,rgb_features)
        if mode=='sample':
            output=self.bridge.sample(self.net,chosen['prior'],local,condition)
            available=batch['candidate_mask'].any(1)
            return torch.where(available[:,None,None],output,torch.zeros_like(output))
        # Per-example weighting is essential: do not force unrelated route modes
        # to reconstruct the same expert trajectory or train on fabricated priors.
        c=self.bridge.config
        clean=encode_trajectory(batch['gt_pose'],c.action_step_m)
        prior=encode_trajectory(chosen['prior'],c.action_step_m)
        sigma=torch.empty(len(clean),device=clean.device).uniform_(c.sigma_min,c.sigma_max)
        ratio=(sigma[:,None,None]/c.sigma_max).square()
        noisy=(1-ratio)*clean+ratio*prior+sigma[:,None,None]*(1-ratio).sqrt()*torch.randn_like(clean)
        denoised=self.bridge.denoise(self.net,noisy,sigma,local,condition)
        per=((sigma.reciprocal().square()+c.sigma_data**-2)[:,None,None]*(denoised-clean).square()).mean((1,2))
        bridge_loss=(per*pair_ok).sum()/valid_denominator(pair_ok)
        valid=chosen['prior_valid']*pair_ok[:,None]
        target=reliability_target(chosen['prior'],batch['gt_pose'],chosen['prior_valid']).detach()
        count=valid_denominator(valid)
        reliability=(F.binary_cross_entropy_with_logits(aux['astar_reliability_logits'],target,reduction='none')*valid).sum()/count
        consistency=(F.smooth_l1_loss(aux['local_condition_gate'],target*(1-aux['depth_risk_target']),reduction='none')*valid).sum()/count
        residual=(aux['visual_risk_residual'].square()*valid).sum()/count
        gate_loss=reliability+.1*consistency+.1*residual
        loss=bridge_loss+float(self.config.get('gate_loss_weight',.01))*gate_loss+float(self.config.get('route_loss_weight',.1))*rank_loss
        entropy=-(probability*probability.clamp_min(1e-8).log()).sum(-1)
        return dict(loss=loss,bridge_loss=bridge_loss.detach(),gate_loss=gate_loss.detach(),route_loss=rank_loss.detach(),
                    bridge_coverage=pair_ok.float().mean(),label_coverage=covered.float().mean(),
                    predicted_compatible=compatible[torch.arange(len(index),device=index.device),predicted].float().mean(),
                    route_entropy=entropy.mean().detach(),shallow_weight=aux['shallow_weight'].mean().detach())
