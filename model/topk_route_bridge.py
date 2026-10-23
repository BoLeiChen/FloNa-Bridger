"""Train only goal-hypothesis route scoring/rejection; preserve the PointGoal controller."""
import torch
from torch import nn
import torch.nn.functional as F
from model.path_distribution_v2 import PathDistributionV2,GoalMatchingRouteScorer
from model.multimodal_path_bridge import valid_denominator
from model.route_objective import route_objective
CHECKPOINT_FORMAT='floverse-imagegoal-topk-route-bridge-v1'

class HypothesisScorer(GoalMatchingRouteScorer):
    def __init__(self,dim):
        super().__init__(dim)
        self.hypothesis_encoder=nn.Sequential(nn.Linear(8,dim),nn.GELU(),nn.Linear(dim,1))
        # Explicit unknown-goal-set prediction, conditioned on visual matching and
        # valid goal-hypothesis evidence. Never consumes training target fields.
        self.reject_head=nn.Sequential(nn.Linear(dim*3+4,dim),nn.GELU(),nn.Linear(dim,1))
    def forward(self,batch,observations,goal):
        logits=super().forward(batch,observations,goal)+self.hypothesis_encoder(batch['candidate_hypothesis']).squeeze(-1)
        observed=self.visual_projection(observations).mean((1,2));target=self.visual_projection(goal).mean(1)
        valid=batch['candidate_mask']&batch['candidate_is_goal'];den=valid.sum(-1).clamp_min(1)
        values=torch.where(valid,logits,torch.zeros_like(logits));mean=values.sum(-1)/den
        maximum=logits.masked_fill(~valid,-1e4).amax(-1);maximum=torch.where(valid.any(-1),maximum,torch.zeros_like(maximum))
        prior=(batch['candidate_hypothesis'][...,0]*valid).sum(-1)/den
        statistics=torch.stack([mean,maximum,prior,valid.float().mean(-1)],-1)
        rejection=self.reject_head(torch.cat([observed,target,observed*target,statistics],-1)).squeeze(-1)
        return logits,rejection

class TopKRouteBridge(PathDistributionV2):
    def __init__(self,config,load_pretrained=True):
        if not 1<=int(config['goal_top_k'])<=64:raise ValueError('goal_top_k must be in [1,64]')
        super().__init__(config,load_pretrained)
        self.scorer=HypothesisScorer(int(config['condition_dim']));self.train(False)
    def _scores(self,batch,observed,goal):
        logits,rejection=self.scorer(batch,observed,goal)
        logits=logits-float(self.config.get('route_revisit_penalty',.25))*batch['candidate_stats'][...,1]-float(self.config.get('route_failure_penalty',1.))*batch['candidate_stats'][...,3]
        mask=batch['candidate_mask'].bool();safe=mask.clone();safe[~mask.any(-1),0]=True
        return logits.masked_fill(~safe,-1e4),rejection
    def score_paths(self,batch):
        if not (batch['task_type']==1).all():raise ValueError('TopK model supports ImageGoal only')
        observed,goals,rgb=self.visual_features(batch)
        logits,rejection=self._scores(batch,observed,goals[:,0])
        # Reject mass is a routing decision, not a navigation-success/stop signal.
        reject=rejection.sigmoid();goal=batch['candidate_is_goal'];mask=batch['candidate_mask']
        group_goal=mask&goal;group_explore=mask&~goal
        mass=torch.where(group_goal.any(-1),1-reject,torch.zeros_like(reject))
        mass=torch.where(group_explore.any(-1),mass,torch.ones_like(mass))
        def normalized(group):
            safe=group.clone();safe[~group.any(-1),0]=True
            return logits.masked_fill(~safe,-1e4).softmax(-1)*group.any(-1)[:,None]
        probability=mass[:,None]*normalized(group_goal)+(1-mass[:,None])*normalized(group_explore)
        return probability,logits,reject,rgb
    def forward(self,batch,mode='loss',selected_index=None,prepared=None):
        if mode!='loss':return super().forward(batch,mode,selected_index,prepared)
        observed,goals,_=self.visual_features(batch)
        logits,rejection=self._scores(batch,observed,goals[:,0]);target=batch['route_target'];valid=target.sum(-1)>0
        loss,ce,reject_loss,_,_,valid,eligible=route_objective(logits,rejection,batch,self.config,distributed=self.scorer.training)
        p=logits.softmax(-1);zero=loss.detach()*0
        return dict(loss=loss,route_loss=ce.detach(),bridge_loss=zero,gate_loss=zero,shallow_weight=zero,
          bridge_coverage=batch['bridge_compatible'].any(-1).float().mean(),label_coverage=valid.float().mean(),
          predicted_compatible=batch['bridge_compatible'].gather(1,p.argmax(-1)[:,None]).float().mean(),
          route_entropy=-(p*p.clamp_min(1e-8).log()).sum(-1).mean().detach(),
          reject_loss=reject_loss.detach(),reject_accuracy=((((rejection.sigmoid()>=.5)==batch['goal_set_reject'].bool())&eligible).float().sum()/eligible.sum().clamp_min(1)),
          goal_coverage=((1-batch['goal_set_reject'].float())*eligible).sum()/eligible.sum().clamp_min(1),
          active_fraction=eligible.float().mean())
