"""Hierarchical route loss, matching goal-vs-exploration inference groups."""
import torch
import torch.nn.functional as F


def route_objective(logits, rejection, batch, config, distributed=False):
    target=batch['route_target']
    eligible=batch.get('route_sample_valid',torch.ones_like(rejection,dtype=torch.bool)).bool()
    reject_target=batch['goal_set_reject'].float()
    active=(target.sum(-1)>0)&eligible
    if config.get('route_objective')=='hierarchical-v2':
        group=torch.where(reject_target[:,None].bool(),~batch['candidate_is_goal'],batch['candidate_is_goal'])
        group=group&batch['candidate_mask']
        if ((target>0)&~group&eligible[:,None]).any():raise ValueError('Route target lies outside supervised group')
        safe=group.clone();safe[~safe.any(-1),0]=True
        scored=logits.masked_fill(~safe,-1e4)
    else:scored=logits
    nll=-(target*F.log_softmax(scored,-1)).sum(-1)
    bce=F.binary_cross_entropy_with_logits(rejection,reject_target,reduction='none')
    counts=torch.stack([active.sum(),eligible.sum()]).to(dtype=logits.dtype)
    scale=1
    if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        # DDP averages rank gradients. Normalize by global active samples so
        # ranks with more arrived/masked windows do not receive extra weight.
        torch.distributed.all_reduce(counts)
        scale=torch.distributed.get_world_size()
    ce=scale*(nll*active).sum()/counts[0].clamp_min(1)
    reject_loss=scale*(bce*eligible).sum()/counts[1].clamp_min(1)
    return ce+float(config.get('goal_reject_loss_weight',.5))*reject_loss,ce,reject_loss,nll,bce,active,eligible
