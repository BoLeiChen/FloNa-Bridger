import numpy as np
import pytest
import torch
from test_multimodal_paths import config,inputs,planner
from model.path_distribution_v2 import PathDistributionV2, CHECKPOINT_FORMAT
from model.pointgoal_bridge import PointBridgePlanner
from datasets.path_distribution import PathProposer
from datasets.route_goal_supervision import GoalDistanceCache,route_goal_labels,route_labels_from_fields

def cfg():return dict(config(),point_goal_train_prob=0.,img_goal_train_prob=1.,goal_pair_margin=.25,goal_pair_loss_weight=.1)
def batch(tmp_path):
    b=inputs(tmp_path);b['task_type'][:]=1
    b['alternate_goal_rgb']=torch.flip(b['goal_rgb'],[0]);b['alternate_route_target']=torch.flip(b['route_target'],[-1]);b['goal_pair_valid']=torch.ones(2,dtype=torch.bool)
    # Use valid but goal-distinguishing targets.
    b['route_target'].zero_();b['route_target'][:,0]=1
    b['alternate_route_target'].zero_();b['alternate_route_target'][:,1]=1
    return b

def test_only_route_head_trains_and_no_unused_parameters(tmp_path):
    torch.manual_seed(7);m=PathDistributionV2(cfg(),load_pretrained=False).train();b=batch(tmp_path)
    assert not m.obs_encoder.training and not m.net.training and m.scorer.training
    trainable=[n for n,p in m.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith('scorer.') for n in trainable)
    loss=m(b)['loss'];assert torch.isfinite(loss);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.scorer.parameters())
    assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith('scorer.'))

def test_v2_ignores_supervision_and_responds_to_goal(tmp_path):
    torch.manual_seed(11);m=PathDistributionV2(cfg(),load_pretrained=False).eval();b=batch(tmp_path)
    with torch.no_grad():
        a=m(b,mode='score')['probabilities']
        altered=dict(b,task_point_world=b['task_point_world']+1000,label_goal_world=torch.ones(2,2)*999,
                     route_target=torch.zeros_like(b['route_target']),gt_pose=b['gt_pose']+100)
        c=m(altered,mode='score')['probabilities'];torch.testing.assert_close(a,c,atol=0,rtol=0)
        other=m(dict(b,goal_rgb=torch.zeros_like(b['goal_rgb'])),mode='score')['probabilities']
        assert not torch.allclose(a,other,atol=1e-7,rtol=1e-7)
        permutation=torch.randperm(a.shape[1]);swapped=dict(b)
        for k in b:
            if k.startswith('candidate_'):swapped[k]=b[k][:,permutation]
        torch.testing.assert_close(m(swapped,mode='score')['probabilities'],a[:,permutation],atol=1e-6,rtol=1e-5)

def test_frozen_point_controller_conditions_preserved(tmp_path):
    b=batch(tmp_path);base=PointBridgePlanner(dict(cfg(),point_goal_train_prob=1.,img_goal_train_prob=0.),load_pretrained=False).eval()
    m=PathDistributionV2(cfg(),load_pretrained=False).eval();m.load_state_dict(base.state_dict(),strict=False)
    index=torch.zeros(2,dtype=torch.long);rows=torch.arange(2)
    chosen=dict(b,prior=b['candidate_priors'][rows,index],prior_valid=b['candidate_valid'][rows,index],goal_world=b['candidate_goals'][rows,index])
    with torch.no_grad():
        _,_,rgb=m.visual_features(b);local,glob,_=m.conditions(chosen,rgb);l2,g2,_=base.conditions(chosen)
    torch.testing.assert_close(local,l2,atol=1e-5,rtol=1e-5);torch.testing.assert_close(glob,g2,atol=1e-5,rtol=1e-5)

def test_vectorized_goal_labels_match_reference(tmp_path):
    p=planner(tmp_path);pos=np.array([-5.,0.]);c=PathProposer(cfg()).propose(p,pos,.3);cache=GoalDistanceCache()
    goals=[c['candidate_goals'][i] for i in np.flatnonzero(c['candidate_mask'])[:3]]
    fields=np.stack([cache.distances(p,g) for g in goals]);many,costs=route_labels_from_fields(c,p,fields)
    for i,g in enumerate(goals):
        one,cost=route_goal_labels(c,p,pos,.3,g,cache)
        np.testing.assert_allclose(many[i],one,rtol=1e-5,atol=1e-5)
        np.testing.assert_allclose(costs[i],cost,rtol=1e-5,atol=1e-5)

def test_empty_labels_finite_and_checkpoint_roundtrip(tmp_path):
    from eval.multimodal_path import load_models
    b=batch(tmp_path);m=PathDistributionV2(cfg(),load_pretrained=False).eval()
    b['route_target'].zero_();b['alternate_route_target'].zero_();b['goal_pair_valid'][:]=False
    loss=m(b)['loss'];assert torch.isfinite(loss);loss.backward()
    torch.save(dict(format=CHECKPOINT_FORMAT,config=cfg(),planner=m.state_dict()),tmp_path/'m.pth')
    _,loaded=load_models(tmp_path/'m.pth','cpu')
    with torch.no_grad():torch.testing.assert_close(m(b,mode='score')['probabilities'],loaded(b,mode='score')['probabilities'])


def test_route_objective_does_not_copy_expert_future(tmp_path):
    torch.manual_seed(13);m=PathDistributionV2(cfg(),load_pretrained=False).eval();b=batch(tmp_path)
    with torch.no_grad():
        before=m(b)['loss']
        after=m(dict(b,gt_pose=b['gt_pose']+100,bridge_compatible=torch.zeros_like(b['bridge_compatible'])))['loss']
    torch.testing.assert_close(before,after,atol=0,rtol=0)
