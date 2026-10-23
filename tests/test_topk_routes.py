import json
import numpy as np
import pytest
import torch
from test_multimodal_paths import config,planner,inputs
from datasets.hypothesis_routes import HypothesisRoutes,hypothesis_labels
from datasets.route_goal_supervision import GoalDistanceCache
from eval.hypothesis_policy_state import HypothesisPolicyState
from model.topk_route_bridge import TopKRouteBridge
from localization.goal_hypotheses import GoalHypothesisProvider,signature

def cfg(k=16):return dict(config(),point_goal_train_prob=0.,img_goal_train_prob=1.,goal_top_k=k,exploration_candidates=2)
def hypotheses(n=20):
    return dict(poses=np.array([[4.,float(i%7-3),.1*i] for i in range(n)],np.float32),ids=np.arange(n),scores=np.linspace(2,-2,n,dtype=np.float32),density=np.ones(n,np.float32),match_logits=np.ones(n,np.float32))

def make_batch(tmp_path,k=16):
    p=planner(tmp_path);c=HypothesisRoutes(cfg(k),hypotheses()).propose(p,np.array([-5.,0.]),0.)
    batch=inputs(tmp_path)
    for name,value in c.items():batch[name]=torch.from_numpy(value)[None].repeat(2,*([1]*value.ndim))
    batch['task_type'][:]=1;batch['bridge_compatible']=batch['candidate_mask'].clone()
    batch['route_target']=batch['candidate_mask'].float()/batch['candidate_mask'].sum(-1,keepdim=True)
    batch['goal_set_reject']=torch.tensor([0.,1.]);return batch

@pytest.mark.parametrize('k',[4,8,12,16,64])
def test_k_geometry_and_padding(tmp_path,k):
    p=planner(tmp_path);g=HypothesisRoutes(cfg(k),hypotheses(6));c=g.propose(p,np.array([-5.,0.]),0.)
    assert c['candidate_mask'].shape==(k+2,)
    assert c['candidate_is_goal'].sum()==min(k,6)
    assert c['candidate_mask'][:min(k,6)].all()
    assert not c['candidate_mask'][min(k,6):k].any()
    for path in g.raw_paths.values():
        pix=np.rint(p.to_pixel(path)).astype(int);assert p.walkable[pix[:,1],pix[:,0]].all()
    np.testing.assert_allclose(c['candidate_goals'][:min(k,6)],hypotheses(6)['poses'][:min(k,6),:2])

def test_unknown_labels_do_not_force_wrong_goal(tmp_path):
    p=planner(tmp_path);pos=np.array([-5.,0.]);c=HypothesisRoutes(cfg(4),hypotheses()).propose(p,pos,0.)
    label=hypothesis_labels(c,p,pos,0.,np.array([-8.,8.]),GoalDistanceCache(),cfg(4))
    assert label['goal_set_reject']==1 and not label['goal_candidate_positive'].any()
    assert label['route_target'][:4].sum()==0 and label['route_target'][4:].sum()>0
    good=hypothesis_labels(c,p,pos,0.,c['candidate_goals'][0],GoalDistanceCache(),cfg(4))
    assert good['goal_set_reject']==0 and good['route_target'][4:].sum()==0

def test_absent_unreachable_and_arrived(tmp_path):
    p=planner(tmp_path);h=hypotheses(1);h['poses'][0,:2]=[999,999]
    c=HypothesisRoutes(cfg(4),h).propose(p,np.array([-5.,0.]),0.);assert not c['candidate_mask'][:4].any()
    h['poses'][0,:2]=[-5,0];c=HypothesisRoutes(cfg(4),h).propose(p,np.array([-5.,0.]),0.)
    assert c['candidate_mask'][0] and not c['candidate_priors'][0].any()

def test_reservoir_changes_identity_not_count(tmp_path):
    g=HypothesisRoutes(cfg(4),hypotheses(12));old=g.hypotheses['ids'].copy();assert g.refresh()
    assert len(g.hypotheses['ids'])==4 and not np.array_equal(old,g.hypotheses['ids'])
    assert not HypothesisRoutes(cfg(16),hypotheses(6)).refresh()

def test_model_freezing_gradients_and_gt_independence(tmp_path):
    torch.manual_seed(2);b=make_batch(tmp_path);m=TopKRouteBridge(cfg(),False).train()
    result=m(b);assert torch.isfinite(result['loss']);result['loss'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.scorer.parameters())
    assert all(not p.requires_grad and p.grad is None for n,p in m.named_parameters() if not n.startswith('scorer.'))
    assert not m.net.training and not m.obs_encoder.training
    m.eval()
    with torch.no_grad():
        a=m(b,mode='score')['probabilities'];other=dict(b,label_goal_world=torch.ones(2,2)*999,gt_pose=b['gt_pose']+999,goal_set_reject=1-b['goal_set_reject'])
        torch.testing.assert_close(a,m(other,mode='score')['probabilities'],rtol=0,atol=0)
        order=torch.randperm(a.shape[-1]);shuffled=dict(b)
        for k in b:
            if k.startswith('candidate_'):shuffled[k]=b[k][:,order]
        torch.testing.assert_close(m(shuffled,mode='score')['probabilities'],a[:,order],rtol=1e-5,atol=1e-6)
        b['candidate_mask'][:]=False
        assert torch.isfinite(m(b)['loss'])
        assert not m(b,mode='sample').any()

@pytest.mark.parametrize('k',[4,8,12,16])
def test_model_variable_k_and_checkpoint(tmp_path,k):
    m=TopKRouteBridge(cfg(k),False).eval();b=make_batch(tmp_path,k)
    with torch.no_grad():a=m.score_paths(b)
    assert a[0].shape==(2,k+2);torch.testing.assert_close(a[0].sum(-1),torch.ones(2))
    restored=TopKRouteBridge(cfg(k),False);restored.load_state_dict(m.state_dict(),strict=True)
    with torch.no_grad():torch.testing.assert_close(restored.score_paths(b)[0],a[0])

def test_reject_explores_and_arrival_is_not_success(tmp_path):
    p=planner(tmp_path);c=HypothesisRoutes(cfg(4),hypotheses()).propose(p,np.array([-5.,0.]),0.)
    state=HypothesisPolicyState(cfg(4));state.observe(np.array([-5.,0.]))
    selected=state.select(c,np.ones(6)/6,np.array([-5.,0.]),reject_probability=.99)
    assert not c['candidate_is_goal'][selected]
    state=HypothesisPolicyState(cfg(4));state.observe(c['candidate_goals'][0]);prob=np.array([.9,.02,.02,.02,.02,.02]);state.select(c,prob,c['candidate_goals'][0])
    state.observe(c['candidate_goals'][0]);state.select(c,prob,c['candidate_goals'][0])
    assert state.last_reason=='arrived_unconfirmed' and state.cooldown

def test_stable_id_hold_and_windowed_oscillation(tmp_path):
    p=planner(tmp_path);c=HypothesisRoutes(cfg(4),hypotheses()).propose(p,np.array([-5.,0.]),0.);state=HypothesisPolicyState(dict(cfg(4),route_min_hold_cycles=100))
    for t in range(7):
        pos=np.array([-5.+(t%2)*.3,0.]);state.observe(pos);state.select(c,np.array([.6,.3,.03,.03,.02,.02]),pos)
    assert state.cooldown # Net-zero oscillation is recognized despite >0.15m per-step movement.

def test_cache_k_independent_and_training_cannot_infer(tmp_path):
    c=dict(cfg(),goal_localization={'pin':'test'},goal_hypothesis_cache=str(tmp_path))
    assert signature(c)==signature(dict(c,goal_top_k=4))
    with pytest.raises(FileNotFoundError):GoalHypothesisProvider(c).get(tmp_path/'not-there.png')


def test_group_size_does_not_bias_exploration(tmp_path):
    p=planner(tmp_path);c=HypothesisRoutes(cfg(16),hypotheses()).propose(p,np.array([-5.,0.]),0.)
    state=HypothesisPolicyState(cfg());state.observe(np.array([-5.,0.]))
    probability=np.array([.8/16]*16+[.1,.1])
    selected=state.select(c,probability,np.array([-5.,0.]),reject_probability=.2)
    assert c['candidate_is_goal'][selected]


def test_exhausted_goal_hypotheses_explore_without_gt(tmp_path):
    p=planner(tmp_path);c=HypothesisRoutes(cfg(4),hypotheses(1)).propose(p,np.array([-5.,0.]),0.)
    state=HypothesisPolicyState(cfg(4));state.cooldown[0]=20;state.observe(np.array([-5.,0.]))
    index=state.select(c,np.array([.99,0,0,0,.005,.005]),np.array([-5.,0.]),reject_probability=.01)
    assert not c['candidate_is_goal'][index]
