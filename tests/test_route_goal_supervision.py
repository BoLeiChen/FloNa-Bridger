import numpy as np
import pytest
from datasets.path_distribution import PathProposer
from datasets.route_goal_supervision import GoalDistanceCache,route_goal_labels
from test_multimodal_paths import planner,config

def test_goal_labels_change_without_changing_observation_or_candidates(tmp_path):
    p=planner(tmp_path);pos=np.array([-5.,0.]);c=PathProposer(config()).propose(p,pos,0.)
    valid=np.flatnonzero(c['candidate_mask']);i,j=valid[0],valid[-1]
    saved={k:v.copy() for k,v in c.items()};cache=GoalDistanceCache()
    a,_=route_goal_labels(c,p,pos,0.,c['candidate_goals'][i],cache)
    b,_=route_goal_labels(c,p,pos,0.,c['candidate_goals'][j],cache)
    assert a.sum()==pytest.approx(1);assert b.sum()==pytest.approx(1)
    assert not np.allclose(a,b)
    for k in c:np.testing.assert_array_equal(c[k],saved[k])
    assert not a[~c['candidate_mask']].any()

def test_unreachable_goal_has_no_fake_labels(tmp_path):
    p=planner(tmp_path);pos=np.array([-5.,0.]);c=PathProposer(config()).propose(p,pos,0.)
    weights,cost=route_goal_labels(c,p,pos,0.,[10000,10000],GoalDistanceCache())
    assert weights.sum()==0
