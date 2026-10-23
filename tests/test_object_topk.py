import inspect
import torch
import numpy as np
from test_topk_routes import cfg,make_batch,hypotheses
from test_multimodal_paths import planner
from model.object_topk_bridge import ObjectTopKRouteBridge
from datasets.hypothesis_routes import HypothesisRoutes
from localization.object_hypotheses import ObjectHypothesisProvider

def test_object_scorer_freezing_category_and_gt_isolation(tmp_path):
    torch.manual_seed(5);c=dict(cfg(4),route_task='object',img_goal_train_prob=0.,object_goal_train_prob=1.)
    b=make_batch(tmp_path,4);b['task_type'][:]=2;b['goal_text']=torch.randn(2,512)
    m=ObjectTopKRouteBridge(c,False).train();loss=m(b)['loss'];assert torch.isfinite(loss);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.scorer.parameters())
    assert all(not p.requires_grad and p.grad is None for n,p in m.named_parameters() if not n.startswith('scorer.'))
    m.eval()
    with torch.no_grad():
        a=m.score_paths(b)[0]
        other=dict(b,goal_rgb=b['goal_rgb']+100,label_goal_world=torch.ones(2,2)*999,gt_pose=b['gt_pose']+999,goal_set_reject=1-b['goal_set_reject'])
        torch.testing.assert_close(a,m.score_paths(other)[0],rtol=0,atol=0)
        changed=m.score_paths(dict(b,goal_text=torch.randn_like(b['goal_text'])))[0]
        assert not torch.allclose(a,changed)
        selected=a.argmax(-1);path=m(b,mode='sample',selected_index=selected)
        assert torch.isfinite(path).all()

def test_object_hypotheses_have_no_orientation_condition(tmp_path):
    p=planner(tmp_path);h=hypotheses(4);h['yaw_valid']=np.zeros(4,bool)
    c=HypothesisRoutes(cfg(4),h).propose(p,np.array([-5.,0.]),.4)
    assert (c['candidate_hypothesis'][:4,2:4]==0).all()
    assert 'target' not in inspect.signature(ObjectHypothesisProvider.get).parameters
