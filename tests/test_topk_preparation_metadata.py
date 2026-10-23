import numpy as np
import pytest
from utils.prepare_topk_routes import trajectory_pose_count

def test_goal_frame_count_without_rgb_decode(tmp_path):
    root=tmp_path/'train';path=root/'scene_0'/'traj_0';path.mkdir(parents=True)
    np.savetxt(path/'traj_0.txt',np.zeros((41,4)))
    item=dict(key='traj::%s::scene_0::traj_0'%root,floor_key='floor::%s::scene_0'%root,sample_count=4)
    assert trajectory_pose_count(item,dict(len_traj_pred=16,traj_pace=8))==('traj_0',41)
    with pytest.raises(ValueError,match='window mismatch'):
        trajectory_pose_count(dict(item,sample_count=5),dict(len_traj_pred=16,traj_pace=8))
    np.savetxt(path/'traj_0.txt',[[np.nan]*4])
    with pytest.raises(ValueError,match='Invalid trajectory'):
        trajectory_pose_count(item,dict(len_traj_pred=16,traj_pace=8))


def test_distributed_loss_counts_and_validation_no_collective(monkeypatch):
    import torch
    from model.route_objective import route_objective
    monkeypatch.setattr(torch.distributed,'is_initialized',lambda:True)
    monkeypatch.setattr(torch.distributed,'get_world_size',lambda:2)
    calls=[]
    def reduce(counts):
        calls.append(1);counts+=torch.tensor([2.,2.]) # remote rank has two active samples
    monkeypatch.setattr(torch.distributed,'all_reduce',reduce)
    logits=torch.tensor([[1.,0.],[0.,1.]],requires_grad=True);reject=torch.zeros(2,requires_grad=True)
    batch=dict(route_target=torch.tensor([[1.,0.],[0.,0.]]),goal_set_reject=torch.zeros(2),
               route_sample_valid=torch.tensor([True,False]),candidate_mask=torch.ones(2,2,dtype=torch.bool),
               candidate_is_goal=torch.ones(2,2,dtype=torch.bool))
    cfg=dict(route_objective='hierarchical-v2')
    local=route_objective(logits,reject,batch,cfg)[0]
    assert not calls # main-rank-only validation must never wait for other ranks
    distributed=route_objective(logits,reject,batch,cfg,distributed=True)[0]
    torch.testing.assert_close(distributed,local*2/3)
    assert len(calls)==1
    distributed.backward();assert torch.isfinite(logits.grad).all()
