import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from PIL import Image
from datasets.bridge_maps import FloorplanPlanner
from datasets.path_distribution import PathProposer, VisitMemory, training_labels
from datasets.multimodal_path_dataset import task_for_window, MultimodalPathLMDB
from model.multimodal_path_bridge import MultimodalPathBridge
from eval.path_policy_state import PathPolicyState

torch.set_num_threads(2)


def config():
    return dict(point_goal_train_prob=.5,img_goal_train_prob=.5,object_goal_train_prob=0.,history_length=3,
                global_coordinate_mode='world',global_coordinate_step_m=.1,condition_dim=32,
                mha_num_attention_layers=1,down_dims=[16,32,64],len_traj_pred=16,astar_step_m=.1,
                bridge=dict(steps=2,churn=0.,action_step_m=.1),path_candidates=8,path_route_points=16,path_proposal_pool=48)


def planner(tmp_path, blocked=False):
    tmp_path.mkdir(exist_ok=True)
    free=np.ones((30,30),np.uint8);free[:,14]=0;free[5:8,14]=1;free[22:25,14]=1
    if blocked:free[:]=0
    np.save(tmp_path/'walkable.npy',free)
    (tmp_path/'metadata.json').write_text(json.dumps(dict(size=30,source_size=[300,300],scale=10,offset=[150,150])))
    return FloorplanPlanner(tmp_path)


def inputs(tmp_path):
    p=planner(tmp_path)
    candidate=PathProposer(config()).propose(p,np.array([-5.,0.]),0.)
    batch={k:torch.from_numpy(v)[None].repeat(2,*([1]*v.ndim)) for k,v in candidate.items()}
    batch.update(rgb=torch.rand(2,3,3,96,96),current_rgb=torch.rand(2,3,96,96),
                 floorplan=torch.rand(2,3,96,96),agent_pose=torch.tensor([[-5.,0.,0.],[-5.,0.,0.]]),
                 task_type=torch.tensor([0,1]),task_point_world=torch.tensor([[5.,2.],[0.,0.]]),
                 goal_rgb=torch.rand(2,3,96,96))
    batch['gt_pose']=batch['candidate_priors'][:,0].clone()
    batch['route_target']=batch['candidate_mask'].float()/batch['candidate_mask'].sum(1,keepdim=True)
    batch['bridge_compatible']=batch['candidate_mask'].clone()
    return batch


def test_geometry_reachable_routes_no_wall_shortcuts(tmp_path):
    p=planner(tmp_path);generator=PathProposer(config());pos=np.array([-5.,0.])
    paths=generator._explore(p,pos)
    assert 2<=len(paths)<=8
    for path in paths:
        pixels=np.rint(p.to_pixel(path)).astype(int)
        assert p.walkable[pixels[:,1],pixels[:,0]].all()
        assert (np.abs(np.diff(pixels,axis=0))<=1).all()
    a=generator.propose(p,pos,0.);b=generator.propose(p,pos,0.)
    for key in a:np.testing.assert_array_equal(a[key],b[key])
    point=generator.propose(p,pos,0.,point_goal=np.array([5.,0.]))
    assert point['candidate_mask'].sum()==1
    assert point['candidate_priors'][0,0].tolist()==[0.,0.]


def test_memory_changes_features_not_geometry(tmp_path):
    p=planner(tmp_path);g=PathProposer(config());pos=np.array([-5.,0.]);a=g.propose(p,pos,0.)
    memory=VisitMemory()
    for route in a['candidate_routes'][a['candidate_mask']]:
        for xy in route+pos:memory.update(xy)
    b=g.propose(p,pos,0.,memory)
    np.testing.assert_array_equal(a['candidate_routes'],b['candidate_routes'])
    assert b['candidate_stats'][:,1:].sum()>a['candidate_stats'][:,1:].sum()


def test_missing_candidates_hold_and_zero_loss_safe(tmp_path):
    data=inputs(tmp_path);data['candidate_mask'][:]=False;data['candidate_valid'][:]=0
    data['bridge_compatible'][:]=False;data['route_target'][:]=0
    model=MultimodalPathBridge(config(),load_pretrained=False)
    result=model(data);assert torch.isfinite(result['loss']);result['loss'].backward()
    assert result['bridge_coverage']==0
    with torch.no_grad():assert model(data,mode='sample').count_nonzero()==0


def test_multimodal_backward_and_discrete_selection(tmp_path):
    torch.manual_seed(5);data=inputs(tmp_path);model=MultimodalPathBridge(config(),load_pretrained=False)
    loss=model(data)['loss'];assert torch.isfinite(loss);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    assert model.scorer.score[-1].weight.grad.abs().sum()>0
    model.eval()
    seen={}
    def sample(net,prior,local,condition):seen['prior']=prior.clone();return prior
    model.bridge.sample=sample
    with torch.no_grad():
        index=torch.tensor([0,1]);out=model(data,mode='sample',selected_index=index)
    torch.testing.assert_close(out, data['candidate_priors'][torch.arange(2),index])
    torch.testing.assert_close(seen['prior'],out)


def test_image_policy_ignores_gt_and_point_coordinates(tmp_path):
    data=inputs(tmp_path);data['task_type'][:]=1
    model=MultimodalPathBridge(config(),load_pretrained=False).eval()
    with torch.no_grad():
        first=model(data,mode='score')['probabilities']
        altered=dict(data,gt_pose=data['gt_pose']+100,route_target=torch.zeros_like(data['route_target']),
                     task_point_world=data['task_point_world']+123,point_goal=torch.ones(2,1,2)*300,goal_world=torch.ones(2,2)*200)
        second=model(altered,mode='score')['probabilities']
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        changed=model(dict(data,goal_rgb=torch.zeros_like(data['goal_rgb'])),mode='score')['probabilities']
        assert not torch.allclose(first,changed,atol=1e-7,rtol=1e-7)
        permutation=torch.randperm(first.shape[1]);permuted=dict(data)
        for key in data:
            if key.startswith('candidate_'):permuted[key]=data[key][:,permutation]
        actual=model(permuted,mode='score')['probabilities']
        torch.testing.assert_close(actual,first[:,permutation],atol=1e-6,rtol=1e-5)


def test_oracle_labels_do_not_modify_candidates(tmp_path):
    p=planner(tmp_path);g=PathProposer(config());pos=np.array([-5.,0.]);c=g.propose(p,pos,0.)
    before={k:v.copy() for k,v in c.items()};expert=c['candidate_priors'][0]
    labels=training_labels(c,expert,p,np.array([5.,0.]),p.goal_tree([5.,0.]))
    assert labels['route_target'].sum()==pytest.approx(1)
    for k in c:np.testing.assert_array_equal(before[k],c[k])


def test_task_sampling_deterministic_and_training_root_guard():
    cfg=config();assert task_for_window('sample',2,cfg)==task_for_window('sample',2,cfg)
    assert {task_for_window(str(i),0,cfg) for i in range(50)}=={0,1}
    with pytest.raises(ValueError,match='non-training'):
        MultimodalPathLMDB(cfg,[dict(floor_key='floor::/dataset/test/HM3D::scene')],1)


def test_commitment_then_stall_allows_switch():
    c=dict(candidate_mask=np.array([True,True]),candidate_goals=np.array([[3.,0.],[0.,3.]]))
    state=PathPolicyState(min_hold=2,stall_cycles=2)
    assert state.select(c,np.array([.6,.4]),np.zeros(2))==0
    assert state.select(c,np.array([.1,.9]),np.zeros(2))==0
    for _ in range(4):state.observe(np.zeros(2))
    assert state.select(c,np.array([.1,.9]),np.zeros(2))==1


def test_checkpoint_roundtrip(tmp_path):
    from model.multimodal_path_bridge import CHECKPOINT_FORMAT
    from eval.multimodal_path import load_models
    cfg=config();model=MultimodalPathBridge(cfg,load_pretrained=False).eval()
    torch.save(dict(format=CHECKPOINT_FORMAT,config=cfg,planner=model.state_dict()),tmp_path/'model.pth')
    _,restored=load_models(tmp_path/'model.pth','cpu')
    data=inputs(tmp_path/'map')
    with torch.no_grad():torch.testing.assert_close(model(data,mode='score')['probabilities'],restored(data,mode='score')['probabilities'])


def test_failed_waypoint_is_not_a_visit():
    memory=VisitMemory();memory.update([0,0]);memory.update([5,5],failed=True)
    assert memory.key([5,5]) not in memory.cells
    assert memory.last==memory.key([0,0])
    assert memory.failures[memory.key([5,5])]==1


def test_image_rollout_never_uses_gt_goal_for_planning(tmp_path,monkeypatch):
    from eval import multimodal_path as entry
    p=planner(tmp_path/'map');cfg=config();batches=[];fallback_goals=[]
    rgb_dir=tmp_path/'rgb';rgb_dir.mkdir();Image.fromarray(np.full((96,96,3),128,np.uint8)).save(rgb_dir/'rgb_19.png')
    class Robot:
        def set_position(self,xy):self.position=np.array(xy)
        def set_orientation(self,q):pass
        def get_position(self):return self.position
        def get_rpy(self):return [0,0,0]
    robot=Robot()
    env=SimpleNamespace(robots=[robot],simulator_step=lambda:None,test_valid_position=lambda *a,**k:False)
    monkeypatch.setattr(entry,'capture',lambda *a:torch.zeros(3,96,96))
    def fallback(env,floor,goal,z,yaw):fallback_goals.append(goal.copy());return None,None
    monkeypatch.setattr(entry,'move_to_fallback',fallback)
    class Policy:
        def score_paths(self,batch):
            batches.append({k:v.clone() for k,v in batch.items()})
            mask=batch['candidate_mask'].float();return (mask/mask.sum(-1,keepdim=True),)
        def __call__(self,batch,mode,selected_index,prepared):return batch['candidate_priors'][torch.arange(1),selected_index]
    args=SimpleNamespace(goal_type='image',robot_z=.5,success_distance=1.,max_steps=1,collision_limit=15,execute_steps=9,save_plans=False,floor_level=0)
    for i,end in enumerate(([5.,5.],[10.,10.])):
        poses=np.tile([-5.,0.,-4.,0.],(20,1));poses[-1]=[*end,end[0]+1,end[1]]
        trajectory=tmp_path/'traj.txt';np.savetxt(trajectory,poses)
        result=entry.rollout(env,trajectory,torch.ones(1,96,96),p,Policy(),cfg,args,tmp_path/str(i),'cpu')
        assert not result['policy_uses_gt_target']
    assert len(batches)==2 and len(fallback_goals)==2
    assert 'goal_world' not in batches[0] and 'gt_pose' not in batches[0]
    assert batches[0]['task_point_world'].count_nonzero()==0
    for key in batches[0]:torch.testing.assert_close(batches[0][key],batches[1][key],rtol=0,atol=0)
    np.testing.assert_array_equal(fallback_goals[0],fallback_goals[1])


def test_point_warm_start_preserves_base_conditions(tmp_path):
    from model.pointgoal_bridge import PointBridgePlanner
    cfg=config();base=PointBridgePlanner(dict(cfg,point_goal_train_prob=1.,img_goal_train_prob=0.),load_pretrained=False).eval()
    new=MultimodalPathBridge(cfg,load_pretrained=False).eval();new.load_state_dict(base.state_dict(),strict=False)
    data=inputs(tmp_path);data['task_type'][:]=0;index=torch.tensor([0,0]);rows=torch.arange(2)
    chosen=dict(data,prior=data['candidate_priors'][rows,index],prior_valid=data['candidate_valid'][rows,index],goal_world=data['candidate_goals'][rows,index])
    with torch.no_grad():
        ref_local,ref_global,_=base.conditions(chosen)
        prepared=new.score_paths(data)
        _,local,global_cond,_=new.selected_conditions(data,index,prepared[2],prepared[3])
    torch.testing.assert_close(local,ref_local,atol=0,rtol=0)
    torch.testing.assert_close(global_cond,ref_global,atol=0,rtol=0)


def test_geometry_disk_cache_is_independent_of_tasks_and_memory(tmp_path):
    p=planner(tmp_path/'map');p.metadata['fingerprint']='test-map'
    cfg=dict(config(),path_candidate_cache=str(tmp_path/'geometry'))
    first=PathProposer(cfg);a=first.propose(p,np.array([-5.,0.]),0.)
    first.disk.close();first.disk=None
    second=PathProposer(cfg)
    # Reading existing geometry must not invoke graph search again.
    p.goal_tree=lambda *args: (_ for _ in ()).throw(AssertionError('cache miss'))
    b=second.propose(p,np.array([-5.,0.]),0.)
    for k in a:np.testing.assert_array_equal(a[k],b[k])
    second.disk.close();second.disk=None


def test_route_preferences_respond_to_new_observations_and_visits(tmp_path):
    torch.manual_seed(9)
    data=inputs(tmp_path);data['task_type'][:]=1
    model=MultimodalPathBridge(config(),load_pretrained=False).eval()
    with torch.no_grad():
        before=model(data,mode='score')['probabilities']
        observed=model(dict(data,current_rgb=torch.zeros_like(data['current_rgb'])),mode='score')['probabilities']
        assert not torch.allclose(before,observed,rtol=1e-7,atol=1e-7)
        stats=data['candidate_stats'].clone();stats[:,0,3]=3.
        after=model(dict(data,candidate_stats=stats),mode='score')['probabilities']
        assert (after[:,0]<before[:,0]).all()


def test_image_only_training_updates_scorer_and_bridge(tmp_path):
    torch.manual_seed(17)
    cfg=dict(config(),point_goal_train_prob=0.,img_goal_train_prob=1.)
    assert {task_for_window(str(i),epoch,cfg) for epoch in (0,1,4) for i in range(100)}=={1}
    data=inputs(tmp_path);data['task_type'][:]=1;data['task_point_world'].zero_()
    model=MultimodalPathBridge(cfg,load_pretrained=False)
    result=model(data);result['loss'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)
    assert model.scorer.score[-1].weight.grad.abs().sum()>0
    assert any(p.grad.abs().sum()>0 for p in model.net.parameters() if p.requires_grad)


def test_image_selected_bridge_conditions_ignore_supervision(tmp_path):
    torch.manual_seed(19)
    data=inputs(tmp_path);data['task_type'][:]=1
    model=MultimodalPathBridge(config(),load_pretrained=False).eval()
    altered=dict(data,gt_pose=data['gt_pose']+100,
                 route_target=torch.zeros_like(data['route_target']),
                 bridge_compatible=torch.zeros_like(data['bridge_compatible']),
                 task_point_world=data['task_point_world']+123,
                 goal_world=torch.ones(2,2)*200)
    with torch.no_grad():
        first=model.score_paths(data);second=model.score_paths(altered)
        index=first[0].argmax(-1)
        chosen_a,local_a,global_a,_=model.selected_conditions(data,index,first[2],first[3])
        chosen_b,local_b,global_b,_=model.selected_conditions(altered,index,second[2],second[3])
    for key in ('prior','prior_valid','goal_world'):
        torch.testing.assert_close(chosen_a[key],chosen_b[key],atol=0,rtol=0)
    torch.testing.assert_close(local_a,local_b,atol=0,rtol=0)
    torch.testing.assert_close(global_a,global_b,atol=0,rtol=0)


def test_previous_candidate_does_not_repeat_graph_search(tmp_path):
    p=planner(tmp_path);g=PathProposer(config());pos=np.array([-5.,0.])
    before=g.propose(p,pos,0.)
    goal=before['candidate_goals'][np.flatnonzero(before['candidate_mask'])[0]]
    p.astar=lambda *args: (_ for _ in ()).throw(AssertionError('redundant A*'))
    p.goal_tree=lambda *args: (_ for _ in ()).throw(AssertionError('redundant tree'))
    after=g.propose(p,pos,0.,previous_goal=goal)
    for key in before:np.testing.assert_array_equal(before[key],after[key])


def test_missing_previous_endpoint_reuses_current_source_tree(tmp_path):
    p=planner(tmp_path);cfg=dict(config(),path_candidates=1)
    g=PathProposer(cfg);pos=np.array([-5.,0.]);before=g.propose(p,pos,0.)
    goal=np.array([-3.,0.])
    assert np.linalg.norm(before['candidate_goals'][0]-goal)>.5
    reference=p.astar(pos,goal)
    p.astar=lambda *args: (_ for _ in ()).throw(AssertionError('extra A*'))
    p.goal_tree=lambda *args: (_ for _ in ()).throw(AssertionError('extra tree'))
    after=g.propose(p,pos,0.,previous_goal=goal)
    np.testing.assert_allclose(after['candidate_goals'][0],reference[-1])
    assert after['candidate_mask'][0]
    # A move to a different cell must create a new source tree.
    with pytest.raises(AssertionError,match='extra tree'):
        g.propose(p,pos+np.array([1.,0.]),0.,previous_goal=goal)
