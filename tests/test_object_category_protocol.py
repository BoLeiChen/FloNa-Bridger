import json
import numpy as np
import pytest
from eval.object_category_protocol import (PROTOCOL, load_category_targets, distance_to_category,
    score_category_trajectory, category_reference_length, aggregate_category_results)


def targets():
    return dict(object_ids=['chair_1','chair_2'],world_xy=[[10.,0.],[0.,5.]])


def test_any_instance_real_center_and_floor_alignment(tmp_path):
    assets=tmp_path/'assets'/'building';assets.mkdir(parents=True)
    episode=tmp_path/'building_1'/'traj_0'/'object';episode.mkdir(parents=True)
    first=dict(object_id='chair_1',object_category='dining_chair',floor=0,
               obj_position=[10.,0.,.5],agent_position=[8.,0.,.5])
    records=[first,dict(first,object_id='chair_2',obj_position=[0.,5.,.5]),
             dict(first,object_id='chair_1',floor=1,obj_position=[0.,0.,3.5],agent_position=[0.,1.,3.5]),
             dict(first,object_id='table_1',object_category='table',obj_position=[0.,0.,.5])]
    (assets/'building.json').write_text(json.dumps(records))
    (episode/'object.json').write_text(json.dumps([dict(first,object_category='dining chair')]))
    got=load_category_targets(tmp_path/'assets','building_1',episode.parent)
    assert got['object_ids']==['chair_1','chair_2']
    assert got['semantic_floor']==0 and got['simulator_floor']==1
    assert distance_to_category([8.,0.],got)==2.  # observer != physical object
    assert score_category_trajectory([[0.,4.,0.,0.]], [0.,0.],got,4.)['success']
    assert not score_category_trajectory([[0.,0.,0.,0.]], [0.,0.],got,4.)['success']


def test_first_hit_inclusive_radius_start_and_traveled_length():
    rows=[[0.,2.,0.,0.],[0.,4.,0.,0.],[30.,30.,0.,1.]]
    got=score_category_trajectory(rows,[0.,0.],targets(),4.)
    assert got['success'] and got['nearest_object_id']=='chair_2'
    assert got['scored_moves']==2 and got['traveled_m']==4. and got['collisions']==0
    assert got['SPL_category_map']==1. and got['SoftSPL_category_map']==1.
    start=score_category_trajectory([], [0.,4.],targets(),0.)
    assert start['success'] and start['SPL_category_map']==1.
    assert not score_category_trajectory([], [0.,0.],targets(),4.)['success']


def test_collision_budget_and_missing_map_reference():
    rows=[[0.,0.,0.,1.]]*15+[[0.,5.,0.,0.]]
    got=score_category_trajectory(rows,[0.,0.],targets())
    assert not got['success'] and got['collisions']==15 and got['SPL_category_map'] is None
    rows[14]=[0.,4.,0.,1.]
    assert score_category_trajectory(rows,[0.,0.],targets())['success']


def test_map_reference_is_to_nearest_success_region(tmp_path):
    from datasets.bridge_maps import FloorplanPlanner
    np.save(tmp_path/'walkable.npy',np.ones((20,20),bool))
    (tmp_path/'metadata.json').write_text(json.dumps(dict(size=20,source_size=[20,20],scale=1.,offset=[0.,0.])))
    planner=FloorplanPlanner(tmp_path)
    assert category_reference_length(planner,[0.,0.],targets(),1.)==4.
    assert category_reference_length(planner,[0.,4.],targets(),1.)==0.


def test_aggregate_rejects_old_protocol_and_reports_spl_coverage(tmp_path):
    d=tmp_path/'s'/'t';d.mkdir(parents=True)
    r=score_category_trajectory([[0.,4.,0.,0.]], [0.,0.],targets())
    (d/'result.json').write_text(json.dumps(r))
    got=aggregate_category_results(tmp_path,[('s','t')])
    assert got['SR']==1. and got['SPL'] is None and got['spl_unavailable_episodes']==1
    r['object_success_protocol']='original_reference_trajectory_endpoint'
    (d/'result.json').write_text(json.dumps(r))
    with pytest.raises(ValueError,match='Mixed/old'):
        aggregate_category_results(tmp_path,[('s','t')])


@pytest.mark.parametrize('other_x,expected_moves',[(10.,1),(8.,0)])
def test_rollout_stops_at_other_instance_without_gt_policy_input(tmp_path,monkeypatch,other_x,expected_moves):
    import torch
    from types import SimpleNamespace
    from eval import multimodal_path as entry
    from test_floverse_eval_protocol import Env
    import localization.object_hypotheses as provider_module
    import datasets.hypothesis_routes as route_module
    directory=tmp_path/'data'/'building_0'/'traj_0';(directory/'object').mkdir(parents=True)
    poses=np.array([[i,0,i+1,0] for i in range(20)],float)
    path=directory/'traj_0.txt';np.savetxt(path,poses)
    ref=dict(object_id='chair_1',object_category='chair',floor=0,
             agent_position=[19.,0.,.5],obj_position=[30.,0.,.5])
    (directory/'object/object.json').write_text(json.dumps([ref]))
    assets=tmp_path/'assets'/'building';assets.mkdir(parents=True)
    (assets/'building.json').write_text(json.dumps([ref,dict(ref,object_id='chair_2',obj_position=[other_x,0.,.5])]))
    monkeypatch.setattr(entry,'category_reference_length',lambda *args:float(expected_moves))
    monkeypatch.setattr(entry,'capture',lambda e,d:torch.zeros(3,2,2))
    monkeypatch.setattr(entry,'mark_floorplan',lambda *args:torch.ones(3,2,2))
    class Provider:
        def __init__(self,*args,**kwargs):pass
        def get(self,path,category,**kwargs):
            assert category=='chair' and set(kwargs)=={'origin_xy','extent_xy'}
            return dict(goal_text=np.zeros(512))
    class Routes:
        offset=0
        def __init__(self,*args):pass
        def propose(self,*args,**kwargs):
            assert kwargs['point_goal'] is None
            return dict(candidate_mask=np.array([True]),candidate_ids=np.array([0]),
                        candidate_is_goal=np.array([True]),candidate_goals=np.array([[25.,0.]]),
                        candidate_priors=np.zeros((1,16,2)))
        def commit(self,*args):pass
    class Policy:
        def score_paths(self,batch):
            assert not any(k in batch for k in ('category_targets','label_goal_world','diagnostic_gt_goal'))
            assert torch.count_nonzero(batch['task_point_world'])==0
            return torch.ones(1,1),None,torch.zeros(1),None
        def __call__(self,*args,**kwargs):
            path=torch.zeros(1,16,2);path[0,:,0]=torch.arange(16)
            return path
    monkeypatch.setattr(provider_module,'ObjectHypothesisProvider',Provider)
    monkeypatch.setattr(route_module,'HypothesisRoutes',Routes)
    args=SimpleNamespace(goal_type='object',object_success_mode='category',diagnostic_route_mode='learned',
        controller_checkpoint=None,robot_z=.5,floor_level=0,execute_steps=9,max_steps=5,
        collision_limit=15,success_distance=1.,save_plans=False)
    config=dict(goal_top_k=16,exploration_candidates=2,history_length=3,len_traj_pred=16,astar_step_m=.1,
                scene_assets=str(tmp_path/'assets'))
    maps=SimpleNamespace(metadata=dict(offset=[0,0],scale=1,source_size=[256,256]))
    env=Env()
    result=entry.rollout(env,path,torch.zeros(3,2,2),maps,Policy(),config,args,tmp_path/'out',torch.device('cpu'))
    assert result['success'] and result['moves']==expected_moves
    assert len(env.checks)==expected_moves
    assert result['nearest_object_id']=='chair_2'
    assert result['final_distance']==result['terminal_distance']
