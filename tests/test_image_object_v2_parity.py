"""Paired modality configs share navigation, losses, state and success accounting."""
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from datasets.category_route_supervision import CategoryDistanceCache, category_route_labels
from datasets.hypothesis_routes import HypothesisRoutes
from eval.hypothesis_policy_state import HypothesisPolicyState
from eval.object_category_protocol import score_category_trajectory
from eval.image_region_protocol import image_region_result, aggregate_image_region_results
from test_topk_routes import cfg

def v2(k=4):
    return dict(cfg(k), route_supervision="object-category-center-region-v2", route_policy_version="category-v2", route_objective="hierarchical-v2", validation_design="category-balanced-v2")
from test_topk_routes import hypotheses
from test_multimodal_paths import planner

ROOT=Path(__file__).resolve().parents[1]

def test_release_config_and_holdout():
    image=OmegaConf.to_container(OmegaConf.load(ROOT/'config/imagegoal_v3.yaml'))
    assert image['route_task']=='image'
    assert len(json.loads((ROOT/image['validation_buildings_file']).read_text()))>=40
    assert image['route_policy_version']=='category-v2'
    assert image['route_objective']=='hierarchical-v2'
    assert image['validation_design']=='image-source-floor-balanced-v2'


def test_same_hypotheses_same_geometry_labels_state_for_both_modalities(tmp_path):
    p=planner(tmp_path);pos=np.array([-5.,0.]);h=hypotheses(20)
    image=dict(v2(),route_task='image',route_supervision='image-goal-region-v2')
    obj=dict(v2(),route_task='object')
    generators=[HypothesisRoutes(c,h) for c in (image,obj)]
    states=[HypothesisPolicyState(c) for c in (image,obj)]
    target=dict(world_xy=[[4.,-3.]],object_ids=['target'])
    for step in range(14):
        candidates=[g.propose(p,pos,0.) for g in generators]
        for key in candidates[0]:np.testing.assert_equal(candidates[0][key],candidates[1][key])
        labels=[category_route_labels(c,p,pos,target,CategoryDistanceCache(),cfg) for c,cfg in zip(candidates,(image,obj))]
        for key in labels[0]:np.testing.assert_equal(labels[0][key],labels[1][key])
        selected=[]
        for state,c,g in zip(states,candidates,generators):
            state.observe(pos)
            index=state.select(c,np.ones(6)/6,pos,reject_probability=.99 if step<6 else .01)
            g.commit(index,c);selected.append((index,state.last_reason,state.last_rejected))
            if state.refresh_due():g.refresh()
        assert selected[0]==selected[1]
        pos=pos+np.array([0.,.02])


def test_image_region_scoring_and_aggregation_matches_object(tmp_path):
    obj=score_category_trajectory([[1.,0.,0.,0.],[2.,0.,0.,0.]],[0,0],dict(world_xy=[[3,0]],object_ids=['target']),2.)
    image=image_region_result(obj)
    assert image['success']==obj['success']
    assert image['SPL_goal_map']==obj['SPL_category_map']
    assert image['SoftSPL_goal_map']==obj['SoftSPL_category_map']
    d=tmp_path/'scene'/'traj';d.mkdir(parents=True)
    (d/'result.json').write_text(json.dumps(image))
    assert aggregate_image_region_results(tmp_path,[('scene','traj')])['SR']==1.
    (d/'result.json').write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='Mixed/old'):aggregate_image_region_results(tmp_path,[('scene','traj')])


@pytest.mark.parametrize('goal_x,expected_moves',[(10.,1),(8.,0)])
def test_image_rollout_first_hit_and_gt_input_isolation(tmp_path,monkeypatch,goal_x,expected_moves):
    from types import SimpleNamespace
    from PIL import Image
    from eval import multimodal_path as entry
    from test_floverse_eval_protocol import Env
    import localization.goal_hypotheses as provider_module
    import datasets.hypothesis_routes as route_module
    directory=tmp_path/'data'/'building_0'/'traj_0';(directory/'rgb').mkdir(parents=True)
    poses=np.array([[i,0,i+1,0] for i in range(20)],float);poses[-1]=[goal_x,0,goal_x+1,0]
    path=directory/'traj_0.txt';np.savetxt(path,poses)
    Image.new('RGB',(96,96)).save(directory/'rgb/rgb_19.png')
    monkeypatch.setattr(entry,'category_reference_length',lambda *args:float(expected_moves))
    monkeypatch.setattr(entry,'capture',lambda e,d:torch.zeros(3,2,2))
    monkeypatch.setattr(entry,'mark_floorplan',lambda *args:torch.ones(3,2,2))
    class Provider:
        def __init__(self,*args,**kwargs):pass
        def get(self,*args,**kwargs):
            assert len(args)==1 and not kwargs
            return {}
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
            assert not any(k in batch for k in ('category_targets','label_goal_world','diagnostic_gt_goal','goal_text'))
            assert torch.count_nonzero(batch['task_point_world'])==0
            return torch.ones(1,1),None,torch.zeros(1),None
        def __call__(self,*args,**kwargs):
            path=torch.zeros(1,16,2);path[0,:,0]=torch.arange(16);return path
    monkeypatch.setattr(provider_module,'GoalHypothesisProvider',Provider)
    monkeypatch.setattr(route_module,'HypothesisRoutes',Routes)
    args=SimpleNamespace(goal_type='image',diagnostic_route_mode='learned',controller_checkpoint=None,
        robot_z=.5,floor_level=0,execute_steps=9,max_steps=5,collision_limit=15,success_distance=1.,save_plans=False)
    config=dict(goal_top_k=16,exploration_candidates=2,history_length=3,len_traj_pred=16,astar_step_m=.1,
                route_supervision='image-goal-region-v2',route_policy_version='category-v2')
    env=Env();maps=SimpleNamespace(metadata=dict(offset=[0,0],scale=1,source_size=[256,256]))
    result=entry.rollout(env,path,torch.zeros(3,2,2),maps,Policy(),config,args,tmp_path/'out',torch.device('cpu'))
    assert result['success'] and result['moves']==expected_moves and len(env.checks)==expected_moves
    assert result['protocol']=='floverse-image-goal-region-1m-v2'
    assert result['final_distance']==result['terminal_distance']
    assert result['route_policy']=='shared_topk_v2_persistent_exploration_hysteresis'
