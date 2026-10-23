import importlib.util
from pathlib import Path
import torch
from model.bridge_diffusion import BridgeConfig, DiffusionBridge, encode_trajectory, decode_trajectory
from model.bridge_guidance.geometry import LocalMap, depth_to_local_map
from model.bridge_guidance.bridge_energy import BridgeEnergyGuidance


def config():
    return dict(scale=1., sigma_max=.5, max_update_m=.025,
        costs=dict(robot_radius=.25, safety_margin=.1, subdivisions=32,
                   weights=dict(collision=1., unknown=1., boundary=1., shape=.1, motion=1., goal=.25)))


def world(obstacle=True):
    occupied = torch.zeros(1, 121, 121, dtype=torch.bool)
    if obstacle:
        occupied[:, 60, :] = True  # x=0.5 m wall, forward motion should retreat.
    return LocalMap.from_masks(occupied, torch.ones_like(occupied), .025, -1., 1.5)


def test_closed_form_bridge_score_shift():
    bridge=DiffusionBridge(BridgeConfig(action_step_m=.1))
    x=torch.randn(1,16,2); clean=torch.randn_like(x); endpoint=torch.randn_like(x)
    sigma=torch.tensor([.3]); g=torch.randn_like(x)
    d0=bridge.derivative(x,clean,endpoint,sigma)
    d1=bridge.derivative(x,clean-sigma[:,None,None]**2*g,endpoint,sigma)
    torch.testing.assert_close(d1-d0,sigma[:,None,None]*g,atol=2e-6,rtol=2e-5)


def test_baseline_sampler_bitwise_parity():
    p=Path(__file__).parent/'fixtures/baseline_bridge.py'
    spec=importlib.util.spec_from_file_location('original_bridge',str(p)); original=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name]=original;spec.loader.exec_module(original)
    class Net:
        def __init__(self):self.calls=0
        def __call__(self,sample,timestep,**kw):self.calls+=1;return sample.sin()*.05
    path=torch.zeros(1,16,2);path[0,:,0]=torch.arange(16)*.1
    a,b=Net(),Net();cfg=dict(action_step_m=.1)
    torch.manual_seed(47); expected=original.DiffusionBridge(original.BridgeConfig(**cfg)).sample(a,path,None,None)
    torch.manual_seed(47); actual=DiffusionBridge(BridgeConfig(**cfg)).sample(b,path,None,None)
    assert torch.equal(expected,actual);assert a.calls==b.calls==29
    guidance=BridgeEnergyGuidance(world(),None,dict(config(),scale=0),.1)
    torch.manual_seed(47); zero=DiffusionBridge(BridgeConfig(**cfg)).sample(Net(),path,None,None,guidance)
    assert torch.equal(expected,zero)
    assert torch.equal(zero[:,0],torch.zeros(1,2))


def test_collision_update_direction_and_trust_region():
    path=torch.zeros(1,16,2);path[0,:,0]=torch.linspace(0,.4,16)
    x=encode_trajectory(path,.1); guide=BridgeEnergyGuidance(world(),None,config(),.1)
    y=guide(x,torch.tensor(.4),'predictor')
    after=decode_trajectory(y,.1);before=decode_trajectory(x,.1)
    assert after[0,-1,0]<before[0,-1,0]
    assert (after-before).norm(dim=-1).max()<=.0250001
    assert torch.equal(y[:,0],x[:,0])
    record=guide.records[-1]
    assert record['after']['collision'][0]<record['before']['collision'][0]
    assert guide.objective(after)<=guide.objective(before)+1e-7


def test_no_fabricated_boundary_obstacle_and_sigma_gate():
    local=world(False)
    assert local.sdf.min()>2
    x=torch.zeros(1,16,2)
    guide=BridgeEnergyGuidance(local,None,config(),.1)
    assert guide(x,torch.tensor(.6),'churn') is x


def test_invalid_depth_produces_no_obstacles_or_goal_force():
    cfg=dict(depth_units='meters',image_width=16,image_height=16,fx=8,fy=8,cx=8,cy=8,camera_height=.8)
    local=depth_to_local_map(torch.full((1,1,16,16),float('nan')),cfg)
    assert not local.known.any();assert (local.sdf>0).all()
    guide=BridgeEnergyGuidance(local,torch.tensor([[1.,0.]]),config(),.1)
    assert not guide.objective.visible_goal.any()


def test_radial_floor_projection_metric_and_y_sign():
    n=64;yy,xx=torch.meshgrid(torch.arange(n),torch.arange(n),indexing='ij')
    rays=torch.stack([torch.ones_like(xx),-(xx-32)/32.,-(yy-32)/32.],-1).float()
    rays=rays/rays.norm(dim=-1,keepdim=True)
    depth=torch.where(rays[...,2]<-.1,-.8/rays[...,2],torch.zeros_like(rays[...,2]))
    local=depth_to_local_map(depth[None,None],dict(depth_units='meters',image_width=n,image_height=n,
        fx=32,fy=32,cx=32,cy=32,camera_height=.8,resolution=.05))
    assert local.known.any();assert not (local.sdf<0).any()
    # A left pixel's body-height obstacle must project to positive ego y.
    depth.zero_();depth[32,16]=1.
    local=depth_to_local_map(depth[None,None],dict(depth_units='meters',image_width=n,image_height=n,
        fx=32,fy=32,cx=32,cy=32,camera_height=.8,resolution=.05,minimum_valid_fraction=0))
    ij=(local.sdf[0]<0).nonzero();assert len(ij)==1
    assert local.y_max-ij[0,1]*local.resolution>0
