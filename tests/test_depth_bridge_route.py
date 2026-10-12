import math
import numpy as np
import torch
from model.depth_bridge_guidance import DepthMap,depth_to_map,BridgeDepthGuidance,decode_metric
from model.depth_bridge_route import BridgeRouteGuidance,DepthMapMemory,DepthObservationBuffer,rasterize_floor_surface,calibrate_floor_scale
from test_depth_bridge_guidance import encode


def observed_map(wall=False):
    occ=np.zeros((121,121),bool);known=np.zeros_like(occ)
    # Observed region x=.6..2.0; origin remains unobserved.
    known[64:121,20:101]=True
    if wall:occ[83:85,:]=True
    m=DepthMap(occ,known,.025,-1.,1.5,'cpu');m.valid_fraction=1.
    return m


def route_guide(m):
    prior=np.stack((np.linspace(0,1.8,32),np.zeros(32)),-1)
    return BridgeRouteGuidance(m,[[3,0]],dict(weights={'goal':.25}),prior,np.ones(32),16)


def straight():
    x=np.concatenate((np.linspace(.0625,1.,16),np.linspace(1.04,1.64,16)))
    return encode([np.stack((x,np.zeros(32)),-1).tolist()]).repeat(32,1,1)


def test_local_route_works_with_unknown_origin_and_moves_executed_endpoint():
    m=observed_map();old=BridgeDepthGuidance(m,[[3.,0.]],{})
    assert not old.visible
    guide=route_guide(m);x=straight();out=guide(x,torch.full((32,),.4),'predictor')
    assert guide.visible and guide.records[-1]['route_active']
    torch.testing.assert_close(guide.route_target,torch.tensor([[1.25,0.]]))
    assert guide.records[-1]['before']['collision']==0
    assert guide.records[-1]['after']['goal']<guide.records[-1]['before']['goal']
    assert decode_metric(out,.045).mean(0)[16,0]>decode_metric(x,.045).mean(0)[16,0]
    assert guide.records[-1]['max_update_m']<=.01000011
    assert m.query(m.clearance,torch.zeros(1,1,2))[0].item()==0


def test_route_blocks_wall_unknown_and_invalid_prior():
    x=straight()
    for mode in ['wall','unknown','invalid']:
        m=observed_map(mode=='wall');guide=route_guide(m)
        if mode=='unknown':m.clearance.zero_()
        if mode=='invalid':guide.valid.zero_()
        guide(x,torch.full((32,),.4),'corrector')
        assert not guide.visible
        assert guide.records[-1]['before']['goal']==0


def test_free_surface_interpolation_never_fills_blind_origin_or_invalid_gap():
    h=w=128;v,u=np.mgrid[:h,:w]
    depth=np.zeros((h,w),np.float32);valid=v>h/2
    depth[valid]=.85*(h/2)/(v[valid]-h/2)
    depth[:,60:68]=0
    k=np.array([[64,0,64],[0,64,64],[0,0,1]])
    old=depth_to_map(depth,k,{},'cpu');new=depth_to_map(depth,k,dict(floor_surface=True),'cpu')
    assert new.known_fraction>old.known_fraction
    assert new.query(new.clearance,torch.zeros(1,1,2))[0].item()==0
    assert new.query(new.clearance,torch.tensor([[[1.5,0.]]]))[0].item()==0


def test_surface_rejects_depth_discontinuity_and_unobserved_vertices():
    forward=np.array([[1.,1.],[2.,2.]])
    left=np.array([[0.,.02],[0.,.02]])
    mask=rasterize_floor_surface(forward,left,np.ones((2,2),bool),(121,121),.025,-1,1.5,.1)
    assert not mask.any()
    forward=np.array([[1.,1.],[1.02,1.02]])
    mask=rasterize_floor_surface(forward,left,np.zeros((2,2),bool),(121,121),.025,-1,1.5,.1)
    assert not mask.any()


def test_memory_pose_fusion_rotation_occupied_wins_and_episode_reset():
    shape=(121,121);occ=np.zeros(shape,bool);known=np.zeros(shape,bool)
    # Source observes at x=1,y=0. Target at world(.5,0), yaw=pi/2 sees x=0,y=-.5.
    occ[80,60]=True;known[78:83,58:63]=True
    src=DepthMap(occ,known,.025,-1,1.5,'cpu');src.valid_fraction=1
    target=DepthMap(np.zeros(shape,bool),np.ones(shape,bool),.025,-1,1.5,'cpu');target.valid_fraction=1
    mem=DepthMapMemory(16);mem.add(src,[0,0,0]);mem.add(target,[.5,0,math.pi/2]);fused=mem.current()
    assert fused.query(fused.sdf,torch.tensor([[[0.,-.5]]]))[0].item()<0
    assert len(DepthMapMemory(16).frames)==0
    mem.add(target,[.5,0,math.pi/2]);assert len(mem.frames)==2


def test_route_no_valid_segments_and_disabled_guidance_are_identity():
    guide=route_guide(observed_map());guide.valid.zero_();x=straight()
    assert torch.equal(guide(x,torch.full((32,),.4),'predictor'),x)
    guide.scale=0
    assert guide(x,torch.full((32,),.4),'predictor') is x


def test_memory_does_not_create_free_blind_space():
    source=observed_map()
    empty=DepthMap(np.zeros((121,121),bool),np.zeros((121,121),bool),.025,-1,1.5,'cpu')
    empty.valid_fraction=1.
    mem=DepthMapMemory(16);mem.add(source,[0,0,0]);mem.add(empty,[.25,0,0])
    fused=mem.current()
    assert fused.query(fused.clearance,torch.zeros(1,1,2))[0].item()==0
    assert fused.query(fused.clearance,torch.tensor([[[1.,0.]]]))[0].item()>.15


def test_route_cannot_jump_invalid_prefix_or_past_its_end():
    guide=route_guide(observed_map());x=straight()
    guide.valid[:,10]=0
    guide(x,torch.full((32,),.4),'predictor')
    assert not guide.visible
    assert guide.gate_reason in ['no_progress','cross_track']


def test_observation_buffer_copies_actual_pose_and_subsamples_close_views():
    buffer=DepthObservationBuffer(2,.1)
    rgb=np.zeros((4,4,3),np.uint8);position=np.array([0.,0.,.85]);heading=np.array([1.,0.,.85])
    buffer.observe(rgb,position,heading)
    rgb[:]=123;position[0]=.01
    buffer.observe(rgb,position,heading)
    assert len(buffer.frames)==1
    position[0]=.2;buffer.observe(rgb,position,heading)
    rows=buffer.drain();assert len(rows)==2 and not buffer.frames
    assert not rows[0][0].any() and rows[0][1][0]==0
    assert rows[1][1][0]==.2


def test_floor_scale_recovers_metric_ground_and_keeps_origin_unknown():
    v,u=np.mgrid[:128,:128];depth=np.zeros((128,128),np.float32)
    depth[v>64]=2*.85*64/(v[v>64]-64)
    k=np.array([[64.,0,64],[0,64,64],[0,0,1]])
    calibrated,info=calibrate_floor_scale(depth,k,.85)
    assert info['accepted'] and abs(info['scale']-.5)<1e-6
    np.testing.assert_allclose(calibrated,depth*.5,rtol=1e-6)
    m=depth_to_map(depth,k,dict(floor_scale_calibration=True,floor_surface=True),'cpu')
    assert m.clearance.max()>.2 and m.depth_calibration['accepted']
    assert m.query(m.clearance,torch.zeros(1,1,2))[0].item()==0


def test_wall_or_invalid_depth_cannot_calibrate_floor_or_steer():
    k=np.array([[64.,0,64],[0,64,64],[0,0,1]])
    for depth in [np.full((128,128),2.,np.float32),np.full((128,128),np.nan,np.float32)]:
        m=depth_to_map(depth,k,dict(floor_scale_calibration=True),'cpu')
        assert not m.depth_calibration['accepted'] and m.valid_fraction==0
        assert not m.known.any() and not m.occupied.any()
        guide=route_guide(m);x=straight()
        assert guide(x,torch.full((32,),.4),'predictor') is x


def test_legacy_v2_enabled_guidance_matches_evaluated_version_exactly():
    import subprocess,types
    source=subprocess.check_output(['git','show','e621486:model/depth_bridge_guidance.py'],text=True)
    old=types.ModuleType('_evaluated_v2');exec(compile(source,'evaluated_v2','exec'),old.__dict__)
    m=observed_map(True);x=straight()
    a=old.BridgeDepthGuidance(m,[[3.,0.]],{})
    b=BridgeDepthGuidance(m,[[3.,0.]],{})
    for stage in ['churn','predictor','corrector']:
        torch.testing.assert_close(a(x,torch.full((32,),.4),stage),b(x,torch.full((32,),.4),stage),rtol=0,atol=0)
    assert a.records==b.records
