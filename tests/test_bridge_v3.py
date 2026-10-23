import copy
import torch
from model.bridge_guidance.geometry import LocalMap, ObservationMapMemory
from model.bridge_guidance.navigation_v2 import NavigationObjectiveV2
from model.bridge_guidance.navigation_v3 import NavigationObjectiveV3
from model.bridge_guidance.bridge_energy_v3 import BridgeEnergyGuidanceV3
from model.bridge_diffusion import encode_trajectory,decode_trajectory


def setup(route=True,robust=False,wall=False):
    occ=torch.zeros(1,161,161,dtype=torch.bool)
    known=torch.zeros_like(occ);known[:,56:]=True  # Only x >= .4m observed.
    if wall:occ[:,80]=True  # x=1m
    local=LocalMap.from_masks(occ,known,.025,-1.,2.)
    prior=torch.zeros(1,16,2);prior[0,:,0]=torch.arange(16)*.1
    valid=torch.ones(1,16)
    cfg=dict(scale=1.,sigma_max=.5,max_update_m=.025,correction_budget=robust,total_update_budget_m=.05,
        costs=dict(route_energy=route,prefix_risk=robust,robot_radius=.25,safety_margin=.1,subdivisions=32,
                   weights=dict(collision=1.,unknown=1.,boundary=1.,shape=.1,motion=1.,goal=.25)))
    return local,prior,valid,cfg


def test_route_goal_enabled_without_certifying_blind_origin():
    m,p,v,c=setup()
    old=NavigationObjectiveV2(m,c['costs'],torch.tensor([[1.5,0.]]))
    assert not old.visible_goal.any()
    new=NavigationObjectiveV3(m,c['costs'],torch.tensor([[1.5,0.]]),p,v)
    x=p.clone().requires_grad_(True);new.bind_reference(x)
    assert new.visible_goal.item()==1
    assert abs(new.route_target[0,0].item()-1.05)<1e-5
    grad=torch.autograd.grad(new.terms(x)['goal'].sum(),x)[0]
    assert grad[0,8,0]<0
    assert not m.known[0,40,80]  # Origin remains unknown.


def test_route_gate_blocks_wall_and_invalid_prior():
    m,p,v,c=setup(wall=True)
    for valid in [v,torch.zeros_like(v)]:
        o=NavigationObjectiveV3(m,c['costs'],None,p,valid);o.bind_reference(p)
        assert not o.visible_goal.any()
        assert torch.isfinite(o(p)).all()


def test_route_guidance_moves_executed_endpoint_forward():
    m,p,v,c=setup()
    g=BridgeEnergyGuidanceV3(m,torch.tensor([[1.5,0.]]),c,.1,p,v)
    x=encode_trajectory(p,.1);y=g(x,torch.tensor(.3),'predictor')
    assert decode_trajectory(y,.1)[0,8,0]>decode_trajectory(x,.1)[0,8,0]
    assert g.records[-1]['route_active']==[1.]
    assert g.records[-1]['after']['goal'][0]<g.records[-1]['before']['goal'][0]


def test_budget_is_cumulative_across_sampler_calls():
    m,p,v,c=setup(route=False,robust=True,wall=True)
    g=BridgeEnergyGuidanceV3(m,torch.tensor([[1.5,0.]]),c,.1,p,v)
    x=encode_trajectory(p,.1)
    for _ in range(30):g(x,torch.tensor(.3),'predictor')
    assert 0<g.spent<=.050001
    assert sum(max(a['max_shift_m']) for a in g.records)<=.050001


def test_memory_rotation_translation_and_unknown_preservation():
    occ=torch.zeros(1,121,121,dtype=torch.bool);occ[0,80,60]=True # x=1,y=0
    old=LocalMap.from_masks(occ,occ,.025,-1.,1.5)
    empty=LocalMap.from_masks(torch.zeros_like(occ),torch.zeros_like(occ),.025,-1.,1.5)
    memory=ObservationMapMemory(3);memory.add(old,[0.,0.,0.]);memory.add(empty,[.5,0.,torch.pi/2])
    fused=memory.current()
    hits=(fused.sdf[0]<0).nonzero()
    xy=torch.stack([-1.+hits[:,0]*.025,1.5-hits[:,1]*.025],-1)
    assert ((xy-torch.tensor([0.,-.5])).norm(dim=-1)<.04).any()
    assert not fused.known[0,40,60]  # Fusion must not create known robot footprint.
    assert torch.equal(old.known,occ)


def test_v3_disabled_components_match_v2():
    from model.bridge_guidance.bridge_energy import BridgeEnergyGuidance
    m,p,v,c=setup(route=False,wall=True)
    a=BridgeEnergyGuidance(m,torch.tensor([[1.5,0.]]),c,.1)
    b=BridgeEnergyGuidanceV3(m,torch.tensor([[1.5,0.]]),c,.1,p,v)
    x=encode_trajectory(p,.1)
    assert torch.equal(a(x,torch.tensor(.3),'predictor'),b(x,torch.tensor(.3),'predictor'))
