import subprocess
import types
import numpy as np
import pytest
import torch
from model.navibridger import NaviBridgeConfig,NaviBridgeDenoiser
from model.depth_bridge_guidance import DepthMap,depth_to_map,decode_metric,BridgeDepthGuidance
from test_navibridger import DummyPolicy


def local_map(wall=False):
    occ=np.zeros((101,101),bool)
    if wall:occ[60:62,:]=True
    m=DepthMap(occ,np.ones_like(occ),.025,-1.,1.25,'cpu')
    m.valid_fraction=1.
    return m


def encode(points):
    p=torch.tensor(points,dtype=torch.float32)
    d=torch.diff(torch.cat([torch.zeros_like(p[:,:1]),p],1),dim=1)/.045
    return 2*(d-d.new_tensor([-2.5,-4.]))/d.new_tensor([7.5,8.])-1


def test_metric_decode_has_0045_scale_and_correct_first_displacement():
    x=encode([[[.045,0.],[.09,.045]]])
    torch.testing.assert_close(decode_metric(x,.045),torch.tensor([[[0.,0.],[.045,0.],[.09,.045]]]))


@pytest.mark.parametrize('with_disabled_hook',[False,True])
def test_exact_original_bridge_parity_and_endpoint_preservation(with_disabled_hook):
    source=subprocess.check_output(['git','show','11f0f4b:model/navibridger.py'],text=True)
    original=types.ModuleType('_original_bridge')
    import sys
    sys.modules[original.__name__]=original
    exec(compile(source,'original_bridge','exec'),original.__dict__)
    cfg=dict(num_inference_steps=10,churn_step_ratio=.33)
    x=torch.randn(32,8,2).clamp(-1,1);frozen=x.clone();condition=torch.zeros(32,4)
    torch.manual_seed(712)
    a=original.NaviBridgeDenoiser(original.NaviBridgeConfig(**cfg)).sample(DummyPolicy(),x,condition,local_cond=x)
    guide=BridgeDepthGuidance(local_map(),[[1.,0.]],dict(scale=0)) if with_disabled_hook else None
    torch.manual_seed(712)
    b=NaviBridgeDenoiser(NaviBridgeConfig(**cfg)).sample(DummyPolicy(),x,condition,local_cond=x,energy_guidance=guide)
    assert torch.equal(a,b)
    assert torch.equal(frozen,x)


def test_projection_uses_axial_depth_and_left_positive():
    # Pixel two columns left: optical Z=.5, local y=+.2 (no ray normalization).
    depth=np.zeros((5,5),np.float32);depth[2,0]=.5
    k=np.array([[5.,0.,2.],[0.,5.,2.],[0.,0.,1.]])
    m=depth_to_map(depth,k,{},'cpu')
    point=torch.tensor([[[.5,.2]]])
    assert m.query(m.sdf,point)[0].item()<0
    assert m.query(m.sdf,torch.tensor([[[.5,-.2]]]))[0].item()>0


def test_invalid_depth_never_creates_obstacles_or_free_evidence():
    depth=np.full((8,8),np.nan,np.float32)
    m=depth_to_map(depth,np.array([[4.,0.,4.],[0.,4.,4.],[0.,0.,1.]]),{},'cpu')
    assert m.occupied_cells==0 and m.known_fraction==0 and m.valid_fraction==0
    guide=BridgeDepthGuidance(m,[[1.,0.]],{})
    x=torch.zeros(32,8,2)
    assert guide(x,torch.ones(32)*.1,'predictor') is x


def test_mean_path_guide_checks_clipped_proposal_and_bounds_every_candidate():
    guide=BridgeDepthGuidance(local_map(True),[[1.,0.]],dict(max_update_m=.01))
    points=[[[.10,0.],[.20,0.],[.30,0.],[.40,0.]]]
    x=encode(points).repeat(32,1,1)
    before=x.clone();result=guide(x,torch.full((32,),.4),'corrector')
    assert torch.equal(x,before)
    assert torch.isfinite(result).all()
    assert (result.abs()<=1).all()
    shift=(decode_metric(result,.045)-decode_metric(x,.045)).norm(dim=-1).max()
    assert shift<=.0100001
    row=guide.records[-1]
    assert row['after']['collision']<=row['before']['collision']+1e-7
    assert row['max_update_m']>0


def test_high_noise_bridge_endpoint_is_not_guided():
    guide=BridgeDepthGuidance(local_map(True),[[1.,0.]],{})
    x=torch.zeros(32,8,2)
    assert guide(x,torch.full((32,),9.99),'churn') is x
    assert not guide.records


def test_guidance_applies_to_churn_predictor_and_corrector_without_extra_network_calls():
    policy=DummyPolicy();stages=[]
    def identity(x,sigma,stage):stages.append(stage);return x
    x=torch.zeros(32,8,2)
    bridge=NaviBridgeDenoiser(NaviBridgeConfig(num_inference_steps=3))
    bridge.sample(policy,x,torch.zeros(32,4),local_cond=x,energy_guidance=identity)
    assert policy.calls==8
    assert stages.count('churn')==3 and stages.count('predictor')==3 and stages.count('corrector')==2


def test_no_obstacle_no_visible_goal_has_no_artificial_shrink_gradient():
    m=local_map();m.clearance.zero_()
    guide=BridgeDepthGuidance(m,[[3.,0.]],{})
    x=encode([[[.10,0.],[.20,0.],[.30,0.],[.40,0.]]]).repeat(32,1,1)
    result=guide(x,torch.full((32,),.4),'predictor')
    assert torch.equal(result,x)
    assert guide.records[-1]['gradient_norm']==0


def test_evaluation_model_output_forwards_bridge_guidance():
    from training.train_utils import model_output
    class Policy(torch.nn.Module):
        def forward(self,name,**kwargs):
            if name=='vision_encoder':return torch.zeros(1,8)
            if name=='noise_pred_net':return torch.zeros_like(kwargs['sample'])
            if name=='dist_pred_net':return torch.zeros(len(kwargs['obsfloorplan_cond']),1)
            raise AssertionError(name)
    calls=[]
    def hook(x,sigma,stage):calls.append(stage);return x
    result=model_output(Policy(),None,torch.zeros(1,15,98,98),torch.zeros(1,3,96,96),
        8,2,32,torch.zeros(1,2),torch.zeros(1,2),torch.ones(1,2),torch.zeros(1,8,5),
        torch.device('cpu'),bridge_diffusion=NaviBridgeDenoiser(NaviBridgeConfig(num_inference_steps=3)),
        energy_guidance=hook)
    assert result['actions'].shape==(32,8,2)
    assert len(calls)==8
