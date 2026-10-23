"""Unit-aware bridge endpoints and backwards-compatible checkpoint semantics."""
import pytest
import torch
from model.bridge_diffusion import BridgeConfig, DiffusionBridge, encode_trajectory, decode_trajectory


def path():
    x=torch.arange(16,dtype=torch.float32)*.1
    return torch.stack((x,.1*x.square()),-1)[None]


@pytest.mark.parametrize('scale',[1.,.1,.045])
def test_metric_roundtrip(scale):
    target=path()
    torch.testing.assert_close(decode_trajectory(encode_trajectory(target,scale),scale),target,atol=2e-6,rtol=1e-6)


def test_waypoint_unit_reference_and_legacy_default():
    target=path()
    delta=torch.diff(target/.1,dim=1,prepend=torch.zeros_like(target[:,:1]))
    expected=2*(delta-torch.tensor([-2.5,-4.]))/torch.tensor([7.5,8.])-1
    torch.testing.assert_close(encode_trajectory(target,.1),expected,rtol=0,atol=0)
    assert BridgeConfig().action_step_m==1.
    torch.testing.assert_close(encode_trajectory(target),encode_trajectory(target,1.),rtol=0,atol=0)


@pytest.mark.parametrize('scale',[0.,-1.,float('nan'),float('inf')])
def test_invalid_scale_rejected(scale):
    with pytest.raises(ValueError): BridgeConfig(action_step_m=scale)
    with pytest.raises(ValueError): encode_trajectory(path(),scale)
    with pytest.raises(ValueError): decode_trajectory(path(),scale)


def test_both_loss_endpoints_are_scaled_and_prediction_is_metric():
    target=path();prior=target.flip(-1);bridge=DiffusionBridge(BridgeConfig(action_step_m=.1));seen={}
    def denoise(net,x,sigma,local,global_cond):
        seen.update(x=x,sigma=sigma)
        return encode_trajectory(target,.1)
    bridge.denoise=denoise
    torch.manual_seed(17)
    loss,prediction=bridge.loss(None,target,prior,None,None)
    torch.manual_seed(17)
    sigma=torch.empty(1).uniform_(.002,10.)
    noise=torch.randn_like(target);s=sigma[:,None,None];r=(s/10).square()
    expected=(1-r)*encode_trajectory(target,.1)+r*encode_trajectory(prior,.1)+s*(1-r).sqrt()*noise
    torch.testing.assert_close(seen['x'],expected)
    torch.testing.assert_close(prediction,target,atol=2e-6,rtol=1e-6)
    assert loss.item()==0.


@pytest.mark.parametrize('churn',[0.,.33])
def test_sampler_initial_prior_and_final_units(churn):
    target=path();prior=target.flip(-1);bridge=DiffusionBridge(BridgeConfig(action_step_m=.1,churn=churn));seen=[]
    def denoise(net,x,sigma,local,global_cond):
        seen.append(x.clone())
        return encode_trajectory(target,.1)
    bridge.denoise=denoise
    torch.manual_seed(33)
    output=bridge.sample(None,prior,None,None)
    torch.testing.assert_close(seen[0],encode_trajectory(prior,.1),atol=0,rtol=0)
    torch.testing.assert_close(output,target,atol=2e-5,rtol=1e-5)


def test_checkpoint_config_carries_scale(tmp_path):
    import dataclasses
    f=tmp_path/'checkpoint.pt'
    torch.save({'config':{'bridge':dataclasses.asdict(BridgeConfig(action_step_m=.1))}},f)
    restored=BridgeConfig(**torch.load(f)['config']['bridge'])
    assert restored.action_step_m==.1
    assert BridgeConfig(**{'steps':10}).action_step_m==1.
