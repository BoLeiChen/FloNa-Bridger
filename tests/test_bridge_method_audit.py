"""Independent mathematical and inference-boundary checks for the method audit."""
import pytest
import torch

from model.bridge_diffusion import BridgeConfig, DiffusionBridge, encode_trajectory
from model.pointgoal_bridge import PointBridgePlanner
from test_pointgoal_bridge import batch, config


def test_bridge_drift_matches_conditional_gaussian_score():
    # q(x_s | x0,xT) = N((1-r)x0+r*xT, s^2(1-r)), r=s^2/T^2.
    # With forward bridge drift f=2s(xT-x)/(T^2-s^2), g^2=2s:
    # probability-flow drift is f-g^2*score/2; reverse-SDE is f-g^2*score.
    bridge = DiffusionBridge(BridgeConfig())
    torch.manual_seed(10)
    x, clean, endpoint = [torch.randn(3, 16, 2, dtype=torch.float64) for _ in range(3)]
    sigma = torch.tensor([.002, 2., 9.99], dtype=torch.float64)
    s = sigma[:, None, None]
    ratio = s.square() / bridge.config.sigma_max**2
    score = ((1-ratio)*clean + ratio*endpoint-x) / (s.square()*(1-ratio))
    forward = 2*s*(endpoint-x)/(bridge.config.sigma_max**2-s.square())
    torch.testing.assert_close(bridge.derivative(x, clean, endpoint, sigma), forward-s*score,
                               rtol=1e-10, atol=1e-8)
    torch.testing.assert_close(bridge.derivative(x, clean, endpoint, sigma, stochastic=True),
                               forward-2*s*score, rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize('churn', [0., .33, .8])
def test_oracle_denoiser_recovers_expert_from_different_priors(churn):
    target = batch()['gt_pose']
    target[..., 1] = .2 * target[..., 0].square()

    class OracleBridge(DiffusionBridge):
        def denoise(self, net, x, sigma, local, global_cond):
            return encode_trajectory(target).expand_as(x)

    bridge = OracleBridge(BridgeConfig(churn=churn))
    for prior in (torch.zeros_like(target), target.flip(-1), target*2):
        torch.manual_seed(22)
        output = bridge.sample(None, prior, None, None)
        torch.testing.assert_close(output, target, atol=2e-5, rtol=1e-5)


def test_loss_uses_bridge_marginal_and_weighted_clean_target():
    target = batch()['gt_pose']
    prior = torch.zeros_like(target)
    bridge = DiffusionBridge(BridgeConfig())
    captured = {}
    def denoise(net, x, sigma, local, global_cond):
        captured.update(x=x, sigma=sigma)
        return torch.zeros_like(x)
    bridge.denoise = denoise
    torch.manual_seed(31)
    loss, _ = bridge.loss(None, target, prior, None, None)
    torch.manual_seed(31)
    sigma = torch.empty(1).uniform_(.002, 10.)
    noise = torch.randn_like(target)
    ratio = (sigma[:, None, None]/10).square()
    x0, endpoint = encode_trajectory(target), encode_trajectory(prior)
    expected = (1-ratio)*x0 + ratio*endpoint + sigma[:, None, None]*(1-ratio).sqrt()*noise
    torch.testing.assert_close(captured['x'], expected)
    torch.testing.assert_close(loss, ((sigma[:, None, None].reciprocal().square()+4)*x0.square()).mean())


def test_inference_never_reads_expert_labels():
    planner = PointBridgePlanner(config(), load_pretrained=False).eval()
    data = batch()
    torch.manual_seed(81)
    expected = planner(data, mode='sample')
    torch.manual_seed(81)
    actual = planner({k: v for k, v in data.items() if k != 'gt_pose'}, mode='sample')
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.manual_seed(81)
    actual = planner(dict(data, gt_pose=torch.full_like(data['gt_pose'], float('nan'))), mode='sample')
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
