import torch

from model.navibridger import NaviBridgeConfig, NaviBridgeDenoiser, waypoints_to_normalized_deltas


class DummyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.local_conditions = []

    def forward(self, func_name, **kwargs):
        self.calls += 1
        assert func_name == "noise_pred_net"
        assert "local_cond" in kwargs
        if kwargs["local_cond"] is not None:
            assert kwargs["local_cond"].shape == kwargs["sample"].shape
        self.local_conditions.append(kwargs["local_cond"])
        assert kwargs["global_cond"].shape[0] == kwargs["sample"].shape[0]
        return torch.zeros_like(kwargs["sample"])


def test_astar_waypoints_use_existing_delta_normalization():
    waypoints = torch.tensor([[[1.0, 0.0], [3.0, 2.0]]])
    actual = waypoints_to_normalized_deltas(waypoints, [-2.5, -4.0], [5.0, 4.0])
    expected = torch.tensor([[[-0.0666667, 0.0], [0.2, 0.5]]])
    torch.testing.assert_close(actual, expected)


def test_bridge_loss_and_sampling_preserve_trajectory_shape():
    policy = DummyPolicy()
    bridge = NaviBridgeDenoiser(NaviBridgeConfig(num_inference_steps=3))
    x0 = torch.randn(2, 8, 2)
    xT = torch.randn(2, 8, 2).clamp(-1, 1)
    local_condition = torch.randn(2, 8, 2)
    condition = torch.randn(2, 16)
    loss = bridge.training_loss(policy, x0, xT, condition, local_cond=local_condition)
    assert loss.ndim == 0 and torch.isfinite(loss)
    sample = bridge.sample(policy, xT, condition, local_cond=local_condition)
    assert sample.shape == xT.shape
    assert torch.isfinite(sample).all()
    assert all(actual is local_condition for actual in policy.local_conditions)


def test_bridge_loss_uses_navibridger_default_karras_weighting():
    bridge = NaviBridgeDenoiser(NaviBridgeConfig(sigma_data=0.5, sigma_max=10.0))
    sigma = torch.tensor([2.0])
    torch.testing.assert_close(bridge._loss_weights(sigma), sigma.reciprocal().square() + 0.5**-2)


def test_bridge_scalings_match_navibridger_ve_formulas():
    config = NaviBridgeConfig(sigma_data=0.5, sigma_max=10.0)
    bridge = NaviBridgeDenoiser(config)
    sigma = torch.tensor([2.0])
    ratio = (sigma / config.sigma_max).square()
    variance = (
        ratio.square() * config.sigma_data**2
        + (1.0 - ratio).square() * config.sigma_data**2
        + sigma.square() * (1.0 - ratio)
    )
    expected = (
        (1.0 - ratio) * config.sigma_data**2 / variance,
        (
            ratio.square() * config.sigma_data**4
            + config.sigma_data**2 * sigma.square() * (1.0 - ratio)
        ).sqrt()
        / variance.sqrt(),
        variance.rsqrt(),
    )
    for actual, target in zip(bridge._scalings(sigma), expected):
        torch.testing.assert_close(actual, target)


def test_churn_matches_navibridger_sampling_structure():
    xT = torch.zeros(2, 8, 2)
    local_condition = torch.randn(2, 8, 2)
    condition = torch.randn(2, 16)

    deterministic_policy = DummyPolicy()
    deterministic = NaviBridgeDenoiser(
        NaviBridgeConfig(num_inference_steps=3, churn_step_ratio=0.0)
    )
    deterministic.sample(
        deterministic_policy, xT, condition, local_cond=local_condition
    )
    assert deterministic_policy.calls == 5

    churn_policy = DummyPolicy()
    churned = NaviBridgeDenoiser(
        NaviBridgeConfig(num_inference_steps=3, churn_step_ratio=0.33)
    )
    sample = churned.sample(churn_policy, xT, condition, local_cond=local_condition)
    assert churn_policy.calls == 8
    assert torch.isfinite(sample).all()
    assert not torch.equal(sample[0], sample[1])


def test_astar_local_condition_switch_does_not_remove_bridge_endpoint():
    policy = DummyPolicy()
    bridge = NaviBridgeDenoiser(
        NaviBridgeConfig(
            num_inference_steps=1,
            churn_step_ratio=0.0,
            use_astar_local_cond=False,
        )
    )
    xT = torch.zeros(2, 8, 2)
    condition = torch.randn(2, 16)
    local_condition = torch.randn_like(xT)
    sample = bridge.sample(policy, xT, condition, local_cond=local_condition)
    assert sample.shape == xT.shape
    assert policy.local_conditions == [None]


def test_astar_local_condition_dropout_is_training_only():
    policy = DummyPolicy()
    bridge = NaviBridgeDenoiser(
        NaviBridgeConfig(astar_local_cond_dropout_prob=1.0)
    )
    x0 = torch.randn(2, 8, 2)
    xT = torch.randn(2, 8, 2).clamp(-1, 1)
    condition = torch.randn(2, 16)
    local_condition = torch.randn_like(xT)

    policy.train()
    bridge.training_loss(policy, x0, xT, condition, local_cond=local_condition)
    assert torch.count_nonzero(policy.local_conditions[-1]) == 0

    policy.eval()
    bridge.training_loss(policy, x0, xT, condition, local_cond=local_condition)
    assert policy.local_conditions[-1] is local_condition
