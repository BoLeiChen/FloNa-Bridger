import numpy as np
import torch

from model.data_utils import path_xy_to_local_condition, sample_path_prefix_points
from model.flona_vint import CurvatureResidualFusion, compute_path_turn_risk
from training.train_utils import apply_paper_train_perturbations


def test_sample_path_prefix_valid_mask_marks_terminal_padding():
    path = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    sampled, valid = sample_path_prefix_points(
        path, num_points=4, step_size_m=1.0, return_valid_mask=True
    )

    np.testing.assert_allclose(sampled[:, 0], [1.0, 2.0, 2.0, 2.0])
    np.testing.assert_array_equal(valid, [1.0, 1.0, 0.0, 0.0])


def test_local_condition_encodes_signed_turn_and_masks_padding():
    path = np.array(
        [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [2.0, 1.0]],
        dtype=np.float32,
    )
    valid = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    condition = path_xy_to_local_condition(path, valid)

    assert condition.shape == (4, 5)
    np.testing.assert_allclose(condition[:, :2], path)
    np.testing.assert_allclose(condition[1, 2:4], [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(condition[-1, 2:], [0.0, 0.0, 0.0])


def test_training_perturbation_preserves_five_dimensional_semantics():
    local_condition = torch.tensor(
        [[[1.0, 0.0, 0.0, 1.0, 1.0],
          [2.0, 0.0, 0.0, 1.0, 1.0],
          [2.0, 1.0, 1.0, 0.0, 1.0],
          [2.0, 1.0, 0.0, 0.0, 0.0]]]
    )
    config = {
        "metric_waypoint_spacing": 0.045,
        "waypoint_spacing": 1,
        "pose_noise_apply": False,
        "path_noise_apply": False,
        "path_noise_alpha_m": 0.0,
    }
    _, _, actual, noise_bound = apply_paper_train_perturbations(
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
        local_condition,
        schedule_epoch=1,
        schedule_total_epochs=20,
        perturb_config=config,
    )

    assert actual.shape == (1, 4, 5)
    torch.testing.assert_close(actual[..., :2], local_condition[..., :2])
    torch.testing.assert_close(actual[..., 4], local_condition[..., 4])
    torch.testing.assert_close(actual[0, 1, 2:4], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(actual[0, -1, 2:4], torch.tensor([0.0, 0.0]))
    assert noise_bound == 0.0


def test_turn_risk_ignores_invalid_padding_and_increases_with_curvature():
    straight = torch.tensor(
        [[[1.0, 0.0, 0.0, 1.0, 1.0],
          [2.0, 0.0, 0.0, 1.0, 1.0],
          [3.0, 0.0, 0.0, 1.0, 1.0],
          [4.0, 0.0, 0.0, 1.0, 1.0],
          [4.0, 0.0, 0.0, 0.0, 0.0]]]
    )
    turning = straight.clone()
    turning[0, 2, :2] = torch.tensor([2.0, 1.0])
    turning[0, 3, :2] = torch.tensor([2.0, 2.0])

    straight_risk = compute_path_turn_risk(straight)
    turning_risk = compute_path_turn_risk(turning)

    torch.testing.assert_close(straight_risk, torch.zeros_like(straight_risk))
    assert turning_risk.item() > straight_risk.item()


def test_curvature_fusion_starts_at_exact_deep_baseline_without_hard_prior():
    gate = CurvatureResidualFusion(feature_dim=8, shallow_dim=12, attention_heads=3)
    shallow = torch.randn(2, 7, 12)
    deep = torch.randn(2, 3, 8)
    risk = torch.tensor([0.0, 0.5])

    fused, shallow_weight, attention = gate(shallow, deep, risk)

    torch.testing.assert_close(fused, deep)
    assert shallow_weight.shape == (2,)
    torch.testing.assert_close(shallow_weight[0], shallow_weight[1])
    assert torch.all((shallow_weight >= 0.0) & (shallow_weight <= 0.25))
    assert attention.shape == (2, 7)


def test_structured_path_noise_is_continuous_and_keeps_padded_tail_together():
    condition = torch.tensor(
        [[[1.0, 0.0, 0.0, 1.0, 1.0],
          [2.0, 0.0, 0.0, 1.0, 1.0],
          [3.0, 0.0, 0.0, 1.0, 1.0],
          [4.0, 0.0, 0.0, 1.0, 1.0],
          [4.0, 0.0, 0.0, 0.0, 0.0],
          [4.0, 0.0, 0.0, 0.0, 0.0]]]
    )
    config = {
        "metric_waypoint_spacing": 0.045,
        "waypoint_spacing": 1,
        "pose_noise_apply": False,
        "path_noise_apply": True,
        "path_noise_mode": "structured",
        "path_noise_alpha_m": 0.02,
        "path_noise_control_points": 3,
        "path_noise_translation_std_m": 0.0,
    }
    torch.manual_seed(4)
    _, _, perturbed, bound = apply_paper_train_perturbations(
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
        condition,
        schedule_epoch=20,
        schedule_total_epochs=20,
        perturb_config=config,
    )

    assert np.isclose(bound, 0.02)
    displacement = perturbed[..., :2] - condition[..., :2]
    adjacent_change = torch.linalg.vector_norm(
        displacement[:, 1:4] - displacement[:, :3], dim=-1
    )
    assert adjacent_change.max().item() < 0.02 / 0.045
    torch.testing.assert_close(perturbed[0, 4, :2], perturbed[0, 3, :2])
    torch.testing.assert_close(perturbed[0, 5, :2], perturbed[0, 3, :2])
