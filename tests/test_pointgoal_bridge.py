"""Behavioral tests for bridge math, calibrated planning, and RGB-only conditioning and Planner gradients."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / 'diffusion_policy'))

import numpy as np
import pytest
import torch
from PIL import Image

from datasets.bridge_maps import FloorplanPlanner, prepare_map, sample_prefix, world_to_local, local_to_world
from model.bridge_diffusion import BridgeConfig, DiffusionBridge, encode_trajectory, decode_trajectory
from model.pointgoal_bridge import PointBridgePlanner, reliability_target, path_features
from model.vgc_rgb_condition import LocalConditionTrustGate, compute_path_turn_risk

torch.set_num_threads(2)


def config():
    return dict(point_goal_train_prob=1., img_goal_train_prob=0., object_goal_train_prob=0.,
                history_length=3, global_coordinate_mode="world", len_traj_pred=16, condition_dim=32, mha_num_attention_layers=1,
                down_dims=[16, 32, 64], bridge=dict(steps=2, churn=0.),
                occupancy_height=40, occupancy_width=50, grid_size=0.1,
                camera_intrinsics=dict(camera_fx=128, camera_fy=128, camera_cx=128, camera_cy=128))


def batch():
    path = torch.zeros(1, 16, 2)
    path[0, :, 0] = torch.arange(16) * 0.1
    return dict(rgb=torch.rand(1, 3, 3, 96, 96), current_rgb=torch.rand(1, 3, 96, 96),
                gt_pose=path, prior=path.clone(), prior_valid=torch.ones(1, 16),
                agent_pose=torch.tensor([[2., -1., .5]]), goal_world=torch.tensor([[4., 2.]]),
                point_goal=torch.tensor([[[3., 0.]]]), floorplan=torch.rand(1, 3, 96, 96))



def test_metric_delta_round_trip():
    path = torch.tensor([[[0., 0.], [0.1, -0.2], [0.3, -0.1], [0.3, -0.1]]])
    torch.testing.assert_close(decode_trajectory(encode_trajectory(path)), path, atol=1e-6, rtol=1e-5)


class ZeroNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, sample, **kwargs):
        self.calls += 1
        return torch.zeros_like(sample)


def test_bridge_endpoint_controls_samples_and_solver_call_count():
    bridge = DiffusionBridge(BridgeConfig(steps=3, churn=0.33))
    prior = batch()['prior']
    net = ZeroNet()
    torch.manual_seed(42)
    first = bridge.sample(net, prior, None, None)
    assert net.calls == 8
    changed = prior.clone()
    changed[..., 1] = prior[..., 0]
    torch.manual_seed(42)
    second = bridge.sample(net, changed, None, None)
    assert torch.isfinite(first).all() and not torch.equal(first, second)
    assert torch.count_nonzero(first[:, 0]) == 0


@pytest.mark.parametrize('kwargs', [dict(steps=1), dict(churn=1), dict(sigma_min=11), dict(sigma_data=0)])
def test_invalid_bridge_config_fails(kwargs):
    with pytest.raises(ValueError):
        BridgeConfig(**kwargs)


def make_map(tmp_path, free):
    directory = tmp_path / 'map'
    directory.mkdir()
    np.save(directory / 'walkable.npy', np.asarray(free, np.uint8))
    size = len(free)
    (directory / 'metadata.json').write_text(json.dumps(dict(size=size, source_size=[size * 10, size * 10],
                                                            scale=100, offset=[size * 5, size * 5])))
    return FloorplanPlanner(directory, max_snap_m=0.01)


def test_calibration_round_trip(tmp_path):
    planner = make_map(tmp_path, np.ones((20, 20)))
    points = np.array([[0., 0.], [0.12, -0.2]])
    np.testing.assert_allclose(planner.to_world(planner.to_pixel(points)), points, atol=1e-7)
    local = world_to_local(points, np.array([1., 2.]), 1.2)
    np.testing.assert_allclose(local_to_world(local, np.array([1., 2.]), 1.2), points, atol=1e-7)


def test_astar_cannot_cut_diagonal_wall_corners(tmp_path):
    free = np.eye(4, dtype=bool)
    planner = make_map(tmp_path, free)
    path = planner.astar(planner.to_world([0, 0]), planner.to_world([3, 3]))
    assert len(path) == 0


def test_astar_and_cached_goal_tree_have_same_path_cost(tmp_path):
    free = np.ones((20, 20), bool)
    free[:15, 10] = False
    planner = make_map(tmp_path, free)
    start, goal = planner.to_world([[2, 2], [18, 2]])
    first = planner.astar(start, goal)
    second = planner.path_from_tree(start, planner.goal_tree(goal))
    cost = lambda path: np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    np.testing.assert_allclose(cost(first), cost(second), rtol=1e-6)
    assert len(first) > 0


def test_failure_prior_is_stationary_and_marked_invalid(tmp_path):
    planner = make_map(tmp_path, np.zeros((20, 20)))
    prior, valid = planner.prior(np.zeros(2), 0., np.ones(2))
    assert np.count_nonzero(prior) == 0 and np.count_nonzero(valid) == 0


def test_prefix_origin_metric_step_and_terminal_mask():
    prior, valid = sample_prefix(np.array([[0., 0.], [0.25, 0.]]), np.zeros(2), 0, 5, 0.1)
    np.testing.assert_allclose(prior[:, 0], [0, .1, .2, .25, .25])
    np.testing.assert_array_equal(valid, [1, 1, 1, 0, 0])


def test_prepare_preserves_thin_wall_and_fails_without_calibration(tmp_path):
    root, assets, output = tmp_path / 'data', tmp_path / 'assets', tmp_path / 'cache'
    (root / 'room_0').mkdir(parents=True)
    image = np.full((1600, 1600), 255, np.uint8)
    image[:, 800:803] = 0
    image[500, 500] = 0
    Image.fromarray(image).save(root / 'room_0/floorplan.png')
    with pytest.raises(ValueError, match='calibration'):
        prepare_map(root, 'room_0', assets, output)
    (assets / 'room').mkdir(parents=True)
    (assets / 'room/scale.txt').write_text('100\n')
    (assets / 'room/offset.txt').write_text('800 800\n')
    directory = prepare_map(root, 'room_0', assets, output, clearance_m=0)
    planner = FloorplanPlanner(directory)
    assert planner.walkable.shape == (256, 256)
    assert not planner.walkable[:, 128].any()
    assert not planner.walkable[80, 80]
    assert Image.open(root / 'room_0/floorplan.png').size == (1600, 1600)


def test_path_gate_preserves_xy_and_direction_and_invalid_padding():
    gate = LocalConditionTrustGate(visual_dim=32)
    features = torch.randn(1, 16, 5)
    features[..., 4] = 1
    features[:, -2:, 4] = 0
    aux = gate(torch.randn(1, 9, 32), features)
    local = aux['gated_local_cond']
    torch.testing.assert_close(local[..., :4], features[..., :4])
    assert torch.count_nonzero(local[:, -2:, 4]) == 0
    assert (local[:, :-2, 4] >= 0.5).all()


def trusted_turn_summary(path, valid, trust):
    return compute_path_turn_risk(path_features(path, valid), waypoint_trust=trust)


def test_reliability_is_geometry_based_not_same_sample_index():
    expert = torch.tensor([[[0., 0.], [.1, 0.], [.2, 0.], [.3, 0.]]])
    path = torch.tensor([[[0., 0.], [.05, 0.], [.1, 0.], [.15, 0.]]])
    target = reliability_target(path, expert, torch.ones(1, 4))
    torch.testing.assert_close(target, torch.ones_like(target))


def test_global_curvature_respects_confidence_and_terminal_padding():
    path = batch()['prior'].clone()
    path[:, 8:, 0] = 0.7
    path[:, 8:, 1] = torch.arange(8) * 0.1
    valid = torch.ones(1, 16)
    full = trusted_turn_summary(path, valid, valid)
    untrusted = trusted_turn_summary(path, valid, torch.zeros_like(valid))
    assert full.item() > 0 and untrusted.item() == 0
    valid[:, 8:] = 0
    torch.testing.assert_close(trusted_turn_summary(path, valid, valid), torch.zeros(1), atol=1e-6, rtol=0)


def test_pointgoal_only_rejects_other_goal_probabilities():
    cfg = config()
    cfg['object_goal_train_prob'] = 0.1
    with pytest.raises(ValueError, match='PointGoal'):
        PointBridgePlanner(cfg)


def test_rgb_planner_backward_sampling_and_frozen_backbone():
    data = batch()
    planner = PointBridgePlanner(config(), load_pretrained=False).train()
    assert not planner.obs_encoder.training
    assert all(not p.requires_grad for p in planner.obs_encoder.parameters())
    result = planner(data)
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    assert all(p.grad is not None for p in planner.parameters() if p.requires_grad)
    assert all(p.grad is None for p in planner.obs_encoder.parameters())
    planner.eval()
    with torch.no_grad():
        planned = planner(data, mode='sample')
    assert planned.shape == data['prior'].shape and torch.isfinite(planned).all()
    assert torch.count_nonzero(planned[:, 0]) == 0


def test_condition_uses_four_frames_and_floorplan_but_ignores_depth():
    planner = PointBridgePlanner(config(), load_pretrained=False).eval()
    data = batch()
    with torch.no_grad():
        local, baseline, aux = planner.conditions(data)
        # Initial residual is exactly zero; alpha is bounded, trainable, and nonzero.
        assert aux['shallow_attention'].shape == (1, 49)
        assert 0 < aux['shallow_weight'].item() <= .25
        injected = dict(data, current_depth=torch.full((1, 256, 256), float('nan')),
                        depth=torch.full((1, 4, 1, 96, 96), float('nan')))
        other_local, other, _ = planner.conditions(injected)
        torch.testing.assert_close(other, baseline, rtol=0, atol=0)
        torch.testing.assert_close(other_local, local, rtol=0, atol=0)
        for frame in range(4):
            changed = {k: v.clone() for k, v in data.items()}
            if frame < 3:
                changed['rgb'][:, frame] = 0
            else:
                changed['current_rgb'][:] = 0
            _, condition, _ = planner.conditions(changed)
            assert not torch.allclose(condition, baseline)
        _, map_changed, _ = planner.conditions(dict(data, floorplan=torch.zeros_like(data['floorplan'])))
        assert not torch.allclose(map_changed, baseline)
        with pytest.raises(KeyError, match='floorplan'):
            planner.conditions({k: v for k, v in data.items() if k != 'floorplan'})
        data['rgb'] = data['rgb'][:, :2]
        with pytest.raises(ValueError, match='four'):
            planner.conditions(data)


def test_pretrained_checkpoint_is_required_for_training(tmp_path):
    cfg = dict(config(), dinov2_checkpoint=str(tmp_path / 'missing.pth'))
    with pytest.raises(FileNotFoundError, match='Depth Anything'):
        PointBridgePlanner(cfg)


@pytest.mark.parametrize('rgb_floor', [False, True])
@pytest.mark.parametrize('pace', [4, 8])
def test_files_and_lmdb_bridge_samples_match_without_source_writes(tmp_path, rgb_floor, pace):
    import lmdb
    import msgpack
    from datasets.pointgoal_bridge_dataset import PointBridgeFiles, PointBridgeLMDB, select_training_manifest
    from datasets.lmdb_dataset import encode_record

    root = tmp_path / 'source'
    scene, traj = root / 'room_0', root / 'room_0/traj_0'
    (traj / 'rgb').mkdir(parents=True)
    (traj / 'object').mkdir()
    (traj / 'object/object.json').write_text('invalid, deliberately unused object metadata')
    floor_array = np.full((1600, 1600, 3) if rgb_floor else (1600, 1600), 255, np.uint8)
    if rgb_floor:
        floor_array[300:500, 300:500] = [120, 180, 220]
    Image.fromarray(floor_array).save(scene / 'floorplan.png')
    poses = np.array([[i * 0.1, 0., i * 0.1 + 1, 0.] for i in range(28)])
    np.savetxt(traj / 'traj_0.txt', poses)
    rgb, depth, native = [], [], []
    for index in range(28):
        image = Image.fromarray(np.full((256, 256, 3), 50 + index, np.uint8))
        image.save(traj / 'rgb' / ('rgb_%d.png' % index))
        rgb.append(np.asarray(image.resize((96, 96), Image.Resampling.BILINEAR)))
    assets = tmp_path / 'assets/room'
    assets.mkdir(parents=True)
    (assets / 'scale.txt').write_text('100\n')
    (assets / 'offset.txt').write_text('800 800\n')
    cfg = dict(config(), data_folder=[str(root)], traj_pace=pace, astar_step_m=0.1,
               map_cache=str(tmp_path / 'maps'), prior_cache=str(tmp_path / 'priors'), lmdb_path=str(tmp_path / 'lmdb'))
    prepare_map(root, 'room_0', assets.parent, cfg['map_cache'])
    before = sorted(str(p.relative_to(root)) for p in root.rglob('*'))
    files = PointBridgeFiles(cfg)
    expected = files[0]
    floor_key = 'floor::%s::room_0' % root
    manifest = [dict(key='traj0', floor_key=floor_key, sample_count=4)]
    record = dict(scene='room_0', traj_name='traj_0', poses=poses, rgb=np.stack(rgb),
                  current_indices=np.array([0, 4, 8, 12]), object_goal='null')
    env = lmdb.open(cfg['lmdb_path'], map_size=10 * 1024**2)
    with env.begin(write=True) as txn:
        txn.put(b'__manifest__', msgpack.packb(manifest, use_bin_type=True))
        small_floor = np.asarray(Image.fromarray(floor_array).resize((96, 96), Image.Resampling.BILINEAR)).astype(np.float32) / 255
        small_floor = small_floor.transpose(2, 0, 1) if rgb_floor else small_floor[None]
        txn.put(floor_key.encode(), encode_record(dict(floorplan=small_floor)))
        txn.put(b'traj0', encode_record(record))
    env.close()
    selected = select_training_manifest(manifest, cfg)
    lmdb_data = PointBridgeLMDB(cfg, selected, max_samples=selected[0]['sample_count'])
    actual_samples = list(lmdb_data)
    for index, actual in enumerate(actual_samples):
        expected = files[index]
        assert set(actual) == {'rgb', 'current_rgb', 'gt_pose', 'point_goal', 'sample_id', 'prior', 'prior_valid', 'floorplan', 'agent_pose', 'goal_world'}
        for key in expected:
            if torch.is_tensor(expected[key]):
                torch.testing.assert_close(actual[key], expected[key])
            else:
                assert actual[key] == expected[key]
        # Sparse starts must still use dense consecutive history frames.
        values = torch.cat([actual['rgb'][:, 0, 0, 0], actual['current_rgb'][0, :1, 0]])
        start = index * pace
        indices = [max(0, i) for i in range(start-3, start+1)]
        torch.testing.assert_close(values, torch.tensor([(50 + i) / 255 for i in indices]))
    assert before == sorted(str(p.relative_to(root)) for p in root.rglob('*'))


def test_planner_checkpoint_portability_and_legacy_rejection(tmp_path):
    from eval.pointgoal_bridge import load_models
    from model.pointgoal_bridge import CHECKPOINT_FORMAT
    cfg = dict(config(), dinov2_checkpoint=str(tmp_path / 'missing-pretrained.pth'))
    planner = PointBridgePlanner(cfg, load_pretrained=False).eval()
    checkpoint = tmp_path / 'planner.pth'
    torch.save(dict(format=CHECKPOINT_FORMAT, config=cfg, planner=planner.state_dict(), step=1), checkpoint)
    _, restored = load_models(checkpoint, torch.device('cpu'))
    data = batch()
    with torch.no_grad():
        torch.testing.assert_close(planner.conditions(data)[1], restored.conditions(data)[1])
    for legacy in ['floverse-pointgoal-bridge-v1', 'floverse-pointgoal-rgb-planner-v2', 'floverse-pointgoal-rgb-floorplan-planner-v3']:
        torch.save(dict(format=legacy), checkpoint)
        with pytest.raises(ValueError, match='incompatible'):
            load_models(checkpoint, torch.device('cpu'))


def test_turn_is_at_its_vertex_and_invalid_support_does_not_create_turns():
    path = torch.tensor([[[0., 0.], [1., 0.], [1., 1.], [1., 2.], [1., 2.]]])
    valid = torch.tensor([[1., 1., 1., 1., 0.]])
    features = path_features(path, valid)
    torch.testing.assert_close(features[0, :, 2], torch.tensor([0., 1., 0., 0., 0.]), atol=1e-6, rtol=0)
    torch.testing.assert_close(features[0, :, 3], torch.tensor([1., 0., 1., 1., 0.]), atol=1e-6, rtol=0)
    valid[:, 2:] = 0
    features = path_features(path, valid)
    torch.testing.assert_close(features[0, :, 2], torch.zeros(5))
    torch.testing.assert_close(features[0, :, 3], torch.tensor([1., 1., 0., 0., 0.]))


@pytest.mark.parametrize('yaw', [0., np.pi/2, -np.pi/2, np.pi, .73])
def test_training_goal_labels_prior_and_execution_share_ego_frame(yaw):
    from datasets.pointgoal_bridge_dataset import PointBridgeFiles, PointBridgeLMDB
    position = np.array([4.3, -2.7])
    ego = np.stack([np.arange(20)*.1, np.sin(np.arange(20)*.1)*.2], -1)
    world = local_to_world(ego, position, yaw)
    direction = np.array([np.cos(yaw), np.sin(yaw)])
    poses = np.concatenate([world, world + direction], -1)
    for cls in [PointBridgeFiles, PointBridgeLMDB]:
        dataset = cls.__new__(cls)
        dataset.len_traj_pred = 16
        target = dataset._future_pose_tensor(poses.tolist(), 0)
        point = dataset._point_goal_tensor(poses.tolist(), 0)
        np.testing.assert_allclose(target.numpy(), ego[:16], atol=1e-6)
        np.testing.assert_allclose(point.numpy()[0], ego[-1], atol=1e-6)
        np.testing.assert_allclose(local_to_world(target.numpy(), position, yaw), world[:16], atol=1e-6)
    prior, valid = sample_prefix(world, position, yaw, 16, .1)
    reference, reference_valid = sample_prefix(ego, np.zeros(2), 0., 16, .1)
    np.testing.assert_allclose(prior, reference, atol=1e-6)
    np.testing.assert_array_equal(valid, reference_valid)
    np.testing.assert_allclose(world_to_local(world[-1], position, yaw), ego[-1], atol=1e-6)


def test_floorplan_is_unmarked_and_does_not_mutate_cache(tmp_path):
    from datasets.pointgoal_bridge_dataset import mark_floorplan
    planner = make_map(tmp_path, np.ones((20, 20)))
    floor = torch.rand(1, 96, 96)
    original = floor.clone()
    position, goal = planner.to_world([[5, 5], [15, 10]])
    plain = mark_floorplan(floor, planner, position, 0., goal)
    torch.testing.assert_close(plain, original.expand(3, -1, -1), rtol=0, atol=0)
    moved = mark_floorplan(floor, planner, goal, 1.3, np.array([100., 100.]))
    torch.testing.assert_close(plain, moved, rtol=0, atol=0)
    plain.zero_()
    torch.testing.assert_close(floor, original, rtol=0, atol=0)


def test_hm3d_sparse_manifest_filters_before_rank_assignment(tmp_path):
    from datasets.pointgoal_bridge_dataset import select_training_manifest
    from datasets.lmdb_dataset import assign_lmdb_manifest
    root = str(tmp_path)
    raw = [dict(key=str(i), floor_key='floor::%s::%s' % (root, scene), sample_count=count)
           for i, (scene, count) in enumerate([
               ('00001-abcdef_0', 5), ('00002-ghijk_1', 4), ('Gibson_0', 100)])]
    cfg = dict(data_folder=[root], train_scene_dataset='hm3d', traj_pace=8, len_traj_pred=16)
    selected = select_training_manifest(raw, cfg)
    assert [x['sample_count'] for x in selected] == [3, 2]
    assert [x['source_sample_count'] for x in selected] == [5, 4]
    assert raw[0]['sample_count'] == 5  # immutable source manifest
    _, totals = assign_lmdb_manifest(selected, 2)
    assert sorted(totals) == [2, 3]
    with pytest.raises(ValueError, match='multiple'):
        select_training_manifest(raw, dict(cfg, traj_pace=6))
    with pytest.raises(ValueError, match='horizon'):
        select_training_manifest(raw, dict(cfg, len_traj_pred=32))


def test_world_coordinates_match_ddpm_and_ignore_ego_goal():
    model = PointBridgePlanner(config(), load_pretrained=False).eval()
    data = batch()
    captured = []
    hook = model.obs_goal_pos_ori_enc.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    with torch.no_grad():
        local, condition, _ = model.conditions(data)
        pose = data['agent_pose']; xy = pose[:, :2]
        endpoint = xy + torch.stack([pose[:, 2].cos(), pose[:, 2].sin()], -1)
        expected = torch.cat([xy / .1, data['goal_world'] / .1, endpoint / .1], -1)
        torch.testing.assert_close(captured[-1][:, -6:], expected, rtol=0, atol=0)
        _, unchanged, _ = model.conditions(dict(data, point_goal=data['point_goal'] * 100))
        torch.testing.assert_close(condition, unchanged, rtol=0, atol=0)
        for key in ('agent_pose', 'goal_world'):
            shifted = dict(data); shifted[key] = data[key] + .5
            local2, changed, _ = model.conditions(shifted)
            assert not torch.allclose(condition, changed)
            torch.testing.assert_close(local, local2, rtol=0, atol=0)
    hook.remove()
