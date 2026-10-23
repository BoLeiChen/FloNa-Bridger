"""PointGoal bridge under the original FloVerse evaluation protocol."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / 'diffusion_policy'))

import numpy as np
import torch
import pybullet as p
from PIL import Image
from omegaconf import OmegaConf

from datasets.bridge_maps import prepare_map, FloorplanPlanner, local_to_world, world_to_local
from datasets.pointgoal_bridge_dataset import mark_floorplan, load_floorplan_image
from eval.floverse_protocol import PROTOCOL, get_pose_from_position, move_to_fallback, score_trajectory
from model.pointgoal_bridge import PointBridgePlanner, CHECKPOINT_FORMAT


def capture(env, device):
    rgb, = env.simulator.renderer.render_robot_cameras(modes=('rgb',), cache=False)
    rgb = Image.fromarray(np.clip(rgb[:, :, :3] * 255, 0, 255).astype(np.uint8))
    def small(image):
        array = np.asarray(image.resize((96, 96), Image.Resampling.BILINEAR)).copy()
        if array.ndim == 2:
            array = array[:, :, None]
        return torch.from_numpy(array).permute(2, 0, 1).float().to(device) / 255
    return small(rgb)


def load_models(checkpoint, device):
    saved = torch.load(checkpoint, map_location='cpu')
    if saved.get('format') != CHECKPOINT_FORMAT:
        raise ValueError('Expected world-coordinate RGB4 Planner v4 checkpoint; legacy weights are incompatible')
    config = saved['config']
    planner = PointBridgePlanner(config, load_pretrained=False).to(device).eval()
    planner.load_state_dict(saved['planner'], strict=True)
    return config, planner


@torch.no_grad()
def rollout(env, trajectory, floor, map_planner, planner, config, args, directory, device, navigation_goal=None, navigation_goals=None):
    poses = np.atleast_2d(np.loadtxt(trajectory))
    start = 8  # Same navigation start as original; frame count is model-specific.
    if len(poses) <= start:
        raise ValueError('Original FloVerse protocol requires trajectory index 8')
    gt_goal = poses[-1, :2].copy()
    goal = gt_goal.copy() if navigation_goal is None else np.asarray(navigation_goal, dtype=np.float64).copy()
    if goal.shape != (2,) or not np.isfinite(goal).all():
        raise ValueError("Navigation goal must be finite world XY")
    robot = env.robots[0]
    history_length = int(config['history_length'])
    if history_length != 3:
        raise ValueError('This Planner uses three history frames plus current RGB')
    guidance_config = getattr(args, 'guidance_config', None)
    memory = None
    if guidance_config is not None and int(guidance_config.get('memory_frames', 0)) > 0:
        from model.bridge_guidance.geometry import ObservationMapMemory
        memory = ObservationMapMemory(int(guidance_config['memory_frames']))
    last_depth_pose = None
    def capture_observation():
        nonlocal last_depth_pose
        rgb = capture(env, device)
        if memory is not None:
            measured = np.array([*robot.get_position()[:2], float(robot.get_rpy()[2])])
            angle = 0. if last_depth_pose is None else np.arctan2(np.sin(measured[2]-last_depth_pose[2]), np.cos(measured[2]-last_depth_pose[2]))
            if last_depth_pose is None or np.linalg.norm(measured[:2]-last_depth_pose[:2])>.025 or abs(angle)>.02:
                from eval.depth_sensor import observed_map
                observation_map, _ = observed_map(env, config['scene_assets'], args.scene_name,
                    args.floor_level, guidance_config, device)
                memory.add(observation_map, measured)
                last_depth_pose = measured
        return rgb
    rgbs = []
    for index in range(start - history_length, start + 1):
        pose = poses[index]
        yaw = float(np.arctan2(pose[3]-pose[1], pose[2]-pose[0]))
        robot.set_position([*pose[:2], args.robot_z])
        robot.set_orientation(p.getQuaternionFromEuler([0, 0, yaw]))
        env.simulator_step()
        observation = capture_observation()
        if index < start:
            rgbs.append(observation)
        else:
            current_rgb = observation
    position = poses[start, :2].copy()
    guidance_records = []
    guidance_config = getattr(args, 'guidance_config', None)
    rows, plans = [], []  # Original output does not prepend the initial pose.
    controller = None
    if navigation_goals is not None:
        from eval.candidate_navigation import CandidateNavigator
        controller = CandidateNavigator(
            navigation_goals, max_steps=args.max_steps, collision_limit=args.collision_limit,
            reach_radius=getattr(args, 'candidate_reach_radius', .5),
            stall_steps=getattr(args, 'candidate_stall_steps', 20),
            min_progress=getattr(args, 'candidate_min_progress', .2),
            unreachable_steps=getattr(args, 'candidate_unreachable_steps', 3),
            dedup_radius=getattr(args, 'candidate_dedup_radius', .5))
        goal = controller.goal
    goal_tree = map_planner.goal_tree(goal)
    collisions, moves, replans, failed_priors, fallback_failures = 0, 0, 0, 0, 0
    while (np.linalg.norm(position-gt_goal) > args.success_distance
           and replans < args.max_steps and collisions < args.collision_limit):
        # Use GT pose at the planning instant for all ego/world transforms.
        position = np.asarray(robot.get_position()[:2]).copy()
        yaw = float(robot.get_rpy()[2])
        if controller is not None:
            while not controller.exhausted:
                map_path = map_planner.path_from_tree(position, goal_tree)
                remaining = (float(np.linalg.norm(np.diff(map_path, axis=0), axis=1).sum())
                             + float(np.linalg.norm(position-map_path[0]))
                             + float(np.linalg.norm(map_path[-1]-goal))) if len(map_path) else float('inf')
                reason = controller.observe(position, remaining, replans, collisions)
                if reason is None:
                    break
                if not controller.exhausted:
                    goal = controller.goal
                    goal_tree = map_planner.goal_tree(goal)
            if controller.exhausted:
                break
        prior, valid = map_planner.prior(position, yaw, goal, config['len_traj_pred'], config['astar_step_m'], tree=goal_tree)
        failed_priors += int(valid.sum() == 0)
        batch = dict(agent_pose=torch.tensor([[position[0], position[1], yaw]], dtype=torch.float32, device=device),
                     goal_world=torch.tensor(goal, dtype=torch.float32, device=device)[None],
                     rgb=torch.stack(rgbs)[None], current_rgb=current_rgb[None],
                     floorplan=mark_floorplan(floor, map_planner, position, yaw, goal).to(device)[None],
                     point_goal=torch.tensor(world_to_local(goal, position, yaw), dtype=torch.float32, device=device)[None, None],
                     prior=torch.from_numpy(prior).to(device)[None], prior_valid=torch.from_numpy(valid).to(device)[None])
        energy_guidance = None
        if guidance_config is not None:
            from eval.depth_sensor import observed_map
            from model.bridge_guidance.bridge_energy import BridgeEnergyGuidance
            local_map, sensing = observed_map(env, config['scene_assets'],
                args.scene_name, args.floor_level, guidance_config, device)
            if memory is not None:
                measured = np.array([*robot.get_position()[:2], float(robot.get_rpy()[2])])
                # Add the actual planning view as the exact target frame of fusion.
                if memory.frames and torch.allclose(memory.frames[-1][1], torch.as_tensor(measured, device=device, dtype=torch.float32)[None], atol=1e-5):
                    memory.frames.pop()
                memory.add(local_map, measured)
                local_map = memory.current()
                sensing.update(memory_frames=len(memory.frames), fused_known_cells=int(local_map.known.sum()),
                    fused_occupied_cells=int((local_map.sdf<0).sum()))
            if guidance_config.get('version') == 'v3':
                from model.bridge_guidance.bridge_energy_v3 import BridgeEnergyGuidanceV3
                energy_guidance = BridgeEnergyGuidanceV3(local_map, batch['point_goal'], guidance_config,
                    float(config['bridge']['action_step_m']), batch['prior'], batch['prior_valid'], args.execute_steps)
            else:
                energy_guidance = BridgeEnergyGuidance(local_map, batch['point_goal'],
                    guidance_config, float(config['bridge']['action_step_m']))
        path = planner(batch, mode='sample', energy_guidance=energy_guidance)
        if energy_guidance is not None:
            guidance_records.append(dict(replan=replans, sensing=sensing,
                visible_goal=energy_guidance.objective.visible_goal.cpu().tolist(),
                updates=energy_guidance.records))
        if args.save_plans:
            plans.append(dict(astar=prior.tolist(), planner=path[0].cpu().tolist(), position=position.tolist(), yaw=yaw))
        # Original execute_steps=9 includes origin. get_pose drops it => 8 moves.
        predicted = local_to_world(path[0, :args.execute_steps].cpu().numpy(), position, yaw)
        predicted_pose, yaws = get_pose_from_position(predicted)
        for waypoint_index, waypoint_pose in enumerate(predicted_pose):
            next_yaw = float(yaws[waypoint_index])
            valid_motion = env.test_valid_position(robot, np.array([*waypoint_pose[:2], args.robot_z]),
                np.array([0., 0., next_yaw]), ignore_self_collision=True)
            if not valid_motion:
                next_position, next_yaw = move_to_fallback(env, args.floor_level, goal, args.robot_z, yaw)
                if next_position is None:
                    fallback_failures += 1
                    break  # Match original: no counted collision without fallback.
                # Original refreshes all history slots after fallback; adapt to configured history.
                for _ in range(history_length):
                    rgbs = rgbs[1:] + [current_rgb]
                    current_rgb = capture_observation()
                    position = np.asarray(robot.get_position()[:2]).copy()
                collisions += 1
                rows.append([*next_position, next_yaw, 1])
                break
            robot.set_position([*waypoint_pose[:2], args.robot_z])
            robot.set_orientation(waypoint_pose[2:])
            env.simulator.step()
            rgbs = rgbs[1:] + [current_rgb]
            current_rgb = capture_observation()
            position = np.asarray(robot.get_position()[:2]).copy()
            rows.append([*waypoint_pose[:2], next_yaw, 0])
            moves += 1
            # Like original, finish this chunk before the outer success check.
        replans += 1
    array = np.asarray(rows).reshape(-1, 4)
    scored = score_trajectory(array, poses, args.collision_limit, args.success_distance)
    result = dict(scored, moves=moves, replans=replans, failed_astar=failed_priors,
                  runtime_collisions=collisions, fallback_failures=fallback_failures,
                  terminal_distance=float(np.linalg.norm(position-gt_goal)),
                  predicted_goal_distance=float(np.linalg.norm(position-goal)),
                  success_distance=args.success_distance, collision_limit=args.collision_limit,
                  start_index=start, history_indices=list(range(start-history_length,start)),
                  oracle_recovery=True, protocol=PROTOCOL, max_planning_cycles=args.max_steps,
                  execute_steps_including_origin=args.execute_steps)
    if controller is not None:
        result['candidate_navigation'] = controller.summary()
    directory.mkdir(parents=True, exist_ok=True)
    np.savetxt(directory / trajectory.name, array)
    (directory / 'result.json').write_text(json.dumps(result, indent=2))
    if guidance_config is not None:
        (directory / 'guidance.json').write_text(json.dumps(guidance_records))
    if args.save_plans:
        (directory / 'plans.json').write_text(json.dumps(plans))
    return result


def main():
    from eval.common import build_env, set_seed, stable_string_seed, list_traj_ids, list_scene_ids
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--paths-config', help='Override only scene_assets, map_cache and prior_cache in a saved checkpoint')
    parser.add_argument('--scenes-dir', default='data/eval_without_obj/HM3D')
    parser.add_argument('--output', required=True)
    parser.add_argument('--scene-ids', nargs='+')
    parser.add_argument('--max-trajs', type=int, default=30)
    parser.add_argument('--execute-steps', type=int, default=9, help='Includes origin; 9 gives eight executed future points')
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--collision-limit', type=int, default=15)
    parser.add_argument('--success-distance', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-plans', action='store_true')
    parser.add_argument('--guidance-config', help='Optional VE bridge depth guidance YAML')
    parser.add_argument('--episodes-file', help='Explicit disjoint validation episode list')
    parser.add_argument('--paired-episode-seed', action='store_true')
    args = parser.parse_args()
    if args.guidance_config:
        args.guidance_config = OmegaConf.to_container(OmegaConf.load(args.guidance_config), resolve=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config, planner = load_models(args.checkpoint, device)
    if args.paths_config:
        paths = OmegaConf.to_container(OmegaConf.load(args.paths_config), resolve=True)
        for key in ('scene_assets', 'map_cache', 'prior_cache'):
            config[key] = paths[key]
    if not 2 <= args.execute_steps <= config['len_traj_pred']:
        raise ValueError('execute-steps must be in [2, horizon], including origin')
    for name in ['max_steps', 'collision_limit', 'success_distance']:
        if getattr(args, name) <= 0:
            raise ValueError(name + ' must be positive')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Use a fresh output directory: ' + str(output))
    output.mkdir(parents=True)
    (output / 'run.json').write_text(json.dumps(vars(args), indent=2))
    selected = None
    if args.episodes_file:
        selected = json.loads(Path(args.episodes_file).read_text())
        if len({tuple(x) for x in selected}) != len(selected):
            raise ValueError('Duplicate selected episodes')
        args.scene_ids = sorted({x[0] for x in selected})
    scenes = list_scene_ids(args.scenes_dir, args.scene_ids)
    if not scenes:
        raise ValueError('No HM3D evaluation scenes selected')
    for scene in scenes:
        set_seed(args.seed + stable_string_seed(scene))
        directory = prepare_map(args.scenes_dir, scene, config['scene_assets'], config['map_cache'],
                                256, config['astar_free_threshold'], config['astar_clearance_m'], config.get('astar_exclude_exterior', False))
        map_planner = FloorplanPlanner(directory, config['astar_max_snap_m'])
        floor = load_floorplan_image(str(Path(args.scenes_dir) / scene / 'floorplan.png'))
        name, level = scene.rsplit('_', 1)
        args.floor_level = int(level)
        args.scene_name = name
        env, args.robot_z = build_env(name, args.floor_level, 0, config['scene_assets'])
        try:
            trajectories = [Path(args.scenes_dir) / scene / name for name in list_traj_ids(args.scenes_dir, scene)[:args.max_trajs]]
            if selected is not None:
                trajectories = [Path(args.scenes_dir)/scene/name for s,name in selected if s==scene]
            for trajectory in trajectories:
                if args.paired_episode_seed:
                    import hashlib
                    seed = int.from_bytes(hashlib.sha256((str(args.seed)+'/'+scene+'/'+trajectory.name).encode()).digest()[:4], 'little')
                    set_seed(seed)
                result = rollout(env, trajectory / (trajectory.name + '.txt'), floor, map_planner,
                                 planner, config, args, output / scene / trajectory.name, device)
                print(scene, trajectory.name, json.dumps(result), flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
