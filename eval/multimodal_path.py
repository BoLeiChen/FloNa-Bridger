"""Shared PointGoal/ImageGoal path policy under the FloVerse scoring protocol."""
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
from model.multimodal_path_bridge import MultimodalPathBridge as PointBridgePlanner, CHECKPOINT_FORMAT
from datasets.path_distribution import PathProposer
from eval.path_policy_state import PathPolicyState
from eval.object_category_protocol import (load_category_targets, distance_to_category,
    category_reference_length, score_category_trajectory)

V2_POLICY_OVERRIDE_TYPES = {
    'goal_reject_enter': float, 'goal_reject_leave': float,
    'goal_reject_ema_alpha': float, 'goal_reject_confirm_cycles': int,
    'route_switch_log_ratio': float, 'route_switch_confirm_cycles': int,
    'route_min_hold_cycles': int, 'route_progress_window': int,
    'route_cooldown_cycles': int, 'route_cooldown_penalty': float,
    'hypothesis_refresh_cycles': int, 'route_arrival_m': float,
}


def apply_v2_policy_overrides(config, values):
    """Apply an explicit inference-only allowlist and validate it as one policy."""
    if values and config.get('route_policy_version') != 'category-v2':
        raise ValueError('V2 policy overrides require a category-v2 checkpoint')
    applied = {}
    for item in values:
        if '=' not in item:
            raise ValueError('Expected --v2-policy-override NAME=VALUE')
        name, raw = item.split('=', 1)
        if name not in V2_POLICY_OVERRIDE_TYPES:
            raise ValueError('Unsupported V2 policy override: ' + name)
        if name in applied:
            raise ValueError('Duplicate V2 policy override: ' + name)
        cast = V2_POLICY_OVERRIDE_TYPES[name]
        try:
            value = cast(raw)
        except ValueError as error:
            raise ValueError(f'Invalid value for {name}: {raw}') from error
        applied[name] = value
        config[name] = value
    if values:
        leave=float(config.get('goal_reject_leave',.35))
        enter=float(config.get('goal_reject_enter',.65))
        alpha=float(config.get('goal_reject_ema_alpha',.5))
        if not 0 <= leave < enter <= 1 or not 0 < alpha <= 1:
            raise ValueError('Require 0 <= reject_leave < reject_enter <= 1 and EMA alpha in (0,1]')
        for name in ('goal_reject_confirm_cycles','route_switch_confirm_cycles',
                     'route_min_hold_cycles','route_progress_window',
                     'route_cooldown_cycles','hypothesis_refresh_cycles'):
            if int(config.get(name, 1)) < 1:
                raise ValueError(name + ' must be at least one')
        for name in ('route_switch_log_ratio','route_cooldown_penalty'):
            if float(config.get(name, 0)) < 0:
                raise ValueError(name + ' must be nonnegative')
        if float(config.get('route_arrival_m',.5)) <= 0:
            raise ValueError('route_arrival_m must be positive')
    return applied


def capture(env, device):
    rgb, = env.simulator.renderer.render_robot_cameras(modes=('rgb',), cache=False)
    rgb = Image.fromarray(np.clip(rgb[:, :, :3] * 255, 0, 255).astype(np.uint8))
    def small(image):
        array = np.asarray(image.resize((96, 96), Image.Resampling.BILINEAR)).copy()
        if array.ndim == 2:
            array = array[:, :, None]
        return torch.from_numpy(array).permute(2, 0, 1).float().to(device) / 255
    return small(rgb)


def load_models(checkpoint, device, paths_config=None):
    saved = torch.load(checkpoint, map_location='cpu')
    from model.path_distribution_v2 import PathDistributionV2, CHECKPOINT_FORMAT as V2_FORMAT
    from model.topk_route_bridge import TopKRouteBridge, CHECKPOINT_FORMAT as TOPK_FORMAT
    from model.object_topk_bridge import ObjectTopKRouteBridge, CHECKPOINT_FORMAT as OBJECT_FORMAT
    if saved.get('format') not in (CHECKPOINT_FORMAT,V2_FORMAT,TOPK_FORMAT,OBJECT_FORMAT):
        raise ValueError('Expected a supported path-distribution checkpoint')
    config = dict(saved['config'])
    if paths_config is not None:
        for key in ('scene_assets','map_cache','prior_cache','goal_hypothesis_cache'):
            config[key] = paths_config[key]
        config['goal_localization'] = dict(config['goal_localization'])
        for key in ('diffusion_checkpoint','cf2loc_disco_checkpoint','gibson_dataset_root','dinov2_checkpoint'):
            config['goal_localization'][key] = paths_config['goal_localization'][key]
    model_class=ObjectTopKRouteBridge if saved['format']==OBJECT_FORMAT else TopKRouteBridge if saved['format']==TOPK_FORMAT else PathDistributionV2 if saved['format']==V2_FORMAT else PointBridgePlanner
    planner = model_class(config, load_pretrained=False).to(device).eval()
    planner.load_state_dict(saved['planner'], strict=True)
    return config, planner


@torch.no_grad()
def rollout(env, trajectory, floor, map_planner, planner, config, args, directory, device):
    poses = np.atleast_2d(np.loadtxt(trajectory))
    start = 8  # Same navigation start as original; frame count is model-specific.
    if len(poses) <= start:
        raise ValueError('Original FloVerse protocol requires trajectory index 8')
    gt_goal = poses[-1, :2].copy()
    task = {'point':0,'image':1,'object':2}[args.goal_type]
    category_targets = None
    category_length = None
    if task == 2 and getattr(args, 'object_success_mode', 'category') == 'category':
        category_targets = load_category_targets(config['scene_assets'], trajectory.parents[1].name, trajectory.parent)
    region_image = task==1 and config.get('route_supervision')=='image-goal-region-v2'
    if region_image:
        category_targets=dict(world_xy=[gt_goal.tolist()],object_ids=['goal_image'])
    def success_distance(position):
        return (distance_to_category(position, category_targets) if category_targets is not None
                else float(np.linalg.norm(position-gt_goal)))
    goal_image_path = trajectory.parent / 'rgb' / ('rgb_%d.png' % (len(poses)-1))
    if task==1 and getattr(args,'diagnostic_goal_mode','correct')=='shuffled':
        alternatives=sorted(trajectory.parent.parent.glob('traj_*/traj_*.txt'))
        alternatives=[p for p in alternatives if p!=trajectory]
        if not alternatives:raise ValueError('No alternative image goal for diagnostic')
        # Deterministic different trajectory; never changes the scoring GT.
        index=sum(trajectory.parent.name.encode())%len(alternatives)
        source=alternatives[index];count=len(np.atleast_2d(np.loadtxt(source)))
        goal_image_path=source.parent/'rgb'/('rgb_%d.png'%(count-1))
    if task==1:
        with Image.open(goal_image_path) as image:
            goal_rgb = torch.from_numpy(np.asarray(image.convert('RGB').resize((96,96), Image.Resampling.BILINEAR)).copy()).permute(2,0,1).float()/255
    else:
        goal_rgb = torch.zeros(3,96,96)
    # GT goal is scoring-only for ImageGoal. No target coordinate in its inputs.
    state = PathPolicyState(float(config.get('route_switch_margin',.15)),
                            int(config.get('route_min_hold_cycles',2)),int(config.get('route_stall_cycles',5)))
    proposer = PathProposer(dict(config, path_candidate_cache=None))
    if 'goal_top_k' in config:
        if task not in (1,2) or args.diagnostic_route_mode!='learned' or args.controller_checkpoint:
            raise ValueError('TopK evaluation requires ordinary ImageGoal mode')
        from localization.goal_hypotheses import GoalHypothesisProvider
        from datasets.hypothesis_routes import HypothesisRoutes
        from eval.hypothesis_policy_state import HypothesisPolicyState
        # One frozen provider per evaluator, shared across episodes; no GT argument.
        if task==2:
            from localization.object_hypotheses import ObjectHypothesisProvider
            from eval.common import read_object_goal
            category=read_object_goal(str(trajectory.parents[2]),trajectory.parents[1].name,trajectory.parent.name)[0]
            if not hasattr(planner,'goal_hypothesis_provider'):
                planner.goal_hypothesis_provider=ObjectHypothesisProvider(config,device,allow_inference=True)
            calibration=map_planner.metadata
            origin=-np.asarray(calibration['offset'])/calibration['scale']
            extent=np.asarray(calibration['source_size'])/calibration['scale']
            hypotheses=planner.goal_hypothesis_provider.get(trajectory.parents[1]/'floorplan.png',category,origin_xy=origin,extent_xy=extent)
        else:
            if not hasattr(planner,'goal_hypothesis_provider'):
                planner.goal_hypothesis_provider=GoalHypothesisProvider(config,device,allow_inference=True)
            hypotheses=planner.goal_hypothesis_provider.get(goal_image_path)
        if task==2:goal_text=torch.as_tensor(hypotheses.pop('goal_text'),dtype=torch.float32,device=device)
        proposer=HypothesisRoutes(config,hypotheses);state=HypothesisPolicyState(config)
    point_tree = map_planner.goal_tree(gt_goal) if task == 0 else None
    robot = env.robots[0]
    history_length = int(config['history_length'])
    if history_length != 3:
        raise ValueError('This Planner uses three history frames plus current RGB')
    rgbs = []
    for index in range(start - history_length, start + 1):
        pose = poses[index]
        yaw = float(np.arctan2(pose[3]-pose[1], pose[2]-pose[0]))
        robot.set_position([*pose[:2], args.robot_z])
        robot.set_orientation(p.getQuaternionFromEuler([0, 0, yaw]))
        env.simulator_step()
        state.memory.update(pose[:2])
        observation = capture(env, device)
        if index < start:
            rgbs.append(observation)
        else:
            current_rgb = observation
    position = poses[start, :2].copy()
    category_start = None
    if category_targets is not None:
        # Physics can slightly displace the commanded pose. Score the same actual
        # robot positions used by online termination, not commanded waypoints.
        position = np.asarray(robot.get_position()[:2]).copy()
        category_start = position.copy()
        category_length = category_reference_length(map_planner, category_start, category_targets, args.success_distance)
    rows, plans = [], []  # Original output does not prepend the initial pose.
    route_records = []
    guidance_records = []
    collisions, moves, replans, failed_priors, fallback_failures = 0, 0, 0, 0, 0
    while (success_distance(position) > args.success_distance
           and replans < args.max_steps and collisions < args.collision_limit):
        # Use GT pose at the planning instant for all ego/world transforms.
        position = np.asarray(robot.get_position()[:2]).copy()
        yaw = float(robot.get_rpy()[2])
        state.observe(position)
        candidates = proposer.propose(map_planner, position, yaw, state.memory,
            point_goal=gt_goal if task == 0 else None, previous_goal=state.goal if task else None, point_tree=point_tree)
        if not candidates['candidate_mask'].any():
            failed_priors += 1
            break
        batch = dict(rgb=torch.stack(rgbs)[None], current_rgb=current_rgb[None],
            floorplan=mark_floorplan(floor,map_planner,None,None,None).to(device)[None],
            agent_pose=torch.tensor([[*position,yaw]],dtype=torch.float32,device=device),
            task_type=torch.tensor([task],dtype=torch.long,device=device),
            task_point_world=torch.tensor(gt_goal if task==0 else np.zeros(2),dtype=torch.float32,device=device)[None],
            goal_rgb=goal_rgb.to(device)[None])
        if task==2:batch['goal_text']=goal_text[None]
        batch.update({k:torch.as_tensor(v,device=device)[None] for k,v in candidates.items()})
        if getattr(args,'diagnostic_route_mode','learned')=='oracle':
            batch['diagnostic_gt_goal']=torch.as_tensor(gt_goal,dtype=torch.float32,device=device)[None]
        prepared=planner.score_paths(batch)
        probabilities=prepared[0][0].cpu().numpy()
        if 'goal_top_k' in config:
            index=state.select(candidates,probabilities,position,reject_probability=float(prepared[2][0]))
            proposer.commit(index,candidates)
        else:
            index=state.select(candidates,probabilities,position,point_goal=task==0)
        goal=candidates['candidate_goals'][index].astype(np.float64)
        prior=candidates['candidate_priors'][index]
        route_records.append(dict(probabilities=probabilities.tolist(),selected=index,subgoal=goal.tolist(),
            entropy=float(-(probabilities*np.log(np.maximum(probabilities,1e-8))).sum()),
            valid_candidates=int(candidates['candidate_mask'].sum())))
        if 'goal_top_k' in config:
            route_records[-1].update(candidate_ids=candidates['candidate_ids'].tolist(),candidate_is_goal=candidates['candidate_is_goal'].tolist(),
                selected_id=int(candidates['candidate_ids'][index]),reject_probability=float(prepared[2][0]),
                switch_reason=state.last_reason,reservoir_offset=proposer.offset)
            route_records[-1].update(reject_probability_smoothed=state.reject_ema,
                                     rejected_goal_set=state.last_rejected)
        if getattr(args,'navigation_controller_checkpoint',None):
            from eval.frozen_depth_controller import sample_selected
            path,diagnostics=sample_selected(args.navigation_controller,args.navigation_controller_config,
                batch,index,args.depth_guidance,env,config['scene_assets'],
                trajectory.parents[1].name.rsplit('_',1)[0],args.floor_level,args.execute_steps)
            if diagnostics is not None:
                diagnostics['replan']=replans
                guidance_records.append(diagnostics)
        else:
            path=planner(batch,mode='sample',selected_index=torch.tensor([index],device=device),prepared=prepared)
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
                state.memory.update(waypoint_pose[:2], failed=True)
                next_position, next_yaw = move_to_fallback(env, args.floor_level, goal, args.robot_z, yaw)
                if next_position is None:
                    fallback_failures += 1
                    break  # Match original: no counted collision without fallback.
                # Original refreshes all history slots after fallback; adapt to configured history.
                for _ in range(history_length):
                    rgbs = rgbs[1:] + [current_rgb]
                    current_rgb = capture(env, device)
                    position = np.asarray(robot.get_position()[:2]).copy()
                state.memory.update(position)
                collisions += 1
                rows.append([*(position if category_targets is not None else next_position), next_yaw, 1])
                break
            robot.set_position([*waypoint_pose[:2], args.robot_z])
            robot.set_orientation(waypoint_pose[2:])
            env.simulator.step()
            rgbs = rgbs[1:] + [current_rgb]
            current_rgb = capture(env, device)
            position = np.asarray(robot.get_position()[:2]).copy()
            state.memory.update(position)
            rows.append([*(position if category_targets is not None else waypoint_pose[:2]), next_yaw, 0])
            moves += 1
            if category_targets is not None and success_distance(position) <= args.success_distance:
                break
            # Like original, finish this chunk before the outer success check.
        if 'goal_top_k' in config and state.refresh_due():
            route_records[-1]['reservoir_refreshed']=proposer.refresh()
        replans += 1
    array = np.asarray(rows).reshape(-1, 4)
    scored = (score_category_trajectory(array, category_start, category_targets, category_length,
                                      args.collision_limit, args.success_distance)
              if category_targets is not None else score_trajectory(array, poses, args.collision_limit, args.success_distance))
    if region_image:
        from eval.image_region_protocol import image_region_result
        scored=image_region_result(scored)
    result = dict(scored, moves=moves, replans=replans, failed_astar=failed_priors,
                  runtime_collisions=collisions, fallback_failures=fallback_failures,
                  terminal_distance=success_distance(position),
                  predicted_goal_distance=float(np.linalg.norm(position-state.goal)) if state.goal is not None else None,
                  success_distance=args.success_distance, collision_limit=args.collision_limit,
                  start_index=start, history_indices=list(range(start-history_length,start)),
                  oracle_recovery=True, protocol=scored.get('protocol', PROTOCOL), max_planning_cycles=args.max_steps,
                  execute_steps_including_origin=args.execute_steps)
    if 'goal_top_k' in config:result.update(goal_top_k=config['goal_top_k'],target_source='object_diffusion_xy_nms_disco' if task==2 else 'cf2loc_nms64_disco_hypotheses',exploration_candidates=config['exploration_candidates'])
    result.update(goal_type=args.goal_type, route_switches=state.switches,
                  goal_image_source=str(goal_image_path) if task==1 else None,
                  policy_uses_gt_target=(task==0 or getattr(args,'diagnostic_route_mode','learned')=='oracle'), benchmark_gt_success_termination=True,
                  diagnostic_route_mode=getattr(args,'diagnostic_route_mode','learned'),
                  diagnostic_goal_mode=getattr(args,'diagnostic_goal_mode','correct'),
                  controller_checkpoint=getattr(args,'controller_checkpoint',None))
    if config.get('route_policy_version')=='category-v2':
        result.update(route_policy='shared_topk_v2_persistent_exploration_hysteresis',
                      route_supervision=config.get('route_supervision'),route_objective=config.get('route_objective'))
    if region_image:result['evaluation_start_xy']=category_start.tolist()
    if task==2:
        result.update(object_category=category,object_success_protocol=scored.get('object_success_protocol','original_reference_trajectory_endpoint'),route_policy='same_learned_topk_route_and_rejection_as_imagegoal')
        if config.get('route_policy_version')=='category-v2':
            result.update(route_policy='shared_topk_v2_persistent_exploration_hysteresis',
                          route_supervision=config.get('route_supervision'),route_objective=config.get('route_objective'))
        if category_targets is not None:
            result.update(category_target_provenance=category_targets,
                          evaluation_start_xy=category_start.tolist(),
                          reference_endpoint_distance=float(np.linalg.norm(position-gt_goal)))
    directory.mkdir(parents=True, exist_ok=True)
    np.savetxt(directory / trajectory.name, array)
    result.update(navigation_controller_checkpoint=getattr(args,'navigation_controller_checkpoint',None),
                  depth_guidance_config=getattr(args,'guidance_config',None))
    (directory / 'result.json').write_text(json.dumps(result, indent=2))
    if getattr(args,'guidance_config',None):
        (directory/'guidance.json').write_text(json.dumps(guidance_records))
    (directory / 'route_decisions.json').write_text(json.dumps(route_records))
    if args.save_plans:
        (directory / 'plans.json').write_text(json.dumps(plans))
    return result


def main():
    from eval.common import build_env, set_seed, stable_string_seed, list_traj_ids, list_scene_ids
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--paths-config', default='config/imagegoal_v3.yaml')
    parser.add_argument('--goal-top-k',type=int,help='Inference K ablation for TopK models; record distribution change')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--scenes-dir')
    parser.add_argument('--goal-type', choices=['point','image','object'], default='image')
    parser.add_argument('--output', required=True)
    parser.add_argument('--scene-ids', nargs='+')
    parser.add_argument('--episode-manifest', help='Override fixed ImageGoal evaluation manifest')
    parser.add_argument('--max-trajs', type=int, default=30)
    parser.add_argument('--execute-steps', type=int, default=9, help='Includes origin; 9 gives eight executed future points')
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--collision-limit', type=int, default=15)
    parser.add_argument('--success-distance', type=float, default=1.0)
    parser.add_argument('--object-success-mode', choices=['category','reference'], default='category',
                        help='ObjectGoal: any same-category real object within 1m; reference reproduces old scoring')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-plans', action='store_true')
    parser.add_argument('--diagnostic-route-mode',choices=['learned','oracle'],default='learned')
    parser.add_argument('--diagnostic-goal-mode',choices=['correct','shuffled'],default='correct')
    parser.add_argument('--controller-checkpoint')
    parser.add_argument('--navigation-controller-checkpoint')
    parser.add_argument('--guidance-config')
    parser.add_argument('--v2-policy-override',action='append',default=[],metavar='NAME=VALUE',
                        help='Inference-only category-v2 override from a strict allowlist')
    parser.add_argument('--diagnostic-seed-per-episode',action='store_true')
    args = parser.parse_args()
    if args.guidance_config and not args.navigation_controller_checkpoint:
        raise ValueError('Depth guidance requires explicit navigation controller')
    if args.navigation_controller_checkpoint and (args.controller_checkpoint or args.diagnostic_route_mode!='learned'):
        raise ValueError('Controller substitution requires ordinary learned goal policy')
    if args.scenes_dir is None:
        args.scenes_dir = 'data/eval_with_obj' if args.goal_type in ('image','object') else 'data/eval_without_obj/HM3D'
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    paths_config = OmegaConf.to_container(OmegaConf.load(args.paths_config), resolve=True)
    config, planner = load_models(args.checkpoint, device, paths_config)
    args.effective_v2_policy_overrides=apply_v2_policy_overrides(config,args.v2_policy_override)
    trained_k=config.get('goal_top_k')
    if args.goal_top_k is not None:
        if trained_k is None or not 1<=args.goal_top_k<=64:raise ValueError('Top-K override requires TopK checkpoint and K in [1,64]')
        config['goal_top_k']=args.goal_top_k
    args.trained_goal_top_k=trained_k
    args.evaluation_goal_top_k=config.get('goal_top_k')
    if args.diagnostic_route_mode=='oracle' or args.controller_checkpoint:
        from eval.path_diagnostics import DiagnosticPolicy
        controller=None
        if args.controller_checkpoint:
            from eval.pointgoal_bridge import load_models as load_controller
            controller_config,controller=load_controller(args.controller_checkpoint,device)
            for key in ('history_length','len_traj_pred','global_coordinate_mode','global_coordinate_step_m'):
                if config[key]!=controller_config[key]:raise ValueError('Controller mismatch: '+key)
            if config['bridge']['action_step_m']!=controller_config['bridge']['action_step_m']:
                raise ValueError('Controller action scale mismatch')
        planner=DiagnosticPolicy(planner,controller,args.diagnostic_route_mode=='oracle')
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
    if args.navigation_controller_checkpoint:
        from eval.frozen_depth_controller import load_controller
        args.navigation_controller_config,args.navigation_controller=load_controller(args.navigation_controller_checkpoint,config,device)
        args.depth_guidance=OmegaConf.to_container(OmegaConf.load(args.guidance_config),resolve=True) if args.guidance_config else None
    from eval.path_eval_manifest import select_episodes
    episodes = select_episodes(args.scenes_dir, args.goal_type, args.max_trajs,
                               args.episode_manifest, args.scene_ids)
    (output / 'episodes.json').write_text(json.dumps(episodes))
    scenes = sorted({scene for scene, traj in episodes})
    if not scenes:
        raise ValueError('No HM3D evaluation scenes selected')
    for scene in scenes:
        set_seed(args.seed + stable_string_seed(scene))
        directory = prepare_map(args.scenes_dir, scene, config['scene_assets'], config['map_cache'],
                                256, config['astar_free_threshold'], config['astar_clearance_m'], config.get('astar_exclude_exterior', False))
        map_planner = FloorplanPlanner(directory, config['astar_max_snap_m'])
        if hasattr(planner,'map_planner'):planner.map_planner=map_planner
        floor = load_floorplan_image(str(Path(args.scenes_dir) / scene / 'floorplan.png'))
        name, level = scene.rsplit('_', 1)
        args.floor_level = int(level)
        env, args.robot_z = build_env(name, args.floor_level, 0, scene_assets=config['scene_assets'])
        try:
            trajectories = [Path(args.scenes_dir) / scene / traj for episode_scene, traj in episodes if episode_scene == scene]
            for trajectory in trajectories:
                if args.diagnostic_seed_per_episode:set_seed(args.seed+stable_string_seed(scene+'/'+trajectory.name))
                result = rollout(env, trajectory / (trajectory.name + '.txt'), floor, map_planner,
                                 planner, config, args, output / scene / trajectory.name, device)
                print(scene, trajectory.name, json.dumps(result), flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
