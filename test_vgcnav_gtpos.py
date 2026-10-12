import os
import json
import hashlib
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_script_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

diffusion_policy_path = os.path.join(project_root, 'diffusion_policy')
if diffusion_policy_path not in sys.path:
    sys.path.insert(0, diffusion_policy_path)

import PIL.Image as Image
import igibson
import numpy as np
import yaml
import torch
import torch.nn as nn
from torchvision import transforms
import torch.backends.cudnn as cudnn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

from igibson.simulator import Simulator
from igibson.render.profiler import Profiler
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from model.data_utils import (
    astar_path_on_walkable_map,
    floorplan_to_walkable_map,
    img_path_to_data,
    metric_positions_to_resized_pixels,
    resized_pixels_to_metric_positions,
    resize_and_aspect_crop,
    sample_path_prefix_points,
    path_xy_to_local_condition,
    to_local_coords,
)
from model.flona import flona, DenseNetwork
from model.flona_vint import flona_ViNT, replace_bn_with_gn
from model.navibridger import NaviBridgeConfig, NaviBridgeDenoiser
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from training.train_eval_loop import load_model
from training.train_utils import execute_model

DEFAULT_OBS_IMG_SIZE = (224, 224)       # 观测图像 for DINOv2
FLOORPLAN_IMG_SIZE = (96, 96)   # 平面图 for EfficientNet
PLANNER_FLOORPLAN_SIZE = (256, 256)
IMAGE_ASPECT_RATIO = 1 / 1 

def check_collision(pos, travers_map):
    pos_in_map = (pos * 100 + np.array([travers_map.shape[0], travers_map.shape[1]]) // 2).astype(np.int16)
    return travers_map[pos_in_map[1], pos_in_map[0]] == 0

def l2_distance(v1, v2):
    return np.linalg.norm(np.array(v1) - np.array(v2))

def camera_set_and_record(env, current_position, current_heading, obs_img_size):
    if env.viewer is not None:
        env.viewer.initial_pos = current_position
        env.viewer.initial_view_direction = current_heading - current_position
        env.viewer.reset_viewer()
    env.renderer.set_camera(current_position, current_heading, [0, 0, 1])
    
    frame = env.renderer.render(modes=("rgb"))
    img = Image.fromarray((255 * np.concatenate(frame, axis=1)[:, :, :3]).astype(np.uint8))
    resized_img = resize_and_aspect_crop(img, obs_img_size, IMAGE_ASPECT_RATIO)
    env.step()
    return resized_img
    
def camera_execute_actions(
    env,
    current_position,
    trajectory,
    cur_obs_list,
    context_size,
    travers_map,
    save_state,
    obs_img_size,
    collision_accounting_mode="legacy",
    recovery_turn_mode="continuous_right",
):
    if collision_accounting_mode not in {"legacy", "movement_only"}:
        raise ValueError(
            "collision_accounting_mode must be 'legacy' or 'movement_only'"
        )
    if recovery_turn_mode not in {"random", "continuous_right"}:
        raise ValueError(
            "recovery_turn_mode must be 'random' or 'continuous_right'"
        )
    turn = False
    for subgoal in trajectory:
        sub_waypoints = sub_waypoints_generator(subgoal, current_position)
        last_sub_waypoint = current_position
        for sub_waypoint in sub_waypoints:
            collision = check_collision(sub_waypoint[:2], travers_map)          
            current_position = last_sub_waypoint
            current_heading_point = sub_waypoint
            resized_img = camera_set_and_record(env, current_position, current_heading_point, obs_img_size)
            save_state.append(np.array([current_position[0], current_position[1], current_heading_point[0], current_heading_point[1], collision]))
          
            last_sub_waypoint = sub_waypoint
            if collision:
                print("collide")               
                break
        
        if len(cur_obs_list) < context_size + 1:
            cur_obs_list.append(resized_img.unsqueeze(0))
        else:
            cur_obs_list.pop(0)
            cur_obs_list.append(resized_img.unsqueeze(0))
            
        if collision:
            current_yaw = np.arctan2(current_heading_point[1] - current_position[1], current_heading_point[0] - current_position[0])
            collision_num = 0
            coin = 1.0 if recovery_turn_mode == "continuous_right" else np.random.rand()
            if coin >= 0.5:
                while collision and collision_num < 8:
                    turn = True
                    collision_num += 1
                    current_yaw -= 45 / 180 * np.pi
                    current_heading_point[:2] = current_position[:2] + np.array([np.cos(current_yaw), np.sin(current_yaw)]) * 0.02
                    resized_img = camera_set_and_record(env, current_position, current_heading_point, obs_img_size)
                    recovery_collision = collision if collision_accounting_mode == "legacy" else False
                    collision = check_collision(current_heading_point[:2], travers_map)
                    save_state.append(np.array([
                        current_position[0], current_position[1],
                        current_heading_point[0], current_heading_point[1],
                        recovery_collision,
                    ]))
                    if len(cur_obs_list) < context_size + 1:
                         cur_obs_list.append(resized_img.unsqueeze(0))
                    else:
                         cur_obs_list.pop(0)
                         cur_obs_list.append(resized_img.unsqueeze(0))
            else:
                while collision and collision_num < 8:
                    turn = True
                    collision_num += 1
                    current_yaw += 45 / 180 * np.pi
                    current_heading_point[:2] = current_position[:2] + np.array([np.cos(current_yaw), np.sin(current_yaw)]) * 0.02
                    resized_img = camera_set_and_record(env, current_position, current_heading_point, obs_img_size)
                    recovery_collision = collision if collision_accounting_mode == "legacy" else False
                    collision = check_collision(current_heading_point[:2], travers_map)
                    save_state.append(np.array([
                        current_position[0], current_position[1],
                        current_heading_point[0], current_heading_point[1],
                        recovery_collision,
                    ]))
                    if len(cur_obs_list) < context_size + 1:
                         cur_obs_list.append(resized_img.unsqueeze(0))
                    else:
                         cur_obs_list.pop(0)
                         cur_obs_list.append(resized_img.unsqueeze(0))
            break
    return current_position, current_heading_point, collision, turn

def camera_follow_traj(env, current_position, trajectory, orientation, cur_obs_list, context_size, obs_img_size):
    for i in range(len(trajectory)):
        if env.viewer is not None:
            env.viewer.initial_pos = trajectory[i]
            env.viewer.initial_view_direction = orientation[i] - trajectory[i]
            env.viewer.reset_viewer()
        frame = env.renderer.render(modes=("rgb"))
        img = Image.fromarray((255 * np.concatenate(frame, axis=1)[:, :, :3]).astype(np.uint8))
        resized_img = resize_and_aspect_crop(img, obs_img_size, IMAGE_ASPECT_RATIO)
        if len(cur_obs_list) < context_size + 1:
            cur_obs_list.append(resized_img.unsqueeze(0))
        else:
            cur_obs_list.pop(0)
            cur_obs_list.append(resized_img.unsqueeze(0))
        env.step()
    return trajectory[-1], orientation[-1]

def sub_waypoints_generator(subgoal, current_position):
    sub_waypoints = []
    for i in range(5):
        sub_waypoints.append(current_position + (subgoal - current_position) * (i + 1) / 5)
    return sub_waypoints


def build_exp_name_from_checkpoint(checkpoint_path: str) -> str:
    checkpoint = Path(checkpoint_path)
    parent_name = checkpoint.parent.name
    checkpoint_name = checkpoint.stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{parent_name}_{checkpoint_name}_{timestamp}"


def build_astar_local_cond(
    cur_pos_metric: np.ndarray,
    cur_heading_metric: np.ndarray,
    walkable_map: np.ndarray,
    goal_xy: np.ndarray,
    planner_metric_size_cm: float,
    len_traj_pred: int,
    metric_waypoint_spacing: float,
    waypoint_spacing: int,
) -> np.ndarray:
    start_xy = np.round(
        metric_positions_to_resized_pixels(cur_pos_metric, planner_metric_size_cm, PLANNER_FLOORPLAN_SIZE)
    ).astype(np.int32)
    path_xy = astar_path_on_walkable_map(walkable_map, start_xy, goal_xy)
    planner_succeeded = len(path_xy) > 0
    if not planner_succeeded:
        path_xy = np.stack([start_xy, goal_xy], axis=0)
    path_metric = resized_pixels_to_metric_positions(path_xy, planner_metric_size_cm, PLANNER_FLOORPLAN_SIZE)
    sampled_metric, valid_mask = sample_path_prefix_points(
        path_metric,
        num_points=len_traj_pred,
        step_size_m=metric_waypoint_spacing * waypoint_spacing,
        return_valid_mask=True,
    )
    if not planner_succeeded:
        valid_mask.fill(0.0)
    local_path_xy = to_local_coords(sampled_metric, cur_pos_metric, cur_heading_metric)
    return path_xy_to_local_condition(local_path_xy, valid_mask)


def realign_heading_to_astar(
    env,
    current_position,
    current_heading_point,
    astar_local_cond,
    cur_obs_list,
    obs_img_size,
    angle_threshold_deg=150.0,
):
    """Snap heading to the full-horizon A* chord when facing nearly backwards."""
    path_delta_local = np.asarray(
        astar_local_cond[-1, :2] - astar_local_cond[0, :2], dtype=np.float32
    )
    if np.linalg.norm(path_delta_local) < 1e-6:
        return current_heading_point, False

    relative_yaw = float(np.arctan2(path_delta_local[1], path_delta_local[0]))
    if abs(np.rad2deg(relative_yaw)) <= angle_threshold_deg:
        return current_heading_point, False

    current_yaw = np.arctan2(
        current_heading_point[1] - current_position[1],
        current_heading_point[0] - current_position[0],
    )
    path_yaw = current_yaw + relative_yaw
    new_heading_xy = current_position[:2] + np.array(
        [np.cos(path_yaw), np.sin(path_yaw)], dtype=np.float32
    ) * 0.01
    current_heading_point = np.array(current_heading_point, copy=True)
    current_heading_point[:2] = new_heading_xy

    resized_img = camera_set_and_record(
        env, current_position, current_heading_point, obs_img_size
    )
    refreshed_obs = resized_img.unsqueeze(0)
    for idx in range(len(cur_obs_list)):
        cur_obs_list[idx] = refreshed_obs

    print(f"A* heading realign: {np.rad2deg(relative_yaw):.1f} deg")
    return current_heading_point, True

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="test_vgc.yaml")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument(
        "--traj_per_scene",
        type=int,
        default=None,
        help="Maximum trajectories per scene. Defaults to traj_per_scene in the config file.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel worker processes. Each worker handles a shard of scenes.",
    )
    parser.add_argument(
        "--worker_id",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def load_eval_config(config_path: str, ckpt_path: Optional[str], exp_name: Optional[str]):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if ckpt_path:
        config["checkpoint_path"] = ckpt_path

    if exp_name:
        config["state_save_dir"] = os.path.join("results/exps", exp_name)
    else:
        auto_exp_name = build_exp_name_from_checkpoint(config["checkpoint_path"])
        config["state_save_dir"] = os.path.join("results/exps", auto_exp_name)

    config["obs_image_size"] = tuple(config.get("obs_image_size", DEFAULT_OBS_IMG_SIZE))

    return config


def normalize_gpu_ids(config):
    gpu_ids = config.get("gpu_ids", [0])
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]
    config["gpu_ids"] = gpu_ids
    return gpu_ids


def build_model_and_device(config, worker_id: int):
    worker_seed = config.get("seed")
    if worker_seed is not None:
        worker_seed += worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        cudnn.deterministic = True

    gpu_ids = normalize_gpu_ids(config)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[worker_id % len(gpu_ids)])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    cudnn.benchmark = True
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    vision_encoder = flona_ViNT(
        context_size=config["context_size"],
        obs_encoding_size=config["encoding_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
        shallow_feature_layers=tuple(config.get("shallow_feature_layers", [2, 5])),
        deep_feature_layer=int(config.get(
            "deep_feature_layer", config.get("deep_feature_layers", [11])[-1]
        )),
        curvature_gate_horizon=int(config.get("curvature_gate_horizon", 20)),
        curvature_stride=int(config.get("curvature_stride", 3)),
        curvature_smoothing_window=int(config.get("curvature_smoothing_window", 3)),
        shallow_residual_alpha_max=float(config.get("shallow_residual_alpha_max", 0.25)),
        use_shallow_visual_residual=bool(config.get("use_shallow_visual_residual", True)),
        obs_already_normalized=True,
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        local_cond_dim=5,
        global_cond_dim=config["encoding_size"],
        down_dims=config["down_dims"],
        cond_predict_scale=config["cond_predict_scale"],
    )
    dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
    model = flona(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    )
    bridge_diffusion = None
    if config.get("diffusion_mode", "ddpm") == "navibridger":
        bridge_diffusion = NaviBridgeDenoiser(NaviBridgeConfig(
            sigma_data=float(config.get("bridge_sigma_data", 0.5)),
            sigma_min=float(config.get("bridge_sigma_min", 0.002)),
            sigma_max=float(config.get("bridge_sigma_max", 10.0)),
            rho=float(config.get("bridge_rho", 7.0)),
            num_inference_steps=int(config["num_diffusion_iters"]),
            guidance=float(config.get("bridge_guidance", 1.0)),
            clip_denoised=bool(config.get("bridge_clip_denoised", True)),
            churn_step_ratio=float(config.get("bridge_churn_step_ratio", 0.33)),
            use_astar_local_cond=bool(config.get("use_astar_local_cond", True)),
            # Dropout is training-only; retain the configured value so train
            # and test configs describe the same model behavior explicitly.
            astar_local_cond_dropout_prob=float(
                config.get("astar_local_cond_dropout_prob", 0.0)
            ),
        ))
        noise_scheduler = None
    else:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon',
        )

    latest_flona_checkpoint = torch.load(config["checkpoint_path"], map_location='cpu')
    load_model(model, latest_flona_checkpoint)
    model = model.to(device)
    model.eval()
    return model, noise_scheduler, bridge_diffusion, transform, device


def get_assigned_scenes(test_scenes: List[str], num_workers: int, worker_id: int) -> List[str]:
    if num_workers <= 1:
        return list(test_scenes)
    return [scene for idx, scene in enumerate(test_scenes) if idx % num_workers == worker_id]


def ensure_output_dirs(state_save_dir_base: str):
    os.makedirs(state_save_dir_base, exist_ok=True)
    img_save_dir = os.path.join(state_save_dir_base, 'image')
    os.makedirs(img_save_dir, exist_ok=True)
    traj_state_save_dir = os.path.join(state_save_dir_base, 'trajectory')
    os.makedirs(traj_state_save_dir, exist_ok=True)
    return img_save_dir, traj_state_save_dir


def evaluate_scene(
    scene_name_floor,
    config,
    simulator,
    model,
    noise_scheduler,
    bridge_diffusion,
    transform,
    device,
    img_save_dir,
    traj_state_save_dir,
    traj_per_scene,
    planner_free_thresh,
    floor_shapes,
    progress_desc,
    depth_estimator=None,
):
    scene_id = scene_name_floor.split('_')[0]
    floor = int(scene_name_floor.split('_')[1])

    planner_floorplan_path = os.path.join("datasets", "scenes_117", "test", scene_name_floor, "floorplan.png")
    model_floorplan_path = os.path.join(config["testdataset"], scene_name_floor, "floorplan.png")
    if floor == 0:
        travers_path = os.path.join(config["trav_maps_path"], scene_id, 'floor_trav_test_0_modified.png')
    else:
        travers_path = os.path.join(config["trav_maps_path"], scene_name_floor, f'floor_trav_test_{floor}_modified.png')

    floorplan = img_path_to_data(model_floorplan_path, FLOORPLAN_IMG_SIZE).unsqueeze(0)
    if floorplan.shape[1] > 3:
        floorplan = floorplan[:, :3]

    floorplan_ary = np.array(Image.open(planner_floorplan_path).convert('RGBA'))
    planner_metric_size_cm = floor_shapes[scene_name_floor]
    walkable_map = floorplan_to_walkable_map(planner_floorplan_path, free_thresh=planner_free_thresh)
    travers_map = np.array(Image.open(travers_path).convert('L'))

    trajs_dir = os.path.join(config["testdataset"], f"{scene_id}_{floor}")
    if not os.path.exists(trajs_dir):
        return

    trajs_id = sorted([int(f.split('_')[1]) for f in os.listdir(trajs_dir) if f.startswith('traj')])
    selected_trajs = trajs_id[:min(traj_per_scene, len(trajs_id))]

    with open(os.path.join(config["scene_path"], scene_id, "floors.txt"), "r") as ff:
        heights = sorted(list(map(float, ff.readlines())))
    height = 0.85 + heights[floor]

    short_exec = True
    metric_waipoint_spacing = 0.045
    waypoint_spacing = 1
    len_traj_pred = int(config.get("len_traj_pred", 32))
    num_trajectory_samples = int(config.get("num_trajectory_samples", 30))
    execute_horizon = int(config.get("execute_horizon", 20))
    if len_traj_pred < 1:
        raise ValueError("len_traj_pred must be at least 1")
    if num_trajectory_samples < 1:
        raise ValueError("num_trajectory_samples must be at least 1")
    if not 1 <= execute_horizon <= len_traj_pred:
        raise ValueError(
            "execute_horizon must be between 1 and len_traj_pred"
        )
    max_steps = -1 if not short_exec else 90
    context_size = config["context_size"]
    obs_img_size = tuple(config.get("obs_image_size", DEFAULT_OBS_IMG_SIZE))
    arrive_th = float(config.get("arrive_th", 0.3))
    max_collision_count = 11
    astar_heading_realign = bool(config.get("astar_heading_realign", False))
    astar_heading_realign_threshold_deg = float(
        config.get("astar_heading_realign_threshold_deg", 150.0)
    )
    collision_accounting_mode = config.get("collision_accounting_mode", "legacy")
    if collision_accounting_mode not in {"legacy", "movement_only"}:
        raise ValueError(
            "collision_accounting_mode must be 'legacy' or 'movement_only'"
        )
    recovery_turn_mode = config.get(
        "recovery_turn_mode", "continuous_right"
    )
    if recovery_turn_mode not in {"random", "continuous_right"}:
        raise ValueError(
            "recovery_turn_mode must be 'random' or 'continuous_right'"
        )

    for traj_id in tqdm(selected_trajs, desc=progress_desc, leave=False):
        traj_name = f'traj_{traj_id}'
        if config.get('paired_episode_seed') is not None:
            key = f"{config['paired_episode_seed']}:{scene_name_floor}:{traj_id}"
            episode_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little')
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(episode_seed)
        traj_file = os.path.join(trajs_dir, traj_name, f'{traj_name}.npy')
        all_traj = np.load(traj_file)
        start_time, goal_time = 4, -1

        cur_obs_list = []
        trajectory_to_execute = [np.array([all_traj[idx][0], all_traj[idx][1], height]) for idx in range(start_time - context_size, start_time + 1)]
        heading_to_execute = [np.array([all_traj[idx][2], all_traj[idx][3], height]) for idx in range(start_time - context_size, start_time + 1)]
        goal_pos = np.array([[all_traj[goal_time][0], all_traj[goal_time][1]]])

        current_position, current_heading_point = camera_follow_traj(
            simulator,
            trajectory_to_execute[0],
            trajectory_to_execute,
            heading_to_execute,
            cur_obs_list,
            context_size,
            obs_img_size,
        )

        save_num, save_state, step = 0, [], 1
        total_collision_count = 0
        collision = False
        traj_img_save_dir = os.path.join(img_save_dir, f"{scene_name_floor}_{traj_name}")
        os.makedirs(traj_img_save_dir, exist_ok=True)
        goal_xy = np.round(
            metric_positions_to_resized_pixels(goal_pos[0], planner_metric_size_cm, PLANNER_FLOORPLAN_SIZE)
        ).astype(np.int32)

        while step <= max_steps:
            turn = False
            if len(trajectory_to_execute) > 0:
                current_position, current_heading_point, collision, turn = camera_execute_actions(
                    simulator,
                    current_position,
                    trajectory_to_execute,
                    cur_obs_list,
                    context_size,
                    travers_map,
                    save_state,
                    obs_img_size,
                    collision_accounting_mode=collision_accounting_mode,
                    recovery_turn_mode=recovery_turn_mode,
                )
                trajectory_to_execute = []

            if collision:
                total_collision_count += 1
                if max_collision_count > 0 and total_collision_count >= max_collision_count:
                    print(f'Reached {max_collision_count} collisions, moving to next trajectory')
                    break
                print('collision occurred, attempting recovery...')

            if len(cur_obs_list) == context_size + 1:
                cur_pos = np.array([current_position[:2]])
                cur_heading_raw = np.array([current_heading_point[:2]])
                cur_heading = cur_pos + (cur_heading_raw - cur_pos) / np.linalg.norm(cur_heading_raw - cur_pos)

                if np.linalg.norm(cur_pos - goal_pos) < arrive_th:
                    print("arrive at the target!")
                    break

                astar_local_cond = build_astar_local_cond(
                    cur_pos_metric=cur_pos[0],
                    cur_heading_metric=cur_heading[0],
                    walkable_map=walkable_map,
                    goal_xy=goal_xy,
                    planner_metric_size_cm=planner_metric_size_cm,
                    len_traj_pred=len_traj_pred,
                    metric_waypoint_spacing=metric_waipoint_spacing,
                    waypoint_spacing=waypoint_spacing,
                )
                if astar_heading_realign and not turn:
                    current_heading_point, did_realign = realign_heading_to_astar(
                        simulator,
                        current_position,
                        current_heading_point,
                        astar_local_cond,
                        cur_obs_list,
                        obs_img_size,
                        angle_threshold_deg=astar_heading_realign_threshold_deg,
                    )
                    if did_realign:
                        cur_heading_raw = np.array([current_heading_point[:2]])
                        cur_heading = cur_pos + (cur_heading_raw - cur_pos) / np.linalg.norm(
                            cur_heading_raw - cur_pos
                        )
                        astar_local_cond = build_astar_local_cond(
                            cur_pos_metric=cur_pos[0],
                            cur_heading_metric=cur_heading[0],
                            walkable_map=walkable_map,
                            goal_xy=goal_xy,
                            planner_metric_size_cm=planner_metric_size_cm,
                            len_traj_pred=len_traj_pred,
                            metric_waypoint_spacing=metric_waipoint_spacing,
                            waypoint_spacing=waypoint_spacing,
                        )

                cur_obs = torch.cat(cur_obs_list, dim=0)
                energy_guidance = None
                if depth_estimator is not None:
                    from model.depth_bridge_guidance import depth_to_map, BridgeDepthGuidance
                    # Additional RGB rendering only, at the actual level-camera
                    # pose. No simulator stepping and no ground-truth depth.
                    simulator.renderer.set_camera(current_position, current_heading_point, [0,0,1])
                    rgb_frame = simulator.renderer.render(modes=("rgb"))
                    rgb = (np.clip(np.concatenate(rgb_frame, axis=1)[:,:,:3],0,1)*255).astype(np.uint8)
                    depth = depth_estimator.predict(rgb)
                    guidance_config = config['depth_bridge_guidance']
                    local_map = depth_to_map(depth, simulator.renderer.get_intrinsics(), guidance_config, device)
                    goal_local = to_local_coords(goal_pos, cur_pos[0], cur_heading[0])
                    energy_guidance = BridgeDepthGuidance(local_map, goal_local, guidance_config,
                        metric_waipoint_spacing * waypoint_spacing)
                actions_2d = execute_model(
                    model,
                    cur_pos,
                    cur_heading,
                    goal_pos,
                    astar_local_cond[None],
                    cur_obs,
                    floorplan,
                    metric_waipoint_spacing,
                    waypoint_spacing,
                    transform,
                    device,
                    noise_scheduler,
                    floorplan_ary.copy(),
                    planner_metric_size_cm,
                    os.path.join(traj_img_save_dir, f'{save_num}.png'),
                    bridge_diffusion=bridge_diffusion,
                    pred_horizon=len_traj_pred,
                    num_trajectory_samples=num_trajectory_samples,
                    execute_horizon=execute_horizon,
                    energy_guidance=energy_guidance,
                )
                if energy_guidance is not None:
                    with open(os.path.join(traj_img_save_dir, 'depth_guidance.jsonl'), 'a') as diagnostic:
                        diagnostic.write(json.dumps(dict(cycle=save_num,
                            depth_valid_fraction=local_map.valid_fraction,
                            known_fraction=local_map.known_fraction, occupied_cells=local_map.occupied_cells,
                            visible_goal=energy_guidance.visible, updates=energy_guidance.records))+'\n')
                    if save_num == 0:
                        Image.fromarray(rgb).save(os.path.join(traj_img_save_dir, 'depth_input_rgb.png'))
                        np.save(os.path.join(traj_img_save_dir, 'predicted_depth_m.npy'), depth)
                save_num += 1
                for k in range(execute_horizon):
                    trajectory_to_execute.append(np.array([actions_2d[k][0], actions_2d[k][1], height]))
            else:
                frame = simulator.renderer.render(modes=("rgb"))
                img = Image.fromarray((255 * np.concatenate(frame, axis=1)[:, :, :3]).astype(np.uint8))
                cur_obs_list.append(resize_and_aspect_crop(img, obs_img_size, IMAGE_ASPECT_RATIO).unsqueeze(0))
            step += 1

        save_state_arr = np.array(save_state)
        state_save_file = os.path.join(traj_state_save_dir, f"{scene_name_floor}_{traj_name}.txt")
        np.savetxt(state_save_file, save_state_arr, fmt='%f')


def run_worker(args):
    config = load_eval_config(args.config, args.ckpt_path, args.exp_name)
    worker_id = 0 if args.worker_id is None else args.worker_id
    traj_per_scene = int(config.get("traj_per_scene", 10)) if args.traj_per_scene is None else args.traj_per_scene
    assigned_scenes = get_assigned_scenes(config["test_scenes"], args.num_workers, worker_id)

    print(
        f"Worker {worker_id}/{args.num_workers}: {len(assigned_scenes)} scenes, "
        f"traj_per_scene={traj_per_scene}, results -> {config['state_save_dir']}"
    )
    if not assigned_scenes:
        return

    model, noise_scheduler, bridge_diffusion, transform, device = build_model_and_device(config, worker_id)
    depth_estimator = None
    if config.get('depth_bridge_guidance', {}).get('enabled', False):
        if bridge_diffusion is None:
            raise ValueError('depth_bridge_guidance is only supported by NaviBridge')
        from model.depth_bridge_guidance import MetricDepthEstimator
        depth_estimator = MetricDepthEstimator(config['depth_bridge_guidance'], device)
    planner_free_thresh = float(config.get("planner_free_thresh", 0.96))
    floor_shapes = np.load(os.path.join(config["trav_maps_path"], "floor_shapes.npy"), allow_pickle=True).item()

    headless = True
    settings = MeshRendererSettings(enable_shadow=False, msaa=False)
    img_save_dir, traj_state_save_dir = ensure_output_dirs(config["state_save_dir"])

    for scene_name_floor in tqdm(assigned_scenes, desc=f"Scenes[w{worker_id}]"):
        scene_id = scene_name_floor.split('_')[0]
        simulator = Simulator(
            mode="gui_interactive" if not headless else "headless",
            image_width=512,
            image_height=512,
            rendering_settings=settings,
            device_idx=0,
        )
        scene = StaticIndoorScene(scene_id, build_graph=True)
        simulator.import_scene(scene)
        try:
            evaluate_scene(
                scene_name_floor=scene_name_floor,
                config=config,
                simulator=simulator,
                model=model,
                noise_scheduler=noise_scheduler,
                bridge_diffusion=bridge_diffusion,
                transform=transform,
                device=device,
                img_save_dir=img_save_dir,
                traj_state_save_dir=traj_state_save_dir,
                traj_per_scene=traj_per_scene,
                planner_free_thresh=planner_free_thresh,
                floor_shapes=floor_shapes,
                progress_desc=f"Trajs in {scene_name_floor} [w{worker_id}]",
                depth_estimator=depth_estimator,
            )
        finally:
            simulator.disconnect()


def build_worker_command(args, worker_id: int):
    command = [
        sys.executable,
        current_script_path,
        "--config",
        args.config,
        "--num_workers",
        str(args.num_workers),
        "--worker_id",
        str(worker_id),
    ]
    if args.traj_per_scene is not None:
        command.extend(["--traj_per_scene", str(args.traj_per_scene)])
    if args.exp_name:
        command.extend(["--exp_name", args.exp_name])
    if args.ckpt_path:
        command.extend(["--ckpt_path", args.ckpt_path])
    return command


def launch_workers(args):
    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1")

    processes = []
    try:
        for worker_id in range(args.num_workers):
            command = build_worker_command(args, worker_id)
            print(f"Launching worker {worker_id}: {' '.join(command)}")
            processes.append((worker_id, subprocess.Popen(command)))

        failed_workers = []
        for worker_id, process in processes:
            return_code = process.wait()
            if return_code != 0:
                failed_workers.append((worker_id, return_code))

        if failed_workers:
            failed_msg = ", ".join([f"worker {worker_id} -> {return_code}" for worker_id, return_code in failed_workers])
            raise RuntimeError(f"Parallel evaluation failed: {failed_msg}")
    except KeyboardInterrupt:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        raise


def main():
    args = parse_args()

    torch.multiprocessing.set_start_method("spawn", force=True)
    config = load_eval_config(args.config, args.ckpt_path, args.exp_name)
    if args.exp_name is None:
        args.exp_name = os.path.basename(config["state_save_dir"])
    print(f"Results will be saved to: {config['state_save_dir']}")

    if args.worker_id is None and args.num_workers > 1:
        launch_workers(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
