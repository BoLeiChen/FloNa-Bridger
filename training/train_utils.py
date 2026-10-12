
import os
import itertools
import math
from typing import Any, Dict, List, Optional, Tuple
# from prettytable import PrettyTable

import tqdm
import numpy as np
import wandb
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt 
import copy
import matplotlib.gridspec as gridspec

from training.Logger import Logger
from model.data_utils import VISUALIZATION_IMAGE_SIZE
from model.data_utils import to_global_coords, metric_positions_to_resized_pixels
from model.navibridger import NaviBridgeDenoiser, waypoints_to_normalized_deltas

ACTION_STATS = {}
ACTION_STATS["min"] = np.array([-2.5, -4])
ACTION_STATS["max"] = np.array([5, 4])    
RED = np.array([1, 0, 0])
GREEN = np.array([0, 0.5, 0])
CYAN = np.array([0, 1, 1])
MAGENTA = np.array([1, 0, 1])
ORANGE = np.array([1.0, 0.55, 0.0])


def _unwrap_dataset(dataset: Any):
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def _metric_positions_to_travers_map_pixels(
    metric_positions_m: np.ndarray,
    travers_map: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(metric_positions_m, dtype=np.float32)
    squeeze = positions.ndim == 1
    if squeeze:
        positions = positions[None, :]
    pixel_positions = np.round(
        positions * 100.0
        + np.array([travers_map.shape[0], travers_map.shape[1]], dtype=np.float32) / 2.0
    ).astype(np.int32)
    in_bounds = (
        (pixel_positions[:, 0] >= 0)
        & (pixel_positions[:, 0] < travers_map.shape[1])
        & (pixel_positions[:, 1] >= 0)
        & (pixel_positions[:, 1] < travers_map.shape[0])
    )
    return (pixel_positions[0], in_bounds[0]) if squeeze else (pixel_positions, in_bounds)


def _analyze_predicted_path_collision(
    cur_pos_m: np.ndarray,
    goal_pos_m: np.ndarray,
    pred_waypoints_m: np.ndarray,
    travers_map: np.ndarray,
    execute_horizon: int = 20,
    substeps_per_segment: int = 5,
) -> Dict[str, Any]:
    waypoints_m = np.asarray(pred_waypoints_m[:execute_horizon], dtype=np.float32)
    cur_pos_m = np.asarray(cur_pos_m, dtype=np.float32)
    goal_pos_m = np.asarray(goal_pos_m, dtype=np.float32)
    sampled_points_m: List[np.ndarray] = []
    collision_points_m: List[np.ndarray] = []
    collision_flags: List[bool] = []

    prev = cur_pos_m
    for waypoint_m in waypoints_m:
        for step_idx in range(substeps_per_segment):
            point_m = prev + (waypoint_m - prev) * float(step_idx + 1) / float(substeps_per_segment)
            point_xy, in_bounds = _metric_positions_to_travers_map_pixels(point_m, travers_map)
            collided = (not bool(in_bounds)) or travers_map[point_xy[1], point_xy[0]] == 0
            sampled_points_m.append(point_m.copy())
            collision_flags.append(collided)
            if collided:
                collision_points_m.append(point_m.copy())
        prev = waypoint_m

    path_points_m = np.concatenate([cur_pos_m[None], waypoints_m], axis=0) if len(waypoints_m) > 0 else cur_pos_m[None]
    return {
        "has_collision": any(collision_flags),
        "collision_count": int(sum(collision_flags)),
        "sample_count": int(len(collision_flags)),
        "scene_name": "",
        "path_points_m": path_points_m,
        "sampled_points_m": np.stack(sampled_points_m, axis=0) if sampled_points_m else np.zeros((0, 2), dtype=np.float32),
        "collision_points_m": np.stack(collision_points_m, axis=0) if collision_points_m else np.zeros((0, 2), dtype=np.float32),
        "goal_pos_m": goal_pos_m,
        "cur_pos_m": cur_pos_m,
        "travers_map": travers_map,
    }


def _compute_batch_collision_metrics(
    pred_actions: torch.Tensor,
    curr_pos: torch.Tensor,
    curr_ori: torch.Tensor,
    goal_pos: torch.Tensor,
    scene_names: List[str],
    collision_dataset: Any,
) -> Dict[str, Any]:
    if collision_dataset is None or scene_names is None:
        return {
            "collision_rate": float("nan"),
            "collision_hits": 0,
            "collision_total": 0,
            "collision_debug": None,
        }

    dataset = _unwrap_dataset(collision_dataset)
    metric_scale = float(dataset.metric_waypoint_spacing) * float(dataset.waypoint_spacing)
    pred_actions_np = to_numpy(pred_actions)
    curr_pos_np = to_numpy(curr_pos)
    curr_ori_np = to_numpy(curr_ori)
    goal_pos_np = to_numpy(goal_pos)

    collision_hits = 0
    collision_total = len(scene_names)
    collision_debug = None

    for idx, scene_name in enumerate(scene_names):
        pred_global_norm = to_global_coords(pred_actions_np[idx], curr_pos_np[idx], curr_ori_np[idx])
        pred_global_m = pred_global_norm * metric_scale
        cur_pos_m = curr_pos_np[idx] * metric_scale
        goal_pos_m = goal_pos_np[idx] * metric_scale
        travers_map = dataset.get_travers_map(scene_name)
        collision_info = _analyze_predicted_path_collision(
            cur_pos_m=cur_pos_m,
            goal_pos_m=goal_pos_m,
            pred_waypoints_m=pred_global_m,
            travers_map=travers_map,
        )
        if collision_info["has_collision"]:
            collision_hits += 1
            if collision_debug is None:
                collision_info["scene_name"] = scene_name
                collision_debug = collision_info

    collision_rate = float(collision_hits) / float(collision_total) if collision_total > 0 else float("nan")
    return {
        "collision_rate": collision_rate,
        "collision_hits": collision_hits,
        "collision_total": collision_total,
        "collision_debug": collision_debug,
    }


def visualize_collision_prediction(
    collision_debug: Dict[str, Any],
    type: str,
    project_folder: str,
    epoch: int,
    step_idx: int,
    use_wandb: bool = True,
):
    if collision_debug is None:
        return

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        type,
        f"epoch{epoch}",
        "collision_prediction",
    )
    os.makedirs(visualize_path, exist_ok=True)

    travers_map = collision_debug["travers_map"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    path_xy, _ = _metric_positions_to_travers_map_pixels(collision_debug["path_points_m"], travers_map)
    sampled_xy, sampled_in_bounds = _metric_positions_to_travers_map_pixels(collision_debug["sampled_points_m"], travers_map)
    collision_xy, _ = _metric_positions_to_travers_map_pixels(collision_debug["collision_points_m"], travers_map)
    start_xy, _ = _metric_positions_to_travers_map_pixels(collision_debug["cur_pos_m"], travers_map)
    goal_xy, _ = _metric_positions_to_travers_map_pixels(collision_debug["goal_pos_m"], travers_map)

    for ax in axes:
        ax.imshow(travers_map, cmap="gray", origin="upper")
        ax.plot(path_xy[:, 0], path_xy[:, 1], color="lime", linewidth=2, marker="o", markersize=2)
        if len(sampled_xy) > 0:
            valid_sampled = sampled_xy[np.asarray(sampled_in_bounds, dtype=bool)]
            if len(valid_sampled) > 0:
                ax.scatter(valid_sampled[:, 0], valid_sampled[:, 1], s=8, c="yellow", alpha=0.5)
        if len(collision_xy) > 0:
            ax.scatter(collision_xy[:, 0], collision_xy[:, 1], s=24, c="red")
        ax.scatter([start_xy[0]], [start_xy[1]], c="cyan", s=40)
        ax.scatter([goal_xy[0]], [goal_xy[1]], c="blue", s=40)
        ax.set_axis_off()

    axes[0].set_title(f"{type} collision path: {collision_debug['scene_name']}")
    axes[1].set_title("Zoomed start region")
    half_window = 120
    axes[1].set_xlim(max(0, start_xy[0] - half_window), min(travers_map.shape[1], start_xy[0] + half_window))
    axes[1].set_ylim(min(travers_map.shape[0], start_xy[1] + half_window), max(0, start_xy[1] - half_window))
    fig.tight_layout()

    save_path = os.path.join(visualize_path, f"step_{step_idx}.png")
    plt.savefig(save_path)
    plt.close(fig)

    if use_wandb:
        wandb.log({f"{type}_collision_example": wandb.Image(save_path)}, commit=False)

def _compute_losses_flona(
    ema_model,
    noise_scheduler,
    batch_obs_images,
    batch_floorplan_images,
    batch_dist_label: torch.Tensor,
    batch_action_label: torch.Tensor,
    goal_pos: torch.Tensor,
    curr_pos: torch.Tensor,  
    curr_ori: torch.Tensor,
    astar_local_cond: torch.Tensor,
    device: torch.device,
    collision_dataset: Any = None,
    scene_names: Optional[List[str]] = None,
    bridge_diffusion: Optional[NaviBridgeDenoiser] = None,
    astar_global_cond: Optional[torch.Tensor] = None,
):
    """
    Compute losses for distance and action prediction.
    """
    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]
    num_samples = 1
    model_output_dict = model_output(
        ema_model,
        noise_scheduler,
        batch_obs_images,
        batch_floorplan_images,
        pred_horizon,
        action_dim,
        num_samples,
        goal_pos,
        curr_pos,
        curr_ori,
        astar_local_cond,
        device=device,
        bridge_diffusion=bridge_diffusion,
        astar_global_cond=astar_global_cond,
    )
    actions = model_output_dict['actions']
    # distance = model_output_dict['distance']

    # dist_loss = F.mse_loss(distance, batch_dist_label.unsqueeze(-1))

    def action_reduce(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        return unreduced_loss.mean()


    action_loss = action_reduce(F.mse_loss(actions, batch_action_label, reduction="none"))

    action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        actions[:, :, :2], batch_action_label[:, :, :2], dim=-1
    ))

    multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(actions[:, :, :2], start_dim=1),
        torch.flatten(batch_action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    results = {
        "action_loss": action_loss,
        "action_waypts_cos_sim": action_waypts_cos_similairity,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim,
        # "dist_loss": dist_loss,
    }
    collision_metrics = _compute_batch_collision_metrics(
        pred_actions=actions,
        curr_pos=curr_pos,
        curr_ori=curr_ori,
        goal_pos=goal_pos,
        scene_names=scene_names,
        collision_dataset=collision_dataset,
    )
    results["collision_rate"] = collision_metrics["collision_rate"]
    results["collision_hits"] = collision_metrics["collision_hits"]
    results["collision_total"] = collision_metrics["collision_total"]
    results["collision_debug"] = collision_metrics["collision_debug"]
    return results

@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def _heading_points_to_yaw(curr_pos_m: torch.Tensor, curr_ori_m: torch.Tensor) -> torch.Tensor:
    direction = curr_ori_m - curr_pos_m
    return torch.atan2(direction[:, 1], direction[:, 0])


def _yaw_to_heading_points(curr_pos_m: torch.Tensor, yaw: torch.Tensor, length_m: float = 1.0) -> torch.Tensor:
    direction = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1) * length_m
    return curr_pos_m + direction


def _local_to_global_torch(local_points_m: torch.Tensor, curr_pos_m: torch.Tensor, curr_ori_m: torch.Tensor) -> torch.Tensor:
    yaw = _heading_points_to_yaw(curr_pos_m, curr_ori_m)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    rot = torch.stack(
        [
            torch.stack([cos_yaw, -sin_yaw], dim=-1),
            torch.stack([sin_yaw, cos_yaw], dim=-1),
        ],
        dim=1,
    )
    return curr_pos_m[:, None, :] + torch.matmul(local_points_m, rot.transpose(1, 2))


def _global_to_local_torch(global_points_m: torch.Tensor, curr_pos_m: torch.Tensor, curr_ori_m: torch.Tensor) -> torch.Tensor:
    yaw = _heading_points_to_yaw(curr_pos_m, curr_ori_m)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    rot = torch.stack(
        [
            torch.stack([cos_yaw, sin_yaw], dim=-1),
            torch.stack([-sin_yaw, cos_yaw], dim=-1),
        ],
        dim=1,
    )
    centered = global_points_m - curr_pos_m[:, None, :]
    return torch.matmul(centered, rot.transpose(1, 2))


def _path_noise_bound(schedule_epoch: int, schedule_total_epochs: int, alpha_m: float) -> float:
    if schedule_total_epochs <= 0:
        progress = 1.0
    else:
        progress = float(schedule_epoch) / float(schedule_total_epochs)
    progress = min(max(progress, 0.0), 1.0)
    return float(alpha_m) * math.sin(progress * math.pi / 2.0)


def _structured_path_noise(
    path_m: torch.Tensor,
    valid_mask: torch.Tensor,
    bound_m: float,
    control_points: int,
    translation_std_m: float,
) -> torch.Tensor:
    """Generate spatially correlated path uncertainty, not pointwise jitter."""
    batch_size, horizon, _ = path_m.shape
    control_points = max(2, min(int(control_points), horizon))
    coarse = torch.rand(
        batch_size, 2, control_points, device=path_m.device, dtype=path_m.dtype
    ) * 2.0 - 1.0
    smooth = F.interpolate(
        coarse, size=horizon, mode="linear", align_corners=True
    ).transpose(1, 2)
    # Anchor the first waypoint so low-frequency deformation changes shape
    # smoothly instead of teleporting the beginning of the path.
    smooth = (smooth - smooth[:, :1]).clamp(-1.0, 1.0) * float(bound_m)
    translation = torch.randn(
        batch_size, 1, 2, device=path_m.device, dtype=path_m.dtype
    ) * float(translation_std_m)
    noise = smooth + translation

    # Padded endpoint copies must stay copies; otherwise xT receives a noisy
    # tail even though the local-condition mask marks it invalid.
    valid = valid_mask > 0.5
    indices = torch.arange(horizon, device=path_m.device).expand(batch_size, -1)
    last_valid = torch.where(valid, indices, torch.zeros_like(indices)).cummax(dim=1).values
    noise = noise.gather(1, last_valid[..., None].expand(-1, -1, 2))
    return noise


def _normalize_astar_local_condition(
    astar_local_cond: torch.Tensor,
    metric_scale: float,
) -> torch.Tensor:
    """Normalize only local-condition coordinates; geometry features are unitless."""
    normalized = astar_local_cond.float().clone()
    normalized[..., :2] /= metric_scale
    return normalized


def _prepare_astar_local_condition_for_model(
    astar_local_cond: torch.Tensor,
    perturb_config: dict,
) -> torch.Tensor:
    """Convert dataset A* coordinates from meters to model waypoint units."""
    metric_scale = (
        float(perturb_config["metric_waypoint_spacing"])
        * float(perturb_config["waypoint_spacing"])
    )
    return _normalize_astar_local_condition(astar_local_cond, metric_scale)


def _path_xy_to_local_condition_torch(
    path_local_xy: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Torch equivalent of [x, y, sin(turn), cos(turn), valid] encoding."""
    valid_mask = valid_mask.to(dtype=path_local_xy.dtype)
    valid = valid_mask > 0.5
    sin_turn = torch.zeros_like(valid_mask)
    cos_turn = valid.to(dtype=path_local_xy.dtype)
    if path_local_xy.shape[-2] >= 3:
        incoming = path_local_xy[..., 1:-1, :] - path_local_xy[..., :-2, :]
        outgoing = path_local_xy[..., 2:, :] - path_local_xy[..., 1:-1, :]
        usable = (
            valid[..., :-2]
            & valid[..., 1:-1]
            & valid[..., 2:]
            & (torch.linalg.vector_norm(incoming, dim=-1) > 1e-6)
            & (torch.linalg.vector_norm(outgoing, dim=-1) > 1e-6)
        )
        incoming_yaw = torch.atan2(incoming[..., 1], incoming[..., 0])
        outgoing_yaw = torch.atan2(outgoing[..., 1], outgoing[..., 0])
        turn = torch.remainder(outgoing_yaw - incoming_yaw + math.pi, 2.0 * math.pi) - math.pi
        sin_turn[..., 1:-1] = torch.where(usable, torch.sin(turn), sin_turn[..., 1:-1])
        cos_turn[..., 1:-1] = torch.where(usable, torch.cos(turn), cos_turn[..., 1:-1])
    return torch.cat(
        [
            path_local_xy,
            sin_turn.unsqueeze(-1),
            cos_turn.unsqueeze(-1),
            valid_mask.unsqueeze(-1),
        ],
        dim=-1,
    )


def _clamp_marker_xy(point_xy: torch.Tensor, width: int, height: int) -> Tuple[int, int]:
    x = int(torch.clamp(point_xy[0].round(), 0, width - 1).item())
    y = int(torch.clamp(point_xy[1].round(), 0, height - 1).item())
    return x, y


def apply_paper_train_perturbations(
    curr_pos: torch.Tensor,
    curr_ori: torch.Tensor,
    astar_local_cond: torch.Tensor,
    schedule_epoch: int,
    schedule_total_epochs: int,
    perturb_config: Optional[dict],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if perturb_config is None:
        return curr_pos.float(), curr_ori.float(), astar_local_cond.float(), 0.0

    if astar_local_cond.shape[-1] != 5:
        raise ValueError(
            "A* local condition must be [x, y, sin(turn), cos(turn), valid]"
        )

    metric_scale = float(perturb_config["metric_waypoint_spacing"]) * float(perturb_config["waypoint_spacing"])
    curr_pos_m = curr_pos.float() * metric_scale
    curr_ori_m = curr_ori.float() * metric_scale
    astar_local_xy_m = astar_local_cond[..., :2].float() * metric_scale
    valid_mask = astar_local_cond[..., 4].float()

    global_path_m = _local_to_global_torch(astar_local_xy_m, curr_pos_m, curr_ori_m)
    noisy_curr_pos_m = curr_pos_m.clone()
    noisy_curr_ori_m = curr_ori_m.clone()

    if bool(perturb_config.get("pose_noise_apply", False)):
        pos_std_m = float(perturb_config["pose_noise_pos_std_m"])
        rot_std_rad = float(perturb_config["pose_noise_rot_std_rad"])
        noisy_curr_pos_m = curr_pos_m + torch.randn_like(curr_pos_m) * pos_std_m
        noisy_yaw = _heading_points_to_yaw(curr_pos_m, curr_ori_m) + torch.randn(
            curr_pos_m.shape[0],
            device=curr_pos_m.device,
            dtype=curr_pos_m.dtype,
        ) * rot_std_rad
        noisy_curr_ori_m = _yaw_to_heading_points(noisy_curr_pos_m, noisy_yaw)

    path_noise_bound_m = 0.0
    if bool(perturb_config.get("path_noise_apply", False)):
        path_noise_bound_m = _path_noise_bound(
            schedule_epoch=schedule_epoch,
            schedule_total_epochs=schedule_total_epochs,
            alpha_m=float(perturb_config["path_noise_alpha_m"]),
        )
        if path_noise_bound_m > 0.0:
            noise_mode = str(perturb_config.get("path_noise_mode", "structured"))
            if noise_mode != "structured":
                raise ValueError(
                    f"Unsupported path_noise_mode={noise_mode!r}; use 'structured'"
                )
            global_path_m = global_path_m + _structured_path_noise(
                global_path_m,
                valid_mask,
                bound_m=path_noise_bound_m,
                control_points=int(perturb_config.get("path_noise_control_points", 4)),
                translation_std_m=float(
                    perturb_config.get("path_noise_translation_std_m", 0.01)
                ),
            )

    noisy_astar_local_xy_m = _global_to_local_torch(global_path_m, noisy_curr_pos_m, noisy_curr_ori_m)
    noisy_curr_pos = noisy_curr_pos_m / metric_scale
    noisy_curr_ori = noisy_curr_ori_m / metric_scale
    noisy_astar_local_xy = noisy_astar_local_xy_m / metric_scale
    noisy_astar_local_cond = _path_xy_to_local_condition_torch(
        noisy_astar_local_xy, valid_mask
    )
    return noisy_curr_pos, noisy_curr_ori, noisy_astar_local_cond, path_noise_bound_m
def train_flona(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: AdamW,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    project_folder: str,
    epoch: int,
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    total_epochs: int = 1,
    schedule_epoch: Optional[int] = None,
    schedule_total_epochs: Optional[int] = None,
    perturb_config: Optional[dict] = None,
    bridge_diffusion: Optional[NaviBridgeDenoiser] = None,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        wand_log_freq: how often to log to wandb
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    model.train()
    ema_model.averaged_model.eval()
    num_batches = len(dataloader)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", "train", window_size=print_log_freq)
    multi_action_waypts_cos_sim_logger = Logger("multi_action_waypts_cos_sim", "train", window_size=print_log_freq)
    collision_rate_logger = Logger("collision_rate", "train", window_size=print_log_freq)
    loggers = {
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
        "collision_rate": collision_rate_logger,
    }
    collision_dataset = _unwrap_dataset(dataloader.dataset)
   
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                floorplan_image,
                actions,
                distance,
                goal_pos,
                curr_pos,
                curr_ori,
                goal_pos_resized,
                curr_pos_resized,
                curr_ori_resized,
                goal_pos_local,
                curr_pos_local,
                astar_local_cond,
                scene_names,
            ) = data
           
            obs_images = torch.split(obs_image, 3, dim=1)
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_floorplan_images = TF.resize(floorplan_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_floorplan_images = transform(floorplan_image).to(device)

            # Plot floorplan markers using floorplan image scaling, not observation scaling.
            fh0, fw0 = floorplan_image.shape[-2:]
            fh1, fw1 = batch_viz_floorplan_images.shape[-2:]
            floorplan_scale = torch.tensor([fw1 / fw0, fh1 / fh0], dtype=torch.float32)
            goal_pos_resized = goal_pos_resized * floorplan_scale
            curr_pos_resized = curr_pos_resized * floorplan_scale
            curr_ori_resized = curr_ori_resized * floorplan_scale
            for b in range(goal_pos_resized.shape[0]):
                goal_x, goal_y = _clamp_marker_xy(goal_pos_resized[b], fw1, fh1)
                curr_x, curr_y = _clamp_marker_xy(curr_pos_resized[b], fw1, fh1)
                ori_x, ori_y = _clamp_marker_xy(curr_ori_resized[b], fw1, fh1)
                batch_viz_floorplan_images[b, :, goal_y, goal_x] = torch.tensor(RED)
                batch_viz_floorplan_images[b, :, curr_y, curr_x] = torch.tensor(GREEN)
                batch_viz_floorplan_images[b, :, ori_y, ori_x] = torch.tensor(CYAN)

            # Dataset A* coordinates are metric for every diffusion mode. Keep
            # DDPM and NaviBridge on the same waypoint scale before applying
            # metric-space pose/path perturbations and before conditioning.
            model_astar_local_cond = _prepare_astar_local_condition_for_model(
                astar_local_cond, perturb_config
            )

            noisy_curr_pos, noisy_curr_ori, noisy_astar_local_cond, path_noise_bound_m = apply_paper_train_perturbations(
                curr_pos,
                curr_ori,
                model_astar_local_cond,
                schedule_epoch=schedule_epoch if schedule_epoch is not None else epoch + 1,
                schedule_total_epochs=schedule_total_epochs if schedule_total_epochs is not None else total_epochs,
                perturb_config=perturb_config,
            )

            B = actions.shape[0] 
            obsfloorplan_cond = model("vision_encoder", obs_img=batch_obs_images,
                                      floorplan_img=batch_floorplan_images,
                                      obs_pos = noisy_curr_pos,
                                      goal_pos = goal_pos,
                                      obs_ori = noisy_curr_ori,
                                      # The global encoder sees only a robust
                                      # summary of the clean A* geometry.  The
                                      # perturbed full path remains confined to
                                      # the bridge endpoint and UNet local branch.
                                      astar_global_cond=model_astar_local_cond.to(device),
                                      )
            # Get distance label
            distance = distance.float().to(device)
            deltas = get_delta(actions)
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = from_numpy(ndeltas).to(device)
            assert naction.shape[-1] == 2, "action dim must be 2"

            # Predict distance
            dist_pred = model("dist_pred_net", obsfloorplan_cond=obsfloorplan_cond)
            dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), distance)

            def action_reduce(unreduced_loss: torch.Tensor):
                # Reduce over non-batch dimensions to get loss per batch element
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)              
                return unreduced_loss.mean()
            if bridge_diffusion is not None:
                astar_prior = waypoints_to_normalized_deltas(
                    noisy_astar_local_cond[..., :2].to(device),
                    ACTION_STATS["min"],
                    ACTION_STATS["max"],
                )
                diffusion_loss = bridge_diffusion.training_loss(
                    model,
                    naction,
                    astar_prior,
                    obsfloorplan_cond,
                    local_cond=noisy_astar_local_cond.to(device),
                )
            else:
                noise = torch.randn(naction.shape, device=device)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (B,), device=device
                ).long()
                noisy_action = noise_scheduler.add_noise(naction, noise, timesteps)
                noise_pred = model(
                    "noise_pred_net",
                    sample=noisy_action,
                    timestep=timesteps,
                    local_cond=noisy_astar_local_cond.to(device),
                    global_cond=obsfloorplan_cond,
                )
                diffusion_loss = action_reduce(F.mse_loss(noise_pred, noise, reduction="none"))
            
            # Total loss
            loss = alpha * dist_loss + (1 - alpha) * diffusion_loss

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema(ema_model.averaged_model, model)

            # Logging
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            
            log_dict = {
                "total_loss": loss_cpu,
                "diffusion_loss": diffusion_loss.item(),
                "dist_loss": dist_loss.item(),
                "path_noise_bound_m": path_noise_bound_m,
            }
            unwrapped_model = model.module if hasattr(model, "module") else model
            vision_encoder = unwrapped_model.vision_encoder
            if hasattr(vision_encoder, "last_shallow_weight"):
                residual_alpha = vision_encoder.last_shallow_weight.float()
                turn_risk = vision_encoder.last_turn_risk.float()
                alpha_max = vision_encoder.curvature_visual_fusion.alpha_max
                log_dict.update({
                    "visual_fusion/shallow_residual_alpha_mean": float(residual_alpha.mean().item()),
                    "visual_fusion/shallow_residual_alpha_p10": float(torch.quantile(residual_alpha, 0.10).item()),
                    "visual_fusion/shallow_residual_alpha_p90": float(torch.quantile(residual_alpha, 0.90).item()),
                    "visual_fusion/shallow_residual_saturation_rate": float(
                        (residual_alpha >= 0.99 * alpha_max).float().mean().item()
                    ),
                    "visual_fusion/turn_risk_mean": float(turn_risk.mean().item()),
                    "visual_fusion/turn_risk_p90": float(torch.quantile(turn_risk, 0.90).item()),
                })
            losses = None

            if i % print_log_freq == 0:
                losses = _compute_losses_flona(
                            model,
                            noise_scheduler,
                            batch_obs_images,
                            batch_floorplan_images,
                            distance.to(device),
                            actions.to(device),
                            goal_pos,
                            noisy_curr_pos,
                            noisy_curr_ori,
                            noisy_astar_local_cond,
                            device,
                            collision_dataset=collision_dataset,
                            scene_names=list(scene_names),
                            bridge_diffusion=bridge_diffusion,
                            astar_global_cond=model_astar_local_cond.to(device),
                        )
                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(float(value))
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    log_dict.update(data_log)

            if use_wandb:
                if losses is not None:
                    log_dict["train/collision_rate"] = float(losses["collision_rate"])
                wandb.log(log_dict)
            
            if image_log_freq != 0 and i % image_log_freq == 0:
                if losses is None:
                    losses = _compute_losses_flona(
                        model,
                        noise_scheduler,
                        batch_obs_images,
                        batch_floorplan_images,
                        distance.to(device),
                        actions.to(device),
                        goal_pos,
                        noisy_curr_pos,
                        noisy_curr_ori,
                        noisy_astar_local_cond,
                        device,
                        collision_dataset=collision_dataset,
                        scene_names=list(scene_names),
                        bridge_diffusion=bridge_diffusion,
                        astar_global_cond=model_astar_local_cond.to(device),
                    )
                visualize_diffusion_action_distribution(
                    model,
                    noise_scheduler,
                    batch_obs_images,
                    batch_floorplan_images,
                    batch_viz_obs_images,
                    batch_viz_floorplan_images,
                    actions,
                    distance,
                    goal_pos,
                    noisy_curr_pos,
                    noisy_curr_ori,
                    goal_pos_local,
                    goal_pos_resized,
                    curr_pos_resized,
                    curr_ori_resized,
                    noisy_astar_local_cond,
                    device,
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,
                    use_wandb,
                    bridge_diffusion=bridge_diffusion,
                    batch_astar_global_cond=model_astar_local_cond.to(device),
                )
                if losses["collision_debug"] is not None:
                    visualize_collision_prediction(
                        losses["collision_debug"],
                        "train",
                        project_folder,
                        epoch,
                        i,
                        use_wandb,
                    )

def evaluate_flona(
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    project_folder: str,
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,
    perturb_config: Optional[dict] = None,
    bridge_diffusion: Optional[NaviBridgeDenoiser] = None,
):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        ema_model (nn.Module): exponential moving average version of model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        noise_scheduler: noise scheduler to evaluate with 
        project_folder (string): path to project folder
        epoch (int): current epoch
        print_log_freq (int): how often to print logs 
        wandb_log_freq (int): how often to log to wandb
        image_log_freq (int): how often to log images
        num_images_log (int): number of images to log
        eval_fraction (float): fraction of data to use for evaluation
        use_wandb (bool): whether to use wandb for logging
    """
    model.eval()
    
    num_batches = len(dataloader)

    action_loss_logger = Logger("action_loss", "test", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger(
        "action_waypts_cos_sim", "test", window_size=print_log_freq
    )
    multi_action_waypts_cos_sim_logger = Logger(
        "multi_action_waypts_cos_sim", "test", window_size=print_log_freq
    )
    collision_rate_logger = Logger("collision_rate", "test", window_size=print_log_freq)
    # dist_loss_logger = Logger("dist_loss", "test", window_size=print_log_freq)
    loggers = {
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
        "collision_rate": collision_rate_logger,
        # "dist_loss": dist_loss_logger,
    }
    num_batches = max(int(num_batches * eval_fraction), 1)
    collision_dataset = _unwrap_dataset(dataloader.dataset)

    total_diffusion_loss = 0.0
    total_action_loss = {
        "action_loss": 0.0,
        "action_waypts_cos_sim": 0.0,
        "multi_action_waypts_cos_sim": 0.0,
        # "dist_loss": 0.0
    }
    total_collision_hits = 0
    total_collision_total = 0
    with torch.no_grad():
        with tqdm.tqdm(
            itertools.islice(dataloader, num_batches),        # only evaluate on a subset of the data
            total=num_batches,
            dynamic_ncols=True,
            desc=f"Evaluating for epoch {epoch}",
            leave=False) as tepoch:
            for i, data in enumerate(tepoch):
                (
                    obs_image,
                    floorplan_image,
                    actions,    # (b, predict_len, 2)      local coordinate    waypoints(steps) metric
                    distance,   # (b,)    step from current to goal
                    goal_pos,   # (b, 2)
                    curr_pos,   # (b, 2)
                    curr_ori,   # (b, 2)      global coordinate   waypoints metric
                    goal_pos_resized, # (b, 2)  goal_pos_resized
                    curr_pos_resized, # (b, 2)  curr_pos_resized
                    curr_ori_resized,  # (b, 2)  curr_ori_resized
                    goal_pos_local, # (b, 2)  goal_pos_local
                    curr_pos_local,  # (b, 2)  curr_pos_local
                    astar_local_cond,
                    scene_names,
                ) = data

                obs_images = torch.split(obs_image, 3, dim=1)   # tuple, containing L*(b,3,h,w) tensors
                batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])   # (height, width)
                batch_viz_floorplan_images = TF.resize(floorplan_image, VISUALIZATION_IMAGE_SIZE[::-1])
                batch_obs_images = [transform(obs) for obs in obs_images]
                batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)  # (b,3*L,h,w)
                batch_floorplan_images = transform(floorplan_image).to(device)  # (b,3,h,w)

                # Plot floorplan markers using floorplan image scaling, not observation scaling.
                fh0, fw0 = floorplan_image.shape[-2:]
                fh1, fw1 = batch_viz_floorplan_images.shape[-2:]
                floorplan_scale = torch.tensor([fw1 / fw0, fh1 / fh0], dtype=torch.float32)
                goal_pos_resized = goal_pos_resized * floorplan_scale
                curr_pos_resized = curr_pos_resized * floorplan_scale
                curr_ori_resized = curr_ori_resized * floorplan_scale
                for b in range(goal_pos_resized.shape[0]):
                    goal_x, goal_y = _clamp_marker_xy(goal_pos_resized[b], fw1, fh1)
                    curr_x, curr_y = _clamp_marker_xy(curr_pos_resized[b], fw1, fh1)
                    ori_x, ori_y = _clamp_marker_xy(curr_ori_resized[b], fw1, fh1)
                    batch_viz_floorplan_images[b, :, goal_y, goal_x] = torch.tensor(RED)
                    batch_viz_floorplan_images[b, :, curr_y, curr_x] = torch.tensor(GREEN)
                    batch_viz_floorplan_images[b, :, ori_y, ori_x] = torch.tensor(CYAN)

                B = actions.shape[0]
                # Match training and closed-loop inference for DDPM as well.
                model_astar_local_cond = _prepare_astar_local_condition_for_model(
                    astar_local_cond.to(device), perturb_config
                )
                obsfloorplan_cond = model("vision_encoder",
                                          obs_img=batch_obs_images,
                                          floorplan_img=batch_floorplan_images,
                                          obs_pos = curr_pos,
                                          goal_pos = goal_pos,
                                          obs_ori = curr_ori,
                                          astar_global_cond=model_astar_local_cond,
                                          )
                if image_log_freq != 0 and i % image_log_freq == 0:
                    visualize_shallow_feature_fusion(
                        model.vision_encoder,
                        batch_viz_obs_images,
                        project_folder,
                        epoch,
                        num_images_log,
                        split="evaluate",
                    )
                obsfloorplan_cond = obsfloorplan_cond.flatten(start_dim=1)
                distance = distance.to(device)
                deltas = get_delta(actions)
                ndeltas = normalize_data(deltas, ACTION_STATS)
                naction = from_numpy(ndeltas).to(device)   # seem like direction of action
                assert naction.shape[-1] == 2, "action dim must be 2"

                if bridge_diffusion is not None:
                    astar_prior = waypoints_to_normalized_deltas(
                        model_astar_local_cond[..., :2],
                        ACTION_STATS["min"],
                        ACTION_STATS["max"],
                    )
                    loss = bridge_diffusion.training_loss(
                        model,
                        naction,
                        astar_prior,
                        obsfloorplan_cond,
                        local_cond=model_astar_local_cond,
                    )
                else:
                    noise = torch.randn(naction.shape, device=device)
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (B,), device=device
                    ).long()
                    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
                    noise_pred = model(
                        "noise_pred_net",
                        sample=noisy_actions,
                        timestep=timesteps,
                        local_cond=model_astar_local_cond,
                        global_cond=obsfloorplan_cond,
                    )
                    loss = nn.functional.mse_loss(noise_pred, noise)

                # Logging
                loss_cpu = loss.item()
                total_diffusion_loss += loss_cpu
                tepoch.set_postfix(loss=loss_cpu)

                # if use_wandb:
                #    wandb.log({"diffusion_eval_loss": loss})
                # else:
                #    print("diffusion_eval_loss", loss)

                losses = _compute_losses_flona(
                                model,
                                noise_scheduler,
                                batch_obs_images,
                                batch_floorplan_images,
                                distance.to(device),
                                actions.to(device),
                                goal_pos,
                                curr_pos,
                                curr_ori,
                                model_astar_local_cond,
                                device,
                                collision_dataset=collision_dataset,
                                scene_names=list(scene_names),
                                bridge_diffusion=bridge_diffusion,
                            )

                total_action_loss["action_loss"] += losses["action_loss"].item()
                total_action_loss["action_waypts_cos_sim"] += losses["action_waypts_cos_sim"].item()
                total_action_loss["multi_action_waypts_cos_sim"] += losses["multi_action_waypts_cos_sim"].item()
                total_collision_hits += int(losses["collision_hits"])
                total_collision_total += int(losses["collision_total"])
                # total_action_loss["dist_loss"] += losses["dist_loss"].item()

                if image_log_freq != 0 and i % image_log_freq == 0:
                    visualize_diffusion_action_distribution(
                        model,
                        noise_scheduler,
                        batch_obs_images,
                        batch_floorplan_images,
                        batch_viz_obs_images,
                        batch_viz_floorplan_images,
                        actions,
                        distance,
                        goal_pos,
                        curr_pos,
                        curr_ori,
                        goal_pos_local,
                        goal_pos_resized,
                        curr_pos_resized,
                        curr_ori_resized,
                        model_astar_local_cond,
                        device,
                        "evaluate",
                        project_folder,
                        epoch,
                        num_images_log,
                        30,
                        use_wandb,
                        bridge_diffusion=bridge_diffusion,
                    )
                    if losses["collision_debug"] is not None:
                        visualize_collision_prediction(
                            losses["collision_debug"],
                            "evaluate",
                            project_folder,
                            epoch,
                            i,
                            use_wandb,
                        )

    for key, value in total_action_loss.items():
        if key in loggers:
            logger = loggers[key]
            logger.log_data(value / num_batches)
    collision_rate = float(total_collision_hits) / float(total_collision_total) if total_collision_total > 0 else float("nan")
    collision_rate_logger.log_data(collision_rate)

    data_log = {}
    for key, logger in loggers.items():
        data_log[logger.full_name()] = logger.latest()
        print(f"(epoch {epoch}) {logger.display()}")
    
    # Add aggregated diffusion loss
    avg_diffusion_loss = total_diffusion_loss / num_batches
    data_log["test/diffusion_loss"] = avg_diffusion_loss

    if use_wandb:
        wandb.log(data_log, commit=True)
    
    print(f"Epoch {epoch} Average Diffusion Eval Loss: {avg_diffusion_loss}")
    
def execute_model(
    model: nn.Module,
    cur_pos: np.ndarray,        # np.array (1,2)   
    cur_heading: np.ndarray,    # np.array (1,2)
    # cur_pos_f3: np.ndarray, # np.array (1,3)      
    # cur_heading_f3: np.ndarray, # np.array (1,3)  
    goal_pos: np.ndarray,       # np.array (1,2)
    astar_local_cond: np.ndarray,
    # img_paths: List[str],       # list of image paths
    # floorplan_path: str,        # floorplan path
    cur_obs: torch.Tensor,        # torch.tensor(L,3,h,w)
    floorplan: torch.Tensor,      # torch.tensor (1,3,h,w)
    metric_waipoint_spacing: float,
    waypoint_spacing: float,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    floorplan_ary: np.ndarray,
    planner_metric_size_cm: Optional[float] = None,
    log_add: str = None,
    bridge_diffusion: Optional[NaviBridgeDenoiser] = None,
    pred_horizon: int = 32,
    num_trajectory_samples: int = 30,
    execute_horizon: int = 20,
    energy_guidance=None,
):
    """
    Execute the model on the given data.
    Args:
        ema_model: exponential moving average model
        cur_pos: current position
        goal_pos: goal position
        # img_paths: list of image paths
        # floorplan_path: floorplan path
        cur_obs: current observation
        floorplan: floorplan
        transform: transform to apply to images
        device: device to use for evaluation
        noise_scheduler: noise scheduler to evaluate with 
        log_folder: folder to save images to
    """
    model.eval()
    
    cur_pos = torch.as_tensor(cur_pos, dtype=torch.float32)
    cur_heading = torch.as_tensor(cur_heading, dtype=torch.float32)
    goal_pos = torch.as_tensor(goal_pos, dtype=torch.float32)
    astar_local_cond = torch.as_tensor(astar_local_cond, dtype=torch.float32)
    if astar_local_cond.ndim == 2:
        astar_local_cond = astar_local_cond.unsqueeze(0)
    cur_obs = torch.as_tensor(cur_obs, dtype=torch.float32)
    floorplan = torch.as_tensor(floorplan, dtype=torch.float32)
    cur_pos /= metric_waipoint_spacing * waypoint_spacing
    cur_heading /= metric_waipoint_spacing * waypoint_spacing
    goal_pos /= metric_waipoint_spacing * waypoint_spacing
    astar_local_cond = _normalize_astar_local_condition(
        astar_local_cond, metric_waipoint_spacing * waypoint_spacing
    )
    cur_obss = torch.split(cur_obs, 1, dim=0)
    batch_cur_obss = [transform(obs) for obs in cur_obss]
    batch_cur_obss = torch.cat(batch_cur_obss, dim=1).to(device)  # (1,3*L,h,w)
    batch_floorplan = transform(floorplan).to(device)  # (1,3,h,w)

    
    if pred_horizon < 1:
        raise ValueError("pred_horizon must be at least 1")
    if num_trajectory_samples < 1:
        raise ValueError("num_trajectory_samples must be at least 1")
    if not 1 <= execute_horizon <= pred_horizon:
        raise ValueError("execute_horizon must be between 1 and pred_horizon")

    model_output_dict = model_output(
        model,
        noise_scheduler,
        batch_cur_obss,
        batch_floorplan,
        pred_horizon,
        2,
        num_trajectory_samples,
        goal_pos,
        cur_pos,
        cur_heading,
        astar_local_cond,
        device=device,
        bridge_diffusion=bridge_diffusion,
        energy_guidance=energy_guidance,
    )
    actions = model_output_dict['actions'].mean(dim=0)  # [1,8,2]
    

    distance = model_output_dict['distance']

    actions_normed_global = to_global_coords(to_numpy(actions), to_numpy(cur_pos).squeeze(0), to_numpy(cur_heading).squeeze(0))
    actions_meter_global = actions_normed_global * metric_waipoint_spacing * waypoint_spacing

    if log_add is not None:
        save_action = actions.cpu().detach().numpy()
        gs = gridspec.GridSpec(6, 6)
        gs.update(wspace = 0.9, hspace = 0.7)
        ax1 = plt.subplot(gs[:2, :2])
        ax2 = plt.subplot(gs[:2, 2:])
        ax3 = plt.subplot(gs[2:, :3])
        ax4 = plt.subplot(gs[2:, 3:])
        
        goal_pos_metric = goal_pos * metric_waipoint_spacing * waypoint_spacing
        floor_width = floorplan_ary.shape[0]
        cur_pos_metric = to_numpy(cur_pos)[0] * metric_waipoint_spacing * waypoint_spacing
        image_size = (floorplan_ary.shape[1], floorplan_ary.shape[0])

        def metric_to_map_xy(metric_xy):
            metric_xy = np.asarray(metric_xy, dtype=np.float32)
            if planner_metric_size_cm is None:
                return np.flip((metric_xy / 0.01 + floor_width / 2.0)).astype(int)
            return np.flip(
                np.round(
                    metric_positions_to_resized_pixels(metric_xy, planner_metric_size_cm, image_size)
                ).astype(int)
            )

        def in_bounds(map_xy):
            return 0 <= map_xy[0] < floorplan_ary.shape[0] and 0 <= map_xy[1] < floorplan_ary.shape[1]

        if planner_metric_size_cm is None:
            end_xy = metric_to_map_xy(np.array(goal_pos_metric[0]))
            start_xy = metric_to_map_xy(cur_pos_metric)
            planner_global_path = to_global_coords(
                to_numpy(astar_local_cond)[0, :, :2],
                to_numpy(cur_pos).squeeze(0),
                to_numpy(cur_heading).squeeze(0),
            ) * metric_waipoint_spacing * waypoint_spacing
            planner_path_xy = np.flip((planner_global_path / 0.01 + floor_width / 2.0), axis=1).astype(int)
        else:
            end_xy = metric_to_map_xy(np.array(goal_pos_metric[0]))
            start_xy = metric_to_map_xy(cur_pos_metric)
            planner_global_path = to_global_coords(
                to_numpy(astar_local_cond)[0, :, :2],
                to_numpy(cur_pos).squeeze(0),
                to_numpy(cur_heading).squeeze(0),
            ) * metric_waipoint_spacing * waypoint_spacing
            planner_path_xy = np.flip(np.round(metric_positions_to_resized_pixels(planner_global_path, planner_metric_size_cm, image_size)).astype(int), axis=1)
        
        # Draw Goal (Blue square) - Existing logic
        floorplan_ary[max(0, end_xy[0]-5) : min(end_xy[0]+5, floorplan_ary.shape[0]), max(0, end_xy[1]-5) : min(end_xy[1]+5, floorplan_ary.shape[1]), :] = np.array([0, 0, 255, 255])
        
        # Draw Start Position Marker (Blue circle/square) - NEW
        # cur_pos is tensor (1,2) normalized. Convert to metric then pixel.
        cur_xy = metric_to_map_xy(cur_pos_metric)
        if in_bounds(cur_xy):
             # Draw a blue marker (e.g., 4 pixel radius)
             marker_color = np.array([0, 0, 255, 255])
             r = 4
             floorplan_ary[max(0, cur_xy[0]-r) : min(cur_xy[0]+r, floorplan_ary.shape[0]), max(0, cur_xy[1]-r) : min(cur_xy[1]+r, floorplan_ary.shape[1]), :] = marker_color

        for planner_xy in planner_path_xy:
            py, px = int(planner_xy[0]), int(planner_xy[1])
            if 0 <= px < floorplan_ary.shape[1] and 0 <= py < floorplan_ary.shape[0]:
                floorplan_ary[max(0, py-1):min(py+2, floorplan_ary.shape[0]), max(0, px-1):min(px+2, floorplan_ary.shape[1]), :] = np.array([255, 165, 0, 255])

        ax1.imshow(cur_obs[-1].permute(1,2,0).cpu().detach().numpy())
        ax2.plot(save_action[:,0], save_action[:,1], marker = '.')
        
        for i, xy in enumerate(actions_meter_global):
            map_xy = metric_to_map_xy(xy)
            if i == 0:
                start_xy = np.clip(
                    map_xy,
                    [0, 0],
                    [floorplan_ary.shape[0] - 1, floorplan_ary.shape[1] - 1],
                )
            if not in_bounds(map_xy):
                continue
            if i < execute_horizon:
                color = np.array([0, 128, 0, 255]) # Dark green for the executed prefix
                floorplan_ary[map_xy[0]-2 : map_xy[0]+2, map_xy[1]-2 : map_xy[1]+2, :] = color
            else:
                color = np.array([0, 200, 0, 255]) # Slightly lighter green for the remaining predicted suffix
                floorplan_ary[map_xy[0]-2 : map_xy[0]+2, map_xy[1]-2 : map_xy[1]+2, :] = color

        start_xy = np.clip(
            start_xy,
            [0, 0],
            [floorplan_ary.shape[0] - 1, floorplan_ary.shape[1] - 1],
        )
        local_view = floorplan_ary[
            max(0, start_xy[0]-200) : min(start_xy[0]+200, floorplan_ary.shape[0]),
            max(0, start_xy[1]-200) : min(start_xy[1]+200, floorplan_ary.shape[1]),
            :
        ]
        if local_view.size == 0:
            local_view = floorplan_ary
        ax3.imshow(local_view)
        ax4.imshow(floorplan_ary)

        plt.savefig(os.path.join(log_add))
    
    return actions_meter_global    
    
# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def get_delta(actions):       # (0,0)->first action point, first action point->second action point, ...
    # append zeros to first action
    ex_actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[-1])), actions], axis=1)
    delta = ex_actions[:,1:] - ex_actions[:,:-1]
    return delta

def get_action(diffusion_output, action_stats=ACTION_STATS):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    device = diffusion_output.device
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return from_numpy(actions).to(device)

def model_output(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_floorplan_images: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    goal_pos: torch.Tensor,
    curr_pos: torch.Tensor,
    curr_ori: torch.Tensor,
    astar_local_cond: torch.Tensor,
    device: torch.device,
    bridge_diffusion: Optional[NaviBridgeDenoiser] = None,
    astar_global_cond: Optional[torch.Tensor] = None,
    energy_guidance=None,
):

    astar_local_cond = astar_local_cond.to(device)
    if astar_global_cond is None:
        astar_global_cond = astar_local_cond
    else:
        astar_global_cond = astar_global_cond.to(device)
    obsfloorplan_cond_fused = model("vision_encoder",
                                    obs_img=batch_obs_images,
                                    floorplan_img=batch_floorplan_images,
                                    obs_pos=curr_pos,
                                    goal_pos=goal_pos,
                                    obs_ori = curr_ori,
                                    astar_global_cond=astar_global_cond,
                                    )  #==============modified by weiqi========
    obsfloorplan_cond_fused = obsfloorplan_cond_fused.repeat_interleave(num_samples, dim=0)
    astar_local_cond = astar_local_cond.repeat_interleave(num_samples, dim=0)

    if bridge_diffusion is not None:
        astar_prior = waypoints_to_normalized_deltas(
            astar_local_cond[..., :2], ACTION_STATS["min"], ACTION_STATS["max"]
        )
        diffusion_output = bridge_diffusion.sample(
            model,
            astar_prior,
            obsfloorplan_cond_fused,
            local_cond=astar_local_cond,
            energy_guidance=energy_guidance,
        )
    else:
        if energy_guidance is not None:
            raise ValueError("RGB depth guidance requires NaviBridge, not DDPM")
        diffusion_output = torch.randn(
            (len(obsfloorplan_cond_fused), pred_horizon, action_dim), device=device)
        for k in noise_scheduler.timesteps[:]:
            noise_pred = model(
                "noise_pred_net",
                sample=diffusion_output,
                timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
                local_cond=astar_local_cond,
                global_cond=obsfloorplan_cond_fused,
            )
            diffusion_output = noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=diffusion_output,
            ).prev_sample

    actions = get_action(diffusion_output, ACTION_STATS)
    distance = model("dist_pred_net", obsfloorplan_cond=obsfloorplan_cond_fused)
    # pos_ori = model("pos_ori_pred_net", obsgoal_cond=obsgoal_cond)

    return {
        'actions': actions,
        'distance': distance,
        # 'pos_ori': pos_ori
    }

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()

def from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()


def visualize_shallow_feature_fusion(
    vision_encoder: nn.Module,
    batch_viz_obs_images: torch.Tensor,
    project_folder: str,
    epoch: int,
    num_images_log: int,
    split: str = "evaluate",
) -> None:
    """Save fixed-sample shallow-attention evidence for fusion ablations."""
    required = (
        "last_shallow_weight",
        "last_turn_risk",
        "last_shallow_spatial_attention",
        "last_effective_residual_ratio",
    )
    if not all(hasattr(vision_encoder, name) for name in required):
        return

    alphas = vision_encoder.last_shallow_weight.detach().float().cpu()
    risks = vision_encoder.last_turn_risk.detach().float().cpu()
    attentions = vision_encoder.last_shallow_spatial_attention.detach().float().cpu()
    ratios = vision_encoder.last_effective_residual_ratio.detach().float().cpu()
    if attentions.ndim != 2:
        return
    grid_size = int(round(math.sqrt(attentions.shape[1])))
    if grid_size * grid_size != attentions.shape[1]:
        return

    output_dir = os.path.join(
        project_folder, "visualize", split, f"epoch{epoch}", "shallow_feature_fusion"
    )
    os.makedirs(output_dir, exist_ok=True)
    count = min(int(num_images_log), batch_viz_obs_images.shape[0], attentions.shape[0])
    for sample_idx in range(count):
        image = batch_viz_obs_images[sample_idx].detach().float().cpu()
        image = np.moveaxis(image.numpy(), 0, -1)
        image = np.clip(image, 0.0, 1.0)
        heatmap = attentions[sample_idx].reshape(1, 1, grid_size, grid_size)
        heatmap = F.interpolate(
            heatmap,
            size=image.shape[:2],
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        heat_min, heat_max = float(heatmap.min()), float(heatmap.max())
        heatmap = (heatmap - heat_min) / max(heat_max - heat_min, 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)
        axes[0].imshow(image)
        axes[0].set_title("Current RGB")
        axes[1].imshow(heatmap, cmap="magma", vmin=0.0, vmax=1.0)
        axes[1].set_title("Shallow spatial attention")
        axes[2].imshow(image)
        axes[2].imshow(heatmap, cmap="magma", alpha=0.48, vmin=0.0, vmax=1.0)
        axes[2].set_title(
            f"Overlay\nalpha={alphas[sample_idx]:.4f}, "
            f"turn risk={risks[sample_idx]:.3f}, "
            f"residual/deep={ratios[sample_idx]:.4f}"
        )
        for axis in axes:
            axis.axis("off")
        fig.savefig(
            os.path.join(output_dir, f"sample_{sample_idx}.png"),
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)

def visualize_diffusion_action_distribution(
    ema_model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_floorplan_images: torch.Tensor,
    batch_viz_obs_images: torch.Tensor,
    batch_viz_floorplan_images: torch.Tensor,
    batch_action_label: torch.Tensor,
    batch_distance_labels: torch.Tensor,
    batch_goal_pos: torch.Tensor,
    batch_curr_pos: torch.Tensor,
    batch_curr_ori: torch.Tensor,
    batch_goal_pos_local: torch.Tensor,
    batch_goal_pos_resized: torch.Tensor,
    batch_curr_pos_resized: torch.Tensor,
    batch_curr_ori_resized: torch.Tensor,
    batch_astar_local_cond: torch.Tensor,
    device: torch.device,
    type: str,
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,
    use_wandb: bool = True,
    bridge_diffusion: Optional[NaviBridgeDenoiser] = None,
    batch_astar_global_cond: Optional[torch.Tensor] = None,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    max_batch_size = batch_obs_images.shape[0]
    num_images_log = min(num_images_log, batch_obs_images.shape[0], batch_floorplan_images.shape[0], batch_action_label.shape[0], batch_goal_pos.shape[0])
    batch_obs_images = batch_obs_images[:num_images_log]
    batch_floorplan_images = batch_floorplan_images[:num_images_log]
    batch_action_label = batch_action_label[:num_images_log]
    batch_goal_pos = batch_goal_pos[:num_images_log]
    batch_curr_pos = batch_curr_pos[:num_images_log]
    batch_curr_ori = batch_curr_ori[:num_images_log]
    batch_goal_pos_local = batch_goal_pos_local[:num_images_log]
    batch_goal_pos_resized = batch_goal_pos_resized[:num_images_log]
    batch_curr_pos_resized = batch_curr_pos_resized[:num_images_log]
    batch_curr_ori_resized = batch_curr_ori_resized[:num_images_log]
    batch_astar_local_cond = batch_astar_local_cond[:num_images_log]
    if batch_astar_global_cond is not None:
        batch_astar_global_cond = batch_astar_global_cond[:num_images_log]
    wandb_list = []

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    # split into batches
    batch_obs_images_list = torch.split(batch_obs_images, max_batch_size, dim=0)
    batch_floorplan_images_list = torch.split(batch_floorplan_images, max_batch_size, dim=0)
    actions_list = []
    distances_list = []

    for obs, floorplan in zip(batch_obs_images_list, batch_floorplan_images_list):
        model_output_dict = model_output(
            ema_model,
            noise_scheduler,
            obs,
            floorplan,
            pred_horizon,
            action_dim,
            num_samples,
            batch_goal_pos,
            batch_curr_pos,
            batch_curr_ori,
            batch_astar_local_cond,
            device,
            bridge_diffusion=bridge_diffusion,
            astar_global_cond=batch_astar_global_cond,
        )
        actions_list.append(to_numpy(model_output_dict['actions'])) # local, waypoints metric
        distances_list.append(to_numpy(model_output_dict['distance']))

    # concatenate
    actions_list = np.concatenate(actions_list, axis=0)
    distances_list = np.concatenate(distances_list, axis=0)

    # split into actions per observation
    actions_list = np.split(actions_list, num_images_log, axis=0)
    distances_list = np.split(distances_list, num_images_log, axis=0)
    distances_avg = [np.mean(dist) for dist in distances_list]
    distances_std = [np.std(dist) for dist in distances_list]
    assert len(actions_list) == len(actions_list) == num_images_log
    np_distance_labels = to_numpy(batch_distance_labels)
    for i in range(num_images_log):
        fig, ax = plt.subplots(1, 3)
        actions = actions_list[i]
        action_label = to_numpy(batch_action_label[i])
        astar_local_cond = to_numpy(batch_astar_local_cond[i])
        traj_list = np.concatenate([
            actions,
            astar_local_cond[None, :, :2],
            action_label[None],
        ], axis=0)
        traj_colors = ["red"] * len(actions) + [ORANGE] + ["magenta"]
        traj_alphas = [0.1] * len(actions) + [1.0, 1.0]

        # make points numpy array of robot positions (0, 0) and goal positions
        point_list = [np.array([0, 0]), to_numpy(batch_goal_pos_local[i])]
        point_colors = ["green", "red"]
        point_alphas = [1.0, 1.0]
        plot_trajs_and_points(
            ax[0],
            traj_list,
            point_list,
            traj_colors,
            point_colors,
            traj_labels=None,
            point_labels=None,
            traj_alphas=traj_alphas,
            point_alphas=point_alphas,
            default_coloring=True, 
        )
        
        obs_image = to_numpy(batch_viz_obs_images[i])
        floorplan_image = to_numpy(batch_viz_floorplan_images[i])
        # move channel to last dimension
        obs_image = np.moveaxis(obs_image, 0, -1)
        floorplan_image = np.moveaxis(floorplan_image, 0, -1)
        ax[1].imshow(obs_image)
        ax[2].imshow(floorplan_image)

        # set title
        ax[0].set_title(f"diffusion predictions + A* local cond")
        ax[1].set_title(f"observation")
        ax[2].set_title(f"goal: label={np_distance_labels[i]} gc_dist={distances_avg[i]:.2f}±{distances_std[i]:.2f}")
        str_text = f'goal_resized:{batch_goal_pos_resized[i].cpu().numpy()} curr_pos_resized:{batch_curr_pos_resized[i].cpu().numpy()} curr_ori_resized:{batch_curr_ori_resized[i].cpu().numpy()}'
        fig.text(0, 0, str_text)
        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)
        save_path = os.path.join(visualize_path, f"sample_{i}.png")
        plt.savefig(save_path)
        if use_wandb:
            wandb_list.append(wandb.Image(save_path))
        plt.close(fig)
    if len(wandb_list) > 0 and use_wandb:
        wandb.log({f"{type}_action_samples": wandb_list}, commit=False)

def plot_trajs_and_points(
    ax: plt.Axes,
    list_trajs: list,
    list_points: list,
    traj_colors: list = [CYAN, MAGENTA],
    point_colors: list = [RED, GREEN],
    traj_labels: Optional[list] = ["prediction", "ground truth"],
    point_labels: Optional[list] = ["robot", "goal"],
    traj_alphas: Optional[list] = None,
    point_alphas: Optional[list] = None,
    default_coloring: bool = True,
):
    """
    Plot trajectories and points that could potentially have a yaw.

    Args:
        ax: matplotlib axis
        list_trajs: list of trajectories, each trajectory is a numpy array of shape (horizon, 2) 
        list_points: list of points, each point is a numpy array of shape (2,)
        traj_colors: list of colors for trajectories
        point_colors: list of colors for points
        traj_labels: list of labels for trajectories
        point_labels: list of labels for points
        traj_alphas: list of alphas for trajectories
        point_alphas: list of alphas for points
    """
    assert (
        len(list_trajs) <= len(traj_colors) or default_coloring
    ), "Not enough colors for trajectories"
    assert len(list_points) <= len(point_colors), "Not enough colors for points"
    assert (
        traj_labels is None or len(list_trajs) == len(traj_labels) or default_coloring
    ), "Not enough labels for trajectories"
    assert point_labels is None or len(list_points) == len(point_labels), "Not enough labels for points"

    for i, traj in enumerate(list_trajs):
        if traj_labels is None:
            ax.plot(
                traj[:, 0], 
                traj[:, 1], 
                color=traj_colors[i],
                alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
                marker="o",
            )
        else:
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=traj_colors[i],
                label=traj_labels[i],
                alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
                marker="o",
            )
    for i, pt in enumerate(list_points):
        if point_labels is None:
            ax.plot(
                pt[0], 
                pt[1], 
                color=point_colors[i], 
                alpha=point_alphas[i] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0
            )
        else:
            ax.plot(
                pt[0],
                pt[1],
                color=point_colors[i],
                alpha=point_alphas[i] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0,
                label=point_labels[i],
            )
    # put the legend below the plot
    if traj_labels is not None or point_labels is not None:
        ax.legend()
        ax.legend(bbox_to_anchor=(0.0, -0.5), loc="upper left", ncol=2)
    ax.set_aspect("equal", "box")

# def model_output_bak(
#     model: nn.Module,
#     noise_scheduler: DDPMScheduler,
#     depth_cond: torch.Tensor,
#     shortest_actions: torch.Tensor, 
#     pred_horizon: int,
#     action_dim: int,
#     device: torch.device,
# ):
#     noisy_diffusion_output = torch.randn(
#         (depth_cond.shape[0], pred_horizon, action_dim)).to("cuda:0")
#     diffusion_output = noisy_diffusion_output
#     print("shape of shortest_actions:", shortest_actions.shape) 
#     for k in noise_scheduler.timesteps[:]:
#         # predict noise
#         noise_pred = model(
#             "noise_pred_net",
#             sample=diffusion_output,
#             timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to("cuda:0"),
#             global_cond=depth_cond,
#             local_cond = shortest_actions
#         )

#         # inverse diffusion step (remove noise)
#         diffusion_output = noise_scheduler.step(
#             model_output=noise_pred,
#             timestep=k,
#             sample=diffusion_output
#         ).prev_sample

#     actions = get_action(diffusion_output, ACTION_STATS)  

#     return {
#         'actions': actions
#     }
    

# def execute_model(
#     model: nn.Module,
#     cur_obs: torch.Tensor,  
#     cur_pos: torch.Tensor,
#     cur_heading: torch.Tensor,
#     goal_pos: torch.Tensor,
#     shortest_actions: torch.Tensor,
#     depth_latent: torch.Tensor, 
#     metric_waipoint_spacing: float,
#     waypoint_spacing: float,
#     device: torch.device,
#     noise_scheduler: DDPMScheduler,
#     floorplan_ary: np.ndarray,  
#     log_add: str = None,
# ):
#     """
#     Execute the model on the given data.
#     """
#     model.eval()

#     depth_cond = model("depth_latent_processor", depth_latent=depth_latent)
#     depth_cond = depth_cond.repeat_interleave(30, dim=0)
#     shortest_actions = shortest_actions.repeat_interleave(30, dim=0)
#     pred_horizon, action_dim = shortest_actions.shape[1], shortest_actions.shape[2]

#     print("shape of depth_cond:", depth_cond.shape)
#     model_output_dict = model_output(
#         model,
#         noise_scheduler,
#         depth_cond,
#         shortest_actions,
#         pred_horizon,
#         action_dim,
#         device=device,
#     )
#     actions = model_output_dict['actions'].mean(dim=0)  # [1,8,2]
#     cur_pos = torch.from_numpy(cur_pos).to(device) 
#     cur_heading = torch.from_numpy(cur_heading).to(device)
#     actions = actions * metric_waipoint_spacing * waypoint_spacing
#     actions_meter_global = to_global_coords(to_numpy(actions), to_numpy(cur_pos).squeeze(0), to_numpy(cur_heading).squeeze(0))
    
    
    
#     if log_add is not None:
#         save_action = actions.cpu().detach().numpy()
#         gs = gridspec.GridSpec(6, 6)
#         gs.update(wspace = 0.9, hspace = 0.7)
#         ax1 = plt.subplot(gs[:2, :2])
#         ax2 = plt.subplot(gs[:2, 2:])
#         ax3 = plt.subplot(gs[2:, :3])
#         ax4 = plt.subplot(gs[2:, 3:])
        
#         goal_pos_metric = goal_pos 
#         floor_width = floorplan_ary.shape[0]
#         end_xy = np.flip((np.array(goal_pos_metric[0]) / 0.01 + floor_width / 2.0)).astype(int)
#         start_xy = np.flip((np.array(goal_pos_metric[0]) / 0.01 + floor_width / 2.0)).astype(int)
#         floorplan_ary[max(0, end_xy[0]-5) : min(end_xy[0]+5, floorplan_ary.shape[0]), max(0, end_xy[1]-5) : min(end_xy[1]+5, floorplan_ary.shape[1]), :] = np.array([0, 0, 255, 255])
        
#         ax1.imshow(cur_obs[-1].permute(1,2,0).cpu().detach().numpy())
#         ax2.plot(save_action[:,0], save_action[:,1], marker = '.')
#         for i, xy in enumerate(actions_meter_global):
#             map_xy = np.flip((np.array(xy) / 0.01 + floor_width / 2.0)).astype(int)
#             if i == 0:
#                 start_xy = map_xy
#             if i < 8:
#                 color = np.array([255, 0, 0, 255])
#                 floorplan_ary[map_xy[0]-2 : map_xy[0]+2, map_xy[1]-2 : map_xy[1]+2, :] = color
#             else:
#                 color = np.array([0, 255, 0, 30])
#                 floorplan_ary[map_xy[0]-1 : map_xy[0]+1, map_xy[1]-1 : map_xy[1]+1, :] = color
            
#         ax3.imshow(floorplan_ary[max(0, start_xy[0]-200) : min(start_xy[0]+200, floorplan_ary.shape[0]), max(0, start_xy[1]-200) : min(start_xy[1]+200, floorplan_ary.shape[1]), :])
#         ax4.imshow(floorplan_ary)
#         # plt.plot(actions_meter_global[:,0], actions_meter_global[:,1], marker = 'o')
#         plt.savefig(os.path.join(log_add))
    
#     return actions_meter_global
