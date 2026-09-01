import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
import yaml
import time
import sys

# Add project root to sys.path
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also add the outer diffusion_policy directory to sys.path
diffusion_policy_path = os.path.join(project_root, 'diffusion_policy')
if diffusion_policy_path not in sys.path:
    sys.path.insert(0, diffusion_policy_path)

import numpy as np
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from model.flona import flona, DenseNetwork

from model.flona_vint import flona_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from model.flona_dataset import flona_Dataset
from model.navibridger import NaviBridgeConfig, NaviBridgeDenoiser
from train_eval_loop import train_eval_loop_flona, load_model


def main(config):
    # ==============================Dataset==============================
    data_config = config["datasets"]
    train_dataset = flona_Dataset(
        data_folder=os.path.join(data_config["data_folder"], "train"),
        trav_folder=data_config["trav_map_folder"],
        scene_names=data_config['scene_names'],
        obs_image_size=tuple(config["obs_image_size"]),          # 224x224 for DINOv2
        floorplan_image_size=tuple(config["floorplan_image_size"]),  # 96x96 for EfficientNet
        waypoint_spacing=data_config["waypoint_spacing"],
        len_traj_pred=config["len_traj_pred"],
        context_size=config["context_size"],
        end_slack=data_config["end_slack"],
        normalize=config["normalize"],
        planner_floorplan_size=tuple(config["planner_floorplan_size"]),
        planner_free_thresh=float(config["planner_free_thresh"]),
    )
    test_dataset = flona_Dataset(
        data_folder=os.path.join(data_config["data_folder"], "test"),
        trav_folder=data_config["trav_map_folder"],
        scene_names=data_config['scene_test_names'],
        obs_image_size=tuple(config["obs_image_size"]),          # 224x224 for DINOv2
        floorplan_image_size=tuple(config["floorplan_image_size"]),  # 96x96 for EfficientNet
        waypoint_spacing=data_config["waypoint_spacing"],
        len_traj_pred=config["len_traj_pred"],
        context_size=config["context_size"],
        end_slack=data_config["end_slack"],
        normalize=config["normalize"],
        planner_floorplan_size=tuple(config["planner_floorplan_size"]),
        planner_free_thresh=float(config["planner_free_thresh"]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=True,
        pin_memory=True,
    )
    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["eval_batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    # ==============================Model==============================
    vision_encoder = flona_ViNT(
        obs_encoding_size=config["encoding_size"],
        context_size=config["context_size"],
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
    is_navibridger = config.get("diffusion_mode", "ddpm") == "navibridger"
    noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            local_cond_dim=5,
            global_cond_dim=config["encoding_size"],    # +6
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )
    dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])   # +6
    model = flona(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    )
    bridge_diffusion = None
    if is_navibridger:
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

    # ==============================Training Configuration==============================
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in config["gpu_ids"]])
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True  
    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)
    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = None
  
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
    if config["warmup"]:
        print("Using warmup")
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=config["warmup_epochs"],
            after_scheduler=scheduler,
        )
    current_epoch = 0    

    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        print("Loading model from ", load_project_folder)
        latest_path = os.path.join(load_project_folder, "latest.pth")
        latest_checkpoint = torch.load(latest_path) #f"cuda:{}" if torch.cuda.is_available() else "cpu")
        load_model(model, latest_checkpoint)
        if "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1

    # Multi-GPU
    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    model = model.to(device)

    if "load_run" in config:  # load optimizer and scheduler after data parallel
        if "optimizer" in latest_checkpoint:
            optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
        if scheduler is not None and "scheduler" in latest_checkpoint:
            scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())

    # ==============================Train==============================
    train_eval_loop_flona(
        train_model=config["train"],
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        noise_scheduler=noise_scheduler,
        train_loader=train_loader,
        test_loader=test_dataloader,
        transform=transform,
        epochs=config["epochs"],
        device=device,
        project_folder=config["project_folder"],
        print_log_freq=config["print_log_freq"],
        wandb_log_freq=config["wandb_log_freq"],
        image_log_freq=config["image_log_freq"],
        num_images_log=config["num_images_log"],
        current_epoch=current_epoch,
        alpha=float(config["alpha"]),
        use_wandb=config["use_wandb"],
        eval_fraction=config["eval_fraction"],
        eval_freq=config["eval_freq"],
        save_interval_epochs=config["save_interval_epochs"],
        perturb_config={
            "metric_waypoint_spacing": float(config["metric_waypoint_spacing"]),
            "waypoint_spacing": int(data_config["waypoint_spacing"]),
            "path_noise_apply": bool(config.get("path_noise_apply", True)),
            "path_noise_mode": str(config.get("path_noise_mode", "structured")),
            "path_noise_alpha_m": float(config.get("path_noise_alpha_m", 0.02)),
            "path_noise_control_points": int(config.get("path_noise_control_points", 4)),
            "path_noise_translation_std_m": float(
                config.get("path_noise_translation_std_m", 0.01)
            ),
            "path_noise_schedule_epochs": config.get("path_noise_schedule_epochs", 40),
            "pose_noise_apply": bool(config.get("pose_noise_apply", True)),
            "pose_noise_pos_std_m": float(config.get("pose_noise_pos_std_m", 0.1)),
            "pose_noise_rot_std_rad": float(config.get("pose_noise_rot_std_rad", np.pi / 36.0)),
        },
        bridge_diffusion=bridge_diffusion,
    )

    print("Done!!!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")
    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="flona.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name of the run. If not provided, the name in config + timestamp will be used.",
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    if args.run_name:
        config["run_name"] = args.run_name
    else:
        config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
        
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    os.makedirs(
        config["project_folder"],  # should error if dir already exists to avoid overwriting and old project
    )
    if config["use_wandb"]:
        wandb.login()
        wandb.init(
            project=config["project_name"],
            settings=wandb.Settings(),          
        )
        wandb.save(args.config, policy="now")  # save the config file
        wandb.run.name = config["run_name"]
        # update the wandb args with the training configurations
        if wandb.run:
            wandb.config.update(config)

    main(config)
