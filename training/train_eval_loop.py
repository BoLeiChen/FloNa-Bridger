import os
from typing import List, Optional, Dict

import wandb
import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from training.train_utils import train_flona, evaluate_flona
from model.navibridger import NaviBridgeDenoiser
import copy

def train_eval_loop_flona(
    train_model: bool,
    model: nn.Module,
    optimizer: AdamW, 
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    noise_scheduler: DDPMScheduler,
    train_loader: DataLoader,
    test_loader: DataLoader,
    transform: transforms,
    epochs: int,
    device: torch.device,
    project_folder: str,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 1e-4,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    eval_freq: int = 1,
    save_interval_epochs: int = 1,
    perturb_config: Optional[Dict] = None,
    bridge_diffusion: Optional[NaviBridgeDenoiser] = None,
):
    """
    Train and evaluate the model for several epochs
    Args:
        train_model: train or eval model
        model: model to train
        optimizer: optimizer to use
        lr_scheduler: learning rate scheduler to use
        noise_scheduler: noise scheduler to use
        dataloader: dataloader for train dataset
        test_dataloaders: dict of dataloaders for testing
        transform: transform to apply to images
        epochs: number of epochs to train
        device: device to train on
        project_folder: folder to save checkpoints and logs
        wandb_log_freq: frequency of logging to wandb
        print_log_freq: frequency of printing to console
        image_log_freq: frequency of logging images to wandb
        num_images_log: number of images to log to wandb
        current_epoch: epoch to start training from
        alpha: tradeoff between distance and action loss
        use_wandb: whether to log to wandb or not
        eval_fraction: fraction of training data to use for evaluation
        eval_freq: frequency of evaluation
    """
    # ==============================Train==============================
    latest_path = os.path.join(project_folder, f"latest.pth")
    ema_model = EMAModel(parameters=model.parameters(), decay=0.75)
    ema_model.averaged_model = copy.deepcopy(model)
    for p in ema_model.averaged_model.parameters():
        p.requires_grad = False

    schedule_total_epochs = epochs
    if perturb_config is not None and perturb_config.get("path_noise_schedule_epochs") is not None:
        schedule_total_epochs = int(perturb_config["path_noise_schedule_epochs"])

    for epoch in range(current_epoch, current_epoch + epochs):
        schedule_epoch = epoch - current_epoch + 1
        if train_model:
            print(
            f"Start flona Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            train_flona(
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                device=device,
                noise_scheduler=noise_scheduler,
                project_folder=project_folder,
                epoch=epoch,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                alpha=alpha,
                total_epochs=epochs,
                schedule_epoch=schedule_epoch,
                schedule_total_epochs=schedule_total_epochs,
                perturb_config=perturb_config,
                bridge_diffusion=bridge_diffusion,
            )
            lr_scheduler.step()

        if (epoch + 1) % save_interval_epochs == 0:
            numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
            torch.save(ema_model.averaged_model.state_dict(), numbered_path)
            latest_ema_path = os.path.join(project_folder, f"ema_latest.pth")
            torch.save(ema_model.averaged_model.state_dict(), latest_ema_path)
            print(f"Saved EMA model to {numbered_path} and {latest_ema_path}")

            numbered_path = os.path.join(project_folder, f"{epoch}.pth")
            torch.save(model.state_dict(), numbered_path)
            torch.save(model.state_dict(), latest_path)
            print(f"Saved model to {numbered_path}")

            # save optimizer
            latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
            torch.save(optimizer.state_dict(), latest_optimizer_path)

            # save scheduler
            latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
            torch.save(lr_scheduler.state_dict(), latest_scheduler_path)

        #==============================Eval==============================
        if (epoch + 1) % eval_freq == 0: 
            print(
                f"Start flona Testing Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            loader = test_loader
            evaluate_flona(
                model=model,
                dataloader=loader,
                transform=transform,
                device=device,
                noise_scheduler=noise_scheduler,
                project_folder=project_folder,
                epoch=epoch,
                print_log_freq=print_log_freq,
                num_images_log=num_images_log,
                wandb_log_freq=wandb_log_freq,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
                perturb_config=perturb_config,
                bridge_diffusion=bridge_diffusion,
            )
        
        if use_wandb:
            wandb.log({
                "lr": optimizer.param_groups[0]["lr"],
            })
       
    print()

def load_model(model, checkpoint: dict) -> None:
    """Load model from checkpoint."""
    state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
