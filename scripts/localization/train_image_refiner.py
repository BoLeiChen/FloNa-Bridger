import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
"""Three-GPU local-refiner training on the disjoint HM3D LMDB."""

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, DistributedSampler, Subset

from third_party.cf2loc_hm3d.DisCo_model.floverse_full_dataset import LocalityDistributedSampler
from third_party.cf2loc_hm3d.DisCo_model.floverse_full_refiner_dataset import FloVerseFullRefinerDataset
from third_party.cf2loc_hm3d.DisCo_model.pose_local_refiner import PoseLocalRefinerLightning


class DistributedRefiner(PoseLocalRefinerLightning):
    """Synchronize validation metrics before checkpoint selection."""

    def log(self, name, value, *args, **kwargs):
        if name.startswith("val_"):
            kwargs.setdefault("sync_dist", True)
        return super().log(name, value, *args, **kwargs)


class FullRefinerData(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        config = self.config
        self.train_data = FloVerseFullRefinerDataset(config, "train")
        full_validation = FloVerseFullRefinerDataset(config, "val", deterministic=True)
        requested = min(
            len(full_validation), int(config.get("validation_frames", 19200))
        )
        global_batch = self.trainer.world_size * int(config["val_batch_size"])
        validation_count = (requested // global_batch) * global_batch
        if validation_count == 0:
            raise ValueError("Validation selection is smaller than one global batch")
        generator = np.random.default_rng(int(config.get("validation_seed", 42)))
        self.validation_indices = np.sort(
            generator.choice(len(full_validation), validation_count, replace=False)
        )
        self.val_data = Subset(full_validation, self.validation_indices.tolist())
        if self.trainer.is_global_zero:
            run_dir = Path(config["run_dir"])
            run_dir.mkdir(parents=True, exist_ok=True)
            np.save(run_dir / "validation_indices.npy", self.validation_indices)
            print(
                "REFINER_DATA_READY",
                json.dumps(
                    {
                        "train_frames": len(self.train_data),
                        "validation_index_frames": len(full_validation),
                        "validation_per_epoch": len(self.val_data),
                        "buildings": {
                            split: len(info["buildings"])
                            for split, info in self.train_data.frames.meta[
                                "splits"
                            ].items()
                        },
                        "candidate_supervision": "online_synthetic_local_se2",
                    }
                ),
                flush=True,
            )

    def train_dataloader(self):
        config = self.config
        sampler = LocalityDistributedSampler(
            len(self.train_data),
            self.trainer.global_rank,
            self.trainer.world_size,
            int(config["batch_size"]),
            int(config.get("io_block_size", 2048)),
            seed=int(config.get("seed", 42)),
        )
        return DataLoader(
            self.train_data,
            batch_size=int(config["batch_size"]),
            sampler=sampler,
            num_workers=int(config.get("num_workers", 4)),
            pin_memory=True,
            persistent_workers=int(config.get("num_workers", 4)) > 0,
            prefetch_factor=2 if int(config.get("num_workers", 4)) > 0 else None,
        )

    def val_dataloader(self):
        config = self.config
        sampler = DistributedSampler(
            self.val_data,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
            shuffle=False,
            drop_last=False,
        )
        return DataLoader(
            self.val_data,
            batch_size=int(config["val_batch_size"]),
            sampler=sampler,
            num_workers=int(config.get("num_workers", 4)),
            pin_memory=True,
            persistent_workers=int(config.get("num_workers", 4)) > 0,
            prefetch_factor=2 if int(config.get("num_workers", 4)) > 0 else None,
        )


class Progress(pl.Callback):
    def on_train_start(self, trainer, module):
        self.started = time.monotonic()
        self.start_step = trainer.global_step

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return
        if trainer.global_step % 100 and trainer.global_step > 3:
            return
        elapsed = time.monotonic() - self.started
        steps = trainer.global_step - self.start_step
        global_fps = (
            steps
            * int(module.config["batch_size"])
            * trainer.world_size
            / max(elapsed, 1e-6)
        )
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        print(
            "REFINER_TRAIN_PROGRESS",
            json.dumps(
                {
                    "epoch": trainer.current_epoch + 1,
                    "step": trainer.global_step,
                    "loss": float(loss.detach()),
                    "global_frames_s": round(global_fps, 2),
                    "elapsed_s": round(elapsed),
                    "gpu_memory_MiB": round(
                        torch.cuda.max_memory_allocated() / 1024 ** 2
                    ),
                }
            ),
            flush=True,
        )

    def on_validation_epoch_end(self, trainer, module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        keys = (
            "val_loss",
            "val_candidate_0.5m_recall",
            "val_refined_0.1m_recall",
            "val_refined_0.5m_recall",
            "val_refined_1m_recall",
            "val_refined_1m_30deg_recall",
            "val_mean_xy_err_m",
        )
        print(
            "REFINER_VALIDATION",
            json.dumps(
                {
                    "epoch": trainer.current_epoch + 1,
                    "step": trainer.global_step,
                    **{
                        key: float(trainer.callback_metrics[key])
                        for key in keys
                        if key in trainer.callback_metrics
                    },
                }
            ),
            flush=True,
        )


def _load_weights_only(model, checkpoint_path):
    """Load model parameters without restoring optimizer/scheduler state."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=True)
    print(
        "REFINER_WEIGHTS_ONLY_LOADED",
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "missing_keys": list(incompatible.missing_keys),
                "unexpected_keys": list(incompatible.unexpected_keys),
            }
        ),
        flush=True,
    )


def main(config, resume=None, weights_only=None, smoke=False):
    config = dict(config)
    if smoke:
        config["run_dir"] = str(Path(config["run_dir"]).with_name(Path(config["run_dir"]).name + "_smoke"))
    pl.seed_everything(int(config.get("seed", 42)), workers=True)
    torch.set_num_threads(4)
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    model = DistributedRefiner(config)
    if resume and weights_only:
        raise ValueError("--resume and --weights-only are mutually exclusive")
    if weights_only:
        _load_weights_only(model, weights_only)
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="{epoch:02d}-{val_refined_0.5m_recall:.4f}",
        monitor="val_refined_0.5m_recall",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    recovery = ModelCheckpoint(
        dirpath=run_dir / "recovery",
        filename="step-{step}",
        monitor=None,
        save_top_k=1,
        save_last=True,
        train_time_interval=timedelta(minutes=30),
        save_on_train_epoch_end=False,
    )
    callbacks = [checkpoint, recovery, Progress()]
    patience = config.get("early_stopping_patience")
    if patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val_refined_0.5m_recall",
                mode="max",
                patience=int(patience),
                min_delta=float(config.get("early_stopping_min_delta", 0.001)),
                check_finite=True,
                verbose=True,
            )
        )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=config["gpu_ids"],
        strategy=DDPStrategy(find_unused_parameters=False),
        precision=config.get("precision", "bf16-mixed"),
        max_epochs=1 if smoke else int(config.get("epochs", 5)),
        callbacks=callbacks,
        logger=CSVLogger(run_dir, name="metrics"),
        default_root_dir=run_dir,
        enable_progress_bar=False,
        gradient_clip_val=float(config.get("gradient_clip_val", 1.0)),
        log_every_n_steps=int(config.get("log_every_n_steps", 10)),
        num_sanity_val_steps=1,
        use_distributed_sampler=False,
        limit_train_batches=2 if smoke else 1.0,
        limit_val_batches=2 if smoke else 1.0,
    )
    trainer.fit(model, datamodule=FullRefinerData(config), ckpt_path=resume)
    if trainer.is_global_zero:
        (run_dir / "completed.json").write_text(
            json.dumps(
                {
                    "epochs": trainer.current_epoch + 1,
                    "step": trainer.global_step,
                    "best_checkpoint": checkpoint.best_model_path,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--weights-only")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as stream:
        configuration = yaml.safe_load(stream)
    main(
        configuration,
        resume=args.resume,
        weights_only=args.weights_only,
        smoke=args.smoke,
    )
