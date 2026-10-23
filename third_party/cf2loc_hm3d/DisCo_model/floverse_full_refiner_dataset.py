"""On-the-fly local-refiner supervision over the indexed HM3D LMDB."""

import io
import math
import os
from functools import lru_cache

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .disco_dataset import DisCo_Dataset
from .floverse_full_dataset import FloVerseFullDataset
from .pose_local_refiner import (
    crop_to_refiner_tensor,
    pose_delta_to_local_m,
    wrap_to_pi,
)


class FloVerseFullRefinerDataset(Dataset):
    """Generate candidate/correction pairs without duplicating the full image set."""

    def __init__(self, config, split, deterministic=False):
        dataset_cfg = config["datasets"]
        self.frames = FloVerseFullDataset(
            config["full_index"],
            split,
            augment=False,
            lmdb_root=config.get("lmdb_root"),
        )
        self.split = split
        self.map_res = float(dataset_cfg.get("map_res", 0.02))
        self.crop_size_meters = float(config.get("refiner_crop_size_meters", 5.0))
        self.crop_output_size = int(config.get("refiner_crop_output_size", 256))
        self.max_delta_m = float(config.get("refiner_max_delta_m", 1.5))
        self.max_delta_theta = math.radians(
            float(config.get("refiner_max_delta_theta_deg", 45.0))
        )
        self.score_sigma_m = float(config.get("refiner_score_sigma_m", 0.5))
        self.score_sigma_theta = math.radians(
            float(config.get("refiner_score_sigma_deg", 20.0))
        )
        self.deterministic = bool(deterministic)
        self.seed = int(config.get("refiner_val_seed", 42))
        self.representation = dataset_cfg.get(
            "refiner_floorplan_representation", "gray"
        )
        self.oriented_crop = bool(dataset_cfg.get("refiner_oriented_crop", True))
        augmentation = dataset_cfg.get("map_pose_rot_aug", {})
        self.rotation_enabled = bool(augmentation.get("enable", False))
        self.rotation_probability = float(augmentation.get("p", 0.0))
        self.rotation_angles = tuple(
            int(angle // 90) % 4 for angle in augmentation.get("angles", [90, 180, 270])
        )

    def __len__(self):
        return len(self.frames)

    def _rng(self, index):
        if self.deterministic:
            return np.random.default_rng(self.seed + int(index))
        return np.random

    def _sample_rotation(self, rng):
        if (
            self.split != "train"
            or not self.rotation_enabled
            or rng.random() >= self.rotation_probability
        ):
            return 0
        return int(rng.choice(self.rotation_angles))

    def _sample_candidate(self, pose, wh, rng):
        mode = float(rng.random())
        if mode < 0.5:
            offset_m = rng.normal(0.0, 0.3, size=2)
            theta_delta = float(rng.normal(0.0, math.radians(10.0)))
        elif mode < 0.8:
            offset_m = rng.normal(0.0, 0.8, size=2)
            theta_delta = float(rng.normal(0.0, math.radians(25.0)))
        else:
            radius = self.max_delta_m * math.sqrt(float(rng.random()))
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
            offset_m = radius * np.asarray([math.cos(angle), math.sin(angle)])
            theta_delta = float(
                rng.uniform(-self.max_delta_theta, self.max_delta_theta)
            )
        norm = float(np.linalg.norm(offset_m))
        if norm > self.max_delta_m:
            offset_m = offset_m / max(norm, 1e-6) * self.max_delta_m
        theta_delta = float(
            np.clip(theta_delta, -self.max_delta_theta, self.max_delta_theta)
        )
        candidate = pose.astype(np.float32).copy()
        candidate[:2] += offset_m.astype(np.float32) / self.map_res
        candidate[0] = np.clip(candidate[0], 0.0, max(float(wh[0] - 1), 0.0))
        candidate[1] = np.clip(candidate[1], 0.0, max(float(wh[1] - 1), 0.0))
        candidate[2] = np.mod(candidate[2] + theta_delta, 2.0 * math.pi)
        return candidate

    @lru_cache(maxsize=64)
    def _raw_map(self, scene_index):
        path = self.frames.scenes[int(scene_index)]["map_path"]
        raw_map = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if raw_map is None:
            raise FileNotFoundError(f"Failed to load floorplan {path}")
        return raw_map

    def _observation_and_metadata(self, index):
        original_index = (
            int(self.frames.selection[index])
            if self.frames.selection is not None
            else int(index)
        )
        row = self.frames.rows[original_index]
        trajectory = self.frames.trajectories[int(row["traj"])]
        scene_index = int(trajectory["scene"])
        scene = self.frames.scenes[scene_index]
        path = os.path.join(trajectory["path"], "rgb", f"rgb_{int(row['frame'])}.png")
        source = (
            io.BytesIO(self.frames._image_payload(index))
            if self.frames.lmdb_path
            else path
        )
        with Image.open(source) as image:
            if image.size != (256, 256):
                raise ValueError(f"Unexpected image dimensions: {path}")
            observation = self.frames.transform(image.convert("RGB"))
        return observation, row["pose"].copy(), scene_index, tuple(scene["map_size"])

    def __getitem__(self, index):
        rng = self._rng(index)
        observation, pose, scene_index, wh = self._observation_and_metadata(index)
        rotation = self._sample_rotation(rng)
        raw_map = self._raw_map(scene_index)
        if rotation:
            pose, width, height = DisCo_Dataset._rotate_pose_wh_90(
                pose, wh[0], wh[1], rotation
            )
            wh = (width, height)
            raw_map = DisCo_Dataset._rotate_array_90(raw_map, rotation)
        candidate = self._sample_candidate(pose, wh, rng)
        local_map = crop_to_refiner_tensor(
            raw_map,
            candidate,
            crop_size_meters=self.crop_size_meters,
            map_res=self.map_res,
            output_size=self.crop_output_size,
            representation=self.representation,
            oriented=self.oriented_crop,
        )
        pose_tensor = torch.from_numpy(pose.astype(np.float32))
        candidate_tensor = torch.from_numpy(candidate.astype(np.float32))
        delta_xy_m, delta_theta = pose_delta_to_local_m(
            pose_tensor, candidate_tensor, self.map_res
        )
        position_error = torch.linalg.norm(
            (candidate_tensor[:2] - pose_tensor[:2]) * self.map_res
        )
        theta_error = torch.abs(wrap_to_pi(candidate_tensor[2] - pose_tensor[2]))
        score = torch.exp(
            -0.5 * (position_error / max(self.score_sigma_m, 1e-6)).square()
            - 0.5 * (theta_error / max(self.score_sigma_theta, 1e-6)).square()
        )
        return {
            "obs_img": observation.float(),
            "local_map": local_map.float(),
            "pose": pose_tensor.float(),
            "candidate_pose": candidate_tensor.float(),
            "wh": torch.tensor(wh, dtype=torch.float32),
            "target_delta_xy_m": delta_xy_m.float(),
            "target_delta_theta": delta_theta.float(),
            "score_target": score.float(),
        }
