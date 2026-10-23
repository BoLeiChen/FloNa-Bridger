import io
import json
import logging
import math
import os
import random
import tarfile
from collections import Counter
from pathlib import PurePosixPath

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import IterableDataset
from datasets.data_selection import scene_selected


LOGGER = logging.getLogger(__name__)


def build_tar_records(samples_index):
    """Map the existing validated sample index to its original scene tar files."""
    counts = Counter()
    for item in samples_index:
        item_info, data_folder = item.rsplit("|", 1)
        scene = item_info.split("::", 1)[0]
        if not scene_selected(scene, {"train_scene_dataset": "hm3d"}):
            continue
        # Resolve subset scene links to locate existing source archives.
        data_folder = os.path.dirname(os.path.realpath(os.path.join(data_folder, scene)))
        counts[(data_folder, scene)] += 1

    records = []
    missing = []
    for (data_folder, scene), sample_count in counts.items():
        tar_folder = data_folder.replace("/floverse_extracted/", "/floverse/")
        tar_path = os.path.join(tar_folder, f"{scene}.tar")
        if os.path.isfile(tar_path):
            records.append((tar_path, data_folder, scene, sample_count))
        else:
            missing.append((tar_path, sample_count))
    return records, missing


def assign_tar_records(records, world_size):
    """Greedily balance whole scene archives across DDP ranks."""
    assignments = [[] for _ in range(world_size)]
    totals = [0] * world_size
    for record in sorted(records, key=lambda value: value[3], reverse=True):
        rank = min(range(world_size), key=lambda idx: totals[idx])
        assignments[rank].append(record)
        totals[rank] += record[3]
    return assignments, totals


class TarNavDataset(IterableDataset):
    """Stream navigation samples directly from scene tar files.

    Reading each archive linearly avoids millions of random small-file reads on
    the HDD. Archives are shuffled between epochs, while samples inside a scene
    stay local to preserve sequential I/O.
    """

    def __init__(
        self,
        records,
        len_traj_pred,
        traj_pace,
        history_length,
        need_currdep,
        max_samples=None,
        seed=0,
    ):
        self.records = [r for r in records if scene_selected(r[2], {"train_scene_dataset": "hm3d"})]
        self.len_traj_pred = int(len_traj_pred)
        self.traj_pace = int(traj_pace)
        self.history_length = int(history_length)
        self.need_currdep = bool(need_currdep)
        available = sum(r[3] for r in self.records)
        self.max_samples = min(int(max_samples), available) if max_samples is not None else available
        self.seed = int(seed)
        self.epoch = 0
        self.transform = transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor()])

    def __len__(self):
        return self.max_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    @staticmethod
    def _yaw_rotmat(yaw):
        return np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    @staticmethod
    def _get_yaw(x, y, x1, y1):
        dx = x1 - x
        dy = y1 - y
        if dx == 0 and dy == 0:
            raise ValueError("The current point coincides with the target point.")
        return np.arctan2(dy, dx)

    def _to_local_coords(self, positions, curr_pos, curr_yaw):
        positions = np.array(positions)
        curr_pos = np.array(curr_pos)
        rotmat = self._yaw_rotmat(float(curr_yaw))
        if positions.shape[-1] == 2:
            rotmat = rotmat[:2, :2]
        return (positions - curr_pos).dot(rotmat)

    def _compute_actions(self, traj_data):
        traj_data = np.array(traj_data)
        positions = traj_data[:, :2].copy()
        yaw = self._get_yaw(traj_data[0][0], traj_data[0][1], traj_data[0][2], traj_data[0][3])
        return self._to_local_coords(positions, positions[0], yaw)

    def _decode_frame(self, payload, convert_rgb=False):
        with Image.open(io.BytesIO(payload)) as image:
            if convert_rgb:
                image = image.convert("RGB")
            return self.transform(image)

    @staticmethod
    def _parse_poses(payload):
        poses = []
        for line in payload.decode("utf-8").splitlines():
            values = list(map(float, line.strip().split()))
            if len(values) == 4:
                poses.append(values)
        return poses

    def _history_pose_tensor(self, all_pose, start_idx):
        if start_idx - self.history_length < 0:
            history = [all_pose[0] for _ in range(self.history_length - start_idx)] + all_pose[: start_idx + 1]
        else:
            history = all_pose[start_idx - self.history_length : start_idx + 1]
        history.reverse()
        values = self._compute_actions(history)
        return torch.tensor(values[1:].tolist()[::-1], dtype=torch.float32)[:, :2]

    def _future_pose_tensor(self, all_pose, start_idx):
        values = self._compute_actions(all_pose[start_idx : start_idx + self.len_traj_pred])
        return torch.tensor(values, dtype=torch.float32)[:, :2]

    def _point_goal_tensor(self, all_pose, start_idx):
        point_goal = all_pose[-1]
        curr_pos = all_pose[start_idx]
        yaw = self._get_yaw(curr_pos[0], curr_pos[1], curr_pos[2], curr_pos[3])
        value = self._to_local_coords(point_goal[:2], curr_pos[:2], yaw)
        return torch.tensor(value, dtype=torch.float32).unsqueeze(0)[:, :2]

    def _trajectory_samples(self, bundle, floorplan, data_folder, scene, traj_name):
        if bundle["poses"] is None:
            raise RuntimeError(f"Missing trajectory pose file in {scene}/{traj_name}")
        all_pose = self._parse_poses(bundle["poses"])
        goal_idx = len(all_pose) - 1
        object_goal = "null"
        if bundle["object"] is not None:
            object_data = json.loads(bundle["object"].decode("utf-8"))
            if isinstance(object_data, list):
                object_data = object_data[0] if object_data else {}
            object_goal = object_data.get("object_category", "null")

        # Decode every unique frame only once. Consecutive training samples have
        # heavily overlapping history windows, so decoding inside the sample
        # loop would throw away most of the benefit of streaming by trajectory.
        decoded_rgb = {
            idx: self._decode_frame(payload, convert_rgb=True)
            for idx, payload in bundle["rgb"].items()
        }
        decoded_depth = {}
        current_depth_frames = {}
        for idx, payload in bundle["depth"].items():
            with Image.open(io.BytesIO(payload)) as image:
                decoded_depth[idx] = self.transform(image)
                if idx % self.traj_pace == 0:
                    current_depth_frames[idx] = np.array(image)

        for start_idx in range(0, len(all_pose), self.traj_pace):
            if start_idx + self.len_traj_pred > len(all_pose):
                break
            history_indices = [max(0, idx) for idx in range(start_idx - self.history_length, start_idx)]
            try:
                image_goal = decoded_rgb[goal_idx]
                stack_rgb = torch.stack(
                    [decoded_rgb[idx] for idx in history_indices]
                )
                stack_depth = torch.stack([decoded_depth[idx] for idx in history_indices])
                current_depth = current_depth_frames[start_idx]
            except KeyError as exc:
                raise RuntimeError(f"Missing frame {exc} in {scene}/{traj_name}") from exc

            item_info = f"{scene}::{traj_name}::{start_idx}"
            result = (
                torch.as_tensor(current_depth, dtype=torch.float32),
                stack_rgb.to(dtype=torch.float32),
                stack_depth.to(dtype=torch.float32),
                self._history_pose_tensor(all_pose, start_idx),
                self._future_pose_tensor(all_pose, start_idx),
                floorplan,
                image_goal.to(dtype=torch.float32),
                self._point_goal_tensor(all_pose, start_idx),
                object_goal,
                item_info,
            )
            if self.need_currdep:
                yield result
            else:
                yield result[1:]

    def _archive_samples(self, record):
        tar_path, data_folder, scene, _ = record
        floorplan_path = os.path.join(data_folder, scene, "floorplan.png")
        with Image.open(floorplan_path) as image:
            floorplan = self.transform(image).to(dtype=torch.float32)

        current_traj = None
        bundle = None
        with tarfile.open(tar_path, mode="r|") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                parts = PurePosixPath(member.name).parts
                if len(parts) < 3 or not parts[1].startswith("traj_"):
                    continue
                traj_name = parts[1]
                if current_traj is not None and traj_name != current_traj:
                    yield from self._trajectory_samples(bundle, floorplan, data_folder, scene, current_traj)
                    bundle = None
                if bundle is None:
                    current_traj = traj_name
                    bundle = {"poses": None, "object": None, "rgb": {}, "depth": {}}

                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                payload = extracted.read()
                relative = parts[2:]
                filename = parts[-1]
                if len(relative) == 1 and filename == f"{traj_name}.txt":
                    bundle["poses"] = payload
                elif relative[0] == "rgb" and filename.endswith(".png"):
                    bundle["rgb"][int(PurePosixPath(filename).stem.rsplit("_", 1)[1])] = payload
                elif relative[0] == "depth" and filename.endswith(".png"):
                    bundle["depth"][int(PurePosixPath(filename).stem.rsplit("_", 1)[1])] = payload
                elif relative[:2] == ("object", "object.json") or filename == "object.json":
                    bundle["object"] = payload

            if bundle is not None:
                yield from self._trajectory_samples(bundle, floorplan, data_folder, scene, current_traj)

    def __iter__(self):
        records = list(self.records)
        random.Random(self.seed + self.epoch).shuffle(records)
        yielded = 0
        for record in records:
            for sample in self._archive_samples(record):
                if yielded >= self.max_samples:
                    return
                yielded += 1
                yield sample
        if yielded != self.max_samples:
            raise RuntimeError(f"Tar stream yielded {yielded} samples, expected {self.max_samples}")
