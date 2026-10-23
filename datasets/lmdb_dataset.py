import json
import logging
import random

import lmdb
import msgpack
import numpy as np
import torch
import zstandard as zstd

from datasets.tar_dataset import TarNavDataset
from datasets.data_selection import select_training_manifest, read_subset_manifest


LOGGER = logging.getLogger(__name__)


def assign_lmdb_manifest(manifest, world_size):
    assignments = [[] for _ in range(world_size)]
    totals = [0] * world_size
    for item in sorted(manifest, key=lambda value: value["sample_count"], reverse=True):
        rank = min(range(world_size), key=lambda idx: totals[idx])
        assignments[rank].append(item)
        totals[rank] += item["sample_count"]
    return assignments, totals


def _pack_array(array):
    array = np.ascontiguousarray(array)
    return {"shape": array.shape, "dtype": array.dtype.str, "data": array.tobytes()}


def _unpack_array(value):
    return np.frombuffer(value["data"], dtype=np.dtype(value["dtype"])).reshape(value["shape"])


def encode_record(value, compression_level=3):
    packed = {}
    for key, item in value.items():
        packed[key] = _pack_array(item) if isinstance(item, np.ndarray) else item
    return zstd.ZstdCompressor(level=compression_level).compress(msgpack.packb(packed, use_bin_type=True))


def decode_record(payload):
    packed = msgpack.unpackb(zstd.ZstdDecompressor().decompress(payload), raw=False)
    result = {}
    for key, item in packed.items():
        result[key] = _unpack_array(item) if isinstance(item, dict) and "dtype" in item else item
    return result


class LmdbNavDataset(TarNavDataset):
    """Read pre-resized, trajectory-deduplicated training data from LMDB."""

    def __init__(
        self,
        lmdb_path,
        len_traj_pred,
        traj_pace,
        history_length,
        need_currdep,
        max_samples=None,
        seed=0,
        manifest=None,
        data_folder=None,
        scene_dataset="hm3d",
        source_traj_pace=4,
        source_len_traj_pred=16,
        subset_manifest=None,
    ):
        super().__init__([], len_traj_pred, traj_pace, history_length, need_currdep, max_samples=0, seed=seed)
        self.lmdb_path = str(lmdb_path)
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, subdir=True)
        if manifest is None and subset_manifest:
            self.manifest = read_subset_manifest(subset_manifest, self.lmdb_path)
        elif manifest is None:
            with env.begin() as txn:
                manifest_payload = txn.get(b"__manifest__")
                if manifest_payload is None:
                    raise RuntimeError(f"LMDB manifest missing from {self.lmdb_path}")
                self.manifest = msgpack.unpackb(manifest_payload, raw=False)
        else:
            self.manifest = list(manifest)
        env.close()
        if manifest is None:
            roots = data_folder or sorted({x['floor_key'].split('::')[1] for x in self.manifest})
            self.manifest = select_training_manifest(self.manifest, dict(
                data_folder=roots, train_scene_dataset=scene_dataset, traj_pace=traj_pace,
                len_traj_pred=len_traj_pred, lmdb_source_traj_pace=source_traj_pace,
                lmdb_source_len_traj_pred=source_len_traj_pred))
        available = sum(item["sample_count"] for item in self.manifest)
        self.max_samples = min(int(max_samples), available) if max_samples is not None else available

    def _record_samples(self, record, floorplan):
        poses = record["poses"].astype(np.float64).tolist()
        rgb = torch.from_numpy(record["rgb"].copy()).permute(0, 3, 1, 2).to(dtype=torch.float32).div_(255.0)
        depth_array = record["depth"].copy()
        depth = torch.from_numpy(depth_array).unsqueeze(1).to(dtype=torch.float32)
        if depth_array.dtype == np.uint8:
            depth.div_(255.0)
        current_indices = record["current_indices"].astype(np.int64)
        current_lookup = {int(idx): pos for pos, idx in enumerate(current_indices)}
        current_depth = record["current_depth"]
        goal_idx = len(poses) - 1
        scene = record["scene"]
        traj_name = record["traj_name"]

        for start_idx in range(0, len(poses), self.traj_pace):
            if start_idx + self.len_traj_pred > len(poses):
                break
            history_indices = [max(0, idx) for idx in range(start_idx - self.history_length, start_idx)]
            item_info = f"{scene}::{traj_name}::{start_idx}"
            result = (
                torch.as_tensor(current_depth[current_lookup[start_idx]].copy(), dtype=torch.float32),
                rgb[history_indices],
                depth[history_indices],
                self._history_pose_tensor(poses, start_idx),
                self._future_pose_tensor(poses, start_idx),
                floorplan,
                rgb[goal_idx],
                self._point_goal_tensor(poses, start_idx),
                record["object_goal"],
                item_info,
            )
            yield result if self.need_currdep else result[1:]

    def __iter__(self):
        manifest = list(self.manifest)
        random.Random(self.seed + self.epoch).shuffle(manifest)
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=True, subdir=True, max_readers=256)
        yielded = 0
        floorplan_cache = {}
        try:
            with env.begin(buffers=True) as txn:
                for item in manifest:
                    floor_key = item["floor_key"]
                    if floor_key not in floorplan_cache:
                        floorplan_cache[floor_key] = torch.from_numpy(
                            decode_record(bytes(txn.get(floor_key.encode("utf-8"))))["floorplan"].copy()
                        ).to(dtype=torch.float32)
                    record = decode_record(bytes(txn.get(item["key"].encode("utf-8"))))
                    count = len(range(0, len(record['poses'])-self.len_traj_pred+1, self.traj_pace))
                    if count != item['sample_count']:
                        raise ValueError('Selected LMDB manifest count does not match requested window stride')
                    starts = np.arange(0, len(record['poses'])-self.len_traj_pred+1, self.traj_pace)
                    if not np.isin(starts, record['current_indices']).all():
                        raise ValueError('Requested depth observations are absent from source LMDB')
                    for sample in self._record_samples(record, floorplan_cache[floor_key]):
                        if yielded >= self.max_samples:
                            return
                        yielded += 1
                        yield sample
        finally:
            env.close()
        if yielded != self.max_samples:
            raise RuntimeError(f"LMDB yielded {yielded} samples, expected {self.max_samples}")
