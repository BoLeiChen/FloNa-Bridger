"""PointGoal views over existing files/LMDB; sidecars never modify the source data."""
import hashlib
import json
import os
import random
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import lmdb
import msgpack
import numpy as np
import torch
from torch.utils.data import IterableDataset

from datasets.data_selection import scene_selected, select_training_manifest, read_subset_manifest
from datasets.dataset import NavDataset
from datasets.lmdb_dataset import LmdbNavDataset, decode_record
from datasets.bridge_maps import FloorplanPlanner, map_key


def filter_manifest(manifest, roots, overlap_path=None):
    allowed = {str(Path(root).resolve()) for root in roots}
    overlap = set()
    if overlap_path:
        overlap = {line.strip() for line in Path(overlap_path).read_text().splitlines() if line.strip()}
    selected = []
    for item in manifest:
        prefix, root, scene = item['floor_key'].split('::')
        if prefix != 'floor':
            raise ValueError('Unexpected LMDB floor key: ' + item['floor_key'])
        if str(Path(root).resolve()) in allowed and scene.rsplit('_', 1)[0] not in overlap:
            selected.append(item)
    if not selected:
        raise ValueError('No LMDB trajectories match the configured training roots')
    return selected


def load_manifest(path, subset_manifest=None):
    if subset_manifest:
        return read_subset_manifest(subset_manifest, path)
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False)
    try:
        with env.begin() as txn:
            value = txn.get(b'__manifest__')
            if value is None:
                raise ValueError('LMDB has no manifest')
            return msgpack.unpackb(value, raw=False)
    finally:
        env.close()


class PriorStore:
    def __init__(self, config):
        self.config = config
        self.maps = OrderedDict()

    def planner(self, root, scene):
        key = map_key(root, scene)
        if key not in self.maps:
            directory = Path(self.config['map_cache']) / key
            if not (directory / 'metadata.json').exists():
                raise FileNotFoundError('Missing calibrated map; run utils/prepare_pointgoal_bridge.py: ' + str(directory))
            self.maps[key] = FloorplanPlanner(directory, float(self.config.get('astar_max_snap_m', 0.5)))
            if len(self.maps) > 2:
                self.maps.popitem(last=False)
        self.maps.move_to_end(key)
        return self.maps[key]

    def trajectory(self, root, scene, traj_name, poses):
        planner = self.planner(root, scene)
        horizon, pace = int(self.config['len_traj_pred']), int(self.config['traj_pace'])
        step = float(self.config['astar_step_m'])
        starts = np.arange(0, max(0, len(poses) - horizon + 1), pace)
        signature = hashlib.sha256(np.asarray(poses, dtype=np.float64).tobytes()).hexdigest()
        metadata = dict(map=planner.metadata['fingerprint'], pose_hash=signature,
                        horizon=horizon, pace=pace, step=step, max_snap_m=planner.max_snap_m, version=1)
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:20]
        path = Path(self.config['prior_cache']) / map_key(root, scene) / (traj_name + '_' + digest + '.npz')
        if path.exists():
            with np.load(path) as cache:
                return cache['prior'], cache['valid']
        source_pace = int(self.config.get('lmdb_source_traj_pace', 4))
        if pace > source_pace and pace % source_pace == 0:
            # Same geometry: reuse dense prior sidecars at the selected starts.
            dense_metadata = dict(metadata, pace=source_pace)
            dense_digest = hashlib.sha256(json.dumps(dense_metadata, sort_keys=True).encode()).hexdigest()[:20]
            dense_path = path.with_name(traj_name + '_' + dense_digest + '.npz')
            if dense_path.exists():
                with np.load(dense_path) as cache:
                    return cache['prior'][::pace//source_pace], cache['valid'][::pace//source_pace]
        tree = planner.goal_tree(np.asarray(poses)[-1, :2])
        priors, masks = [], []
        for start in starts:
            pose = poses[start]
            yaw = np.arctan2(pose[3] - pose[1], pose[2] - pose[0])
            prior, valid = planner.prior(pose[:2], yaw, poses[-1, :2], horizon, step, tree)
            priors.append(prior)
            masks.append(valid)
        priors = np.asarray(priors, dtype=np.float32).reshape(-1, horizon, 2)
        masks = np.asarray(masks, dtype=np.float32).reshape(-1, horizon)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, prior=priors, valid=masks)
        os.replace(temporary, path)
        return priors, masks


@lru_cache(maxsize=8)
def load_floorplan_image(path):
    """Match the legacy LMDB builder: preserve L/RGB channels, PIL bilinear 96px."""
    from PIL import Image
    with Image.open(path) as image:
        array = np.asarray(image.resize((96, 96), Image.Resampling.BILINEAR)).copy()
    if array.ndim == 2:
        array = array[:, :, None]
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255


def mark_floorplan(floor, planner, position, yaw, goal):
    """VGCNav global map input is unmarked; state is supplied separately."""
    floor=torch.as_tensor(floor,dtype=torch.float32)
    if floor.shape==(1,96,96):
        floor=floor.expand(3,-1,-1)
    if floor.shape!=(3,96,96):
        raise ValueError('Expected RGB or grayscale 96px floorplan')
    return floor.clone()


def pack_sample(rgb, current_rgb, gt, point, sample_id, prior, valid, floorplan, poses, start):
    """RGB history and unmarked floorplan; no depth, object goal, or future RGB."""
    poses=np.asarray(poses)
    pose=poses[start]
    yaw=np.arctan2(pose[3]-pose[1],pose[2]-pose[0])
    return dict(agent_pose=torch.tensor([pose[0],pose[1],yaw],dtype=torch.float32),
                goal_world=torch.tensor(poses[-1,:2].copy(),dtype=torch.float32),
                rgb=rgb, current_rgb=current_rgb, floorplan=floorplan, gt_pose=gt, point_goal=point,
                sample_id=sample_id, prior=torch.as_tensor(prior.copy()),
                prior_valid=torch.as_tensor(valid.copy()))


class PointBridgeFiles(NavDataset):
    def __init__(self, config):
        self.config = config
        super().__init__(config['data_folder'], config['len_traj_pred'], config['traj_pace'], config['history_length'], False)
        self.priors = PriorStore(config)
        self._prior_records = OrderedDict()

    def build_index(self):
        # Unlike the legacy loader, never write an index into source data roots.
        signature = dict(roots=self._data_folders(), horizon=self.len_traj_pred, pace=self.traj_pace,
                         overlap=sorted(self._overlap_scenes()), scene_dataset=self.config.get('train_scene_dataset', 'all'))
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
        cache = Path(self.config['prior_cache']) / ('file_index_' + key + '.json')
        if cache.exists():
            return json.loads(cache.read_text())
        index = []
        for root in self._data_folders():
            for scene in sorted(Path(root).iterdir()):
                if not scene_selected(scene.name, self.config):
                    continue
                if not (scene / 'floorplan.png').is_file() or scene.name.rsplit('_', 1)[0] in self._overlap_scenes():
                    continue
                for trajectory in sorted(scene.iterdir()):
                    path = trajectory / (trajectory.name + '.txt')
                    if not path.is_file():
                        continue
                    count, _ = self.read_traj_txt(str(path))
                    for start in range(0, max(0, count - self.len_traj_pred + 1), self.traj_pace):
                        index.append('%s::%s::%d|%s' % (scene.name, trajectory.name, start, root))
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(cache.name + '.' + str(os.getpid()) + '.tmp')
        temporary.write_text(json.dumps(index))
        os.replace(temporary, cache)
        return index

    @lru_cache(maxsize=128)
    def _bridge_trajectory_poses(self, path):
        _, poses = self.read_traj_txt(path)
        return tuple(tuple(pose) for pose in poses)

    def __getitem__(self, index):
        _, root, scene, traj, start = self._parse_index_item(index)
        directory = Path(root) / scene / traj
        poses = self._bridge_trajectory_poses(str(directory / (traj + '.txt')))
        poses = [list(pose) for pose in poses]
        key = (root, scene, traj)
        if key not in self._prior_records:
            self._prior_records[key] = self.priors.trajectory(root, scene, traj, np.asarray(poses))
            if len(self._prior_records) > 16:
                self._prior_records.popitem(last=False)
        priors, masks = self._prior_records[key]
        slot = start // self.traj_pace
        return pack_sample(
            torch.stack(self._load_history_frames(str(directory / 'rgb'), 'rgb', start, True)),
            self._load_cached_frame(str(directory / 'rgb' / ('rgb_%d.png' % start)), True),
            self._future_pose_tensor(poses, start), self._point_goal_tensor(poses, start),
            '%s::%s::%d' % (scene, traj, start), priors[slot], masks[slot],
            mark_floorplan(load_floorplan_image(str(Path(root) / scene / 'floorplan.png')), self.priors.planner(root, scene),
                           poses[start][:2], np.arctan2(poses[start][3]-poses[start][1], poses[start][2]-poses[start][0]), poses[-1][:2]), poses, start)


class PointBridgeLMDB(LmdbNavDataset):
    def __init__(self, config, manifest, max_samples, seed=0):
        super().__init__(config['lmdb_path'], config['len_traj_pred'], config['traj_pace'],
                         config['history_length'], False, max_samples=max_samples, seed=seed, manifest=manifest)
        self.priors = PriorStore(config)

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        if worker is not None and worker.num_workers != 1:
            raise ValueError('LMDB already sharded by rank; use 0 or 1 loader worker')
        manifest = list(self.manifest)
        random.Random(self.seed + self.epoch).shuffle(manifest)
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=True, max_readers=256)
        yielded = 0
        floors = OrderedDict()
        try:
            with env.begin(buffers=True) as txn:
                for item in manifest:
                    if yielded >= self.max_samples:
                        break
                    floor_key = item['floor_key']
                    _, root, scene = floor_key.split('::')
                    if floor_key not in floors:
                        floors[floor_key] = torch.from_numpy(decode_record(bytes(txn.get(floor_key.encode())))['floorplan'].copy()).float()
                        if len(floors) > 8:
                            floors.popitem(last=False)
                    record = decode_record(bytes(txn.get(item['key'].encode())))
                    poses = record['poses'].astype(np.float64)
                    priors, masks = self.priors.trajectory(root, scene, record['traj_name'], poses)
                    expected_starts = np.arange(0, len(poses) - self.len_traj_pred + 1, self.traj_pace)
                    source_pace = int(self.priors.config.get('lmdb_source_traj_pace', 4))
                    source_horizon = int(self.priors.config.get('lmdb_source_len_traj_pred', 16))
                    cached_starts = np.arange(0, len(poses)-source_horizon+1, source_pace)
                    if not np.array_equal(cached_starts, record['current_indices']):
                        raise ValueError('LMDB record does not match declared source geometry')
                    if len(cached_starts) != item.get('source_sample_count', item['sample_count']):
                        raise ValueError('LMDB source manifest count differs from record')
                    if not np.isin(expected_starts, cached_starts).all():
                        raise ValueError('Requested starts are not a subset of cached starts')
                    if len(priors) != item['sample_count']:
                        raise ValueError('LMDB manifest sample count does not match the bridge geometry')
                    # The legacy record is a single compressed trajectory, so decoding
                    # decompresses all fields. Only RGB/poses become tensors or reach the GPU.
                    rgb = torch.from_numpy(record['rgb'].copy()).permute(0, 3, 1, 2).float() / 255
                    for slot, start in enumerate(expected_starts):
                        if yielded >= self.max_samples:
                            break
                        start = int(start)
                        indices = [max(0, i) for i in range(start - self.history_length, start)]
                        yielded += 1
                        yield pack_sample(rgb[indices], rgb[start], self._future_pose_tensor(poses, start),
                            self._point_goal_tensor(poses, start),
                            '%s::%s::%d' % (scene, record['traj_name'], start), priors[slot], masks[slot],
                            mark_floorplan(floors[floor_key], self.priors.planner(root, scene), poses[start, :2],
                                np.arctan2(poses[start, 3]-poses[start, 1], poses[start, 2]-poses[start, 0]), poses[-1, :2]), poses, start)
        finally:
            env.close()
        if yielded != self.max_samples:
            raise RuntimeError('LMDB rank exhausted early: %d != %d' % (yielded, self.max_samples))
