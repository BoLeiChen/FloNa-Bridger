import os
import sys
import io
from typing import Dict, List, Tuple

import numpy as np
import tqdm
import lmdb
import torch
from torch.utils.data import Dataset
from PIL import Image

from .data_utils import (
    astar_path_on_walkable_map,
    floorplan_to_walkable_map,
    img_path_to_data,
    img_path_to_data_and_point_transfer,
    get_data_path,
    metric_positions_to_resized_pixels,
    resized_pixels_to_metric_positions,
    sample_path_prefix_points,
    path_xy_to_local_condition,
    to_local_coords,
)

DATASET_INDEX_VERSION = "terminalpad_v1"
PLANNER_CACHE_VERSION = "v3_turnmask"

class flona_Dataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        trav_folder: str,   
        scene_names: List[str],
        obs_image_size: Tuple[int, int] = (224, 224),      # 观测图像 for DINOv2
        floorplan_image_size: Tuple[int, int] = (96, 96),  # 平面图 for EfficientNet
        waypoint_spacing: int = 1,
        len_traj_pred: int = 8,
        context_size: int = 5,
        end_slack: int = 0,
        normalize: bool = True,
        planner_floorplan_size: Tuple[int, int] = (256, 256),
        planner_free_thresh: float = 0.96,
    ):
        self.data_folder = data_folder
        self.trav_folder = trav_folder
        self.scene_names = scene_names
        self.obs_image_size = obs_image_size
        self.floorplan_image_size = floorplan_image_size
        self.waypoint_spacing = waypoint_spacing
        self.len_traj_pred = len_traj_pred
        self.context_size = context_size
        self.end_slack = end_slack
        self.normalize = normalize
        self.planner_floorplan_size = planner_floorplan_size
        self.planner_free_thresh = planner_free_thresh
        self.metric_waypoint_spacing = 0.045 #0.255    # 0.045
        self.metric_step_size = self.metric_waypoint_spacing * self.waypoint_spacing
        self._image_caches = {}
        self.traj_names = {}    # {scene_name: [traj_name]}
        for scene_name in scene_names:
            self.traj_names[scene_name] = []
            for file in os.listdir(os.path.join(data_folder, scene_name)):
                if os.path.isdir(os.path.join(data_folder, scene_name, file)) and file.startswith("traj"):
                    self.traj_names[scene_name].append(file)
            self.traj_names[scene_name].sort()
        print('traj_names produced and size ', sys.getsizeof(self.traj_names))
        self.trajectory_cache = {} # {scene_name: {traj_name: traj_data}}
        self.floor_shapes_ori = np.load(os.path.join(trav_folder, "floor_shapes.npy"), allow_pickle=True).item()
        self.num_action_params = 2
        self._load_index()
        self._build_caches()
        self.floorplan_cache = {}
        self.planner_split = os.path.basename(os.path.normpath(self.data_folder))
        self.planner_floorplan_root = os.path.join("datasets", "scenes_117", self.planner_split)
        self.astar_local_cond_cache: Dict[str, np.ndarray] = {}
        self.walkable_map_cache: Dict[str, np.ndarray] = {}
        self._build_astar_local_cond_caches()
        self.travers_map_cache: Dict[str, np.ndarray] = {}

    def _build_caches(self, use_tqdm: bool = True):
        """
        Build a cache of images for faster loading using LMDB.
        Only creates the files, does not keep them open to avoid pickling errors.
        """
        print("Building LMDB cache")
        # Ensure caches are empty initially to avoid pickling issues
        self._image_caches = {}
        
        # Load all the trajectories into memory. These should already be loaded, but just in case.
        for scene_name in self.scene_names:
            for traj_name in self.traj_names[scene_name]:
                self._get_trajectory(scene_name, traj_name)
        print('trajectories loaded and size ', sys.getsizeof(self.trajectory_cache))
        
        for scene_name in self.scene_names:
            cache_filename = os.path.join(self.data_folder, scene_name, f"dataset_{scene_name}_new.lmdb")
            if not os.path.exists(cache_filename):
                start_idx = self.scene_start_index[self.scene_names.index(scene_name)]
                end_idx = self.scene_end_index[self.scene_names.index(scene_name)]
                if start_idx < 0 or end_idx < 0:
                    continue
                base_idx = start_idx
                tqdm_iterator = tqdm.tqdm(
                    self.index_to_data_str[start_idx:end_idx + 1],
                    disable=not use_tqdm,
                    dynamic_ncols=True,
                    desc=f"Building LMDB cache for {scene_name}",
                )
                with lmdb.open(cache_filename, map_size=2**40) as image_cache:
                    with image_cache.begin(write=True) as txn:
                        for idx, scene_name_and_traj_name in enumerate(tqdm_iterator):
                            sn, traj_name = scene_name_and_traj_name.split('-')
                            time = self.index_to_data_int[idx + base_idx, 0]
                            image_path = get_data_path(os.path.join(self.data_folder, scene_name), traj_name, time)
                            with open(image_path, "rb") as f:
                                txn.put(image_path.encode(), f.read())
                            
                        trajs = self.traj_names[scene_name]
                        for traj_name in trajs:
                            image_path_0 = get_data_path(os.path.join(self.data_folder, scene_name), traj_name, 0)
                            image_path_1 = get_data_path(os.path.join(self.data_folder, scene_name), traj_name, 1) 
                            image_path_2 = get_data_path(os.path.join(self.data_folder, scene_name), traj_name, 2)
                            with open(image_path_0, "rb") as f:
                                txn.put(image_path_0.encode(), f.read())
                            with open(image_path_1, "rb") as f:
                                txn.put(image_path_1.encode(), f.read())
                            with open(image_path_2, "rb") as f:
                                txn.put(image_path_2.encode(), f.read())                      
                        image_path = get_data_path(self.data_folder, scene_name, "floorplan")
                        with open(image_path, "rb") as f:
                            txn.put(image_path.encode(), f.read())

    def _get_lmdb_env(self, scene_name):
        """Helper to lazy-load LMDB environment for a scene."""
        if scene_name not in self._image_caches:
            cache_filename = os.path.join(self.data_folder, scene_name, f"dataset_{scene_name}_new.lmdb")
            # Open in read-only mode, with no lock (faster for multi-process reading)
            self._image_caches[scene_name] = lmdb.open(cache_filename, readonly=True, lock=False)
        return self._image_caches[scene_name]

    def _build_index(self, use_tqdm: bool = False):
        """
        Build an index consisting of tuples (trajectory name, time)
        """
        samples_index = []
        skip_k = []
        goals_pos = []
        scene_start_index = []
        for scene_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            trajs = self.traj_names[scene_name]
            scene_start_index.append(len(samples_index))
            for traj_name in trajs:
                scene_name_traj_name = scene_name + '-' + traj_name
                # print(scene_name_traj_name)
                traj_data = self._get_trajectory(scene_name, traj_name)
                traj_len = traj_data.shape[0]
                
                # Use fixed k=5 for sparse sampling per paper. Near-goal
                # samples are still included; missing future steps are padded
                # with the terminal pose in _compute_actions.
                k = 5
                begin_time = self.context_size * self.waypoint_spacing
                goal_time = traj_len - 1 - self.end_slack
                min_frames_needed = begin_time + self.end_slack + 1
                
                if traj_len < min_frames_needed:
                    continue
                    
                goal_pos = traj_data[goal_time, :2].copy()
                goals_pos.append(goal_pos)
                for curr_time in range(begin_time, goal_time + 1, k * self.waypoint_spacing):
                    samples_index.append((scene_name_traj_name, curr_time, goal_time))
                    skip_k.append(k)

        scene_end_index = []
        for i in range(len(scene_start_index) - 1):
            scene_end_index.append(scene_start_index[i + 1] - 1)
        scene_end_index.append(len(samples_index) - 1)
        return samples_index, skip_k, goals_pos, scene_start_index, scene_end_index

    def _load_index(self) -> None:
        """
        Generates a list of tuples of (obs_traj_name, goal_traj_name, obs_time, goal_time) for each observation in the dataset
        """
        index_to_data_path = os.path.join(
            self.data_folder,
            f"dataset_n{self.context_size}_slack_{self.end_slack}_{DATASET_INDEX_VERSION}.npz",
        )
        try:
            # load the index_to_data if it already exists (to save time)
            npz = np.load(index_to_data_path, allow_pickle=True)
            # self.index_to_data = npz["index_to_data"]
            index_to_data = npz["index_to_data"]
            skip_k = npz["skip_k"]
            self.index_to_data_str = index_to_data[:,0].copy()
            self.index_to_data_int = index_to_data[:,1:].copy().astype(np.int16)
            self.skip_k = skip_k.copy().astype(np.int16)

            del index_to_data
            del skip_k
            self.goals_pos = npz["goals_pos"]
            print('Index loaded, index_to_data size', sys.getsizeof(self.index_to_data_str))
            self._filter_index_to_selected_scenes()
            self._rebuild_scene_metadata()

        except:
            # if the index_to_data file doesn't exist, create it
            print('Building dataset index.')
            index_to_data, skip_k, self.goals_pos, self.scene_start_index, self.scene_end_index = self._build_index()

            index_to_data = np.array(index_to_data, dtype=object)
            skip_k = np.array(skip_k, dtype=object)
            self.goals_pos = np.array(self.goals_pos, dtype=object)
            self.index_to_data_str = index_to_data[:,0].copy()
            self.index_to_data_int = index_to_data[:,1:].copy().astype(np.int16)
            self.skip_k = skip_k.copy().astype(np.int16)
            self._rebuild_scene_metadata()

            np.savez(index_to_data_path, index_to_data=index_to_data, skip_k=skip_k, goals_pos=self.goals_pos)
            print('get index')
            del index_to_data
            del skip_k

    def _filter_index_to_selected_scenes(self) -> None:
        scene_set = set(self.scene_names)
        mask = np.array([
            entry.split('-')[0] in scene_set
            for entry in self.index_to_data_str
        ], dtype=bool)
        self.index_to_data_str = self.index_to_data_str[mask]
        self.index_to_data_int = self.index_to_data_int[mask]
        self.skip_k = self.skip_k[mask]

    def _rebuild_scene_metadata(self) -> None:
        self.scene_start_index = []
        self.scene_end_index = []
        self.scene_global_indices = {}
        self.scene_local_indices = np.full(len(self.index_to_data_str), -1, dtype=np.int32)
        for scene_name in self.scene_names:
            indices = np.where(
                np.char.startswith(self.index_to_data_str.astype(str), f"{scene_name}-")
            )[0]
            self.scene_global_indices[scene_name] = indices
            if len(indices) == 0:
                self.scene_start_index.append(-1)
                self.scene_end_index.append(-1)
                continue
            self.scene_start_index.append(int(indices[0]))
            self.scene_end_index.append(int(indices[-1]))
            self.scene_local_indices[indices] = np.arange(len(indices), dtype=np.int32)

    def _planner_cache_path(self, scene_name: str) -> str:
        return os.path.join(
            self.data_folder,
            scene_name,
            (
                f"astar_local_cond_floorplan_v1_{scene_name}"
                f"_{PLANNER_CACHE_VERSION}"
                f"_pyastar2d"
                f"_ctx{self.context_size}_pred{self.len_traj_pred}_slack{self.end_slack}"
                f"_planner{self.planner_floorplan_size[0]}x{self.planner_floorplan_size[1]}.npy"
            ),
        )

    def _get_future_segment_with_terminal_pad(
        self,
        traj_data: np.ndarray,
        curr_time: int,
        goal_time: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        target_len = self.len_traj_pred + 1
        stop_index = min(
            curr_time + self.len_traj_pred * self.waypoint_spacing + 1,
            goal_time + 1,
        )
        segment = traj_data[curr_time:stop_index:self.waypoint_spacing].copy()
        if segment.shape[0] == 0:
            raise IndexError(f"Empty future segment for curr_time={curr_time}, goal_time={goal_time}")

        if segment.shape[0] < target_len:
            goal_state = traj_data[goal_time:goal_time + 1].copy()
            pad = np.repeat(goal_state, target_len - segment.shape[0], axis=0)
            segment = np.concatenate([segment, pad], axis=0)
        else:
            segment = segment[:target_len]

        positions = segment[:, :2].copy()
        yaw = segment[:, 2:].copy()
        return positions, yaw

    def _get_planner_floorplan_path(self, scene_name: str) -> str:
        return os.path.join(self.planner_floorplan_root, scene_name, "floorplan.png")

    def _get_walkable_map(self, scene_name: str) -> np.ndarray:
        if scene_name not in self.walkable_map_cache:
            floorplan_path = self._get_planner_floorplan_path(scene_name)
            self.walkable_map_cache[scene_name] = floorplan_to_walkable_map(
                floorplan_path,
                free_thresh=self.planner_free_thresh,
            )
        return self.walkable_map_cache[scene_name]

    def _compute_astar_local_cond_metric(
        self,
        scene_name: str,
        traj_name: str,
        curr_pos_metric: np.ndarray,
        goal_pos_metric: np.ndarray,
        cur_ori_metric: np.ndarray,
    ) -> np.ndarray:
        walkable_map = self._get_walkable_map(scene_name)
        metric_size_cm = self.floor_shapes_ori[scene_name]
        start_xy = np.round(
            metric_positions_to_resized_pixels(curr_pos_metric, metric_size_cm, self.planner_floorplan_size)
        ).astype(np.int32)
        goal_xy = np.round(
            metric_positions_to_resized_pixels(goal_pos_metric, metric_size_cm, self.planner_floorplan_size)
        ).astype(np.int32)
        path_xy = astar_path_on_walkable_map(walkable_map, start_xy, goal_xy)
        planner_succeeded = len(path_xy) > 0
        if not planner_succeeded:
            path_xy = np.stack([start_xy, goal_xy], axis=0)
        path_metric = resized_pixels_to_metric_positions(path_xy, metric_size_cm, self.planner_floorplan_size)
        sampled_metric, valid_mask = sample_path_prefix_points(
            path_metric,
            num_points=self.len_traj_pred,
            step_size_m=self.metric_step_size,
            return_valid_mask=True,
        )
        if not planner_succeeded:
            valid_mask.fill(0.0)
        astar_local_xy = to_local_coords(sampled_metric, curr_pos_metric, cur_ori_metric)
        return path_xy_to_local_condition(astar_local_xy, valid_mask)

    def _build_astar_local_cond_caches(self, use_tqdm: bool = True) -> None:
        for scene_name in self.scene_names:
            cache_path = self._planner_cache_path(scene_name)
            if os.path.exists(cache_path):
                continue
            indices = self.scene_global_indices.get(scene_name, np.array([], dtype=np.int32))
            if len(indices) == 0:
                continue
            scene_cache = np.zeros((len(indices), self.len_traj_pred, 5), dtype=np.float32)
            iterator = tqdm.tqdm(
                enumerate(indices),
                total=len(indices),
                disable=not use_tqdm,
                dynamic_ncols=True,
                desc=f"Building A* local-cond cache for {scene_name}",
            )
            for local_idx, global_idx in iterator:
                scene_name_cur_and_f_curr = self.index_to_data_str[global_idx]
                curr_time, goal_time = self.index_to_data_int[global_idx]
                _, traj_name = scene_name_cur_and_f_curr.split('-')
                traj_data = self._get_trajectory(scene_name, traj_name)
                goal_pos_metric = traj_data[goal_time, :2].copy()
                cur_pos_metric = traj_data[curr_time, :2].copy()
                cur_ori_metric = traj_data[curr_time, 2:].copy()
                cur_ori_metric = cur_pos_metric + (cur_ori_metric - cur_pos_metric) / np.linalg.norm(cur_ori_metric - cur_pos_metric)
                scene_cache[local_idx] = self._compute_astar_local_cond_metric(
                    scene_name,
                    traj_name,
                    cur_pos_metric,
                    goal_pos_metric,
                    cur_ori_metric,
                )
            np.save(cache_path, scene_cache)

    def _get_astar_local_cond(self, scene_name: str, global_index: int) -> np.ndarray:
        if scene_name not in self.astar_local_cond_cache:
            self.astar_local_cond_cache[scene_name] = np.load(self._planner_cache_path(scene_name), mmap_mode="r")
        local_index = self.scene_local_indices[global_index]
        return np.array(self.astar_local_cond_cache[scene_name][local_index], dtype=np.float32)

    def _get_travers_map_path(self, scene_name: str) -> str:
        scene_id, floor_str = scene_name.rsplit("_", 1)
        floor = int(floor_str)
        if floor == 0:
            scene_dir = os.path.join(self.trav_folder, scene_id)
            if not os.path.isdir(scene_dir):
                scene_dir = os.path.join(self.trav_folder, scene_name)
        else:
            scene_dir = os.path.join(self.trav_folder, scene_name)
            if not os.path.isdir(scene_dir):
                scene_dir = os.path.join(self.trav_folder, scene_id)
        return os.path.join(scene_dir, f"floor_trav_test_{floor}_modified.png")

    def get_travers_map(self, scene_name: str) -> np.ndarray:
        if scene_name not in self.travers_map_cache:
            travers_map_path = self._get_travers_map_path(scene_name)
            self.travers_map_cache[scene_name] = np.array(
                Image.open(travers_map_path).convert("L")
            )
        return self.travers_map_cache[scene_name]


    def _load_image(self, scene_name, trajectory_name, name): 
        if name == "floorplan":
            image_size = self.floorplan_image_size
            image_path = get_data_path(self.data_folder, scene_name, name)
        else:
            image_size = self.obs_image_size  # 观测图像使用 224x224 for DINOv2
            image_path = get_data_path(os.path.join(self.data_folder, scene_name), trajectory_name, name)
        
        try:
            # Try loading from LMDB cache first using lazy loader
            lmdb_env = self._get_lmdb_env(scene_name)
            with lmdb_env.begin() as txn:
                image_data = txn.get(image_path.encode())
                if image_data is not None:
                    with io.BytesIO(image_data) as f:
                        result = img_path_to_data(f, image_size)
                    return result

            # Fallback to loading from disk
            with open(image_path, "rb") as f:
                result = img_path_to_data(f, image_size)

            return result
            
        except TypeError:
            print(f"Failed to load image {image_path}")
            
    def _load_image_and_transform_points(self, scene_name, trajectory_name, cur_pos, goal_pos, cur_ori, name):
        cur_pos_metric = cur_pos * self.metric_waypoint_spacing * self.waypoint_spacing # trans from waypoints to meters
        goal_pos_metric = goal_pos * self.metric_waypoint_spacing * self.waypoint_spacing
        cur_ori_metric = cur_ori * self.metric_waypoint_spacing * self.waypoint_spacing

        # 平面图使用 floorplan_image_size
        image_size = self.floorplan_image_size

        if name == "floorplan":
            # Check in-memory cache for floorplan
            if scene_name in self.floorplan_cache:
                img_bytes = self.floorplan_cache[scene_name]
                with io.BytesIO(img_bytes) as f:
                    result = img_path_to_data_and_point_transfer(f, self.floor_shapes_ori[scene_name], image_size, cur_pos_metric, goal_pos_metric, cur_ori_metric)
                return result

            image_path = get_data_path(self.data_folder, scene_name, name)
        else:
            image_path = get_data_path(os.path.join(self.data_folder, scene_name), trajectory_name, name)

        try:
            image_data = None
            # Try loading from LMDB cache first using lazy loader
            lmdb_env = self._get_lmdb_env(scene_name)
            with lmdb_env.begin() as txn:
                image_data = txn.get(image_path.encode())
            
            # If found in LMDB
            if image_data is not None:
                if name == "floorplan":
                    self.floorplan_cache[scene_name] = image_data # Cache to memory
                with io.BytesIO(image_data) as f:
                    result = img_path_to_data_and_point_transfer(f, self.floor_shapes_ori[scene_name], image_size, cur_pos_metric, goal_pos_metric, cur_ori_metric)
                return result

            # Fallback to loading from disk
            with open(image_path, "rb") as f:
                image_data = f.read()
                if name == "floorplan":
                     self.floorplan_cache[scene_name] = image_data # Cache to memory
                with io.BytesIO(image_data) as f_obj:
                    result = img_path_to_data_and_point_transfer(f_obj, self.floor_shapes_ori[scene_name], image_size, cur_pos_metric, goal_pos_metric, cur_ori_metric)
            return result
        except TypeError:
            print(f"Failed to load image {image_path}")
            
    def _compute_actions(self, traj_data, curr_time, goal_time):
        positions, yaw = self._get_future_segment_with_terminal_pad(traj_data, curr_time, goal_time)
        goal_pos = traj_data[goal_time, :2].copy()
        cur_pos = positions[0]
        cur_ori = yaw[0]
        cur_ori = cur_pos + (cur_ori - cur_pos) / np.linalg.norm(cur_ori - cur_pos)
        waypoints = to_local_coords(positions, positions[0], yaw[0])
        goal_pos_local = to_local_coords(goal_pos, positions[0], yaw[0])
        cur_pos_local = to_local_coords(cur_pos, positions[0], yaw[0])
        assert waypoints.shape == (self.len_traj_pred + 1, 2), f"{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal"
        actions = waypoints[1:]
        
        if self.normalize:
            # Modified: normalize by dense step size (removed k)
            actions[:, :2] /= self.metric_waypoint_spacing * self.waypoint_spacing  # transform meters to waypoints(steps)
            goal_pos /= self.metric_waypoint_spacing * self.waypoint_spacing
            cur_pos /= self.metric_waypoint_spacing * self.waypoint_spacing
            cur_ori /= self.metric_waypoint_spacing * self.waypoint_spacing
            goal_pos_local /= self.metric_waypoint_spacing * self.waypoint_spacing

        assert actions.shape == (self.len_traj_pred, self.num_action_params), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"

        return actions, goal_pos, cur_pos, cur_ori, goal_pos_local, cur_pos_local
    
    def _compute_actions_ori(self, k, traj_data, curr_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * k * self.waypoint_spacing + 1
        yaw = traj_data[start_index:end_index:(k * self.waypoint_spacing), 2:].copy()
        positions = traj_data[start_index:end_index:(k * self.waypoint_spacing), :2].copy()
        goal_pos = traj_data[-1 - self.end_slack, :2].copy()
        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])
        waypoints = to_local_coords(positions, positions[0], yaw[0])
        cur_pos = waypoints[0]
       
        assert waypoints.shape == (self.len_traj_pred + 1, 2), f"{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        actions = waypoints[1:]
        if self.normalize:
            actions[:, :2] /= self.metric_waypoint_spacing * k * self.waypoint_spacing  # transform meters to waypoints(steps)
            goal_pos /= self.metric_waypoint_spacing * k * self.waypoint_spacing
            cur_pos /= self.metric_waypoint_spacing * k * self.waypoint_spacing

        assert actions.shape == (self.len_traj_pred, self.num_action_params), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"

        return actions, goal_pos, cur_pos
    
    def _get_trajectory(self, scene_name, trajectory_name):
        k = scene_name+'-'+trajectory_name
        if k in self.trajectory_cache:
            return self.trajectory_cache[k]
        else:
            with open(os.path.join(self.data_folder, scene_name, trajectory_name, trajectory_name + ".npy"), "rb") as f:
                traj_data = np.load(f, allow_pickle=True)
            self.trajectory_cache[k] = traj_data
            return traj_data

    def __len__(self) -> int:
        return len(self.index_to_data_str)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        scene_name_cur_and_f_curr = self.index_to_data_str[i]
        curr_time, goal_time = self.index_to_data_int[i]
        scene_name_cur, f_curr = scene_name_cur_and_f_curr.split('-')
        # Load images
        context = []
        # sample the last self.context_size times from interval [0, curr_time)
        context_step = self.waypoint_spacing
        context_times = list(
            range(
                curr_time - self.context_size * self.waypoint_spacing,
                curr_time + 1,
                context_step,
            )
        )
        context = [(f_curr, t) for t in context_times]

        obs_image = torch.cat([
            self._load_image(scene_name_cur, f, t) for f, t in context
        ])
        # Load actions and current position and goal position
        curr_traj_data = self._get_trajectory(scene_name_cur, f_curr) # meters metric
        curr_traj_len = curr_traj_data.shape[0]
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"
        actions, goal_pos, cur_pos, cur_ori, goal_pos_local, cur_pos_local = self._compute_actions(curr_traj_data, curr_time, goal_time)   # waypoints(steps) metric in local
        astar_local_cond = self._get_astar_local_cond(scene_name_cur, i)
        # Load goal image
        floorplan_image, cur_pos_resized, goal_pos_resized, cur_ori_resized = self._load_image_and_transform_points(scene_name_cur, f_curr, cur_pos, goal_pos, cur_ori, "floorplan")

        # Compute distances
        # distance = (len(curr_traj_data) - self.end_slack - curr_time) // self.waypoint_spacing
        distance = (goal_time - curr_time) // (self.waypoint_spacing)
        distance = np.log1p(distance) # Apply log(1 + distance) for normalization
        # assert (len(curr_traj_data) - self.end_slack - curr_time) % self.waypoint_spacing == 0, f"{len(curr_traj_data) - self.end_slack} and {curr_time} should be separated by an integer multiple of {k * self.waypoint_spacing}"
        return (
            torch.as_tensor(obs_image, dtype=torch.float32),   # [3*L, H, W]
            torch.as_tensor(floorplan_image, dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.float32),           # waypoints(steps) metric in local
            torch.as_tensor(distance, dtype=torch.float32), # Changed to float32 for log values
            torch.as_tensor(goal_pos, dtype=torch.float32),
            torch.as_tensor(cur_pos, dtype=torch.float32),
            torch.as_tensor(cur_ori, dtype=torch.float32),
            torch.as_tensor(goal_pos_resized, dtype=torch.float32),
            torch.as_tensor(cur_pos_resized, dtype=torch.float32),
            torch.as_tensor(cur_ori_resized, dtype=torch.float32),
            torch.as_tensor(goal_pos_local, dtype=torch.float32),
            torch.as_tensor(cur_pos_local, dtype=torch.float32),
            torch.as_tensor(astar_local_cond, dtype=torch.float32),
            scene_name_cur,
        )
