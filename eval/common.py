import json
"""Simulator helpers retained from the evaluated PointGoal source."""
import os
import re
import subprocess
from pathlib import Path
import numpy as np
import torch
import yaml
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_CONFIG_PATH = REPO_ROOT / 'config/locobot_floverse_nav.yaml'
DEFAULT_GIBSON_DATASET_ROOT = REPO_ROOT / 'iGibson/igibson/data/g_dataset'


def load_floor_height(scene_name, floor_id, dataset_root=DEFAULT_GIBSON_DATASET_ROOT):
    """Load the absolute world z height for one scene floor."""
    floors_file = os.path.join(dataset_root, scene_name, "floors.txt")
    if not os.path.isfile(floors_file):
        raise FileNotFoundError("Missing floor heights; check scene_assets: " + floors_file)
    with open(floors_file, "r") as f:
        heights = [float(line.strip()) for line in f.readlines() if line.strip()]
    if not 0 <= floor_id < len(heights):
        raise ValueError("Invalid floor index %s for %s" % (floor_id, floors_file))
    height = heights[floor_id]
    if not np.isfinite(height):
        raise ValueError("Non-finite floor height: " + floors_file)
    return height

def set_seed(seed):
    """Seed NumPy and PyTorch RNGs used during evaluation."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def stable_string_seed(value):
    """Return a deterministic integer offset derived from a string."""
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(value))

def resolve_gibson_device_id(device_id):
    """Map a logical CUDA index onto the EGL minor id used by iGibson."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        entries = [entry.strip() for entry in visible_devices.split(",") if entry.strip()]
        if device_id < len(entries):
            physical_device = entries[device_id]
            try:
                output = subprocess.check_output(["nvidia-smi", "-q", "-i", physical_device], text=True)
                for line in output.splitlines():
                    if "Minor Number" in line:
                        return line.split(":")[-1].strip()
            except Exception:
                return physical_device
    return str(device_id)

def build_env(scene_name, floor, device_id, scene_assets=DEFAULT_GIBSON_DATASET_ROOT):
    """Create one headless iGibson env pinned to the requested EGL device."""
    import igibson
    from igibson.envs.igibson_env import iGibsonEnv
    igibson.g_dataset_path = str(Path(scene_assets).resolve())
    os.environ["GIBSON_DEVICE_ID"] = resolve_gibson_device_id(device_id)
    os.environ.pop("DISPLAY", None)
    config_data = yaml.load(open(DEFAULT_ENV_CONFIG_PATH, "r"), Loader=yaml.FullLoader)
    config_data["enable_shadow"] = False
    config_data["enable_pbr"] = False
    config_data["scene_id"] = scene_name
    env = iGibsonEnv(config_file=config_data, mode="headless")
    floor_height = load_floor_height(scene_name, floor, scene_assets)
    robot_z = floor_height + config_data.get("initial_pos_z_offset", 0.1)
    return env, robot_z

def list_traj_ids(scenes_dir, scene_id):
    """List trajectory directories in numeric order."""
    trajs = [
        traj_id
        for traj_id in os.listdir(os.path.join(scenes_dir, scene_id))
        if os.path.isdir(os.path.join(scenes_dir, scene_id, traj_id))
    ]
    return sorted(trajs, key=lambda name: int(re.search(r"\d+", name).group()))

def list_scene_ids(scenes_dir, explicit_scene_ids=None, excluded_scene_ids=None):
    """Return valid scene ids from the evaluation directory, with optional filtering."""
    from datasets.data_selection import scene_selected
    excluded_scene_ids = set(excluded_scene_ids or [])
    if explicit_scene_ids:
        scene_ids = explicit_scene_ids
    else:
        scene_ids = os.listdir(scenes_dir)
    return sorted(
        scene_id
        for scene_id in scene_ids
        if os.path.isdir(os.path.join(scenes_dir, scene_id)) and scene_id not in excluded_scene_ids
        and scene_selected(scene_id, {"train_scene_dataset": "hm3d"})
    )


def read_object_goal(scenes_dir, scene_id, traj_id):
    """Read the object category for one object-goal trajectory."""
    obj_json_path = os.path.join(scenes_dir, scene_id, traj_id, "object", "object.json")
    with open(obj_json_path, "r") as f:
        obj_data = json.load(f)
    if isinstance(obj_data, list):
        obj_data = obj_data[0] if obj_data else {}
    return [obj_data.get("object_category", "null")]
