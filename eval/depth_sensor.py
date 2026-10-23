"""Native radial depth and measured camera calibration; no simulator map query."""
from pathlib import Path
import numpy as np
import pybullet as p
import torch
from model.bridge_guidance.geometry import depth_to_local_map


def observed_map(env, scene_assets, scene_name, floor_level, config, device):
    renderer = env.simulator.renderer
    xyz = renderer.render_robot_cameras(modes=('3d',), cache=False)[0][:, :, :3]
    # iGibson 3d buffers contain camera-space metric XYZ; norm is radial range.
    radial = np.linalg.norm(xyz, axis=-1)
    radial[~np.isfinite(xyz).all(-1)] = 0
    depth = torch.from_numpy(radial.copy()).to(device=device, dtype=torch.float32)[None, None]
    k = renderer.get_intrinsics()
    robot = env.robots[0]
    yaw = float(robot.get_rpy()[2])
    c, s = np.cos(yaw), np.sin(yaw)
    level = np.array([[c,-s,0], [s,c,0], [0,0,1]])
    camera = np.asarray(p.getMatrixFromQuaternion(robot.eyes.get_orientation())).reshape(3,3)
    offset = level.T @ (robot.eyes.get_position()-robot.get_position())
    heights = [float(v) for v in (Path(scene_assets)/scene_name/'floors.txt').read_text().splitlines() if v.strip()]
    mapping = dict(config['map'], depth_units='meters', image_height=radial.shape[0],
        image_width=radial.shape[1], fx=float(k[0,0]), fy=float(k[1,1]), cx=float(k[0,2]), cy=float(k[1,2]),
        camera_rotation=(level.T@camera).tolist(), camera_offset_xy=offset[:2].tolist(),
        camera_height=float(robot.eyes.get_position()[2]-heights[floor_level]))
    local_map = depth_to_local_map(depth, mapping)
    return local_map, dict(calibration=mapping, valid_fraction=float(((radial>.05)&(radial<9.98)).mean()),
        occupied_cells=int((local_map.sdf<0).sum()), known_cells=int(local_map.known.sum()))
