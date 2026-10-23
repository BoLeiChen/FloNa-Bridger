"""Metric robot-local maps: x forward, y left; image row forward, column right."""
from dataclasses import dataclass
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


@dataclass
class LocalMap:
    sdf: torch.Tensor             # [B,H,W], metres; positive outside obstacles
    known: torch.Tensor           # [B,H,W], bool; observed free OR occupied
    free_clearance: torch.Tensor  # distance to occupied, unknown or map boundary
    resolution: float
    x_min: float
    y_max: float

    @classmethod
    def from_masks(cls, occupied, known, resolution=0.1, x_min=-1.0, y_max=3.0):
        if occupied.shape != known.shape or occupied.ndim != 3 or min(occupied.shape) < 1 or min(occupied.shape[-2:]) < 2:
            raise ValueError("occupied and known must have shape [B,H,W]")
        if not np.isfinite([resolution, x_min, y_max]).all() or resolution <= 0:
            raise ValueError("resolution must be positive")
        occ = occupied.detach().cpu().numpy().astype(bool)
        obs = known.detach().cpu().numpy().astype(bool) | occ
        signed, clearance = [], []
        for o, k in zip(occ, obs):
            # Map extent is not a sensed wall. Unknown/boundary have separate costs.
            if o.any():
                outside = distance_transform_edt(~o)
                inside = distance_transform_edt(np.pad(o, 1, constant_values=False))[1:-1, 1:-1]
                signed.append((outside - inside) * resolution)
            else:
                signed.append(np.full(o.shape, np.hypot(*o.shape) * resolution))
            observed_free = np.pad(k & ~o, 1, constant_values=False)
            clearance.append(distance_transform_edt(observed_free)[1:-1, 1:-1] * resolution)
        kwargs = dict(device=occupied.device, dtype=torch.float32)
        return cls(torch.as_tensor(np.stack(signed), **kwargs),
                   torch.as_tensor(obs, device=occupied.device),
                   torch.as_tensor(np.stack(clearance), **kwargs),
                   resolution, x_min, y_max)

    def query(self, field, points):
        """Bilinear differentiable query. Return values and explicit out-of-map distance."""
        if points.ndim != 3 or points.shape[-1] != 2 or points.shape[0] != field.shape[0]:
            raise ValueError("points must be [B,N,2] matching map batch")
        h, w = field.shape[-2:]
        row = (points[..., 0] - self.x_min) / self.resolution
        col = (self.y_max - points[..., 1]) / self.resolution
        grid = torch.stack((2 * col / (w - 1) - 1, 2 * row / (h - 1) - 1), -1)
        values = F.grid_sample(field[:, None].to(points), grid[:, :, None],
                               align_corners=True, padding_mode="border")[:, 0, :, 0]
        outside = (F.relu(-row) + F.relu(row - h + 1) +
                   F.relu(-col) + F.relu(col - w + 1)) * self.resolution
        return values, outside


def camera_rotation(rpy):
    """Camera x-forward/y-left/z-up -> level robot frame, Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = map(float, rpy)
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def depth_to_local_map(depth, cfg):
    """Project radial depth with calibrated extrinsics and observed floor support.

    Only observed floor cells are free. A distant floor return does not prove
    that the unobserved ground before it has no hole/drop-off. Body-height hits
    block floor evidence farther away in the same horizontal angular bin.
    """
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError("depth must be [B,1,H,W]")
    units = cfg.get("depth_units", "uint8_range")
    if units not in ("uint8_range", "meters"):
        raise ValueError("unknown depth_units")
    raw = depth.detach().float().cpu().numpy()[:, 0]
    ranges = raw * (10.0 / 255.0) if units == "uint8_range" else raw
    b, h, w = ranges.shape
    fx, fy = float(cfg["fx"]), float(cfg["fy"])
    cx, cy = float(cfg["cx"]), float(cfg["cy"])
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0 or h != int(cfg["image_height"]) or w != int(cfg["image_width"]):
        raise ValueError("native depth resolution/intrinsics mismatch")
    res = float(cfg.get("resolution", 0.1))
    xmin, xmax = float(cfg.get("x_min", -1)), float(cfg.get("x_max", 5))
    ymin, ymax = float(cfg.get("y_min", -3)), float(cfg.get("y_max", 3))
    if not np.isfinite([res, xmin, xmax, ymin, ymax]).all() or res <= 0 or xmin >= xmax or ymin >= ymax:
        raise ValueError("invalid local map extent/resolution")
    nh, nw = round((xmax - xmin) / res) + 1, round((ymax - ymin) / res) + 1
    if min(nh, nw) < 3:
        raise ValueError("invalid local map extent")
    offset = np.asarray(cfg.get("camera_offset_xy", [0., 0.]), dtype=float)
    rotation = np.asarray(cfg.get("camera_rotation", camera_rotation(cfg.get("camera_rpy", [0., 0., 0.]))))
    if (offset.shape != (2,) or rotation.shape != (3, 3) or not np.isfinite(offset).all()
            or not np.isfinite(rotation).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)):
        raise ValueError("invalid camera extrinsics")
    yy, xx = np.mgrid[:h, :w]
    rays = np.stack((np.ones_like(xx), -(xx-cx)/fx, -(yy-cy)/fy), -1)
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    rays = rays @ rotation.T
    # Tilt/roll changes azimuth across image rows, so image columns are not rays
    # in the horizontal map. Bin transformed azimuth instead.
    bin_width = float(cfg.get("azimuth_bin_radians", 1.0 / fx))
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("invalid azimuth bin width")
    bins_count = int(np.ceil(2*np.pi/bin_width))
    bins = np.floor((np.arctan2(rays[..., 1], rays[..., 0]) + np.pi) / bin_width).astype(int) % bins_count
    occ, known = np.zeros((b, nh, nw), bool), np.zeros((b, nh, nw), bool)
    for batch in range(b):
        r = ranges[batch]
        valid = np.isfinite(r) & (r > 0.05) & (r < float(cfg.get("sensor_max_range", 10)) - 0.02)
        if valid.mean() < float(cfg.get("minimum_valid_fraction", 0.05)):
            continue
        xyz = np.where(valid, r, 0)[..., None] * rays
        height = cfg.get("camera_height")
        if height is None:
            samples = -xyz[..., 2][valid & (yy >= int(0.6 * h)) & (rays[..., 2] < -0.1)]
            height = float(np.quantile(samples, 0.9)) if samples.size else None
        if height is None or not np.isfinite(height) or not 0.1 <= float(height) <= 2.5:
            continue
        above_floor = float(height) + xyz[..., 2]
        floor = valid & (np.abs(above_floor) <= float(cfg.get("floor_tolerance", 0.08)))
        obstacle = valid & (above_floor > float(cfg.get("obstacle_min_height", 0.08))) & (
            above_floor < float(cfg.get("robot_height", 0.9)))
        horizontal_distance = np.linalg.norm(xyz[..., :2], axis=-1)
        nearest = np.full(bins_count, np.inf)
        np.minimum.at(nearest, bins[obstacle], horizontal_distance[obstacle])
        # Neighbor bins cover angular quantization at obstacle boundaries.
        nearest = np.minimum(nearest, np.minimum(np.roll(nearest, 1), np.roll(nearest, -1)))
        floor &= horizontal_distance < nearest[bins] - res
        robot_xy = xyz[..., :2] + offset
        gi = np.rint((robot_xy[..., 0] - xmin) / res).astype(int)
        gj = np.rint((ymax - robot_xy[..., 1]) / res).astype(int)
        inside = (gi >= 0) & (gi < nh) & (gj >= 0) & (gj < nw)
        for mask, output in ((floor, known), (obstacle, occ)):
            use = mask & inside
            output[batch, gi[use], gj[use]] = True
        # No depth-validity shortcut grants free space around the robot. Initial
        # footprint/blind spots need actual observations or an explicit supplied map.
        known[batch] |= occ[batch]
    return LocalMap.from_masks(torch.as_tensor(occ, device=depth.device),
                               torch.as_tensor(known, device=depth.device), res, xmin, ymax)


def interpolate_segments(trajectory, subdivisions=8):
    if trajectory.ndim != 3 or trajectory.shape[-1] != 2 or trajectory.shape[1] < 2:
        raise ValueError("trajectory must be [B,H>=2,2]")
    if subdivisions < 1:
        raise ValueError("subdivisions must be positive")
    t = torch.linspace(0, 1, subdivisions + 1, device=trajectory.device, dtype=trajectory.dtype)
    points = trajectory[:, :-1, None] * (1 - t[None, None, :, None])
    points = points + trajectory[:, 1:, None] * t[None, None, :, None]
    return points.flatten(1, 2)


@torch.no_grad()
def fuse_local_maps(maps, poses):
    """Fuse static observations into the first map's coordinate frame.

    poses[k] is [B,3] = source-frame (x,y,yaw) in the target frame.
    Occlusion is handled during ray carving. Occupied evidence wins over free.
    This interface also supports odometry-aligned historical depth observations.
    """
    if not maps or len(maps) != len(poses):
        raise ValueError("supply one source pose per map")
    target = maps[0]
    b, h, w = target.sdf.shape
    rows, cols = torch.meshgrid(torch.arange(h, device=target.sdf.device),
                                torch.arange(w, device=target.sdf.device), indexing="ij")
    points = torch.stack((target.x_min + rows * target.resolution,
                          target.y_max - cols * target.resolution), -1).float()
    points = points.reshape(1, -1, 2).expand(b, -1, -1)
    occupied = torch.zeros_like(target.known)
    known = torch.zeros_like(target.known)
    for local, pose in zip(maps, poses):
        pose = pose.to(points)
        shifted = points - pose[:, None, :2]
        cosine, sine = pose[:, 2:3].cos(), pose[:, 2:3].sin()
        query = torch.stack((cosine * shifted[..., 0] + sine * shifted[..., 1],
                              -sine * shifted[..., 0] + cosine * shifted[..., 1]), -1)
        seen, outside = local.query(local.known.float(), query)
        hits, _ = local.query((local.sdf < 0).float(), query)
        inside = outside == 0
        known |= ((seen > 0.999) & inside).reshape(b, h, w)
        occupied |= ((hits > 0.001) & inside).reshape(b, h, w)
    return LocalMap.from_masks(occupied, known | occupied, target.resolution, target.x_min, target.y_max)


class ObservationMapMemory:
    """Fuse actual past depth maps with measured planar poses (static scenes).

    Never fills an initial footprint or unseen cells. Retains original maps to
    avoid repeated raster resampling, and bounds memory per episode.
    """
    def __init__(self, capacity=32):
        if capacity < 1:
            raise ValueError("map memory capacity must be positive")
        self.frames = deque(maxlen=capacity)

    def add(self, local_map, world_pose):
        pose = torch.as_tensor(world_pose, device=local_map.sdf.device,
                               dtype=local_map.sdf.dtype).reshape(-1, 3)
        if pose.shape[0] != local_map.sdf.shape[0] or not torch.isfinite(pose).all():
            raise ValueError("one finite world pose is required for each map")
        self.frames.append((local_map, pose.clone()))

    def current(self):
        if not self.frames:
            raise ValueError("map memory is empty")
        target, origin = self.frames[-1]
        maps, poses = [target], [torch.zeros_like(origin)]
        for local, world in list(self.frames)[:-1]:
            shift = world[:, :2] - origin[:, :2]
            cosine, sine = origin[:, 2].cos(), origin[:, 2].sin()
            relative = torch.stack((cosine*shift[:, 0] + sine*shift[:, 1],
                                    -sine*shift[:, 0] + cosine*shift[:, 1],
                                    world[:, 2] - origin[:, 2]), -1)
            maps.append(local)
            poses.append(relative)
        return fuse_local_maps(maps, poses)
