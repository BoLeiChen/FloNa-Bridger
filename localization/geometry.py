"""SE(2) and FloVerse/CF2Loc map-coordinate conversions."""

import math

import numpy as np


def wrap_angle(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def relative_odometry(previous_pose, current_pose):
    """Return current motion [forward, left, yaw] in the previous body frame."""
    previous = np.asarray(previous_pose, dtype=np.float64)
    current = np.asarray(current_pose, dtype=np.float64)
    delta = current[:2] - previous[:2]
    c, s = math.cos(previous[2]), math.sin(previous[2])
    return np.asarray(
        [
            c * delta[0] + s * delta[1],
            -s * delta[0] + c * delta[1],
            wrap_angle(current[2] - previous[2]),
        ],
        dtype=np.float64,
    )


def apply_odometry(pose, odometry):
    pose = np.asarray(pose, dtype=np.float64)
    odometry = np.asarray(odometry, dtype=np.float64)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.asarray(
        [
            pose[0] + c * odometry[0] - s * odometry[1],
            pose[1] + s * odometry[0] + c * odometry[1],
            wrap_angle(pose[2] + odometry[2]),
        ],
        dtype=np.float64,
    )


def pose_errors(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    return (
        float(np.linalg.norm(first[:2] - second[:2])),
        abs(math.degrees(wrap_angle(first[2] - second[2]))),
    )


def blend_poses(predicted, measured, gain):
    predicted = np.asarray(predicted, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    gain = float(np.clip(gain, 0.0, 1.0))
    xy = predicted[:2] + gain * (measured[:2] - predicted[:2])
    yaw = wrap_angle(predicted[2] + gain * wrap_angle(measured[2] - predicted[2]))
    return np.asarray([xy[0], xy[1], yaw], dtype=np.float64)


def world_to_cf2loc_pose(world_pose, scale, offset_xy, map_resolution_m):
    """Match prepare_floverse_hm3d.py's isotropic 2 cm map conversion."""
    world_pose = np.asarray(world_pose, dtype=np.float64)
    offset_xy = np.asarray(offset_xy, dtype=np.float64)
    xy = world_pose[:2] / float(map_resolution_m) + offset_xy / (
        float(scale) * float(map_resolution_m)
    )
    return np.asarray([xy[0], xy[1], wrap_angle(world_pose[2])], dtype=np.float64)


def cf2loc_to_world_pose(map_pose, scale, offset_xy, map_resolution_m):
    map_pose = np.asarray(map_pose, dtype=np.float64)
    offset_xy = np.asarray(offset_xy, dtype=np.float64)
    xy = map_pose[:2] * float(map_resolution_m) - offset_xy / float(scale)
    return np.asarray([xy[0], xy[1], wrap_angle(map_pose[2])], dtype=np.float64)
