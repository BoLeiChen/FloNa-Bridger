"""Protocol helpers frozen from server original FloVerse on 2026-09-07.

Preserve the original reference-based scorer and collision fallback semantics.
See POINTGOAL_BRIDGE.md for deliberate historical metric conventions.
"""
import numpy as np
import pybullet as p
_DEFAULT_FALLBACK_PATH_INDEX = 0
PROTOCOL = 'floverse-original-pointgoal-v1'

def choose_fallback_path_index(path_length, fallback_path_index=_DEFAULT_FALLBACK_PATH_INDEX):
    """Select a fallback waypoint along the shortest path."""
    return min(fallback_path_index, max(path_length - 1, 0))

def get_pose_from_position(predicted_position):
    """Convert a sequence of global XY positions into poses and per-step yaw values."""
    delta = predicted_position[1:] - predicted_position[:-1]
    yaws = np.arctan2(delta[:, 1], delta[:, 0])
    quaternions = np.array([p.getQuaternionFromEuler([0, 0, yaw]) for yaw in yaws])
    poses = np.concatenate([predicted_position[1:], quaternions], axis=1)
    return poses, yaws

def move_to_fallback(env, floor, point_goal, robot_z, current_yaw):
    """Move to the configured shortest-path fallback waypoint and return its pose."""
    shortest_path_output = env.simulator.scene.get_shortest_path(
        floor,
        env.robots[0].get_position()[:2],
        point_goal,
        entire_path=True,
    )
    if len(shortest_path_output) == 3:
        safe_path, _, _ = shortest_path_output
    elif len(shortest_path_output) == 2:
        safe_path, _ = shortest_path_output
    else:
        raise ValueError(
            f"Unexpected get_shortest_path return format: {len(shortest_path_output)} values"
        )
    if len(safe_path) < 2:
        return None, None

    fallback_idx = choose_fallback_path_index(len(safe_path))
    next_position = safe_path[fallback_idx]
    ori_from_idx = max(fallback_idx - 1, 0)
    next_orientation = safe_path[fallback_idx] - safe_path[ori_from_idx]
    if np.linalg.norm(next_orientation) < 1e-6 and fallback_idx + 1 < len(safe_path):
        next_orientation = safe_path[fallback_idx + 1] - safe_path[fallback_idx]
    if np.linalg.norm(next_orientation) < 1e-6:
        next_orientation = np.array([np.cos(current_yaw), np.sin(current_yaw)])

    next_yaw = np.arctan2(next_orientation[1], next_orientation[0])
    next_quat = p.getQuaternionFromEuler([0, 0, next_yaw])
    env.robots[0].set_position([next_position[0], next_position[1], robot_z])
    env.robots[0].set_orientation(next_quat)
    env.simulator.step()
    return next_position, next_yaw

def path_length(points):
    """
    Compute the 2D path length of a trajectory.

    Args:
        points: Trajectory array with shape [T, D], where the first two
            dimensions correspond to x and y.

    Returns:
        Total 2D path length.
    """
    points = np.asarray(points)
    points = np.atleast_2d(points)

    if points.shape[0] < 2 or points.shape[1] < 2:
        return 0.0

    return float(
        np.sum(
            np.linalg.norm(
                points[1:, :2] - points[:-1, :2],
                axis=1,
            )
        )
    )

def judge_success(
    data,
    shortest_traj,
    collision_th=15,
    success_distance=1.0,
):
    """
    Evaluate one predicted trajectory.

    Assumptions:
        - data[:, :2] contains the predicted 2D positions.
        - data[:, 3] is the collision indicator.
        - A value > 0.5 in data[:, 3] is treated as a collision.
        - At most `collision_th` collisions are allowed.
        - The trajectory terminates when the number of collisions exceeds
          `collision_th`.
        - The task succeeds if the distance to the goal is smaller than
          `success_distance`.

    Args:
        data: Predicted trajectory with shape [T, D].
        shortest_traj: Reference shortest trajectory.
        collision_th: Maximum allowed number of collisions.
        success_distance: Goal success radius.

    Returns:
        Dictionary containing evaluation results.
    """
    shortest_traj = np.asarray(shortest_traj)
    shortest_traj = np.atleast_2d(shortest_traj)

    shortest_dis = path_length(shortest_traj)

    start = shortest_traj[0, :2]
    goal = shortest_traj[-1, :2]

    # Initial straight-line distance from start to goal.
    d_0 = float(np.linalg.norm(start - goal))

    # Handle invalid or empty predicted trajectories.
    if (
        data.size == 0
        or data.ndim != 2
        or data.shape[1] < 4
    ):
        return {
            "arrive": False,
            "collision_num": 0,
            "shortest_dis": shortest_dis,
            "actual_dis": 0.0,
            "d_0": d_0,
            "d_t": d_0,
        }

    collision_num = 0
    actual_dis = 0.0
    arrive = False

    # Initialize final distance using the first predicted pose.
    d_t = float(
        np.linalg.norm(
            data[0, :2] - goal
        )
    )

    for idx, pose in enumerate(data):

        # --------------------------------------------------
        # 1. Compute the actual traveled distance.
        # --------------------------------------------------
        if idx > 0:
            step_distance = float(
                np.linalg.norm(
                    pose[:2] - data[idx - 1, :2]
                )
            )
            actual_dis += step_distance

        # --------------------------------------------------
        # 2. Update the current distance to the goal.
        # --------------------------------------------------
        d_t = float(
            np.linalg.norm(
                pose[:2] - goal
            )
        )

        # --------------------------------------------------
        # 3. Count collisions.
        # --------------------------------------------------
        if pose[3] > 0.5:
            collision_num += 1

        # --------------------------------------------------
        # 4. Terminate if collision budget is exceeded.
        #
        # collision_th = 15 means that up to 15 collisions
        # are allowed. The trajectory terminates at the 16th
        # collision.
        # --------------------------------------------------
        if collision_num > collision_th:
            break

        # --------------------------------------------------
        # 5. Check task success.
        # --------------------------------------------------
        if d_t < success_distance:
            arrive = True
            break

    return {
        "arrive": arrive,
        "collision_num": collision_num,
        "shortest_dis": shortest_dis,
        "actual_dis": actual_dis,
        "d_0": d_0,
        "d_t": d_t,
    }

def score_trajectory(rows, reference, collision_limit=15, success_distance=1.0):
    sample = judge_success(np.asarray(rows, dtype=np.float32).reshape(-1, 4), np.asarray(reference, dtype=np.float32),
                           collision_th=collision_limit, success_distance=success_distance)
    ratio = sample['shortest_dis'] / max(sample['shortest_dis'], sample['actual_dis'], 1e-8)
    progress = max(0., 1 - sample['d_t'] / max(sample['d_0'], 1e-8))
    return dict(success=sample['arrive'], collisions=sample['collision_num'],
                final_distance=sample['d_t'], reference_length_m=sample['shortest_dis'],
                traveled_m=sample['actual_dis'], SPL_reference=float(sample['arrive'])*ratio,
                SoftSPL_reference=progress*ratio)
