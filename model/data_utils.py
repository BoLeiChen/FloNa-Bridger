import os
import io
import heapq
from typing import Union, Tuple, Optional

from PIL import Image
import numpy as np
import torch
import pyastar2d
from torchvision import transforms
import torchvision.transforms.functional as TF
from typing import Tuple, List

VISUALIZATION_IMAGE_SIZE = (160, 160)
IMAGE_ASPECT_RATIO = (
    1 / 1
)  

def get_data_path(data_folder: str, f: str, name):
    if type(name) == int or type(name) == np.int16:
        return os.path.join(data_folder, f, "{:05d}.png".format(name))
    elif type(name) == str:
        return os.path.join(data_folder, f, name + ".png") 
    else:
        raise ValueError("name should be either int or str") 


def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )


def to_local_coords(            
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw
) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    if len(curr_yaw) == 2:
        dir = curr_yaw - curr_pos
        #conver vector to yaw
        curr_yaw = np.arctan2(dir[1], dir[0])
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos).dot(rotmat)

def to_global_coords(           
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw
) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert     in local coordinates  (b,2) 
        curr_pos (np.ndarray): current position         in global coordinates  (2,)
        curr_yaw (float): current yaw                  in global coordinates   (2,)
    Returns:
        np.ndarray: positions in local coordinates
    """
    if len(curr_yaw) == 2:
        dir = curr_yaw - curr_pos
        #conver vector to yaw
        curr_yaw = np.arctan2(dir[1], dir[0])
    rotmat = yaw_rotmat(curr_yaw)  # R from local to global
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return curr_pos + positions.dot(rotmat.transpose())


def calculate_deltas(waypoints: torch.Tensor) -> torch.Tensor:
    """
    Calculate deltas between waypoints

    Args:
        waypoints (torch.Tensor): waypoints
    Returns:
        torch.Tensor: deltas
    """
    num_params = waypoints.shape[1]
    origin = torch.zeros(1, num_params)
    prev_waypoints = torch.concat((origin, waypoints[:-1]), axis=0)
    deltas = waypoints - prev_waypoints
    if num_params > 2:
        return calculate_sin_cos(deltas)
    return deltas


def calculate_sin_cos(waypoints: torch.Tensor) -> torch.Tensor:
    """
    Calculate sin and cos of the angle

    Args:
        waypoints (torch.Tensor): waypoints
    Returns:
        torch.Tensor: waypoints with sin and cos of the angle
    """
    assert waypoints.shape[1] == 3
    angle_repr = torch.zeros_like(waypoints[:, :2])
    angle_repr[:, 0] = torch.cos(waypoints[:, 2])
    angle_repr[:, 1] = torch.sin(waypoints[:, 2])
    return torch.concat((waypoints[:, :2], angle_repr), axis=1)


def transform_images(
    img: Image.Image, transform: transforms, image_resize_size: Tuple[int, int], aspect_ratio: float = IMAGE_ASPECT_RATIO
):
    w, h = img.size
    if w > h:
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))  
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))
    viz_img = img.resize(VISUALIZATION_IMAGE_SIZE)
    viz_img = TF.to_tensor(viz_img)
    img = img.resize(image_resize_size)
    transf_img = transform(img)
    return viz_img, transf_img


def resize_and_aspect_crop(
    img: Image.Image, image_resize_size: Tuple[int, int], aspect_ratio: float = IMAGE_ASPECT_RATIO
):
    w, h = img.size
    if w > h:
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))  
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))
    img = img.resize(image_resize_size)
    resize_img = TF.to_tensor(img)
    return resize_img


def img_path_to_data(path: Union[str, io.BytesIO], image_resize_size: Tuple[int, int]) -> torch.Tensor:
    """
    Load an image from a path and transform it
    Args:
        path (str): path to the image
        image_resize_size (Tuple[int, int]): size to resize the image to
    Returns:
        torch.Tensor: resized image as tensor
    """
    # return transform_images(Image.open(path), transform, image_resize_size, aspect_ratio)
    return resize_and_aspect_crop(Image.open(path), image_resize_size)    


def metric_positions_to_resized_pixels(
    metric_positions: Union[np.ndarray, List[float]],
    metric_size_cm: float,
    image_size: Tuple[int, int],
) -> np.ndarray:
    positions = np.asarray(metric_positions, dtype=np.float32)
    squeeze = positions.ndim == 1
    if squeeze:
        positions = positions[None, :]
    pixel_positions = positions * 100.0 + np.array([metric_size_cm / 2.0, metric_size_cm / 2.0], dtype=np.float32)
    pixel_positions[:, 0] *= float(image_size[0]) / float(metric_size_cm)
    pixel_positions[:, 1] *= float(image_size[1]) / float(metric_size_cm)
    return pixel_positions[0] if squeeze else pixel_positions


def resized_pixels_to_metric_positions(
    pixel_positions: Union[np.ndarray, List[float]],
    metric_size_cm: float,
    image_size: Tuple[int, int],
) -> np.ndarray:
    positions = np.asarray(pixel_positions, dtype=np.float32)
    squeeze = positions.ndim == 1
    if squeeze:
        positions = positions[None, :]
    metric_positions = positions.copy()
    metric_positions[:, 0] *= float(metric_size_cm) / float(image_size[0])
    metric_positions[:, 1] *= float(metric_size_cm) / float(image_size[1])
    metric_positions -= np.array([metric_size_cm / 2.0, metric_size_cm / 2.0], dtype=np.float32)
    metric_positions /= 100.0
    return metric_positions[0] if squeeze else metric_positions


def floorplan_to_walkable_map(
    floorplan: Union[str, Image.Image, np.ndarray, torch.Tensor],
    free_thresh: float = 0.96,
) -> np.ndarray:
    if isinstance(floorplan, str):
        gray = np.array(Image.open(floorplan).convert("L"), dtype=np.float32) / 255.0
    elif isinstance(floorplan, Image.Image):
        gray = np.array(floorplan.convert("L"), dtype=np.float32) / 255.0
    elif torch.is_tensor(floorplan):
        arr = floorplan.detach().cpu().numpy()
        if arr.ndim == 3:
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] == 1:
            gray = arr[..., 0].astype(np.float32)
        else:
            gray = arr[..., :3].mean(axis=-1).astype(np.float32)
        if gray.max() > 1.0:
            gray /= 255.0
    else:
        arr = np.asarray(floorplan)
        if arr.ndim == 3:
            gray = arr[..., :3].mean(axis=-1).astype(np.float32)
        else:
            gray = arr.astype(np.float32)
        if gray.max() > 1.0:
            gray /= 255.0
    walkable = gray >= free_thresh
    return walkable.astype(np.uint8)


def snap_point_to_walkable(
    walkable_map: np.ndarray,
    xy: Union[np.ndarray, List[int], Tuple[int, int]],
    max_radius: int = 16,
) -> Optional[np.ndarray]:
    xy = np.asarray(xy, dtype=np.int32)
    x = int(np.clip(xy[0], 0, walkable_map.shape[1] - 1))
    y = int(np.clip(xy[1], 0, walkable_map.shape[0] - 1))
    if walkable_map[y, x] > 0:
        return np.array([x, y], dtype=np.int32)
    for radius in range(1, max_radius + 1):
        x0 = max(0, x - radius)
        x1 = min(walkable_map.shape[1], x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(walkable_map.shape[0], y + radius + 1)
        patch = walkable_map[y0:y1, x0:x1]
        ys, xs = np.where(patch > 0)
        if len(xs) == 0:
            continue
        coords = np.stack([xs + x0, ys + y0], axis=1)
        dists = np.sum((coords - np.array([x, y])) ** 2, axis=1)
        return coords[np.argmin(dists)].astype(np.int32)
    return None


def build_goal_distance_tree(
    walkable_map: np.ndarray,
    goal_xy: Union[np.ndarray, List[int], Tuple[int, int]],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    goal_xy = snap_point_to_walkable(walkable_map, goal_xy)
    if goal_xy is None:
        parents = np.full((walkable_map.shape[0], walkable_map.shape[1], 2), -1, dtype=np.int16)
        return parents, None

    parents = np.full((walkable_map.shape[0], walkable_map.shape[1], 2), -1, dtype=np.int16)
    distances = np.full(walkable_map.shape, np.inf, dtype=np.float32)
    heap = [(0.0, int(goal_xy[0]), int(goal_xy[1]))]
    distances[goal_xy[1], goal_xy[0]] = 0.0
    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, np.sqrt(2.0)), (-1, 1, np.sqrt(2.0)),
        (1, -1, np.sqrt(2.0)), (1, 1, np.sqrt(2.0)),
    ]
    while heap:
        current_dist, x, y = heapq.heappop(heap)
        if current_dist > distances[y, x]:
            continue
        for dx, dy, step_cost in neighbors:
            nx = x + dx
            ny = y + dy
            if nx < 0 or ny < 0 or nx >= walkable_map.shape[1] or ny >= walkable_map.shape[0]:
                continue
            if walkable_map[ny, nx] == 0:
                continue
            new_dist = current_dist + step_cost
            if new_dist < distances[ny, nx]:
                distances[ny, nx] = new_dist
                parents[ny, nx] = np.array([x, y], dtype=np.int16)
                heapq.heappush(heap, (new_dist, nx, ny))
    return parents, goal_xy.astype(np.int32)


def trace_path_from_goal_tree(
    goal_parents: np.ndarray,
    start_xy: Union[np.ndarray, List[int], Tuple[int, int]],
    snapped_goal_xy: Optional[np.ndarray],
    walkable_map: np.ndarray,
    max_steps: int = 4096,
) -> np.ndarray:
    if snapped_goal_xy is None:
        return np.zeros((0, 2), dtype=np.int32)
    start_xy = snap_point_to_walkable(walkable_map, start_xy)
    if start_xy is None:
        return np.zeros((0, 2), dtype=np.int32)
    path = [start_xy.astype(np.int32)]
    current = start_xy.astype(np.int32)
    for _ in range(max_steps):
        if np.array_equal(current, snapped_goal_xy):
            break
        parent = goal_parents[current[1], current[0]]
        if parent[0] < 0 or parent[1] < 0:
            return np.zeros((0, 2), dtype=np.int32)
        current = parent.astype(np.int32)
        path.append(current)
        if np.array_equal(current, snapped_goal_xy):
            break
    return np.asarray(path, dtype=np.int32)


def sample_path_prefix_points(
    path_metric: np.ndarray,
    num_points: int,
    step_size_m: float,
    return_valid_mask: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    path_metric = np.asarray(path_metric, dtype=np.float32)
    if len(path_metric) == 0:
        sampled = np.zeros((num_points, 2), dtype=np.float32)
        valid = np.zeros(num_points, dtype=np.float32)
        return (sampled, valid) if return_valid_mask else sampled
    if len(path_metric) == 1:
        sampled = np.repeat(path_metric, num_points, axis=0)
        valid = np.zeros(num_points, dtype=np.float32)
        return (sampled, valid) if return_valid_mask else sampled
    segment_lengths = np.linalg.norm(path_metric[1:] - path_metric[:-1], axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    targets = step_size_m * (np.arange(num_points, dtype=np.float32) + 1.0)
    sampled = []
    total_length = cumulative[-1]
    for target in targets:
        if target >= total_length:
            sampled.append(path_metric[-1])
            continue
        seg_idx = np.searchsorted(cumulative, target, side="right") - 1
        seg_idx = int(np.clip(seg_idx, 0, len(segment_lengths) - 1))
        seg_start = path_metric[seg_idx]
        seg_end = path_metric[seg_idx + 1]
        seg_dist = cumulative[seg_idx]
        ratio = (target - seg_dist) / max(segment_lengths[seg_idx], 1e-6)
        sampled.append(seg_start + ratio * (seg_end - seg_start))
    sampled = np.asarray(sampled, dtype=np.float32)
    valid = (targets <= total_length + 1e-6).astype(np.float32)
    return (sampled, valid) if return_valid_mask else sampled


def path_xy_to_local_condition(
    path_local_xy: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Encode a sampled local path as [x, y, sin(turn), cos(turn), valid]."""
    path_local_xy = np.asarray(path_local_xy, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=np.float32)
    if path_local_xy.ndim != 2 or path_local_xy.shape[-1] != 2:
        raise ValueError(f"Expected path shape [N, 2], got {path_local_xy.shape}")
    if valid_mask.shape != (path_local_xy.shape[0],):
        raise ValueError(
            f"Expected valid mask shape {(path_local_xy.shape[0],)}, got {valid_mask.shape}"
        )

    valid_bool = valid_mask > 0.5
    sin_turn = np.zeros(path_local_xy.shape[0], dtype=np.float32)
    cos_turn = valid_bool.astype(np.float32)  # Boundary valid points mean zero turn.
    if path_local_xy.shape[0] >= 3:
        incoming = path_local_xy[1:-1] - path_local_xy[:-2]
        outgoing = path_local_xy[2:] - path_local_xy[1:-1]
        usable = (
            valid_bool[:-2]
            & valid_bool[1:-1]
            & valid_bool[2:]
            & (np.linalg.norm(incoming, axis=-1) > 1e-6)
            & (np.linalg.norm(outgoing, axis=-1) > 1e-6)
        )
        incoming_yaw = np.arctan2(incoming[:, 1], incoming[:, 0])
        outgoing_yaw = np.arctan2(outgoing[:, 1], outgoing[:, 0])
        turn = (outgoing_yaw - incoming_yaw + np.pi) % (2.0 * np.pi) - np.pi
        interior_indices = np.nonzero(usable)[0] + 1
        sin_turn[interior_indices] = np.sin(turn[usable])
        cos_turn[interior_indices] = np.cos(turn[usable])

    return np.concatenate(
        [
            path_local_xy,
            sin_turn[:, None],
            cos_turn[:, None],
            valid_mask[:, None],
        ],
        axis=-1,
    ).astype(np.float32)


def astar_path_on_walkable_map(
    walkable_map: np.ndarray,
    start_xy: Union[np.ndarray, List[int], Tuple[int, int]],
    goal_xy: Union[np.ndarray, List[int], Tuple[int, int]],
    max_snap_radius: int = 16,
) -> np.ndarray:
    snapped_start = snap_point_to_walkable(walkable_map, start_xy, max_radius=max_snap_radius)
    snapped_goal = snap_point_to_walkable(walkable_map, goal_xy, max_radius=max_snap_radius)
    if snapped_start is None or snapped_goal is None:
        return np.zeros((0, 2), dtype=np.int32)
    if np.array_equal(snapped_start, snapped_goal):
        return snapped_start[None].astype(np.int32)

    costs = np.where(walkable_map > 0, 1.0, np.inf).astype(np.float32)
    path_yx = pyastar2d.astar_path(
        costs,
        (int(snapped_start[1]), int(snapped_start[0])),
        (int(snapped_goal[1]), int(snapped_goal[0])),
        allow_diagonal=True,
    )
    if path_yx is None or len(path_yx) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    path_yx = np.asarray(path_yx, dtype=np.int32)
    return np.stack([path_yx[:, 1], path_yx[:, 0]], axis=1)

def img_path_to_data_and_point_transfer(path: Union[str, io.BytesIO], ori_size: float, image_resize_size: Tuple[int, int], cur_pos: np.ndarray, goal_pos: np.ndarray, cur_ori: np.ndarray) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load an image and two points, transform the image and transfer the points to local coordinates
    Args:
        path (str): path to the image
        cur_pos (np.ndarray): current position in the pixel coordinate of original image [x,y]
        goal_pos (np.ndarray): goal position in the pixel coordinate of original image
        image_resize_size (Tuple[int, int]): size to resize the image to [width, height]
    Returns:
        Tuple[torch.Tensor, np.ndarray, np.ndarray]: resized image as tensor, current position in the transformed image size coordinate, goal position in the same coordinate
    """
    
    with Image.open(path) as img:
        w0 = ori_size
        h0 = ori_size
        w, h = img.size
        cur_pos = cur_pos * 100 + np.array([w0 / 2, h0 / 2])
        goal_pos = goal_pos * 100 + np.array([w0 / 2, h0 / 2])
        cur_ori = cur_ori * 100 + np.array([w0 / 2, h0 / 2])      
        aspect_ratio = IMAGE_ASPECT_RATIO

        img = img.resize(image_resize_size)
        cur_pos_in_resizeSize = cur_pos * image_resize_size[0] / w0
        goal_pos_in_resizeSize = goal_pos * image_resize_size[0] / w0
        cur_ori_in_resizeSize = cur_ori * image_resize_size[0] / w0       
        resize_img = TF.to_tensor(img)
        
        return (resize_img, cur_pos_in_resizeSize, goal_pos_in_resizeSize, cur_ori_in_resizeSize)
