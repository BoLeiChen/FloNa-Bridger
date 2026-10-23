"""Calibrated 256px floorplan planning shared by preparation, training and eval."""
import hashlib
import heapq
import json
import os
from pathlib import Path
from functools import lru_cache

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


MAP_VERSION = 2


def world_to_local(points, position, yaw):
    co, si = np.cos(yaw), np.sin(yaw)
    return (np.asarray(points) - position) @ np.array([[co, -si], [si, co]])


def local_to_world(points, position, yaw):
    co, si = np.cos(yaw), np.sin(yaw)
    return np.asarray(points) @ np.array([[co, si], [-si, co]]) + position


def read_calibration(asset_root, scene):
    name, floor = scene.rsplit('_', 1)
    floor = int(floor)
    root = Path(asset_root) / name
    try:
        scale = float((root / 'scale.txt').read_text().splitlines()[floor])
        offset = [float(v) for v in (root / 'offset.txt').read_text().splitlines()[floor].split()]
    except (OSError, IndexError, ValueError) as exc:
        raise ValueError('Missing/invalid world-to-floorplan calibration: ' + str(root)) from exc
    if scale <= 0 or len(offset) != 2 or not np.isfinite([scale] + offset).all():
        raise ValueError('Invalid floorplan calibration: ' + str(root))
    return scale, offset


def map_key(root, scene):
    identity = str((Path(root) / scene).resolve())
    return hashlib.sha256(identity.encode()).hexdigest()[:16] + '_' + scene


def prepare_map(root, scene, asset_root, output, size=256, free_threshold=0.96,
                clearance_m=0.10, exclude_exterior=False):
    source = Path(root) / scene / 'floorplan.png'
    raw = source.read_bytes()
    scale, offset = read_calibration(asset_root, scene)
    with Image.open(source) as image:
        width, height = image.size
        gray = image.convert('L')
        # BOX averages all source pixels, unlike nearest-neighbor wall aliasing.
        reduced = gray.resize((size, size), Image.Resampling.BOX)
        # Conservative obstacle pooling retains even an isolated thin-wall pixel.
        # Thresholding only the averaged gray image could erase diagonal corners.
        source_obstacle = (np.asarray(gray).astype(np.float32) / 255 < free_threshold).astype(np.uint8) * 255
        obstacle_coverage = Image.fromarray(source_obstacle).resize((size, size), Image.Resampling.BOX)
    metadata = dict(version=MAP_VERSION, source=str(source.resolve()),
                    source_sha256=hashlib.sha256(raw).hexdigest(), source_size=[width, height],
                    size=size, scale=scale, offset=offset, free_threshold=free_threshold,
                    clearance_m=clearance_m, exclude_exterior=exclude_exterior)
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    metadata['fingerprint'] = digest
    directory = Path(output) / map_key(root, scene)
    directory.mkdir(parents=True, exist_ok=True)
    saved = directory / 'metadata.json'
    if saved.exists() and json.loads(saved.read_text()) == metadata:
        if (directory / 'walkable.npy').exists() and (directory / 'floorplan_256.png').exists():
            return directory
    free = np.asarray(obstacle_coverage) == 0
    if exclude_exterior:
        components, _ = label(free)
        border = np.unique(np.concatenate([components[0], components[-1], components[:, 0], components[:, -1]]))
        free &= ~np.isin(components, border[border > 0])
    if clearance_m > 0:
        sampling = (height / (size * scale), width / (size * scale))
        free &= distance_transform_edt(free, sampling=sampling) > clearance_m
    reduced.save(directory / 'floorplan_256.png')
    np.save(directory / 'walkable.npy', free.astype(np.uint8))
    saved.write_text(json.dumps(metadata, indent=2))
    return directory


def sample_prefix(world_path, position, yaw, horizon, step_m):
    """Include the current origin, then uniform metric samples; terminal pad is invalid."""
    path = np.asarray(world_path, dtype=np.float64)
    if len(path) == 0:
        return np.zeros((horizon, 2), np.float32), np.zeros(horizon, np.float32)
    if np.linalg.norm(path[0] - position) > 1e-8:
        path = np.vstack([position, path])
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    keep = np.r_[True, np.diff(lengths) > 1e-8]
    path, lengths = path[keep], lengths[keep]
    distances = np.arange(horizon) * step_m
    samples = np.stack([np.interp(distances, lengths, path[:, dim]) for dim in range(2)], -1)
    valid = (distances <= lengths[-1] + 1e-7).astype(np.float32)
    result = world_to_local(samples, position, yaw).astype(np.float32)
    result[0] = 0
    return result, valid


class FloorplanPlanner:
    def __init__(self, directory, max_snap_m=0.5):
        directory = Path(directory)
        self.metadata = json.loads((directory / 'metadata.json').read_text())
        self.walkable = np.load(directory / 'walkable.npy').astype(bool)
        self.max_snap_m = max_snap_m
        self.size = self.metadata['size']
        self.pixel_scale = np.asarray([self.size, self.size]) / self.metadata['source_size']
        self.meters_per_pixel = 1 / (self.pixel_scale * self.metadata['scale'])
        self._graph = None

    def to_pixel(self, points):
        return (np.asarray(points) * self.metadata['scale'] + self.metadata['offset']) * self.pixel_scale

    def to_world(self, pixels):
        return (np.asarray(pixels) / self.pixel_scale - self.metadata['offset']) / self.metadata['scale']

    def snap(self, point):
        point = np.asarray(point)
        if np.any(point < 0) or np.any(point >= self.size):
            return None
        nearest = np.rint(point).astype(int).clip(0, self.size - 1)
        if self.walkable[nearest[1], nearest[0]]:
            return tuple(nearest)
        radius = np.ceil(self.max_snap_m / self.meters_per_pixel).astype(int)
        x0, y0 = np.maximum(0, nearest - radius)
        x1, y1 = np.minimum(self.size, nearest + radius + 1)
        ys, xs = np.where(self.walkable[y0:y1, x0:x1])
        if not len(xs):
            return None
        candidates = np.stack([xs + x0, ys + y0], -1)
        distance = np.linalg.norm((candidates - point) * self.meters_per_pixel, axis=1)
        index = distance.argmin()
        return tuple(candidates[index]) if distance[index] <= self.max_snap_m else None

    def neighbors(self, x, y):
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.size and 0 <= ny < self.size) or not self.walkable[ny, nx]:
                continue
            # Never squeeze diagonally between two blocked wall cells.
            if dx and dy and not (self.walkable[y, nx] and self.walkable[ny, x]):
                continue
            yield nx, ny, float(np.linalg.norm(np.array([dx, dy]) * self.meters_per_pixel))

    def astar(self, start, goal):
        start, goal = self.snap(self.to_pixel(start)), self.snap(self.to_pixel(goal))
        if start is None or goal is None:
            return np.empty((0, 2))
        cost, parents = {start: 0.}, {}
        queue = [(0., 0., start)]
        while queue:
            _, g, node = heapq.heappop(queue)
            if g > cost[node]:
                continue
            if node == goal:
                path = [node]
                while node in parents:
                    node = parents[node]
                    path.append(node)
                return self.to_world(np.asarray(path[::-1]))
            for x, y, step in self.neighbors(*node):
                nxt = (x, y)
                value = g + step
                if value < cost.get(nxt, float('inf')):
                    cost[nxt], parents[nxt] = value, node
                    h = np.linalg.norm((np.array(nxt) - goal) * self.meters_per_pixel)
                    heapq.heappush(queue, (value + h, value, nxt))
        return np.empty((0, 2))

    def goal_tree(self, goal):
        """One C-level Dijkstra per trajectory goal; same graph/costs as online A*."""
        goal = self.snap(self.to_pixel(goal))
        if goal is None:
            return None, None
        if self._graph is None:
            rows, cols, costs = [], [], []
            ids = np.arange(self.size**2).reshape(self.size, self.size)
            # Vectorized construction matters for thousands of scenes on HDD training.
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                x0, x1 = max(0, -dx), min(self.size, self.size - dx)
                y0, y1 = max(0, -dy), min(self.size, self.size - dy)
                src = (slice(y0, y1), slice(x0, x1))
                dst = (slice(y0 + dy, y1 + dy), slice(x0 + dx, x1 + dx))
                mask = self.walkable[src] & self.walkable[dst]
                if dx and dy:
                    mask &= self.walkable[y0:y1, x0 + dx:x1 + dx] & self.walkable[y0 + dy:y1 + dy, x0:x1]
                rows.append(ids[src][mask])
                cols.append(ids[dst][mask])
                costs.append(np.full(mask.sum(), np.linalg.norm(np.array([dx, dy]) * self.meters_per_pixel)))
            self._graph = coo_matrix((np.concatenate(costs), (np.concatenate(rows), np.concatenate(cols))),
                                     shape=(self.size**2, self.size**2)).tocsr()
        goal_id = goal[1] * self.size + goal[0]
        _, predecessors = dijkstra(self._graph, directed=True, indices=goal_id, return_predecessors=True)
        return goal_id, predecessors

    def path_from_tree(self, start, tree):
        goal, parents = tree
        start = self.snap(self.to_pixel(start))
        if start is None or parents is None:
            return np.empty((0, 2))
        index = start[1] * self.size + start[0]
        path = [index]
        while index != goal:
            index = int(parents[index])
            if index < 0:
                return np.empty((0, 2))
            path.append(index)
        ids = np.asarray(path)
        return self.to_world(np.stack([ids % self.size, ids // self.size], -1))

    def prior(self, position, yaw, goal, horizon=16, step_m=0.1, tree=None):
        path = self.astar(position, goal) if tree is None else self.path_from_tree(position, tree)
        return sample_prefix(path, position, yaw, horizon, step_m)
