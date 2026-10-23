"""Goal-blind geometric proposals, causal visit memory and training-only labels."""
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from datasets.bridge_maps import sample_prefix, world_to_local

VERSION = 1


def resample(path, n):
    path = np.asarray(path, np.float64)
    d = np.r_[0., np.linalg.norm(np.diff(path, axis=0), axis=1).cumsum()]
    keep = np.r_[True, np.diff(d) > 1e-8]
    path, d = path[keep], d[keep]
    return np.stack([np.interp(np.linspace(0, d[-1], n), d, path[:, i]) for i in range(2)], -1)


class VisitMemory:
    """Only actually visited positions; no unobserved region is declared target-free."""
    def __init__(self, cell_m=.5):
        self.cell_m = cell_m
        self.cells = {}
        self.failures = {}
        self.last = None

    def key(self, xy):
        return tuple(np.floor(np.asarray(xy) / self.cell_m).astype(int))

    def update(self, xy, failed=False):
        key = self.key(xy)
        if failed:
            self.failures[key] = min(10, self.failures.get(key, 0) + 1)
            return  # A rejected waypoint was NOT visited.
        if key != self.last:
            self.cells[key] = min(20, self.cells.get(key, 0) + 1)
        self.last = key

    def features(self, path):
        keys = [self.key(p) for p in path[1:]]
        if not keys:
            return [0., 0., 0.]
        return [np.mean([min(self.cells.get(k, 0), 4) / 4 for k in keys]),
                min(self.cells.get(keys[-1], 0), 4) / 4,
                np.mean([min(self.failures.get(k, 0), 3) / 3 for k in keys])]


class PathProposer:
    """Dijkstra on the exact A* graph: one source tree, diverse reachable endpoints.

    ImageGoal calls pass no goal. The proposal signature has no image/GT labels.
    Cache contains geometry only; visits and headings are recomputed per call.
    """
    def __init__(self, config):
        self.k = int(config.get('path_candidates', 12))
        self.points = int(config.get('path_route_points', 32))
        self.horizon = int(config['len_traj_pred'])
        self.step = float(config['astar_step_m'])
        self.pool = int(config.get('path_proposal_pool', 96))
        self.cache = OrderedDict()
        self.disk_path = config.get('path_candidate_cache')
        self.disk = None
        self.source_tree_key = None
        self.source_tree = None
        self.cache_size = int(config.get('path_geometry_cache', 256))
        if not 1 <= self.k <= 32 or self.points < 4 or self.pool < self.k:
            raise ValueError('Invalid path proposal dimensions')

    def _tree(self, planner, position):
        snapped = planner.snap(planner.to_pixel(position))
        key = (planner.metadata.get('fingerprint', id(planner)), snapped)
        if key != self.source_tree_key:
            self.source_tree = ((None, None) if snapped is None else
                                planner.goal_tree(planner.to_world(np.asarray(snapped))))
            self.source_tree_key = key
        return self.source_tree

    def _explore(self, planner, position):
        snapped = planner.snap(planner.to_pixel(position))
        if snapped is None:
            return []
        key = (planner.metadata.get('fingerprint', id(planner)), snapped)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        disk_key = hashlib.sha256(json.dumps([VERSION, key, self.k, self.pool], default=str).encode()).hexdigest().encode()
        if self.disk_path and 'fingerprint' in planner.metadata:
            if self.disk is None:
                import lmdb
                Path(self.disk_path).mkdir(parents=True, exist_ok=True)
                self.disk = lmdb.open(str(self.disk_path), map_size=2**39, max_readers=256)
            with self.disk.begin() as txn:
                raw = txn.get(disk_key)
            if raw is not None:
                with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
                    result = [arrays[k].copy() for k in sorted(arrays.files)]
                self.cache[key] = result
                while len(self.cache) > self.cache_size: self.cache.popitem(last=False)
                return result
        # Deterministic within a snapped source cell, independent of first caller.
        position = planner.to_world(np.asarray(snapped))
        # goal_tree(position) is a source-rooted tree on this undirected grid.
        tree = self._tree(planner, position)
        root, parents = tree
        if parents is None:
            return []
        ids = np.flatnonzero(parents >= 0)
        if not len(ids):
            return []
        pixels = np.stack([ids % planner.size, ids // planner.size], -1)
        endpoints = planner.to_world(pixels)
        delta = endpoints - position
        distance = np.linalg.norm(delta, axis=1)
        # Stratify both bearing and Euclidean range, then deduplicate ROUTES.
        bearing = np.arctan2(delta[:, 1], delta[:, 0])
        sector = np.floor((bearing + np.pi) / (2*np.pi) * 16).astype(int).clip(0, 15)
        selected = []
        for s in range(16):
            idx = np.flatnonzero((sector == s) & (distance >= .5))
            if len(idx):
                idx = idx[np.argsort(distance[idx], kind='stable')]
                selected.extend(idx[np.linspace(0, len(idx)-1, max(1, self.pool//16)).astype(int)].tolist())
        if not selected:
            selected = [int(distance.argmax())]
        paths = []
        for i in sorted(set(selected)):
            reverse = planner.path_from_tree(endpoints[i], tree)
            if len(reverse) >= 2:
                paths.append(reverse[::-1].copy())
        if not paths:
            return []
        summaries = np.stack([resample(p, 8) for p in paths])
        lengths = np.array([np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in paths])
        # Farthest-point selection over full route signatures retains branches,
        # unlike endpoint-only NMS or averaging routes through walls.
        take = [int(lengths.argmax())]
        separation = np.full(len(paths), np.inf)
        for _ in range(min(self.k, len(paths))-1):
            separation = np.minimum(separation, np.linalg.norm(summaries-summaries[take[-1]], axis=-1).mean(-1))
            separation[take] = -1
            nxt = int(separation.argmax())
            if separation[nxt] < .3:
                break
            take.append(nxt)
        result = [paths[i] for i in take]
        if self.disk is not None:
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **{'path_%02d'%i:p for i,p in enumerate(result)})
            with self.disk.begin(write=True) as txn: txn.put(disk_key, buffer.getvalue())
        self.cache[key] = result
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return result

    def propose(self, planner, position, yaw, memory=None, point_goal=None, previous_goal=None, point_tree=None):
        if point_goal is not None:
            p = planner.astar(position, point_goal) if point_tree is None else planner.path_from_tree(position, point_tree)
            paths = [p] if len(p) else []
        else:
            paths = list(self._explore(planner, position))
            # Keep the previous hypothesis available after the source moves.
            if previous_goal is not None and np.linalg.norm(previous_goal-position) > .5:
                snapped = planner.snap(planner.to_pixel(previous_goal))
                endpoint = None if snapped is None else planner.to_world(np.asarray(snapped))
                # Check the snapped endpoint BEFORE search. Usually it is already
                # present. Otherwise reuse the current source tree on the same graph.
                if endpoint is not None and all(np.linalg.norm(endpoint-q[-1]) > .5 for q in paths):
                    reverse = planner.path_from_tree(endpoint, self._tree(planner, position))
                    if len(reverse):
                        paths = [reverse[::-1].copy()] + paths[:self.k-1]
        memory = memory or VisitMemory()
        values = dict(candidate_routes=np.zeros((self.k,self.points,2),np.float32),
                      candidate_grid=np.zeros((self.k,self.points,2),np.float32),
                      candidate_priors=np.zeros((self.k,self.horizon,2),np.float32),
                      candidate_valid=np.zeros((self.k,self.horizon),np.float32),
                      candidate_goals=np.zeros((self.k,2),np.float32),
                      candidate_stats=np.zeros((self.k,4),np.float32),
                      candidate_mask=np.zeros(self.k,bool))
        for i, raw in enumerate(paths[:self.k]):
            path = raw if np.linalg.norm(raw[0]-position)<1e-8 else np.vstack([position,raw])
            full = resample(path,self.points)
            values['candidate_routes'][i] = world_to_local(full,position,yaw)
            values['candidate_grid'][i] = (planner.to_pixel(full)+.5)/planner.size*2-1
            values['candidate_priors'][i],values['candidate_valid'][i] = sample_prefix(path,position,yaw,self.horizon,self.step)
            values['candidate_goals'][i] = path[-1] if point_goal is None else point_goal
            length = np.linalg.norm(np.diff(path,axis=0),axis=1).sum()
            values['candidate_stats'][i] = [length/10]+memory.features(full)
            values['candidate_mask'][i] = length > 1e-5
        # No fabricated straight line on a failed map. Model has an explicit hold fallback.
        return values


def training_labels(candidates, expert, planner, goal, goal_tree, temperature=.5, max_prefix_error=.5):
    """GT is used only here, after goal-blind ImageGoal proposal generation."""
    mask = candidates['candidate_mask']
    valid = candidates['candidate_valid']
    error = np.linalg.norm(candidates['candidate_priors']-expert[None],axis=-1)
    ade = (error*valid).sum(-1)/np.maximum(valid.sum(-1),1)
    costs = np.full(len(mask), 1e4, np.float32)
    for i in np.flatnonzero(mask):
        remaining = planner.path_from_tree(candidates['candidate_goals'][i], goal_tree)
        if len(remaining):
            distance = np.linalg.norm(np.diff(remaining,axis=0),axis=1).sum()
            costs[i] = ade[i]/max_prefix_error + .1*distance + .15*candidates['candidate_stats'][i,0]
    feasible = mask & (costs < 1e4)
    weights = np.zeros(len(mask),np.float32)
    if feasible.any():
        weights[feasible] = np.exp(-(costs[feasible]-costs[feasible].min())/temperature)
        weights /= weights.sum()
    compatible = feasible & (ade <= max_prefix_error) & (valid.sum(-1)>=4)
    return dict(route_target=weights, bridge_compatible=compatible,
                candidate_prefix_error=ade.astype(np.float32))
