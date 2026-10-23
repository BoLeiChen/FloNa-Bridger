"""Shared success-region labels: ImageGoal singleton or ObjectGoal target set.

Ground-truth targets are supervision only, never policy feature tensors.
"""
from collections import OrderedDict
import hashlib
import numpy as np
from scipy.sparse.csgraph import dijkstra
from datasets.object_category_targets import distance_to_category
from datasets.route_goal_supervision import route_labels_from_fields

LABEL_VERSION = 'object-category-center-region-v2'


class CategoryDistanceCache:
    def __init__(self, maxsize=16):
        self.maxsize = maxsize
        self.cache = OrderedDict()

    def distances(self, planner, position, targets, radius):
        xy = np.asarray(targets['world_xy'], dtype=np.float64)
        key = (planner.metadata.get('fingerprint', id(planner)),
               hashlib.sha256(xy.tobytes()).hexdigest(), float(radius))
        if key not in self.cache:
            if planner._graph is None:
                planner.goal_tree(position)
            if planner._graph is None:
                return None
            ys, xs = np.where(planner.walkable)
            world = planner.to_world(np.stack([xs, ys], axis=-1))
            inside = np.zeros(len(xs), bool)
            for point in xy:
                inside |= np.linalg.norm(world-point, axis=-1) <= radius
            sources = ys[inside]*planner.size + xs[inside]
            field = (dijkstra(planner._graph, directed=True, indices=sources, min_only=True)
                     if len(sources) else np.full(planner.size**2, np.inf))
            self.cache[key] = field
            while len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]


def category_route_labels(candidates, planner, position, targets, cache, config):
    radius = float(config.get('goal_label_radius_m', 1.))
    if radius != 1.:
        raise ValueError('Category labels must match the 1m evaluation protocol')
    mask = candidates['candidate_mask'].astype(bool)
    is_goal = candidates['candidate_is_goal'].astype(bool)
    xy = np.asarray(targets['world_xy'], float)
    error = np.linalg.norm(candidates['candidate_goals'][:, None, :] - xy[None], axis=-1).min(-1)
    positive = mask & is_goal & (error <= radius)
    weights = np.zeros(len(mask), np.float32)
    temperature = float(config.get('route_label_temperature', .5))
    if temperature <= 0:
        raise ValueError('Invalid label temperature')
    if positive.any():
        # All matching instances are correct. Prefer shorter feasible routes
        # softly, rather than favoring the single demonstration endpoint.
        cost = candidates['candidate_stats'][positive, 0]  # full path length / 10
        weights[positive] = np.exp(-(cost-cost.min())/temperature)
        weights /= weights.sum()
    else:
        field = cache.distances(planner, position, targets, radius)
        if field is not None:
            fallback = dict(candidates, candidate_mask=mask & ~is_goal)
            weights = route_labels_from_fields(fallback, planner, field[None], temperature)[0][0]
    # Such windows should terminate immediately at inference, not teach explore.
    active = distance_to_category(position, targets) > radius
    if not active:
        weights[:] = 0
    return dict(route_target=weights, goal_set_reject=np.array(not positive.any(), np.float32),
                goal_candidate_positive=positive, route_sample_valid=np.array(active, bool),
                bridge_compatible=positive, candidate_prefix_error=error.astype(np.float32))
