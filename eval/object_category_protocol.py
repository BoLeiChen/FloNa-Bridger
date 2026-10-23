"""Evaluation-only, same-category object-center success; never policy inputs.

Distances are horizontal world meters, as in FloVerse. SPL uses the 256-map
shortest route to the union of success disks, not the old single-instance path.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

PROTOCOL = 'floverse-object-category-center-1m-v1'


from datasets.object_category_targets import category_name, load_category_targets, distance_to_category


def category_reference_length(planner, start, targets, radius):
    """Map geodesic to any free grid cell inside any object success disk.

    Missing map connectivity affects SPL availability, never category SR.
    The evaluator does not pass this GT distance field to the navigation model.
    """
    if distance_to_category(start, targets) <= radius:
        return 0.0
    root, parents = planner.goal_tree(start)
    if parents is None:
        return None
    from scipy.sparse.csgraph import dijkstra
    distances = dijkstra(planner._graph, directed=True, indices=root)
    ys, xs = np.where(planner.walkable)
    world = planner.to_world(np.stack([xs, ys], axis=-1))
    eligible = np.zeros(len(world), dtype=bool)
    for xy in targets['world_xy']:
        eligible |= np.linalg.norm(world - xy, axis=1) <= radius
    reachable = distances[ys * planner.size + xs][eligible]
    reachable = reachable[np.isfinite(reachable)]
    if not len(reachable):
        return None
    snapped_start = planner.to_world(np.array([root % planner.size, root // planner.size]))
    return float(reachable.min() + np.linalg.norm(np.asarray(start)[:2] - snapped_start))


def score_category_trajectory(rows, start, targets, reference_length=None,
                              collision_limit=15, success_distance=1.0):
    """Score the first reached success region, including a successful start.

    The 15th collision may land at a valid target; no moves after it are allowed.
    A regular final success check uses this same function as offline aggregation.
    """
    rows = np.asarray(rows, dtype=float).reshape(-1, 4)
    if not np.isfinite(rows).all():
        raise ValueError('Nonfinite evaluated trajectory')
    position = np.asarray(start, dtype=float)[:2].copy()
    d0 = distance_to_category(position, targets)
    dt = d0
    success = dt <= success_distance
    collisions, traveled, used = 0, 0.0, 0
    for row in rows:
        if success or collisions >= collision_limit:
            break
        traveled += float(np.linalg.norm(row[:2] - position))
        position = row[:2]
        collisions += int(row[3] > .5)
        used += 1
        dt = distance_to_category(position, targets)
        success = dt <= success_distance
    nearest = int(np.linalg.norm(np.asarray(targets['world_xy']) - position, axis=1).argmin())
    ratio = None if reference_length is None else (1. if reference_length == 0 and traveled == 0
             else reference_length / max(reference_length, traveled, 1e-8))
    progress = 1. if d0 <= success_distance else max(0., 1. - max(dt-success_distance, 0.) / (d0-success_distance))
    return dict(success=bool(success), collisions=collisions, final_distance=dt,
                initial_category_distance=d0, traveled_m=traveled, scored_moves=used,
                reference_length_m=reference_length,
                SPL_category_map=None if ratio is None else float(success)*ratio,
                SoftSPL_category_map=None if ratio is None else progress*ratio,
                nearest_object_id=targets['object_ids'][nearest], protocol=PROTOCOL,
                object_success_protocol=PROTOCOL, success_distance=success_distance,
                collision_limit=collision_limit, target_count=len(targets['object_ids']))


def aggregate_category_results(directory, episodes):
    """Never run the legacy single-endpoint evaluator on category episodes."""
    results = []
    for scene, traj in episodes:
        path = Path(directory) / scene / traj / 'result.json'
        if not path.is_file():
            continue
        result = json.loads(path.read_text())
        if result.get('object_success_protocol') != PROTOCOL or result.get('success_distance') != 1.0:
            raise ValueError('Mixed/old ObjectGoal protocol: ' + str(path))
        results.append(result)
    available = [r for r in results if r['SPL_category_map'] is not None]
    metrics = dict(scored_episodes=len(results), protocol=PROTOCOL,
                   spl_reference='256-grid shortest path to any same-category object 1m disk',
                   spl_scored_episodes=len(available), spl_unavailable_episodes=len(results)-len(available))
    if results:
        metrics.update(SR=float(np.mean([r['success'] for r in results])),
                       mean_collisions=float(np.mean([r['collisions'] for r in results])),
                       SPL=None if not available else float(np.mean([r['SPL_category_map'] for r in available])),
                       SoftSPL=None if not available else float(np.mean([r['SoftSPL_category_map'] for r in available])))
    return metrics
