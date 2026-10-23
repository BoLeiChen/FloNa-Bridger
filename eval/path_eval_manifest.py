"""Fixed historical ImageGoal episode identities; never reads evaluation outcomes."""
import json
from pathlib import Path

DEFAULT_IMAGE_MANIFEST = Path(__file__).resolve().parents[1] / 'config/imagegoal_eval_1470.json'


def select_episodes(root, task, max_trajs=30, manifest=None, scene_ids=None):
    from eval.common import list_scene_ids, list_traj_ids
    if max_trajs <= 0:
        raise ValueError('max-trajs must be positive')
    available_scenes = set(list_scene_ids(root, None))
    source = manifest or (DEFAULT_IMAGE_MANIFEST if task == 'image' else None)
    if source is not None:
        raw = json.loads(Path(source).read_text())
        if not isinstance(raw, list) or not raw:
            raise ValueError('Episode manifest must be a nonempty list')
        if any(not isinstance(e, list) or len(e) != 2 or
               not all(isinstance(x, str) for x in e) for e in raw):
            raise ValueError('Expected [scene, trajectory] episode pairs')
        pairs = [tuple(e) for e in raw]
        if len(set(pairs)) != len(pairs):
            raise ValueError('Duplicate episodes in manifest')
        by_scene = {}
        for scene, traj in pairs:
            by_scene.setdefault(scene, []).append(traj)
        for scene, trajectories in by_scene.items():
            if scene not in available_scenes:
                raise ValueError('Manifest scene unavailable: '+scene)
            available = set(list_traj_ids(root, scene))
            if not set(trajectories) <= available:
                raise ValueError('Manifest trajectories unavailable: '+scene)
    else:
        by_scene = {s: list_traj_ids(root, s) for s in sorted(available_scenes)}
    selected = set(by_scene) if scene_ids is None else set(scene_ids)
    if not selected or not selected <= set(by_scene):
        raise ValueError('Requested scenes outside evaluation manifest')
    return [(s,t) for s in sorted(selected) for t in by_scene[s][:max_trajs]]
