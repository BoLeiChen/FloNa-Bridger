"""Shared physical object targets for training supervision and evaluation only."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np

def category_name(value):
    # Spelling normalization only: do not silently merge semantic subclasses.
    return ' '.join(str(value).lower().replace('_', ' ').split())


@lru_cache(maxsize=128)
def _asset(path):
    raw = Path(path).read_bytes()
    records = json.loads(raw)
    if not isinstance(records, list) or not records:
        raise ValueError('Missing scene object annotations: ' + path)
    return records, hashlib.sha256(raw).hexdigest()


def load_category_targets(scene_assets, scene, episode_dir):
    """Use the episode instance ONLY to align asset/simulator floor numbering.

    Some assets number the ground floor 0 while simulator floors.txt includes a
    basement at 0. The identical object ID + agent_position disambiguate this.
    All matching category instances on that semantic floor are then accepted.
    """
    building, level = scene.rsplit('_', 1)
    annotation = json.loads((Path(episode_dir) / 'object/object.json').read_text())
    reference = annotation[0] if isinstance(annotation, list) and annotation else annotation
    if not isinstance(reference, dict):
        raise ValueError('Missing episode object annotation')
    category = category_name(reference['object_category'])
    path = str(Path(scene_assets) / building / (building + '.json'))
    records, digest = _asset(path)
    anchors = [v for v in records if str(v['object_id']) == str(reference['object_id'])
               and category_name(v['object_category']) == category
               and np.allclose(v['agent_position'], reference['agent_position'], atol=1e-5, rtol=0)]
    if len(anchors) != 1:
        raise ValueError('Cannot uniquely align object/floor: ' + str(episode_dir))
    anchor = anchors[0]
    if not np.allclose(anchor['agent_position'], reference['agent_position'], atol=1e-5, rtol=0):
        raise ValueError('Object asset and episode coordinate frames disagree')
    semantic_floor = int(anchor['floor'])
    selected = {}
    for item in records:
        if int(item['floor']) != semantic_floor or category_name(item['object_category']) != category:
            continue
        xyz = np.asarray(item['obj_position'], dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError('Invalid real object position')
        identity = str(item['object_id'])
        if identity in selected and not np.array_equal(selected[identity], xyz[:2]):
            raise ValueError('Conflicting object positions')
        selected[identity] = xyz[:2]
    ids = sorted(selected)
    return dict(category=category, object_ids=ids, world_xy=[selected[i].tolist() for i in ids],
                position_source='scene_asset.obj_position', source_path=path, source_sha256=digest,
                scene=scene, simulator_floor=int(level), semantic_floor=semantic_floor,
                floor_alignment='episode_object_id_and_matching_agent_position')


def distance_to_category(position, targets):
    xy = np.asarray(targets['world_xy'], dtype=float).reshape(-1, 2)
    if not len(xy) or not np.isfinite(xy).all():
        raise ValueError('Empty or invalid category targets')
    return float(np.linalg.norm(xy - np.asarray(position)[:2], axis=1).min())
