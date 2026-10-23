"""Shared scene and window selection for all navigation goal types."""
from pathlib import Path
import re
import json

def scene_selected(scene, config):
    selection = config.get('train_scene_dataset', 'all')
    if selection not in ('all', 'hm3d'):
        raise ValueError('train_scene_dataset must be all or hm3d')
    return selection == 'all' or re.fullmatch(r'[0-9]{5}-[A-Za-z0-9]+_[0-9]+', scene) is not None


def select_training_manifest(manifest, config):
    """Select roots/scenes and count a sparse view of the immutable LMDB index.

    Source cache geometry is explicit and checked against records during reading.
    A multiple of its stride can select existing starts without decoding the cache
    or changing the dense RGB/pose sequences.
    """
    source_pace = int(config.get('lmdb_source_traj_pace', 4))
    source_horizon = int(config.get('lmdb_source_len_traj_pred', 16))
    pace = int(config['traj_pace'])
    if source_pace < 1 or pace < source_pace or pace % source_pace:
        raise ValueError('traj_pace must be a positive multiple of the cached stride')
    if int(config['len_traj_pred']) != source_horizon:
        raise ValueError('Prediction horizon must match the cached LMDB horizon')
    allowed = {str(Path(root).resolve()) for root in config['data_folder']}
    # Scene-directory links form the isolated subset view while LMDB keys retain
    # their original source roots. Resolve once per scene, not once per sample.
    allowed_scenes = {str(p.resolve()) for root in config['data_folder']
                      if Path(root).is_dir() for p in Path(root).iterdir() if p.is_dir()}
    result = []
    for item in manifest:
        _, root, scene = item['floor_key'].split('::')
        if (str(Path(root).resolve()) not in allowed
                and str(Path(root)/scene) not in allowed_scenes) or not scene_selected(scene, config):
            continue
        count = int(item['sample_count'])
        if count > 0:
            result.append(dict(item, source_sample_count=count,
                               sample_count=(count-1)//(pace//source_pace)+1))
    if not result:
        raise ValueError('No training trajectories match the selected roots/scenes')
    return result



def read_subset_manifest(path, lmdb_path):
    payload = json.loads(Path(path).read_text())
    if payload.get('version') != 1 or Path(payload['source_lmdb']).resolve() != Path(lmdb_path).resolve():
        raise ValueError('Subset manifest does not match the source LMDB')
    return payload['records']
