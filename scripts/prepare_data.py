"""Build the frozen V3 training selection from extracted FloVerse data."""
import argparse
import gzip
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lmdb
import msgpack
import numpy as np
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
from datasets.lmdb_dataset import encode_record
from utils.build_lmdb_cache import process_and_encode


def frame_bytes(directory):
    return {int(p.stem.rsplit('_', 1)[-1]): p.read_bytes() for p in directory.glob('*.png')}


def prepare(config, selection, max_trajectories=None, map_size_gb=600, check_evaluation=False):
    roots = [Path(p).resolve() for p in config['data_folder']]
    rows = selection['records'][:max_trajectories]
    output = Path(config['lmdb_path'])
    if output.exists():
        raise FileExistsError('Use a fresh LMDB output path: ' + str(output))
    # Verify references first; RGB/depth coverage is checked during decoding.
    for group, scene, trajectory, count in rows:
        folder = roots[group] / scene / trajectory
        if not (folder / (trajectory + '.txt')).is_file():
            raise FileNotFoundError(folder / (trajectory + '.txt'))
    episodes = json.loads(Path('config/evaluation_episodes.json').read_text())
    if check_evaluation:
        for scene, trajectory in episodes:
            path = Path(config.get('eval_data_folder', 'data/eval_without_obj/HM3D')) / scene / trajectory / (trajectory + '.txt')
            if not path.is_file():
                raise FileNotFoundError(path)
    output.mkdir(parents=True)
    env = lmdb.open(str(output), map_size=int(map_size_gb * 1024**3))
    resize = transforms.Resize((96, 96))
    manifest, floors = [], set()
    try:
        for index, (group, scene, trajectory, expected_count) in enumerate(rows):
            root = roots[group]
            floor_key = 'floor::{}::{}'.format(root, scene)
            folder = root / scene / trajectory
            if floor_key not in floors:
                with Image.open(root / scene / 'floorplan.png') as image:
                    floor = transforms.ToTensor()(resize(image)).numpy()
                with env.begin(write=True) as txn:
                    txn.put(floor_key.encode(), encode_record({'floorplan': floor}, 3))
                floors.add(floor_key)
            obj = folder / 'object.json'
            bundle = dict(poses=(folder / (trajectory + '.txt')).read_bytes(), rgb=frame_bytes(folder / 'rgb'), depth=frame_bytes(folder / 'depth'), object=obj.read_bytes() if obj.exists() else None)
            payload, count = process_and_encode(bundle, scene, trajectory, 4, 16, 3)
            if count != expected_count:
                raise ValueError('Trajectory length differs from the frozen training selection: ' + str(folder))
            key = 'traj::{}::{}::{}'.format(root, scene, trajectory)
            with env.begin(write=True) as txn:
                txn.put(key.encode(), payload)
            manifest.append(dict(key=key, floor_key=floor_key, sample_count=count))
            if (index+1) % 100 == 0:
                print('Cached', index+1, '/', len(rows), flush=True)
        with env.begin(write=True) as txn:
            txn.put(b'__manifest__', msgpack.packb(manifest, use_bin_type=True))
        env.sync()
    finally:
        env.close()
    report = dict(trajectories=len(manifest), training_windows=sum((x['sample_count']+1)//2 for x in manifest), evaluation_episodes=len(episodes))
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/pointgoal_v3.yaml')
    parser.add_argument('--selection', default='config/training_selection.json.gz')
    parser.add_argument('--max-trajectories', type=int, help='Smoke preparation only; use a separate cache path')
    parser.add_argument('--map-size-gb', type=float, default=600)
    parser.add_argument('--check-evaluation', action='store_true', help='Also check evaluation pose files; not needed for training-cache construction')
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    with gzip.open(args.selection, 'rt') as stream:
        selection = json.load(stream)
    prepare(config, selection, args.max_trajectories, args.map_size_gb, args.check_evaluation)
