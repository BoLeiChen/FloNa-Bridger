"""Validate prepared V3 data before constructing training caches."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import yaml


def validate(root, config, tasks, check_frames=True):
    root = Path(root)
    shapes = np.load(root / 'trav_maps/floor_shapes.npy', allow_pickle=True).item()
    report = {}
    actual_tasks = set()
    for split, key in [('train', 'scene_names'), ('test', 'scene_test_names')]:
        counts = dict(scene_floors=0, trajectories=0, frames=0, training_windows=0)
        for scene in config['datasets'][key]:
            print('Validating', split, scene, file=sys.stderr, flush=True)
            directory = root / 'scenes_117' / split / scene
            if scene not in shapes:
                raise ValueError('Missing metric map dimensions: ' + scene)
            scene_id, floor = scene.rsplit('_', 1)
            choices = [root / 'trav_maps' / scene_id, root / 'trav_maps' / scene] if floor == '0' else [root / 'trav_maps' / scene, root / 'trav_maps' / scene_id]
            map_dir = next((p for p in choices if p.is_dir()), choices[0])
            with Image.open(map_dir / ('floor_trav_test_' + floor + '_modified.png')) as image:
                image.verify()
            for name in ('floorplan.png', 'planner_floorplan.png'):
                with Image.open(directory / name) as image:
                    image.verify()
            paths = sorted(directory.glob('traj_*/*.npy'), key=lambda p: int(p.parent.name.split('_')[1]))
            if not paths:
                raise ValueError('No trajectories: ' + str(directory))
            if split == 'test':
                actual_tasks.update((scene, int(p.parent.name.split('_')[1])) for p in paths[:10])
            counts['scene_floors'] += 1
            for path in paths:
                array = np.load(path, allow_pickle=False)
                if array.ndim != 2 or array.shape[1] != 4 or not np.isfinite(array).all():
                    raise ValueError('Invalid trajectory array: ' + str(path))
                length = len(array)
                if check_frames:
                    frames = {int(p.stem) for p in path.parent.glob('*.png') if p.stem.isdigit()}
                    missing = set(range(length)) - frames
                    if missing:
                        raise ValueError('Missing frames in {}: {}'.format(path.parent, sorted(missing)[:10]))
                counts['trajectories'] += 1
                counts['frames'] += length
                begin = config['context_size'] * config['datasets']['waypoint_spacing']
                goal = length - 1 - config['datasets']['end_slack']
                counts['training_windows'] += len(range(begin, goal + 1, 5 * config['datasets']['waypoint_spacing']))
        report[split] = counts
    expected = {(task['scene'], task['id']) for task in tasks}
    if actual_tasks != expected or len(expected) != 500:
        raise ValueError('Prepared data does not match the fixed 500 evaluation tasks')
    report['evaluation_episodes'] = len(expected)
    report['floor_shape_entries'] = len(shapes)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='training/pointgoal_v3.yaml')
    parser.add_argument('--data-root', help='Override the config data root')
    parser.add_argument('--skip-frames', action='store_true', help='Skip frame-name coverage check')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    tasks = json.loads(Path('experiments/pointgoal_v3_results.json').read_text())['tasks']
    print(json.dumps(validate(args.data_root or config['data_root'], config, tasks, not args.skip_frames), indent=2))
