"""Create one dataset root from the legacy trees without changing source data."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import yaml


def prepare(model_root, reference_root, output, copy=False, scene_lists=None, eval_count=10):
    model_root = Path(model_root).resolve()
    reference_root = Path(reference_root).resolve()
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if not (model_root / 'trav_maps' / 'floor_shapes.npy').is_file():
        raise FileNotFoundError(model_root / 'trav_maps' / 'floor_shapes.npy')
    # Validate before creating output: keep both floorplans and verify that the
    # shared trajectory serves as the original scoring reference without changes.
    entries = []
    for split in ('train', 'test'):
        scenes = model_root / 'scenes_117' / split
        if not scenes.is_dir():
            raise FileNotFoundError(scenes)
        selected = [scenes / name for name in scene_lists[split]] if scene_lists else sorted(p for p in scenes.iterdir() if p.is_dir())
        for scene in selected:
            if not scene.is_dir():
                raise FileNotFoundError(scene)
            reference = reference_root / 'scenes_117' / split / scene.name
            for floorplan in (scene / 'floorplan.png', reference / 'floorplan.png'):
                if not floorplan.is_file():
                    raise FileNotFoundError(floorplan)
            trajectories = sorted((p for p in scene.glob('traj_*') if p.is_dir()), key=lambda p: int(p.name.split('_')[1]))
            evaluation_tasks = set(trajectories[:eval_count])
            for traj in trajectories:
                path = traj / (traj.name + '.npy')
                if not path.is_file():
                    raise FileNotFoundError(path)
                old = reference / traj.name / path.name
                if split == 'test' and traj in evaluation_tasks and not old.is_file():
                    raise FileNotFoundError(old)
                if split == 'test' and old.is_file() and not np.array_equal(np.load(path), np.load(old)):
                    raise ValueError(f'Trajectory mismatch: {path} and {old}')
            entries.append((split, scene, reference, trajectories))

    def transfer(source, target):
        if copy:
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
        else:
            target.symlink_to(source, target_is_directory=source.is_dir())

    output.mkdir(parents=True)
    transfer(model_root / 'trav_maps', output / 'trav_maps')
    for split, scene, reference, trajectories in entries:
        dest = output / 'scenes_117' / split / scene.name
        dest.mkdir(parents=True)
        transfer(scene / 'floorplan.png', dest / 'floorplan.png')
        transfer(reference / 'floorplan.png', dest / 'planner_floorplan.png')
        for traj in trajectories:
            transfer(traj, dest / traj.name)
        # Old LMDB keys contain old paths. Rebuild image/index/A* caches under
        # the new root rather than copying stale scene-level caches.
    return len(entries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', required=True, help='Legacy datasets_96 directory')
    parser.add_argument('--reference-root', required=True, help='Legacy datasets directory')
    parser.add_argument('--output', default='data', help='New, nonexistent dataset directory')
    parser.add_argument('--copy', action='store_true', help='Copy data for a standalone directory; default creates symlinks')
    parser.add_argument("--train-config", default="training/pointgoal_v3.yaml")
    parser.add_argument("--eval-config", default="experiments/alpha_0p7_depth_route_full.yaml")
    args = parser.parse_args()
    with open(args.train_config) as f:
        train = yaml.safe_load(f)
    with open(args.eval_config) as f:
        evaluation = yaml.safe_load(f)
    scene_lists = {"train": train["datasets"]["scene_names"], "test": sorted(set(train["datasets"]["scene_test_names"]) | set(evaluation["test_scenes"]))}
    count = prepare(args.model_root, args.reference_root, args.output, args.copy, scene_lists, evaluation.get("traj_per_scene", 10))
    print(f'Prepared {count} scene-floor directories in {args.output}; source data unchanged.')
