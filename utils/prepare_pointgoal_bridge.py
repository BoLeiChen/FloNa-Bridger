"""Build small calibrated floorplan/path sidecars; never rewrite RGB-D LMDB."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from omegaconf import OmegaConf
from datasets.bridge_maps import prepare_map
from datasets.pointgoal_bridge_dataset import PriorStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--root', action='append', help='Override roots; repeat for eval data')
    parser.add_argument('--scene', action='append')
    parser.add_argument('--max-scenes', type=int)
    parser.add_argument('--priors', action='store_true', help='Also precompute per-trajectory A*; potentially long')
    parser.add_argument('--max-trajs', type=int)
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    store = PriorStore(config)
    report = []
    for root in args.root or config['data_folder']:
        scenes = sorted(p for p in Path(root).iterdir() if (p / 'floorplan.png').is_file())
        if args.scene:
            scenes = [p for p in scenes if p.name in args.scene]
        if args.max_scenes:
            scenes = scenes[:args.max_scenes]
        for scene in scenes:
            directory = prepare_map(root, scene.name, config['scene_assets'], config['map_cache'],
                                    256, config['astar_free_threshold'], config['astar_clearance_m'],
                                    config.get('astar_exclude_exterior', False))
            entry = {'scene': str(scene), 'cache': str(directory), 'samples': 0, 'failed_priors': 0}
            if args.priors:
                trajectories = sorted(p for p in scene.iterdir() if (p / (p.name + '.txt')).is_file())
                if args.max_trajs:
                    trajectories = trajectories[:args.max_trajs]
                for traj in trajectories:
                    poses = np.atleast_2d(np.loadtxt(traj / (traj.name + '.txt')))
                    _, valid = store.trajectory(root, scene.name, traj.name, poses)
                    entry['samples'] += len(valid)
                    entry['failed_priors'] += int((valid.sum(-1) == 0).sum())
            report.append(entry)
            print(json.dumps(entry), flush=True)
    if not report:
        raise ValueError('No matching scenes')
    output = Path(config['map_cache']) / 'last_preparation_report.json'
    output.write_text(json.dumps(report, indent=2))
    print('Prepared %d scenes; report: %s' % (len(report), output))


if __name__ == '__main__':
    main()
