"""Score exactly the frozen PointGoal evaluation episodes using original rules."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from eval.floverse_protocol import score_trajectory


def score(output, scenes_dir, episodes):
    output, scenes_dir = Path(output), Path(scenes_dir)
    expected = {(scene, traj) for scene, traj in episodes}
    actual = {(p.parent.parent.name, p.parent.name) for p in output.glob('*/*/result.json')}
    if len(expected) != len(episodes) or actual != expected:
        raise ValueError('Episode coverage mismatch: {} missing, {} extra'.format(len(expected-actual),len(actual-expected)))
    rows=[]
    for scene, traj in episodes:
        path=output/scene/traj/(traj+'.txt')
        reference=np.loadtxt(scenes_dir/scene/traj/(traj+'.txt'),dtype=np.float32,ndmin=2)
        rollout=np.loadtxt(path,dtype=np.float32,ndmin=2) if path.stat().st_size else np.empty((0,4),dtype=np.float32)
        if not np.isfinite(reference).all() or not np.isfinite(rollout).all():
            raise ValueError('Non-finite trajectory: '+str(path))
        rows.append(score_trajectory(rollout,reference,15,1.0))
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--scenes-dir',default='data/eval_without_obj/HM3D')
    parser.add_argument('--episodes-file',default='config/evaluation_episodes.json')
    args=parser.parse_args()
    episodes=json.loads(Path(args.episodes_file).read_text())
    rows=score(args.output,args.scenes_dir,episodes)
    metrics={name:float(np.mean([row[key] for row in rows])) for name,key in [('SR','success'),('SPL_reference','SPL_reference'),('SoftSPL_reference','SoftSPL_reference'),('mean_collisions','collisions')]}
    metrics.update(episodes=len(rows),complete=True)
    (Path(args.output)/'summary.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))
