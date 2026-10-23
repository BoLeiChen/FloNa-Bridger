"""Aggregate the fixed ImageGoal tasks; reject missing, extra or partial rollouts."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.image_region_protocol import aggregate_image_region_results


def score(output, episodes):
    output = Path(output)
    expected = {tuple(e) for e in episodes}
    if len(expected) != len(episodes) or not expected:
        raise ValueError('Empty or duplicate episode list')
    actual = {(p.parent.parent.name, p.parent.name) for p in output.glob('*/*/result.json')}
    if actual != expected:
        raise ValueError('Episode coverage mismatch: %d missing, %d extra' % (len(expected-actual), len(actual-expected)))
    for scene, traj in expected:
        if not (output/scene/traj/(traj+'.txt')).is_file():
            raise ValueError('Missing rollout: '+scene+'/'+traj)
    result = aggregate_image_region_results(output, episodes)
    result.update(expected_episodes=len(episodes), complete=result['scored_episodes']==len(episodes))
    return {'image': result}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--episodes', default='config/imagegoal_eval_1470.json')
    a = p.parse_args()
    result = score(a.output, json.loads(Path(a.episodes).read_text()))
    (Path(a.output)/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
