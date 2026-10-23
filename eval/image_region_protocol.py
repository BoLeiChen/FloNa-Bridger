"""ImageGoal counterpart of category protocol, with one image-position target."""
import json
from pathlib import Path
import numpy as np

PROTOCOL='floverse-image-goal-region-1m-v2'

def image_region_result(result):
    result=dict(result)
    result['protocol']=PROTOCOL
    result['initial_goal_distance']=result.pop('initial_category_distance')
    result['SPL_goal_map']=result.pop('SPL_category_map')
    result['SoftSPL_goal_map']=result.pop('SoftSPL_category_map')
    result.pop('object_success_protocol');result.pop('nearest_object_id')
    return result

def aggregate_image_region_results(directory,episodes):
    results=[]
    for scene,traj in episodes:
        path=Path(directory)/scene/traj/'result.json'
        if not path.is_file():continue
        r=json.loads(path.read_text())
        if r.get('protocol')!=PROTOCOL or r.get('success_distance')!=1.:
            raise ValueError('Mixed/old ImageGoal region protocol: '+str(path))
        results.append(r)
    available=[r for r in results if r['SPL_goal_map'] is not None]
    out=dict(scored_episodes=len(results),protocol=PROTOCOL,
             spl_reference='256-grid shortest path to image-position 1m disk',
             spl_scored_episodes=len(available),spl_unavailable_episodes=len(results)-len(available))
    if results:
        out.update(SR=float(np.mean([r['success'] for r in results])),
                   mean_collisions=float(np.mean([r['collisions'] for r in results])),
                   SPL=float(np.mean([r['SPL_goal_map'] for r in available])) if available else None,
                   SoftSPL=float(np.mean([r['SoftSPL_goal_map'] for r in available])) if available else None)
    return out
