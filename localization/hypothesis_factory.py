"""Task dispatch only; route geometry and learned navigation policy remain shared."""
from pathlib import Path
import json
import numpy as np
from PIL import Image

def object_task(config):return config.get('route_task')=='object'

def provider_for(config,device='cpu',allow_inference=False):
    if object_task(config):
        from localization.object_hypotheses import ObjectHypothesisProvider
        return ObjectHypothesisProvider(config,device,allow_inference)
    from localization.goal_hypotheses import GoalHypothesisProvider
    return GoalHypothesisProvider(config,device,allow_inference)

def trajectory_hypotheses(provider,config,root,scene,traj,pose_count):
    directory=Path(root)/scene/traj
    if not object_task(config):return provider.get(directory/'rgb'/('rgb_%d.png'%(pose_count-1)))
    from datasets.bridge_maps import read_calibration
    raw=json.loads((directory/'object/object.json').read_text());entry=raw[0] if isinstance(raw,list) and raw else raw
    if not isinstance(entry,dict) or not entry.get('object_category'):raise ValueError('Missing target category')
    scale,offset=read_calibration(config['scene_assets'],scene)
    path=Path(root)/scene/'floorplan.png'
    with Image.open(path) as im:size=np.array(im.size,dtype=float)
    return provider.get(path,entry['object_category'],origin_xy=-np.asarray(offset)/scale,extent_xy=size/scale)
