"""Replay saved initial RGB-depth maps; no simulator ground truth is used."""
import sys,json,time
from pathlib import Path
import numpy as np
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from model.depth_bridge_guidance import depth_to_map

torch.set_num_threads(2)
folder=root/'results/exps/depth_bridge_full_20261012_081018_depth_bridge/image'
rows=[];start=time.time()
for p in sorted(folder.glob('*/predicted_depth_m.npy')):
    depth=np.load(p);h,w=depth.shape
    intrinsic=np.array([[h/2,0,w/2],[0,h/2,h/2],[0,0,1]])
    maps=[depth_to_map(depth,intrinsic,config,'cpu') for config in [
        {},{'floor_surface':True},{'floor_surface':True,'floor_scale_calibration':True}]]
    rows.append(dict(episode=p.parent.name,
        known_fraction=[m.known_fraction for m in maps],
        max_free_clearance=[float(m.clearance.max()) for m in maps],
        origin_clearance=[float(m.query(m.clearance,torch.zeros(1,1,2))[0]) for m in maps],
        occupied_cells=[m.occupied_cells for m in maps],calibration=maps[-1].depth_calibration))
assert len(rows)==500
assert all(r['occupied_cells'][0]==r['occupied_cells'][1] and not any(r['origin_clearance']) for r in rows)
summary=dict(maps=len(rows),seconds=time.time()-start,
    maps_with_footprint_support=[sum(r['max_free_clearance'][i]>=.15+2**.5*.025 for r in rows) for i in [0,1,2]],
    calibrated_frames=sum(r['calibration']['accepted'] for r in rows),
    mean_known_fraction=np.mean([r['known_fraction'] for r in rows],axis=0).tolist(),
    note='Initial frames only; columns [point projection, surface interpolation, scale calibration plus surfaces]. No memory replay without historical depth.')
output=root/'logs/depth_route_floor_replay.json'
output.write_text(json.dumps(dict(summary=summary,rows=rows),indent=2));print(json.dumps(summary,indent=2))
