"""Expand immutable multimodal labels to ALL rows of the existing CF2Loc index."""
import argparse,copy,hashlib,json,shutil
from pathlib import Path
import numpy as np
from .data import LocalizationData
from .prepare import DTYPE
from .encoders import file_hash


def expand(source,output):
    source=Path(source);out=Path(output)
    if out.exists():raise FileExistsError(out)
    # Validate published parent labels before reusing them.
    parent=LocalizationData(source,'train','object');meta=parent.meta;info=copy.deepcopy(parent.info);parent.close()
    out.mkdir(parents=True)
    shutil.copy2(source/'source_index.json',out/'source_index.json')
    shutil.copy2(source/'rejected_object_targets.json',out/'rejected_object_targets.json')
    audit={};scene_by_traj=np.array([x['scene'] for x in meta['trajectories']])
    sizes=np.array([x['map_size'] for x in meta['scenes']])
    for split in ('train','val','test'):
        rows=np.memmap(meta['splits'][split]['index_path'],mode='r',dtype=DTYPE)
        scene_ids=scene_by_traj[rows['traj']]
        valid=np.isfinite(rows['pose']).all(1)&(rows['pose'][:,:2]>=0).all(1)&(rows['pose'][:,:2]<=sizes[scene_ids]).all(1)
        if not valid.all():raise ValueError('%s has %d uncalibrated/nonfinite rows; inspect before expanding'%(split,(~valid).sum()))
        if any(meta['trajectories'][int(t)]['split']!=split for t in np.unique(rows['traj'])):raise ValueError('Frame split mismatch')
        ids=np.arange(len(rows),dtype=np.int64);np.save(out/(split+'_image_rows.npy'),ids)
        shutil.copy2(source/(split+'_objects.json'),out/(split+'_objects.json'))
        info['summary']['splits'][split]['images']=len(rows)
        audit[split]=dict(frames=len(rows),trajectories=len(np.unique(rows['traj'])),source_bin_sha256=file_hash(meta['splits'][split]['index_path']))
    info.update(version=2,parent_dataset=str(source),parent_fingerprint=(source/'COMPLETE').read_text(),image_selection='all_source_index_rows')
    info['summary']['image_stride']=1
    (out/'full_index_audit.json').write_text(json.dumps(audit,indent=2))
    info['artifacts']={p.name:file_hash(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out/'metadata.json').write_text(json.dumps(info,indent=2))
    (out/'COMPLETE').write_text(hashlib.sha256(json.dumps(info,sort_keys=True).encode()).hexdigest())
    print(json.dumps(info['summary']['splits'],indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();expand(a.source,a.output)
