"""Build immutable multimodal labels from navigation TRAIN sources only."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
from functools import lru_cache
import numpy as np
from PIL import Image
import math

DTYPE = np.dtype([('traj','<u4'),('frame','<u4'),('pose','<f4',(3,))])
ALIASES = {'couch': 'sofa'}

def category_name(name):
    name = ' '.join(str(name).lower().replace('_', ' ').split())
    return ALIASES.get(name, name)

@lru_cache(maxsize=4096)
def training_path(path):
    parts = Path(path).resolve().parts
    return ('train_with_obj' in parts or 'train_without_obj' in parts) and not any(p in ('test','test_with_obj','test_without_obj') for p in parts)

def validate_splits(meta):
    sets = {s:set(meta['splits'][s]['buildings']) for s in ('train','val','test')}
    if any(sets[a]&sets[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Building leakage across splits')
    for scene in meta['scenes']:
        if scene['source'] != 'HM3D' or not training_path(scene['path']):
            raise ValueError('Only HM3D navigation training sources are allowed: '+scene['path'])
        if scene['building'] not in sets[scene['split']]:
            raise ValueError('Scene/building split mismatch')
    for t in meta['trajectories']:
        if Path(t['path']).parent != Path(meta['scenes'][t['scene']]['path']) or t['split'] != meta['scenes'][t['scene']]['split']:
            raise ValueError('Trajectory split/source mismatch')
    return sets

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-index',required=True);p.add_argument('--object-manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--image-stride',type=int,default=8);p.add_argument('--workers',type=int,default=8);p.add_argument('--scene-assets',required=True);p.add_argument('--reserved-buildings',required=True)
    args=p.parse_args()
    out=Path(args.output)
    if out.exists():raise FileExistsError('Use a new immutable dataset output: '+str(out))
    if args.image_stride<1:raise ValueError('image-stride must be positive')
    raw=Path(args.source_index).read_bytes();meta=json.loads(raw);validate_splits(meta)
    manifest_raw=Path(args.object_manifest).read_bytes();manifest=json.loads(manifest_raw)['records']
    reserved=set(Path(args.reserved_buildings).read_text().split())
    if reserved & (set(meta['splits']['train']['buildings']) | set(meta['splits']['val']['buildings'])):
        raise ValueError('Source image split overlaps reserved navigation test buildings')
    building_split={b:split for split,v in meta['splits'].items() for b in v['buildings']}
    source_scenes={s['path'] for s in meta['scenes']}
    object_roots=sorted({x['floor_key'].split('::')[1]+'/'+x['floor_key'].split('::')[2] for x in manifest if 'train_with_obj' in x['floor_key']})
    out.mkdir(parents=True);(out/'maps').mkdir()
    added=[]
    for path in object_roots:
        if path in source_scenes:continue
        if not training_path(path):raise ValueError('Object source is not navigation train')
        scene=Path(path).name;building,floor=scene.rsplit('_',1);floor=int(floor)
        if building not in building_split:
            bucket=int(hashlib.sha256(('cf2loc-mm-v1:'+building).encode()).hexdigest()[:8],16)%10
            split='test' if building in reserved else ('train' if bucket<8 else ('val' if bucket==8 else 'test'))
            building_split[building]=split;meta['splits'][split]['buildings'].append(building)
        split=building_split[building];assets=Path(args.scene_assets)/building
        scale=float((assets/'scale.txt').read_text().splitlines()[floor])
        offset=[float(v) for v in (assets/'offset.txt').read_text().splitlines()[floor].split()[:2]]
        factor=1/(meta['map_res']*scale)
        with Image.open(Path(path)/'floorplan.png') as image:
            size=[int(math.ceil(v*factor)) for v in image.size]
            mapped=image.convert('RGB').transform((256,256),Image.Transform.AFFINE,(size[0]/(factor*256),0,0,0,size[1]/(factor*256),0),resample=Image.Resampling.BILINEAR,fillcolor=(255,255,255))
            mp=out/'maps'/(scene+'.png');mapped.save(mp)
        meta['scenes'].append(dict(source='HM3D',path=path,building=building,floor=floor,scale=scale,offset=offset,factor=factor,map_path=str(mp),map_size=size,split=split))
        added.append(scene)
    validate_splits(meta)
    scenes={str(Path(s['path']).resolve()):i for i,s in enumerate(meta['scenes'])}
    def read(item):
        _,root,scene,traj=item['key'].split('::')
        if not training_path(root):raise ValueError('Navigation test/unknown source rejected: '+root)
        if 'train_with_obj' not in Path(root).parts:return None
        si=scenes.get(str((Path(root)/scene).resolve()))
        if si is None:raise ValueError('Object scene absent from image split index: '+scene)
        a=json.loads((Path(root)/scene/traj/'object/object.json').read_text());a=a if isinstance(a,list) else [a]
        records=[]
        for v in a:
            xy=np.asarray(v['agent_position'][:2],dtype=float);s=meta['scenes'][si]
            pixel=(xy*s['scale']+np.asarray(s['offset']))*s['factor']
            if not np.isfinite(xy).all():raise ValueError('Nonfinite object target')
            if not ((pixel>=0)&(pixel<np.asarray(s['map_size']))).all():
                records.append(dict(scene=scene,object_id=str(v['object_id']),category=v['object_category'],world_xy=xy.tolist(),map_xy=pixel.tolist(),map_size=s['map_size'],reason='outside_calibrated_map'))
                continue
            cat=category_name(v['object_category'])
            if not cat or cat=='null':raise ValueError('Missing object category')
            records.append((si,cat,str(v['object_id']),xy.tolist(),v['object_category']))
        return records
    unique={};raw_count=0;object_trajs=0;rejected={}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i,records in enumerate(pool.map(read,manifest)):
            if records is not None:
                object_trajs+=1
                for entry in records:
                    raw_count+=1
                    if isinstance(entry,dict):
                        rejected[(entry['scene'],entry['object_id'])]=entry
                        continue
                    si,cat,oid,xy,original=entry
                    key=(si,cat,oid)
                    if key in unique and not np.allclose(unique[key]['world_xy'],xy,atol=1e-6,rtol=0):raise ValueError('Conflicting instance position')
                    unique[key]=dict(scene=si,category=cat,object_id=oid,world_xy=xy,original_category=original)
            if (i+1)%10000==0: print('manifest records',i+1,'/',len(manifest),flush=True)
    groups=defaultdict(list)
    for v in unique.values():groups[(v['scene'],v['category'])].append(v)
    queries=[]
    for (si,cat),instances in sorted(groups.items()):
        # Coincident instance observation points do not gain extra target mass.
        points=sorted({tuple(v['world_xy']) for v in instances})
        queries.append(dict(scene=si,category=cat,split=meta['scenes'][si]['split'],world_xy=points,object_ids=sorted(v['object_id'] for v in instances)))
    summary=dict(rejected_object_instances=len(rejected),navigation_test_used=False,additional_object_scene_floors=added,image_stride=args.image_stride,object_source_trajectories=object_trajs,annotation_records=raw_count,unique_instance_ids=len(unique),category_aliases=ALIASES,splits={})
    for split in ('train','val','test'):
        original=np.memmap(meta['splits'][split]['index_path'],mode='r',dtype=DTYPE)
        selected=(original['frame']%args.image_stride)==0
        for t in meta['trajectories']:
            if t['split']==split and t['end']>t['start']:
                last=t['start']+int(np.argmax(original['frame'][t['start']:t['end']]))
                selected[last]=True
        ids=np.flatnonzero(selected).astype(np.int64);np.save(out/(split+'_image_rows.npy'),ids)
        q=[x for x in queries if x['split']==split]
        (out/(split+'_objects.json')).write_text(json.dumps(q))
        summary['splits'][split]=dict(images=len(ids),object_queries=len(q),object_instances=sum(len(x['object_ids']) for x in q),categories=len({x['category'] for x in q}),buildings=len(meta['splits'][split]['buildings']))
    expanded=json.dumps(meta).encode()
    (out/'source_index.json').write_bytes(expanded)
    metadata=dict(version=1,source_index_sha256=hashlib.sha256(expanded).hexdigest(),image_lmdb_index_sha256=hashlib.sha256(raw).hexdigest(),object_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),summary=summary)
    (out/'rejected_object_targets.json').write_text(json.dumps(list(rejected.values()),indent=2))
    metadata['artifacts']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    # Complete marker published last; consumers refuse partial datasets.
    (out/'COMPLETE').write_text(hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest())
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
