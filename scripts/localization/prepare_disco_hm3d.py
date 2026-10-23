"""Leakage-audited HM3D query selection and frozen diffusion candidate mining.

Run manifest once, then torchrun --nproc_per_node=3 ... mine --split train/val.
Test images/poses are never opened; test directory names are exclusion metadata.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from functools import lru_cache
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from localization.disco import diverse_modes, crop_maps


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2)); tmp.replace(path)


def building_id(name):
    # Ignore numeric dataset ordering prefixes; compare the actual HM3D IDs.
    return name.rsplit('_',1)[0].split('-')[-1]


def assert_disjoint(meta, navigation_roots):
    groups={k:{building_id(b) for b in v['buildings']} for k,v in meta['splits'].items()}
    if any(groups[a]&groups[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Localization building splits overlap')
    excluded=set()
    for root in navigation_roots:
        p=Path(root)
        if not p.is_dir(): raise FileNotFoundError('Required navigation exclusion root: '+root)
        names=[s.name for s in p.iterdir() if s.is_dir() and '-' in s.name and '_' in s.name]
        if not names: raise ValueError('Empty navigation exclusion root: '+root)
        excluded.update(building_id(s) for s in names)
    for split in ('train','val','test'):
        overlap=groups[split]&excluded
        if overlap: raise ValueError('Navigation test building leakage: '+str(sorted(overlap)))
    for scene in meta['scenes']:
        if scene['source']!='HM3D' or building_id(scene['building']) not in groups[scene['split']]:
            raise ValueError('Unexpected scene source/split')
    for traj in meta['trajectories']:
        scene=meta['scenes'][traj['scene']]
        if traj['split'] != scene['split'] or 'test' in Path(traj['path']).parts:
            raise ValueError('Trajectory split/source mismatch')
    return {'navigation_test_buildings': sorted(excluded),
            'localization_buildings':{k:sorted(v) for k,v in groups.items()},
            'overlap':[], 'test_images_or_poses_read':False}


def open_source(c, split):
    if split not in ('train','val'): raise ValueError('DisCo preparation accepts train/val only')
    from third_party.cf2loc_hm3d.DisCo_model.floverse_full_dataset import FloVerseFullDataset
    return FloVerseFullDataset(c['index'],split,augment=False,lmdb_root=c['lmdb_root'])


def select_rows(ds,n,seed,terminal_fraction):
    # Endpoint-enriched, spatially deduplicated, inverse-building-frequency sampling.
    original=np.asarray(ds.selection) if ds.selection is not None else np.arange(len(ds.rows))
    rows=ds.rows[original]
    tids=rows['traj'].astype(np.int64)
    sids=np.array([t['scene'] for t in ds.trajectories])[tids]
    terminal=rows['frame']==np.array([t['pose_rows']-1 for t in ds.trajectories])[tids]
    pose=rows['pose']
    cells=np.column_stack((sids,np.floor(pose[:,:2]*ds.map_res/0.5),
                          np.floor(np.mod(pose[:,2],2*np.pi)/math.radians(15)))).astype(np.int64)
    rng=np.random.default_rng(seed)
    # Prefer terminal representatives where terminal and ordinary samples share a cell.
    order=np.concatenate((rng.permutation(np.flatnonzero(terminal)),rng.permutation(np.flatnonzero(~terminal))))
    _,keep=np.unique(cells[order],axis=0,return_index=True)
    pool=order[keep]
    buildings=np.array([s['building'] for s in ds.scenes])[sids]
    _,inv,counts=np.unique(buildings[pool],return_inverse=True,return_counts=True)
    weights=1.0/counts[inv]
    def choose(candidates,count):
        count=min(count,len(candidates))
        w=weights[candidates]; w=w/w.sum()
        return rng.choice(candidates,count,replace=False,p=w)
    ep=choose(np.flatnonzero(terminal[pool]),int(n*terminal_fraction))
    remaining=np.setdiff1d(np.arange(len(pool)),ep)
    rest=choose(remaining,max(0,n-len(ep)))
    ids=np.sort(pool[np.concatenate((ep,rest))])
    return ids, {'frames':len(ids),'available_frames':len(ds),'unique_pose_cells':len(pool),
                 'terminal_frames':int(terminal[ids].sum()),'buildings':int(len(set(buildings[ids])))}


def manifest(c):
    run=Path(c['run_dir']); run.mkdir(parents=True,exist_ok=True)
    if (run/'manifest.json').exists(): raise FileExistsError('Manifest already exists; use a new run directory')
    meta=json.loads(Path(c['index']).read_text())
    audit=assert_disjoint(meta,c['navigation_test_roots'])
    if digest(c['diffusion_checkpoint'])!=c['diffusion_sha256']: raise ValueError('Diffusion SHA mismatch')
    if digest(c['refiner_checkpoint'])!=c['refiner_sha256']: raise ValueError('Refiner SHA mismatch')
    info={'config':c,'index_sha256':digest(c['index']),'leakage_audit':audit,'splits':{}}
    for split in ('train','val'):
        ds=open_source(c,split)
        ids,stats=select_rows(ds,c[split+'_frames'],c['seed']+(split=='val'),c['terminal_fraction'])
        np.save(run/(split+'_ids.npy'),ids)
        stats['selection_sha256']=digest(run/(split+'_ids.npy'))
        stats['source_rows_sha256']=digest(ds.info['index_path'])
        info['splits'][split]=stats
    atomic_json(run/'manifest.json',info)
    print(json.dumps({k:v for k,v in info.items() if k!='config'}),flush=True)


class Queries(Dataset):
    def __init__(self,source,ids): self.source=source; self.ids=ids
    def __len__(self): return len(self.ids)
    def __getitem__(self,i):
        idx=int(self.ids[i]); ds=self.source
        obs,pose,_,fmap,wh,*_=ds[idx]
        row=ds.rows[int(ds.selection[idx]) if ds.selection is not None else idx]
        return obs,pose,fmap,wh,int(ds.trajectories[int(row['traj'])]['scene'])


@torch.inference_mode()
def mine(c,split):
    rank=int(os.environ.get('RANK',0)); world=int(os.environ.get('WORLD_SIZE',1))
    device=torch.device('cuda',int(os.environ.get('LOCAL_RANK',0)))
    torch.cuda.set_device(device); torch.set_num_threads(4)
    torch.manual_seed(c['seed']+rank+10000*(split=='val'))
    run=Path(c['run_dir']); info=json.loads((run/'manifest.json').read_text())
    if info['config']!=c or info['index_sha256']!=digest(c['index']): raise ValueError('Stale manifest/config')
    if digest(run/(split+'_ids.npy'))!=info['splits'][split]['selection_sha256']: raise ValueError('Selection checksum mismatch')
    if digest(c['diffusion_checkpoint'])!=c['diffusion_sha256']: raise ValueError('Diffusion SHA mismatch')
    all_ids=np.load(run/(split+'_ids.npy')); ids=np.array_split(all_ids,world)[rank]
    out=run/'cache'/split/str(rank); out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        done=json.loads((out/'complete.json').read_text())
        if done['world_size']!=world or done['ids_sha256']!=hashlib.sha256(ids.tobytes()).hexdigest(): raise ValueError('Stale cache shard')
        print('CACHE_ALREADY_COMPLETE',split,rank,flush=True); return
    from third_party.cf2loc_hm3d.DisCo_model.pose_query_diffusion import PoseQueryDiffusionLocalizer
    model_config=dict(torch.load(c['diffusion_checkpoint'],map_location='cpu')['hyper_parameters'])
    model_config['dptv2_ckpt_path']=c['dinov2_checkpoint']
    model=PoseQueryDiffusionLocalizer.load_from_checkpoint(c['diffusion_checkpoint'],map_location='cpu',config=model_config).eval().to(device)
    model.requires_grad_(False)
    source=open_source(c,split)
    expected_rows=info['splits'][split].get('source_rows_sha256')
    if expected_rows and digest(source.info['index_path'])!=expected_rows: raise ValueError('Source pose rows changed')
    data=Queries(source,ids)
    loader=DataLoader(data,batch_size=c['mining_batch'],num_workers=c['workers'],pin_memory=True)
    n=len(ids); k=c['modes']+(2 if split=='train' else 0)
    shapes={'tokens':((n,256,128),'float16'),'maps':((n,k,1,c['crop_size'],c['crop_size']),'uint8'),
            'poses':((n,k,3),'float32'),'gt':((n,3),'float32'),'valid':((n,k),'bool'),
            'density':((n,k),'float32'),'errors':((n,k,2),'float32')}
    arrays={key:np.lib.format.open_memmap(out/(key+'.npy'),mode='w+',dtype=dtype,shape=shape) for key,(shape,dtype) in shapes.items()}
    captured={}
    handle=model.condition_encoder.image_mixer.register_forward_hook(lambda m,i,o:captured.update(tokens=o.detach()))
    @lru_cache(maxsize=32)
    def raw_map(si):
        with Image.open(source.scenes[si]['map_path']) as img: return np.asarray(img.convert('L'))
    started=time.time(); offset=0
    rng=np.random.default_rng(c['seed']+rank)
    for obs,gt,fmap,wh,scenes in loader:
        obs=obs.to(device); fmap=fmap.to(device); wh_gpu=wh.to(device)
        context=model.condition_encoder(obs,fmap)
        samples=model.sample_from_context(*context,wh_gpu,num_particles=c['particles'],num_steps=c['sample_steps'])
        _,density=model.select_pose_mode(samples)
        tokens=captured['tokens'].float().cpu().numpy()
        for j in range(len(gt)):
            idx=offset+j
            selected=diverse_modes(samples[j],density[j],c['map_res'],c['modes'],c['nms_m'],c['nms_deg'])
            poses=samples[j,selected].float().cpu().numpy()
            dens=density[j,selected].float().cpu().numpy()
            count=len(poses)
            arrays['poses'][idx]=0; arrays['density'][idx]=0; arrays['valid'][idx]=False
            arrays['maps'][idx]=255
            arrays['errors'][idx]=1e6
            arrays['poses'][idx,:count]=poses; arrays['density'][idx,:count]=dens; arrays['valid'][idx,:count]=True
            if split=='train':
                # GT-near positive + hard reversed-heading example, TRAIN ONLY.
                positive=gt[j].numpy().copy()
                positive[:2]+=rng.uniform(-.2,.2,2)/c['map_res']
                positive[:2]=np.clip(positive[:2],0,wh[j].numpy()-1)
                positive[2]+=rng.uniform(-math.radians(10),math.radians(10))
                negative=gt[j].numpy().copy(); negative[2]+=math.pi+rng.uniform(-.2,.2)
                arrays['poses'][idx,-2:]=np.stack((positive,negative))
                arrays['valid'][idx,-2:]=True
            valid=arrays['valid'][idx]
            actual=arrays['poses'][idx,valid]
            crops=crop_maps(raw_map(int(scenes[j])),actual,c)
            arrays['maps'][idx,valid]=crops.mul(255).round().clamp(0,255).byte().numpy()
            arrays['errors'][idx,valid,0]=np.linalg.norm((actual[:,:2]-gt[j,:2].numpy())*c['map_res'],axis=-1)
            arrays['errors'][idx,valid,1]=np.abs((actual[:,2]-float(gt[j,2])+np.pi)%(2*np.pi)-np.pi)
            arrays['tokens'][idx]=tokens[j]; arrays['gt'][idx]=gt[j].numpy()
        offset+=len(gt)
        if offset%(c['mining_batch']*20)==0 or offset==n:
            print(json.dumps({'stage':'mining','split':split,'rank':rank,'done':offset,'total':n,'frames_s':round(offset/(time.time()-started),2)}),flush=True)
    handle.remove()
    for arr in arrays.values(): arr.flush()
    np.save(out/'source_ids.npy',ids)
    atomic_json(out/'complete.json',{'frames':n,'world_size':world,'rank':rank,'split':split,
        'ids_sha256':hashlib.sha256(ids.tobytes()).hexdigest(), 'gt_candidates_added':split=='train',
        'elapsed_s':time.time()-started,'manifest_sha256':digest(run/'manifest.json')})


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['manifest','mine']);p.add_argument('--config',required=True);p.add_argument('--split',choices=['train','val'])
    a=p.parse_args(); c=yaml.safe_load(Path(a.config).read_text())
    if a.action=='manifest': manifest(c)
    else: mine(c,a.split)
if __name__=='__main__': main()
