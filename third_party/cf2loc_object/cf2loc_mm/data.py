"""Data readers independent of FloVerse and the old CF2Loc Python packages."""
import hashlib,io,json,os
from pathlib import Path
from functools import lru_cache
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from .prepare import DTYPE,validate_splits
from .encoders import image_tensor

@lru_cache(maxsize=8)
def verify_artifacts(root, fingerprint):
    root=Path(root);info=json.loads((root/'metadata.json').read_text())
    for name,digest in info.get('artifacts',{}).items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Dataset artifact changed: '+name)

class LocalizationData(Dataset):
    def __init__(self,root,split,task,lmdb_root=None):
        self.root=Path(root);self.split=split;self.task=task
        if not (self.root/'COMPLETE').is_file():raise ValueError('Dataset preparation is incomplete')
        self.info=json.loads((self.root/'metadata.json').read_text())
        if hashlib.sha256(json.dumps(self.info,sort_keys=True).encode()).hexdigest()!=(self.root/'COMPLETE').read_text():raise ValueError('Dataset metadata changed')
        raw=(self.root/'source_index.json').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=self.info['source_index_sha256']:raise ValueError('Source index hash mismatch')
        verify_artifacts(str(self.root), (self.root/'COMPLETE').read_text())
        self.meta=json.loads(raw);validate_splits(self.meta)
        self.env=None;self.pid=None;self.inverse=None;self.lmdb_path=None
        if task=='object':
            self.queries=json.loads((self.root/(split+'_objects.json')).read_text())
            cats=defaultdict(list)
            for i,q in enumerate(self.queries):
                if q['split']!=split or self.meta['scenes'][q['scene']]['split']!=split:raise ValueError('Object split mismatch')
                cats[q['category']].append(i)
            self.category_groups=[np.array(ids) for _,ids in sorted(cats.items())]
        elif task=='image':
            self.ids=np.load(self.root/(split+'_image_rows.npy'),mmap_mode='r')
            self.rows=np.memmap(self.meta['splits'][split]['index_path'],mode='r',dtype=DTYPE)
            if lmdb_root:
                self.lmdb_path=str(Path(lmdb_root)/split)
                schema=json.loads((Path(self.lmdb_path)/'complete.json').read_text())
                if schema['index_sha256']!=self.info['image_lmdb_index_sha256'] or schema['split']!=split:raise ValueError('Image LMDB index mismatch')
                selection=np.load(Path(self.lmdb_path)/'selection.npy',mmap_mode='r')
                if hashlib.sha256(selection.tobytes()).hexdigest()!=schema['selection_sha256']:raise ValueError('LMDB selection hash mismatch')
                self.inverse=np.full(len(self.rows),-1,dtype=np.int64);self.inverse[selection]=np.arange(len(selection))
                if (self.inverse[self.ids]<0).any():raise ValueError('Requested image rows absent from LMDB')
        else:raise ValueError(task)
        if not len(self):raise ValueError('Empty split/task')

    def close(self):
        if getattr(self,'env',None) is not None:
            self.env.close();self.env=None

    def __del__(self):
        self.close()

    def __len__(self):return len(self.queries) if self.task=='object' else len(self.ids)

    @lru_cache(maxsize=8)
    def floor(self,si):
        with Image.open(self.meta['scenes'][si]['map_path']) as im:
            return image_tensor(im,256)

    def image_payload(self,row):
        import lmdb
        if self.env is None or self.pid!=os.getpid():
            self.env=lmdb.open(self.lmdb_path,readonly=True,lock=False,readahead=False,max_readers=256);self.pid=os.getpid()
        with self.env.begin() as txn:value=txn.get(b'r'+int(self.inverse[row]).to_bytes(8,'big'))
        if value is None:raise KeyError('Missing LMDB image')
        return io.BytesIO(value)

    def __getitem__(self,idx):
        if self.task=='object':
            q=self.queries[idx];si=q['scene'];category=q['category']
            # Object-only query never loads terminal RGB or trajectory endpoint.
            targets=np.column_stack([np.asarray(q['world_xy']),np.zeros(len(q['world_xy']))]).astype('float32')
            extra=dict(category=category,key='object:%d:%s'%(si,category))
        else:
            row_index=int(self.ids[idx]);row=self.rows[row_index];traj=self.meta['trajectories'][int(row['traj'])];si=traj['scene'];scene=self.meta['scenes'][si]
            if traj['split']!=self.split:raise ValueError('Image split mismatch')
            pose=row['pose'].copy();pose[:2]=(pose[:2]/scene['factor']-np.asarray(scene['offset']))/scene['scale']
            targets=pose[None].astype('float32')
            path=Path(traj['path'])/'rgb'/('rgb_%d.png'%int(row['frame']))
            source=self.image_payload(row_index) if self.lmdb_path else path
            with Image.open(source) as im:rgb=image_tensor(im,224)
            extra=dict(rgb=rgb,key='image:%d:%d'%(int(row['traj']),int(row['frame'])))
        scene=self.meta['scenes'][si]
        origin=-np.asarray(scene['offset'],dtype=np.float32)/scene['scale']
        extent=np.asarray(scene['map_size'],dtype=np.float32)/(scene['factor']*scene['scale'])
        return dict(task=self.task,floorplan=self.floor(si),origin=torch.from_numpy(origin),extent=torch.from_numpy(extent),targets=torch.from_numpy(targets),scene=si,**extra)

class EpochSamples(Dataset):
    """Fixed object exposure per epoch; joint runs add an equal image budget."""
    def __init__(self,data,count,seed,epoch):self.data=data;self.count=count;self.seed=seed;self.epoch=epoch
    def __len__(self):return self.count
    def __getitem__(self,index):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index]))
        if self.data.task=='object':
            # Uniform categories -> uniform scene/category queries -> uniform
            # distinct target positions in the diffusion loss.
            group=self.data.category_groups[int(rng.integers(len(self.data.category_groups)))]
            i=int(group[int(rng.integers(len(group)))])
        else:i=int(rng.integers(len(self.data)))
        return self.data[i]

class FullEpochSamples(EpochSamples):
    """Full image permutation; mandatory object positives plus balanced extras.

    Mandatory object slots are spread uniformly through the epoch. Their first
    diffusion particle uses the specified positive; other particles remain random.
    This preserves multi-positive supervision and makes coverage auditable.
    """
    def __init__(self,data,count,seed,epoch):
        super().__init__(data,count,seed,epoch)
        rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,1701]))
        self.mandatory={};self.order=None
        if data.task=='image':
            if count<len(data):raise ValueError('Full epoch cannot truncate images')
            self.order=np.resize(rng.permutation(len(data)),count)
            self.audit=dict(unique_images=len(data),presentations=count,padding=count-len(data),order_sha256=hashlib.sha256(self.order.tobytes()).hexdigest())
        else:
            pairs=[(q,p) for q,x in enumerate(data.queries) for p in range(len(x['world_xy']))]
            if count<len(pairs):raise ValueError('Epoch cannot cover all object positions')
            slots=rng.choice(count,len(pairs),replace=False)
            self.mandatory={int(slot):pairs[int(j)] for slot,j in zip(slots,rng.permutation(len(pairs)))}
            self.audit=dict(unique_queries=len(data),mandatory_positive_positions=len(pairs),presentations=count,balanced_extra_presentations=count-len(pairs),schedule_sha256=hashlib.sha256(json.dumps(sorted(self.mandatory.items())).encode()).hexdigest())

    def __getitem__(self,index):
        if self.order is not None:return self.data[int(self.order[index])]
        if index in self.mandatory:
            q,p=self.mandatory[index]
            return dict(self.data[q],forced_target=p)
        return super().__getitem__(index)

def collate(rows):
    if len({x['task'] for x in rows})!=1:raise ValueError('Use homogeneous modality batches')
    n=max(len(x['targets']) for x in rows);targets=torch.zeros(len(rows),n,3);mask=torch.zeros(len(rows),n,dtype=torch.bool)
    for i,r in enumerate(rows):targets[i,:len(r['targets'])]=r['targets'];mask[i,:len(r['targets'])]=True
    out={k:torch.stack([r[k] for r in rows]) for k in ('floorplan','origin','extent')}
    out.update(task=rows[0]['task'],targets=targets,target_mask=mask,key=[r['key'] for r in rows],scene=[r['scene'] for r in rows])
    if any('forced_target' in r for r in rows):out['forced_target']=torch.tensor([r.get('forced_target',-1) for r in rows],dtype=torch.long)
    if out['task']=='object':out['category']=[r['category'] for r in rows]
    else:out['rgb']=torch.stack([r['rgb'] for r in rows])
    return out

def featurize(batch,features,device):
    out={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
    if out['task']=='object':out['text_features']=features.text(out['category'])
    else:out.update(features.image(out.pop('rgb')))
    return out
