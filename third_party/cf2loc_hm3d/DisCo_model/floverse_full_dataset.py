"""Compact full-resolution HM3D dataset for diffusion-only training."""
import json, math, os, io, hashlib
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from .disco_dataset import DisCo_Dataset

DTYPE=np.dtype([('traj','<u4'),('frame','<u4'),('pose','<f4',(3,))])

class FloVerseFullDataset(Dataset):
    def __init__(self, index_path, split, augment=False, lmdb_root=None):
        self.meta=json.loads(Path(index_path).read_text())
        self.split=split
        self.augment=augment
        self.map_res=self.meta['map_res']
        self.scenes=self.meta['scenes']
        self.trajectories=self.meta['trajectories']
        self.info=self.meta['splits'][split]
        self.rows=np.memmap(self.info['index_path'],dtype=DTYPE,mode='r')
        assert len(self.rows)==self.info['frames']
        self.lmdb_path=str(Path(lmdb_root)/split) if lmdb_root else None
        self.selection=None
        self._env=None
        self._env_pid=None
        if self.lmdb_path:
            schema=json.loads((Path(self.lmdb_path)/'complete.json').read_text())
            if schema['index_sha256']!=hashlib.sha256(Path(index_path).read_bytes()).hexdigest() or schema['split']!=split:
                raise ValueError('LMDB source index mismatch')
            self.selection=np.load(Path(self.lmdb_path)/'selection.npy',mmap_mode='r')
            if len(self.selection)!=schema['frames'] or hashlib.sha256(self.selection.tobytes()).hexdigest()!=schema['selection_sha256']:
                raise ValueError('LMDB selection mismatch')
        sets={k:set(v['buildings']) for k,v in self.meta['splits'].items()}
        assert not (sets['train'] & sets['val'] or sets['train'] & sets['test'] or sets['val'] & sets['test'])
        for s in self.scenes:
            if s['source']!='HM3D' or 'Gibson' in s['path']:
                raise ValueError('Non-HM3D source rejected')
        self.transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
        self.unused_ray=torch.zeros(40)
        self.unused_local=torch.zeros(1,1,1)

    def __len__(self): return len(self.selection) if self.selection is not None else len(self.rows)

    def _image_payload(self, idx):
        import lmdb
        pid=os.getpid()
        if self._env is None or self._env_pid!=pid:
            if self._env is not None: self._env.close()
            self._env=lmdb.open(self.lmdb_path,readonly=True,lock=False,readahead=True,max_readers=256,subdir=True)
            self._env_pid=pid
        with self._env.begin() as txn:
            value=txn.get(b'r'+int(idx).to_bytes(8,'big'))
        if value is None: raise KeyError(f'Missing LMDB record {self.split}/{idx}')
        return value

    @lru_cache(maxsize=64)
    def _map(self, si):
        scene=self.scenes[si]
        with Image.open(scene['map_path']) as image:
            image=image.convert('RGB').resize((256,256),Image.Resampling.BICUBIC)
            return transforms.functional.to_tensor(image)

    def __getitem__(self, idx):
        original_idx=int(self.selection[idx]) if self.selection is not None else idx
        row=self.rows[original_idx]; traj=self.trajectories[int(row['traj'])]
        assert traj['split']==self.split
        si=traj['scene']; scene=self.scenes[si]
        path=os.path.join(traj['path'],'rgb',f"rgb_{int(row['frame'])}.png")
        source=io.BytesIO(self._image_payload(idx)) if self.lmdb_path else path
        with Image.open(source) as image:
            if image.size!=(256,256): raise ValueError(f'Unexpected image dimensions: {path}')
            obs=self.transform(image.convert('RGB'))
        pose=row['pose'].copy(); w,h=scene['map_size']
        fmap=self._map(si)
        if self.augment and np.random.random()<.5:
            k=int(np.random.randint(1,4))
            pose,w,h=DisCo_Dataset._rotate_pose_wh_90(pose,w,h,k)
            fmap=torch.rot90(fmap,k,(1,2))
        pose=torch.from_numpy(pose)
        return obs,pose,self.unused_ray,fmap,torch.tensor([w,h],dtype=torch.float32),self.unused_local,self.unused_local,pose

class LocalityDistributedSampler(Sampler):
    """Shuffle blocks and frames, then shard the SAME order across DDP ranks.

    Every rank has equal, complete batches. Only the normal final partial
    global batch is dropped, with different frames dropped each epoch.
    """
    def __init__(self, size, rank, world_size, batch_size, block_size=2048, seed=42):
        self.size=size; self.rank=rank; self.world_size=world_size
        self.batch_size=batch_size; self.block_size=block_size; self.seed=seed; self.epoch=0
        self.total=(size//(world_size*batch_size))*(world_size*batch_size)
    def set_epoch(self, epoch): self.epoch=epoch
    def __len__(self): return self.total//self.world_size
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch)
        blocks=rng.permutation(math.ceil(self.size/self.block_size))
        offset=0
        for b in blocks:
            start=int(b)*self.block_size
            indices=np.arange(start,min(start+self.block_size,self.size),dtype=np.int64)
            rng.shuffle(indices)
            keep=min(len(indices),self.total-offset)
            if keep<=0: break
            first=(self.rank-offset)%self.world_size
            yield from indices[first:keep:self.world_size].tolist()
            offset+=keep
