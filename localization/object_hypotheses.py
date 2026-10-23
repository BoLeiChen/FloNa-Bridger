"""Category-only Object DisCo provider. No target coordinates or episode RGB accepted."""
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

class ObjectHypothesisProvider:
    def __init__(self,config,device='cpu',allow_inference=False):
        source=Path(config['object_localization']['code_root']).resolve()
        sys.path.insert(0,str(source))
        from cf2loc_mm.object_disco import ObjectDisCoLocalizer
        from cf2loc_mm.encoders import file_hash
        c=config['object_localization'];self.device=torch.device(device)
        # Hash the actual artifacts; changing a checkpoint cannot reuse old caches.
        self.provenance=dict(version='object-hypotheses-xy-disco-v1',diffusion_sha256=file_hash(c['diffusion_checkpoint']),disco_sha256=file_hash(c['disco_checkpoint']))
        for name in ('diffusion','disco'):
            if c.get(name+'_sha256') and c[name+'_sha256']!=self.provenance[name+'_sha256']:raise ValueError('Checkpoint changed: '+name)
        self.engine=None;self.allow_inference=allow_inference;self.config=c
        self.root=Path(config['object_hypothesis_cache']);self.seed=int(config.get('seed',0))
    @torch.inference_mode()
    def get(self,map_path,category,*,origin_xy,extent_xy):
        from cf2loc_mm.prepare import category_name
        from cf2loc_mm.encoders import file_hash
        category=category_name(category)
        if not category or category=='null':raise ValueError('Missing object category')
        identity=dict(self.provenance,map_sha256=file_hash(map_path),category=category,origin=list(map(float,origin_xy)),extent=list(map(float,extent_xy)),seed=self.seed)
        signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();path=self.root/(signature+'.npz')
        if not path.exists():
            if not self.allow_inference:raise FileNotFoundError('Prepare object hypothesis cache before training: '+str(path))
            from cf2loc_mm.object_disco import ObjectDisCoLocalizer
            devices=[] if self.device.type=='cpu' else [self.device.index or 0]
            with torch.random.fork_rng(devices=devices):
                if self.engine is None:self.engine=ObjectDisCoLocalizer(self.config['diffusion_checkpoint'],self.config['disco_checkpoint'],self.device)
                with Image.open(map_path) as im:
                    r=self.engine.predict(im.convert('RGB'),origin_xy=origin_xy,extent_xy=extent_xy,category=category,top_k=64,seed=int(signature[:15],16))
            # Third column is a storage placeholder, never an estimated orientation.
            xy=np.asarray(r['world_xy'],np.float32);poses=np.column_stack([xy,np.zeros(len(xy),np.float32)])
            record=dict(poses=poses,ids=np.asarray(r['ids'],np.int64),scores=np.asarray(r['scores'],np.float32),density=np.asarray(r['density'],np.float32),match_logits=np.asarray(r['match_logits'],np.float32),yaw_valid=np.zeros(len(xy),bool))
            path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.stem+'.%d.tmp.npz'%os.getpid())
            np.savez_compressed(temp,signature=signature,goal_text=self.engine.base.features.text([category])[0].cpu().numpy(),**record);os.replace(temp,path)
        with np.load(path,allow_pickle=False) as data:
            if str(data['signature'])!=signature:raise ValueError('Object cache mismatch')
            goal_text=data['goal_text'].copy()
            record={k:data[k].copy() for k in ('poses','ids','scores','density','match_logits','yaw_valid')}
        n=len(record['poses'])
        if not 1<=n<=64 or record['poses'].shape!=(n,3) or len(set(record['ids'].tolist()))!=n or any(not np.isfinite(v).all() for v in record.values()):raise ValueError('Invalid object candidates')
        if goal_text.shape!=(512,) or not np.isfinite(goal_text).all():raise ValueError('Invalid category feature')
        return dict(record,goal_text=goal_text)
