"""Frozen image-only localization hypotheses. K-independent, provenance-keyed cache."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from localization.image_goal import goal_localization_seed

VERSION='goal-hypotheses-nms64-disco-v1'

def signature(config):
    if config.get('route_task')=='object':
        return hashlib.sha256(json.dumps(dict(version='object-hypotheses-xy-disco-v1',localization=config['object_localization']),sort_keys=True).encode()).hexdigest()
    return hashlib.sha256(json.dumps(dict(version=VERSION,localization=config['goal_localization']),sort_keys=True).encode()).hexdigest()

class GoalHypothesisProvider:
    def __init__(self,config,device='cpu',allow_inference=False):
        self.config=config;self.device=torch.device(device);self.allow_inference=allow_inference
        self.signature=signature(config);self.root=Path(config['goal_hypothesis_cache'])/self.signature
        self.engine=None;self.scene=None
    def path(self,image_path):
        key=str(Path(image_path).resolve());digest=hashlib.sha256(key.encode()).hexdigest()
        return key,self.root/digest[:2]/(digest+'.npz')
    def get(self,image_path):
        key,path=self.path(image_path)
        if not path.exists():
            if not self.allow_inference:raise FileNotFoundError('Prepare goal hypotheses before training: '+key)
            record=self._infer(key);path.parent.mkdir(parents=True,exist_ok=True)
            temp=path.with_name(path.stem+'.%d.tmp.npz'%os.getpid())
            np.savez_compressed(temp,goal_path=key,signature=self.signature,**record);os.replace(temp,path)
        with np.load(path,allow_pickle=False) as data:
            if str(data['goal_path'])!=key or str(data['signature'])!=self.signature:raise ValueError('Goal cache provenance mismatch')
            record={k:data[k].copy() for k in ('poses','ids','scores','density','match_logits')}
        n=len(record['poses'])
        if not 1<=n<=64 or record['poses'].shape!=(n,3) or len(set(record['ids'].tolist()))!=n:raise ValueError('Invalid hypotheses')
        if any(not np.isfinite(v).all() for v in record.values()):raise ValueError('Nonfinite hypotheses')
        if any(v.shape!=(n,) for k,v in record.items() if k!='poses'):raise ValueError('Invalid hypothesis fields')
        return record
    @torch.inference_mode()
    def _infer(self,key):
        from localization.cf2loc import CF2LocEngine
        from localization.disco import diverse_modes,crop_maps,fused_scores
        cfg=dict(self.config['goal_localization'])
        if not cfg.get('cf2loc_disco_enabled') or cfg['num_particles']!=64:raise ValueError('Requires frozen 64-particle diffusion + DisCo')
        cfg['cf2loc_refiner_enabled']=False # Preserve audited 67.55% proposal definition.
        devices=[] if self.device.type=='cpu' else [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        # Construction and sampling must not perturb navigation/training random streams.
        with torch.random.fork_rng(devices=devices):
            if self.engine is None:self.engine=CF2LocEngine(cfg,self.device)
            goal=Path(key);root=goal.parents[2];scene=root.name;traj=goal.parents[1].name
            if self.scene!=str(root):
                name,floor=scene.rsplit('_',1);floor=int(floor);geo=Path(cfg['gibson_dataset_root'])/name
                scale=float([x for x in (geo/'scale.txt').read_text().splitlines() if x.strip()][floor])
                offset=np.array([float(v) for v in [x for x in (geo/'offset.txt').read_text().splitlines() if x.strip()][floor].split()[:2]])
                self.engine.reset_scene(root/'floorplan.png',scale,offset);self.scene=str(root)
            seed=goal_localization_seed(0,scene,traj);torch.random.default_generator.manual_seed(seed)
            if devices:torch.cuda.manual_seed(seed)
            with Image.open(goal) as image:observation=np.asarray(image.convert('RGB'))
            _,samples,_,density=self.engine._sample_pose_candidates(observation)
            ranker=self.engine.disco;c=ranker.config
            ids=diverse_modes(samples,density,c['map_res'],64,c['nms_m'],c['nms_deg'])
            logits=[]
            for start in range(0,len(ids),8):
                maps=crop_maps(self.engine._raw_map,samples[ids[start:start+8]].cpu().numpy(),c).unsqueeze(0).to(self.device)
                with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
                    logits.append(ranker.model(self.engine._disco_image_tokens.half().float(),maps).float())
            logits=torch.cat(logits,1)
            scores=fused_scores(logits,density[ids][None],torch.ones_like(logits,dtype=torch.bool),ranker.weight)[0]
            order=torch.argsort(scores,descending=True,stable=True);ids=ids[order]
            return dict(poses=np.stack([self.engine._map_pose_to_world(p) for p in samples[ids].cpu().numpy()]).astype(np.float32),
                ids=ids.cpu().numpy().astype(np.int64),scores=scores[order].cpu().numpy(),density=density[ids].cpu().numpy(),match_logits=logits[0,order].cpu().numpy())
