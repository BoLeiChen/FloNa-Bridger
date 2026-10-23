"""Evaluate a fixed diffusion/ranker pair on held-out localization frames."""
import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from third_party.cf2loc_hm3d.DisCo_model.floverse_full_dataset import FloVerseFullDataset
from third_party.cf2loc_hm3d.DisCo_model.pose_query_diffusion import PoseQueryDiffusionLocalizer
from localization.disco import DisCoRanker


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


class IndexedFrames(Dataset):
    def __init__(self,source,limit):self.source=source;self.count=min(len(source),limit or len(source))
    def __len__(self):return self.count
    def __getitem__(self,i):return i,self.source[i]


@torch.inference_mode()
def evaluate(args):
    if args.limit is not None and args.limit<1:raise ValueError('--limit must be positive')
    if args.batch_size<1:raise ValueError('--batch-size must be positive')
    output=Path(args.output)
    if output.exists():raise FileExistsError('Use a fresh evaluation output: '+str(output))
    c=yaml.safe_load(Path(args.config).read_text());device=torch.device(args.device)
    torch.manual_seed(args.seed);np.random.seed(args.seed);torch.set_num_threads(4)
    parent_hash=digest(c['diffusion_checkpoint'])
    if parent_hash!=c['diffusion_sha256']:raise ValueError('Diffusion checkpoint SHA mismatch')
    checkpoint=torch.load(c['diffusion_checkpoint'],map_location='cpu')
    model_config=dict(checkpoint['hyper_parameters']);model_config['dptv2_ckpt_path']=c['dinov2_checkpoint']
    model=PoseQueryDiffusionLocalizer(model_config).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'],strict=True);del checkpoint
    ranker=DisCoRanker(args.ranker,parent_hash,device,c['map_res']) if args.ranker else None
    source=FloVerseFullDataset(c['index'],args.split,augment=False,lmdb_root=c.get('lmdb_root'))
    data=IndexedFrames(source,args.limit)
    if not len(data):raise ValueError('Empty evaluation dataset')
    loader=DataLoader(data,batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
    captured={}
    hook=model.condition_encoder.image_mixer.register_forward_hook(lambda m,i,o:captured.update(tokens=o.detach()))
    @lru_cache(maxsize=32)
    def raw_map(scene):
        with Image.open(source.scenes[scene]['map_path']) as image:return np.asarray(image.convert('L')).copy()
    totals={'diffusion':{}}
    if ranker:totals['ranked']={}
    count=0
    try:
        for indices,batch in loader:
            obs,target,_,floorplan,wh,*_=batch
            target=target.to(device);context=model.condition_encoder(obs.to(device),floorplan.to(device))
            samples=model.sample_from_context(*context,wh.to(device),num_particles=int(c['particles']),num_steps=int(c['sample_steps']))
            selected,density=model.select_pose_mode(samples)
            predictions={'diffusion':selected}
            if ranker:
                poses=[]
                for j,index in enumerate(indices.tolist()):
                    original=int(source.selection[index]) if source.selection is not None else index
                    scene=int(source.trajectories[int(source.rows[original]['traj'])]['scene'])
                    ranked,_=ranker.rank(captured['tokens'][j:j+1],samples[j],density[j],raw_map(scene))
                    poses.append(ranked[0])
                predictions['ranked']=torch.stack(poses)
            for group,prediction in predictions.items():
                for key,value in model.sample_metrics(samples,prediction,target).items():
                    totals[group][key]=totals[group].get(key,0.)+float(value)*len(indices)
            count+=len(indices)
            if count%(args.batch_size*100)==0:print('Evaluated',count,'/',len(data),flush=True)
    finally:hook.remove()
    report=dict(split=args.split,frames=count,available_frames=len(source),expected_split_frames=len(source.rows),
        complete_split=count==len(source.rows),navigation_metrics=False,seed=args.seed,batch_size=args.batch_size,
        diffusion_sha256=parent_hash,ranker_sha256=digest(args.ranker) if ranker else None,
        fusion_weight=ranker.weight if ranker else None,candidate_modes=int(ranker.config['modes']) if ranker else None,
        metrics={group:{key:value/count for key,value in values.items()} for group,values in totals.items()})
    output.mkdir(parents=True);(output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='config/localization/image_disco.yaml');p.add_argument('--ranker')
    p.add_argument('--split',choices=['val','test'],default='test');p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--seed',type=int,default=42);p.add_argument('--limit',type=int)
    evaluate(p.parse_args())
