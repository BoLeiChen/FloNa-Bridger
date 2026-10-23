"""Offline held-out localization evaluation, never a navigation simulator test."""
import argparse,hashlib,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from .data import LocalizationData,collate,featurize
from .encoders import FrozenFeatures
from .model import MultimodalLocalizer,FORMAT
from .metrics import rank_candidates,query_metrics

def evaluation_indices(data,limit,seed=713,image_scope='last_available'):
    if image_scope not in ('last_available','all_frames'):raise ValueError('Unknown image evaluation scope')
    if data.task=='image' and image_scope=='last_available':
        # ImageGoal-facing evaluation: one last AVAILABLE indexed image per
        # trajectory, rather than many nearly identical adjacent frames.
        rows=data.rows[data.ids];last={}
        for i,row in enumerate(rows):
            key=int(row['traj']);frame=int(row['frame'])
            if key not in last or frame>last[key][0]:last[key]=(frame,i)
        ids=np.array([v[1] for v in last.values()],dtype=np.int64)
    else:ids=np.arange(len(data),dtype=np.int64)
    if limit and len(ids)>limit:
        ids=np.random.default_rng(seed).choice(ids,int(limit),replace=False)
    return sorted(ids.tolist())

@torch.no_grad()
def evaluate_model(model,features,config,split='val',limit=None,device='cpu',output=None):
    ranking_mode=config.get('candidate_ranking','pose')
    scope=config.get('image_eval_scope','last_available')
    model.eval();result={'split':split,'tasks':{},'navigation_metrics':False,'candidate_ranking':ranking_mode,'image_eval_scope':scope};records=[]
    if output:Path(output).mkdir(parents=True,exist_ok=True)
    for task in (['object'] if config['setting']=='object_only' else ['object','image']):
        data=LocalizationData(config['data_root'],split,task,config.get('image_lmdb_root'))
        cap=limit if limit is not None else int(config.get(task+'_eval_limit',0))
        available=len(evaluation_indices(data,0,image_scope=scope))
        indices=evaluation_indices(data,cap,image_scope=scope);taskrows=[];catrows=defaultdict(list);begin=time.monotonic()
        for index in indices:
            example=data[index];batch=featurize(collate([example]),features,device)
            seed=int.from_bytes(hashlib.sha256((str(config['seed'])+example['key']).encode()).digest()[:8],'little')%(2**63-1)
            generator=torch.Generator(device=device).manual_seed(seed)
            candidates=model.sample(batch,int(config.get('eval_candidates',64)),int(config.get('sample_steps',20)),generator)[0].cpu().numpy()
            keep,weights=rank_candidates(candidates,task,ranking_mode=ranking_mode)
            targets=example['targets'].numpy();metrics=query_metrics(candidates[keep],targets)
            metrics['raw64_hit_1m']=float(np.linalg.norm(candidates[:,None,:2]-targets[None,:,:2],axis=-1).min()<1.)
            metrics['retained_candidates']=len(keep)
            if task=='image':
                delta=abs(np.arctan2(np.sin(candidates[keep[0],2]-targets[0,2]),np.cos(candidates[keep[0],2]-targets[0,2])))
                metrics['top1_yaw_error_deg']=float(np.rad2deg(delta))
                metrics['top1_1m_30deg']=float(metrics['top1_error_m']<1. and delta<np.pi/6)
            taskrows.append(metrics)
            if task=='object':catrows[example['category']].append(metrics)
            records.append(dict(key=example['key'],task=task,category=example.get('category'),metrics=metrics,candidates_xy=candidates[keep,:2].tolist(),relative_weights=weights.tolist()))
            if len(taskrows)%100==0 or len(taskrows)==len(indices):
                elapsed=time.monotonic()-begin
                progress=dict(split=split,task=task,done=len(taskrows),total=len(indices),elapsed_s=elapsed,eta_s=elapsed/max(1,len(taskrows))*(len(indices)-len(taskrows)))
                print('EVAL_PROGRESS '+json.dumps(progress),flush=True)
                if output:(Path(output)/'progress.json').write_text(json.dumps(progress,indent=2))
        if not taskrows:raise ValueError('Empty evaluation')
        averaged={k:float(np.mean([r[k] for r in taskrows])) for k in taskrows[0]}
        result['tasks'][task]=dict(count=len(taskrows),available_queries=available,limited=bool(cap and available>cap),metrics=averaged)
        if catrows:
            percat={c:{k:float(np.mean([r[k] for r in values])) for k in values[0]} for c,values in catrows.items()}
            result['tasks'][task]['category_macro']={k:float(np.mean([v[k] for v in percat.values()])) for k in averaged}
            result['tasks'][task]['per_category']=percat
        data.close()
    obj=result['tasks']['object']['metrics'];result['object_selection_score']=(obj['hit_1m_top1']+obj['hit_1m_top16'])/2
    if 'image' in result['tasks']:
        img=result['tasks']['image']['metrics'];result['image_selection_score']=(img['hit_1m_top1']+img['hit_1m_top16'])/2
        result['joint_selection_score']=(result['object_selection_score']+result['image_selection_score'])/2
    else:result['joint_selection_score']=result['object_selection_score']
    if output:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        (output/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        (output/'predictions.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--split',choices=['val','test'],default='test');p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--limit',type=int);p.add_argument('--clip-checkpoint');p.add_argument('--depth-checkpoint');a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError(a.output)
    saved=torch.load(a.checkpoint,map_location='cpu')
    if saved.get('format')!=FORMAT:raise ValueError('Checkpoint format mismatch')
    cfg=saved['config'];cfg.update({k:v for k,v in vars(a).items() if k in ('clip_checkpoint','depth_checkpoint') and v})
    if (Path(cfg['data_root'])/'COMPLETE').read_text()!=saved['dataset_fingerprint']:raise ValueError('Evaluation dataset differs from training dataset')
    features=FrozenFeatures(cfg,a.device);model=MultimodalLocalizer(cfg).to(a.device);model.load_state_dict(saved['model'],strict=True)
    result=evaluate_model(model,features,cfg,a.split,a.limit,a.device,a.output)
    (Path(a.output)/'provenance.json').write_text(json.dumps(dict(checkpoint=str(Path(a.checkpoint).resolve()),dataset_fingerprint=saved['dataset_fingerprint'],epoch=saved['epoch'],setting=cfg['setting']),indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='tasks'},indent=2))

if __name__=='__main__':main()
