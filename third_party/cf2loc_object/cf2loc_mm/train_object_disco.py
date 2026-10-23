"""Distributed candidate preparation/training; held-out test only after val selection."""
import argparse,json,os,time,hashlib,random
from pathlib import Path
from collections import Counter,defaultdict
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset,DataLoader,DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from .object_disco import ObjectDisCo,FORMAT,VERSION,candidates_xy,candidate_seed,fused_scores,ranking_loss
from .data import LocalizationData,collate,featurize
from .api import Localizer
from .encoders import file_hash
from .metrics import query_metrics
from .train import atomic_save

def write_json(path,value):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)

def setup():
    rank=int(os.getenv('RANK',0));world=int(os.getenv('WORLD_SIZE',1));local=int(os.getenv('LOCAL_RANK',0));torch.cuda.set_device(local)
    if world>1:dist.init_process_group('nccl',timeout=timedelta(hours=2))
    return rank,world,torch.device('cuda',local)

def provenance(checkpoint):
    s=torch.load(checkpoint,map_location='cpu')
    if s['config']['setting']!='object_only':raise ValueError('Requires a frozen object_only checkpoint selected on validation')
    if (Path(s['config']['data_root'])/'COMPLETE').read_text()!=s['dataset_fingerprint']:raise ValueError('Dataset fingerprint mismatch')
    return dict(diffusion_sha256=file_hash(checkpoint),dataset_fingerprint=s['dataset_fingerprint'],version=VERSION),s

@torch.inference_mode()
def prepare(a):
    rank,world,device=setup();root=Path(a.cache);root.mkdir(parents=True,exist_ok=True)
    prov,saved=provenance(a.diffusion);localizer=Localizer(a.diffusion,device);cfg=localizer.config
    data=LocalizationData(cfg['data_root'],a.split,'object');rows=[];floors={};texts={};start=time.monotonic()
    variants=a.variants if a.split=='train' else 1
    indices=list(range(len(data)))[rank::world]
    if a.limit:indices=indices[:a.limit]
    for done,index in enumerate(indices):
        example=data[index];scene=example['scene'];category=example['category']
        floors[scene]=example['floorplan'].half();texts[category]=localizer.features.text([category])[0].cpu()
        batch=featurize(collate([example]),localizer.features,device)
        for variant in range(variants):
            seed=candidate_seed(cfg['seed'],example['key'],variant)
            poses=localizer.model.sample(batch,64,int(cfg.get('sample_steps',20)),torch.Generator(device=device).manual_seed(seed))[0].cpu().numpy()
            ids,density=candidates_xy(poses);targets=example['targets'].numpy()[:,:2]
            xy=poses[ids,:2];error=np.linalg.norm(xy[:,None]-targets[None],axis=-1).min(-1)
            rows.append(dict(key=example['key'],scene=scene,category=category,variant=variant,seed=seed,xy=xy,density=density[ids],errors=error,targets=targets,origin=example['origin'].numpy(),extent=example['extent'].numpy(),raw_hit=float(np.linalg.norm(poses[:,None,:2]-targets[None],axis=-1).min()<1.)))
        if (done+1)%100==0 or done+1==len(indices):
            progress=dict(stage='prepare',split=a.split,rank=rank,done=done+1,total=len(indices),elapsed_s=time.monotonic()-start)
            write_json(root/('%s_progress_%d.json'%(a.split,rank)),progress);print(json.dumps(progress),flush=True)
    target=root/('%s_%d.pt'%(a.split,rank))
    if target.exists():raise FileExistsError(target)
    atomic_save(dict(provenance=prov,split=a.split,rank=rank,world=world,total_queries=len(data),limited=bool(a.limit),variants=variants,rows=rows,floors=floors,texts=texts),target)
    if world>1:dist.barrier();dist.destroy_process_group()

class Candidates(Dataset):
    def __init__(self,cache,split,prov):
        self.rows=[];self.floors={};self.texts={};seen=set();shards=sorted(Path(cache).glob(split+'_[0-9]*.pt'))
        if not shards:raise ValueError('No completed candidate shards')
        ranks=set()
        for f in shards:
            p=torch.load(f,map_location='cpu')
            if p['provenance']!=prov or p['split']!=split or p['limited']:raise ValueError('Cache provenance/split/coverage mismatch')
            ranks.add(p['rank']);self.floors.update(p['floors']);self.texts.update(p['texts'])
            for row in p['rows']:
                key=(row['key'],row['variant'])
                if key in seen:raise ValueError('Duplicate cached query')
                seen.add(key);self.rows.append(row)
        if ranks!=set(range(p['world'])) or len(seen)!=p['total_queries']*p['variants']:raise ValueError('Incomplete candidate cache')
        counts=Counter(r['category'] for r in self.rows);average=len(self.rows)/len(counts)
        weights=np.array([np.clip(np.sqrt(average/counts[r['category']]),.25,4.) for r in self.rows])
        self.weights=weights/weights.mean()
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i];n=len(row['xy']);out={}
        for key,width in [('xy',2),('density',None),('errors',None)]:
            x=torch.zeros((64,width) if width else (64,),dtype=torch.float32)
            x[:n]=torch.as_tensor(row[key]);out[key]=x
        out.update(mask=torch.arange(64)<n,floor=self.floors[row['scene']].float(),text=self.texts[row['category']].float(),origin=torch.tensor(row['origin']),extent=torch.tensor(row['extent']),index=i,query_weight=torch.tensor(self.weights[i],dtype=torch.float32))
        return out

def device_batch(batch,device):return {k:v.to(device) for k,v in batch.items()}

@torch.no_grad()
def evaluate(model,data,device,weights=(0.,.25,.5,.75,1.),predictions=False):
    model.eval();metrics={w:[] for w in weights};records=[]
    for batch in DataLoader(data,batch_size=8,shuffle=False):
        batch=device_batch(batch,device);logits=model(batch)
        for w in weights:
            score=fused_scores(logits,batch['density'],batch['mask'],w)
            order=score.argsort(dim=-1,descending=True,stable=True).cpu().numpy()
            for j,index in enumerate(batch['index'].cpu().tolist()):
                row=data.rows[index];n=len(row['xy']);ids=order[j,:n];m=query_metrics(row['xy'][ids],row['targets'],ks=(1,4,8,16,32,64))
                m.update(raw64_hit_1m=row['raw_hit'],retained_candidates=n);metrics[w].append(m)
                if predictions:records.append(dict(key=row['key'],category=row['category'],weight=w,world_xy=row['xy'][ids].tolist(),scores=score[j,torch.as_tensor(ids,device=device)].cpu().tolist(),metrics=m))
    results={str(w):{k:float(np.mean([m[k] for m in rows])) for k in rows[0]} for w,rows in metrics.items()}
    return results,records

def audit_data(trainset,val):
    def audit(data):
        groups=defaultdict(list);categories=Counter();positive=[];hard=[];counts=[]
        for r in data.rows:
            groups[r['key']].append(r);categories[r['category']]+=1
            xy=np.asarray(r['xy']);targets=np.asarray(r['targets']);origin=r['origin'];extent=r['extent']
            if not np.isfinite(xy).all() or not np.isfinite(targets).all():raise ValueError('Nonfinite training coordinates')
            if (extent<=0).any() or ((targets-origin)/extent < -1e-5).any() or ((targets-origin)/extent > 1.00001).any():raise ValueError('Target outside calibrated map')
            errors=np.linalg.norm(xy[:,None]-targets[None],axis=-1).min(-1)
            if not np.allclose(errors,r['errors'],atol=1e-5):raise ValueError('Candidate label mismatch')
            counts.append(len(xy));positive.append(int((errors<1).sum()));hard.append(int(((errors>=1)&(errors<3)).sum()))
        covered=0;targets_total=0;query_hit=0
        for values in groups.values():
            targets=values[0]['targets'];xy=np.concatenate([r['xy'] for r in values]);d=np.linalg.norm(xy[:,None]-targets[None],axis=-1)
            hit=d.min(0)<1;covered+=int(hit.sum());targets_total+=len(targets);query_hit+=int(hit.any())
        return dict(queries=len(groups),candidate_pools=len(data),categories=len(categories),mean_candidates=float(np.mean(counts)),positive_pool_fraction=float(np.mean(np.array(positive)>0)),all_negative_pool_fraction=float(np.mean(np.array(positive)==0)),query_any_positive_fraction=query_hit/len(groups),annotated_positions=targets_total,covered_annotated_positions=covered,mean_positive_candidates=float(np.mean(positive)),mean_hard_negatives_1to3m=float(np.mean(hard)),category_pool_counts=dict(categories),query_weight_min=float(data.weights.min()),query_weight_max=float(data.weights.max()))
    train_scenes={r['scene'] for r in trainset.rows};val_scenes={r['scene'] for r in val.rows}
    if train_scenes&val_scenes:raise ValueError('Train/validation scene overlap')
    return dict(train=audit(trainset),validation=audit(val),scene_overlap=0,building_split_verified_by_source_loader=True,test_used_for_training=False,gt_candidates_injected=False,category_weight='clipped inverse sqrt frequency; all training queries retained',label_semantics='distance to any annotated class-associated observer position')

def train(a):
    rank,world,device=setup();random.seed(42);np.random.seed(42);torch.manual_seed(42)
    prov,_=provenance(a.diffusion);trainset=Candidates(a.cache,'train',prov);val=Candidates(a.cache,'val',prov) if rank==0 else None
    root=Path(a.output)
    if rank==0:root.mkdir(parents=True,exist_ok=False)
    if world>1:dist.barrier()
    model=ObjectDisCo().to(device);wrapped=DDP(model,device_ids=[device.index]) if world>1 else model
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,a.epochs)
    sampler=DistributedSampler(trainset,num_replicas=world,rank=rank,shuffle=True,seed=42)
    loader=DataLoader(trainset,batch_size=8,sampler=sampler,num_workers=0)
    if rank==0:write_json(root/'data_quality_audit.json',audit_data(trainset,val))
    best=-1.;step=0;start=time.monotonic()
    for epoch in range(a.epochs):
        sampler.set_epoch(epoch);model.train();losses=[]
        for batch in loader:
            batch=device_batch(batch,device);optimizer.zero_grad(set_to_none=True);logits=wrapped(batch);loss=ranking_loss(logits,batch['errors'],batch['mask'],batch['query_weight'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite DisCo loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();step+=1;losses.append(float(loss.detach()))
            if rank==0 and step%20==0:write_json(root/'progress.json',dict(stage='train',epoch=epoch+1,epochs=a.epochs,step=step,steps_per_epoch=len(loader),loss=float(loss.detach()),elapsed_s=time.monotonic()-start))
        scheduler.step()
        if rank==0:
            results,_=evaluate(model,val,device)
            weight=max(results,key=lambda w:(results[w]['hit_1m_top1']+results[w]['hit_1m_top16'])/2)
            score=(results[weight]['hit_1m_top1']+results[weight]['hit_1m_top16'])/2
            record=dict(epoch=epoch+1,loss=float(np.mean(losses)),fusion_weight=float(weight),selection_score=score,metrics=results,elapsed_s=time.monotonic()-start)
            write_json(root/('val_%03d.json'%(epoch+1)),record)
            package=dict(format=FORMAT,model=model.state_dict(),fusion_weight=float(weight),epoch=epoch+1,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),validation=record,**prov)
            atomic_save(package,root/'last.pt')
            if score>best:best=score;atomic_save(package,root/'best.pt')
            print(json.dumps({k:v for k,v in record.items() if k!='metrics'}),flush=True)
        if world>1:dist.barrier()
    if rank==0:write_json(root/'status.json',dict(stage='trained',epochs=a.epochs,best_validation_score=best))
    if world>1:dist.destroy_process_group()

def test(a):
    prov,_=provenance(a.diffusion);saved=torch.load(Path(a.output)/'best.pt',map_location='cpu')
    if any(saved[k]!=v for k,v in prov.items()):raise ValueError('Checkpoint provenance mismatch')
    data=Candidates(a.cache,'test',prov);model=ObjectDisCo().cuda().eval();model.load_state_dict(saved['model'])
    weights=tuple(dict.fromkeys([0.,saved['fusion_weight']]))
    result,records=evaluate(model,data,'cuda:0',weights,True)
    root=Path(a.output);write_json(root/'test_metrics.json',dict(count=len(data),split='test',epoch=saved['epoch'],selected_weight=saved['fusion_weight'],baseline=result['0.0'],reranked=result[str(saved['fusion_weight'])],navigation_metrics=False))
    (root/'test_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    write_json(root/'status.json',dict(stage='complete',epoch=saved['epoch'],test_queries=len(data)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','test']);p.add_argument('--diffusion',required=True);p.add_argument('--cache',required=True);p.add_argument('--output');p.add_argument('--split',choices=['train','val','test'],default='train');p.add_argument('--variants',type=int,default=8);p.add_argument('--epochs',type=int,default=30);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    {'prepare':prepare,'train':train,'test':test}[a.action](a)
