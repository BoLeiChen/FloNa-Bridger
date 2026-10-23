"""Train only DisCo on fixed HM3D candidates. Select checkpoints on VAL only."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import yaml
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from localization.disco import DisCoMatcher, fused_scores, ranking_loss
from prepare_disco_hm3d import digest, atomic_json


class CachedQueries(Dataset):
    def __init__(self,run,split):
        self.shards=[]; self.ends=[]; self.count=0
        manifest=Path(run)/'manifest.json'
        info=json.loads(manifest.read_text()); expected=digest(manifest)
        roots=sorted((Path(run)/'cache'/split).iterdir(),key=lambda p:int(p.name))
        for path in roots:
            done=json.loads((path/'complete.json').read_text())
            if done['manifest_sha256']!=expected or done['split']!=split or done['gt_candidates_added']!=(split=='train'):
                raise ValueError('Cache provenance mismatch')
            shard={key:np.load(path/(key+'.npy'),mmap_mode='r') for key in ('tokens','maps','poses','valid','density','errors','gt')}
            if any(len(v)!=done['frames'] for v in shard.values()):raise ValueError('Truncated cache')
            self.shards.append(shard);self.count+=done['frames'];self.ends.append(self.count)
        if self.count!=info['splits'][split]['frames']: raise ValueError('Incomplete cache')
    def __len__(self):return self.count
    def __getitem__(self,idx):
        s=int(np.searchsorted(self.ends,idx,side='right')); j=idx-(self.ends[s-1] if s else 0)
        return {key:torch.from_numpy(np.array(value[j],copy=True)) for key,value in self.shards[s].items()}


def to_device(batch,device):
    return {k:v.to(device,non_blocking=True) for k,v in batch.items()}


@torch.inference_mode()
def validate(model,loader,device,weights):
    model.eval()
    # n, baseline R1, baseline mean, oracle R1; per weight R1/R.5/mean/yaw30joint
    totals=torch.zeros(4+4*len(weights),device=device,dtype=torch.float64)
    for batch in loader:
        b=to_device(batch,device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            scores=model(b['tokens'].float(),b['maps'].float()/255.)
        errors=b['errors']; valid=b['valid']; n=len(errors)
        totals[0]+=n; totals[1]+=(errors[:,0,0]<1).sum();totals[2]+=errors[:,0,0].sum()
        totals[3]+=((errors[:,:,0]<1)&valid).any(-1).sum()
        for i,weight in enumerate(weights):
            ids=fused_scores(scores,b['density'],valid,weight).argmax(-1)
            e=errors[torch.arange(n,device=device),ids]
            totals[4+i*4:8+i*4]+=torch.stack(((e[:,0]<1).sum(),(e[:,0]<.5).sum(),e[:,0].sum(),((e[:,0]<1)&(e[:,1]<np.pi/6)).sum())).double()
    if dist.is_initialized():dist.all_reduce(totals)
    values=totals.cpu().tolist();n=values[0]
    if n<=0:raise ValueError('Empty validation')
    rows=[{'weight':w,'recall_1m':values[4+4*i]/n,'recall_05m':values[5+4*i]/n,
           'mean_xy_m':values[6+4*i]/n,'recall_1m_30deg':values[7+4*i]/n} for i,w in enumerate(weights)]
    best=max(rows,key=lambda v:(v['recall_1m'],-v['mean_xy_m']))
    return {'frames':int(n),'stage1_recall_1m':values[1]/n,'stage1_mean_xy_m':values[2]/n,
            'mode_oracle_recall_1m':values[3]/n,'fusion_grid':rows,'selected':best}


def save_checkpoint(path,state):
    tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true')
    args=p.parse_args();c=yaml.safe_load(Path(args.config).read_text())
    world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0))
    device=torch.device('cuda',int(os.environ.get('LOCAL_RANK',0)));torch.cuda.set_device(device);torch.set_num_threads(4)
    if world>1:dist.init_process_group('nccl')
    torch.manual_seed(c['seed']);np.random.seed(c['seed'])
    run=Path(c['run_dir']);manifest=json.loads((run/'manifest.json').read_text())
    if manifest['config']!=c:raise ValueError('Training config differs from audited preparation')
    train=CachedQueries(run,'train');val=CachedQueries(run,'val')
    sampler=DistributedSampler(train,num_replicas=world,rank=rank,shuffle=True,seed=c['seed'],drop_last=True)
    # Exact nonpadding sharding: no duplicated validation examples.
    val_ids=list(range(rank,len(val),world))
    train_loader=DataLoader(train,batch_size=c['batch_size'],sampler=sampler,drop_last=True,num_workers=c['workers'],pin_memory=True,persistent_workers=c['workers']>0)
    val_loader=DataLoader(torch.utils.data.Subset(val,val_ids),batch_size=c['val_batch_size'],shuffle=False,num_workers=c['workers'],pin_memory=True)
    model=DisCoMatcher(c).to(device)
    if digest(c['refiner_checkpoint'])!=c['refiner_sha256']:raise ValueError('Refiner SHA mismatch')
    model.initialize_map_from_refiner(torch.load(c['refiner_checkpoint'],map_location='cpu'))
    if world>1:model=DDP(model,device_ids=[device.index],broadcast_buffers=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(c['lr']),weight_decay=float(c['weight_decay']))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,c['epochs'],eta_min=float(c['min_lr']))
    best_key=(-1.,-float('inf'));start_epoch=0;stale=0;best_record=None
    if args.resume:
        state=torch.load(run/'last.pt',map_location='cpu')
        if state['manifest_sha256']!=digest(run/'manifest.json'):raise ValueError('Resume manifest mismatch')
        (model.module if world>1 else model).load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start_epoch=state['epoch'];best_key=tuple(state['best_key']);best_record=state['best_record'];stale=state['stale']
    started=time.time()
    epochs_completed=start_epoch
    for epoch in range(start_epoch,c['epochs']):
        epochs_completed=epoch+1
        sampler.set_epoch(epoch);torch.manual_seed(c['seed']+epoch*world+rank)
        model.train();loss_sum=0.;steps=0
        for batch in train_loader:
            b=to_device(batch,device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                scores=model(b['tokens'].float(),b['maps'].float()/255.)
                loss=ranking_loss(scores,b['errors'][:,:,0],b['errors'][:,:,1],b['valid'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            steps+=1;loss_sum+=float(loss.detach())
            if rank==0 and (steps==1 or steps%100==0):
                print(json.dumps({'stage':'train','epoch':epoch+1,'step':steps,'steps_per_epoch':len(train_loader),'loss':loss_sum/steps,'elapsed_s':round(time.time()-started)}),flush=True)
            if args.smoke and steps==3:break
        scheduler.step()
        raw=model.module if world>1 else model
        metrics=validate(raw,val_loader,device,c['fusion_weights'])
        selected=metrics['selected'];key=(selected['recall_1m'],-selected['mean_xy_m'])
        improved=key>best_key
        if improved:best_key=key;stale=0;best_record={'epoch':epoch+1,**metrics}
        else:stale+=1
        if rank==0:
            print('VALIDATION '+json.dumps({'epoch':epoch+1,**metrics}),flush=True)
            with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps({'epoch':epoch+1,'train_loss_rank0':loss_sum/steps,**metrics})+'\n')
            state={'format':'floverse_disco_v1','config':c,'model':raw.state_dict(),'epoch':epoch+1,
                   'fusion_weight':selected['weight'],'validation':metrics,'manifest_sha256':digest(run/'manifest.json')}
            if improved:
                save_checkpoint(run/'best.pt',state)
                atomic_json(run/'best.json',{'checkpoint':str(run/'best.pt'),'checkpoint_sha256':digest(run/'best.pt'),
                    'selection_metric':'val_recall_1m_then_lower_mean_xy_m','navigation_test_used':False,
                    'diffusion_sha256':c['diffusion_sha256'],'refiner_sha256':c['refiner_sha256'],**best_record})
            save_checkpoint(run/'last.pt',{**state,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                                           'best_key':best_key,'best_record':best_record,'stale':stale})
        if world>1:dist.barrier()
        if args.smoke or stale>=c['patience']:break
    if rank==0:
        # Separate opt-in navigation config. Existing evaluation defaults are untouched.
        base=yaml.safe_load((ROOT/'config/imagegoal_v3.yaml').read_text())
        base['goal_localization'].update(cf2loc_disco_enabled=True,cf2loc_disco_checkpoint=str(run/'best.pt'),
                                    cf2loc_disco_checkpoint_sha256=digest(run/'best.pt'),
                                    diffusion_checkpoint=c['diffusion_checkpoint'],diffusion_checkpoint_sha256=c['diffusion_sha256'])
        (run/'imagegoal_disco_best.yaml').write_text(yaml.safe_dump(base,sort_keys=False))
        atomic_json(run/'training_completed.json',{'epochs_completed':epochs_completed,'elapsed_s':time.time()-started,
                    'best':best_record,'navigation_config':str(run/'imagegoal_disco_best.yaml'),'smoke':args.smoke})
    if world>1:dist.destroy_process_group()
if __name__=='__main__':main()
