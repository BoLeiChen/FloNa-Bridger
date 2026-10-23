"""Single-GPU/CPU/torchrun training with paired task budgets and exact resume."""
import argparse,copy,json,os,random,time,math
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
import yaml
from .data import LocalizationData,EpochSamples,FullEpochSamples,collate,featurize
from .encoders import FrozenFeatures,file_hash
from .model import MultimodalLocalizer,FORMAT
from .evaluate import evaluate_model


def rng_state():
    return dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)

def restore_rng(state):
    torch.set_rng_state(state['torch']);np.random.set_state(state['numpy']);random.setstate(state['python'])
    if state['cuda'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state(state['cuda'])

def atomic_save(package,path):
    temp=path.with_suffix('.tmp');torch.save(package,temp);os.replace(temp,path)

def prepare_run(run,config,resume,rank,world):
    # Only rank zero inspects/creates the directory. Otherwise a slower worker
    # can mistake the directory just created by rank zero for an existing run.
    error=[None]
    if rank==0:
        try:
            run.mkdir(parents=True,exist_ok=bool(resume))
            (run/'config.yaml').write_text(yaml.safe_dump(config))
        except OSError as exc:
            error[0]='Cannot prepare run directory: '+str(exc)
    if world>1:dist.broadcast_object_list(error,src=0)
    if error[0] is not None:raise RuntimeError(error[0])

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run',required=True);p.add_argument('--device',default='cuda');p.add_argument('--resume');p.add_argument('--max-steps',type=int);p.add_argument('--eval-limit',type=int);a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text());rank=int(os.getenv('RANK',0));world=int(os.getenv('WORLD_SIZE',1));local=int(os.getenv('LOCAL_RANK',0))
    device=torch.device('cpu' if a.device=='cpu' else ('cuda:%d'%local))
    if device.type=='cuda':torch.cuda.set_device(device)
    if world>1:dist.init_process_group('gloo' if device.type=='cpu' else 'nccl',timeout=timedelta(seconds=int(cfg.get('distributed_timeout_seconds',14400))))
    random.seed(cfg['seed']+rank);np.random.seed(cfg['seed']+rank);torch.manual_seed(cfg['seed']+rank)
    for name in (['clip','depth'] if cfg['setting']=='joint_geometry' else ['clip']):
        digest=file_hash(cfg[name+'_checkpoint'])
        if cfg.get(name+'_sha256') and cfg[name+'_sha256']!=digest:raise ValueError(name+' checkpoint hash mismatch')
        cfg[name+'_sha256']=digest
    root=Path(cfg['data_root']);fingerprint=(root/'COMPLETE').read_text()
    run=Path(a.run)
    prepare_run(run,cfg,a.resume,rank,world)
    features=FrozenFeatures(cfg,device)
    if features.dim!=int(cfg.get('clip_dim',512)):raise ValueError('CLIP output dimension mismatch')
    model=MultimodalLocalizer(cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg['learning_rate']),weight_decay=float(cfg.get('weight_decay',.01)))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=int(cfg['epochs']))
    step=0;start_epoch=0;skip=0;best={};resume_rng=None
    if a.resume:
        saved=torch.load(a.resume,map_location='cpu')
        if saved.get('format')!=FORMAT or saved['config']!=cfg or saved['world_size']!=world or saved['dataset_fingerprint']!=fingerprint:raise ValueError('Resume requires matching format/config/world size/data')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        step=saved['step'];start_epoch=saved['next_epoch'];skip=saved['next_batch'];best=saved['best'];resume_rng=saved['rng'][rank]
    wrapped=DDP(model,device_ids=[local] if device.type=='cuda' else None,find_unused_parameters=False) if world>1 else model
    tasks=['object'] if cfg['setting']=='object_only' else ['object','image']
    datasets={task:LocalizationData(root,'train',task,cfg.get('image_lmdb_root')) for task in tasks}
    batch_size=int(cfg['batch_size'])
    full=cfg.get('epoch_sampling','random_budget')=='full_coverage'
    if cfg.get('epoch_sampling','random_budget') not in ('random_budget','full_coverage'):raise ValueError('Unknown epoch sampling mode')
    object_positions=sum(len(q['world_xy']) for q in datasets['object'].queries)
    if full:
        if 'image' in datasets:
            steps_per_epoch=math.ceil(len(datasets['image'])/(batch_size*world))
        else:
            # Cover every distinct object positive, then use balanced repeats.
            # Preserve an explicit optimization budget independently of image count.
            steps_per_epoch=max(math.ceil(object_positions/(batch_size*world)),int(cfg.get('steps_per_epoch',0)))
    else:steps_per_epoch=int(cfg['steps_per_epoch'])
    if rank==0:
        (run/'budget.json').write_text(json.dumps(dict(epochs=cfg['epochs'],steps_per_epoch=steps_per_epoch,total_steps=steps_per_epoch*cfg['epochs'],global_batch_per_task=batch_size*world,image_queries=len(datasets['image']) if 'image' in datasets else 0,object_queries=len(datasets['object']),sampling=cfg.get('epoch_sampling','random_budget')),indent=2))
    if min(batch_size,steps_per_epoch,int(cfg['epochs']))<1:raise ValueError('Invalid training budget')
    if resume_rng:restore_rng(resume_rng)
    for epoch in range(start_epoch,int(cfg['epochs'])):
        loaders={}
        for task in tasks:
            sampled=(FullEpochSamples if full else EpochSamples)(datasets[task],steps_per_epoch*batch_size*world,int(cfg['seed']),epoch)
            if full and rank==0:(run/('coverage_%03d_%s.json'%(epoch+1,task))).write_text(json.dumps(sampled.audit,indent=2))
            sampler=DistributedSampler(sampled,num_replicas=world,rank=rank,shuffle=False,drop_last=False)
            generator=torch.Generator().manual_seed(cfg['seed']+epoch)
            loaders[task]=iter(DataLoader(sampled,batch_size=batch_size,sampler=sampler,num_workers=int(cfg.get('workers',0)),collate_fn=collate,generator=generator))
        model.train();next_batch=skip;stop=False;begin=time.monotonic()
        for index in range(steps_per_epoch):
            raw={task:next(loaders[task]) for task in tasks}
            if index<skip:continue
            batches=[featurize(raw[task],features,device) for task in tasks]
            optimizer.zero_grad(set_to_none=True);loss,parts=wrapped(batches[0] if len(batches)==1 else batches)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();step+=1;next_batch=index+1
            metrics=torch.cat([loss.detach()[None],parts])
            if world>1:dist.all_reduce(metrics);metrics/=world
            if rank==0 and (step%int(cfg.get('log_every',20))==0 or a.max_steps):
                record=dict(epoch=epoch+1,batch=index+1,step=step,loss=float(metrics[0]),object_loss=float(metrics[1]),lr=optimizer.param_groups[0]['lr'],elapsed_s=time.monotonic()-begin)
                if len(tasks)>1:record['image_loss']=float(metrics[2])
                with (run/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True)
                (run/'progress.json').write_text(json.dumps(dict(phase='training',epoch=epoch+1,batch=index+1,steps_per_epoch=steps_per_epoch,step=step,total_steps=steps_per_epoch*cfg['epochs'],elapsed_epoch_seconds=time.monotonic()-begin,time=time.time()),indent=2))
            interval=int(cfg.get('save_every_steps',0))
            if interval and step%interval==0 and next_batch<steps_per_epoch:
                states=[None]*world;own=rng_state()
                if world>1:dist.all_gather_object(states,own)
                else:states=[own]
                if rank==0:
                    atomic_save(dict(format=FORMAT,config=cfg,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),world_size=world,dataset_fingerprint=fingerprint,epoch=epoch+1,step=step,next_epoch=epoch,next_batch=next_batch,best=best,rng=states,validation=None),run/'last.pt')
                if world>1:dist.barrier()
            if a.max_steps and step>=a.max_steps:stop=True;break
        complete=next_batch==steps_per_epoch
        if complete:scheduler.step()
        own_rng=rng_state();states=[None]*world
        if world>1:dist.all_gather_object(states,own_rng)
        else:states=[own_rng]
        if rank==0:
            # Validation never changes the training RNG or selects using test data.
            result=None
            if complete:
                (run/'progress.json').write_text(json.dumps(dict(phase='validation',epoch=epoch+1,step=step,total_steps=steps_per_epoch*cfg['epochs'],time=time.time()),indent=2))
                result=evaluate_model(model,features,cfg,'val',a.eval_limit,device,run/('val_epoch_%03d'%(epoch+1)))
            restore_rng(own_rng)
            improved=[]
            if result:
                for kind in ('object','image','joint'):
                    key=kind+'_selection_score'
                    if key in result and result[key]>best.get(kind,-1):best[kind]=result[key];improved.append(kind)
            package=dict(format=FORMAT,config=cfg,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),world_size=world,dataset_fingerprint=fingerprint,epoch=epoch+1,step=step,next_epoch=epoch+int(complete),next_batch=0 if complete else next_batch,best=best,rng=states,validation=result)
            atomic_save(package,run/'last.pt')
            if complete:atomic_save(package,run/('epoch_%03d.pt'%(epoch+1)))
            for kind in improved:atomic_save(package,run/('best_'+kind+'.pt'))
            (run/'status.json').write_text(json.dumps(dict(phase='complete' if complete and epoch+1==cfg['epochs'] else ('stopped_at_max_steps' if stop else 'training'),epoch=epoch+1,step=step,best=best),indent=2))
        if world>1:dist.barrier()
        skip=0
        if stop:break
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
