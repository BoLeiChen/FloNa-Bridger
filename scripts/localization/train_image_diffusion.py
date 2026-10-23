import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse, json, os, sys, time
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch, yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, Subset, DistributedSampler
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from third_party.cf2loc_hm3d.DisCo_model.floverse_full_dataset import FloVerseFullDataset, LocalityDistributedSampler
from third_party.cf2loc_hm3d.DisCo_model.pose_query_diffusion import PoseQueryDiffusionLocalizer

class DistributedDiffusion(PoseQueryDiffusionLocalizer):
    def log(self, name, value, *args, **kwargs):
        if name.startswith('val_'): kwargs.setdefault('sync_dist',True)
        return super().log(name,value,*args,**kwargs)

class FullData(pl.LightningDataModule):
    def __init__(self, config, smoke=False):
        super().__init__(); self.config=config; self.smoke=smoke
    def setup(self, stage=None):
        c=self.config
        self.train_data=FloVerseFullDataset(c['full_index'],'train',augment=True,lmdb_root=c.get('lmdb_root'))
        full_val=FloVerseFullDataset(c['full_index'],'val',lmdb_root=c.get('lmdb_root'))
        multiple=self.trainer.world_size*c['val_batch_size']
        n=min(len(full_val),int(c['validation_frames']))
        n=(n//multiple)*multiple
        if n==0: raise ValueError('Validation has fewer than one global batch')
        rng=np.random.default_rng(42)
        self.val_indices=np.sort(rng.choice(len(full_val),n,replace=False))
        self.val_data=Subset(full_val,self.val_indices.tolist())
        if self.trainer.is_global_zero:
            np.save(Path(c['run_dir'])/'validation_indices.npy',self.val_indices)
            print('DATA_READY',json.dumps({'train_frames':len(self.train_data),'validation_index_frames':len(full_val),'validation_per_epoch':n,'buildings':{k:len(v['buildings']) for k,v in self.train_data.meta['splits'].items()},'gibson_frames':0}),flush=True)
    def train_dataloader(self):
        c=self.config
        sampler=LocalityDistributedSampler(len(self.train_data),self.trainer.global_rank,self.trainer.world_size,c['batch_size'],c['io_block_size'])
        return DataLoader(self.train_data,batch_size=c['batch_size'],sampler=sampler,num_workers=c['num_workers'],pin_memory=True,persistent_workers=bool(c['num_workers']),prefetch_factor=2 if c['num_workers'] else None)
    def val_dataloader(self):
        c=self.config
        sampler=DistributedSampler(self.val_data,num_replicas=self.trainer.world_size,rank=self.trainer.global_rank,shuffle=False,drop_last=False)
        return DataLoader(self.val_data,batch_size=c['val_batch_size'],sampler=sampler,num_workers=c['num_workers'],pin_memory=True,persistent_workers=bool(c['num_workers']),prefetch_factor=2 if c['num_workers'] else None)

class Progress(pl.Callback):
    def on_train_start(self,trainer,module): self.started=time.monotonic(); self.start_step=trainer.global_step
    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        if trainer.is_global_zero and (trainer.global_step%100==0 or trainer.global_step<=3):
            elapsed=time.monotonic()-self.started
            steps=trainer.global_step-self.start_step
            fps=steps*module.config['batch_size']*trainer.world_size/max(elapsed,1e-6)
            loss=float(outputs['loss'] if isinstance(outputs,dict) else outputs)
            print('TRAIN_PROGRESS',json.dumps({'epoch':trainer.current_epoch+1,'step':trainer.global_step,'loss':loss,'global_frames_s':round(fps,2),'elapsed_s':round(elapsed),'gpu_memory_MiB':round(torch.cuda.max_memory_allocated()/1024**2)}),flush=True)
    def on_validation_epoch_end(self,trainer,module):
        if trainer.is_global_zero and not trainer.sanity_checking:
            keys=['val_loss','val_1m_recall','val_mean_xy_err_m','val_mean_theta_err_deg']
            print('VALIDATION',json.dumps({'epoch':trainer.current_epoch+1,'step':trainer.global_step,**{k:float(trainer.callback_metrics[k]) for k in keys if k in trainer.callback_metrics}}),flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--resume'); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    c=yaml.safe_load(Path(a.config).read_text())
    pl.seed_everything(c['seed'],workers=True)
    torch.set_num_threads(4)
    run=Path(c['run_dir']); run.mkdir(parents=True,exist_ok=True)
    model=DistributedDiffusion(c)
    ckpt=ModelCheckpoint(dirpath=run/'checkpoints',filename='{epoch:02d}-{val_1m_recall:.4f}',monitor='val_1m_recall',mode='max',save_top_k=3,save_last=True)
    recovery=ModelCheckpoint(dirpath=run/'recovery',filename='step-{step}',monitor=None,save_top_k=1,save_last=True,train_time_interval=timedelta(minutes=30),save_on_train_epoch_end=False)
    trainer=pl.Trainer(accelerator='gpu',devices=c['gpu_ids'],strategy=DDPStrategy(find_unused_parameters=False),
        precision=c['precision'],max_epochs=1 if a.smoke else c['epochs'],
        callbacks=[ckpt,recovery,Progress()],logger=CSVLogger(run,name='metrics'),
        default_root_dir=run,enable_progress_bar=False,gradient_clip_val=1.,
        log_every_n_steps=10,num_sanity_val_steps=1,use_distributed_sampler=False,
        limit_train_batches=10 if a.smoke else 1.0,limit_val_batches=2 if a.smoke else 1.0)
    trainer.fit(model,datamodule=FullData(c,a.smoke),ckpt_path=a.resume)
    if trainer.is_global_zero: (run/'completed.json').write_text(json.dumps({'epochs':trainer.current_epoch,'step':trainer.global_step,'best_checkpoint':ckpt.best_model_path}))

if __name__=='__main__': main()
