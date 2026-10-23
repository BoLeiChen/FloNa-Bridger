"""HM3D training-only LMDB view. ImageGoal target coordinates are supervision only."""
import hashlib
import random
from pathlib import Path
from collections import OrderedDict
import lmdb
import numpy as np
import torch
from datasets.lmdb_dataset import LmdbNavDataset, decode_record
from datasets.pointgoal_bridge_dataset import PriorStore, mark_floorplan
from datasets.path_distribution import PathProposer, VisitMemory, training_labels


def task_for_window(sample_id, epoch, config):
    probability=float(config['img_goal_train_prob'])
    digest=hashlib.sha256(('%s:%s:%s'%(config.get('seed',0),epoch,sample_id)).encode()).digest()
    return int(int.from_bytes(digest[:8],'big')/2**64 < probability)


class MultimodalPathLMDB(LmdbNavDataset):
    def __init__(self, config, manifest, max_samples, seed=0):
        if float(config.get('object_goal_train_prob',0)) != 0:
            raise ValueError('Stage one supports PointGoal and ImageGoal only')
        if abs(float(config['point_goal_train_prob'])+float(config['img_goal_train_prob'])-1)>1e-6:
            raise ValueError('PointGoal and ImageGoal probabilities must sum to one')
        allowed={Path(p).resolve() for p in config.get('data_folder',[])}
        allowed_scenes={p.resolve() for root in config.get('data_folder',[])
                        if Path(root).is_dir() for p in Path(root).iterdir() if p.is_dir()}
        for floor_key in {item['floor_key'] for item in manifest}:
            _,source_root,scene=floor_key.split('::')
            root=Path(source_root)
            parts=root.resolve().parts
            is_train=any(p=='train' or p.startswith('train_') for p in parts)
            is_heldout=any(p in ('test','val','validation') or p.startswith(('test_','val_','validation_')) for p in parts)
            if not is_train or is_heldout:
                raise ValueError('Training loader refuses non-training roots: '+str(root))
            if allowed and root.resolve() not in allowed and (root/scene).resolve() not in allowed_scenes:
                raise ValueError('Manifest root is outside the configured training subset')
        super().__init__(config['lmdb_path'],config['len_traj_pred'],config['traj_pace'],
                         config['history_length'],False,max_samples=max_samples,seed=seed,manifest=manifest)
        self.config=config
        self.maps=PriorStore(config)
        self.proposer=PathProposer(config)

    def __iter__(self):
        worker=torch.utils.data.get_worker_info()
        if worker is not None and worker.num_workers!=1:
            raise ValueError('Use one LMDB loader worker per rank')
        manifest=list(self.manifest);random.Random(self.seed+self.epoch).shuffle(manifest)
        env=lmdb.open(self.lmdb_path,readonly=True,lock=False,readahead=True,max_readers=256)
        yielded=0;floors=OrderedDict()
        try:
            with env.begin(buffers=True) as txn:
                for item in manifest:
                    if yielded>=self.max_samples:break
                    _,root,scene=item['floor_key'].split('::')
                    if item['floor_key'] not in floors:
                        raw=decode_record(bytes(txn.get(item['floor_key'].encode())))['floorplan'].copy()
                        floors[item['floor_key']]=mark_floorplan(raw,None,None,None,None)
                        if len(floors)>8:floors.popitem(last=False)
                    record=decode_record(bytes(txn.get(item['key'].encode())))
                    poses=record['poses'].astype(np.float64)
                    rgb=torch.from_numpy(record['rgb'].copy()).permute(0,3,1,2).float()/255
                    starts=np.arange(0,len(poses)-self.len_traj_pred+1,self.traj_pace)
                    cached=np.arange(0,len(poses)-int(self.config['lmdb_source_len_traj_pred'])+1,int(self.config['lmdb_source_traj_pace']))
                    if not np.array_equal(cached,record['current_indices']) or not np.isin(starts,cached).all() or len(starts)!=item['sample_count']:
                        raise ValueError('LMDB manifest/stride does not match sample geometry')
                    planner=self.maps.planner(root,scene)
                    label_tree=planner.goal_tree(poses[-1,:2])
                    memory=VisitMemory();last=-1
                    for start in starts:
                        if yielded>=self.max_samples:break
                        start=int(start)
                        for past in range(last+1,start+1):memory.update(poses[past,:2])
                        last=start
                        sample_id='%s::%s::%d'%(scene,record['traj_name'],start)
                        task=task_for_window(sample_id,self.epoch,self.config)
                        position=poses[start,:2];yaw=np.arctan2(poses[start,3]-poses[start,1],poses[start,2]-poses[start,0])
                        candidates=self.proposer.propose(planner,position,yaw,memory,
                                                        point_goal=poses[-1,:2] if task==0 else None,
                                                        point_tree=label_tree if task==0 else None)
                        expert=self._future_pose_tensor(poses,start)
                        labels=training_labels(candidates,expert.numpy(),planner,poses[-1,:2],label_tree,
                            temperature=float(self.config.get('route_label_temperature',.5)),
                            max_prefix_error=float(self.config.get('bridge_pair_max_error_m',.5)))
                        sample=dict(rgb=rgb[[max(0,i) for i in range(start-3,start)]],current_rgb=rgb[start],
                            floorplan=floors[item['floor_key']],agent_pose=torch.tensor([*position,yaw],dtype=torch.float32),
                            task_type=torch.tensor(task,dtype=torch.long),
                            task_point_world=torch.tensor(poses[-1,:2].copy() if task==0 else np.zeros(2),dtype=torch.float32),
                            goal_rgb=rgb[-1] if task==1 else torch.zeros_like(rgb[-1]),
                            gt_pose=expert,sample_id=sample_id,
                            floor_key=item['floor_key'],label_goal_world=torch.tensor(poses[-1,:2].copy(),dtype=torch.float32))
                        sample.update({k:torch.from_numpy(v) for k,v in {**candidates,**labels}.items()})
                        yielded+=1
                        yield sample
        finally:env.close()
        if yielded!=self.max_samples:raise RuntimeError('LMDB rank exhausted early')
