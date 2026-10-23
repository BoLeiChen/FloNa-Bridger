"""Training-only ImageGoal hypotheses: cached image localization, GT labels isolated."""
import hashlib,random
from pathlib import Path
from datasets.sampling_identity import sampling_root
from collections import OrderedDict
import lmdb
import numpy as np
import torch
from datasets.lmdb_dataset import decode_record
from datasets.pointgoal_bridge_dataset import mark_floorplan
from datasets.multimodal_path_dataset import MultimodalPathLMDB,task_for_window
from datasets.path_distribution import VisitMemory,training_labels
from datasets.hypothesis_routes import HypothesisRoutes,hypothesis_labels,refresh_offsets,dynamic_candidate_mask
from datasets.route_goal_supervision import GoalDistanceCache
from localization.hypothesis_factory import provider_for,trajectory_hypotheses,object_task
from datasets.object_category_targets import load_category_targets
from datasets.category_route_supervision import CategoryDistanceCache, category_route_labels

class HypothesisRouteLMDB(MultimodalPathLMDB):
    def __init__(self,config,manifest,max_samples,seed=0):
        expected=[0.,0.,1.] if object_task(config) else [0.,1.,0.]
        if [float(config[k]) for k in ('point_goal_train_prob','img_goal_train_prob','object_goal_train_prob')]!=expected:raise ValueError('Task probabilities do not match route task')
        parent=dict(config,point_goal_train_prob=0.,img_goal_train_prob=1.,object_goal_train_prob=0.)
        super().__init__(parent,manifest,max_samples,seed)
        self.config=config
        self.provider=provider_for(config,allow_inference=False)
        self.label_distances=GoalDistanceCache(32)
        self.category_distances=CategoryDistanceCache(16)
        self.category_labels=config.get('route_supervision')=='object-category-center-region-v2'
        if self.category_labels and not object_task(config):raise ValueError('Category supervision is ObjectGoal only')

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
                    label_tree=None if self.category_labels else planner.goal_tree(poses[-1,:2])
                    targets=(load_category_targets(self.config['scene_assets'],scene,Path(root)/scene/record['traj_name'])
                             if self.category_labels else None)
                    goal_path=Path(root)/scene/record['traj_name']/'rgb'/('rgb_%d.png'%(len(poses)-1))
                    hypotheses=trajectory_hypotheses(self.provider,self.config,root,scene,record['traj_name'],len(poses))
                    goal_text=hypotheses.pop('goal_text',None)
                    proposer=HypothesisRoutes(self.config,hypotheses)
                    memory=VisitMemory();last=-1
                    wanted=self.config.get('validation_window_indices')
                    wanted=None if wanted is None else set(map(int,wanted))
                    for window_index,start in enumerate(starts):
                        if yielded>=self.max_samples:break
                        if wanted is not None and window_index not in wanted:continue
                        start=int(start)
                        for past in range(last+1,start+1):memory.update(poses[past,:2])
                        last=start
                        sample_id='%s::%s::%d'%(scene,record['traj_name'],start)
                        task=2 if object_task(self.config) else task_for_window(sample_id,self.epoch,self.config)
                        position=poses[start,:2];yaw=np.arctan2(poses[start,3]-poses[start,1],poses[start,2]-poses[start,0])
                        # Expose the reject head to reservoir replenishment during training too.
                        digest=int.from_bytes(hashlib.sha256((sample_id+':'+str(self.epoch)).encode()).digest()[:4],'big')
                        n=len(hypotheses['poses']);k=int(self.config['goal_top_k'])
                        offset=(k+(digest//100)%max(1,n-k)) if n>k and digest%100<int(100*float(self.config.get('hypothesis_tail_train_fraction',.25))) else 0
                        if self.category_labels and offset:
                            offsets=refresh_offsets(n,k)
                            offset=offsets[(digest//100)%len(offsets)] if offsets else 0
                        proposer.set_offset(offset)
                        candidates=proposer.propose(planner,position,yaw,memory)
                        effective_k=int(self.config['goal_top_k']);effective_exploration=int(self.config.get('exploration_candidates',2));dynamic_offset=0
                        if self.config.get('dynamic_candidate_training') and wanted is None:
                            effective_k,effective_exploration,dynamic_offset=dynamic_candidate_mask(candidates,self.config,digest)
                        expert=self._future_pose_tensor(poses,start)
                        if self.category_labels:
                            labels=category_route_labels(candidates,planner,position,targets,self.category_distances,self.config)
                        else:
                            labels=training_labels(candidates,expert.numpy(),planner,poses[-1,:2],label_tree,
                                temperature=float(self.config.get('route_label_temperature',.5)),
                                max_prefix_error=float(self.config.get('bridge_pair_max_error_m',.5)))
                            labels.update(hypothesis_labels(candidates,planner,position,yaw,poses[-1,:2],self.label_distances,self.config))
                        sample=dict(rgb=rgb[[max(0,i) for i in range(start-3,start)]],current_rgb=rgb[start],
                            floorplan=floors[item['floor_key']],agent_pose=torch.tensor([*position,yaw],dtype=torch.float32),
                            task_type=torch.tensor(task,dtype=torch.long),
                            task_point_world=torch.tensor(poses[-1,:2].copy() if task==0 else np.zeros(2),dtype=torch.float32),
                            goal_rgb=rgb[-1] if task==1 else torch.zeros_like(rgb[-1]),
                            gt_pose=expert,sample_id=sample_id,
                            floor_key=item['floor_key'],label_goal_world=torch.tensor(poses[-1,:2].copy(),dtype=torch.float32))
                        if goal_text is not None:sample['goal_text']=torch.from_numpy(goal_text.copy())
                        if targets is not None:
                            sample.update(object_category=targets['category'],label_version=self.config['route_supervision'],
                                          label_asset_sha256=targets['source_sha256'],building=scene.rsplit('_',1)[0],
                                          hypothesis_offset=offset,dynamic_goal_top_k=effective_k,
                                          dynamic_exploration_candidates=effective_exploration,
                                          dynamic_hypothesis_offset=dynamic_offset)
                        sample.update({k:torch.from_numpy(v) for k,v in {**candidates,**labels}.items()})
                        yielded+=1
                        yield sample
        finally:env.close()
        if yielded!=self.max_samples:raise RuntimeError('LMDB rank exhausted early')
