"""Goal-conditioned route supervision on TRAIN scenes; no expert-heading shortcut label."""
import hashlib,json
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from datasets.multimodal_path_dataset import MultimodalPathLMDB
from datasets.route_goal_supervision import GoalDistanceCache,route_labels_from_fields

class GoalPairLMDB(MultimodalPathLMDB):
    def __init__(self,config,manifest,max_samples,seed=0):
        if [float(config[k]) for k in ('point_goal_train_prob','img_goal_train_prob','object_goal_train_prob')]!=[0.,1.,0.]:
            raise ValueError('V2 currently trains ImageGoal only')
        super().__init__(config,manifest,max_samples,seed)
        self.goal_banks=OrderedDict();self.distances=GoalDistanceCache(32)
        meta=json.loads((Path(config['goal_bank'])/'metadata.json').read_text())
        if meta['scope'].split(',')[0]!='training roots only':raise ValueError('Bank must be from training data')
    def bank(self,floor_key,planner):
        if floor_key not in self.goal_banks:
            path=Path(self.config['goal_bank'])/(hashlib.sha256(floor_key.encode()).hexdigest()+'.npz')
            with np.load(path,allow_pickle=False) as raw:
                value={k:raw[k].copy() for k in raw.files}
            if str(value['floor_key'])!=floor_key:raise ValueError('Bank floor mismatch')
            if 'goal_distance' not in value or str(value['map_fingerprint'])!=planner.metadata['fingerprint']:
                raise ValueError('Missing/stale goal distance bank; regenerate before training')
            self.goal_banks[floor_key]=value
            while len(self.goal_banks)>4:self.goal_banks.popitem(last=False)
        self.goal_banks.move_to_end(floor_key);return self.goal_banks[floor_key]
    def __iter__(self):
        for sample in super().__iter__():
            _,root,scene=sample['floor_key'].split('::');planner=self.maps.planner(root,scene)
            bank=self.bank(sample['floor_key'],planner)
            candidates={k:v.numpy() for k,v in sample.items() if k.startswith('candidate_')}
            goal=sample['label_goal_world'].numpy();field=self.distances.distances(planner,goal)
            if field is None:field=np.full(planner.size**2,np.inf)
            fields=np.concatenate([field[None],bank['goal_distance']],axis=0)
            targets,costs=route_labels_from_fields(candidates,planner,fields,float(self.config.get('route_label_temperature',.5)))
            original=targets[0];alternatives=targets[1:]
            tv=.5*np.abs(alternatives-original).sum(-1)
            separated=np.linalg.norm(bank['goal_world']-goal,axis=-1)>=1.
            eligible=(alternatives.sum(-1)>0)&separated&(tv>=float(self.config.get('goal_pair_min_tv',.25)))&(original.sum()>0)
            options=np.flatnonzero(eligible)
            if len(options):
                # Vary hard valid goals reproducibly across epochs, without test feedback.
                digest=hashlib.sha256(('%s:%d:%d'%(sample['sample_id'],self.epoch,self.seed)).encode()).digest()
                index=int(options[int.from_bytes(digest[:8],'big')%len(options)])
                alt_image=torch.from_numpy(bank['goal_rgb'][index].copy()).permute(2,0,1).float()/255
                alt_target=alternatives[index];pair_valid=True;pair_tv=float(tv[index])
            else:
                alt_image=sample['goal_rgb'];alt_target=original;pair_valid=False;pair_tv=0.
            sample['expert_route_target']=sample['route_target']
            sample.update(route_target=torch.from_numpy(original),alternate_goal_rgb=alt_image,
                          alternate_route_target=torch.from_numpy(alt_target),goal_pair_valid=torch.tensor(pair_valid),
                          goal_pair_tv=torch.tensor(pair_tv,dtype=torch.float32))
            yield sample
