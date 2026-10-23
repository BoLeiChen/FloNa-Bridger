"""Explicit mechanism-only oracle/shuffled-goal/controller ablations."""
import torch
from datasets.route_goal_supervision import GoalDistanceCache,route_goal_labels

class DiagnosticPolicy:
    def __init__(self,policy,controller=None,oracle=False):
        self.policy=policy;self.controller=controller;self.oracle=oracle
        self.map_planner=None;self.distances=GoalDistanceCache()
    def score_paths(self,batch):
        prepared=self.policy.score_paths(batch)
        if not self.oracle:return prepared
        if 'diagnostic_gt_goal' not in batch:raise ValueError('Oracle requires explicit diagnostic target')
        candidates={k:v[0].detach().cpu().numpy() for k,v in batch.items() if k.startswith('candidate_')}
        pose=batch['agent_pose'][0].cpu().numpy();goal=batch['diagnostic_gt_goal'][0].cpu().numpy()
        target,_=route_goal_labels(candidates,self.map_planner,pose[:2],pose[2],goal,self.distances)
        if target.sum()==0:return prepared
        probability=torch.as_tensor(target,device=batch['agent_pose'].device)[None]
        return (probability,probability.clamp_min(1e-8).log(),prepared[2],prepared[3])
    def __call__(self,batch,mode,selected_index,prepared):
        if self.controller is None:return self.policy(batch,mode=mode,selected_index=selected_index,prepared=prepared)
        rows=torch.arange(len(selected_index),device=selected_index.device)
        chosen=dict(batch,prior=batch['candidate_priors'][rows,selected_index],
            prior_valid=batch['candidate_valid'][rows,selected_index],goal_world=batch['candidate_goals'][rows,selected_index])
        return self.controller(chosen,mode='sample')
