"""Target-hypothesis routes with fixed IDs; GT is accepted only by label construction."""
import hashlib,json
import math
from pathlib import Path
import numpy as np
from datasets.path_distribution import PathProposer,VisitMemory,resample
from datasets.bridge_maps import world_to_local,sample_prefix
from datasets.route_goal_supervision import route_goal_labels


def route_signature(config):
    keys=('goal_top_k','exploration_candidates','hypothesis_max_snap_m','goal_label_radius_m','route_label_temperature','path_route_points','path_proposal_pool','len_traj_pred','astar_step_m','astar_free_threshold','astar_clearance_m','astar_max_snap_m','astar_exclude_exterior','history_length','traj_pace','validation_buildings_file')
    keys+=('route_supervision','route_policy_version','route_objective','validation_design','data_folder')
    split=config.get('validation_buildings_file')
    split_hash=hashlib.sha256(Path(split).read_bytes()).hexdigest() if split and Path(split).is_file() else None
    return hashlib.sha256(json.dumps(dict(version='hypothesis-routes-v4-source-aware',split_sha256=split_hash,parameters={k:config.get(k) for k in keys}),sort_keys=True).encode()).hexdigest()

def refresh_offsets(size,k):
    """Exactly the noninitial reservoirs reachable by repeated online refresh()."""
    if size<=k:return []
    step=max(1,k//2)
    return [(i*step)%size for i in range(1,size//math.gcd(size,step))]

class HypothesisRoutes:
    def __init__(self,config,hypotheses):
        self.config=config;self.k=int(config['goal_top_k']);self.extra=int(config.get('exploration_candidates',2))
        if not 1<=self.k<=64 or not 1<=self.extra<=8:raise ValueError('Invalid K/exploration count')
        self.reservoir={k:np.asarray(v) for k,v in hypotheses.items()};self.offset=0
        self.set_offset(0)
        self.explorer=PathProposer(dict(config,path_candidates=self.extra,path_candidate_cache=None))
        self.raw_paths={};self.committed_id=None;self.committed_path=None;self.committed_goal=None
    def set_offset(self,offset):
        n=len(self.reservoir['poses']);self.offset=int(offset)%n
        indices=(np.arange(min(n,self.k))+self.offset)%n
        self.hypotheses={k:v[indices] for k,v in self.reservoir.items()}
    def refresh(self):
        if len(self.reservoir['poses'])<=self.k:return False
        self.set_offset(self.offset+max(1,self.k//2));return True
    def commit(self,index,candidates):
        identity=int(candidates['candidate_ids'][index])
        self.committed_id=identity;self.committed_path=self.raw_paths.get(identity)
        self.committed_goal=candidates['candidate_goals'][index].copy()
    def _retain(self,planner,position,path):
        old=self.committed_path
        if old is None or len(old)<2:return path
        nearest=int(np.linalg.norm(old-position,axis=1).argmin())
        if np.linalg.norm(old[nearest]-position)>.5:return path
        join=min(nearest+1,len(old)-1);connector=planner.astar(position,old[join])
        if len(connector)==0 or np.linalg.norm(np.diff(connector,axis=0),axis=1).sum()>1.:return path
        return np.vstack([connector,old[join+1:]])
    def propose(self,planner,position,yaw,memory=None,**unused):
        memory=memory or VisitMemory();n=self.k+self.extra;points=int(self.config.get('path_route_points',32));h=int(self.config['len_traj_pred'])
        v=dict(candidate_routes=np.zeros((n,points,2),np.float32),candidate_grid=np.zeros((n,points,2),np.float32),
          candidate_priors=np.zeros((n,h,2),np.float32),candidate_valid=np.zeros((n,h),np.float32),candidate_goals=np.zeros((n,2),np.float32),
          candidate_stats=np.zeros((n,4),np.float32),candidate_mask=np.zeros(n,bool),candidate_ids=np.full(n,-1,np.int64),
          candidate_is_goal=np.zeros(n,bool),candidate_hypothesis=np.zeros((n,8),np.float32))
        tree=self.explorer._tree(planner,position);self.raw_paths={}
        for i,pose in enumerate(self.hypotheses['poses']):
            v['candidate_ids'][i]=self.hypotheses['ids'][i];v['candidate_is_goal'][i]=True
            cell=planner.snap(planner.to_pixel(pose[:2]))
            if cell is None:continue
            goal=planner.to_world(np.asarray(cell));snap=float(np.linalg.norm(goal-pose[:2]))
            if snap>float(self.config.get('hypothesis_max_snap_m',.5)):continue
            reverse=planner.path_from_tree(goal,tree)
            if len(reverse)==0:continue
            raw=reverse[::-1].copy();identity=int(v['candidate_ids'][i])
            if identity==self.committed_id:raw=self._retain(planner,position,raw)
            if np.linalg.norm(raw[0]-position)>1e-8:raw=np.vstack([position,raw])
            self.raw_paths[identity]=raw
            self._fill(v,i,raw,goal,planner,position,yaw,memory)
            angle=pose[2]-yaw
            v['candidate_hypothesis'][i]=[np.clip(self.hypotheses['scores'][i],-10,10)/5,
               np.clip(np.log(max(float(self.hypotheses['density'][i]),1e-12)),-30,10)/10,
               np.sin(angle) if self.hypotheses.get('yaw_valid',np.ones(len(self.hypotheses['poses']),bool))[i] else 0.,np.cos(angle) if self.hypotheses.get('yaw_valid',np.ones(len(self.hypotheses['poses']),bool))[i] else 0.,snap,float(np.linalg.norm(goal-position))/10,1.,0.]
        exploration=self.explorer.propose(planner,position,yaw,memory)
        for key in ('candidate_routes','candidate_grid','candidate_priors','candidate_valid','candidate_goals','candidate_stats','candidate_mask'):
            v[key][self.k:]=exploration[key]
        # Retain the exact committed exploration endpoint, not merely a nearby
        # replacement with a different pixel ID. Hold/cooldown logic needs a
        # stable candidate identity when the robot moves to another source cell.
        if (self.config.get('route_policy_version')=='category-v2' and self.committed_id is not None
                and self.committed_id>=1000 and self.committed_goal is not None
                and np.linalg.norm(self.committed_goal-position)>float(self.config.get('route_arrival_m',.5))):
            reverse=planner.path_from_tree(self.committed_goal,tree)
            if len(reverse):
                raw=self._retain(planner,position,reverse[::-1].copy())
                if np.linalg.norm(raw[0]-position)>1e-8:raw=np.vstack([position,raw])
                existing=np.flatnonzero(v['candidate_mask'][self.k:] & (np.linalg.norm(v['candidate_goals'][self.k:]-self.committed_goal,axis=-1)<1e-6))
                slot=self.k+int(existing[0]) if len(existing) else self.k
                self._fill(v,slot,raw,self.committed_goal,planner,position,yaw,memory)
                self.raw_paths[self.committed_id]=raw
        seen=set()
        for j in range(self.extra):
            i=self.k+j
            # Geometry-based fallback identity, distinct from diffusion sample IDs.
            pixel=np.rint(planner.to_pixel(v['candidate_goals'][i])).astype(int)
            v['candidate_ids'][i]=1000+int(pixel[1]*planner.size+pixel[0])
            v['candidate_hypothesis'][i,7]=1.
            if self.config.get('route_policy_version')=='category-v2' and v['candidate_mask'][i]:
                identity=int(v['candidate_ids'][i])
                if identity in seen:
                    v['candidate_mask'][i]=False
                    continue
                seen.add(identity)
                if identity not in self.raw_paths:
                    reverse=planner.path_from_tree(v['candidate_goals'][i],tree)
                    if len(reverse):
                        raw=reverse[::-1].copy()
                        if np.linalg.norm(raw[0]-position)>1e-8:raw=np.vstack([position,raw])
                        self.raw_paths[identity]=raw
        return v
    def _fill(self,v,i,path,goal,planner,position,yaw,memory):
        full=resample(path,len(v['candidate_routes'][i]));v['candidate_routes'][i]=world_to_local(full,position,yaw)
        v['candidate_grid'][i]=(planner.to_pixel(full)+.5)/planner.size*2-1
        v['candidate_priors'][i],v['candidate_valid'][i]=sample_prefix(path,position,yaw,self.config['len_traj_pred'],self.config['astar_step_m'])
        v['candidate_goals'][i]=goal;length=np.linalg.norm(np.diff(path,axis=0),axis=1).sum()
        v['candidate_stats'][i]=[length/10]+memory.features(full)
        v['candidate_mask'][i]=True # Zero-length goal is valid; arrival is not a mapping failure.

def hypothesis_labels(candidates,planner,position,yaw,gt_goal,distance_cache,config):
    mask=candidates['candidate_mask'];is_goal=candidates['candidate_is_goal'];cost=np.full(len(mask),np.inf)
    field=distance_cache.distances(planner,gt_goal)
    if field is not None:
        for i in np.flatnonzero(mask&is_goal):
            cell=planner.snap(planner.to_pixel(candidates['candidate_goals'][i]))
            if cell is not None:cost[i]=field[cell[1]*planner.size+cell[0]]
    # Use endpoint geodesic error, not the closest point somewhere along a long route.
    positive=mask&is_goal&(cost<=float(config.get('goal_label_radius_m',1.)))
    target=np.zeros(len(mask),np.float32)
    if positive.any():
        value=np.exp(-cost[positive]/float(config.get('route_label_temperature',.5)));target[positive]=value/value.sum()
    else:
        fallback=dict(candidates,candidate_mask=mask&~is_goal)
        target,_=route_goal_labels(fallback,planner,position,yaw,gt_goal,distance_cache,float(config.get('route_label_temperature',.5)))
    return dict(route_target=target,goal_set_reject=np.array(not positive.any(),np.float32),goal_candidate_positive=positive)
