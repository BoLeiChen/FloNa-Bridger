"""Persistent hypothesis IDs, finite cooldowns and windowed progress. No GT input."""
from collections import deque
import numpy as np
from datasets.path_distribution import VisitMemory

class HypothesisPolicyState:
    def __init__(self,config):
        self.config=config;self.memory=VisitMemory();self.identity=None;self.goal=None;self.age=0;self.cycle=0;self.switches=0
        self.positions=deque(maxlen=int(config.get('route_progress_window',6)));self.cooldown={};self.pending=None;self.pending_count=0
        self.last_reason='initial';self.last_rejected=False;self.exploration_cycles=0
        self.reject_ema=None;self.reject_pending=0
        self.reject_set_ids=None
    def _update_rejection(self,probability):
        if self.config.get('route_policy_version')!='category-v2':
            self.last_rejected=float(probability)>=float(self.config.get('goal_reject_threshold',.5));return
        enter=float(self.config.get('goal_reject_enter',.65));leave=float(self.config.get('goal_reject_leave',.35))
        alpha=float(self.config.get('goal_reject_ema_alpha',.5));confirm=int(self.config.get('goal_reject_confirm_cycles',2))
        if not 0<=leave<enter<=1 or not 0<alpha<=1 or confirm<1:raise ValueError('Invalid rejection hysteresis')
        self.reject_ema=float(probability) if self.reject_ema is None else alpha*float(probability)+(1-alpha)*self.reject_ema
        change=(self.reject_ema<=leave) if self.last_rejected else (self.reject_ema>=enter)
        self.reject_pending=self.reject_pending+1 if change else 0
        if self.reject_pending>=confirm:
            self.last_rejected=not self.last_rejected;self.reject_pending=0
    def observe(self,position):
        self.positions.append(np.asarray(position).copy());self.memory.update(position);self.cycle+=1
    def select(self,candidates,probability,position,point_goal=False,reject_probability=0.):
        mask=candidates['candidate_mask'].copy();ids=candidates['candidate_ids'];goal_mask=candidates['candidate_is_goal']
        if not mask.any():return None
        if self.config.get('route_policy_version')=='category-v2':
            signature=tuple(sorted(int(i) for i in ids[mask&goal_mask]))
            if signature!=self.reject_set_ids:
                self.reject_ema=None;self.reject_pending=0;self.last_rejected=False
                self.reject_set_ids=signature
        self._update_rejection(reject_probability)
        # Decide goal-following versus exploration first. Comparing individual
        # mixture probabilities across unequal group sizes would bias toward
        # the two exploration slots as K grows.
        available_goal=mask&goal_mask&np.array([self.cooldown.get(int(i),0)<=self.cycle for i in ids])
        if (self.last_rejected or not available_goal.any()) and (mask&~goal_mask).any():mask&=~goal_mask
        elif available_goal.any():mask=available_goal
        if self.config.get('route_policy_version')=='category-v2' and (mask&~goal_mask).any():
            fresh=mask&np.array([self.cooldown.get(int(i),0)<=self.cycle for i in ids])
            if fresh.any():mask=fresh
        score=np.log(np.maximum(probability,1e-12))
        for i,identity in enumerate(ids):
            if self.cooldown.get(int(identity),0)>self.cycle:score[i]-=float(self.config.get('route_cooldown_penalty',4.))
        old=np.flatnonzero(mask&(ids==self.identity)) if self.identity is not None else []
        old=int(old[0]) if len(old) else None
        reason='initial' if self.identity is None else 'unavailable'
        if self.config.get('route_policy_version')=='category-v2' and old is None and self.identity in ids:
            reason='group_changed_or_cooldown'
        release=old is None
        if old is not None:
            reached=np.linalg.norm(position-candidates['candidate_goals'][old])<float(self.config.get('route_arrival_m',.5))
            stalled=False
            if len(self.positions)==self.positions.maxlen:
                distances=np.array([np.linalg.norm(p-candidates['candidate_goals'][old]) for p in self.positions])
                net=np.linalg.norm(self.positions[-1]-self.positions[0]);walk=np.linalg.norm(np.diff(np.array(self.positions),axis=0),axis=1).sum()
                positions=np.array(self.positions);diameter=np.linalg.norm(positions[:,None]-positions[None,:],axis=-1).max()
                stalled=net<.2 or (walk>1. and diameter<.5) or (walk>1. and net/walk<.15 and distances[0]-distances[-1]<.1)
            release=reached or stalled or (self.last_rejected and goal_mask[old])
            reason='arrived_unconfirmed' if reached else 'no_progress' if stalled else 'goal_set_rejected' if release else 'hold'
            if reached or stalled:
                self.cooldown[int(ids[old])]=self.cycle+int(self.config.get('route_cooldown_cycles',20));score[old]-=float(self.config.get('route_cooldown_penalty',4.))
        # A failed/reached hypothesis enters a finite cooldown; if none remain,
        # explore instead of repeatedly selecting already attempted coordinates.
        available_goal=candidates['candidate_mask']&goal_mask&np.array([self.cooldown.get(int(i),0)<=self.cycle for i in ids])
        if (mask&goal_mask).any():
            if available_goal.any():mask=available_goal
            elif (candidates['candidate_mask']&~goal_mask).any():
                mask=candidates['candidate_mask']&~goal_mask;release=True
        if self.config.get('route_policy_version')=='category-v2' and (mask&~goal_mask).any():
            fresh=mask&np.array([self.cooldown.get(int(i),0)<=self.cycle for i in ids])
            if fresh.any():mask=fresh
        best=int(np.where(mask,score,-np.inf).argmax())
        if old is not None and not release:
            better=best!=old and score[best]-score[old]>float(self.config.get('route_switch_log_ratio',1.))
            identity=int(ids[best])
            if better:
                self.pending_count=self.pending_count+1 if self.pending==identity else 1;self.pending=identity
            else:self.pending=None;self.pending_count=0
            if self.age<int(self.config.get('route_min_hold_cycles',5)) or self.pending_count<int(self.config.get('route_switch_confirm_cycles',3)):best=old
            else:reason='persistent_new_evidence'
        new_id=int(ids[best]);changed=new_id!=self.identity
        if self.identity is not None and changed:self.switches+=1
        if changed:self.age=0;self.positions.clear();self.pending=None;self.pending_count=0
        self.identity=new_id;self.goal=candidates['candidate_goals'][best].copy();self.age+=1;self.last_reason=reason
        self.exploration_cycles=self.exploration_cycles+1 if not goal_mask[best] else 0
        return best
    def refresh_due(self):
        return self.exploration_cycles>0 and self.exploration_cycles%int(self.config.get('hypothesis_refresh_cycles',10))==0
