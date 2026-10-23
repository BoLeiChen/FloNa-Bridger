"""Online route commitment uses predictions and executed history only."""
import numpy as np
from datasets.path_distribution import VisitMemory


class PathPolicyState:
    def __init__(self, switch_margin=.15, min_hold=2, stall_cycles=5):
        self.memory=VisitMemory()
        self.goal=None
        self.margin=switch_margin
        self.min_hold=min_hold
        self.stall_limit=stall_cycles
        self.age=0;self.stall=0;self.previous_position=None
        self.switches=0

    def observe(self, position):
        position=np.asarray(position)
        self.memory.update(position)
        if self.previous_position is not None:
            self.stall=self.stall+1 if np.linalg.norm(position-self.previous_position)<.15 else 0
        self.previous_position=position.copy()

    def select(self, candidates, probability, position, point_goal=False):
        mask=candidates['candidate_mask']
        if not mask.any():return None
        score=np.where(mask,probability,-np.inf)
        best=int(score.argmax())
        if self.goal is not None and not point_goal:
            distances=np.linalg.norm(candidates['candidate_goals']-self.goal,axis=-1)
            old=int(np.where(mask,distances,np.inf).argmin())
            keep=(distances[old]<.6 and np.linalg.norm(position-self.goal)>.5
                  and self.stall<self.stall_limit)
            if keep and (self.age<self.min_hold or score[best]<score[old]+self.margin):best=old
        new_goal=candidates['candidate_goals'][best].copy()
        if self.goal is not None and np.linalg.norm(new_goal-self.goal)>.6:
            self.switches+=1;self.age=0
        self.goal=new_goal;self.age+=1
        return best
