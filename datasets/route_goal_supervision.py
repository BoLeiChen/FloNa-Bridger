"""Training/explicit-oracle geometry labels. Never called by ordinary online policy."""
from collections import OrderedDict
import numpy as np
from scipy.sparse.csgraph import dijkstra
from datasets.bridge_maps import local_to_world

class GoalDistanceCache:
    def __init__(self,maxsize=32):
        self.maxsize=maxsize;self.cache=OrderedDict()
    def distances(self,planner,goal):
        cell=planner.snap(planner.to_pixel(goal))
        if cell is None:return None
        key=(planner.metadata.get('fingerprint',id(planner)),cell)
        if key not in self.cache:
            if planner._graph is None:planner.goal_tree(goal)
            self.cache[key]=dijkstra(planner._graph,directed=True,indices=cell[1]*planner.size+cell[0])
            while len(self.cache)>self.maxsize:self.cache.popitem(last=False)
        self.cache.move_to_end(key);return self.cache[key]

def route_goal_labels(candidates,planner,position,yaw,goal,cache,temperature=.5):
    """Prefer routes that approach the target without large travel cost.

    Cost = min_along_route(remaining geodesic + 0.1*travelled arc length).
    No expert future or current heading preference. Multiple equivalent routes
    receive soft probability, including routes sharing a prefix toward the goal.
    """
    if temperature<=0:raise ValueError('temperature must be positive')
    field=cache.distances(planner,goal);mask=candidates['candidate_mask'].astype(bool)
    cost=np.full(len(mask),np.inf);target=np.zeros(len(mask),np.float32)
    if field is not None:
        for i in np.flatnonzero(mask):
            world=local_to_world(candidates['candidate_routes'][i],position,yaw)
            arc=np.r_[0.,np.linalg.norm(np.diff(world,axis=0),axis=-1).cumsum()]
            cells=[planner.snap(planner.to_pixel(x)) for x in world]
            remaining=np.array([field[c[1]*planner.size+c[0]] if c is not None else np.inf for c in cells])
            cost[i]=np.min(remaining+.1*arc)
        valid=mask&np.isfinite(cost)
        if valid.any():
            target[valid]=np.exp(-(cost[valid]-cost[valid].min())/temperature);target/=target.sum()
    return target,cost


def route_labels_from_fields(candidates,planner,fields,temperature=.5):
    """Vectorized equivalent of route_goal_labels for several goal distance fields."""
    if temperature<=0:raise ValueError('temperature must be positive')
    fields=np.asarray(fields);k,l=candidates['candidate_grid'].shape[:2]
    pixels=(candidates['candidate_grid'].astype(np.float64)+1)*.5*planner.size-.5
    rounded=np.rint(pixels).astype(int);inside=((rounded>=0)&(rounded<planner.size)).all(-1)
    safe=rounded.clip(0,planner.size-1)
    good=inside&planner.walkable[safe[...,1],safe[...,0]]
    for i,j in np.argwhere(~good):
        snapped=planner.snap(pixels[i,j])
        if snapped is not None:safe[i,j]=snapped;good[i,j]=True
    indices=safe[...,1]*planner.size+safe[...,0]
    remaining=fields[:,indices].copy();remaining[:,~good]=np.inf
    arc=np.concatenate([np.zeros((k,1)),np.linalg.norm(np.diff(candidates['candidate_routes'],axis=1),axis=-1).cumsum(-1)],-1)
    costs=np.min(remaining+.1*arc[None],axis=-1)
    costs[:,~candidates['candidate_mask'].astype(bool)]=np.inf
    targets=np.zeros_like(costs,dtype=np.float32)
    for g in range(len(fields)):
        valid=np.isfinite(costs[g])
        if valid.any():
            targets[g,valid]=np.exp(-(costs[g,valid]-costs[g,valid].min())/temperature)
            targets[g]/=targets[g].sum()
    return targets,costs
