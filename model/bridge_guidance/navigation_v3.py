"""Ablatable local route energy and executed-prefix risk for the RGB4 bridge."""
import torch
from .navigation_v2 import NavigationObjectiveV2
from .geometry import interpolate_segments


class NavigationObjectiveV3(NavigationObjectiveV2):
    def __init__(self, local_map, config, point_goal, prior, valid, execute_points=9):
        self.route_enabled = bool(config.get('route_energy', False))
        self.prefix_enabled = bool(config.get('prefix_risk', False))
        self.prior, self.valid = prior.detach(), valid.detach()
        self.execute_points = int(execute_points)
        if not 2 <= self.execute_points <= prior.shape[1]:
            raise ValueError('Invalid execution prefix')
        # Route-energy variants replace the impossible origin-to-final-goal gate.
        super().__init__(local_map, config, None if self.route_enabled else point_goal)
        self.visible_goal = torch.zeros(len(prior), device=prior.device) if self.route_enabled else self.visible_goal
        self.goal_energy = self.route_enabled or self.goal_energy
        self.route_target = None

    @torch.no_grad()
    def bind_reference(self, path):
        super().bind_reference(path)
        if not self.route_enabled:
            return
        endpoint = path[:, self.execute_points-1]
        start, segment = self.prior[:, :-1], self.prior[:, 1:]-self.prior[:, :-1]
        length = segment.norm(dim=-1)
        usable = (self.valid[:, :-1]>.5)&(self.valid[:, 1:]>.5)&(length>1e-6)
        relative = endpoint[:, None]-start
        fraction = ((relative*segment).sum(-1)/length.square().clamp_min(1e-12)).clamp(0,1)
        nearest = start+fraction[..., None]*segment
        distances = (nearest-endpoint[:, None]).norm(dim=-1).masked_fill(~usable, float('inf'))
        index = distances.argmin(-1)
        rows = torch.arange(len(path),device=path.device)
        arc = torch.cat([torch.zeros_like(length[:, :1]),length.cumsum(-1)],-1)
        current_s = arc[rows,index]+fraction[rows,index]*length[rows,index]
        total = (length*usable).sum(-1)
        target_s = torch.minimum(current_s+float(self.cfg.get('route_lookahead_m',.25)),total)
        target_index = ((arc[:,1:]<target_s[:,None])&usable).sum(-1).clamp(max=length.shape[1]-1)
        t = ((target_s-arc[rows,target_index])/length[rows,target_index].clamp_min(1e-6)).clamp(0,1)
        target = start[rows,target_index]+t[:,None]*segment[rows,target_index]
        # Gate ONLY the added endpoint-to-subgoal correction corridor. The existing
        # path is handled by collision/unknown costs; no blind cell is marked free.
        # This checks a curved-route shortcut against observed depth before using it.
        n = max(2, int(2*(float(self.cfg.get('route_max_cross_track_m',.5))+
                         float(self.cfg.get('route_lookahead_m',.25)))/self.map.resolution)+2)
        u = torch.linspace(0,1,n,device=path.device,dtype=path.dtype)
        points = endpoint[:,None]*(1-u[None,:,None])+target[:,None]*u[None,:,None]
        sdf,outside = self.map.query(self.map.sdf,points)
        known,_ = self.map.query(self.map.free_clearance,points)
        enabled = (usable.any(-1)&(distances[rows,index]<=float(self.cfg.get('route_max_cross_track_m',.5)))
                   &(target_s>current_s+1e-4)&(outside.amax(-1)==0)
                   &(sdf.amin(-1)>=self.radius+self.margin)&(known.amin(-1)>=self.radius))
        self.visible_goal = enabled.to(path)
        self.route_target = target.detach()

    def _geometry(self,path):
        collision,exposure,boundary = super()._geometry(path)
        if self.prefix_enabled and self.execute_points < path.shape[1]:
            # Blend prefix and tail risks; retain both rather than ignoring tail.
            first,_,_ = super()._geometry(path[:,:self.execute_points])
            tail,_,_ = super()._geometry(path[:,self.execute_points-1:])
            collision = .75*first+.25*tail
        return collision,exposure,boundary

    def terms(self,path):
        terms = super().terms(path)
        if self.route_enabled:
            d = ((path[:,self.execute_points-1]-self.route_target)/self.goal_scale).square().sum(-1)
            terms['goal'] = (torch.sqrt(1+d)-1)*self.visible_goal.to(path)
        return terms

    @torch.no_grad()
    def accept_update(self,before,after):
        accept = super().accept_update(before,after)
        # Prefix weighting may not trade away safety anywhere on the full path.
        if self.prefix_enabled:
            old = NavigationObjectiveV2._geometry(self,before)[0]
            new = NavigationObjectiveV2._geometry(self,after)[0]
            accept &= new<=old+1e-7
        return accept
