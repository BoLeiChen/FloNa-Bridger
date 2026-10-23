"""VE-bridge energy guidance with an optional per-rollout-sample correction budget."""
import math
import torch
from model.bridge_diffusion import decode_trajectory
from .bridge_energy import BridgeEnergyGuidance
from .navigation_v3 import NavigationObjectiveV3


class BridgeEnergyGuidanceV3(BridgeEnergyGuidance):
    def __init__(self,local_map,goal,config,action_step_m,prior,valid,execute_points=9):
        super().__init__(local_map,goal,config,action_step_m)
        self.objective=NavigationObjectiveV3(local_map,config['costs'],goal,prior,valid,execute_points)
        self.robust = bool(config.get('correction_budget',False))
        self.budget = float(config.get('total_update_budget_m',.05))
        self.spent = 0.
        if not math.isfinite(self.budget) or self.budget<=0:
            raise ValueError('Invalid correction budget')

    def __call__(self,clean,sigma,stage):
        if self.scale==0 or float(sigma)>self.sigma_max:
            return clean
        cap=self.max_update_m
        if self.robust:
            remaining=max(0.,self.budget-self.spent)
            if remaining<1e-7:
                return clean
            with torch.no_grad():
                risk=self.objective._geometry(decode_trajectory(clean,self.action_step_m))[0]
            # Smoothly smaller corrections for low modeled risk; local goal
            # attraction remains possible at a quarter of the nominal cap.
            factor=.25+.75*float(risk.clamp(0,1).max())
            self.max_update_m=min(cap*factor,remaining)
        try:
            output=super().__call__(clean,sigma,stage)
            with torch.no_grad():
                shift=(decode_trajectory(output,self.action_step_m)-decode_trajectory(clean,self.action_step_m)).norm(dim=-1).amax()
                self.spent+=float(shift)
            if self.robust and self.spent>self.budget+1e-6:
                raise AssertionError('Correction budget violated')
            self.records[-1].update(route_active=self.objective.visible_goal.cpu().tolist(),
                cumulative_injected_shift_m=self.spent,step_cap_m=self.max_update_m)
            return output
        finally:
            self.max_update_m=cap
