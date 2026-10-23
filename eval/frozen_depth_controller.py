"""Frozen PointGoal substitution with the evaluated current-view V3 guidance."""
import random
import numpy as np
import torch
from datasets.bridge_maps import world_to_local

def load_controller(checkpoint, policy_config, device):
    from eval.pointgoal_bridge import load_models
    python_state=random.getstate(); numpy_state=np.random.get_state()
    try:
        with torch.random.fork_rng():
            config,model=load_models(checkpoint,device)
    finally:
        random.setstate(python_state);np.random.set_state(numpy_state)
    for key in ('history_length','len_traj_pred','global_coordinate_mode','global_coordinate_step_m'):
        if config[key]!=policy_config[key]:raise ValueError('Controller mismatch: '+key)
    if config['bridge']['action_step_m']!=policy_config['bridge']['action_step_m']:
        raise ValueError('Controller action scale mismatch')
    model.requires_grad_(False)
    return config,model

def sample_selected(controller,controller_config,batch,index,guidance_config,env,scene_assets,scene_name,floor_level,execute_steps):
    if batch['agent_pose'].shape[0]!=1:raise ValueError('Online controller expects batch size one')
    if not bool(batch['candidate_mask'][0,index]):raise ValueError('Selected padded candidate')
    chosen=dict(batch,prior=batch['candidate_priors'][:,index],
                prior_valid=batch['candidate_valid'][:,index],
                goal_world=batch['candidate_goals'][:,index])
    energy=None;sensing=None
    if guidance_config is not None:
        from eval.depth_sensor import observed_map
        from model.bridge_guidance.bridge_energy_v3 import BridgeEnergyGuidanceV3
        if guidance_config.get('version')!='v3' or guidance_config.get('memory_frames',0)!=0:
            raise ValueError('This integration reproduces current-view V3 only')
        pose=batch['agent_pose'][0].detach().cpu().numpy()
        goal=chosen['goal_world'][0].detach().cpu().numpy()
        point=torch.as_tensor(world_to_local(goal,pose[:2],pose[2]),device=chosen['prior'].device,dtype=chosen['prior'].dtype)[None,None]
        local_map,sensing=observed_map(env,scene_assets,scene_name,floor_level,guidance_config,chosen['prior'].device)
        energy=BridgeEnergyGuidanceV3(local_map,point,guidance_config,
            float(controller_config['bridge']['action_step_m']),chosen['prior'],chosen['prior_valid'],execute_steps)
    path=controller(chosen,mode='sample',energy_guidance=energy)
    if not torch.isfinite(path).all():raise ValueError('Nonfinite controller trajectory')
    report=None if energy is None else dict(sensing=sensing,
        visible_goal=energy.objective.visible_goal.detach().cpu().tolist(),
        updates=energy.records,selected_goal_world=chosen['goal_world'].detach().cpu().tolist())
    return path,report
