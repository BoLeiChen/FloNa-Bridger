"""Regression tests for original FloVerse protocol, using a deterministic fake env."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / 'diffusion_policy'))
from eval import pointgoal_bridge as entry
from eval.floverse_protocol import score_trajectory, move_to_fallback


class Robot:
    def __init__(self): self.pos=np.zeros(3); self.yaw=0.
    def set_position(self, value): self.pos=np.asarray(value)
    def set_orientation(self, value): self.yaw=entry.p.getEulerFromQuaternion(value)[2]
    def get_position(self): return self.pos
    def get_rpy(self): return [0.,0.,self.yaw]


class Env:
    def __init__(self, valid=True):
        self.robots=[Robot()]; self.valid=valid; self.checks=[]
        self.simulator=SimpleNamespace(step=lambda:None)
    def simulator_step(self): pass
    def test_valid_position(self, robot, position, orientation, **kw):
        self.checks.append(position.copy()); return self.valid


def run(tmp_path, monkeypatch, valid=True, fallback=None, budget=1):
    poses=np.array([[i,0,i+1,0] for i in range(20)],float)
    trajectory=tmp_path/'traj_0.txt';np.savetxt(trajectory,poses)
    env=Env(valid)
    monkeypatch.setattr(entry,'capture',lambda e,d: torch.full((3,2,2),float(e.robots[0].pos[0])))
    monkeypatch.setattr(entry,'mark_floorplan',lambda *args:torch.ones(3,96,96))
    seen=[]
    def planner(batch,mode,energy_guidance=None):
        assert energy_guidance is None
        seen.append(batch)
        path=torch.zeros(1,16,2);path[0,:,0]=torch.arange(16).float()
        return path
    maps=SimpleNamespace(goal_tree=lambda goal:None,
        prior=lambda *args,**kwargs:(np.zeros((16,2),np.float32),np.ones(16,np.float32)))
    if fallback is not None: monkeypatch.setattr(entry,'move_to_fallback',fallback)
    args=SimpleNamespace(robot_z=.5,floor_level=0,execute_steps=9,max_steps=budget,
                         collision_limit=1,success_distance=1.,save_plans=False)
    cfg=dict(history_length=3,len_traj_pred=16,astar_step_m=.1)
    result=entry.rollout(env,trajectory,torch.ones(1,96,96),maps,planner,cfg,args,tmp_path/'result',torch.device('cpu'))
    return result,env,seen,tmp_path/'result/traj_0.txt'


def test_index8_history5_to7_chunk_budget_and_endpoint_only(tmp_path,monkeypatch):
    result,env,seen,path=run(tmp_path,monkeypatch)
    assert result['start_index']==8 and result['moves']==8 and result['replans']==1
    assert len(env.checks)==8  # No swept checks, and 1m segments are accepted.
    assert seen[0]['rgb'][0,:,0,0,0].tolist()==[5,6,7]
    torch.testing.assert_close(seen[0]['agent_pose'], torch.tensor([[8., 0., 0.]]))
    torch.testing.assert_close(seen[0]['goal_world'], torch.tensor([[19., 0.]]))
    assert seen[0]['current_rgb'][0,0,0,0].item()==8
    rows=np.loadtxt(path)
    assert rows[0,0]==9 and rows[-1,0]==16  # Initial pose is not recorded.
    assert result['traveled_m']==7  # Original scorer omits initial->first movement.


def test_fallback_collision_count_and_unavailable_fallback(tmp_path,monkeypatch):
    calls=[]
    def fallback(env,floor,goal,z,yaw):
        calls.append(1); env.robots[0].set_position([8.2,0,z]); return np.array([8.2,0]),0.
    result,env,seen,path=run(tmp_path,monkeypatch,False,fallback,budget=5)
    assert len(calls)==1 and result['runtime_collisions']==1 and result['replans']==1
    assert result['oracle_recovery'] and np.loadtxt(path)[3]==1
    other=tmp_path/'none';other.mkdir()
    result,_,_,path=run(other,monkeypatch,False,lambda *args:(None,None),budget=2)
    assert result['runtime_collisions']==0 and result['fallback_failures']==2
    assert result['replans']==2 and path.stat().st_size==0


def test_reference_scoring_truncates_at_first_success_and_counts_collisions():
    ref=np.array([[0,0],[10,0]],float)
    rows=np.array([[8,0,0,0],[9.5,0,0,0],[20,0,0,1]],float)
    score=score_trajectory(rows,ref)
    assert score['success'] and score['traveled_m']==1.5 and score['collisions']==0
    assert score['SPL_reference']==1 and score['SoftSPL_reference']==.95
    assert not score_trajectory(np.empty((0,4)),ref)['success']
    rows=np.tile([0,0,0,1.],(16,1)); rows[-1,0]=10
    assert not score_trajectory(rows,ref)['success']  # 16th collision precedes success check.


def test_original_fallback_uses_simulator_path_and_default_index():
    env=Env();calls=[]
    def shortest(floor,start,goal,entire_path):
        calls.append((floor,entire_path));return np.array([[2.,3.],[3.,3.]]),1.
    env.simulator.scene=SimpleNamespace(get_shortest_path=shortest)
    position,yaw=move_to_fallback(env,2,np.array([9.,9.]),.5,0.)
    assert calls==[(2,True)]
    np.testing.assert_array_equal(position,[2.,3.])
    np.testing.assert_allclose(env.robots[0].get_position(),[2.,3.,.5])
