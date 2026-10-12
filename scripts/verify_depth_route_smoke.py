"""Fixed navigation tasks verify V3 activation, not performance gains."""
from pathlib import Path
import collections
import json
import subprocess
import sys
import time
import yaml
import hashlib
from evaluation_runtime import evaluation_environment

root=Path(__file__).resolve().parents[1]
name='depth_route_smoke_'+time.strftime('%Y%m%d_%H%M%S')
out=root/'results/exps'/name
log=root/'logs'/(name+'.log')
record=dict(name=name,output=str(out),log=str(log),started=time.time())
record['source_sha256']={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
    ['model/depth_bridge_guidance.py','model/depth_bridge_route.py','test_vgcnav_gtpos.py',
     'experiments/alpha_0p7_depth_route_smoke.yaml']}
pointer=root/'logs/depth_route_smoke_latest.json'
pointer.write_text(json.dumps(record,indent=2))
env=evaluation_environment(root)
config=yaml.safe_load((root/'experiments/alpha_0p7_depth_route_smoke.yaml').read_text())
per_scene=int(config['traj_per_scene']);expected=len(config['test_scenes'])*per_scene
check=subprocess.run([sys.executable,'-c',
    'import torch; free,total=torch.cuda.mem_get_info(); print(free,total); assert free>4*1024**3'],
    env=env,check=True,capture_output=True,text=True)
print(check.stdout,flush=True)
command=[sys.executable,'test_vgcnav_gtpos.py','--config','experiments/alpha_0p7_depth_route_smoke.yaml',
    '--exp_name',name,'--num_workers','1','--traj_per_scene',str(per_scene)]
with log.open('w') as f:
    code=subprocess.run(command,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
stats=collections.Counter();gates=collections.Counter()
for p in out.glob('image/*/depth_guidance.jsonl'):
    stats['episodes']+=1
    for line in p.open():
        cycle=json.loads(line);stats['cycles']+=1
        stats['active_cycles']+=cycle['route_active']
        stats['fusion_added_known_cycles']+=cycle['known_fraction']>cycle['single_view_known_fraction']+1e-8
        for u in cycle['updates']:
            stats['updates']+=1;gates[u['gate_reason']]+=1
            stats['goal_decreased']+=u['after']['goal']<u['before']['goal']-1e-8
            stats['collision_decreased']+=u['after']['collision']<u['before']['collision']-1e-7
            stats['collision_violations']+=u['after']['collision']>u['before']['collision']+1e-7
            stats['shift_violations']+=u['max_update_m']>.01000011
record.update(exit_code=code,seconds=time.time()-record['started'],stats=dict(stats),gates=dict(gates),
    trajectories=len(list((out/'trajectory').glob('*.txt'))))
pointer.write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)
(out/'smoke_summary.json').write_text(json.dumps(record,indent=2))
if code or record['trajectories']!=expected or stats['collision_violations'] or stats['shift_violations']:
    raise RuntimeError('Smoke failed; inspect saved diagnostics')
if not stats['active_cycles'] or not stats['goal_decreased']:
    raise RuntimeError('Route energy did not activate on fixed smoke tasks')
