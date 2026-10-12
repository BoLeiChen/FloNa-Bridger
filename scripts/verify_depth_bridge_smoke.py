"""One paired episode integration check; not a performance benchmark."""
import json,os,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
from evaluation_runtime import evaluation_environment
results=[]
for label in ('baseline','depth_bridge'):
 name='depth_bridge_smoke_runtime_20261012_'+label
 out=root/'results/exps'/name
 if out.exists():raise RuntimeError('Use fresh smoke output directories')
 cmd=['/home/cbl/anaconda3/envs/vgcnav/bin/python','test_vgcnav_gtpos.py','--config',
      'experiments/alpha_0p7_'+label+'_smoke.yaml','--exp_name',name,'--num_workers','1','--traj_per_scene','1']
 start=time.time()
 with (root/'logs'/(name+'.log')).open('w') as log:
  p=subprocess.run(cmd,cwd=root,env=evaluation_environment(root),stdout=log,stderr=subprocess.STDOUT)
 result=dict(label=label,returncode=p.returncode,seconds=time.time()-start,files=len(list((out/'trajectory').glob('*.txt'))))
 print(json.dumps(result),flush=True);results.append(result)
 (root/'logs/depth_bridge_smoke_results.json').write_text(json.dumps(results,indent=2))
 if p.returncode or result['files']!=1:raise RuntimeError('Smoke check failed; inspect log')
