"""Frozen full paired evaluation: 50 scenes x 10 tasks x two methods."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import yaml
from evaluation_runtime import evaluation_environment

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    stamp=time.strftime('%Y%m%d_%H%M%S')
    out=ROOT/'logs'/('depth_bridge_full_'+stamp)
    out.mkdir(parents=True,exist_ok=False)
    configs={label:yaml.safe_load((ROOT/('experiments/alpha_0p7_'+label+'_full.yaml')).read_text())
             for label in ('baseline','depth_bridge')}
    assert {k:v for k,v in configs['depth_bridge'].items() if k!='depth_bridge_guidance'}==configs['baseline']
    cfg=configs['baseline']
    assert len(cfg['test_scenes'])==50 and cfg['traj_per_scene']==10
    assert cfg['diffusion_mode']=='navibridger' and cfg['shallow_residual_alpha_max']==.7
    tasks=[]
    for scene in cfg['test_scenes']:
        folder=ROOT/cfg['testdataset']/scene
        ids=sorted(int(p.name.split('_')[1]) for p in folder.iterdir() if p.is_dir() and p.name.startswith('traj_'))[:10]
        assert len(ids)==10
        for i in ids:
            reference=ROOT/'datasets/scenes_117/test'/scene/('traj_'+str(i))/('traj_'+str(i)+'.npy')
            assert reference.is_file(),str(reference)
            tasks.append(dict(scene=scene,id=i,filename=scene+'_traj_'+str(i)+'.txt',reference=str(reference)))
    checkpoint=Path(cfg['checkpoint_path'])
    expected=json.loads((ROOT/'experiments/baseline_provenance.json').read_text())
    assert digest(checkpoint)==expected['checkpoint_sha256']
    depth_checkpoint=Path(configs['depth_bridge']['depth_bridge_guidance']['checkpoint'])
    depth_hash=digest(depth_checkpoint)
    depth_meta=json.loads((ROOT/'experiments/depth_model_provenance.json').read_text())
    assert depth_hash==depth_meta['sha256']
    source=ROOT/configs['depth_bridge']['depth_bridge_guidance']['source_dir']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==depth_meta['upstream_commit']
    jobs=[]
    for label,cfg in configs.items():
        config=out/(label+'.yaml');config.write_text(yaml.safe_dump(cfg,sort_keys=False))
        name='depth_bridge_full_'+stamp+'_'+label
        command=[sys.executable,'-u','test_vgcnav_gtpos.py','--config',str(config),'--exp_name',name,'--num_workers','1','--traj_per_scene','10']
        jobs.append(dict(label=label,name=name,command=command,config_sha256=digest(config)))
    manifest=dict(started=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        checkpoint_sha256=expected['checkpoint_sha256'],depth_checkpoint_sha256=depth_hash,tasks=tasks,jobs=jobs,
        history=expected,new_episodes=1000,paired_episode_seed=cfg['paired_episode_seed'],
        protocol='original GT localization, legacy collision accounting, right-turn recovery, 32 samples mean, 16 executed points')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('RUN_DIRECTORY',out,flush=True)
    env=evaluation_environment(ROOT)
    for job in jobs:
        started=time.time()
        with (out/(job['label']+'.log')).open('w') as log:
            child=subprocess.Popen(job['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            (out/(job['label']+'_running.json')).write_text(json.dumps(dict(pid=child.pid,started=started)))
            code=child.wait()
        (out/(job['label']+'_exit.json')).write_text(json.dumps(dict(returncode=code,seconds=time.time()-started)))
        if code:raise RuntimeError(job['label']+' failed; see log')
        folder=ROOT/'results/exps'/job['name']/'trajectory'
        assert {p.name for p in folder.glob('*.txt')}=={t['filename'] for t in tasks},'Incomplete results'
        with (out/(job['label']+'_metrics.log')).open('w') as log:
            subprocess.run([sys.executable,'results/evaluate.py','--exp_name',job['name']],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('FINISHED',job['label'],flush=True)
    from results.evaluate import judge_success
    records=[];summary={}
    for job in jobs:
        rows=[]
        for task in tasks:
            data=np.atleast_2d(np.loadtxt(ROOT/'results/exps'/job['name']/'trajectory'/task['filename']))
            ref=np.load(task['reference'])
            arrive,collisions,shortest,actual,d0,dt=judge_success(data,10,.30,ref)
            row=dict(method=job['label'],scene=task['scene'],id=task['id'],success=bool(arrive),
                spl=float(arrive)*shortest/max(actual,shortest) if max(actual,shortest)>0 else 0.,
                collision_count=int(collisions),final_distance=float(dt))
            rows.append(row);records.append(row)
        summary[job['label']]=dict(episodes=len(rows),sr=float(np.mean([v['success'] for v in rows])),
            spl=float(np.mean([v['spl'] for v in rows])),collision_rate=float(np.mean([v['collision_count']>0 for v in rows])))
    differences=[]
    for scene in configs['baseline']['test_scenes']:
        a=[float(v['success']) for v in records if v['method']=='baseline' and v['scene']==scene]
        b=[float(v['success']) for v in records if v['method']=='depth_bridge' and v['scene']==scene]
        differences.append(float(np.mean(b)-np.mean(a)))
    boot=np.random.RandomState(0).choice(differences,(10000,50),replace=True).mean(1)
    summary['paired_sr_difference']=dict(mean=float(np.mean(differences)),scene_floor_cluster_95_ci=np.percentile(boot,[2.5,97.5]).tolist())
    assert digest(checkpoint)==expected['checkpoint_sha256'] and digest(depth_checkpoint)==depth_hash
    (out/'comparison.json').write_text(json.dumps(dict(summary=summary,records=records,history=expected),indent=2))
    (out/'complete.json').write_text(json.dumps(dict(completed=time.time(),episodes=1000)))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
