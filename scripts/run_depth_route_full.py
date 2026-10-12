"""Evaluate only repaired V3; compare with the existing paired 500-task runs."""
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
import collections
import numpy as np
import yaml
from evaluation_runtime import evaluation_environment

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'logs/depth_bridge_full_20261012_081018'


def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def values(rows):
    return np.array([[float(x['success']),x['spl'],float(x['collision_count']>0),x['collision_count']] for x in rows])


def main():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    cfg=yaml.safe_load((ROOT/'experiments/alpha_0p7_depth_route_full.yaml').read_text())
    baseline=yaml.safe_load((OLD/'baseline.yaml').read_text())
    assert {k:v for k,v in cfg.items() if k!='depth_bridge_guidance'}==baseline
    assert cfg['paired_episode_seed']==0 and len(cfg['test_scenes'])==50 and cfg['traj_per_scene']==10
    old_manifest=json.loads((OLD/'manifest.json').read_text());tasks=old_manifest['tasks']
    expected={t['filename'] for t in tasks};assert len(expected)==500
    actual=[]
    for scene in cfg['test_scenes']:
        ids=sorted(int(p.name.split('_')[1]) for p in (ROOT/cfg['testdataset']/scene).glob('traj_*') if p.is_dir())[:10]
        actual += [scene+'_traj_'+str(i)+'.txt' for i in ids]
    assert set(actual)==expected
    hashes={cfg['checkpoint_path']:old_manifest['checkpoint_sha256'],
        cfg['depth_bridge_guidance']['checkpoint']:old_manifest['depth_checkpoint_sha256']}
    assert all(sha(p)==h for p,h in hashes.items())
    depth_meta=json.loads((ROOT/'experiments/depth_model_provenance.json').read_text())
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=cfg['depth_bridge_guidance']['source_dir'],text=True).strip()==depth_meta['upstream_commit']
    assert subprocess.check_output(['git','show','e621486:results/evaluate.py'],cwd=ROOT)==(ROOT/'results/evaluate.py').read_bytes()
    old=json.loads((OLD/'comparison.json').read_text())
    historical={}
    for name in ['baseline','depth_bridge']:
        bykey={(v['scene'],v['id']):v for v in old['records'] if v['method']==name}
        assert set(bykey)=={(t['scene'],t['id']) for t in tasks}
        historical[name]=[bykey[(t['scene'],t['id'])] for t in tasks]
        a=values(historical[name]).mean(0)
        assert np.allclose(a[:3],[old['summary'][name][k] for k in ['sr','spl','collision_rate']])
    env=evaluation_environment(ROOT)
    subprocess.run([sys.executable,'-c','import torch; free,total=torch.cuda.mem_get_info(); print(free,total); assert free>4*1024**3'],env=env,check=True)
    stamp=time.strftime('%Y%m%d_%H%M%S');name='depth_route_full_'+stamp
    job=ROOT/'logs'/name;job.mkdir();snapshot=job/'code';snapshot.mkdir()
    archive=job/'source.tar'
    subprocess.run(['git','archive','HEAD','--output',str(archive)],cwd=ROOT,check=True)
    subprocess.run(['tar','-xf',str(archive),'-C',str(snapshot)],check=True)
    for item in ['datasets','datasets_96','iGibson','checkpoints','external','diffusion_policy','depth_anything_v2']:
        dest=snapshot/item
        if not dest.exists():dest.symlink_to(ROOT/item,target_is_directory=True)
    config=job/'route.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    result=snapshot/'results/exps'/name
    manifest=dict(started=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256=sha(archive),config_sha256=sha(config),checkpoint_hashes=hashes,tasks=tasks,
        prior_results=str(OLD),prior_comparison_sha256=sha(OLD/'comparison.json'),
        output=str(result),expected_episodes=500,paired_episode_seed=0,
        score_protocol=dict(collision_threshold=10,arrival_threshold_m=.30),
        note='Only repaired method runs. Baseline and V2 reused. Same per-episode seed and tasks; no in-run tuning.')
    save(job/'manifest.json',manifest)
    save(ROOT/'logs/depth_route_full_latest.json',dict(job=str(job),result=str(result),supervisor_pid=__import__('os').getpid()))
    print('RUN_DIRECTORY',job,flush=True)
    try:
        command=[sys.executable,'-u','test_vgcnav_gtpos.py','--config',str(config),'--exp_name',name,'--num_workers','1','--traj_per_scene','10']
        with (job/'evaluation.log').open('w') as log:
            child=subprocess.Popen(command,cwd=snapshot,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                save(job/'status.json',dict(phase='evaluating',pid=child.pid,completed=len(list(result.glob('trajectory/*.txt'))),expected=500,time=time.time()))
                time.sleep(20)
        if child.returncode:raise RuntimeError('Evaluation exited '+str(child.returncode))
        assert {p.name for p in result.glob('trajectory/*.txt')}==expected
        save(job/'status.json',dict(phase='comparing',completed=500,expected=500,time=time.time()))
        spec=importlib.util.spec_from_file_location('frozen_scorer',str(snapshot/'results/evaluate.py'))
        scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
        rows=[]
        for t in tasks:
            data=np.atleast_2d(np.loadtxt(result/'trajectory'/t['filename']))
            assert len(data)>0 and np.isfinite(data).all()
            success,collisions,shortest,length,d0,dt=scorer.judge_success(data,10,.30,np.load(t['reference']))
            rows.append(dict(method='depth_route',scene=t['scene'],id=t['id'],success=bool(success),
                spl=float(success)*shortest/max(length,shortest) if max(length,shortest)>0 else 0.,
                collision_count=int(collisions),final_distance=float(dt)))
        all_rows=dict(historical,depth_route=rows);columns=['sr','spl','collision_rate','mean_collisions']
        summary={name:dict(zip(columns,values(v).mean(0).tolist()),episodes=len(v)) for name,v in all_rows.items()}
        comparisons={};new=values(rows);scenes=cfg['test_scenes']
        draws=np.random.default_rng(0).integers(0,len(scenes),(10000,len(scenes)))
        for label,oldrows in historical.items():
            reference=values(oldrows);delta=new-reference
            groups=np.array([delta[[i for i,t in enumerate(tasks) if t['scene']==scene]].mean(0) for scene in scenes])
            ci=np.quantile(groups[draws].mean(1),[.025,.975],axis=0)
            comparisons[label]=dict(delta=dict(zip(columns,delta.mean(0).tolist())),
                scene_floor_cluster_ci95={k:ci[:,i].tolist() for i,k in enumerate(columns)},
                gained_success=int(((new[:,0]==1)&(reference[:,0]==0)).sum()),
                lost_success=int(((new[:,0]==0)&(reference[:,0]==1)).sum()))
        diagnostics=collections.Counter()
        for p in result.glob('image/*/depth_guidance.jsonl'):
            diagnostics['episodes']+=1
            for line in p.open():
                c=json.loads(line);diagnostics['cycles']+=1;diagnostics['route_active_cycles']+=c['route_active']
                diagnostics['calibrated_current_cycles']+=c['depth_calibration']['accepted']
                diagnostics['history_fallback_cycles']+=not c['depth_calibration']['accepted'] and bool(c['updates'])
                for u in c['updates']:
                    diagnostics['updates']+=1
                    diagnostics['goal_decreased']+=u['after']['goal']<u['before']['goal']-1e-8
                    diagnostics['collision_decreased']+=u['after']['collision']<u['before']['collision']-1e-7
                    diagnostics['collision_violations']+=u['after']['collision']>u['before']['collision']+1e-7
                    diagnostics['shift_violations']+=u['max_update_m']>.01000011
        assert diagnostics['episodes']==500
        assert all(sha(p)==h for p,h in hashes.items())
        save(job/'comparison.json',dict(summary=summary,comparisons=comparisons,diagnostics=dict(diagnostics),records=rows,
            note='Single paired-seed full evaluation; confidence intervals cluster by scene-floor. Historical baseline and V2 are reused.'))
        save(job/'status.json',dict(phase='complete',completed=500,expected=500,time=time.time(),seconds=time.time()-manifest['started']))
        print(json.dumps(summary),flush=True)
    except BaseException:
        import traceback
        save(job/'failure.json',dict(error=traceback.format_exc(),time=time.time()))
        save(job/'status.json',dict(phase='failed',time=time.time()))
        raise


if __name__=='__main__':main()
