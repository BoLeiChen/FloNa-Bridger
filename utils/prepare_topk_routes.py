"""Prepare frozen goal caches and K-specific validation on training roots only."""
import argparse,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'diffusion_policy')]
import torch,numpy as np
from omegaconf import OmegaConf
from datasets.sampling_identity import sampling_key
from datasets.pointgoal_bridge_dataset import load_manifest,select_training_manifest
from datasets.hypothesis_route_dataset import HypothesisRouteLMDB
from datasets.hypothesis_routes import route_signature
from localization.goal_hypotheses import signature
from localization.hypothesis_factory import provider_for,trajectory_hypotheses
from utils.category_validation import select_category_validation

def trajectory_pose_count(item,config):
    """Read small pose metadata, not all RGB frames, just to locate the goal image."""
    _,root,scene=item['floor_key'].split('::')
    traj=item['key'].split('::')[-1]
    poses=np.atleast_2d(np.loadtxt(Path(root)/scene/traj/(traj+'.txt')))
    if poses.ndim!=2 or poses.shape[1]<4 or not np.isfinite(poses).all():
        raise ValueError('Invalid trajectory metadata: '+item['key'])
    count=len(poses)
    windows=len(range(0,count-int(config['len_traj_pred'])+1,int(config['traj_pace'])))
    if windows!=item['sample_count']:raise ValueError('Pose metadata/manifest window mismatch: '+item['key'])
    return traj,count

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'config/topk_route_bridge.yaml'));p.add_argument('--device',default='cuda:0');p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1);p.add_argument('--max-trajectories',type=int);p.add_argument('--build-validation',action='store_true');p.add_argument('--validation-windows-per-building',type=int,default=8)
    p.add_argument('--validation-manifest-only',action='store_true')
    p.add_argument('--read-only-cache',action='store_true',help='No localization inference/cache writes')
    a=p.parse_args()
    c=OmegaConf.to_container(OmegaConf.load(a.config),resolve=True)
    manifest=select_training_manifest(load_manifest(c['lmdb_path'],c.get('subset_manifest')),c)
    # Reuse strict training-root checks; inference provider has no target-pose input.
    HypothesisRouteLMDB(c,manifest,0)
    heldout=set(json.loads(Path(c['validation_buildings_file']).read_text()))
    design={};v2=c.get('validation_design')=='category-balanced-v2'
    if a.validation_manifest_only and not a.build_validation:raise ValueError('Manifest-only requires build-validation')
    if a.build_validation:
        if a.world_size!=1 or a.rank!=0:raise ValueError('Validation preparation is single-process')
        if v2:
            manifest,design=select_category_validation(manifest,heldout,int(c.get('validation_trajectories_per_building',32)),identity_roots=c.get('sampling_identity_roots'))
        else:
            selected={}
            for item in sorted(manifest,key=lambda m:hashlib.sha256(sampling_key(m['key'],c.get('sampling_identity_roots')).encode()).hexdigest()):
                building=item['floor_key'].split('::')[-1].rsplit('_',1)[0]
                if building in heldout and item['sample_count']>=4:selected.setdefault(building,item)
            manifest=list(selected.values())
    else:manifest=[m for m in manifest if m['floor_key'].split('::')[-1].rsplit('_',1)[0] not in heldout]
    if a.max_trajectories:manifest=manifest[:a.max_trajectories]
    manifest=manifest[a.rank::a.world_size]
    if a.validation_manifest_only:
        destination=Path(c['validation_samples']).with_suffix('.design.json');destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():raise FileExistsError(destination)
        destination.write_text(json.dumps(dict(design=design,manifest=manifest,heldout_buildings=sorted(heldout),navigation_test_used=False),indent=2))
        print('validation design',len(manifest),'trajectories',len(design),'buildings',destination);return
    provider=provider_for(c,a.device,not a.read_only_cache)
    for i,item in enumerate(manifest):
        traj,count=trajectory_pose_count(item,c);_,root,scene=item['floor_key'].split('::')
        trajectory_hypotheses(provider,c,root,scene,traj,count)
        if i%100==0:print('cached',i+1,'/',len(manifest),flush=True)
    del provider
    if torch.cuda.is_available():torch.cuda.empty_cache()
    if a.build_validation:
        c=dict(c,hypothesis_tail_train_fraction=0.) # Fixed initial Top-K, never hash-random tail selection.
        samples=[]
        for item in manifest:
            regimes=[('initial',0.,int(c.get('validation_windows_per_trajectory',4))),('tail',1.,1)] if v2 else [('initial',0.,a.validation_windows_per_building)]
            for regime,tail_fraction,count in regimes:
                first=1 if v2 else 0 # stride8: first evaluated state matches navigation frame8
                wanted=set(np.linspace(first,item['sample_count']-1,min(count,item['sample_count']-first),dtype=int))
                if v2 and regime=='tail':
                    pick=int.from_bytes(hashlib.sha256((sampling_key(item['key'],c.get('sampling_identity_roots'))+':validation-tail').encode()).digest()[:8],'big')
                    wanted={first+pick%(item['sample_count']-first)}
                ds=HypothesisRouteLMDB(dict(c,hypothesis_tail_train_fraction=tail_fraction,validation_window_indices=sorted(wanted)),[item],len(wanted),0)
                for s in ds:
                    if v2 and regime=='tail' and s['hypothesis_offset']==0:continue
                    s['validation_regime']=regime
                    samples.append({k:v.clone() if torch.is_tensor(v) else v for k,v in s.items()})
        package=dict(route_signature=route_signature(c),scope='topk-route-training-building-holdout',goal_top_k=c['goal_top_k'],exploration_candidates=c['exploration_candidates'],hypothesis_signature=signature(c),samples=samples,
          buildings=sorted(heldout),controller_pretraining_caveat='These training-source buildings were seen by the frozen PointGoal controller; held out from scorer training only.')
        package.update(validation_design=design,actual_buildings=sorted({s['floor_key'].split('::')[-1].rsplit('_',1)[0] for s in samples}),
                       manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest())
        destination=Path(c['validation_samples']);destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():raise FileExistsError('Use a new validation cache path')
        torch.save(package,destination);print('validation samples',len(samples),flush=True)
    print('DONE',len(manifest),flush=True)
if __name__=='__main__':main()
