"""GT-free candidate ranking; separate query-hit and multi-target coverage metrics."""
import numpy as np

def rank_candidates(poses,task='object',sigma_m=.75,nms_m=.5,yaw_sigma_deg=20.,nms_yaw_deg=30.,ranking_mode='pose'):
    # Position marginalization is appropriate for ImageGoal -> PointGoal:
    # different camera headings at the same XY must not consume extra goals.
    # Keep pose-aware ranking available for full camera-pose localization.
    if ranking_mode not in ('position','pose'):raise ValueError('Unknown candidate ranking mode')
    poses=np.asarray(poses,float)
    if poses.ndim!=2 or poses.shape[1]!=3 or not np.isfinite(poses[:,:2]).all():raise ValueError('Invalid pose candidates')
    if task=='image' and not np.isfinite(poses[:,2]).all():raise ValueError('Invalid image orientation')
    dist=np.linalg.norm(poses[:,None,:2]-poses[None,:,:2],axis=-1)
    exponent=(dist/sigma_m)**2
    yaw=np.abs(np.arctan2(np.sin(poses[:,None,2]-poses[None,:,2]),np.cos(poses[:,None,2]-poses[None,:,2]))) if task=='image' and ranking_mode=='pose' else None
    if yaw is not None:exponent=exponent+(yaw/np.deg2rad(yaw_sigma_deg))**2
    density=np.exp(-.5*exponent).mean(1)
    order=np.argsort(-density,kind='stable');keep=[]
    for i in order:
        if not any(dist[i,j]<=nms_m and (yaw is None or yaw[i,j]<=np.deg2rad(nms_yaw_deg)) for j in keep):keep.append(int(i))
    # Density weights describe relative sample support, NOT calibrated confidence.
    weights=density[keep]/max(density[keep].sum(),1e-12)
    return np.asarray(keep,dtype=int),weights

def query_metrics(candidates,targets,ks=(1,4,8,16,64),thresholds=(.5,1.,2.)):
    d=np.linalg.norm(np.asarray(candidates)[:,None,:2]-np.asarray(targets)[None,:,:2],axis=-1)
    out={'top1_error_m':float(d[0].min()),'oracle_error_m':float(d.min())}
    for k in ks:
        distances=d[:k]
        for radius in thresholds:
            suffix='%gm_top%d'%(radius,k)
            out['hit_'+suffix]=float(distances.min()<radius)
            out['target_coverage_'+suffix]=float((distances.min(axis=0)<radius).mean())
    return out
