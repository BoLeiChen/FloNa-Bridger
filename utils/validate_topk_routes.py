"""Fixed training-building holdout; no navigation-test checkpoint selection."""
import torch
import json
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from localization.goal_hypotheses import signature
from datasets.hypothesis_routes import route_signature
from model.route_objective import route_objective
from utils.category_validation import summarize_category_validation

@torch.no_grad()
def validate_routes(model,config,device):
    package=torch.load(config['validation_samples'],map_location='cpu')
    if package.get('scope')!='topk-route-training-building-holdout':raise ValueError('Wrong validation cache')
    if package['goal_top_k']!=config['goal_top_k'] or package['hypothesis_signature']!=signature(config) or package['exploration_candidates']!=config['exploration_candidates']:raise ValueError('Stale K/localizer validation cache')
    if package.get('route_signature')!=route_signature(config):raise ValueError('Stale route geometry/labels validation cache')
    if config.get('validation_design') in ('category-balanced-v2','image-source-floor-balanced-v2'):
        heldout=set(json.loads(Path(config['validation_buildings_file']).read_text()))
        seen=set()
        for sample in package['samples']:
            if sample.get('building') not in heldout:raise ValueError('Training building found in validation cache')
            if sample.get('label_version')!=config.get('route_supervision'):raise ValueError('Wrong category supervision in cache')
            if sample.get('validation_regime') not in ('initial','tail'):raise ValueError('Unknown validation regime')
            identity=(sample['sample_id'],sample['validation_regime'])
            if identity in seen:raise ValueError('Duplicate validation window')
            seen.add(identity)
    model.eval();ce=0.;n=0;reject_loss=0.;correct=0;covered=0;total=0;rows=[]
    for start in range(0,len(package['samples']),8):
        batch=default_collate(package['samples'][start:start+8]);batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
        obs,goals,_=model.visual_features(batch);logits,reject=model._scores(batch,obs,goals[:,0]);target=batch['route_target'];valid=target.sum(-1)>0
        if config.get('validation_design') in ('category-balanced-v2','image-source-floor-balanced-v2'):
            _,_,_,nll,bce,active,eligible=route_objective(logits,reject,batch,config)
            goal_mask=batch['candidate_mask']&batch['candidate_is_goal'];positive=batch['goal_candidate_positive']
            ranks=logits.masked_fill(~goal_mask,-1e4).argsort(dim=-1,descending=True)
            hit=positive.gather(1,ranks)&goal_mask.gather(1,ranks)
            prob=reject.sigmoid();prediction=prob>=float(config.get('goal_reject_threshold',.5))
            for i in range(len(target)):
                truth=bool(batch['goal_set_reject'][i]);top1=bool(hit[i,0])
                rows.append(dict(building=batch['building'][i],category=batch['object_category'][i],regime=batch['validation_regime'][i],
                    eligible=bool(eligible[i]),route_valid=bool(active[i]),nll=float(nll[i]),bce=float(bce[i]),reject_target=truth,
                    covered=not truth,predicted_reject=bool(prediction[i]),reject_correct=bool(prediction[i])==truth,
                    brier=float((prob[i]-float(truth))**2),top1_positive=top1,top4_positive=bool(hit[i,:4].any()),
                    selected_positive=top1 and not bool(prediction[i])))
            continue
        ce+=float((-(target*F.log_softmax(logits,-1)).sum(-1)*valid).sum());n+=int(valid.sum())
        reject_loss+=float(F.binary_cross_entropy_with_logits(reject,batch['goal_set_reject'],reduction='sum'))
        correct+=int(((reject.sigmoid()>=.5)==batch['goal_set_reject'].bool()).sum());covered+=int((~batch['goal_set_reject'].bool()).sum());total+=len(target)
    if config.get('validation_design') in ('category-balanced-v2','image-source-floor-balanced-v2'):
        result=summarize_category_validation(rows,float(config.get('goal_reject_loss_weight',.5)),float(config.get('validation_tail_weight',.2)))
        if config.get('route_task')=='image':result['selection_criterion']='macro_building_source_floor_hierarchical_nll_plus_reject_bce'
        return result
    if n==0 or not total:raise ValueError('No usable validation samples')
    return dict(route_nll=ce/n,reject_loss=reject_loss/total,selection_loss=ce/n+float(config.get('goal_reject_loss_weight',.5))*reject_loss/total,reject_accuracy=correct/total,goal_coverage=covered/total,samples=total)
