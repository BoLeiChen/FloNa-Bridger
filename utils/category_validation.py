"""Deterministic training-building holdout design, never navigation-test data."""
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
from datasets.sampling_identity import sampling_key


def select_category_validation(manifest, heldout, per_building=32, identity_roots=None):
    if per_building<1:raise ValueError('Invalid validation trajectory budget')
    groups=defaultdict(lambda:defaultdict(list))
    for item in manifest:
        _,root,scene=item['floor_key'].split('::');building=scene.rsplit('_',1)[0]
        if building not in heldout or item['sample_count']<2:continue
        parts=Path(root).resolve().parts
        if not any(p=='train' or p.startswith('train_') for p in parts) or any(p=='test' or p.startswith('test_') for p in parts):
            raise ValueError('Validation must come only from navigation training buildings')
        traj=item['key'].split('::')[-1]
        raw=json.loads((Path(root)/scene/traj/'object/object.json').read_text());v=raw[0] if isinstance(raw,list) else raw
        from datasets.object_category_targets import category_name
        category=category_name(v['object_category'])
        groups[building][category].append((scene,str(v['object_id']),item))
    selected=[];audit={}
    for building,categories in sorted(groups.items()):
        queues={}
        for cat,values in sorted(categories.items()):
            instances=defaultdict(list)
            for scene,oid,item in values:instances[(scene,oid)].append(item)
            queues[cat]=deque()
            pools=[deque(sorted(v,key=lambda m:hashlib.sha256(sampling_key(m['key'], identity_roots).encode()).hexdigest())) for _,v in sorted(instances.items())]
            while any(pools):
                for pool in pools:
                    if pool:queues[cat].append(pool.popleft())
        chosen=[]
        # Category round-robin, with instance/floor round-robin inside categories.
        # Order rotates deterministically per building to avoid alphabetical bias.
        order=sorted(queues,key=lambda cat:hashlib.sha256((building+cat).encode()).hexdigest())
        while len(chosen)<per_building and any(queues.values()):
            for cat in order:
                if queues[cat] and len(chosen)<per_building:chosen.append((cat,queues[cat].popleft()))
        selected.extend(m for _,m in chosen)
        audit[building]=dict(available_categories=len(categories),selected_categories=len({c for c,m in chosen}),
                             trajectories=len(chosen),categories=sorted({c for c,m in chosen}))
    if not selected:raise ValueError('Empty category validation holdout')
    return selected,audit


def summarize_category_validation(rows, weight=.5, tail_weight=.2):
    """Macro buildings/categories with fixed initial/tail mixture weights."""
    if not 0<tail_weight<1:raise ValueError('Validation regime weights must both be positive')
    eligible=[r for r in rows if r['eligible']]
    if not eligible:raise ValueError('No active category validation samples')
    building_values={};grouped=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for r in eligible:grouped[r['building']][r['category']][r['regime']].append(r)
    for building,categories in grouped.items():
        cats=[]
        for regimes in categories.values():
            values=[];weights=[]
            for regime,items in regimes.items():
                active=[r['nll'] for r in items if r['route_valid']]
                loss=(float(np.mean(active)) if active else 0.)+weight*float(np.mean([r['bce'] for r in items]))
                values.append(loss);weights.append(tail_weight if regime=='tail' else 1-tail_weight)
            cats.append(float(np.average(values,weights=weights)))
        building_values[building]=float(np.mean(cats))
    values=np.array(list(building_values.values()))
    rng=np.random.RandomState(0)
    boot=rng.choice(values,(1000,len(values)),replace=True).mean(1)
    covered=[r for r in eligible if not r['reject_target']];missing=[r for r in eligible if r['reject_target']]
    def mean(items,key):return float(np.mean([r[key] for r in items])) if items else None
    def details(items):
        pos=[r for r in items if not r['reject_target']];neg=[r for r in items if r['reject_target']]
        return dict(samples=len(items),coverage=mean(items,'covered'),reject_accuracy=mean(items,'reject_correct'),
                    false_reject_rate=mean(pos,'predicted_reject'),missing_set_recall=mean(neg,'predicted_reject'),
                    route_top1_positive=mean(pos,'top1_positive'),route_top4_positive=mean(pos,'top4_positive'),
                    selected_goal_positive=mean(items,'selected_positive'),reject_brier=mean(items,'brier'))
    return dict(selection_loss=float(values.mean()),selection_loss_building_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                selection_criterion='macro_building_category_hierarchical_nll_plus_reject_bce',
                building_losses=building_values,buildings=len(values),samples=len(rows),active_samples=len(eligible),
                inactive_buildings=sorted({r['building'] for r in rows}-set(building_values)),
                selection_hit_diagnostic='stateless reject threshold 0.5; not closed-loop navigation SR',
                route_nll=mean([r for r in eligible if r['route_valid']],'nll'),reject_loss=mean(eligible,'bce'),
                reject_accuracy=mean(eligible,'reject_correct'),goal_coverage=mean(eligible,'covered'),
                reject_balanced_accuracy=(1-mean(covered,'predicted_reject')+mean(missing,'predicted_reject'))/2 if covered and missing else None,
                diagnostics=details(eligible),by_regime={s:details([r for r in eligible if r['regime']==s]) for s in sorted({r['regime'] for r in eligible})})
