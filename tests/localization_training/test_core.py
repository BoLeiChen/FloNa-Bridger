import copy
import numpy as np
import pytest
import torch
from cf2loc_mm.model import MultimodalLocalizer
from cf2loc_mm.metrics import rank_candidates,query_metrics
from cf2loc_mm.prepare import category_name,validate_splits
from cf2loc_mm.data import collate

torch.set_num_threads(2)

def cfg(setting):return dict(setting=setting,dim=32,heads=4,map_grid=4,denoiser_layers=1,diffusion_steps=20,train_particles=4,dropout=0.,clip_dim=512)

def batch(task='object'):
    b=dict(task=task,floorplan=torch.rand(2,3,64,64),origin=torch.tensor([[-4.,-6.],[-2.,3.]]),extent=torch.tensor([[20.,30.],[10.,20.]]),targets=torch.tensor([[[0.,0.,.2],[5.,6.,-.3]],[[1.,4.,1.],[0.,0.,0.]]]),target_mask=torch.tensor([[True,True],[True,False]]))
    if task=='object':b['text_features']=torch.randn(2,512)
    else:b.update(image_features=torch.randn(2,5,512),geometry_features=torch.randn(2,4,384))
    return b

@pytest.mark.parametrize('setting',['object_only','joint_clip','joint_geometry'])
def test_backward_sampling_and_bounds(setting):
    m=MultimodalLocalizer(cfg(setting));b=batch();inputs=b if setting=='object_only' else [b,batch('image')]
    loss,_=m(inputs);assert torch.isfinite(loss);loss.backward()
    assert m.text_projection[0].weight.grad.abs().sum()>0
    assert all(p.grad is not None for p in m.parameters())
    if setting!='object_only':assert m.image_projection[0].weight.grad.abs().sum()>0
    if setting=='joint_geometry':assert m.geometry_projection[0].weight.grad.abs().sum()>0
    m.eval();out=m.sample(b,8,4,torch.Generator().manual_seed(1))
    assert torch.isfinite(out[...,:2]).all() and torch.isnan(out[...,2]).all()
    assert (out[...,:2]>=b['origin'][:,None]-1e-5).all()
    assert (out[...,:2]<=b['origin'][:,None]+b['extent'][:,None]+1e-5).all()
    if setting!='object_only':assert torch.isfinite(m.sample(batch('image'),8,4)).all()

@pytest.mark.parametrize('setting',['object_only','joint_clip','joint_geometry'])
def test_condition_has_no_gt_leakage(setting):
    m=MultimodalLocalizer(cfg(setting)).eval();b=batch();c=copy.deepcopy(b)
    c['targets'].fill_(999);c['target_mask'].zero_()
    x=m.context(b);y=m.context(c)
    for a,z in zip(x,y):torch.testing.assert_close(a,z,rtol=0,atol=0)

@pytest.mark.parametrize('setting',['object_only','joint_clip','joint_geometry'])
def test_deterministic_sampling(setting):
    m=MultimodalLocalizer(cfg(setting)).eval();b=batch()
    x=m.sample(b,8,3,torch.Generator().manual_seed(5));y=m.sample(b,8,3,torch.Generator().manual_seed(5))
    torch.testing.assert_close(x,y,equal_nan=True,rtol=0,atol=0)


def test_empty_target_rejected():
    b=batch();b['target_mask'][0]=False
    with pytest.raises(ValueError):MultimodalLocalizer(cfg('object_only'))(b)

def test_object_only_rejects_image():
    with pytest.raises(ValueError):MultimodalLocalizer(cfg('object_only')).context(batch('image'))

def test_multitarget_metric_and_radius_boundary():
    c=np.array([[0.,0.],[10.,0.]])
    t=np.array([[0.,0.],[10.,0.],[20.,0.]])
    v=query_metrics(c,t)
    assert v['hit_1m_top1']==1 and v['target_coverage_1m_top1']==1/3
    assert v['target_coverage_1m_top4']==2/3
    assert query_metrics([[1.,0.]],[[0.,0.]])['hit_1m_top1']==0

def test_density_nms_keeps_different_modes():
    c=np.array([[0,0,np.nan],[.01,0,np.nan],[5,0,np.nan]])
    i,w=rank_candidates(c);assert len(i)==2 and i[0] in (0,1) and i[1]==2;assert np.isclose(w.sum(),1)

def test_object_orientation_cannot_affect_ranking():
    c=np.array([[0.,0.,0.],[0.,0.,3.],[5.,0.,1.]])
    i,w=rank_candidates(c,'object');c[:,2]=np.nan;j,z=rank_candidates(c,'object')
    np.testing.assert_array_equal(i,j);np.testing.assert_allclose(w,z)

def test_image_nms_preserves_opposite_orientations():
    c=np.array([[0.,0.,0.],[0.,0.,np.pi]])
    i,_=rank_candidates(c,'image');assert len(i)==2

def test_position_ranking_marginalizes_image_yaw():
    # Three headings support one place. A distant two-particle mode should not
    # outrank it merely because its camera headings agree.
    c=np.array([[0.,0.,0.],[0.,0.,2.],[0.,0.,-2.],[5.,0.,0.],[5.01,0.,0.]])
    pose,_=rank_candidates(c,'image')
    keep,weights=rank_candidates(c,'image',ranking_mode='position')
    assert pose[0] in (3,4)
    assert keep[0] in (0,1,2) and len(keep)==2
    changed=c.copy();changed[:,2]=np.arange(len(c))
    other,w=rank_candidates(changed,'image',ranking_mode='position')
    np.testing.assert_array_equal(keep,other);np.testing.assert_allclose(weights,w)

@pytest.mark.parametrize('setting',['joint_clip','joint_geometry'])
def test_image_condition_has_no_gt_leakage(setting):
    m=MultimodalLocalizer(cfg(setting)).eval();b=batch('image');c=copy.deepcopy(b)
    c['targets'].fill_(999);c['target_mask'].zero_()
    for x,y in zip(m.context(b),m.context(c)):torch.testing.assert_close(x,y,rtol=0,atol=0)

def test_geometry_contributes_to_image_condition_only():
    m=MultimodalLocalizer(cfg('joint_geometry')).eval();b=batch('image')
    before=m.context(b);b['geometry_features']=torch.randn_like(b['geometry_features'])
    after=m.context(b)
    assert not torch.allclose(before[0],after[0])
    assert not torch.allclose(before[2],after[2])
    obj=batch();x=m.context(obj)
    obj['geometry_features']=torch.full((2,4,384),float('nan'))
    for a,z in zip(x,m.context(obj)):torch.testing.assert_close(a,z,rtol=0,atol=0)

def test_alias_and_no_overmerge():
    assert category_name('washing_machine')=='washing machine'
    assert category_name('couch')=='sofa'
    assert category_name('dining_chair')!='chair'

def test_building_leakage_rejected():
    meta=dict(splits={s:dict(buildings=['same']) for s in ['train','val','test']},scenes=[],trajectories=[])
    with pytest.raises(ValueError,match='leakage'):validate_splits(meta)

def test_wrong_task_mixed_batch_rejected():
    with pytest.raises(ValueError):collate([dict(task='image'),dict(task='object')])

def test_scheduled_positive_is_really_used(monkeypatch):
    monkeypatch.setattr(torch,'multinomial',lambda weights,n,replacement:torch.zeros(len(weights),n,dtype=torch.long,device=weights.device))
    m=MultimodalLocalizer(cfg('object_only'));b=batch()
    b['targets'][0,1,:2]=999
    assert torch.isfinite(m(b)[0])  # Unscheduled sampler only chooses position zero.
    b['forced_target']=torch.tensor([1,-1])
    with pytest.raises(ValueError,match='outside calibrated'):m(b)


def test_scheduled_padding_target_rejected():
    b=batch();b['forced_target']=torch.tensor([-1,1])
    with pytest.raises(ValueError,match='Invalid scheduled'):MultimodalLocalizer(cfg('object_only'))(b)
