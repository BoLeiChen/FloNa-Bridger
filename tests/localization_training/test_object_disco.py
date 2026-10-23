import inspect
import numpy as np
import torch
from cf2loc_mm.object_disco import metric_crops,ranking_loss,fused_scores,candidates_xy,ObjectDisCo,ObjectDisCoLocalizer,candidate_seed

def test_metric_crop_world_axes_and_centers():
    # Pixel value is its world x coordinate in a four-meter map.
    floor=torch.tensor([.125,.375,.625,.875])[None,None,None,:].expand(1,3,4,4)
    crop=metric_crops(floor,torch.tensor([[[12.,22.]]]),torch.tensor([[10.,20.]]),torch.tensor([[4.,4.]]),sizes=(2.,),pixels=2)
    torch.testing.assert_close(crop[0,0,0],torch.tensor([[.375,.625],[.375,.625]]))
    far=metric_crops(floor,torch.tensor([[[100.,100.]]]),torch.tensor([[10.,20.]]),torch.tensor([[4.,4.]]),sizes=(2.,),pixels=2)
    assert torch.allclose(far,torch.ones_like(far))

def test_no_positive_set_has_finite_negative_gradient():
    x=torch.zeros(1,4,requires_grad=True);mask=torch.tensor([[True,True,True,False]])
    loss=ranking_loss(x,torch.tensor([[2.,3.,4.,0.]]),mask);loss.backward()
    assert torch.isfinite(loss) and (x.grad[0,:3]>0).all() and x.grad[0,3]==0

def test_multiple_positives_and_padding():
    x=torch.tensor([[0.,0.,0.,100.]],requires_grad=True);mask=torch.tensor([[True,True,True,False]])
    ranking_loss(x,torch.tensor([[.1,.2,3.,0.]]),mask).backward()
    assert (x.grad[0,:2]<0).all() and x.grad[0,2]>0 and x.grad[0,3]==0
    assert fused_scores(x.detach(),torch.ones_like(x),mask,1)[0,3]<-1e8

def test_xy_nms_ignores_yaw_and_seed_matches_baseline():
    poses=np.array([[1,2,0],[1,2,3],[4,5,0]])
    ids,_=candidates_xy(poses);assert len(ids)==2
    assert candidate_seed(42,'object:1:chair')==candidate_seed(42,'object:1:chair',0)
    assert candidate_seed(42,'object:1:chair',1)!=candidate_seed(42,'object:1:chair',0)
    assert 'targets' not in inspect.signature(ObjectDisCoLocalizer.predict).parameters

def test_candidate_permutation_equivariance():
    torch.manual_seed(3);model=ObjectDisCo().eval()
    b=dict(floor=torch.rand(1,3,64,64),xy=torch.tensor([[[1.,2.],[3.,4.],[5.,6.]]]),origin=torch.zeros(1,2),extent=torch.ones(1,2)*10,text=torch.rand(1,512))
    with torch.no_grad():
        expected=model(b);b['xy']=b['xy'][:,[2,0,1]];actual=model(b)
    torch.testing.assert_close(actual,expected[:,[2,0,1]],atol=1e-5,rtol=1e-5)

def test_validation_ranking_api_and_paired_baseline():
    from cf2loc_mm.train_object_disco import evaluate
    class D:
        rows=[dict(key='synthetic',category='chair',xy=np.array([[1.,1.],[4.,4.]]),targets=np.array([[1.,1.]]),raw_hit=1.)]
        def __len__(self):return 1
        def __getitem__(self,i):return dict(floor=torch.ones(3,32,32),xy=torch.tensor([[1.,1.],[4.,4.]]),origin=torch.zeros(2),extent=torch.ones(2)*8,text=torch.ones(512),mask=torch.ones(2,dtype=torch.bool),density=torch.tensor([.8,.2]),index=i)
    r,records=evaluate(ObjectDisCo(),D(),'cpu',weights=(0.,1.),predictions=True)
    assert r['0.0']['hit_1m_top1']==1 and r['1.0']['hit_1m_top64']==1 and len(records)==2
