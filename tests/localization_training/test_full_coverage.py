import numpy as np
import pytest
import torch
from torch.utils.data import DistributedSampler
from cf2loc_mm.data import FullEpochSamples,collate
from cf2loc_mm.evaluate import evaluation_indices

class FakeImages:
    task='image'
    def __len__(self):return 101
    def __getitem__(self,i):return i

class FakeObjects:
    task='object'
    queries=[dict(world_xy=[[0,0],[1,1]]),dict(world_xy=[[2,2]])]
    category_groups=[np.array([0]),np.array([1])]
    def __len__(self):return 2
    def __getitem__(self,i):return dict(query=i)

def test_all_images_reached_across_three_ranks():
    d=FullEpochSamples(FakeImages(),108,42,0)
    ids=[]
    for rank in range(3):
        sampler=DistributedSampler(d,num_replicas=3,rank=rank,shuffle=False)
        ids.extend(d[i] for i in sampler)
    assert len(ids)==108 and set(ids)==set(range(101))
    assert d.audit['padding']==7
    np.testing.assert_array_equal(d.order,FullEpochSamples(FakeImages(),108,42,0).order)
    assert not np.array_equal(d.order,FullEpochSamples(FakeImages(),108,42,1).order)

def test_all_object_positives_are_scheduled():
    d=FullEpochSamples(FakeObjects(),108,42,0)
    got=set()
    for rank in range(3):
        for i in DistributedSampler(d,num_replicas=3,rank=rank,shuffle=False):
            row=d[i]
            if 'forced_target' in row:got.add((row['query'],row['forced_target']))
    assert got=={(0,0),(0,1),(1,0)}
    assert d.mandatory==FullEpochSamples(FakeObjects(),108,42,0).mandatory

def test_insufficient_epoch_budget_rejected():
    with pytest.raises(ValueError):FullEpochSamples(FakeImages(),100,42,0)
    with pytest.raises(ValueError):FullEpochSamples(FakeObjects(),2,42,0)

def test_full_eval_keeps_all_frames_not_only_terminal():
    class D:
        task='image';ids=np.arange(4)
        rows=np.array([(0,0),(0,1),(1,0),(1,1)],dtype=[('traj','i4'),('frame','i4')])
        def __len__(self):return 4
    assert evaluation_indices(D(),0)==[1,3]
    assert evaluation_indices(D(),0,image_scope='all_frames')==[0,1,2,3]
    assert len(evaluation_indices(D(),2,image_scope='all_frames'))==2
