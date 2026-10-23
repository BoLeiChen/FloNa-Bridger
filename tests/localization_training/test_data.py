import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from cf2loc_mm.prepare import DTYPE
from cf2loc_mm.data import LocalizationData,collate,EpochSamples

@pytest.fixture
def dataset(tmp_path):
    root=tmp_path/'index';root.mkdir();scenes=[];trajectories=[];splits={}
    for i,split in enumerate(['train','val','test']):
        scene=tmp_path/'train_with_obj'/(split+'_0');traj=scene/'traj_0';(traj/'rgb').mkdir(parents=True)
        Image.new('RGB',(100,100),'white').save(scene/'floorplan.png')
        for j in range(2):Image.new('RGB',(32,32),(j*100,50,20)).save(traj/'rgb'/('rgb_%d.png'%j))
        scenes.append(dict(source='HM3D',path=str(scene),building=split,split=split,scale=10.,offset=[50.,50.],factor=1.,map_size=[100,100],map_path=str(scene/'floorplan.png')))
        trajectories.append(dict(scene=i,path=str(traj),split=split,start=0,end=2))
        rows=np.zeros(2,dtype=DTYPE);rows['traj']=i;rows['frame']=[0,1];rows['pose']=[[50,50,0],[60,50,.5]];rows.tofile(root/(split+'.bin'))
        splits[split]=dict(buildings=[split],index_path=str(root/(split+'.bin')),frames=2)
        np.save(root/(split+'_image_rows.npy'),np.array([0,1]))
        (root/(split+'_objects.json')).write_text(json.dumps([dict(scene=i,category='chair',split=split,world_xy=[[1,1],[2,2]],object_ids=['chair0','chair1'])]))
    meta=dict(scenes=scenes,trajectories=trajectories,splits=splits)
    raw=json.dumps(meta).encode();(root/'source_index.json').write_bytes(raw)
    info=dict(source_index_sha256=hashlib.sha256(raw).hexdigest());(root/'metadata.json').write_text(json.dumps(info))
    (root/'COMPLETE').write_text(hashlib.sha256(json.dumps(info,sort_keys=True).encode()).hexdigest())
    return root

def test_world_coordinate_roundtrip(dataset):
    d=LocalizationData(dataset,'train','image');s=d[1]
    np.testing.assert_allclose(s['targets'].numpy(),[[1,0,.5]])
    np.testing.assert_allclose(s['origin'],[-5,-5]);np.testing.assert_allclose(s['extent'],[10,10])
    assert s['rgb'].shape==(3,224,224)

def test_object_query_is_multitarget_and_rgb_free(dataset):
    s=LocalizationData(dataset,'val','object')[0]
    assert len(s['targets'])==2 and 'rgb' not in s and 'object_id' not in s
    assert s['category']=='chair'

def test_collation_mask_does_not_create_extra_positives(dataset):
    a=LocalizationData(dataset,'train','object')[0];b=dict(a,targets=a['targets'][:1])
    z=collate([a,b]);assert z['target_mask'].sum().item()==3

def test_deterministic_object_sampling(dataset):
    d=LocalizationData(dataset,'train','object');a=EpochSamples(d,5,42,0);b=EpochSamples(d,5,42,0)
    assert [a[i]['key'] for i in range(5)]==[b[i]['key'] for i in range(5)]

def test_metadata_tampering_rejected(dataset):
    p=dataset/'source_index.json';p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError,match='hash'):LocalizationData(dataset,'train','object')

def test_incomplete_dataset_rejected(dataset):
    (dataset/'COMPLETE').unlink()
    with pytest.raises(ValueError,match='incomplete'):LocalizationData(dataset,'train','object')
