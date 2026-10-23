"""HM3D subset and sparse-window regressions for the original three-goal loader."""
import ast
import json
import os
from pathlib import Path
import lmdb
import msgpack
import numpy as np
import pytest
import torch
from datasets.data_selection import select_training_manifest, read_subset_manifest
from datasets.lmdb_dataset import LmdbNavDataset, encode_record
from datasets.bridge_maps import map_key

ROOT=Path(__file__).resolve().parents[1]


def test_scene_link_selection_and_map_cache_identity(tmp_path):
    source=tmp_path/'original'; (source/'00001-abc_0').mkdir(parents=True)
    (source/'Gibson_0').mkdir()
    view=tmp_path/'subset';view.mkdir()
    (view/'00001-abc_0').symlink_to(source/'00001-abc_0', target_is_directory=True)
    raw=[dict(key=s,floor_key='floor::%s::%s'%(source,s),sample_count=5)
         for s in ['00001-abc_0','Gibson_0']]
    selected=select_training_manifest(raw,dict(data_folder=[str(view)],train_scene_dataset='hm3d',traj_pace=8,len_traj_pred=16))
    assert [x['key'] for x in selected]==['00001-abc_0']
    assert selected[0]['sample_count']==3
    assert map_key(view,'00001-abc_0')==map_key(source,'00001-abc_0')


def test_three_goal_lmdb_keeps_dense_labels_and_correct_depth_lookup(tmp_path):
    cache=tmp_path/'lmdb';source=tmp_path/'source';source.mkdir()
    poses=np.array([[i*.1,0,i*.1+1,0] for i in range(32)])
    rgb=np.broadcast_to(np.arange(32,dtype=np.uint8)[:,None,None,None],(32,96,96,3)).copy()
    starts=np.arange(0,17,4)
    record=dict(scene='00001-abc_0',traj_name='traj_0',poses=poses,rgb=rgb,
                depth=np.zeros((32,96,96),np.uint8),current_indices=starts,
                current_depth=np.broadcast_to(starts[:,None,None],(5,256,256)).copy(),object_goal='chair')
    raw=[dict(key=s,floor_key='floor::%s::%s'%(source,s),sample_count=5)
         for s in ['00001-abc_0','Gibson_0']]
    env=lmdb.open(str(cache),map_size=16*1024**2)
    with env.begin(write=True) as txn:
        txn.put(b'__manifest__',msgpack.packb(raw,use_bin_type=True))
        for item in raw:
            txn.put(item['key'].encode(),encode_record(dict(record,scene=item['key'])))
            txn.put(item['floor_key'].encode(),encode_record(dict(floorplan=np.zeros((1,96,96),np.float32))))
    env.close()
    subset=tmp_path/'manifest.json'
    subset.write_text(json.dumps(dict(version=1,source_lmdb=str(cache),records=raw[:1])))
    data=LmdbNavDataset(str(cache),16,8,4,True,subset_manifest=str(subset))
    assert len(data)==3 and len(data.manifest)==1
    for sample,start in zip(data,[0,8,16]):
        current_depth,history,depth,history_pose,gt,floor,image_goal,point_goal,obj,idx=sample
        assert obj=='chair' and idx.endswith('::'+str(start))
        assert torch.all(current_depth==start)
        torch.testing.assert_close(history[:,0,0,0],torch.tensor([max(0,t)/255 for t in range(start-4,start)]))
        torch.testing.assert_close(image_goal,torch.full_like(image_goal,31/255))
        torch.testing.assert_close(gt[:,0],torch.arange(16)*.1,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(point_goal,torch.tensor([[(31-start)*.1,0]]),atol=1e-6,rtol=1e-5)
    with pytest.raises(ValueError,match='source LMDB'):
        read_subset_manifest(subset,tmp_path/'wrong')


def test_all_evaluators_shared_scene_selection_excludes_explicit_gibson(tmp_path):
    # Exercise the exact shared function without importing simulator/GL modules.
    tree=ast.parse((ROOT/'eval/common.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='list_scene_ids')
    scope={'os':os};exec(compile(ast.Module(body=[node],type_ignores=[]),'common.py','exec'),scope)
    for name in ['00001-abc_0','00002-def_1','Gibson_0']:(tmp_path/name).mkdir()
    choose=scope['list_scene_ids']
    assert choose(tmp_path)==['00001-abc_0','00002-def_1']
    assert choose(tmp_path,['Gibson_0'])==[]
    assert choose(tmp_path,excluded_scene_ids={'00001-abc_0'})==['00002-def_1']


def test_all_shipped_training_configs_use_hm3d_stride8_subset():
    from omegaconf import OmegaConf
    for path in (ROOT/'config').glob('*.yaml'):
        cfg=OmegaConf.load(path)
        if 'traj_pace' not in cfg:
            continue
        assert cfg.traj_pace==8, path.name
        assert cfg.train_scene_dataset=='hm3d', path.name
        assert len(cfg.data_folder) == (1 if cfg.get('route_task') == 'object' else 2), path.name
    import gzip
    with gzip.open(ROOT/'config/training_selection.json.gz', 'rt') as stream:
        rows=json.load(stream)['records']
    assert len(rows)==175893
    assert sum((row[3]+1)//2 for row in rows)==1772460
