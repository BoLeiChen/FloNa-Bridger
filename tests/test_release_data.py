import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from scripts.prepare_data import prepare
from scripts.score import score
from datasets.lmdb_dataset import decode_record
import lmdb


def test_extracted_cache_preserves_rgb_pose_and_manifest(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'config').mkdir()
    (tmp_path/'config/evaluation_episodes.json').write_text('[["00001-room_0", "traj_0"]]')
    root=tmp_path/'train'; scene=root/'00001-room_0'; traj=scene/'traj_0'
    (traj/'rgb').mkdir(parents=True);(traj/'depth').mkdir()
    (traj/'object').mkdir()
    (traj/'object/object.json').write_text(json.dumps({'object_category': 'chair'}))
    Image.new('L',(96,96),255).save(scene/'floorplan.png')
    poses=np.array([[t*.1,0,t*.1+1,0] for t in range(24)])
    np.savetxt(traj/'traj_0.txt',poses)
    for t in range(24):
        Image.new('RGB',(96,96),(t,t,t)).save(traj/'rgb'/('rgb_%d.png'%t))
        Image.fromarray(np.full((96,96),t+1,dtype=np.uint16)).save(traj/'depth'/('depth_%d.png'%t))
    cfg=dict(data_folder=[str(root),str(root)],lmdb_path=str(tmp_path/'cache'),eval_data_folder=str(root))
    report=prepare(cfg,dict(records=[[0,'00001-room_0','traj_0',3]]),map_size_gb=.1)
    assert report['trajectories']==1 and report['training_windows']==2
    env=lmdb.open(cfg['lmdb_path'],readonly=True,lock=False)
    with env.begin() as txn:
        record=decode_record(txn.get(('traj::%s::00001-room_0::traj_0'%root).encode()))
        np.testing.assert_array_equal(record['poses'],poses)
        assert record['rgb'][7].min()==7
        assert record['object_goal']=='chair'
        assert record['current_indices'].tolist()==[0,4,8]
    env.close()
    with pytest.raises(FileExistsError):prepare(cfg,dict(records=[]))


def test_score_rejects_missing_episodes(tmp_path):
    with pytest.raises(ValueError,match='1 missing'):
        score(tmp_path,tmp_path,[['scene_0','traj_0']])


def test_score_uses_saved_reference_and_rollout(tmp_path):
    traj=tmp_path/'output/scene_0/traj_0';traj.mkdir(parents=True)
    reference=tmp_path/'data/scene_0/traj_0';reference.mkdir(parents=True)
    (traj/'result.json').write_text('{}')
    np.savetxt(traj/'traj_0.txt',[[0,0,0,0],[2,0,0,0]])
    np.savetxt(reference/'traj_0.txt',[[0,0,1,0],[2,0,3,0]])
    rows=score(tmp_path/'output',tmp_path/'data',[['scene_0','traj_0']])
    assert rows[0]['success'] and rows[0]['SPL_reference']==1
