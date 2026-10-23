import pytest
import yaml
from cf2loc_mm.train import prepare_run


def test_fresh_run_cannot_overwrite_existing_directory(tmp_path):
    root=tmp_path/'run';config={'setting':'joint_geometry'}
    prepare_run(root,config,None,0,1)
    with pytest.raises(RuntimeError,match='Cannot prepare'):
        prepare_run(root,{'setting':'object_only'},None,0,1)
    assert yaml.safe_load((root/'config.yaml').read_text())==config


def test_late_worker_does_not_reject_rank_zero_directory(tmp_path,monkeypatch):
    root=tmp_path/'run';config={'setting':'joint_geometry'}
    prepare_run(root,config,None,0,1)
    messages=[]
    monkeypatch.setattr('cf2loc_mm.train.dist.broadcast_object_list',lambda value,src:messages.append(value[:]))
    prepare_run(root,config,None,1,3)
    assert messages==[[None]]


def test_startup_failure_is_broadcast_before_raising(tmp_path,monkeypatch):
    messages=[]
    monkeypatch.setattr('cf2loc_mm.train.dist.broadcast_object_list',lambda value,src:messages.append(value[:]))
    with pytest.raises(RuntimeError):prepare_run(tmp_path,{},None,0,3)
    assert len(messages)==1 and messages[0][0] is not None
