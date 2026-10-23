"""Portable localization manifests and checkpoint restoration contracts."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]

def test_frozen_localization_manifests():
    base=ROOT/'config/localization'
    provenance=json.loads((base/'provenance.json').read_text())
    for name,digest in provenance['index_files'].items():
        assert hashlib.sha256((base/'index'/name).read_bytes()).hexdigest()==digest
    index=json.loads(gzip.decompress((base/'index/index.json.gz').read_bytes()))
    seen=set()
    dtype=np.dtype([('traj','<u4'),('frame','<u4'),('pose','<f4',(3,))])
    for split,meta in index['splits'].items():
        buildings=set(meta['buildings'])
        assert not seen & buildings
        seen.update(buildings)
        assert len(buildings)==provenance['building_counts'][split]
        rows=np.frombuffer(gzip.decompress((base/'index'/(split+'.bin.gz')).read_bytes()),dtype=dtype)
        assert len(rows)==provenance['frame_counts'][split]
        assert np.isfinite(rows['pose']).all()
        assert rows['traj'].max()<len(index['trajectories'])
    assert all(not Path(s['path']).is_absolute() for s in index['scenes'])
    assert all(not Path(t['path']).is_absolute() for t in index['trajectories'])

def test_complete_file_download_integrity_and_overwrite_protection(tmp_path):
    spec=importlib.util.spec_from_file_location('download_weights',ROOT/'scripts/download_weights.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    source=tmp_path/'original.pt';payload=b'complete original checkpoint';source.write_bytes(payload)
    info=dict(url=source.as_uri(),bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
    output=tmp_path/'output'
    mod.download_one('model.pt',info,output)
    assert (output/'model.pt').read_bytes()==payload
    mod.download_one('model.pt',info,output)
    (output/'model.pt').write_bytes(b'locally trained checkpoint')
    with pytest.raises(FileExistsError):mod.download_one('model.pt',info,output)
    (output/'model.pt').unlink();source.write_bytes(b'corrupted download')
    with pytest.raises(ValueError):mod.download_one('model.pt',info,output)
    assert not (output/'model.pt').exists()
    assert not (output/'model.pt.download').exists()


def test_complete_weight_urls_match_reference_pins():
    pins=json.loads((ROOT/'config/pretrained_weights.json').read_text())
    manifest=json.loads((ROOT/'config/weights.json').read_text())['files']
    audit=json.loads((ROOT/'config/reproduction_audit.json').read_text())
    assert manifest['selector.pth']['sha256']==audit['reference_selector_sha256']
    assert manifest['controller_navigation.pth']['sha256']==audit['reference_controller_sha256']
    assert set(pins)-set(manifest) <= {'ViT-B-16.pt'}  # Official CLIP download stays external.
    for name,digest in pins.items():
        if name=='ViT-B-16.pt':continue
        assert manifest[name]['sha256']==digest
        assert manifest[name]['bytes']>0
        assert '/releases/download/hm3d-reference-weights-v1/' in manifest[name]['url']
        assert '.part-' not in manifest[name]['url']
    assert 'refiner_init.ckpt' not in manifest
