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

def test_weight_restore_integrity_and_overwrite_protection(tmp_path):
    spec=importlib.util.spec_from_file_location('restore',ROOT/'scripts/prepare_weights.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    source=tmp_path/'source';source.mkdir();out=tmp_path/'out'
    payload=b'original checkpoint';part=source/'a.part';part.write_bytes(payload)
    h=hashlib.sha256(payload).hexdigest()
    (source/'manifest.json').write_text(json.dumps({'a.pt':dict(bytes=len(payload),sha256=h,parts=[dict(path='a.part',bytes=len(payload),sha256=h)])}))
    mod.assemble(source,out);assert (out/'a.pt').read_bytes()==payload
    mod.assemble(source,out)
    (out/'a.pt').write_bytes(b'local training checkpoint')
    with pytest.raises(FileExistsError):mod.assemble(source,out)
    (out/'a.pt').unlink();part.write_bytes(b'corrupted download')
    with pytest.raises(ValueError):mod.assemble(source,out)
    assert not (out/'a.pt').exists()
    assert not (out/'a.pt.partial').exists()
