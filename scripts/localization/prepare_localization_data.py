"""Rebuild the frozen localization index and calibrated maps from raw FloVerse data."""
import argparse,gzip,json,math,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
DTYPE=np.dtype([('traj','<u4'),('frame','<u4'),('pose','<f4',(3,))])
ROOT=Path(__file__).resolve().parents[2]

def prepare(data_root,scene_assets,output,manifest_dir):
    data_root=Path(data_root).resolve();assets=Path(scene_assets).resolve();out=Path(output).resolve()
    if out.exists():raise FileExistsError('Use a fresh localization output: '+str(out))
    provenance=json.loads((ROOT/'config/localization/provenance.json').read_text())
    for name,digest in provenance['index_files'].items():
        if hashlib.sha256((Path(manifest_dir)/name).read_bytes()).hexdigest()!=digest:raise ValueError('Localization manifest checksum mismatch: '+name)
    index=json.loads(gzip.decompress((Path(manifest_dir)/'index.json.gz').read_bytes()))
    groups={k:set(v['buildings']) for k,v in index['splits'].items()}
    if any(groups[a]&groups[b] for a,b in [('train','val'),('train','test'),('val','test')]):raise ValueError('Overlapping localization buildings')
    payloads={}
    for split,meta in index['splits'].items():
        raw=gzip.decompress((Path(manifest_dir)/(split+'.bin.gz')).read_bytes());rows=np.frombuffer(raw,dtype=DTYPE)
        if len(rows)!=meta['frames'] or not np.isfinite(rows['pose']).all():raise ValueError('Invalid frame manifest: '+split)
        payloads[split]=raw
    # Check calibration before creating an output. RGB availability is checked by the LMDB builder.
    for scene in index['scenes']:
        p=data_root/scene['path'];geo=assets/scene['building'];floor=scene['floor']
        if not (p/'floorplan.png').is_file():raise FileNotFoundError(p/'floorplan.png')
        scale=float((geo/'scale.txt').read_text().splitlines()[floor]);offset=list(map(float,(geo/'offset.txt').read_text().splitlines()[floor].split()))
        if not np.isclose(scale,scene['scale']) or not np.allclose(offset,scene['offset'],atol=1e-5):raise ValueError('Calibration differs from reference: '+str(p))
        if scene['building'] not in groups[scene['split']]:raise ValueError('Scene/split mismatch')
    out.mkdir(parents=True);(out/'maps').mkdir()
    for scene in index['scenes']:
        p=data_root/scene['path'];factor=scene['factor'];dest=out/scene['map_path']
        with Image.open(p/'floorplan.png') as source:
            image=source.convert('L');size=tuple(int(math.ceil(v*factor)) for v in image.size)
            if list(size)!=scene['map_size']:raise ValueError('Floorplan size differs: '+str(p))
            image.transform(size,Image.Transform.AFFINE,(1/factor,0,0,0,1/factor,0),resample=Image.Resampling.BILINEAR,fillcolor=255).save(dest)
        scene.update(path=str(p.resolve()),map_path=str(dest))
    for trajectory in index['trajectories']:trajectory['path']=str((data_root/trajectory['path']).resolve())
    for split,raw in payloads.items():
        p=out/(split+'.bin');p.write_bytes(raw);index['splits'][split]['index_path']=str(p)
    index['roots']=[str((data_root/p).resolve()) for p in index['roots']]
    (out/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    # The object builder needs a trajectory manifest, not the navigation LMDB itself.
    selection=json.loads(gzip.decompress((ROOT/'config/training_selection.json.gz').read_bytes()))['records'];roots=[data_root/'train_without_obj/HM3D',data_root/'train_with_obj']
    records=[dict(key='traj::%s::%s::%s'%(roots[g].resolve(),scene,traj),floor_key='floor::%s::%s'%(roots[g].resolve(),scene),sample_count=n) for g,scene,traj,n in selection]
    (out/'navigation_train_manifest.json').write_text(json.dumps(dict(records=records)))
    # Explicit exclusion metadata: scene names only, no navigation-test observations.
    exclusions=json.loads((ROOT/'config/localization/navigation_test_scenes.json').read_text())
    for group,scenes in exclusions.items():
        for scene in scenes:(out/'navigation_exclusions'/group/scene).mkdir(parents=True,exist_ok=True)
    print(json.dumps({k:dict(frames=v['frames'],buildings=len(v['buildings'])) for k,v in index['splits'].items()},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',default='data');p.add_argument('--scene-assets',default='iGibson/igibson/data/g_dataset');p.add_argument('--output',default='data/localization/image');p.add_argument('--manifest-dir',default=str(ROOT/'config/localization/index'));a=p.parse_args();prepare(a.data_root,a.scene_assets,a.output,a.manifest_dir)
