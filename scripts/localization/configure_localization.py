"""Bind locally trained localization parents and SHA-256 hashes to navigation."""
import argparse,hashlib
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',choices=['image','object'],required=True);p.add_argument('--diffusion',required=True);p.add_argument('--ranker');p.add_argument('--refiner');a=p.parse_args()
 nav=ROOT/'config'/(a.task+'goal_v3.yaml');c=yaml.safe_load(nav.read_text());key='goal_localization' if a.task=='image' else 'object_localization';d=c[key]
 d['diffusion_checkpoint']=str(Path(a.diffusion).resolve());d['diffusion_checkpoint_sha256' if a.task=='image' else 'diffusion_sha256']=digest(a.diffusion)
 if a.ranker:
  d['cf2loc_disco_checkpoint' if a.task=='image' else 'disco_checkpoint']=str(Path(a.ranker).resolve());d['cf2loc_disco_checkpoint_sha256' if a.task=='image' else 'disco_sha256']=digest(a.ranker)
 if a.task=='image':
  path=ROOT/'config/localization/image_disco.yaml';dc=yaml.safe_load(path.read_text());dc.update(diffusion_checkpoint=str(Path(a.diffusion).resolve()),diffusion_sha256=digest(a.diffusion))
  if a.refiner:dc.update(refiner_checkpoint=str(Path(a.refiner).resolve()),refiner_sha256=digest(a.refiner))
  path.write_text(yaml.safe_dump(dc,sort_keys=False))
 nav.write_text(yaml.safe_dump(c,sort_keys=False));print('Updated',nav,'; rebuild goal/validation caches for these weights.')
