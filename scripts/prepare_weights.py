"""Assemble the original localization and navigation checkpoints and verify SHA-256."""
import argparse,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def assemble(source,output):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name,info in json.loads((source/'manifest.json').read_text()).items():
        dest=output/name
        if dest.exists():
            if sha(dest)!=info['sha256']:raise FileExistsError('Refusing to replace a different checkpoint: '+str(dest))
            print('Verified',dest);continue
        tmp=dest.with_suffix(dest.suffix+'.partial')
        try:
            with tmp.open('wb') as f:
                for part in info['parts']:
                    p=source/part['path']
                    if p.stat().st_size!=part['bytes'] or sha(p)!=part['sha256']:raise ValueError('Invalid weight part: '+str(p))
                    with p.open('rb') as stream:
                        for b in iter(lambda:stream.read(8*1024**2),b''):f.write(b)
            if tmp.stat().st_size!=info['bytes'] or sha(tmp)!=info['sha256']:raise ValueError('Checkpoint checksum mismatch')
            os.replace(tmp,dest);print('Restored',dest)
        finally:
            if tmp.exists():tmp.unlink()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='checkpoints');p.add_argument('--source',help='Restore one artifact directory instead of both default groups');a=p.parse_args()
    sources=[Path(a.source)] if a.source else [ROOT/'artifacts/localization',ROOT/'artifacts/navigation']
    for source in sources:assemble(source,a.output)
