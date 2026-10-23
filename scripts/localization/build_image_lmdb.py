"""Resumeable, transactional PNG-byte LMDB builder for CF2Loc indexed frames."""
import argparse, collections, concurrent.futures as cf, hashlib, io, json, time
from pathlib import Path
import lmdb
import numpy as np
from PIL import Image

DTYPE=np.dtype([('traj','<u4'),('frame','<u4'),('pose','<f4',(3,))])

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--index',required=True); p.add_argument('--split',choices=['train','val','test'],required=True)
    p.add_argument('--selection'); p.add_argument('--output',required=True); p.add_argument('--map-gib',type=int,required=True)
    p.add_argument('--workers',type=int,default=8); p.add_argument('--batch',type=int,default=128)
    a=p.parse_args()
    index_path=Path(a.index); metadata=index_path.read_bytes(); meta=json.loads(metadata)
    rows=np.memmap(meta['splits'][a.split]['index_path'],dtype=DTYPE,mode='r')
    selection=np.load(a.selection,mmap_mode='r') if a.selection else np.arange(len(rows),dtype=np.int64)
    assert len(selection)>0 and int(selection.min())>=0 and int(selection.max())<len(rows)
    assert np.all(selection[1:]>selection[:-1]),'Selection must be sorted and unique'
    schema={'version':1,'index_sha256':hashlib.sha256(metadata).hexdigest(),'selection_sha256':hashlib.sha256(selection.tobytes()).hexdigest(),
            'split':a.split,'frames':len(selection),'encoding':'original_png_bytes','image_size':[256,256]}
    output=Path(a.output); output.mkdir(parents=True,exist_ok=True)
    env=lmdb.open(str(output),map_size=a.map_gib*1024**3,subdir=True,max_readers=256)
    already_complete=False
    with env.begin(write=True) as txn:
        old=txn.get(b'__schema__')
        if old and json.loads(old)!=schema: raise ValueError('Refusing to reuse LMDB with a different index/selection')
        txn.put(b'__schema__',json.dumps(schema).encode())
        start=int(txn.get(b'__count__') or b'0')
        if txn.get(b'__complete__'):
            assert start==len(selection)
            already_complete=True
    if already_complete:
        print('ALREADY_COMPLETE',json.dumps(schema),flush=True); env.close(); return
    np.save(output/'selection.npy',selection)
    (output/'schema.json').write_text(json.dumps(schema,indent=2))
    started=time.monotonic(); count=start; byte_count=0
    def read_one(logical):
        row=rows[int(selection[logical])]; traj=meta['trajectories'][int(row['traj'])]
        if traj['split']!=a.split: raise ValueError('Split mismatch')
        scene=meta['scenes'][traj['scene']]
        if scene['source']!='HM3D' or 'Gibson' in traj['path']: raise ValueError('Non-HM3D input')
        source=Path(traj['path'])/'rgb'/f"rgb_{int(row['frame'])}.png"
        payload=source.read_bytes()
        with Image.open(io.BytesIO(payload)) as image:
            if image.size!=(256,256) or image.format!='PNG': raise ValueError(str(source))
            image.verify()
        return logical,payload
    with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
        pending=collections.deque(); next_index=start
        def fill():
            nonlocal next_index
            while next_index<len(selection) and len(pending)<a.workers*4:
                pending.append(pool.submit(read_one,next_index)); next_index+=1
        fill()
        while pending:
            records=[]
            for _ in range(a.batch):
                if not pending: break
                records.append(pending.popleft().result()); fill()
            with env.begin(write=True) as txn:
                for logical,payload in records:
                    assert logical==count
                    key=b'r'+int(logical).to_bytes(8,'big')
                    if not txn.put(key,payload,append=True,overwrite=False): raise ValueError(f'Duplicate/out-of-order record {logical}')
                    count+=1; byte_count+=len(payload)
                txn.put(b'__count__',str(count).encode())
            if count%4096==0 or count==len(selection):
                elapsed=time.monotonic()-started
                result={'written':count,'total':len(selection),'frames_s':round((count-start)/max(elapsed,.001),1),'payload_GiB_this_run':round(byte_count/1024**3,3),'elapsed_s':round(elapsed)}
                print(json.dumps(result),flush=True)
                (output/'progress.json').write_text(json.dumps(result))
    with env.begin(write=True) as txn: txn.put(b'__complete__',b'1')
    env.sync(); env.close()
    (output/'complete.json').write_text(json.dumps(schema,indent=2))
    print('COMPLETE',json.dumps(schema),flush=True)

if __name__=='__main__': main()
