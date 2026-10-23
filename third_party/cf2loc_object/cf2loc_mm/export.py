"""Strip optimizer/RNG state while retaining inference provenance and config."""
import argparse
from pathlib import Path
import torch
from .model import FORMAT

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError(a.output)
    s=torch.load(a.checkpoint,map_location='cpu')
    if s.get('format')!=FORMAT:raise ValueError('Checkpoint format mismatch')
    torch.save({k:s[k] for k in ('format','config','model','dataset_fingerprint','epoch','step','validation')},a.output)
if __name__=='__main__':main()
