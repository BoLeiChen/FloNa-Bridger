"""Copy the validation-selected Lightning checkpoint after a completed run."""
import argparse,json,shutil
from pathlib import Path
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--overwrite',action='store_true');a=p.parse_args()
 source=Path(json.loads((Path(a.run)/'completed.json').read_text())['best_checkpoint']);target=Path(a.output)
 if target.exists() and not a.overwrite:raise FileExistsError(target)
 if not source.is_file():raise FileNotFoundError(source)
 target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);print(target)
