"""Download complete reference checkpoints from GitHub Releases and verify SHA-256."""
import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def download_one(name, info, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / name
    if destination.exists():
        if destination.stat().st_size != info['bytes'] or sha(destination) != info['sha256']:
            raise FileExistsError('Refusing to replace a different checkpoint: ' + str(destination))
        print('Verified', destination, flush=True)
        return
    temporary = destination.with_name(destination.name + '.download')
    try:
        for attempt in range(3):
            try:
                digest = hashlib.sha256()
                size = 0
                print('Downloading complete file:', name, flush=True)
                request = urllib.request.Request(info['url'], headers={'User-Agent': 'FloNa-Bridger-weights'})
                with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as stream:
                    for block in iter(lambda: response.read(8 * 1024**2), b''):
                        stream.write(block)
                        digest.update(block)
                        size += len(block)
                if size != info['bytes'] or digest.hexdigest() != info['sha256']:
                    raise ValueError('Checkpoint size/SHA-256 mismatch: ' + name)
                os.replace(temporary, destination)
                print('Verified', destination, flush=True)
                return
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='checkpoints')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'config/weights.json').read_text())
    for name, info in manifest['files'].items():
        download_one(name, info, args.output)


if __name__ == '__main__':
    main()
