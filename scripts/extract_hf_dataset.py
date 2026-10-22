"""Verify the published shards and extract a unified data/ directory."""
import argparse
import hashlib
import os
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def extract_archive(path, destination):
    """Materialize regular files and safe, backward hard links; reject symlinks."""
    destination = Path(destination)
    written = set()
    count = 0
    with tarfile.open(path, 'r:') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or not name.parts or name.parts[0] != 'data':
                raise ValueError('Unsafe archive path: ' + member.name)
            target = destination.joinpath(*name.parts)
            if member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
            elif member.islnk():
                # Only refer to a regular file already materialized by this shard.
                link = PurePosixPath(member.linkname)
                if link not in written:
                    raise ValueError('Unsafe or forward hard link: ' + member.linkname)
                target.parent.mkdir(parents=True, exist_ok=True)
                with destination.joinpath(*link.parts).open('rb') as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
            else:
                raise ValueError('Unexpected non-file entry: ' + member.name)
            written.add(name)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', default='.', help='Parent directory for data/')
    parser.add_argument('--source', required=True, help='Downloaded HF repository directory')
    args = parser.parse_args()
    source = Path(args.source).resolve()
    destination = Path(args.destination).resolve()
    output = destination / 'data'
    if output.exists() or output.is_symlink():
        raise SystemExit('Refusing to overwrite existing ' + str(output))
    manifest = json.loads((source / 'manifest.json').read_text())
    # Verify every archive before creating output.
    for item in manifest['shards']:
        path = source / item['path']
        if not path.is_file() or path.stat().st_size != item['bytes']:
            raise SystemExit('Missing or incomplete archive: ' + str(path))
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != item['sha256']:
            raise SystemExit('SHA-256 mismatch: ' + str(path))
        print('Verified', item['path'], flush=True)
    staging = destination / 'data.extracting'
    if staging.exists():
        raise SystemExit('Incomplete extraction exists: ' + str(staging) + '; move or remove it before retrying.')
    staging.mkdir(parents=True)
    (staging / 'data').mkdir()
    for item in manifest['shards']:
        count = extract_archive(source / item['path'], staging)
        if count != item['files']:
            raise ValueError('Archive file count mismatch: ' + item['path'])
        print('Extracted', item['path'], flush=True)
    os.rename(staging / 'data', output)
    staging.rmdir()
    print('Dataset ready:', output)


if __name__ == '__main__':
    main()
