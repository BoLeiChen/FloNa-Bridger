import argparse
import concurrent.futures
import io
import multiprocessing
import os
import sys
import tarfile
import time
from pathlib import Path
from pathlib import PurePosixPath

import lmdb
import msgpack
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.lmdb_dataset import encode_record


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--tar", action="append", dest="tar_paths", default=[])
    parser.add_argument("--extracted-root", action="append", dest="extracted_roots", default=[])
    parser.add_argument("--config", help="Build every source archive referenced by a FloVerse config")
    parser.add_argument("--traj-pace", type=int, default=4)
    parser.add_argument("--len-traj-pred", type=int, default=16)
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--map-size-gb", type=int, default=600)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def parse_poses(payload):
    poses = []
    for line in payload.decode("utf-8").splitlines():
        values = list(map(float, line.strip().split()))
        if len(values) == 4:
            poses.append(values)
    return np.asarray(poses, dtype=np.float64)


def process_bundle(bundle, resize, scene, traj_name, traj_pace, len_traj_pred):
    poses = parse_poses(bundle["poses"])
    rgb_values = []
    depth_values = []
    current_indices = []
    current_values = []
    for idx in range(len(poses)):
        with Image.open(io.BytesIO(bundle["rgb"][idx])) as image:
            rgb_values.append(np.array(resize(image.convert("RGB")), dtype=np.uint8))
        with Image.open(io.BytesIO(bundle["depth"][idx])) as image:
            depth_values.append(np.array(resize(image)))
            if idx % traj_pace == 0 and idx + len_traj_pred <= len(poses):
                current_indices.append(idx)
                current_values.append(np.array(image))
    object_goal = "null"
    if bundle["object"] is not None:
        import json
        value = json.loads(bundle["object"].decode("utf-8"))
        if isinstance(value, list):
            value = value[0] if value else {}
        object_goal = value.get("object_category", "null")
    return {
        "scene": scene,
        "traj_name": traj_name,
        "poses": poses,
        "rgb": np.stack(rgb_values),
        "depth": np.stack(depth_values),
        "current_indices": np.asarray(current_indices, dtype=np.int32),
        "current_depth": np.stack(current_values),
        "object_goal": object_goal,
    }, len(current_indices)


def process_and_encode(bundle, scene, traj_name, traj_pace, len_traj_pred, compression_level):
    record, sample_count = process_bundle(
        bundle,
        transforms.Resize((96, 96)),
        scene,
        traj_name,
        traj_pace,
        len_traj_pred,
    )
    return encode_record(record, compression_level), sample_count


def main():
    args = parse_args()
    if args.config:
        from omegaconf import OmegaConf
        from datasets.dataset import NavDataset
        from datasets.tar_dataset import build_tar_records

        config = OmegaConf.load(args.config)
        index_dataset = NavDataset(
            config.data_folder,
            config.len_traj_pred,
            config.traj_pace,
            config.history_length,
            True,
        )
        records, missing = build_tar_records(index_dataset.samples_index)
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} source archives; first: {missing[0][0]}")
        args.tar_paths = [record[0] for record in records]
        args.extracted_roots = [record[1] for record in records]
    if len(args.tar_paths) != len(args.extracted_roots):
        raise ValueError("Each --tar needs a matching --extracted-root")
    if not args.tar_paths:
        raise ValueError("Provide --config or at least one --tar/--extracted-root pair")
    if args.workers < 1:
        raise ValueError("workers must be at least 1")
    os.makedirs(args.output, exist_ok=True)
    executor = None
    if args.workers > 1:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
    env = lmdb.open(args.output, map_size=args.map_size_gb * 1024 ** 3, subdir=True, writemap=True, map_async=True)
    resize = transforms.Resize((96, 96))
    with env.begin() as txn:
        saved_manifest = txn.get(b"__manifest__")
        saved_completed = txn.get(b"__completed_tars__")
    manifest = msgpack.unpackb(saved_manifest, raw=False) if saved_manifest else []
    completed_tars = set(msgpack.unpackb(saved_completed, raw=False)) if saved_completed else set()
    total_source = 0
    started = time.time()
    for tar_path, extracted_root in zip(args.tar_paths, args.extracted_roots):
        if tar_path in completed_tars:
            total_source += os.path.getsize(tar_path)
            print(f"scene={PurePosixPath(tar_path).stem} status=already_complete", flush=True)
            continue
        scene = PurePosixPath(tar_path).stem
        total_source += os.path.getsize(tar_path)
        floor_path = os.path.join(extracted_root, scene, "floorplan.png")
        with Image.open(floor_path) as image:
            floorplan = transforms.ToTensor()(resize(image)).numpy()
        floor_key = f"floor::{extracted_root}::{scene}"
        with env.begin(write=True) as txn:
            txn.put(floor_key.encode(), encode_record({"floorplan": floorplan}, args.compression_level))

        current_traj = None
        bundle = None

        pending = []

        def write_result(key, floor_key_value, result):
            payload, sample_count = result
            with env.begin(write=True) as txn:
                txn.put(key.encode(), payload)
            manifest.append({"key": key, "floor_key": floor_key_value, "sample_count": sample_count})

        def drain_one():
            future, key, floor_key_value = pending.pop(0)
            write_result(key, floor_key_value, future.result())

        def flush(traj_name, value):
            if value is None:
                return
            key = f"traj::{extracted_root}::{scene}::{traj_name}"
            if executor is None:
                write_result(
                    key,
                    floor_key,
                    process_and_encode(
                        value, scene, traj_name, args.traj_pace, args.len_traj_pred, args.compression_level
                    ),
                )
            else:
                future = executor.submit(
                    process_and_encode,
                    value,
                    scene,
                    traj_name,
                    args.traj_pace,
                    args.len_traj_pred,
                    args.compression_level,
                )
                pending.append((future, key, floor_key))
                if len(pending) >= args.workers * 2:
                    drain_one()

        with tarfile.open(tar_path, mode="r|") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                parts = PurePosixPath(member.name).parts
                if len(parts) < 3 or not parts[1].startswith("traj_"):
                    continue
                traj_name = parts[1]
                if current_traj is not None and traj_name != current_traj:
                    flush(current_traj, bundle)
                    bundle = None
                if bundle is None:
                    current_traj = traj_name
                    bundle = {"poses": None, "object": None, "rgb": {}, "depth": {}}
                payload = archive.extractfile(member).read()
                relative = parts[2:]
                filename = parts[-1]
                if len(relative) == 1 and filename == f"{traj_name}.txt":
                    bundle["poses"] = payload
                elif relative[0] == "rgb" and filename.endswith(".png"):
                    bundle["rgb"][int(PurePosixPath(filename).stem.rsplit("_", 1)[1])] = payload
                elif relative[0] == "depth" and filename.endswith(".png"):
                    bundle["depth"][int(PurePosixPath(filename).stem.rsplit("_", 1)[1])] = payload
                elif filename == "object.json":
                    bundle["object"] = payload
            flush(current_traj, bundle)
        while pending:
            drain_one()
        completed_tars.add(tar_path)
        with env.begin(write=True) as txn:
            txn.put(b"__manifest__", msgpack.packb(manifest, use_bin_type=True))
            txn.put(b"__completed_tars__", msgpack.packb(sorted(completed_tars), use_bin_type=True))
        env.sync()
        used = (env.info()["last_pgno"] + 1) * env.stat()["psize"]
        print(
            f"scene={scene} trajectories={len(manifest)} samples={sum(x['sample_count'] for x in manifest)} "
            f"source_gib={total_source / 1024**3:.2f} lmdb_gib={used / 1024**3:.2f} elapsed={time.time()-started:.1f}s",
            flush=True,
        )
    env.sync()
    env.close()
    if executor is not None:
        executor.shutdown()


if __name__ == "__main__":
    main()
