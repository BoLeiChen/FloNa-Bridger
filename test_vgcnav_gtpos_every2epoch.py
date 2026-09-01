import argparse
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import yaml
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple EMA checkpoints and summarize their metrics."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="test_vgc.yaml",
        help="Path to evaluation yaml config.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Directory containing ema_*.pth checkpoints.",
    )
    parser.add_argument(
        "--min_epoch",
        type=int,
        default=None,
        help="Only evaluate checkpoints with epoch >= min_epoch.",
    )
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=None,
        help="Only evaluate checkpoints with epoch <= max_epoch.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Optional epoch stride. Example: --min_epoch 1 --stride 2 evaluates 1,3,5,...",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of checkpoints to evaluate after filtering.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of rollout worker processes forwarded to test_vgcnav_gtpos.py.",
    )
    parser.add_argument(
        "--traj_per_scene",
        type=int,
        default=None,
        help="Maximum trajectories per scene. Defaults to traj_per_scene in the config file.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to launch evaluation subprocesses.",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help="Output CSV for per-checkpoint metrics. Defaults to <ckpt_dir>/checkpoint_eval_summary.csv",
    )
    parser.add_argument(
        "--collision_th",
        type=int,
        default=10,
        help="Collision threshold used for standard metrics.",
    )
    parser.add_argument(
        "--suc_dis",
        type=float,
        default=0.3,
        help="Success distance threshold used for standard metrics.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run checkpoints even if they already exist in the summary CSV.",
    )
    return parser.parse_args()


def extract_ema_epoch(path: Path) -> Optional[int]:
    match = re.fullmatch(r"ema_(\d+)\.pth", path.name)
    if match is None:
        return None
    return int(match.group(1))


def collect_ema_checkpoints(
    ckpt_dir: Path,
    min_epoch: Optional[int],
    max_epoch: Optional[int],
    stride: Optional[int],
    limit: Optional[int],
) -> List[Path]:
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    ckpts = []
    for path in ckpt_dir.iterdir():
        epoch = extract_ema_epoch(path)
        if epoch is None:
            continue
        if min_epoch is not None and epoch < min_epoch:
            continue
        if max_epoch is not None and epoch > max_epoch:
            continue
        if stride is not None:
            stride_base = 0 if min_epoch is None else min_epoch
            if (epoch - stride_base) % stride != 0:
                continue
        ckpts.append((epoch, path))

    ckpts.sort(key=lambda item: item[0])
    paths = [path for _, path in ckpts]
    if limit is not None:
        paths = paths[:limit]
    return paths


def build_exp_name(ckpt_dir: Path, ckpt_path: Path) -> str:
    return f"{ckpt_dir.name}_{ckpt_path.stem}"


def read_completed_exp_names(summary_csv: Path) -> Set[str]:
    if not summary_csv.exists():
        return set()

    completed = set()
    with summary_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp_name = row.get("Exp_Name")
            status = row.get("Status")
            if exp_name and status == "ok":
                completed.add(exp_name)
    return completed


def judge_success(data, collision_th, suc_dis, shortest_traj):
    shortest_dis = 0.0
    for i in range(len(shortest_traj) - 1):
        shortest_dis += np.linalg.norm(shortest_traj[i][:2] - shortest_traj[i + 1][:2])

    goal = shortest_traj[-1][:2]
    d_0 = np.linalg.norm(data[0][:2] - goal)
    collision_num = 0
    arrive = False
    cul_dis = 0.0

    for i, d in enumerate(data):
        if i > 0:
            cul_dis += np.linalg.norm(d[:2] - data[i - 1][:2])

        if d[4] == 1:
            collision_num += 1
        if collision_num >= collision_th:
            break

        if np.linalg.norm(d[:2] - goal) < suc_dis:
            arrive = True
            break

    final_dist = np.linalg.norm(data[min(i, len(data) - 1)][:2] - goal)
    return arrive, collision_num, shortest_dis, cul_dis, d_0, final_dist


def evaluate_experiment(exp_name: str, testdataset: str, collision_th: int, suc_dis: float) -> Dict[str, float]:
    traj_dir = Path("results/exps") / exp_name / "trajectory"
    if not traj_dir.exists():
        raise FileNotFoundError(f"Trajectory directory not found: {traj_dir}")

    traj_files = sorted(traj_dir.glob("*.txt"))
    if not traj_files:
        raise RuntimeError(f"No trajectory files found in {traj_dir}")

    arrives = []
    collision_nums = []
    spl_values = []
    dist_to_goal = []

    for traj_file in tqdm(traj_files, desc=f"Eval {exp_name}", leave=False):
        try:
            data = np.loadtxt(traj_file)
            if data.ndim == 1:
                data = data.reshape(1, -1)
        except Exception:
            continue

        parts = traj_file.stem.split("_")
        if len(parts) < 4:
            continue

        scene_floor = f"{parts[0]}_{parts[1]}"
        traj_name = f"{parts[2]}_{parts[3]}"
        shortest_traj_path = Path(testdataset) / scene_floor / traj_name / f"{traj_name}.npy"
        if not shortest_traj_path.exists():
            continue

        shortest_traj = np.load(shortest_traj_path)
        arrive, collision_num, shortest_dis, cul_dis, d_0, d_t = judge_success(
            data,
            collision_th,
            suc_dis,
            shortest_traj,
        )

        arrives.append(arrive)
        collision_nums.append(collision_num)
        spl_val = arrive * shortest_dis / max(cul_dis, shortest_dis) if max(cul_dis, shortest_dis) > 0 else 0.0
        spl_values.append(spl_val)
        dist_to_goal.append(d_t)

    if not arrives:
        raise RuntimeError(f"No valid trajectories were evaluated for {exp_name}")

    return {
        "SR": float(np.mean(arrives)),
        "SPL": float(np.mean(spl_values)),
        "Collision_Rate": float(np.mean([1 if c > 0 else 0 for c in collision_nums])),
        "Avg_Dist": float(np.mean(dist_to_goal)),
        "Count": int(len(arrives)),
    }


def append_summary_row(summary_csv: Path, row: Dict[str, object]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = summary_csv.exists()
    with summary_csv.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Time",
                "Exp_Name",
                "Epoch",
                "Checkpoint",
                "SR",
                "SPL",
                "Collision_Rate",
                "Avg_Dist",
                "Count",
                "Eval_Seconds",
                "Status",
                "Note",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_command(command: List[str]) -> float:
    start_time = time.time()
    subprocess.run(command, check=True)
    return time.time() - start_time


def main():
    args = parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    summary_csv = Path(args.summary_csv) if args.summary_csv else ckpt_dir / "checkpoint_eval_summary.csv"

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    testdataset = config["testdataset"]
    traj_per_scene = int(config.get("traj_per_scene", 10)) if args.traj_per_scene is None else args.traj_per_scene

    checkpoints = collect_ema_checkpoints(
        ckpt_dir=ckpt_dir,
        min_epoch=args.min_epoch,
        max_epoch=args.max_epoch,
        stride=args.stride,
        limit=args.limit,
    )
    if not checkpoints:
        raise RuntimeError("No checkpoints matched the requested filters.")

    completed = set() if args.force else read_completed_exp_names(summary_csv)
    pending = []
    for ckpt_path in checkpoints:
        exp_name = build_exp_name(ckpt_dir, ckpt_path)
        if exp_name in completed:
            continue
        pending.append(ckpt_path)

    if not pending:
        print(f"All {len(checkpoints)} checkpoints are already recorded in {summary_csv}")
        return

    progress = tqdm(pending, desc="Checkpoints", dynamic_ncols=True)
    for ckpt_path in progress:
        epoch = extract_ema_epoch(ckpt_path)
        exp_name = build_exp_name(ckpt_dir, ckpt_path)
        progress.set_postfix(epoch=epoch, exp=exp_name)

        tqdm.write(f"[{exp_name}] rollout start")
        rollout_cmd = [
            args.python,
            "test_vgcnav_gtpos.py",
            "--config",
            args.config,
            "--ckpt_path",
            str(ckpt_path),
            "--exp_name",
            exp_name,
            "--num_workers",
            str(args.num_workers),
            "--traj_per_scene",
            str(traj_per_scene),
        ]

        try:
            eval_seconds = run_command(rollout_cmd)
            tqdm.write(f"[{exp_name}] rollout done, computing metrics")
            metrics = evaluate_experiment(
                exp_name=exp_name,
                testdataset=testdataset,
                collision_th=args.collision_th,
                suc_dis=args.suc_dis,
            )
            append_summary_row(
                summary_csv,
                {
                    "Time": datetime.now().strftime("%m-%d %H:%M"),
                    "Exp_Name": exp_name,
                    "Epoch": epoch,
                    "Checkpoint": str(ckpt_path),
                    "SR": f"{metrics['SR']:.4f}",
                    "SPL": f"{metrics['SPL']:.4f}",
                    "Collision_Rate": f"{metrics['Collision_Rate']:.4f}",
                    "Avg_Dist": f"{metrics['Avg_Dist']:.4f}",
                    "Count": metrics["Count"],
                    "Eval_Seconds": f"{eval_seconds:.1f}",
                    "Status": "ok",
                    "Note": "",
                },
            )
            tqdm.write(
                f"[{exp_name}] SR={metrics['SR']:.4f} SPL={metrics['SPL']:.4f} "
                f"Col={metrics['Collision_Rate']:.4f} Dist={metrics['Avg_Dist']:.4f}"
            )
        except Exception as exc:
            append_summary_row(
                summary_csv,
                {
                    "Time": datetime.now().strftime("%m-%d %H:%M"),
                    "Exp_Name": exp_name,
                    "Epoch": epoch,
                    "Checkpoint": str(ckpt_path),
                    "SR": "",
                    "SPL": "",
                    "Collision_Rate": "",
                    "Avg_Dist": "",
                    "Count": "",
                    "Eval_Seconds": "",
                    "Status": "failed",
                    "Note": str(exc),
                },
            )
            raise

    print(f"Done. Summary written to {summary_csv}")


if __name__ == "__main__":
    main()
