#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 RUN_NAME" >&2
    exit 2
fi

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PYTHON_BIN="${PYTHON_BIN:-python}"
readonly NUM_WORKERS="${NUM_WORKERS:-1}"
readonly TRAJ_PER_SCENE="${TRAJ_PER_SCENE:-10}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
readonly RUN_NAME="$1"
readonly RUN_DIR="${PROJECT_DIR}/logs/VGCNav/${RUN_NAME}"
readonly CHECKPOINT="${RUN_DIR}/ema_5.pth"
readonly EXP_NAME="${RUN_NAME}_ema5_v3"
readonly PIPELINE_LOG="${PROJECT_DIR}/logs/${RUN_NAME}_pipeline.log"

cd "${PROJECT_DIR}"
mkdir -p "$(dirname "${PIPELINE_LOG}")"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

echo "[$(date --iso-8601=seconds)] Starting training: ${RUN_NAME}"
echo "[$(date --iso-8601=seconds)] Service cgroup: $(cat /proc/self/cgroup | tr '\n' ' ')"

"${PYTHON_BIN}" training/train.py \
    --config training/pointgoal_v3.yaml \
    --run_name "${RUN_NAME}"

if [[ ! -s "${CHECKPOINT}" ]]; then
    echo "Missing final checkpoint: ${CHECKPOINT}" >&2
    exit 3
fi

echo "[$(date --iso-8601=seconds)] Training finished; starting evaluation with ${NUM_WORKERS} worker(s)"
"${PYTHON_BIN}" test_vgcnav_gtpos.py \
    --config experiments/alpha_0p7_depth_route_full.yaml \
    --ckpt_path "${CHECKPOINT}" \
    --exp_name "${EXP_NAME}" \
    --num_workers "${NUM_WORKERS}" \
    --traj_per_scene "${TRAJ_PER_SCENE}"

echo "[$(date --iso-8601=seconds)] Evaluation finished; calculating 3x3 SR/SPL metrics"
"${PYTHON_BIN}" results/evaluate.py --exp_name "${EXP_NAME}"
echo "[$(date --iso-8601=seconds)] Pipeline completed"
