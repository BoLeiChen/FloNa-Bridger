#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 CONFIG CHECKPOINT EXP_NAME PIPELINE_LOG" >&2
    exit 2
fi

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PYTHON_BIN="${PYTHON_BIN:-python}"
readonly NUM_WORKERS="${NUM_WORKERS:-1}"
readonly TRAJ_PER_SCENE="${TRAJ_PER_SCENE:-10}"
export __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
readonly CONFIG="$1"
readonly CHECKPOINT="$2"
readonly EXP_NAME="$3"
readonly PIPELINE_LOG="$4"

cd "${PROJECT_DIR}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

if [[ ! -s "${CHECKPOINT}" ]]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 3
fi

echo "[$(date --iso-8601=seconds)] Starting evaluation: ${EXP_NAME}"
echo "[$(date --iso-8601=seconds)] Service cgroup: $(cat /proc/self/cgroup | tr '\n' ' ')"
"${PYTHON_BIN}" test_vgcnav_gtpos.py \
    --config "${CONFIG}" \
    --ckpt_path "${CHECKPOINT}" \
    --exp_name "${EXP_NAME}" \
    --num_workers "${NUM_WORKERS}" \
    --traj_per_scene "${TRAJ_PER_SCENE}"

echo "[$(date --iso-8601=seconds)] Evaluation finished; calculating 3x3 SR/SPL metrics"
"${PYTHON_BIN}" results/evaluate.py --exp_name "${EXP_NAME}"
echo "[$(date --iso-8601=seconds)] Pipeline completed"
