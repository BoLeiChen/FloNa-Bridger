#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 RUN_NAME PIPELINE_LOG" >&2
    exit 2
fi

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PYTHON_BIN="${PYTHON_BIN:-python}"
readonly RUN_NAME="$1"
readonly PIPELINE_LOG="$2"

cd "${PROJECT_DIR}"
mkdir -p "$(dirname "${PIPELINE_LOG}")"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

echo "[$(date --iso-8601=seconds)] Starting training: ${RUN_NAME}"
echo "[$(date --iso-8601=seconds)] Cgroup: $(cat /proc/self/cgroup | tr '\n' ' ')"
"${PYTHON_BIN}" training/train.py \
    --config training/pointgoal_v3.yaml \
    --run_name "${RUN_NAME}"
echo "[$(date --iso-8601=seconds)] Training completed"
