#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 6 ]]; then
    echo "Usage: $0 TMUX_SOCKET SESSION CONFIG CHECKPOINT EXP_NAME PIPELINE_LOG" >&2
    exit 2
fi

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly TMUX_SOCKET="$1"
readonly SESSION="$2"
readonly CONFIG="$3"
readonly CHECKPOINT="$4"
readonly EXP_NAME="$5"
readonly PIPELINE_LOG="$6"
readonly EVAL_SCRIPT="${PROJECT_DIR}/scripts/run_navibridger_eval.sh"

tmux -L "${TMUX_SOCKET}" new-session -d -s "${SESSION}" \
    "/bin/bash ${EVAL_SCRIPT} ${CONFIG} ${CHECKPOINT} ${EXP_NAME} ${PIPELINE_LOG}"

# Keep the owning systemd service alive until the tmux evaluation and metric
# aggregation have both finished.
while tmux -L "${TMUX_SOCKET}" has-session -t "${SESSION}" 2>/dev/null; do
    sleep 10
done
