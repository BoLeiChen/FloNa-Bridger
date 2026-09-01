#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 TMUX_SOCKET SESSION RUN_NAME" >&2
    exit 2
fi

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly TMUX_SOCKET="$1"
readonly SESSION="$2"
readonly RUN_NAME="$3"
readonly PIPELINE_SCRIPT="${PROJECT_DIR}/scripts/run_navibridger_train_eval.sh"

tmux -L "${TMUX_SOCKET}" new-session -d -s "${SESSION}" \
    "/bin/bash ${PIPELINE_SCRIPT} ${RUN_NAME}"

# Keep the systemd owner alive until training, evaluation, and metric
# aggregation in the tmux session have all completed.
while tmux -L "${TMUX_SOCKET}" has-session -t "${SESSION}" 2>/dev/null; do
    sleep 10
done
