#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 TMUX_SOCKET SESSION RUN_NAME PIPELINE_LOG" >&2
    exit 2
fi

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly TMUX_SOCKET="$1"
readonly SESSION="$2"
readonly RUN_NAME="$3"
readonly PIPELINE_LOG="$4"
readonly TRAIN_SCRIPT="${PROJECT_DIR}/scripts/run_navibridger_train.sh"

tmux -L "${TMUX_SOCKET}" new-session -d -s "${SESSION}" \
    "/bin/bash ${TRAIN_SCRIPT} ${RUN_NAME} ${PIPELINE_LOG}"

# Keep the owning systemd service alive for the entire tmux session so the
# tmux server and all training workers remain in its persistent cgroup.
while tmux -L "${TMUX_SOCKET}" has-session -t "${SESSION}" 2>/dev/null; do
    sleep 10
done
