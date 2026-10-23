#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
if [[ $# != 2 ]]; then
  echo 'Usage: bash scripts/evaluate_imagegoal.sh SELECTOR_CHECKPOINT OUTPUT_DIRECTORY' >&2
  exit 2
fi
export MPLBACKEND="${MPLBACKEND:-Agg}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
python -m eval.multimodal_path --checkpoint "$1" --output "$2" \
  --paths-config config/imagegoal_v3.yaml --goal-type image \
  --scenes-dir "${EVAL_DATA_ROOT:-data/eval_with_obj}" \
  --episode-manifest config/imagegoal_eval_1470.json --max-trajs 100000 \
  --goal-top-k 32 --execute-steps 9 \
  --v2-policy-override route_switch_log_ratio=0.5 \
  --v2-policy-override route_switch_confirm_cycles=2 \
  --v2-policy-override route_min_hold_cycles=3 \
  --navigation-controller-checkpoint "${NAV_CONTROLLER:-checkpoints/controller_navigation.pth}" \
  --guidance-config config/pointcontroller_depth_v3.yaml
python scripts/score_imagegoal.py --output "$2"
