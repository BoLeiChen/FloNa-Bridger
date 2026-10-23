#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
if [[ $# != 2 ]]; then
  echo 'Usage: bash scripts/evaluate_objectgoal.sh SELECTOR_CHECKPOINT OUTPUT_DIRECTORY' >&2
  exit 2
fi
export MPLBACKEND="${MPLBACKEND:-Agg}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
python -m eval.multimodal_path --checkpoint "$1" --output "$2" \
  --paths-config config/objectgoal_v3.yaml --goal-type object \
  --scenes-dir "${EVAL_DATA_ROOT:-data/eval_with_obj}" \
  --episode-manifest config/objectgoal_eval_1710.json --max-trajs 30 \
  --goal-top-k 64 --exploration-candidates 4 --execute-steps 9 \
  --object-success-mode category \
  --navigation-controller-checkpoint "${NAV_CONTROLLER:-checkpoints/controller_navigation.pth}" \
  --guidance-config config/pointcontroller_depth_v3.yaml
python scripts/score_objectgoal.py --output "$2"
