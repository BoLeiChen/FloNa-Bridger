#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
if [[ $# != 2 ]]; then
  echo 'Usage: bash scripts/evaluate.sh CHECKPOINT OUTPUT_DIRECTORY' >&2
  exit 2
fi
export MPLBACKEND="${MPLBACKEND:-Agg}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
python -m eval.pointgoal_bridge --checkpoint "$1" --output "$2" \
  --paths-config config/pointgoal_v3.yaml \
  --scenes-dir "${EVAL_DATA_ROOT:-data/eval_without_obj/HM3D}" \
  --episodes-file config/evaluation_episodes.json \
  --guidance-config config/depth_v3_route.yaml
python scripts/score.py --output "$2" --scenes-dir "${EVAL_DATA_ROOT:-data/eval_without_obj/HM3D}"
