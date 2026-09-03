# FloNavBridger

FloNavBridger is a visual goal-navigation system that refines a metric A* path
with a NaviBridger-style VE denoising bridge. A* supplies both the bridge
endpoint and a persistent per-waypoint condition; visual history, a floorplan,
pose/goal inputs, and a conditional 1-D UNet produce an executable local
trajectory.

This README describes the stable main branch. The consistency-flow experiments
live on the feature/consistency-flow-bridge and
feature/consistency-flow-bridge-v2 branches and have branch-specific READMEs.

## Method

~~~text
RGB history ── frozen DINOv2 ───────────┐
current RGB + floorplan ─ EfficientNet ─┼─ Transformer ─ global condition
current pose + goal pose ───────────────┘

A* local path ─ bridge endpoint xT ─────┐
               └─ UNet local condition ┼─ VE/Heun bridge sampler ─ trajectory
global condition ───────────────────────┘
~~~

The default policy predicts 32 local 2-D waypoints from five RGB observations.
The A* local condition contains position, turn geometry, and a validity mask.
Training adds correlated path/pose perturbations instead of independent
waypoint jitter.

Frozen DINOv2 deep features provide semantic context. A bounded,
zero-initialized shallow spatial residual is activated by a robust, mask-aware
summary of A* curvature, helping the policy retain local obstacle detail around
turns without replacing the deep representation. See
[NAVIBRIDGER_ASTAR_BRIDGE.md](NAVIBRIDGER_ASTAR_BRIDGE.md) for the bridge,
conditioning, and perturbation details.

## Repository layout

~~~text
model/                  policy, vision encoder, dataset and NaviBridge
training/               train/eval loop and experiment configurations
scripts/                train/evaluate and tmux launchers
results/evaluate.py     SR/SPL and collision-budget aggregation
test_vgcnav_gtpos.py    iGibson closed-loop navigation evaluation
tests/                  bridge, A* conditioning and perturbation tests
depth_anything_v2/      minimal local DINOv2 implementation
diffusion_policy/       ConditionalUnet1D dependency and retained license
~~~

Datasets, checkpoints, logs, iGibson assets, and external reference
repositories are intentionally excluded from Git.

## Environment

The validated setup uses Python 3.8, PyTorch 2.4.1, CUDA 12.4, and an NVIDIA
GPU.

~~~bash
conda env create -f environment.yaml
conda activate vgcnav
python -m pip install -e .
~~~

Closed-loop evaluation additionally requires a compatible local iGibson
checkout:

~~~bash
python -m pip install -e ./iGibson
~~~

xFormers is optional. The simulator entry point uses the legacy Gym API
provided by the compatible iGibson checkout.

## External assets and data

Place the Depth Anything V2 ViT-S checkpoint at:

~~~text
checkpoints/depth_anything_v2_vits.pth
~~~

Expected layout:

~~~text
datasets_96/
├── scenes_117/
│   ├── train/<scene>/{traj*,floorplan.png,dataset_<scene>_new.lmdb,...}
│   └── test/<scene>/{traj*,floorplan.png,dataset_<scene>_new.lmdb,...}
└── trav_maps/<scene>/...

datasets/scenes_117/{train,test}/<scene>/floorplan.png
iGibson/igibson/data/g_dataset/<scene>/...
~~~

datasets_96 contains resized offline samples and A* caches. datasets supplies
planner floorplans and reference trajectories. g_dataset contains Gibson scene
meshes. Large directories may be mounted elsewhere and symlinked into the
repository root.

## Training

Review paths, GPU IDs, batch size, and W&B settings in
training/flona_unloc_pose.yaml. The full NaviBridge parameter block is
documented in training/navibridger_astar_bridge.yaml.

Direct launch:

~~~bash
python training/train.py \
  --config training/flona_unloc_pose.yaml \
  --run_name navibridger_run
~~~

Launcher:

~~~bash
scripts/run_navibridger_train.sh navibridger_run logs/navibridger_run.log
~~~

tmux launcher:

~~~bash
scripts/run_training_in_tmux.sh
~~~

Checkpoints are written to logs/VGCNav/<run_name>/; use the EMA checkpoint for
closed-loop comparison.

## Closed-loop evaluation

Quick single-scene evaluation:

~~~bash
python test_vgcnav_gtpos.py \
  --config test_vgc_navibridger.yaml \
  --ckpt_path logs/VGCNav/<run_name>/ema_19.pth \
  --exp_name <experiment_name>
~~~

Full 50-scene evaluation followed by metric aggregation:

~~~bash
scripts/run_navibridger_eval.sh \
  test_vgc_navibridger_full.yaml \
  logs/VGCNav/<run_name>/ema_19.pth \
  <experiment_name> \
  logs/<experiment_name>.log
~~~

The combined launcher trains, evaluates, and aggregates metrics:

~~~bash
scripts/run_navibridger_train_eval.sh <run_name>
~~~

The reported 3x3 table is formed by three collision budgets (10, 30, and 50)
and three arrival thresholds (0.25, 0.30, and 0.35). Keep the trajectory count,
candidate count, execution horizon, collision accounting mode, and recovery
policy identical when comparing runs.

### Collision recovery

test_vgc_navibridger_full.yaml uses deterministic repeated clockwise/right
45-degree recovery turns. test_vgc_navibridger_full_randomturn.yaml chooses
left or right once per collision. The options are:

~~~yaml
collision_accounting_mode: legacy       # or movement_only
recovery_turn_mode: continuous_right    # or random
~~~

Do not mix recovery configurations inside one benchmark comparison.

## Tests

Run the complete maintained suite:

~~~bash
python -m pytest -q tests
~~~

Focused NaviBridge checks:

~~~bash
python -m pytest -q \
  tests/test_navibridger.py \
  tests/test_astar_local_condition_features.py
~~~

## Reproducibility notes

- NaviBridge mode requires a valid A* prior and matching five-channel local
  condition.
- A* cache filenames encode context size, prediction horizon, end slack, and
  planner resolution; changing these settings rebuilds caches.
- DINOv2 remains frozen; trainable shallow fusion starts from the deep-feature
  baseline.
- Train and test must use the same trajectory representation, bridge mode,
  network dimensions, and collision/recovery settings.
- Record the Git commit, YAML configuration, random seed, checkpoint path, and
  result directory for each benchmark.

## Acknowledgements

This project builds on FLoNa/VGCNav, Diffusion Policy, Depth Anything V2/DINOv2,
NaviBridger, and iGibson. Retained third-party components remain under their
included licenses. See [LICENSE](LICENSE) for this repository's license.
