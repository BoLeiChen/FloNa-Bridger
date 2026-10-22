# FloNavBridger: PointGoal with V3 depth and route guidance

This release contains the PointGoal training and evaluation pipeline from
`fix/depth-route-guidance-v3` (evaluated source commit
`4ef718452be31f8d7a4de65c5862ed1ed7f6c98d`). It adds portable configurations and
launchers around that implementation. ImageGoal/ObjectGoal localization and
selection are not part of this release.

## Method and released configuration

A floorplan A* path initializes a variance-exploding diffusion bridge and supplies
persistent five-channel local conditioning (XY, sine/cosine of turn angle, validity).
Five RGB observations, floorplan features, pose and destination supply the global
condition. A frozen DINOv2 backbone provides shallow and deep visual features,
with a learned path-conditioned residual gate capped at 0.7.

The policy predicts 32 future 2-D waypoints. Each navigation cycle averages 32
samples and executes the first 16 waypoints. V3 estimates metric depth from RGB
using Depth Anything V2 Hypersim ViT-S; it does not feed simulator depth into the
policy. A local map with 2.5 cm resolution and a 16-observation history provides
sampling-time guidance. The objective combines collision, additional unknown-space
exposure, additional map-boundary exposure, shape change and route progress, with
weights **1, 1, 1, 0.1, 0.25**, respectively. Guidance starts at noise levels at or
below 0.5. Each correction is limited to 1 cm and accepted only when the total
objective and collision cost do not increase. This is approximate multi-cost
refinement, not a hard collision guarantee or a collision-only energy.

See [bridge details](NAVIBRIDGER_ASTAR_BRIDGE.md),
[V3 map/refinement details](experiments/depth_route_method.json) and
[full evaluation configuration](experiments/alpha_0p7_depth_route_full.yaml).

## Environment

The recorded local environment uses **one NVIDIA GeForce RTX 3090**, Python
3.8.20, PyTorch 2.4.1 and CUDA 12.4. Training requires CUDA; closed-loop evaluation
also requires a working NVIDIA OpenGL/EGL installation.

```bash
git clone https://github.com/BoLeiChen/FloNavBridger.git
cd FloNavBridger
conda env create -f environment.yaml
conda activate vgcnav
python -m pip install -e .
```

Install iGibson separately. The local checkout used commit
`7d9598278b95ec1039e1f4366c66b45346521e44`:

```bash
git clone https://github.com/StanfordVL/iGibson.git iGibson
git -C iGibson checkout 7d9598278b95ec1039e1f4366c66b45346521e44
python -m pip install -e ./iGibson
```

Follow iGibson's installation instructions for its native dependencies and obtain
the Gibson scene assets separately. The environment YAML does not install those
assets or system graphics libraries. Use a compatible legacy Gym environment for
this iGibson checkout. xFormers is optional.

All commands below run from the repository root. Launchers honor `PYTHON_BIN`.
For headless execution, configure your NVIDIA EGL installation; launchers default
`__EGL_VENDOR_LIBRARY_FILENAMES` to `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`
but preserve an explicitly supplied value.

## Required data

The code expects the project's prepared PointGoal demonstrations and floorplans;
a raw Gibson mesh download alone is insufficient. This release does not contain
the demonstration dataset, a dataset-generation pipeline, meshes or trained
navigation weights. Prepare the following directories, or symlink existing ones:

```text
datasets_96/
  scenes_117/
    train/<scene_floor>/floorplan.png
    train/<scene_floor>/traj_<id>/traj_<id>.npy
    train/<scene_floor>/traj_<id>/... RGB images ...
    test/<scene_floor>/... same structure ...
  trav_maps/
    floor_shapes.npy
    <scene>/floor_trav_test_0_modified.png
    <scene_floor>/floor_trav_test_<floor>_modified.png
datasets/
  scenes_117/train/<scene_floor>/floorplan.png
  scenes_117/test/<scene_floor>/floorplan.png
  scenes_117/test/<scene_floor>/traj_<id>/traj_<id>.npy
iGibson/igibson/data/g_dataset/<scene>/... meshes and floors.txt ...
```

`datasets_96` supplies resized model inputs; `datasets` supplies planner floorplans
and reference trajectories for metric aggregation. Both must describe the same
scenes and trajectories. The dataset loader creates image LMDB and A* caches when
needed, so the prepared dataset directory must be writable. See
[dataset loader](model/flona_dataset.py) and [data utilities](model/data_utils.py)
for image naming and trajectory conventions. The training configuration lists 67
training and 50 evaluation **scene-floor entries**, not 117 distinct buildings.

For exact benchmark task identity, see the 500 `(scene, id)` pairs in
[pointgoal_v3_results.json](experiments/pointgoal_v3_results.json). Evaluation
selects the ten lowest numeric trajectory IDs available in each configured test
scene-floor; this must match the recorded task list when reproducing the result.

## Required pretrained assets

Create `checkpoints/` and install the official metric-depth source at the pinned
commit:

```bash
mkdir -p checkpoints external
git clone https://github.com/DepthAnything/Depth-Anything-V2.git external/Depth-Anything-V2
git -C external/Depth-Anything-V2 checkout a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
curl -fL --retry 3 -o checkpoints/depth_anything_v2_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
curl -fL --retry 3 -o checkpoints/depth_anything_v2_metric_hypersim_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth
```

The first checkpoint initializes the policy's frozen visual backbone and is also
required when constructing the policy for evaluation. The second is the separate
RGB-to-metric-depth estimator. Its recorded SHA-256 is:

```text
b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545
```

The navigation checkpoint used for the reported V3 result is named
`checkpoints/navibridger_alpha_0p7_ema5.pth`, with SHA-256:

```text
f400884f9f8c91c6ba4e4c5614ae63dcab0976fe1e77b0d5953000c79cac8b2f
```

That navigation checkpoint is **not included** in Git. Train one using the next
section or supply the original checkpoint separately. A newly trained checkpoint
need not reproduce the exact recorded scores. Weight provenance is recorded in
[baseline_provenance.json](experiments/baseline_provenance.json) and
[depth_model_provenance.json](experiments/depth_model_provenance.json).

## Train the PointGoal bridge

The [released training configuration](training/pointgoal_v3.yaml) preserves the
archived six-epoch, gate-0.7 experiment settings. W&B is disabled by default.

```bash
python training/train.py --config training/pointgoal_v3.yaml --run_name pointgoal_v3
```

Alternatively, capture console output with the launcher:

```bash
bash scripts/run_navibridger_train.sh pointgoal_v3 logs/pointgoal_v3_train.log
```

Use a fresh run name: training refuses to overwrite an existing run directory.
Important settings:

| Setting | Value |
|---|---|
| Epochs / batch size / data-loader workers | 6 / 64 / 8 |
| Optimizer | AdamW, learning rate 1e-4, weight decay 0.01 |
| Adam moments / epsilon | (0.9, 0.999) / 1e-8 |
| Learning-rate schedule | Cosine over 6 epochs; no warm-up |
| Parameter EMA | 0.999 |
| Objective | 0.999 bridge loss + 0.001 distance MSE |
| Bridge sigma min / max / data | 0.002 / 10 / 0.5 |
| Sampling levels / Karras exponent / churn ratio | 10 / 7 / 0.33 |
| Seed | 0 |
| Checkpoint interval | Every 2 epochs |

Checkpoints are saved under `logs/VGCNav/pointgoal_v3/`; the sixth-epoch EMA is
`ema_5.pth` (zero-based epoch index). The active loop does not apply gradient
clipping despite a retained legacy `max_norm` configuration field. Legacy selector
and negative-sampling fields in the archived YAML do not add selector training to
this PointGoal loss. V3 depth/route refinement is applied only at inference and
requires no retraining. Pose/path perturbation and visual fusion details are in
the bridge documentation and source.

## Evaluate

Use the same trained checkpoint for baseline and guided comparisons. The examples
below use the checkpoint produced above; `--ckpt_path` overrides the YAML path.
Start with a small scene subset:

```bash
python test_vgcnav_gtpos.py \
  --config experiments/alpha_0p7_depth_route_smoke.yaml \
  --ckpt_path logs/VGCNav/pointgoal_v3/ema_5.pth \
  --exp_name pointgoal_v3_smoke --traj_per_scene 1
```

Run all 50 scene-floor entries, ten episodes each, then aggregate metrics:

```bash
NUM_WORKERS=1 TRAJ_PER_SCENE=10 bash scripts/run_navibridger_eval.sh \
  experiments/alpha_0p7_depth_route_full.yaml \
  logs/VGCNav/pointgoal_v3/ema_5.pth \
  pointgoal_v3_full logs/pointgoal_v3_full.log
```

The launcher runs both navigation and `results/evaluate.py`. Increase workers
only if GPU memory permits; all workers may share the configured GPU. To score an
existing run separately:

```bash
python results/evaluate.py --exp_name pointgoal_v3_full
```

For the paired baseline, use the identical command with
`experiments/alpha_0p7_baseline_full.yaml` and a different experiment/log name.
`alpha_0p7_depth_bridge_full.yaml` retains the earlier depth-guidance ablation.
A complete train-then-V3-evaluate pipeline is also available:

```bash
bash scripts/run_navibridger_train_eval.sh pointgoal_v3_pipeline
```

Output locations:

```text
logs/VGCNav/<run_name>/              training checkpoints
results/exps/<exp_name>/trajectory/ recorded trajectories
results/exps/<exp_name>/image/      rendered diagnostics
evaluation_summary.csv             standard aggregate metrics
```

Use a new experiment name for each run to avoid mixing old and new trajectories.
For a full benchmark, check that exactly 500 trajectory files were produced and
that aggregation reports 500 valid episodes. The scorer skips missing reference
files, so a successful process exit alone does not establish benchmark completeness.

### Protocol and recorded results

The standard report uses goal distance **0.30 m**, collision threshold **10**, and
legacy collision accounting (including recovery flags). The online rollout stop
radius is 0.25 m. Offline scoring checks budget exhaustion before goal arrival at
each logged state. The evaluator also prints a 3x3 threshold sweep, but those
numbers must not be mixed with the standard result. Recovery uses repeated
clockwise 45-degree turns; A* heading realignment is disabled. Paired episode
seeds are derived from seed 0 and scene/trajectory identity. SPL uses the stored
reference-trajectory length.

| Paired 500-episode experiment | SR | SPL |
|---|---:|---:|
| Frozen bridge baseline | 88.60% | 85.74% |
| Earlier depth bridge | 88.60% | 85.94% |
| **V3 depth + route guidance** | **89.80%** | **86.41%** |

These are archived results, not a fresh benchmark of this packaging commit. The
older independent baseline run scored 89.00% / 86.13%; use the paired baseline
above when measuring V3's improvement. V4 experiments are not included.

## Validation and repository layout

```bash
python -m pytest -q tests
```

The tests exercise A* conditions, the bridge, depth geometry, floor calibration,
route energy and bounded refinement acceptance. They do not replace a complete
GPU/iGibson navigation benchmark.

```text
model/                  policy, dataset, bridge and V3 guidance
training/               training entrypoint, loop and six-epoch config
experiments/            portable evaluation configs, provenance and results
scripts/                portable training/evaluation launchers
results/evaluate.py     offline SR/SPL scoring
test_vgcnav_gtpos.py    iGibson closed-loop evaluation
tests/                  maintained unit tests
depth_anything_v2/      policy visual backbone dependency
diffusion_policy/       conditional U-Net dependency
```

Datasets, weights, simulator assets, external source checkouts and generated logs
are excluded from Git. This release retains the V3 model and guidance behavior;
local driver workarounds and scripts requiring historical log directories have
been removed.

## Acknowledgements and licenses

This project builds on FLoNa/VGCNav, NaviBridger, Diffusion Policy, Depth Anything
V2/DINOv2 and iGibson. See [LICENSE](LICENSE) and the retained third-party license
notices. External datasets and pretrained weights retain their upstream terms.
