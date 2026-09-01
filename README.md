# FloNavBridger

FloNavBridger is a visual goal-navigation system that refines an A* path with
a NaviBridger-style VE denoising diffusion bridge. The A* trajectory supplies
both the bridge endpoint and a persistent local condition; frozen DINOv2
features, an RGB/floorplan encoder, pose inputs, and a conditional 1-D UNet
produce an executable local trajectory.

## Method

```text
RGB history ── frozen DINOv2 ───────────┐
current RGB + floorplan ─ EfficientNet ─┼─ Transformer ─ global condition
current pose + goal pose ───────────────┘

A* local path ─ bridge endpoint xT ─────┐
               └─ UNet local condition ┼─ VE/Heun bridge sampler ─ trajectory
global condition ───────────────────────┘
```

The default experiment predicts 32 two-dimensional local waypoints from five
RGB observations. Deep DINO features provide the semantic backbone. A bounded,
zero-initialized shallow-feature residual is gated by a robust summary of A*
path curvature. See [NAVIBRIDGER_ASTAR_BRIDGE.md](NAVIBRIDGER_ASTAR_BRIDGE.md)
for implementation details.

## Layout

```text
model/                  model, dataset, A* utilities, diffusion bridge
training/               training loop and experiment configurations
scripts/                portable train/evaluate launchers
results/evaluate.py     SR/SPL and collision metric aggregation
test_vgcnav_gtpos.py    iGibson closed-loop evaluation
tests/                  bridge, A* conditioning, and pose-noise tests
depth_anything_v2/      minimal local DINOv2 implementation
diffusion_policy/       minimal ConditionalUnet1D dependency and its license
```

Datasets, checkpoints, logs, iGibson assets, and upstream reference repositories
are intentionally excluded from Git.

## Environment

The validated setup uses Python 3.8, PyTorch 2.4.1, CUDA 12.4, and an NVIDIA GPU.

```bash
conda env create -f environment.yaml
conda activate vgcnav
python -m pip install -e .
```

Closed-loop evaluation additionally requires a compatible local iGibson source
checkout:

```bash
python -m pip install -e ./iGibson
```

`xFormers` is optional. The current code uses the legacy Gym API bundled with
the compatible iGibson checkout.

## External assets

Place the Depth Anything V2 ViT-S checkpoint at:

```text
checkpoints/depth_anything_v2_vits.pth
```

Expected data layout:

```text
datasets_96/
├── scenes_117/
│   ├── train/<scene>/{traj*,floorplan.png,dataset_<scene>_new.lmdb,...}
│   └── test/<scene>/{traj*,floorplan.png,dataset_<scene>_new.lmdb,...}
└── trav_maps/<scene>/...

datasets/scenes_117/{train,test}/<scene>/floorplan.png
iGibson/igibson/data/g_dataset/<scene>/...
```

`datasets_96` contains resized offline samples and A* caches. `datasets`
supplies planner floorplans and reference trajectories. `g_dataset` contains
Gibson scene meshes. Large directories may be mounted elsewhere and linked
into the repository root.

## Training

Review paths, GPU IDs, batch size, and W&B settings in
`training/flona_unloc_pose.yaml`, then run:

```bash
python training/train.py \
  --config training/flona_unloc_pose.yaml \
  --run_name navibridger_run
```

or:

```bash
scripts/run_navibridger_train.sh navibridger_run logs/navibridger_run.log
```

Checkpoints are written under `logs/VGCNav/<run_name>/`.

## Evaluation

Quick single-scene evaluation:

```bash
python test_vgcnav_gtpos.py \
  --config test_vgc_navibridger.yaml \
  --ckpt_path logs/VGCNav/<run_name>/ema_19.pth \
  --exp_name <experiment_name>
```

Full evaluation and metric aggregation:

```bash
scripts/run_navibridger_eval.sh \
  test_vgc_navibridger_full.yaml \
  logs/VGCNav/<run_name>/ema_19.pth \
  <experiment_name> \
  logs/<experiment_name>.log
```

The combined launcher trains, evaluates, and calculates metrics:

```bash
scripts/run_navibridger_train_eval.sh <run_name>
```

Use the same `legacy` or `movement_only` collision accounting mode when
comparing experiments.

## Tests

```bash
python -m pytest -q \
  tests/test_navibridger.py \
  tests/test_astar_local_condition_features.py
```

## Notes

- NaviBridger mode requires an A* prior.
- A* cache filenames encode context size, prediction horizon, end slack, and
  planner resolution; changing them triggers cache rebuilds.
- The pretrained DINOv2 backbone remains frozen during training.
- Evaluation checkpoints must match the model configuration.

## Acknowledgements

This project builds on FLoNa/VGCNav, Diffusion Policy, Depth Anything V2/DINOv2,
NaviBridger, and iGibson. Retained third-party components remain under their
included licenses. See [LICENSE](LICENSE) for this repository's license.
