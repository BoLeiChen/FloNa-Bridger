# FloNavBridger — ObjectGoal V3 depth guidance

ObjectGoal route-selector training and evaluation with a frozen PointGoal controller and V3 depth guidance. Run all commands from the repository root.

## Setup

Use Linux with NVIDIA GPUs; the supplied training protocol uses three GPUs.

```bash
git clone --branch objectgoal-hm3d-v3 --single-branch https://github.com/BoLeiChen/FloNa-Bridger.git
cd FloNa-Bridger
conda env create -f environment.yaml
conda activate flona-objectgoal
python -m pip install -e .
git clone https://github.com/StanfordVL/iGibson.git iGibson
git -C iGibson checkout 7d9598278b95ec1039e1f4366c66b45346521e44
git -C iGibson submodule update --init --recursive
python -m pip install -e ./iGibson
python -m igibson.utils.assets_utils --download_assets
```

Install iGibson's [native dependencies](https://github.com/StanfordVL/iGibson) and ensure NVIDIA OpenGL/EGL works. Extract `floverse_scene_dataset.tar` from [FloVerse-1.6k](https://modelscope.cn/datasets/weiqihuang/floverse-1.6k) into `iGibson/igibson/data/`, producing `g_dataset/`. Preserve the complete scene assets, including object annotations.

## Data and pretrained models

Our dataset is a fixed HM3D subset reorganized from [FloVerse-1.6k (original data download)](https://modelscope.cn/datasets/weiqihuang/floverse-1.6k). Download `train_without_obj/HM3D` and `train_with_obj`, plus `eval_with_obj`. Extract the per-scene archives, preserve scene/trajectory names and all frames, and place or symlink the extracted scene folders under the following paths.

```text
data/
  train_without_obj/HM3D/<scene_floor>/
  train_with_obj/<scene_floor>/
  eval_with_obj/<scene_floor>/
```

Each scene needs `floorplan.png` and `traj_<id>/`, containing `traj_<id>.txt`, `rgb/*.png` and training `depth/*.png`. ObjectGoal trajectories also require `object/object.json`. The cache includes both training sources for controller training; the ObjectGoal selector uses only `train_with_obj`. Edit paths in `config/objectgoal_v3.yaml` if needed.

Download the visual backbones:

```bash
mkdir -p checkpoints
curl -fL --retry 3 -o checkpoints/depth_anything_v2_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
python -c "import clip; clip.load('ViT-B/16', device='cpu', download_root='checkpoints')"
```

Place the experiment's pretrained weights in `checkpoints/`:

| File | Required for |
|---|---|
| `object_localizer.pt` | Frozen category-conditioned goal localization |
| `object_disco.pt` | Frozen object-hypothesis ranking |
| `controller_training.pth` | Frozen controller used to train the selector |
| `controller_navigation.pth` | Controller used during V3 navigation |

These experiment weights are **not included or hosted by this code branch**. Obtain them from the experiment owner before preparing hypotheses; `config/pretrained_weights.json` records their SHA-256 values. Localization and ranking remain frozen in the training workflow below. Depth guidance uses simulator depth.

Prepare the training cache, maps, category hypotheses and validation samples:

```bash
python scripts/prepare_data.py --config config/objectgoal_v3.yaml
python utils/prepare_pointgoal_bridge.py --config config/objectgoal_v3.yaml
CUDA_VISIBLE_DEVICES=0 python utils/prepare_topk_routes.py --config config/objectgoal_v3.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=0 python utils/prepare_topk_routes.py --config config/objectgoal_v3.yaml --device cuda:0 --build-validation
```

`config/training_selection.json.gz` selects 118,143 `without_obj` and 57,750 `with_obj` trajectories for the shared controller cache. `prepare_data.py` checks their lengths and builds a stride-4 LMDB cache; training uses stride-8 windows with 16 waypoints, so keep all raw frames. The ObjectGoal selector uses only `with_obj`: holding out the 40 training buildings in `config/object_validation_buildings.json` leaves 48,000 training trajectories. `--build-validation` creates the selector validation samples; this holdout does not apply to the PointGoal controllers. Evaluation uses only the 1,710 upstream evaluation trajectories in `config/objectgoal_eval_1710.json`, with target categories and positions from the object annotations.

Use fresh LMDB and validation-cache paths. Keep caches writable and rebuild them after changing their inputs or paths.

## Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=2 torchrun --standalone --nproc_per_node=3 \
  utils/train_topk_routes.py --config config/objectgoal_v3.yaml --run-name objectgoal_v3
```

Defaults: 5 epochs, batch size 4 per GPU, 12 accumulation steps, AdamW learning rate `1e-4`, weight decay `0.01`, and cosine decay to `1e-6`. Training varies the active goal and exploration candidate counts. Evaluation uses `runs/objectgoal_v3/planner_best_route_validation.pth`, selected on held-out training buildings.

To retrain the two controllers, prepare maps for both training sources, then train before the selector:

```bash
python utils/prepare_pointgoal_bridge.py --config config/controller_training.yaml
CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=2 torchrun --standalone --nproc_per_node=3 utils/train_pointgoal_bridge.py --config config/controller_training.yaml --run-name controller_training
CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=2 torchrun --standalone --nproc_per_node=3 utils/train_pointgoal_bridge.py --config config/pointgoal_v3.yaml --run-name controller_navigation
cp runs/controller_training/planner_epoch_5_final.pth checkpoints/controller_training.pth
cp runs/controller_navigation/planner_epoch_5_final.pth checkpoints/controller_navigation.pth
```

Keep the controller used for selector training separate from the navigation controller.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 bash scripts/evaluate_objectgoal.sh \
  runs/objectgoal_v3/planner_best_route_validation.pth results/objectgoal_v3
```

The launcher evaluates the fixed 1,710 tasks with 64 goal candidates, 4 exploration candidates and V3 depth guidance. Success uses the same-category object-center region with a 1 m threshold. It writes `summary.json` with SR, SPL and coverage counts. Use a fresh output directory; set `EVAL_DATA_ROOT` or `NAV_CONTROLLER` to override the evaluation data or navigation checkpoint. Configure `__EGL_VENDOR_LIBRARY_FILENAMES` if required by your driver.

```bash
python scripts/score_objectgoal.py --output results/objectgoal_v3
```

## Reproduction checks

The supplied V3 route guidance includes collision, unknown-space, boundary, shape, motion and goal costs.

Keep `sampling_identity_roots` unchanged when relocating data: these are hash namespaces, not paths to open. Retain the `with_obj`/`without_obj` source directory names. They preserve the reference selector sampling and validation choices across machines.

The archived ObjectGoal evaluation silently used zero floor height because its scene-asset link was missing. This release uses the actual floor heights and now fails on missing/invalid `floors.txt`; its corrected evaluation must be rerun in full before reporting new metrics. The historical numbers are not a verified target for this corrected setup. See `config/reproduction_audit.json` for checkpoint hashes and audit scope. The listed pretrained weights are still required separately; full retraining has not been repeated in this audit.
