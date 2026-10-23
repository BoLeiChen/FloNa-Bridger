# FloNavBridger — ImageGoal V3 depth guidance

ImageGoal route-selector training and navigation with a frozen PointGoal controller and V3 depth guidance. Run commands from the repository root.

## Setup

Use Linux with NVIDIA GPUs; the supplied training protocol uses three GPUs.

```bash
git clone --depth 1 --branch imagegoal-hm3d-v3 --single-branch https://github.com/BoLeiChen/FloNa-Bridger.git
cd FloNa-Bridger
conda env create -f environment.yaml
conda activate flona-imagegoal
python -m pip install -e .
git clone https://github.com/StanfordVL/iGibson.git iGibson
git -C iGibson checkout 7d9598278b95ec1039e1f4366c66b45346521e44
git -C iGibson submodule update --init --recursive
python -m pip install -e ./iGibson
python -m igibson.utils.assets_utils --download_assets
```

Install iGibson's [native dependencies](https://github.com/StanfordVL/iGibson) and ensure NVIDIA OpenGL/EGL works. Extract `floverse_scene_dataset.tar` from [FloVerse-1.6k](https://modelscope.cn/datasets/weiqihuang/floverse-1.6k) into `iGibson/igibson/data/`, producing `g_dataset/`.

## Data and pretrained models

Our dataset is a fixed HM3D subset reorganized from [FloVerse-1.6k (original data download)](https://modelscope.cn/datasets/weiqihuang/floverse-1.6k). Download `train_without_obj/HM3D` and `train_with_obj`, plus `eval_with_obj`. Extract the per-scene archives, preserve scene/trajectory names and all frames, and place or symlink the extracted scene folders under the following paths.

```text
data/
  train_without_obj/HM3D/<scene_floor>/
  train_with_obj/<scene_floor>/
  eval_with_obj/<scene_floor>/
```

Each scene needs `floorplan.png` and `traj_<id>/`, containing `traj_<id>.txt`, `rgb/*.png` and training `depth/*.png`. Preserve all RGB frames, including the final goal image. Edit paths in `config/imagegoal_v3.yaml` when necessary.

Download the visual backbone:

```bash
mkdir -p checkpoints
curl -fL --retry 3 -o checkpoints/depth_anything_v2_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
```

Download the reference weights as **complete files** from [GitHub Releases](https://github.com/BoLeiChen/FloNa-Bridger/releases/tag/hm3d-reference-weights-v1). The command below saves them to `checkpoints/`:

| File | Required for |
|---|---|
| [`localizer.ckpt`](https://github.com/BoLeiChen/FloNa-Bridger/releases/download/hm3d-reference-weights-v1/localizer.ckpt) | Frozen CF2Loc goal localization |
| [`disco.pt`](https://github.com/BoLeiChen/FloNa-Bridger/releases/download/hm3d-reference-weights-v1/disco.pt) | Frozen goal-hypothesis ranking |
| [`controller_training.pth`](https://github.com/BoLeiChen/FloNa-Bridger/releases/download/hm3d-reference-weights-v1/controller_training.pth) | Frozen controller used to train the selector |
| [`controller_navigation.pth`](https://github.com/BoLeiChen/FloNa-Bridger/releases/download/hm3d-reference-weights-v1/controller_navigation.pth) | Controller used during V3 navigation |
| [`selector.pth`](https://github.com/BoLeiChen/FloNa-Bridger/releases/download/hm3d-reference-weights-v1/imagegoal_selector.pth) | Reference imagegoal path selector for evaluation |

Download localization, ranking, controller and selector weights with SHA-256 verification:

```bash
python scripts/download_weights.py
```

`config/pretrained_weights.json` records the reference SHA-256 hashes. `config/weights.json` lists complete-file URLs, sizes and hashes. The downloader refuses to overwrite different existing weights. The task-specific selector asset is saved as `checkpoints/selector.pth`. For evaluation with the supplied selector, complete setup, download the visual backbone(s), download weights and prepare only `data/eval_with_obj` plus the scene assets; then go directly to **Evaluation**. Training caches are unnecessary for this path. To train localization/ranking models, follow [Localization training](docs/LOCALIZATION.md). The ImageGoal ranker starts from random initialization with no refiner stage or checkpoint; the original public visual backbone remains frozen.


Build the training cache, maps, goal hypotheses and held-out validation samples:

```bash
python scripts/prepare_data.py --config config/imagegoal_v3.yaml
python utils/prepare_pointgoal_bridge.py --config config/imagegoal_v3.yaml
CUDA_VISIBLE_DEVICES=0 python utils/prepare_topk_routes.py --config config/imagegoal_v3.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=0 python utils/prepare_topk_routes.py --config config/imagegoal_v3.yaml --device cuda:0 --build-validation
```

`config/training_selection.json.gz` selects 118,143 `without_obj` and 57,750 `with_obj` trajectories for the shared controller cache. `prepare_data.py` checks their lengths and builds a stride-4 LMDB cache; training uses stride-8 windows with 16 waypoints, so keep all raw frames. For ImageGoal selector training, `config/image_validation_buildings.json` holds out 92 training buildings, leaving 147,843 trajectories; validation samples are built by `--build-validation`. This holdout applies to the selector, not the PointGoal controllers. Evaluation uses only the 1,470 upstream evaluation trajectories in `config/imagegoal_eval_1470.json`; the final RGB frame provides the image goal.

Use fresh LMDB and validation-cache paths. Regenerate caches after changing data/model paths. The PointGoal branch's LMDB can be reused with the same training roots.

## Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=2 torchrun --standalone --nproc_per_node=3 \
  utils/train_topk_routes.py --config config/imagegoal_v3.yaml --run-name imagegoal_v3
```

Defaults: 5 epochs, batch size 24 per GPU, AdamW learning rate `1e-4`, weight decay `0.01`, and cosine decay to `1e-6`. Navigation uses `runs/imagegoal_v3/planner_best_route_validation.pth`, selected on the held-out training buildings.

To retrain the two PointGoal controllers, use the same prepared training cache and visual backbone:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=2 torchrun --standalone --nproc_per_node=3 utils/train_pointgoal_bridge.py --config config/controller_training.yaml --run-name controller_training
CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=2 torchrun --standalone --nproc_per_node=3 utils/train_pointgoal_bridge.py --config config/pointgoal_v3.yaml --run-name controller_navigation
cp runs/controller_training/planner_epoch_5_final.pth checkpoints/controller_training.pth
cp runs/controller_navigation/planner_epoch_5_final.pth checkpoints/controller_navigation.pth
```

Run controller training before selector training if using newly trained controllers. Keep the two controller checkpoints separate.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 bash scripts/evaluate_imagegoal.sh \
  checkpoints/selector.pth results/imagegoal_v3
```

To evaluate your retrained selector, replace `checkpoints/selector.pth` with `runs/imagegoal_v3/planner_best_route_validation.pth`.

The launcher evaluates all 1,470 fixed tasks with Top-32 hypotheses, the selected route-switching settings and V3 depth guidance. It writes `summary.json` with SR, SPL and coverage counts. Use a fresh output directory; set `EVAL_DATA_ROOT` or `NAV_CONTROLLER` to override the evaluation data or navigation checkpoint. Configure `__EGL_VENDOR_LIBRARY_FILENAMES` if needed by your driver.

```bash
python scripts/score_imagegoal.py --output results/imagegoal_v3
```

## Reproduction checks

The supplied V3 route guidance includes collision, unknown-space, boundary, shape, motion and goal costs.

Keep `sampling_identity_roots` unchanged when relocating data: these are hash namespaces, not paths to open. Retain the `with_obj`/`without_obj` source directory names. They preserve the reference selector sampling and validation choices across machines. Rebuild validation caches created before this fix.

The archived ImageGoal evaluation silently used zero floor height because its scene-asset link was missing. This release uses the actual floor heights and now fails on missing/invalid `floors.txt`; its corrected evaluation must be rerun in full before reporting new metrics. The historical numbers are not a verified target for this corrected setup. See `config/reproduction_audit.json` for checkpoint hashes and audit scope. The supplied weights are the reference experiment files; full retraining has not been repeated in this audit.
