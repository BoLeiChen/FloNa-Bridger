# FloNavBridger — ImageGoal V3 depth guidance

ImageGoal route-selector training and navigation with a frozen PointGoal controller and V3 depth guidance. Run commands from the repository root.

## Setup

Use Linux with NVIDIA GPUs; the supplied training protocol uses three GPUs.

```bash
git clone --branch imagegoal-hm3d-v3 --single-branch https://github.com/BoLeiChen/FloNa-Bridger.git
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

Download and extract the HM3D training and ImageGoal evaluation trajectories from [FloVerse-1.6k](https://modelscope.cn/datasets/weiqihuang/floverse-1.6k). Arrange or link them as follows:

```text
data/
  train_without_obj/HM3D/<scene_floor>/
  train_with_obj/<scene_floor>/
  eval_with_obj/<scene_floor>/
```

Each scene needs `floorplan.png` and `traj_<id>/`, containing `traj_<id>.txt`, `rgb/*.png` and training `depth/*.png`. Preserve all RGB frames, including the final goal image. This branch uses HM3D, not the Gibson dataset linked on `main`. Edit paths in `config/imagegoal_v3.yaml` when necessary.

Download the visual backbone:

```bash
mkdir -p checkpoints
curl -fL --retry 3 -o checkpoints/depth_anything_v2_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
```

Place the experiment's pretrained weights in `checkpoints/`:

| File | Required for |
|---|---|
| `localizer.ckpt` | Frozen CF2Loc goal localization |
| `disco.pt` | Frozen goal-hypothesis ranking |
| `controller_training.pth` | Frozen controller used to train the selector |
| `controller_navigation.pth` | Controller used during V3 navigation |

These experiment weights are **not included or hosted by this code branch**. Obtain them from the experiment owner before preparing hypotheses; `config/pretrained_weights.json` identifies the exact files by SHA-256. The localization and ranking models remain frozen in the training workflow below. Simulator depth supplies guidance, so no additional depth-estimation model is needed.

Build the training cache, maps, goal hypotheses and held-out validation samples:

```bash
python scripts/prepare_data.py --config config/imagegoal_v3.yaml
python utils/prepare_pointgoal_bridge.py --config config/imagegoal_v3.yaml
CUDA_VISIBLE_DEVICES=0 python utils/prepare_topk_routes.py --config config/imagegoal_v3.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=0 python utils/prepare_topk_routes.py --config config/imagegoal_v3.yaml --device cuda:0 --build-validation
```

Use fresh LMDB and validation-cache paths. The included selection fixes training trajectories and held-out buildings. Caches are tied to data and model paths; regenerate them after relocation. A cache prepared on the `pointgoal-hm3d-v3` branch can be reused with the same training roots.

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
  runs/imagegoal_v3/planner_best_route_validation.pth results/imagegoal_v3
```

The launcher evaluates all 1,470 fixed tasks with Top-32 hypotheses, the selected route-switching settings and V3 depth guidance. It writes `summary.json` with SR, SPL and coverage counts. Use a fresh output directory; set `EVAL_DATA_ROOT` or `NAV_CONTROLLER` to override the evaluation data or navigation checkpoint. Configure `__EGL_VENDOR_LIBRARY_FILENAMES` if needed by your driver.

```bash
python scripts/score_imagegoal.py --output results/imagegoal_v3
```
