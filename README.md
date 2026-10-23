# FloNavBridger — HM3D V3 route

PointGoal training and evaluation with V3 route depth guidance. Run all commands from the repository root.

## Setup

Use Linux and NVIDIA GPUs. The training configuration uses three GPUs.

```bash
git clone --branch pointgoal-hm3d-v3 --single-branch https://github.com/BoLeiChen/FloNa-Bridger.git
cd FloNa-Bridger
conda env create -f environment.yaml
conda activate flona-hm3d
python -m pip install -e .

git clone https://github.com/StanfordVL/iGibson.git iGibson
git -C iGibson checkout 7d9598278b95ec1039e1f4366c66b45346521e44
git -C iGibson submodule update --init --recursive
python -m pip install -e ./iGibson
python -m igibson.utils.assets_utils --download_assets
```

Install iGibson's [native dependencies](https://github.com/StanfordVL/iGibson) and ensure NVIDIA OpenGL/EGL works. Download and extract `floverse_scene_dataset.tar` from [FloVerse-1.6k](https://modelscope.cn/datasets/weiqihuang/floverse-1.6k) into `iGibson/igibson/data/`, producing `g_dataset/`.

Download the visual-backbone weights:

```bash
mkdir -p checkpoints
curl -fL --retry 3 -o checkpoints/depth_anything_v2_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
```

Depth guidance uses the simulator's metric depth; no separate depth-estimation checkpoint is required.

## Data preparation

Download the training and PointGoal evaluation data from [FloVerse-1.6k](https://modelscope.cn/datasets/weiqihuang/floverse-1.6k), extract the per-scene archives, and arrange or link the directories as follows. This branch uses HM3D data; the Gibson dataset used by `main` is not compatible.

```text
data/
  train_without_obj/HM3D/<scene_floor>/
  train_with_obj/<scene_floor>/
  eval_without_obj/HM3D/<scene_floor>/
```

Each scene contains `floorplan.png` and `traj_<id>/`. Training trajectories require `traj_<id>.txt`, `rgb/*.png` and `depth/*.png`; keep `object.json` where present. If using different locations, edit `data_folder`, `eval_data_folder`, `lmdb_path`, `map_cache`, `prior_cache` and `scene_assets` in `config/pointgoal_v3.yaml`.

Build the training cache and calibrated maps:

```bash
python scripts/prepare_data.py --config config/pointgoal_v3.yaml
python utils/prepare_pointgoal_bridge.py --config config/pointgoal_v3.yaml
```

Preparation uses the included training selection and checks the fixed evaluation references. Use a fresh LMDB output directory and leave the cache directories writable. First training also builds A* path caches. Data and scene assets are not included in this code repository.

## Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=2 torchrun --standalone --nproc_per_node=3 \
  utils/train_pointgoal_bridge.py --config config/pointgoal_v3.yaml --run-name pointgoal_v3
```

Defaults: 5 epochs, batch size 16 per GPU and Adam learning rate `1e-4`. The final checkpoint is `runs/pointgoal_v3/planner_epoch_5_final.pth`. Changing the GPU count changes the training schedule; use three processes for the supplied protocol.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 bash scripts/evaluate.sh \
  runs/pointgoal_v3/planner_epoch_5_final.pth results/pointgoal_v3
```

The launcher enables `config/depth_v3_route.yaml`, evaluates the fixed 4,084 episodes and writes `results/pointgoal_v3/summary.json`. Use a new run/output directory each time. For another evaluation data location, set `EVAL_DATA_ROOT`. Configure `__EGL_VENDOR_LIBRARY_FILENAMES` if your driver requires a custom EGL path.

To recompute metrics from saved rollouts:

```bash
python scripts/score.py --output results/pointgoal_v3 --scenes-dir data/eval_without_obj/HM3D
```
