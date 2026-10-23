# FloNavBridger — HM3D V3 route

PointGoal training and evaluation with V3 route depth guidance. Run all commands from the repository root.

## Setup

Use Linux and NVIDIA GPUs. The training configuration uses three GPUs.

```bash
git clone --depth 1 --branch pointgoal-hm3d-v3 --single-branch https://github.com/BoLeiChen/FloNa-Bridger.git
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

Our dataset is a fixed HM3D subset reorganized from [FloVerse-1.6k (original data download)](https://modelscope.cn/datasets/weiqihuang/floverse-1.6k). Download `train_without_obj/HM3D` and `train_with_obj`, plus `eval_without_obj/HM3D`. Extract the per-scene archives, preserve scene/trajectory names and all frames, and place or symlink the extracted scene folders under the following paths.

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

`config/training_selection.json.gz` defines our training subset: 118,143 `without_obj` and 57,750 `with_obj` trajectories (175,893 total). Each record identifies the source directory, scene and trajectory; `prepare_data.py` reads only these records and checks trajectory lengths. It builds a stride-4 LMDB cache; training uses stride-8 windows with 16 waypoints. Do not manually subsample the raw frames. Evaluation uses only the 4,084 trajectories listed in `config/evaluation_episodes.json`, from the upstream evaluation split.

Training-cache preparation does not require evaluation trajectories. Add `--check-evaluation` to `prepare_data.py` only when you also want to check their availability.

Use a fresh LMDB output directory and leave caches writable. Map preparation calibrates floorplans against the scene assets; first training builds A* path caches. Data and scene assets are not included in this repository.

## Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=2 torchrun --standalone --nproc_per_node=3 \
  utils/train_pointgoal_bridge.py --config config/pointgoal_v3.yaml --run-name pointgoal_v3
```

Defaults: 5 epochs, batch size 16 per GPU and Adam learning rate `1e-4`. The final checkpoint is `runs/pointgoal_v3/planner_epoch_5_final.pth`. Changing the GPU count changes the training schedule; use three processes for the supplied protocol.

## Evaluation

To evaluate the published reference controller, complete setup and prepare the evaluation data and scene assets, then run:

```bash
python scripts/download_weights.py
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 bash scripts/evaluate.sh checkpoints/controller_navigation.pth results/pointgoal_v3_reference
```

The complete checkpoint is hosted in [GitHub Releases](https://github.com/BoLeiChen/FloNa-Bridger/releases/tag/hm3d-reference-weights-v1); `config/weights.json` pins its size and SHA-256. Training data and LMDB are unnecessary for reference evaluation. To evaluate your own training result:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 bash scripts/evaluate.sh \
  runs/pointgoal_v3/planner_epoch_5_final.pth results/pointgoal_v3
```

The launcher enables `config/depth_v3_route.yaml`, evaluates the fixed 4,084 episodes and writes `results/pointgoal_v3/summary.json`. Use a new run/output directory each time. For another evaluation data location, set `EVAL_DATA_ROOT`. Configure `__EGL_VENDOR_LIBRARY_FILENAMES` if your driver requires a custom EGL path.

To recompute metrics from saved rollouts:

```bash
python scripts/score.py --output results/pointgoal_v3 --scenes-dir data/eval_without_obj/HM3D
```

## Reproduction checks

The supplied V3 route guidance includes collision, unknown-space, boundary, shape, motion and goal costs.

The reference controller is identified in `config/reproduction_audit.json`. With that checkpoint, all 4,084 saved rollouts reproduce the archived metrics, and one fresh simulator rollout matches the archive exactly. Full retraining and a fresh full evaluation have not been repeated in this audit; a newly trained checkpoint need not reproduce the same numbers exactly. Keep the supplied three-GPU training protocol and complete scene assets; missing or invalid `floors.txt` now stops evaluation.
