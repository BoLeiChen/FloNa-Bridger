# FloNavBridger

PointGoal navigation: dataset preparation, training and V3 evaluation. Run all commands from the repository root.

## Dataset

The prepared Gibson-based dataset contains **67 training and 50 evaluation scene-floor entries**. Full evaluation uses 10 trajectories per test entry (500 episodes). Scene lists are in [training/pointgoal_v3.yaml](training/pointgoal_v3.yaml) and [experiments/alpha_0p7_depth_route_full.yaml](experiments/alpha_0p7_depth_route_full.yaml).

Prepare or symlink the following directories:

```text
datasets_96/
  scenes_117/{train,test}/<scene_floor>/
    floorplan.png
    traj_<id>/                 # trajectory .npy and RGB images
  trav_maps/
    floor_shapes.npy
    <scene>/floor_trav_test_0_modified.png
    <scene_floor>/floor_trav_test_<floor>_modified.png
datasets/scenes_117/{train,test}/<scene_floor>/
  floorplan.png
  traj_<id>/traj_<id>.npy       # reference trajectories for evaluation
iGibson/igibson/data/g_dataset/<scene>/  # Gibson meshes and floors.txt
```

`datasets_96` provides model inputs; `datasets` provides planner floorplans and scoring references. Keep their scene/trajectory IDs consistent and allow write access for generated caches. The loader's required image/trajectory conventions are defined in [model/flona_dataset.py](model/flona_dataset.py) and [model/data_utils.py](model/data_utils.py). **Prepared demonstrations, scene assets and navigation checkpoints are not included; raw Gibson meshes alone are insufficient.** Obtain the prepared data separately before training. The recorded evaluation task IDs are in [experiments/pointgoal_v3_results.json](experiments/pointgoal_v3_results.json).

## Training

### 1. Install dependencies

Use Linux with an NVIDIA GPU; the recorded setup used an RTX 3090, Python 3.8.20, PyTorch 2.4.1 and CUDA 12.4.

```bash
git clone https://github.com/BoLeiChen/FloNavBridger.git
cd FloNavBridger
conda env create -f environment.yaml
conda activate vgcnav
python -m pip install -e .

# Required for closed-loop evaluation; install iGibson's native dependencies too.
git clone https://github.com/StanfordVL/iGibson.git iGibson
git -C iGibson checkout 7d9598278b95ec1039e1f4366c66b45346521e44
python -m pip install -e ./iGibson
```

Follow the pinned iGibson checkout's installation instructions for native dependencies and scene assets. Evaluation requires working NVIDIA OpenGL/EGL and its compatible legacy Gym API.

### 2. Prepare pretrained weights

The policy backbone and the separate metric-depth estimator require different weights:

```bash
mkdir -p checkpoints external
git clone https://github.com/DepthAnything/Depth-Anything-V2.git external/Depth-Anything-V2
git -C external/Depth-Anything-V2 checkout a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
curl -fL --retry 3 -o checkpoints/depth_anything_v2_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
curl -fL --retry 3 -o checkpoints/depth_anything_v2_metric_hypersim_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth
```

### 3. Train

After preparing the data, run:

```bash
python training/train.py --config training/pointgoal_v3.yaml --run_name pointgoal_v3
```

Defaults: 6 epochs, batch size 64, AdamW with learning rate `1e-4`, cosine decay, GPU 0 and W&B disabled. Adjust `gpu_ids`, `batch_size` and data paths in the training YAML if needed. Use a new run name each time. The final EMA checkpoint is saved to `logs/VGCNav/pointgoal_v3/ema_5.pth`.

## Evaluation

Keep all pretrained assets above available, including the policy backbone checkpoint. First run the six-scene smoke configuration with one trajectory per scene:

```bash
python test_vgcnav_gtpos.py \
  --config experiments/alpha_0p7_depth_route_smoke.yaml \
  --ckpt_path logs/VGCNav/pointgoal_v3/ema_5.pth \
  --exp_name pointgoal_v3_smoke --traj_per_scene 1
```

Then run the full 500-episode evaluation and automatic metric aggregation:

```bash
NUM_WORKERS=1 TRAJ_PER_SCENE=10 bash scripts/run_navibridger_eval.sh \
  experiments/alpha_0p7_depth_route_full.yaml \
  logs/VGCNav/pointgoal_v3/ema_5.pth \
  pointgoal_v3_full logs/pointgoal_v3_full.log
```

The launcher defaults to NVIDIA EGL at `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`; set `__EGL_VENDOR_LIBRARY_FILENAMES` if your installation uses a different path. Use a fresh experiment name to avoid mixing outputs. For a baseline comparison, replace the configuration with `experiments/alpha_0p7_baseline_full.yaml` and use a different experiment name with the same checkpoint.

Trajectories are saved to `results/exps/<exp_name>/trajectory/`; standard SR/SPL are appended to `evaluation_summary.csv`. Check that **500 valid episodes** are scored: missing reference files are skipped by the scorer. Standard reporting uses a **0.30 m success threshold and collision budget 10**; online stopping uses 0.25 m. Keep the configuration's collision accounting and recovery settings unchanged for comparisons. To recompute metrics:

```bash
python results/evaluate.py --exp_name pointgoal_v3_full
```
