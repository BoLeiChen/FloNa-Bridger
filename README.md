# FloNavBridger

PointGoal navigation: dataset preparation, training and V3 evaluation. Run all commands from the repository root.

## Dataset

The prepared Gibson-based dataset contains **67 training and 50 evaluation scene-floor entries**. Full evaluation uses 10 trajectories per test entry (500 episodes). Scene lists are in [training/pointgoal_v3.yaml](training/pointgoal_v3.yaml) and [experiments/alpha_0p7_depth_route_full.yaml](experiments/alpha_0p7_depth_route_full.yaml).

Download the prepared data from [chenbolei/FloNaBridger-Gibson on Hugging Face](https://huggingface.co/datasets/chenbolei/FloNaBridger-Gibson). The complete release contains **118 archives (about 29.65 GB)**. Allow at least **70 GB** for archives and extracted files. The upload is complete; `manifest.json` records every archive hash.

After installing the environment below, run these commands from the code repository root. Download the complete release, then extract and validate it:

```bash
python -m pip install "huggingface_hub==0.36.2"
hf download chenbolei/FloNaBridger-Gibson \
  --repo-type dataset --local-dir ../FloNaBridger-Gibson
python scripts/extract_hf_dataset.py --source ../FloNaBridger-Gibson --destination .
python scripts/validate_dataset.py
```

The extraction script verifies each archive's SHA-256 and creates `data/`. It safely materializes TAR hard links as regular files and publishes `data/` only after extraction succeeds. It refuses to overwrite an existing `data/` directory; use a fresh destination when needed. Training, navigation and scoring all use `data_root: data` in their YAML configurations:

```text
data/
  scenes_117/{train,test}/<scene_floor>/
    floorplan.png             # model input
    planner_floorplan.png     # original planner map; preserve its resolution
    traj_<id>/                # trajectory .npy and RGB images
  trav_maps/
    floor_shapes.npy
    ... traversability maps ...
```

### Dataset construction

This release packages the **existing V3 demonstrations**, rather than regenerating demonstrations from raw Gibson meshes. The split follows the local `fix/depth-route-guidance-v3` experiment (`4ef7184`): 67 training and 50 test **scene-floor** entries, not necessarily disjoint buildings. Exact lists are in the training YAML; do not randomly re-split the data.

For each entry, RGB frames, trajectory arrays and `floorplan.png` come from the prepared `datasets_96` tree; `planner_floorplan.png` comes from the original `datasets` tree. `trav_maps` comes from `datasets_96`. Each `traj_<id>.npy` is a finite `(T, 4)` array containing position `(x, y)` and a **heading point** `(x_h, y_h)` in meters (not a yaw angle); numbered PNGs index its frames. Preserve map coordinates and resolutions. The 500 evaluation arrays match the original scoring references exactly. Training trajectories must come from the prepared tree, not be substituted from the original scoring tree.

The loader constructs windows starting at frame 4 with stride 5, uses four previous frames plus the current frame, and predicts 32 future waypoints. Its training goal is frame `T-4` (`end_slack: 3`); short futures are padded at that goal. Closed-loop evaluation instead targets the final reference point. RGB files are 96×96 and are resized to 98×98 for the observation encoder. Legacy `shortest_paths`, LMDB, index and A* caches are excluded; they are not portable training inputs. First training builds fresh caches and can take substantially longer before the first optimizer step.

To assemble the same layout from existing legacy data **instead of downloading** (use a nonexistent output directory):

```bash
python scripts/prepare_dataset.py --model-root /path/to/datasets_96 \
  --reference-root /path/to/datasets --output data --copy
python scripts/validate_dataset.py
```

Keep both floorplans. Image/index/A* caches are rebuilt on first training, so `data/` must be writable. If using another location, change `data_root` in both training and evaluation configs. The fixed 500 evaluation tasks are listed in [experiments/pointgoal_v3_results.json](experiments/pointgoal_v3_results.json).

**Gibson scene meshes and model weights are not included in the dataset.** Install simulator assets separately under `iGibson/igibson/data/g_dataset/`, and prepare pretrained weights as described below.

## Training

### 1. Install dependencies

Use Linux with an NVIDIA GPU; the recorded setup used an RTX 3090, Python 3.8.20, PyTorch 2.4.1 and CUDA 12.4.

```bash
git clone https://github.com/BoLeiChen/FloNa-Bridger.git
cd FloNa-Bridger
conda env create -f environment.yaml
conda activate vgcnav
python -m pip install -e .

# Required for closed-loop evaluation; install iGibson's native dependencies too.
git clone https://github.com/StanfordVL/iGibson.git iGibson
git -C iGibson checkout 7d9598278b95ec1039e1f4366c66b45346521e44
git -C iGibson submodule update --init --recursive
python -m pip install -e ./iGibson
```

Follow the pinned iGibson checkout's installation instructions for native dependencies and scene assets. Evaluation requires working NVIDIA OpenGL/EGL and Gym 0.26.2 (pinned in the environment). CUDA availability alone does not verify EGL: if the renderer reports zero devices, check that NVIDIA user-space OpenGL/EGL libraries match the loaded kernel driver.

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
MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen python test_vgcnav_gtpos.py \
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

Trajectories are saved to `results/exps/<exp_name>/trajectory/`; standard SR/SPL are appended to `evaluation_summary.csv`. Check that **500 valid episodes** are scored: use `--require-full-benchmark` below to reject missing, extra or invalid episodes rather than silently score a subset. Standard reporting uses a **0.30 m success threshold and collision budget 10**; online stopping uses 0.25 m. Keep the configuration's collision accounting and recovery settings unchanged for comparisons. To recompute metrics:

```bash
python results/evaluate.py --exp_name pointgoal_v3_full \
  --config experiments/alpha_0p7_depth_route_full.yaml --require-full-benchmark
```

The recorded V3 run achieved SR **89.8%** and SPL **0.8641** on these 500 tasks (see the results JSON). The trained navigation checkpoint is not included: the commands above train a new model. Exact numerical reproduction is not guaranteed by the seed alone, and this release does not provide a raw-mesh demonstration-generation pipeline. Training and simulator evaluation require the separate pretrained weights and Gibson meshes described above.
