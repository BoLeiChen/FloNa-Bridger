# FloNavBridger

PointGoal navigation on Gibson. Run the commands below from the repository root.

## Setup

Use Linux with an NVIDIA GPU.

```bash
git clone --depth 1 --branch main --single-branch https://github.com/BoLeiChen/FloNa-Bridger.git
cd FloNa-Bridger
conda env create -f environment.yaml
conda activate vgcnav
python -m pip install -e .
```

For closed-loop evaluation, install iGibson and its [native dependencies and scene assets](https://github.com/StanfordVL/iGibson):

```bash
git clone https://github.com/StanfordVL/iGibson.git iGibson
git -C iGibson checkout 7d9598278b95ec1039e1f4366c66b45346521e44
git -C iGibson submodule update --init --recursive
python -m pip install -e ./iGibson
python -m igibson.utils.assets_utils --download_assets
```

Place Gibson scene assets under `iGibson/igibson/data/g_dataset/`. Evaluation requires working NVIDIA OpenGL/EGL libraries compatible with the installed driver.

## Data preparation

Download the prepared [Gibson dataset](https://huggingface.co/datasets/chenbolei/FloNaBridger-Gibson), containing 67 training and 50 test scene-floor entries. Allow at least 70 GB for the download and extraction.

```bash
python -m pip install "huggingface_hub==0.36.2"
hf download chenbolei/FloNaBridger-Gibson \
  --repo-type dataset --local-dir ../FloNaBridger-Gibson
python scripts/extract_hf_dataset.py --source ../FloNaBridger-Gibson --destination .
python scripts/validate_dataset.py
```

Extraction creates `data/` and refuses to overwrite an existing directory:

```text
data/
  scenes_117/{train,test}/<scene_floor>/
    floorplan.png
    planner_floorplan.png
    traj_<id>/
  trav_maps/
```

Keep this layout intact. The default configs use `data_root: data`; change it in both training and evaluation configs if needed. The first training run builds caches, so the data directory must be writable. Simulator assets and model weights are downloaded separately.

## Pretrained weights

Prepare both the policy backbone and metric-depth weights:

```bash
mkdir -p checkpoints external
git clone https://github.com/DepthAnything/Depth-Anything-V2.git external/Depth-Anything-V2
git -C external/Depth-Anything-V2 checkout a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
curl -fL --retry 3 -o checkpoints/depth_anything_v2_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
curl -fL --retry 3 -o checkpoints/depth_anything_v2_metric_hypersim_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth
```

## Training

```bash
python training/train.py --config training/pointgoal_v3.yaml --run_name pointgoal_v3
```

This branch trains its navigation checkpoint locally; the HM3D controllers published for the other branches are incompatible with this Gibson model and must not be substituted.

The default configuration trains for 6 epochs with batch size 64. Adjust `gpu_ids` and `batch_size` in the YAML if needed. The final EMA checkpoint is `logs/VGCNav/pointgoal_v3/ema_5.pth`.

## Evaluation

Run the 500-episode evaluation using the trained checkpoint, then verify and aggregate all results:

```bash
NUM_WORKERS=1 TRAJ_PER_SCENE=10 bash scripts/run_navibridger_eval.sh \
  experiments/alpha_0p7_depth_route_full.yaml \
  logs/VGCNav/pointgoal_v3/ema_5.pth \
  pointgoal_v3_full logs/pointgoal_v3_full.log
python results/evaluate.py --exp_name pointgoal_v3_full \
  --config experiments/alpha_0p7_depth_route_full.yaml --require-full-benchmark
```

Use a new run/experiment name for each run. Trajectories are saved under `results/exps/<exp_name>/trajectory/`, and metrics are written to `evaluation_summary.csv`. The launcher uses `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`; override `__EGL_VENDOR_LIBRARY_FILENAMES` if your NVIDIA EGL configuration is elsewhere.
