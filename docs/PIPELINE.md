# ImageGoal training and evaluation pipeline

Run from the repository root after environment and scene-asset setup in the README. Choose one workflow below. Public visual backbones are required by the original architecture; “from scratch” means training task-specific modules, not discarding those frozen backbones.

## A. Evaluate the released reference models

Prepare only `data/eval_with_obj/<scene_floor>/` and `iGibson/igibson/data/g_dataset/`, download the public backbones, then run:

```bash
python scripts/download_weights.py
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 bash scripts/evaluate_imagegoal.sh checkpoints/selector.pth results/imagegoal_reference
```

The evaluator computes floorplan and goal-hypothesis caches on demand. It does not need navigation-training LMDB, localization-training LMDB or selector-validation samples. Use a fresh result directory. The selector contains its frozen training controller; `controller_navigation.pth` is the separate V3 execution controller. Do not mix Gibson checkpoints from `main` with these HM3D models.

## B. Train only the path selector with reference parents

Download the public backbones and reference weights, and prepare the full extracted training roots as described in the README. Then execute:

```bash
python scripts/prepare_data.py --config config/imagegoal_v3.yaml
python utils/prepare_pointgoal_bridge.py --config config/controller_training.yaml
CUDA_VISIBLE_DEVICES=0 python utils/prepare_topk_routes.py --config config/imagegoal_v3.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=0 python utils/prepare_topk_routes.py --config config/imagegoal_v3.yaml --device cuda:0 --build-validation
CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=2 torchrun --standalone --nproc_per_node=3 utils/train_topk_routes.py --config config/imagegoal_v3.yaml --run-name imagegoal_v3
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 bash scripts/evaluate_imagegoal.sh runs/imagegoal_v3/planner_best_route_validation.pth results/imagegoal_trained
```

The last command also needs evaluation data. Training-cache construction does not require downloading evaluation trajectories. Both training roots are included in the shared 96×96 navigation LMDB; the ImageGoal selector uses the two training sources. Validation samples come from reserved training buildings and select the path-selector checkpoint.

## C. Train all task-specific stages

Start with a fresh checkout and fresh run/cache directories and the original raw FloVerse data. Download only the public backbones; reference task checkpoints are not a prerequisite. Follow this order:

| Stage | Data | Required input weights | Output / next consumer |
|---|---|---|---|
| 1. Navigation cache and maps | Frozen trajectory selection from both training roots | None | 96×96 navigation LMDB and calibrated maps |
| 2. PointGoal controllers | Stage 1 navigation data | Public DINOv2 backbone | `controller_training.pth` for selector features; `controller_navigation.pth` for execution |
| 3. Goal localization | Full localization frame manifest, separate building split | Public DINOv2 backbone | Frozen diffusion localizer |
| 4. Goal-hypothesis ranking | Candidates mined from localization train/validation splits | Stage 3 localizer | Frozen goal ranker; random map-encoder initialization, no refiner |
| 5. Path hypotheses and selector-validation cache | Stage 1 training roots + selector building split | Matching stage 3–4 pair | Candidate routes and fixed validation samples |
| 6. Path selector | Stage 1 + stage 5 caches | Stage 2 training controller; stage 3–4 remain frozen | Validation-selected `planner_best_route_validation.pth` |
| 7. Navigation evaluation | Fixed upstream navigation evaluation episodes | Stage 6 selector + stage 2 navigation controller + matching stage 3–4 pair | SR/SPL and coverage reports |

First prepare the navigation LMDB and maps using the first two commands in workflow B, then train both controllers:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=2 torchrun --standalone --nproc_per_node=3 utils/train_pointgoal_bridge.py --config config/controller_training.yaml --run-name controller_training
CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=2 torchrun --standalone --nproc_per_node=3 utils/train_pointgoal_bridge.py --config config/pointgoal_v3.yaml --run-name controller_navigation
cp runs/controller_training/planner_epoch_5_final.pth checkpoints/controller_training.pth
cp runs/controller_navigation/planner_epoch_5_final.pth checkpoints/controller_navigation.pth
```

Next follow [Localization training](LOCALIZATION.md), starting at **Reconstruct training data** and skipping the reference-weight download. Complete diffusion and ranker training and the final `configure_localization.py` command before generating stage 5 caches. Finally run workflow B from `prepare_topk_routes.py` onward. Do not rebuild the already prepared navigation LMDB.

## Keep each stage's artifacts together

- Navigation LMDB uses 96×96 images; ImageGoal localization uses a separate 256×256 RGB LMDB. Object-only localization does not read the image RGB LMDB.
- Localization splits and selector-validation splits serve different stages and are not interchangeable. Navigation evaluation episodes are never used to build training labels or select checkpoints.
- A ranker is tied to its diffusion checkpoint. `configure_localization.py` binds the newly trained pair's paths and SHA-256 hashes to navigation. Rebuild hypothesis and validation caches after changing that pair or data paths; choose new output paths rather than retaining stale caches.
- The evaluation selector embeds the localization identities used in its training. Relocating identical files is supported; replacing them with a different localizer/ranker requires rebuilding caches and training a matching selector. Evaluators retain checkpoint hashes when rebasing paths so mismatched parents fail instead of silently changing the experiment.
- For localization construction, preserve `train_with_obj` / `train_without_obj` in the resolved source paths, including symlink targets; the builders reject inputs that cannot be identified as training sources.
- Reference files and self-trained files are separate workflows. Do not run `download_weights.py` over self-trained checkpoints or combine a reference selector with newly trained localization parents.
- Standalone localization recall and closed-loop navigation SR/SPL are different tests. Full retraining and corrected full navigation evaluation have not been repeated; see the README's reproduction notes.
