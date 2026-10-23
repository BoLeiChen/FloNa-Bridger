# Localization training

Run from the repository root in the environment installed in the main README. Download the complete `train_without_obj/HM3D` and `train_with_obj` archives from [FloVerse-1.6k](https://modelscope.cn/datasets/weiqihuang/floverse-1.6k), preserving scene/trajectory names. Install the scene assets and pretrained visual backbone as described in the README. Localization uses additional frames beyond the navigation selection; do not download only the navigation subset.

## Restore reference weights (skip training)

```bash
python scripts/prepare_weights.py
```

The versioned binary parts reconstruct the original checkpoints under `checkpoints/`, with SHA-256 verification. This restores localization weights, both PointGoal controllers and the branch-specific reference path selector. You can run navigation evaluation directly using the main README. An existing different checkpoint is never overwritten. Use a different `--output` directory if needed. Public backbone weights are downloaded separately.

## Reconstruct training data

```bash
python scripts/localization/prepare_localization_data.py
```

The frozen manifests in `config/localization/index/` specify building splits, trajectory/frame identities and pose labels from the reference training run. The script rebases paths, checks scene calibration, reconstructs calibrated floorplan maps from raw FloVerse floorplans and emits `data/localization/image/index.json`. It does not infer a new random split. The image-localization split contains 997,054 / 123,324 / 118,133 frames from 436 / 54 / 55 disjoint buildings (train / validation / test). `navigation_exclusions/` contains scene-name metadata only, used to exclude navigation-test buildings during ImageGoal ranker preparation; no evaluation observations are consumed. Use a fresh output directory. Override `--data-root`, `--scene-assets` and `--output` when relocating data, and update the localization YAML paths accordingly.

## Train image localization and ranking

Build a separate **256×256 RGB** LMDB for each localization split; the navigation 96×96 LMDB is incompatible. LMDB `--map-gib` is an address-space ceiling, not a request to preallocate that much disk space. Allow sufficient storage for the decoded localization dataset.

```bash
python scripts/localization/build_image_lmdb.py --index data/localization/image/index.json --split train --output data/localization/image/lmdb/train --map-gib 300 --workers 8
python scripts/localization/build_image_lmdb.py --index data/localization/image/index.json --split val --output data/localization/image/lmdb/val --map-gib 40 --workers 8
python scripts/localization/build_image_lmdb.py --index data/localization/image/index.json --split test --output data/localization/image/lmdb/test --map-gib 40 --workers 8
CUDA_VISIBLE_DEVICES=0,1,2 python scripts/localization/train_image_diffusion.py --config config/localization/image_diffusion.yaml
python scripts/localization/export_lightning_best.py --run runs/localization/image_diffusion --output checkpoints/localizer.ckpt --overwrite
```

Lightning launches three GPU workers itself. The diffusion protocol uses 30 epochs, batch size 32 per GPU, AdamW (`lr=1e-4`, weight decay `1e-4`, minimum learning rate `1e-5`), BF16, seed 42 and a frozen image backbone. Validation uses the fixed 19,200-frame subset; the checkpoint is selected by validation recall within 1 m. These are the reference training settings; changing GPU count requires adjusting `gpu_ids` in the YAML.

The image ranker's map encoder is initialized from a separately trained refiner. Train both refiner stages before mining ranking candidates:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python scripts/localization/train_image_refiner.py --config config/localization/image_refiner.yaml
python scripts/localization/export_lightning_best.py --run runs/localization/image_refiner --output checkpoints/refiner_stage1.ckpt --overwrite
CUDA_VISIBLE_DEVICES=0,1,2 python scripts/localization/train_image_refiner.py --config config/localization/image_refiner_finetune.yaml --weights-only checkpoints/refiner_stage1.ckpt
python scripts/localization/export_lightning_best.py --run runs/localization/image_refiner_finetune --output checkpoints/refiner_init.ckpt --overwrite
python scripts/localization/configure_localization.py --task image --diffusion checkpoints/localizer.ckpt --refiner checkpoints/refiner_init.ckpt
python scripts/localization/prepare_disco_hm3d.py manifest --config config/localization/image_disco.yaml
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 scripts/localization/prepare_disco_hm3d.py mine --config config/localization/image_disco.yaml --split train
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 scripts/localization/prepare_disco_hm3d.py mine --config config/localization/image_disco.yaml --split val
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 scripts/localization/train_disco_hm3d.py --config config/localization/image_disco.yaml
python scripts/localization/configure_localization.py --task image --diffusion checkpoints/localizer.ckpt --ranker runs/localization/image_disco/best.pt
```

The refiner uses two five-epoch stages (learning rates `1e-4` and `2e-5`, batch size 64 per GPU). It initializes the ranker; it is not enabled during navigation. Ranker preparation selects 120,000 training and 6,000 validation frames, samples 64 diffusion particles with 10 denoising steps and retains five spatial/angular modes. Training candidates include supervised positive/heading alternatives; validation candidates do not. Ranker training uses batch size 16 per GPU, AdamW (`lr=1e-4`, weight decay `1e-4`), cosine decay to `1e-5`, at most 12 epochs and patience 4. Validation selects the checkpoint and fusion weight; the YAML preserves all mining and fusion settings.

The provided `localizer.ckpt`, `refiner_init.ckpt` and `disco.pt` form the reference weight chain. If retraining the diffusion parent, retrain its downstream stages and run `configure_localization.py` to bind the new paths and hashes. Rebuild navigation hypotheses and validation caches, then follow **Training** and **Evaluation** in the main README. Localization validation is distinct from the navigation evaluation split. Full retraining and corrected full navigation evaluation are not claimed by this code-release audit.
