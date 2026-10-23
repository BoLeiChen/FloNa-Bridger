# Localization training

Run from the repository root in the environment installed in the main README. Download the complete `train_without_obj/HM3D` and `train_with_obj` archives from [FloVerse-1.6k](https://modelscope.cn/datasets/weiqihuang/floverse-1.6k), preserving scene/trajectory names. Install the scene assets and pretrained visual backbone as described in the README. Localization uses additional frames beyond the navigation selection; do not download only the navigation subset.

## Download reference weights (skip training)

```bash
python scripts/download_weights.py
```

The [release](https://github.com/BoLeiChen/FloNa-Bridger/releases/tag/hm3d-reference-weights-v1) provides complete checkpoint files. The downloader verifies SHA-256 and saves localization models, both PointGoal controllers and the branch-specific reference path selector under `checkpoints/`; no assembly is needed. Existing different checkpoints are never overwritten; use a different `--output` directory if needed. Public backbone weights are downloaded separately.

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

Train the ranker from random initialization after training the diffusion localizer. The localizer supplies frozen image features and candidate poses; no refiner checkpoint or refiner-training stage is required. Batch-normalization running statistics are learned for the new map encoder and synchronized across GPUs during distributed training.

```bash
python scripts/localization/configure_localization.py --task image --diffusion checkpoints/localizer.ckpt
python scripts/localization/prepare_disco_hm3d.py manifest --config config/localization/image_disco.yaml
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 scripts/localization/prepare_disco_hm3d.py mine --config config/localization/image_disco.yaml --split train
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 scripts/localization/prepare_disco_hm3d.py mine --config config/localization/image_disco.yaml --split val
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 scripts/localization/train_disco_hm3d.py --config config/localization/image_disco.yaml
python scripts/localization/configure_localization.py --task image --diffusion checkpoints/localizer.ckpt --ranker runs/localization/image_disco/best.pt
```

Ranker preparation selects 120,000 training and 6,000 validation frames, samples 64 diffusion particles with 10 denoising steps and retains five spatial/angular modes. Training candidates include supervised positive/heading alternatives; validation candidates do not. Ranker training uses batch size 16 per GPU, AdamW (`lr=1e-4`, weight decay `1e-4`), cosine decay to `1e-5`, at most 12 epochs and patience 4. Validation selects the checkpoint and fusion weight; the YAML preserves all mining and fusion settings.

## Test localization separately

After checkpoint selection, evaluate the fixed pair on the localization test split:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/localization/evaluate_image_localization.py --config config/localization/image_disco.yaml --ranker runs/localization/image_disco/best.pt --split test --output results/localization/image_test
```

`metrics.json` reports diffusion and reranked localization recall, position/yaw error and split coverage. The ranker's fusion weight and retained-mode count come from its checkpoint; the test set does not select them. This uses the localization ranking protocol (five retained modes in the supplied config), while navigation uses its separate Top-K path-hypothesis protocol. Omit `--ranker` for diffusion-only evaluation. `--limit` is for smoke checks and does not constitute a full test. The complete test command requires the test LMDB constructed above.

The downloadable `disco.pt` remains the historical reference checkpoint, whose map encoder was initialized from a refiner. It already contains all inference weights and does not need that refiner at runtime. The new from-scratch ranker workflow changes this initialization and learns batch-normalization statistics; its final navigation performance has not been measured and must not be equated with the historical reference results. If retraining the diffusion parent, train a new ranker, run `configure_localization.py` to bind both weights, and rebuild navigation hypotheses and validation caches before following the main README.

From-scratch training here refers to the task-specific localization and ranking modules. The original frozen public visual backbone is retained; download it as described in the main README. Full retraining and corrected full navigation evaluation have not been repeated for this release.
