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

## Construct object-localization labels

Object-only training needs category annotations and floorplans; it does not require building the image-localization RGB LMDB. The first command below reads `object/object.json` from the raw `train_with_obj` trajectories selected by the navigation manifest. It groups object instances by category and floor, normalizes category names and uses the annotated `agent_position` as the positive observer pose, not the object's center. The second command retains these labels and attaches the full image index used by the reference multimodal data format.

```bash
export PYTHONPATH="$PWD/third_party/cf2loc_object${PYTHONPATH:+:$PYTHONPATH}"
python -m cf2loc_mm.prepare --source-index data/localization/image/index.json --object-manifest data/localization/image/navigation_train_manifest.json --scene-assets iGibson/igibson/data/g_dataset --reserved-buildings config/localization/reserved_buildings.txt --output data/localization/object_parent
python -m cf2loc_mm.prepare_full --source data/localization/object_parent --output data/localization/object
```

Keep both output directories: metadata may reference maps in the parent directory. The reference object split contains 3,116 / 278 / 522 category-floor queries (train / validation / test); the training queries contain 8,102 distinct positive observer poses. Existing image-building assignments and the supplied reserved-building list determine the split; added buildings use the original deterministic assignment. No navigation evaluation labels are used for optimization.

## Train object localization and ranking

Download `ViT-B-16.pt` as described in the main README. Keep the `PYTHONPATH` export above in the training shell.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 -m cf2loc_mm.train --config config/localization/object_diffusion.yaml --run runs/localization/object_diffusion
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 -m cf2loc_mm.train_object_disco prepare --diffusion runs/localization/object_diffusion/best_object.pt --cache data/localization/object_candidates --split train --variants 8
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 -m cf2loc_mm.train_object_disco prepare --diffusion runs/localization/object_diffusion/best_object.pt --cache data/localization/object_candidates --split val
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 -m cf2loc_mm.train_object_disco train --diffusion runs/localization/object_diffusion/best_object.pt --cache data/localization/object_candidates --output runs/localization/object_disco --epochs 30
python scripts/localization/configure_localization.py --task object --diffusion runs/localization/object_diffusion/best_object.pt --ranker runs/localization/object_disco/best.pt
```

Diffusion training uses a frozen CLIP ViT-B/16 encoder, 30 epochs, batch size 16 per GPU, AdamW (`lr=1e-4`, weight decay `0.01`) and seed 42. Full-coverage sampling visits all 8,102 positive poses per epoch. Validation selects `best_object.pt` using the mean of Top-1 and Top-16 recall within 1 m. Ranking preparation produces eight independent candidate sets per training query and one per validation/test query, each from 64 sampled particles. DisCo trains for 30 epochs with batch size 8 per GPU, AdamW (`lr=1e-4`, weight decay `0.01`) and cosine decay. Validation selects its checkpoint and score-fusion weight.

Evaluate localization on its held-out test queries only after checkpoint selection:

```bash
CUDA_VISIBLE_DEVICES=0 python -m cf2loc_mm.evaluate --checkpoint runs/localization/object_diffusion/best_object.pt --split test --output results/localization/object_diffusion
CUDA_VISIBLE_DEVICES=0 python -m cf2loc_mm.train_object_disco prepare --diffusion runs/localization/object_diffusion/best_object.pt --cache data/localization/object_candidates --split test
CUDA_VISIBLE_DEVICES=0 python -m cf2loc_mm.train_object_disco test --diffusion runs/localization/object_diffusion/best_object.pt --cache data/localization/object_candidates --output runs/localization/object_disco
```

For ranker training, use the locally trained diffusion checkpoint: its stored data fingerprint must match the locally constructed dataset. The packaged reference checkpoints are directly usable for navigation inference; their original training-data paths are not a portable training dataset. After retraining, rebuild navigation hypotheses and validation caches and follow **Training** and **Evaluation** in the main README. Localization recall and navigation SR/SPL are separate evaluations. Full retraining and corrected full navigation evaluation are not claimed by this code-release audit.
