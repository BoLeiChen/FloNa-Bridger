# Object localization localization dependency

This directory contains the localization dependency closure from the frozen `object_disco_quality8_20260930` source snapshot used by ObjectGoal navigation. The upstream MIT license is retained.

The only model-loading adaptation threads `backbone_paths` through `ObjectDisCoLocalizer` to the existing `Localizer` API so that the frozen CLIP checkpoint can be relocated. Learned parameters, sampling, category normalization and ranking are unchanged. Checkpoints are downloaded from GitHub Releases and checked against the hashes in the navigation configuration.

Line endings and trailing whitespace are normalized; upstream source hashes are retained in `UPSTREAM_MANIFEST.sha256`.


Training and dataset construction dependencies are now included. See [the localization workflow](../../docs/LOCALIZATION.md). Reference localization checkpoints are downloaded using `scripts/download_weights.py`; public backbone weights remain external. The training entry points are adapted for repository-local imports and paths.
