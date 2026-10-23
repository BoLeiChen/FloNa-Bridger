# Object localization inference dependency

This directory contains the inference dependency closure from the frozen `object_disco_quality8_20260930` source snapshot used by ObjectGoal navigation. The upstream MIT license is retained.

The only model-loading adaptation threads `backbone_paths` through `ObjectDisCoLocalizer` to the existing `Localizer` API so that the frozen CLIP checkpoint can be relocated. Learned parameters, sampling, category normalization and ranking are unchanged. Checkpoints are external and checked against the hashes in the navigation configuration.

Line endings and trailing whitespace are normalized; upstream source hashes are retained in `UPSTREAM_MANIFEST.sha256`.
