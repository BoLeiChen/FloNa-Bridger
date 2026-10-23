# CF2Loc-HM3D localization snapshot

This directory vendors the minimal localization dependency closure used by FloVerse's
two-stage CF2Loc localization adapter. It was copied on 2026-09-09 from the server
directory the original CF2Loc-HM3D checkout, which was not a Git repository and therefore had
no source commit to pin.

Two mechanical changes were made to the copied model code: line endings and trailing whitespace were normalized and imports beginning
with `DisCo_model` were converted to package-relative imports. This prevents this
snapshot from resolving modules from a separately installed or subsequently edited
CF2Loc checkout.

The original upstream file digests are recorded in `UPSTREAM_MANIFEST.sha256`.
The MIT and bundled Apache-2.0 license texts are preserved in `LICENSE` and `LICENSES`.
Reference checkpoints are provided as versioned binary parts. Their
expected SHA-256 digests are pinned in `config/imagegoal_v3.yaml` and verified
before deserialization.


Training and dataset construction dependencies are now included. See [the localization workflow](../../docs/LOCALIZATION.md). Reference localization checkpoints are restored using `scripts/prepare_weights.py`; public backbone weights remain external. The training entry points are adapted for repository-local imports and paths.
