# Reference localization weights

These are the reference experiment checkpoints, stored in parts no larger than 40 MiB. Run `python scripts/prepare_weights.py` from the repository root to restore and verify them. Public visual backbones and PointGoal controllers are separate.

| Restored file | Bytes | SHA-256 |
|---|---:|---|
| `object_localizer.pt` | 27709775 | `3ed32be20dabd9ce5ebe1a677fbe4704a951d023bcabe8f421ff9c22ed93e3cd` |
| `object_disco.pt` | 3243419 | `e0f2650f85a051be53e3faed0818fc1aed8955d9f14d7b7751b5f33fe2860bac` |

Use the localization workflow to train new checkpoints bound to your rebuilt dataset.
