# Reference localization weights

These are the reference experiment checkpoints, stored in parts no larger than 40 MiB. Run `python scripts/prepare_weights.py` from the repository root to restore and verify them. Public visual backbones and PointGoal controllers are separate.

| Restored file | Bytes | SHA-256 |
|---|---:|---|
| `localizer.ckpt` | 110598104 | `159b032fd83d0b41e7dac9fd4bd986d347e8014ba461d52a05d144b0f91c2155` |
| `disco.pt` | 5678786 | `9a931d509398f9f22ae768097f45aa1f189264db72b44c9f91b91391dd71090e` |
| `refiner_init.ckpt` | 108627756 | `f64a2b3ef664529437054f8e65f0c5bb715a26f0c09aa846654f296442518314` |

ImageGoal refiner weights initialize ranker training only; the navigation refiner remains disabled.
