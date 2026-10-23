# Reference navigation weights

Restore with `python scripts/prepare_weights.py` from the repository root. Every part and restored checkpoint is checked against SHA-256. Files are unchanged from the audited experiment checkpoints.

| File | Role | Bytes | SHA-256 |
|---|---|---:|---|
| `controller_training.pth` | Frozen controller for selector training | 142667445 | `151047a8393506f71834385823c349ff3a573f2e7a996ff756584f9712ce3e31` |
| `controller_navigation.pth` | V3 navigation controller | 142667445 | `764427dfe3fb163cf42fc2fd7e2e6fec83fd3253b86784a1641158ce9a5caf18` |
| `selector.pth` | objectgoal validation-selected path selector | 148277525 | `38c22052763bbaf54d81197ddc8248d451918bf3f82d10161f6791af712dcdf8` |

The two controller files are identical across the ImageGoal and ObjectGoal branches. Each branch provides its own selector. See the main README for data preparation and evaluation commands.
