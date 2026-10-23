"""Read an ImageGoal observation without recreating it through robot physics."""
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image


def load_dataset_goal_rgb(trajectory_dir, pose_count):
    if pose_count < 1:
        raise ValueError("ImageGoal requires a nonempty trajectory")
    path = Path(trajectory_dir) / "rgb" / ("rgb_%d.png" % (pose_count - 1))
    # Never silently substitute a rendered frame or a different frame number.
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy(), str(path)


def goal_localization_seed(seed, scene_id, trajectory_id):
    payload = (str(int(seed)) + "\0" + scene_id + "\0" + trajectory_id).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
