import os
from pathlib import Path

import torch


def save_atomic(obj, path) -> None:
    """Write to a temp file then rename, so a kill mid-write never corrupts the checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_if_exists(path, map_location="cpu"):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as e:  # truncated file after a hard kill
        print(f"WARNING: could not read {path}: {e}")
        return None
