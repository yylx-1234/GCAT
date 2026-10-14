"""Atomic checkpoint save/load utilities."""

import os
import time
from pathlib import Path

import torch


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    for attempt in range(10):
        try:
            os.replace(str(temporary), str(path))
            break
        except PermissionError:
            if attempt == 9:
                raise
            # Windows antivirus/indexers can briefly hold a just-written file.
            time.sleep(0.05 * (attempt + 1))


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
