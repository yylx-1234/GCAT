"""Auditable train-fold-only weights for weighted cross entropy."""

import numpy as np
import torch
from torch import nn


def class_counts_from_rows(rows, num_classes=18):
    counts = np.bincount(
        np.asarray([int(row["label"]) for row in rows], dtype=np.int64),
        minlength=int(num_classes),
    )
    if counts.shape[0] != int(num_classes) or np.any(counts == 0):
        raise ValueError("Every class must occur in the training folds; counts={}".format(counts))
    return counts.astype(np.int64)


def _mean_one_capped(raw, maximum):
    """Scale positive weights to mean one while respecting an upper cap."""
    maximum = float(maximum)
    if maximum < 1.0:
        raise ValueError("max_class_weight must be at least 1.0")
    low, high = 0.0, 1.0
    while np.minimum(raw * high, maximum).mean() < 1.0:
        high *= 2.0
    for _ in range(100):
        middle = (low + high) / 2.0
        if np.minimum(raw * middle, maximum).mean() < 1.0:
            low = middle
        else:
            high = middle
    return np.minimum(raw * high, maximum)


def compute_class_weights(counts, max_class_weight=5.0):
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    classes = float(counts.size)
    raw = total / (classes * counts)
    normalized = raw / raw.mean()
    clipped = _mean_one_capped(raw, max_class_weight)
    return raw, normalized, clipped


def build_weighted_cross_entropy(weights, device):
    tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=tensor)
