"""Leakage-safe use of official PASTIS-R patch normalization files."""

import json
from pathlib import Path

import numpy as np


SATELLITES = ("S2", "S1A", "S1D")


def load_official_train_stats(dataset_root, train_folds, channel_counts):
    """Average official per-fold channel statistics over training folds only.

    This mirrors the official PASTIS loader's combination rule while ensuring
    that validation and test folds never contribute to the selected values.
    """
    dataset_root = Path(dataset_root).expanduser().resolve()
    folds = tuple(int(fold) for fold in train_folds)
    if not folds:
        raise ValueError("At least one training fold is required for normalization")
    result = {}
    for satellite in SATELLITES:
        path = dataset_root / "NORM_{}_patch.json".format(satellite)
        with path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
        means = []
        stds = []
        for fold in folds:
            key = "Fold_{}".format(fold)
            if key not in raw:
                raise KeyError("{} is missing {}".format(path, key))
            means.append(raw[key]["mean"])
            stds.append(raw[key]["std"])
        mean = np.asarray(means, dtype=np.float64).mean(axis=0).astype(np.float32)
        std = np.asarray(stds, dtype=np.float64).mean(axis=0).astype(np.float32)
        expected = int(channel_counts[satellite])
        if mean.shape != (expected,) or std.shape != (expected,):
            raise ValueError(
                "{} has dimensions mean={} std={}, expected {}".format(
                    path, mean.shape, std.shape, expected
                )
            )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("Non-finite normalization value in {}".format(path))
        if np.any(std <= 0):
            raise ValueError("Non-positive normalization std in {}".format(path))
        result[satellite] = {"mean": mean, "std": std}
    return result


def normalize_mean_std(raw_mean, raw_std, channel_mean, channel_std):
    """Apply pixel normalization algebraically to parcel mean/std values."""
    mean = (raw_mean - channel_mean[None, :]) / channel_std[None, :]
    std = raw_std / channel_std[None, :]
    return mean.astype(np.float32), std.astype(np.float32)
