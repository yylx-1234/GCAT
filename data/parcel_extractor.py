"""Vectorized parcel-level temporal mean/std extraction."""

from collections import Counter

import numpy as np


def _majority(values):
    labels, counts = np.unique(values, return_counts=True)
    maximum = counts.max()
    # np.unique is sorted, so ties have a deterministic and auditable outcome.
    winner = labels[np.flatnonzero(counts == maximum)[0]]
    return int(winner), int(labels.size), int(np.sum(counts == maximum) > 1)


def _temporal_features(series, mask):
    """Compute [mean channels, std channels] without a pixel Python loop."""
    pixels = np.asarray(series[:, :, mask], dtype=np.float32)
    raw_mean = pixels.mean(axis=2, dtype=np.float32)
    raw_std = pixels.std(axis=2, dtype=np.float32)
    return np.concatenate((raw_mean, raw_std), axis=1).astype(np.float32, copy=False)


def extract_patch_parcels(
    patch,
    source_class_ids=tuple(range(1, 19)),
    background_label=0,
    void_label=19,
):
    """Extract all valid instances in one patch.

    Loops are per parcel only; all pixel reductions are vectorized NumPy
    operations over every pixel in the parcel.
    """
    source_class_ids = set(int(value) for value in source_class_ids)
    invalid_labels = {int(background_label), int(void_label)}
    instance_ids = np.unique(patch.instances)
    instance_ids = instance_ids[instance_ids != 0]

    rows = []
    features = {"s2": [], "s1a": [], "s1d": []}
    discarded = Counter()
    inconsistent_labels = 0
    label_ties = 0
    inconsistent_parcel_ids = 0

    for instance_id in instance_ids.tolist():
        mask = patch.instances == instance_id
        pixel_count = int(mask.sum())
        if pixel_count == 0:
            discarded["empty_instance"] += 1
            continue

        source_label, label_count, tied = _majority(patch.semantic[mask])
        if label_count > 1:
            inconsistent_labels += 1
        label_ties += tied
        if source_label in invalid_labels:
            discarded["background_or_void"] += 1
            continue
        if source_label not in source_class_ids:
            discarded["unknown_class"] += 1
            continue

        parcel_id, parcel_id_count, _ = _majority(patch.parcel_ids[mask])
        if parcel_id_count > 1:
            inconsistent_parcel_ids += 1
        if parcel_id == 0:
            discarded["missing_parcel_id"] += 1
            continue

        rows.append(
            (parcel_id, int(instance_id), source_label - 1, source_label, pixel_count)
        )
        features["s2"].append(_temporal_features(patch.s2, mask))
        features["s1a"].append(_temporal_features(patch.s1a, mask))
        features["s1d"].append(_temporal_features(patch.s1d, mask))

    count = len(rows)
    s2_dim = patch.s2.shape[1] * 2
    s1a_dim = patch.s1a.shape[1] * 2
    s1d_dim = patch.s1d.shape[1] * 2
    result = {
        "parcel_ids": np.asarray([row[0] for row in rows], dtype=np.int64),
        "instance_ids": np.asarray([row[1] for row in rows], dtype=np.int64),
        "labels": np.asarray([row[2] for row in rows], dtype=np.int64),
        "source_labels": np.asarray([row[3] for row in rows], dtype=np.int16),
        "pixel_counts": np.asarray([row[4] for row in rows], dtype=np.int32),
        "s2": np.stack(features["s2"]).astype(np.float32)
        if count
        else np.empty((0, patch.s2.shape[0], s2_dim), dtype=np.float32),
        "s1a": np.stack(features["s1a"]).astype(np.float32)
        if count
        else np.empty((0, patch.s1a.shape[0], s1a_dim), dtype=np.float32),
        "s1d": np.stack(features["s1d"]).astype(np.float32)
        if count
        else np.empty((0, patch.s1d.shape[0], s1d_dim), dtype=np.float32),
    }
    stats = {
        "candidate_parcels": int(instance_ids.size),
        "valid_parcels": count,
        "discarded_parcels": int(sum(discarded.values())),
        "discard_reasons": dict(discarded),
        "inconsistent_semantic_parcels": inconsistent_labels,
        "majority_label_ties": label_ties,
        "inconsistent_parcel_id_instances": inconsistent_parcel_ids,
    }
    return result, stats
