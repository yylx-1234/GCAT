"""Recreate the paper's patch-disjoint PASTIS-R confirmatory manifests."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FOLD_CYCLE = {
    2: {"test": (2,), "dev": (1,), "train": (3, 4, 5)},
    3: {"test": (3,), "dev": (2,), "train": (4, 5, 1)},
    4: {"test": (4,), "dev": (3,), "train": (5, 1, 2)},
    5: {"test": (5,), "dev": (4,), "train": (1, 2, 3)},
}


def choose_dev_groups(frame, seed=42, trials=20000):
    """Keep the source experiment's patch-level search objective and RNG."""
    patches = np.asarray(sorted(frame.patch_id.unique()), dtype=np.int64)
    patch_index = {int(patch): index for index, patch in enumerate(patches)}
    histograms = np.zeros((len(patches), 18), dtype=np.int64)
    for row in frame.itertuples(index=False):
        histograms[patch_index[int(row.patch_id)], int(row.label)] += 1
    n_val = (len(patches) + 1) // 2
    remaining = len(patches) - n_val
    n_cal_a = (remaining + 1) // 2
    target_ratios = np.asarray(
        [n_val, n_cal_a, remaining - n_cal_a], dtype=np.float64
    ) / len(patches)
    totals = histograms.sum(0).astype(np.float64)
    rng = np.random.RandomState(int(seed))
    best, best_score = None, float("inf")
    for _ in range(int(trials)):
        order = rng.permutation(len(patches))
        groups = (order[:n_val], order[n_val:n_val + n_cal_a], order[n_val + n_cal_a:])
        score = 0.0
        for indices, ratio in zip(groups, target_ratios):
            counts = histograms[indices].sum(0)
            target = totals * ratio
            score += float(np.mean(np.abs(counts - target) / np.maximum(target, 1.0)))
            score += 2.0 * abs(float(counts.sum()) / float(totals.sum()) - ratio)
        if score < best_score:
            best, best_score = groups, score
    return {
        name: {int(patches[index]) for index in indices}
        for name, indices in zip(("val", "cal_a", "cal_b"), best)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int, choices=tuple(FOLD_CYCLE))
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest)
    if not {"patch_id", "parcel_id", "label", "fold"}.issubset(frame):
        raise ValueError("Parcel manifest is missing identity/fold columns")
    cycle = FOLD_CYCLE[args.fold]
    train = frame[frame.fold.isin(cycle["train"])]
    dev = frame[frame.fold.isin(cycle["dev"])]
    test = frame[frame.fold.isin(cycle["test"])]
    assignment = choose_dev_groups(dev)
    groups = {"train": train, "test": test}
    groups.update({name: dev[dev.patch_id.isin(patches)] for name, patches in assignment.items()})
    patch_sets = {name: set(group.patch_id) for name, group in groups.items()}
    if any(left & patch_sets[b] for a, left in patch_sets.items() for b in patch_sets if a != b):
        raise RuntimeError("Patch overlap among splits")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, group in groups.items():
        selected = group[["patch_id", "parcel_id", "label"]].sort_values(["patch_id", "parcel_id"]).copy()
        selected["split"] = name
        selected.to_csv(args.output / f"fold{args.fold}_{name}.csv", index=False, lineterminator="\n")
        print(f"{name}: {len(selected)} parcels; {len(patch_sets[name])} patches")


if __name__ == "__main__":
    main()
