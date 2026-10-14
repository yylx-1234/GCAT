"""Recreate S2 and log1p-SAR normalization from 2018 Train parcels only.

Input NPZ parcels must already use the published Brandenburg parcel-cache
schema. This script never opens Validation, Calibration or 2019 Test.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--train-split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.train_split.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1769 or any("2019" in str(row.get("split", "")) for row in rows):
        raise ValueError("Expected exactly the official 1769-parcel 2018 Train split")
    accum = {name: {"n": 0, "sum": np.zeros(dim), "sumsq": np.zeros(dim),
                    "min": np.full(dim, np.inf), "max": np.full(dim, -np.inf)}
             for name, dim in (("s2_features", 20), ("s1a_features", 4), ("s1d_features", 4))}
    sar_values = {"s1a": [], "s1d": []}
    for row in rows:
        fid = int(row["fid"])
        path = args.data_root / "cache/parcel/train2018" / f"parcel_{fid}.npz"
        with np.load(path, allow_pickle=False) as archive:
            if int(archive["year"]) != 2018 or int(archive["fid"]) != fid:
                raise ValueError(f"2018 Train identity mismatch: {path}")
            for name, item in accum.items():
                raw = np.asarray(archive[name], dtype=np.float64)
                if raw.ndim != 2 or raw.shape[1] != item["sum"].size or not np.isfinite(raw).all():
                    raise ValueError(f"Invalid {name}: {path}")
                item["n"] += len(raw)
                item["sum"] += raw.sum(axis=0)
                item["sumsq"] += np.square(raw).sum(axis=0)
                item["min"] = np.minimum(item["min"], raw.min(axis=0))
                item["max"] = np.maximum(item["max"], raw.max(axis=0))
                if name.startswith("s1"):
                    if np.any(raw < 0):
                        raise ValueError(f"Negative SAR linear-power statistic: {path}")
                    sar_values[name[:3]].append(np.log1p(raw))
    s2 = accum["s2_features"]
    s2_mean = s2["sum"] / s2["n"]
    s2_std = np.sqrt(np.maximum(s2["sumsq"] / s2["n"] - np.square(s2_mean), 0))
    if np.any(s2_std <= 0):
        raise FloatingPointError("Non-positive S2 standard deviation")
    s2_payload = {
        "status": "passed", "feature_space": "raw_parcel_mean_std",
        "normalization_source": "2018 spatial train split only",
        "source_year": 2018, "source_split": "train", "source_parcel_count": len(rows),
        "ddof": 0,
        "statistics": {"s2_features": {"mean": s2_mean.tolist(), "std": s2_std.tolist()}},
    }
    s1_payload = {
        "status": "passed", "source": "2018 Train only", "source_parcel_count": len(rows),
        "source_manifest_sha256": sha256(args.train_split),
        "transform": "log1p(parcel-level linear-power statistic)",
        "external_2019_used": False, "ddof": 0, "statistics": {},
    }
    for orbit in ("s1a", "s1d"):
        values = np.concatenate(sar_values[orbit], axis=0)
        s1_payload["statistics"][orbit] = {
            "mean": values.mean(axis=0, dtype=np.float64).tolist(),
            "std": values.std(axis=0, dtype=np.float64, ddof=0).tolist(),
            "count_timesteps": len(values),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "brandenburg_train2018_stats.json").write_text(
        json.dumps(s2_payload, indent=2), encoding="utf-8"
    )
    (args.output / "brandenburg_s1_log1p_stats.json").write_text(
        json.dumps(s1_payload, indent=2), encoding="utf-8"
    )
    print("Train-only normalization written; no 2019 data opened")


if __name__ == "__main__":
    main()
