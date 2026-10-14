"""Build a resumable parcel-level mean/std cache, one file per patch."""

import argparse
import csv
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.parcel_extractor import extract_patch_parcels
from data.pastis_r import PastisRReader
from utils.config import load_config, resolve_project_path
from utils.doy import dates_to_doy


LABEL_NAMES = {
    0: "Meadow",
    1: "Soft winter wheat",
    2: "Corn",
    3: "Winter barley",
    4: "Winter rapeseed",
    5: "Spring barley",
    6: "Sunflower",
    7: "Grapevine",
    8: "Beet",
    9: "Winter triticale",
    10: "Winter durum wheat",
    11: "Fruits, vegetables, flowers",
    12: "Potatoes",
    13: "Leguminous fodder",
    14: "Soybeans",
    15: "Orchard",
    16: "Mixed cereal",
    17: "Sorghum",
}

MANIFEST_FIELDS = (
    "cache_file",
    "parcel_index",
    "patch_id",
    "parcel_id",
    "instance_id",
    "fold",
    "label",
    "source_label",
    "pixel_count",
    "s2_length",
    "s1a_length",
    "s1d_length",
)
ERROR_FIELDS = ("patch_id", "file", "exception", "reason", "traceback")
REQUIRED_ARRAYS = {
    "parcel_ids",
    "instance_ids",
    "labels",
    "source_labels",
    "pixel_counts",
    "s2",
    "s1a",
    "s1d",
    "dates_s2",
    "dates_s1a",
    "dates_s1d",
    "doy_s2",
    "doy_s1a",
    "doy_s1d",
    "patch_id",
    "fold",
    "stats_json",
    "feature_space",
    "normalized",
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temp), str(path))


def _atomic_csv(path, rows, fieldnames):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temp), str(path))


def _read_json(path, default):
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _validate_patch_file(path, expected_patch_id=None):
    try:
        with np.load(str(path), allow_pickle=False) as data:
            if not REQUIRED_ARRAYS.issubset(data.files):
                return False, "missing arrays: {}".format(
                    sorted(REQUIRED_ARRAYS.difference(data.files))
                )
            patch_id = int(data["patch_id"])
            if expected_patch_id is not None and patch_id != int(expected_patch_id):
                return False, "patch_id mismatch"
            if str(data["feature_space"].item()) != "raw_parcel_mean_std":
                return False, "feature_space is not raw_parcel_mean_std"
            if bool(data["normalized"].item()):
                return False, "raw cache is marked normalized"
            count = int(data["labels"].shape[0])
            for key in ("parcel_ids", "instance_ids", "source_labels", "pixel_counts"):
                if data[key].shape != (count,):
                    return False, "{} shape mismatch".format(key)
            expected_shapes = {
                "s2": (count, data["dates_s2"].shape[0], 20),
                "s1a": (count, data["dates_s1a"].shape[0], 6),
                "s1d": (count, data["dates_s1d"].shape[0], 6),
            }
            for key, shape in expected_shapes.items():
                if data[key].shape != shape:
                    return False, "{} shape {} != {}".format(key, data[key].shape, shape)
                if data[key].dtype != np.float32:
                    return False, "{} dtype is not float32".format(key)
                if not np.all(np.isfinite(data[key])):
                    return False, "{} contains NaN/Inf".format(key)
            for satellite in ("s2", "s1a", "s1d"):
                if data["dates_{}".format(satellite)].shape != data[
                    "doy_{}".format(satellite)
                ].shape:
                    return False, "{} date/DOY mismatch".format(satellite)
            if count and (int(data["labels"].min()) < 0 or int(data["labels"].max()) > 17):
                return False, "classification label outside 0..17"
    except Exception as exc:
        return False, "{}: {}".format(type(exc).__name__, exc)
    return True, "ok"


def _write_patch(path, record, parcel_data, stats):
    temp = path.with_suffix(path.suffix + ".tmp")
    payload = dict(parcel_data)
    payload.update(
        {
            "patch_id": np.asarray(record.patch_id, dtype=np.int64),
            "fold": np.asarray(record.fold, dtype=np.int8),
            "dates_s2": record.dates_s2.astype(np.int32),
            "dates_s1a": record.dates_s1a.astype(np.int32),
            "dates_s1d": record.dates_s1d.astype(np.int32),
            "doy_s2": dates_to_doy(record.dates_s2),
            "doy_s1a": dates_to_doy(record.dates_s1a),
            "doy_s1d": dates_to_doy(record.dates_s1d),
            "stats_json": np.asarray(json.dumps(stats, sort_keys=True)),
            "feature_space": np.asarray("raw_parcel_mean_std"),
            "normalized": np.asarray(False, dtype=np.bool_),
        }
    )
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())
    valid, reason = _validate_patch_file(temp, record.patch_id)
    if not valid:
        raise ValueError("Temporary cache validation failed: {}".format(reason))
    os.replace(str(temp), str(path))


def _manifest_rows(path, cache_root):
    with np.load(str(path), allow_pickle=False) as data:
        relative = path.relative_to(cache_root).as_posix()
        patch_id = int(data["patch_id"])
        fold = int(data["fold"])
        rows = []
        for index in range(data["labels"].shape[0]):
            rows.append(
                {
                    "cache_file": relative,
                    "parcel_index": index,
                    "patch_id": patch_id,
                    "parcel_id": int(data["parcel_ids"][index]),
                    "instance_id": int(data["instance_ids"][index]),
                    "fold": fold,
                    "label": int(data["labels"][index]),
                    "source_label": int(data["source_labels"][index]),
                    "pixel_count": int(data["pixel_counts"][index]),
                    "s2_length": int(data["s2"].shape[1]),
                    "s1a_length": int(data["s1a"].shape[1]),
                    "s1d_length": int(data["s1d"].shape[1]),
                }
            )
    return rows


def _write_fragment(fragment_path, patch_path, cache_root):
    _atomic_csv(fragment_path, _manifest_rows(patch_path, cache_root), MANIFEST_FIELDS)


def _aggregate_manifest(fragment_dir, manifest_path):
    rows = []
    for path in sorted(fragment_dir.glob("patch_*.csv")):
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    rows.sort(key=lambda row: (int(row["patch_id"]), int(row["parcel_index"])))
    _atomic_csv(manifest_path, rows, MANIFEST_FIELDS)
    return len(rows)


def _normalization_metadata():
    return {
        "strategy": "official_patch_train_folds",
        "source_files": ["NORM_{}_patch.json".format(s) for s in ("S2", "S1A", "S1D")],
        "applied_during_cache_build": False,
        "stage": "dataset_or_training",
        "fold_policy": (
            "For each experiment, combine official statistics from that experiment's "
            "training folds only; validation/test folds are excluded"
        ),
        "implementation": (
            "Dataset transforms raw mean as (mean-channel_mean)/channel_std and "
            "raw std as std/channel_std"
        ),
    }


def build(args):
    config = load_config(args.config)
    dataset_config = config["dataset"]
    if dataset_config.get("parcel_stat") != "mean_std":
        raise ValueError("dataset.parcel_stat must be mean_std")
    root = resolve_project_path(dataset_config["root"])
    cache_root = resolve_project_path(dataset_config["cache_root"])
    parcels_dir = cache_root / "parcels"
    fragments_dir = cache_root / "manifests"
    parcels_dir.mkdir(parents=True, exist_ok=True)
    fragments_dir.mkdir(parents=True, exist_ok=True)

    channel_counts = {
        "S2": int(dataset_config["s2_bands"]),
        "S1A": int(dataset_config["s1_channels"]),
        "S1D": int(dataset_config["s1_channels"]),
    }
    reader = PastisRReader(root, channel_counts["S2"], channel_counts["S1A"])
    records = reader.records[: args.max_patches] if args.max_patches else reader.records

    progress_path = cache_root / "progress.json"
    progress = _read_json(
        progress_path,
        {"processed": [], "failed": [], "started_at": _now()},
    )
    processed = set(int(value) for value in progress.get("processed", []))
    failures = {
        int(item["patch_id"]): item for item in progress.get("failed", [])
    }
    run_processed = 0
    run_skipped = 0
    run_failed = 0

    for position, record in enumerate(records, start=1):
        patch_path = parcels_dir / "patch_{}.npz".format(record.patch_id)
        fragment_path = fragments_dir / "patch_{}.csv".format(record.patch_id)
        valid, _ = _validate_patch_file(patch_path, record.patch_id) if patch_path.is_file() else (False, "missing")
        if valid:
            # Rebuild the tiny fragment too, so resume cannot trust a stale or
            # partially written manifest row set merely because the NPZ is valid.
            _write_fragment(fragment_path, patch_path, cache_root)
            processed.add(record.patch_id)
            failures.pop(record.patch_id, None)
            run_skipped += 1
            progress.update(
                {
                    "updated_at": _now(),
                    "processed": sorted(processed),
                    "failed": [failures[key] for key in sorted(failures)],
                    "last_requested_patch_count": len(records),
                }
            )
            _atomic_json(progress_path, progress)
            _atomic_csv(
                cache_root / "cache_errors.csv",
                [failures[key] for key in sorted(failures)],
                ERROR_FIELDS,
            )
            print(
                "[{}/{}] patch {} SKIP (validated cache)".format(
                    position, len(records), record.patch_id
                ),
                flush=True,
            )
            continue

        try:
            patch = reader.load_patch(record)
            parcel_data, stats = extract_patch_parcels(
                patch,
                source_class_ids=dataset_config["source_class_ids"],
                background_label=dataset_config["background_label"],
                void_label=dataset_config["void_label"],
            )
            _write_patch(patch_path, record, parcel_data, stats)
            _write_fragment(fragment_path, patch_path, cache_root)
            processed.add(record.patch_id)
            failures.pop(record.patch_id, None)
            run_processed += 1
            print(
                "[{}/{}] patch {} OK: valid={} discarded={} inconsistent_labels={}".format(
                    position,
                    len(records),
                    record.patch_id,
                    stats["valid_parcels"],
                    stats["discarded_parcels"],
                    stats["inconsistent_semantic_parcels"],
                ),
                flush=True,
            )
        except Exception as exc:
            run_failed += 1
            failures[record.patch_id] = {
                "patch_id": record.patch_id,
                "file": "patch_{}.npz".format(record.patch_id),
                "exception": type(exc).__name__,
                "reason": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }
            print(
                "[{}/{}] patch {} FAILED: {}: {}".format(
                    position, len(records), record.patch_id, type(exc).__name__, exc
                ),
                file=sys.stderr,
                flush=True,
            )
        finally:
            progress.update(
                {
                    "updated_at": _now(),
                    "processed": sorted(processed),
                    "failed": [failures[key] for key in sorted(failures)],
                    "last_requested_patch_count": len(records),
                }
            )
            _atomic_json(progress_path, progress)
            _atomic_csv(
                cache_root / "cache_errors.csv",
                [failures[key] for key in sorted(failures)],
                ERROR_FIELDS,
            )

    parcel_count = _aggregate_manifest(fragments_dir, cache_root / "manifest.csv")
    metadata = {
        "schema_version": 2,
        "updated_at": _now(),
        "dataset_root": str(root),
        "feature_space": "raw_parcel_mean_std",
        "normalized": False,
        "expected_patches": len(reader.records),
        "cached_patches": len(list(fragments_dir.glob("patch_*.csv"))),
        "cached_valid_parcels": parcel_count,
        "feature_layout": {
            "s2": {"shape": "[T, 20]", "order": "10 means then 10 stds"},
            "s1a": {"shape": "[T, 6]", "order": "3 means then 3 stds"},
            "s1d": {"shape": "[T, 6]", "order": "3 means then 3 stds"},
        },
        "label_rule": {
            "semantic_source": "ANNOTATIONS/TARGET_[ID_PATCH].npy channel 0",
            "instance_source": "INSTANCE_ANNOTATIONS/INSTANCES_[ID_PATCH].npy",
            "majority_on_inconsistency": True,
            "excluded_source_labels": [
                int(dataset_config["background_label"]),
                int(dataset_config["void_label"]),
            ],
            "source_to_cache": "source labels 1..18 mapped to cache labels 0..17",
            "class_names": {str(key): value for key, value in LABEL_NAMES.items()},
        },
        "normalization": _normalization_metadata(),
    }
    _atomic_json(cache_root / "metadata.json", metadata)
    print(
        "Run summary: processed={} skipped={} failed={} manifest_parcels={}".format(
            run_processed, run_skipped, run_failed, parcel_count
        )
    )
    print("Cache root: {}".format(cache_root))
    return 1 if run_failed else 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Only consider the first N metadata patches (for smoke tests).",
    )
    args = parser.parse_args()
    if args.max_patches is not None and args.max_patches <= 0:
        parser.error("--max-patches must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(build(parse_args()))
