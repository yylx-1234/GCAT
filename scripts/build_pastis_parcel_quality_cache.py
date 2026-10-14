"""Build a non-destructive parcel-level S2 cloud-quality sidecar for PASTIS-R."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pastis_r import PastisRReader


DEFAULT_PASTIS_ROOT = PROJECT_ROOT / "datasets/PASTIS-R"
DEFAULT_BASE_CACHE = PROJECT_ROOT / "cache/parcel_mean_std"
DEFAULT_OUTPUT = PROJECT_ROOT / "cache/parcel_quality"
QUALITY_NAMES = ["clear_fraction", "thick_cloud_fraction", "thin_cloud_fraction", "cloud_shadow_fraction"]


def atomic_npz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_sidecar(path: Path, base_path: Path, expected_patch_id: int) -> tuple[bool, str]:
    try:
        with np.load(base_path, allow_pickle=False) as base, np.load(path, allow_pickle=False) as data:
            required = {
                "patch_id", "fold", "parcel_ids", "instance_ids", "labels", "pixel_counts",
                "s2_quality", "dates_s2", "dates_s1a", "dates_s1d", "quality_names",
                "base_cache_relative", "join_rule",
            }
            if not required.issubset(data.files):
                return False, f"missing {sorted(required.difference(data.files))}"
            if int(data["patch_id"]) != expected_patch_id:
                return False, "patch_id mismatch"
            count = len(base["labels"])
            if data["s2_quality"].shape != (count, len(base["dates_s2"]), 4):
                return False, "quality shape mismatch"
            for name in ["parcel_ids", "instance_ids", "labels", "pixel_counts", "dates_s2", "dates_s1a", "dates_s1d"]:
                if not np.array_equal(data[name], base[name]):
                    return False, f"{name} differs from frozen base cache"
            quality = data["s2_quality"]
            if quality.dtype != np.float32 or not np.all(np.isfinite(quality)):
                return False, "quality dtype/nonfinite"
            if quality.size and (float(quality.min()) < -1e-7 or float(quality.max()) > 1 + 1e-7):
                return False, "fraction outside [0,1]"
            if not np.allclose(quality.sum(axis=2), 1.0, atol=1e-6, rtol=0):
                return False, "fractions do not sum to one"
        return True, "ok"
    except Exception as error:
        return False, repr(error)


def build_quality(instance_map: np.ndarray, instance_ids: np.ndarray, cloud_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat_instances = np.asarray(instance_map, dtype=np.int64).reshape(-1)
    max_instance = int(max(flat_instances.max(), instance_ids.max(initial=0)))
    total = np.bincount(flat_instances, minlength=max_instance + 1)
    selected_total = total[instance_ids]
    if np.any(selected_total == 0):
        raise ValueError(f"zero-pixel cached instances: {instance_ids[selected_total == 0].tolist()}")
    quality = np.empty((len(instance_ids), cloud_mask.shape[0], 4), dtype=np.float32)
    for time_index in range(cloud_mask.shape[0]):
        labels = np.asarray(cloud_mask[time_index, 0], dtype=np.int8).reshape(-1)
        for class_id in range(4):
            class_counts = np.bincount(
                flat_instances, weights=(labels == class_id).astype(np.uint8), minlength=max_instance + 1
            )
            quality[:, time_index, class_id] = (class_counts[instance_ids] / selected_total).astype(np.float32)
    return quality, selected_total.astype(np.int32)


def build(args: argparse.Namespace) -> int:
    root = args.root.expanduser().resolve()
    base_cache = args.base_cache.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    output_dir = output_root / "parcels"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = PROJECT_ROOT / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    base_metadata = json.loads((base_cache / "metadata.json").read_text(encoding="utf-8"))
    if base_metadata.get("cached_patches") != 2433 or base_metadata.get("cached_valid_parcels") != 85951:
        raise RuntimeError("Frozen base cache metadata does not contain 2433 patches / 85951 parcels")
    if base_metadata.get("feature_space") != "raw_parcel_mean_std" or base_metadata.get("normalized"):
        raise RuntimeError("Frozen base cache semantic gate failed")
    cloud_metadata = json.loads((root / "REAL_MASKS/model_metadata.json").read_text(encoding="utf-8"))
    if cloud_metadata.get("status") != "passed":
        raise RuntimeError("Full cloud-mask build has not passed")

    reader = PastisRReader(root)
    records = reader.records[: args.max_patches] if args.max_patches else reader.records
    started = time.time()
    written = skipped = failed = 0
    failures: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for position, record in enumerate(records, 1):
        base_path = base_cache / "parcels" / f"patch_{record.patch_id}.npz"
        output_path = output_dir / f"patch_{record.patch_id}.npz"
        if output_path.exists() and not args.overwrite:
            valid, _ = validate_sidecar(output_path, base_path, record.patch_id)
        else:
            valid = False
        if valid:
            skipped += 1
        else:
            try:
                paths = reader.paths_for(record)
                instance_map = np.load(paths.instances, mmap_mode="r", allow_pickle=False)
                cloud_mask = np.load(root / "REAL_MASKS" / f"{record.patch_id}.npy", mmap_mode="r", allow_pickle=False)
                with np.load(base_path, allow_pickle=False) as base:
                    quality, source_pixel_counts = build_quality(instance_map, base["instance_ids"], cloud_mask)
                    if not np.array_equal(source_pixel_counts, base["pixel_counts"]):
                        raise ValueError("instance pixel counts differ from frozen parcel cache")
                    payload = {
                        "patch_id": np.asarray(record.patch_id, dtype=np.int64),
                        "fold": np.asarray(int(base["fold"]), dtype=np.int8),
                        "parcel_ids": np.asarray(base["parcel_ids"], dtype=np.int64),
                        "instance_ids": np.asarray(base["instance_ids"], dtype=np.int64),
                        "labels": np.asarray(base["labels"], dtype=np.int8),
                        "pixel_counts": np.asarray(base["pixel_counts"], dtype=np.int32),
                        "s2_quality": quality,
                        "dates_s2": np.asarray(base["dates_s2"], dtype=np.int32),
                        "dates_s1a": np.asarray(base["dates_s1a"], dtype=np.int32),
                        "dates_s1d": np.asarray(base["dates_s1d"], dtype=np.int32),
                        "quality_names": np.asarray(QUALITY_NAMES, dtype="U24"),
                        "base_cache_relative": np.asarray(f"parcels/patch_{record.patch_id}.npz"),
                        "join_rule": np.asarray("join frozen raw features and quality by patch_id + parcel array index"),
                    }
                atomic_npz(output_path, payload)
                valid, reason = validate_sidecar(output_path, base_path, record.patch_id)
                if not valid:
                    raise ValueError(f"written sidecar failed validation: {reason}")
                written += 1
            except Exception as error:
                failed += 1
                failures.append({"patch_id": record.patch_id, "error": repr(error)})
        if output_path.exists():
            with np.load(output_path, allow_pickle=False) as data:
                for index in range(len(data["labels"])):
                    manifest_rows.append(
                        {
                            "quality_file": output_path.relative_to(output_root).as_posix(),
                            "parcel_index": index,
                            "patch_id": record.patch_id,
                            "parcel_id": int(data["parcel_ids"][index]),
                            "instance_id": int(data["instance_ids"][index]),
                            "fold": int(data["fold"]),
                            "label": int(data["labels"][index]),
                            "s2_length": int(data["s2_quality"].shape[1]),
                        }
                    )
        if position == 1 or position % 50 == 0 or position == len(records):
            print(f"[{position}/{len(records)}] written={written} skipped={skipped} failed={failed}", flush=True)
            (audit_dir / "pastis_parcel_quality_progress.json").write_text(
                json.dumps({"processed": position, "written": written, "skipped": skipped, "failed": failed, "failures": failures}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["quality_file", "parcel_index", "patch_id", "parcel_id", "instance_id", "fold", "label", "s2_length"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    metadata = {
        "status": "passed" if args.max_patches is None and not failures and len(manifest_rows) == 85951 else ("partial" if not failures else "failed"),
        "schema_version": 1,
        "storage": "sidecar; frozen feature cache is not modified",
        "join_rule": "patch_id + parcel array index; IDs and labels are redundantly validated",
        "base_cache": str(base_cache),
        "quality_shape": "[N_parcels,T_S2,4]",
        "quality_names": QUALITY_NAMES,
        "cloud_classes": {"0": "clear", "1": "thick cloud", "2": "thin cloud", "3": "cloud shadow"},
        "cached_patches": len(records) - failed,
        "cached_parcels": len(manifest_rows),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "failures": failures,
        "elapsed_seconds": time.time() - started,
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_PASTIS_ROOT)
    parser.add_argument("--base-cache", type=Path, default=DEFAULT_BASE_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(build(parse_args()))
