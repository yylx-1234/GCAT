"""Lazy reader for patch-format PASTIS-R data."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PATCH_ID_RE = re.compile(r"_(\d+)$")


@dataclass(frozen=True)
class PatchRecord:
    patch_id: int
    fold: int
    dates_s2: np.ndarray
    dates_s1a: np.ndarray
    dates_s1d: np.ndarray
    metadata: dict


@dataclass(frozen=True)
class PatchPaths:
    s2: Path
    s1a: Path
    s1d: Path
    semantic: Path
    instances: Path
    parcel_ids: Path


@dataclass
class PatchData:
    record: PatchRecord
    paths: PatchPaths
    s2: np.ndarray
    s1a: np.ndarray
    s1d: np.ndarray
    semantic: np.ndarray
    instances: np.ndarray
    parcel_ids: np.ndarray


def _parse_dates(value, field, patch_id):
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        try:
            ordered = [value[key] for key in sorted(value, key=lambda item: int(item))]
        except (TypeError, ValueError):
            raise ValueError(
                "Patch {} has non-numeric keys in {}".format(patch_id, field)
            )
    elif isinstance(value, (list, tuple)):
        ordered = value
    else:
        raise TypeError(
            "Patch {} has unsupported {} value: {}".format(
                patch_id, field, type(value).__name__
            )
        )
    dates = np.asarray([int(item) for item in ordered], dtype=np.int32)
    if dates.ndim != 1 or dates.size == 0:
        raise ValueError("Patch {} has empty/invalid {}".format(patch_id, field))
    return dates


class PastisRReader:
    """Read metadata eagerly and load only one patch's arrays at a time."""

    REQUIRED_DIRS = (
        "DATA_S2",
        "DATA_S1A",
        "DATA_S1D",
        "ANNOTATIONS",
        "INSTANCE_ANNOTATIONS",
    )

    def __init__(self, root, s2_bands=10, s1_channels=3):
        self.root = Path(root).expanduser().resolve()
        self.s2_bands = int(s2_bands)
        self.s1_channels = int(s1_channels)
        self.metadata_path = self.root / "metadata.geojson"
        if not self.metadata_path.is_file():
            raise FileNotFoundError("Missing metadata: {}".format(self.metadata_path))
        for name in self.REQUIRED_DIRS:
            path = self.root / name
            if not path.is_dir():
                raise FileNotFoundError("Missing data directory: {}".format(path))
        self.records = self._read_metadata()
        self._indices = {
            name: self._index_directory(self.root / name) for name in self.REQUIRED_DIRS
        }

    def _read_metadata(self):
        with self.metadata_path.open("r", encoding="utf-8") as stream:
            geojson = json.load(stream)
        features = geojson.get("features")
        if not isinstance(features, list):
            raise ValueError("metadata.geojson is not a GeoJSON FeatureCollection")
        records = []
        seen = set()
        for feature in features:
            properties = feature.get("properties", {})
            patch_id = int(properties["ID_PATCH"])
            if patch_id in seen:
                raise ValueError("Duplicate ID_PATCH {}".format(patch_id))
            seen.add(patch_id)
            fold = int(properties["Fold"])
            if fold not in (1, 2, 3, 4, 5):
                raise ValueError("Patch {} has invalid Fold {}".format(patch_id, fold))
            records.append(
                PatchRecord(
                    patch_id=patch_id,
                    fold=fold,
                    dates_s2=_parse_dates(properties["dates-S2"], "dates-S2", patch_id),
                    dates_s1a=_parse_dates(
                        properties["dates-S1A"], "dates-S1A", patch_id
                    ),
                    dates_s1d=_parse_dates(
                        properties["dates-S1D"], "dates-S1D", patch_id
                    ),
                    metadata=properties,
                )
            )
        return sorted(records, key=lambda item: item.patch_id)

    @staticmethod
    def _index_directory(directory):
        index = {}
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() != ".npy":
                continue
            match = PATCH_ID_RE.search(path.stem)
            if match is None:
                continue
            index.setdefault(int(match.group(1)), []).append(path)
        return index

    def _choose(self, directory, patch_id, prefixes, validator, role):
        candidates = self._indices[directory].get(int(patch_id), [])
        if not candidates:
            raise FileNotFoundError(
                "No .npy candidate for patch {} in {}".format(patch_id, directory)
            )
        ranked = sorted(
            candidates,
            key=lambda path: (
                0
                if any(path.stem == "{}_{}".format(prefix, patch_id) for prefix in prefixes)
                else 1,
                path.name,
            ),
        )
        valid = []
        errors = []
        for path in ranked:
            try:
                array = np.load(str(path), mmap_mode="r", allow_pickle=False)
                if validator(array):
                    valid.append(path)
            except Exception as exc:
                errors.append("{}: {}".format(path.name, exc))
        preferred = [
            path
            for path in valid
            if any(path.stem == "{}_{}".format(prefix, patch_id) for prefix in prefixes)
        ]
        if len(preferred) == 1:
            return preferred[0]
        if len(valid) == 1:
            return valid[0]
        raise RuntimeError(
            "Could not uniquely identify {} for patch {}. valid={}, errors={}".format(
                role, patch_id, [path.name for path in valid], errors
            )
        )

    def paths_for(self, record):
        patch_id = record.patch_id

        def spatial(array):
            return array.ndim == 2 and array.shape == (128, 128)

        return PatchPaths(
            s2=self._choose(
                "DATA_S2",
                patch_id,
                ("S2",),
                lambda array: array.ndim == 4 and array.shape[1] == self.s2_bands,
                "S2 time series",
            ),
            s1a=self._choose(
                "DATA_S1A",
                patch_id,
                ("S1A",),
                lambda array: array.ndim == 4 and array.shape[1] == self.s1_channels,
                "S1A time series",
            ),
            s1d=self._choose(
                "DATA_S1D",
                patch_id,
                ("S1D",),
                lambda array: array.ndim == 4 and array.shape[1] == self.s1_channels,
                "S1D time series",
            ),
            semantic=self._choose(
                "ANNOTATIONS",
                patch_id,
                ("TARGET",),
                lambda array: (
                    (array.ndim == 3 and array.shape[1:] == (128, 128))
                    or spatial(array)
                )
                and np.issubdtype(array.dtype, np.integer)
                and int(np.max(array[0] if array.ndim == 3 else array)) <= 19,
                "semantic annotation",
            ),
            instances=self._choose(
                "INSTANCE_ANNOTATIONS",
                patch_id,
                ("INSTANCES",),
                lambda array: spatial(array)
                and np.issubdtype(array.dtype, np.integer)
                and int(np.min(array)) == 0
                and int(np.max(array)) > 0,
                "instance annotation",
            ),
            parcel_ids=self._choose(
                "ANNOTATIONS",
                patch_id,
                ("ParcelIDs", "PARCEL_IDS"),
                lambda array: spatial(array)
                and np.issubdtype(array.dtype, np.integer)
                and int(np.max(array)) > 19,
                "global parcel-id annotation",
            ),
        )

    def load_patch(self, record):
        paths = self.paths_for(record)
        arrays = {
            name: np.load(str(getattr(paths, name)), mmap_mode="r", allow_pickle=False)
            for name in ("s2", "s1a", "s1d", "semantic", "instances", "parcel_ids")
        }
        semantic = arrays["semantic"][0] if arrays["semantic"].ndim == 3 else arrays["semantic"]
        expected = {
            "s2": len(record.dates_s2),
            "s1a": len(record.dates_s1a),
            "s1d": len(record.dates_s1d),
        }
        for satellite, length in expected.items():
            actual = int(arrays[satellite].shape[0])
            if actual != length:
                raise ValueError(
                    "Patch {} {} time length {} != date length {}".format(
                        record.patch_id, satellite.upper(), actual, length
                    )
                )
            if arrays[satellite].shape[-2:] != semantic.shape:
                raise ValueError(
                    "Patch {} {} spatial shape {} != annotation {}".format(
                        record.patch_id,
                        satellite.upper(),
                        arrays[satellite].shape[-2:],
                        semantic.shape,
                    )
                )
        if arrays["instances"].shape != semantic.shape:
            raise ValueError("Instance and semantic annotation shapes do not match")
        if arrays["parcel_ids"].shape != semantic.shape:
            raise ValueError("Parcel ID and semantic annotation shapes do not match")
        return PatchData(
            record=record,
            paths=paths,
            s2=arrays["s2"],
            s1a=arrays["s1a"],
            s1d=arrays["s1d"],
            semantic=semantic,
            instances=arrays["instances"],
            parcel_ids=arrays["parcel_ids"],
        )
