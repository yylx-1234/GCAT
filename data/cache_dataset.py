"""PyTorch Dataset that reads only the parcel cache, never raw PASTIS-R."""

import csv
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.normalization import load_official_train_stats, normalize_mean_std
from data.splits import folds_for_split
from utils.config import load_config, resolve_project_path


class ParcelCacheDataset(Dataset):
    AVAILABLE_MODALITIES = ("s2", "s1a", "s1d")
    SATELLITE_KEYS = {"s2": "S2", "s1a": "S1A", "s1d": "S1D"}

    def __init__(
        self,
        cache_root,
        normalization_stats,
        folds=None,
        patch_cache_size=2,
        modalities=None,
    ):
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.modalities = tuple(modalities or self.AVAILABLE_MODALITIES)
        invalid_modalities = set(self.modalities).difference(self.AVAILABLE_MODALITIES)
        if invalid_modalities or not self.modalities:
            raise ValueError(
                "modalities must be a non-empty subset of {}; got {}".format(
                    self.AVAILABLE_MODALITIES, self.modalities
                )
            )
        metadata_path = self.cache_root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError("Missing cache metadata: {}".format(metadata_path))
        with metadata_path.open("r", encoding="utf-8") as stream:
            cache_metadata = json.load(stream)
        if cache_metadata.get("feature_space") != "raw_parcel_mean_std":
            raise ValueError("Dataset requires feature_space=raw_parcel_mean_std")
        if cache_metadata.get("normalized") is not False:
            raise ValueError("Dataset refuses a cache that may already be normalized")
        self.normalization_stats = self._validate_normalization(
            normalization_stats, self.modalities
        )
        manifest_path = self.cache_root / "manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError("Missing cache manifest: {}".format(manifest_path))
        selected = None if folds is None else {int(fold) for fold in folds}
        with manifest_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.rows = [
            {
                "cache_file": row["cache_file"],
                "parcel_index": int(row["parcel_index"]),
                "patch_id": int(row["patch_id"]),
                "parcel_id": int(row["parcel_id"]),
                "instance_id": int(row["instance_id"]),
                "fold": int(row["fold"]),
                "label": int(row["label"]),
                "source_label": int(row["source_label"]),
                "pixel_count": int(row["pixel_count"]),
            }
            for row in rows
            if selected is None or int(row["fold"]) in selected
        ]
        if not self.rows:
            raise ValueError("No parcels match the requested folds in {}".format(manifest_path))
        self.patch_cache_size = max(0, int(patch_cache_size))
        self._patch_cache = OrderedDict()

    @staticmethod
    def _validate_normalization(stats, modalities):
        if stats is None:
            raise ValueError(
                "normalization_stats is required; use ParcelCacheDataset.from_config "
                "to derive it from the current experiment's training folds"
            )
        result = {}
        expected = {"S2": 10, "S1A": 3, "S1D": 3}
        requested = {ParcelCacheDataset.SATELLITE_KEYS[item] for item in modalities}
        for satellite, channels in expected.items():
            if satellite not in requested:
                continue
            if satellite not in stats:
                raise KeyError("Missing normalization statistics for {}".format(satellite))
            mean = np.asarray(stats[satellite]["mean"], dtype=np.float32)
            std = np.asarray(stats[satellite]["std"], dtype=np.float32)
            if mean.shape != (channels,) or std.shape != (channels,):
                raise ValueError(
                    "{} normalization shape mismatch: mean={} std={}".format(
                        satellite, mean.shape, std.shape
                    )
                )
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
                raise ValueError("{} normalization contains NaN/Inf".format(satellite))
            if np.any(std <= 0):
                raise ValueError("{} normalization std must be positive".format(satellite))
            result[satellite] = {"mean": mean.copy(), "std": std.copy()}
        return result

    @classmethod
    def from_config(
        cls,
        config_or_path,
        split=None,
        folds=None,
        patch_cache_size=2,
        modalities=None,
    ):
        """Create a normalized view using only current experiment train folds."""
        config = (
            load_config(config_or_path)
            if isinstance(config_or_path, (str, Path))
            else config_or_path
        )
        if split is not None and folds is not None:
            raise ValueError("Specify either split or folds, not both")
        selected_folds = folds_for_split(config, split) if split is not None else folds
        dataset = config["dataset"]
        root = Path(dataset["root"]).expanduser().resolve()
        train_folds = folds_for_split(config, "train")
        normalization_stats = load_official_train_stats(
            root,
            train_folds,
            {
                "S2": int(dataset["s2_bands"]),
                "S1A": int(dataset["s1_channels"]),
                "S1D": int(dataset["s1_channels"]),
            },
        )
        return cls(
            resolve_project_path(dataset["cache_root"]),
            normalization_stats=normalization_stats,
            folds=selected_folds,
            patch_cache_size=patch_cache_size,
            modalities=modalities,
        )

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Each Windows spawn worker gets a clean, process-local bounded cache.
        state["_patch_cache"] = OrderedDict()
        return state

    def _load_patch(self, relative_path):
        cached = self._patch_cache.get(relative_path)
        if cached is not None:
            self._patch_cache.move_to_end(relative_path)
            return cached
        path = self.cache_root / Path(relative_path)
        if not path.is_file():
            raise FileNotFoundError("Manifest references missing cache file: {}".format(path))
        required = set()
        for satellite in self.modalities:
            required.update(
                (satellite, "dates_{}".format(satellite), "doy_{}".format(satellite))
            )
        with np.load(str(path), allow_pickle=False) as archive:
            cached = {key: archive[key] for key in required}
        for satellite in self.modalities:
            dates_key = "dates_{}".format(satellite)
            doy_key = "doy_{}".format(satellite)
            dates = cached[dates_key]
            order = np.argsort(dates, kind="stable")
            sorted_dates = dates[order]
            if np.any(np.diff(sorted_dates) <= 0):
                raise ValueError(
                    "{} has duplicate/non-increasing {} acquisition dates".format(
                        path, satellite
                    )
                )
            if not np.array_equal(order, np.arange(order.size)):
                cached[satellite] = cached[satellite][:, order, :]
                cached[dates_key] = sorted_dates
                cached[doy_key] = cached[doy_key][order]
        if self.patch_cache_size:
            self._patch_cache[relative_path] = cached
            self._patch_cache.move_to_end(relative_path)
            while len(self._patch_cache) > self.patch_cache_size:
                self._patch_cache.popitem(last=False)
        return cached

    def _normalize_features(self, raw_features, satellite):
        channels = raw_features.shape[1] // 2
        if raw_features.shape[1] != channels * 2:
            raise ValueError("Feature dimension must contain paired mean/std channels")
        stats = self.normalization_stats[satellite]
        mean, std = normalize_mean_std(
            raw_features[:, :channels],
            raw_features[:, channels:],
            stats["mean"],
            stats["std"],
        )
        normalized = np.concatenate((mean, std), axis=1).astype(
            np.float32, copy=False
        )
        if not np.all(np.isfinite(normalized)):
            raise ValueError("Dynamic normalization produced NaN/Inf for {}".format(satellite))
        return torch.from_numpy(normalized)

    def __getitem__(self, item):
        row = self.rows[item]
        data = self._load_patch(row["cache_file"])
        index = row["parcel_index"]
        sample = {
            "label": row["label"],
            "source_label": row["source_label"],
            "fold": row["fold"],
            "patch_id": row["patch_id"],
            "parcel_id": row["parcel_id"],
            "instance_id": row["instance_id"],
            "pixel_count": row["pixel_count"],
        }
        for satellite in self.modalities:
            sample[satellite] = self._normalize_features(
                data[satellite][index], self.SATELLITE_KEYS[satellite]
            )
            sample["dates_{}".format(satellite)] = torch.from_numpy(
                data["dates_{}".format(satellite)].astype(np.int64)
            )
            sample["doy_{}".format(satellite)] = torch.from_numpy(
                data["doy_{}".format(satellite)].astype(np.int64)
            )
        return sample
