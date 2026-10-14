"""Leakage-safe progressive views over the frozen raw PASTIS-R parcel cache."""

import csv
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from data.cache_dataset import ParcelCacheDataset
from data.normalization import load_official_train_stats
from utils.config import resolve_project_path


DEFAULT_MODALITIES = ("s2", "s1a", "s1d")


def date_to_int(value):
    if value is None:
        return None
    return int(str(value).replace("-", ""))


def load_progressive_protocol(path):
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    checkpoints = OrderedDict()
    for item in config["protocol"]["checkpoints"]:
        checkpoints[str(item["name"])] = date_to_int(item.get("cutoff"))
    if list(checkpoints) != [
        "Apr30", "May31", "Jun30", "Jul31", "Aug31", "Sep30", "FullSeason"
    ]:
        raise ValueError("Progressive protocol checkpoint order is not frozen v1")
    if checkpoints["FullSeason"] is not None:
        raise ValueError("FullSeason cutoff must be null")
    return config, checkpoints


class ProgressivePastisDataset(Dataset):
    """Return independently truncated asynchronous sequences and S2 quality.

    ``cutoff_mode`` is either ``random`` for training or a frozen checkpoint
    name for deterministic evaluation. Raw cache features remain untouched;
    normalization is dynamically derived from the explicitly configured
    training folds only.
    """

    def __init__(
        self,
        cache_root,
        quality_root,
        dataset_root,
        split_manifest,
        protocol_path,
        train_folds=(2, 3, 4),
        cutoff_mode="FullSeason",
        seed=42,
        modalities=DEFAULT_MODALITIES,
        patch_cache_size=2500,
    ):
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.quality_root = Path(quality_root).expanduser().resolve()
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split_manifest = Path(split_manifest).expanduser().resolve()
        self.protocol_path = Path(protocol_path).expanduser().resolve()
        self.protocol, self.checkpoints = load_progressive_protocol(self.protocol_path)
        self.train_folds = tuple(int(value) for value in train_folds)
        if len(self.train_folds) != 3 or len(set(self.train_folds)) != 3:
            raise ValueError("Progressive normalization requires three unique training folds")
        if any(fold not in (1, 2, 3, 4, 5) for fold in self.train_folds):
            raise ValueError("Progressive normalization folds must belong to Fold1-Fold5")
        self.modalities = tuple(modalities)
        invalid = set(self.modalities).difference(DEFAULT_MODALITIES)
        if invalid or not self.modalities:
            raise ValueError("Invalid modalities: {}".format(self.modalities))
        self.seed = int(seed)
        self.epoch = 0
        self.cutoff_mode = None

        stats = load_official_train_stats(
            self.dataset_root,
            self.train_folds,
            {"S2": 10, "S1A": 3, "S1D": 3},
        )
        base = ParcelCacheDataset(
            self.cache_root,
            normalization_stats=stats,
            folds=None,
            patch_cache_size=patch_cache_size,
            modalities=self.modalities,
        )
        base_rows = {
            (int(row["patch_id"]), int(row["parcel_id"])): row for row in base.rows
        }
        with self.split_manifest.open("r", encoding="utf-8", newline="") as stream:
            selected = list(csv.DictReader(stream))
        rows = []
        for row in selected:
            key = (int(row["patch_id"]), int(row["parcel_id"]))
            if key not in base_rows:
                raise KeyError("Split row is absent from frozen raw cache: {}".format(key))
            base_row = base_rows[key]
            if int(row["label"]) != int(base_row["label"]):
                raise ValueError("Split label differs from frozen raw cache: {}".format(key))
            rows.append(base_row)
        base.rows = rows
        self.base = base

        metadata_path = self.quality_root / "metadata.json"
        with metadata_path.open("r", encoding="utf-8") as stream:
            quality_metadata = json.load(stream)
        if quality_metadata.get("status") != "passed":
            raise ValueError("Parcel quality cache is not passed")
        self.patch_cache_size = max(0, int(patch_cache_size))
        self._quality_cache = OrderedDict()
        self.set_cutoff(cutoff_mode)

    def __len__(self):
        return len(self.base)

    @property
    def rows(self):
        return self.base.rows

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_quality_cache"] = OrderedDict()
        return state

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def set_cutoff(self, cutoff_mode):
        cutoff_mode = str(cutoff_mode)
        if cutoff_mode != "random" and cutoff_mode not in self.checkpoints:
            raise ValueError("Unknown cutoff mode: {}".format(cutoff_mode))
        self.cutoff_mode = cutoff_mode

    def _sample_cutoff_name(self, parcel_id):
        if self.cutoff_mode != "random":
            return self.cutoff_mode
        value = (
            int(parcel_id) * 2654435761
            + int(self.epoch) * 2246822519
            + self.seed * 3266489917
        ) & 0xFFFFFFFF
        names = tuple(self.checkpoints)
        return names[value % len(names)]

    def _load_quality_patch(self, patch_id):
        relative = "parcels/patch_{}.npz".format(int(patch_id))
        cached = self._quality_cache.get(relative)
        if cached is not None:
            self._quality_cache.move_to_end(relative)
            return cached
        path = self.quality_root / relative
        if not path.is_file():
            raise FileNotFoundError("Missing quality sidecar: {}".format(path))
        with np.load(str(path), allow_pickle=False) as archive:
            cached = {
                key: archive[key]
                for key in (
                    "patch_id", "fold", "parcel_ids", "instance_ids", "labels",
                    "s2_quality", "dates_s2", "dates_s1a", "dates_s1d"
                )
            }
        if self.patch_cache_size:
            self._quality_cache[relative] = cached
            self._quality_cache.move_to_end(relative)
            while len(self._quality_cache) > self.patch_cache_size:
                self._quality_cache.popitem(last=False)
        return cached

    @staticmethod
    def _prefix_mask(dates, cutoff):
        if cutoff is None:
            return torch.ones_like(dates, dtype=torch.bool)
        return dates <= int(cutoff)

    def __getitem__(self, item):
        raw = self.base[item]
        row = self.rows[item]
        quality = self._load_quality_patch(row["patch_id"])
        index = int(row["parcel_index"])
        if int(quality["parcel_ids"][index]) != int(row["parcel_id"]):
            raise ValueError("Quality/raw parcel join mismatch")
        if int(quality["labels"][index]) != int(row["label"]):
            raise ValueError("Quality/raw label join mismatch")
        cutoff_name = self._sample_cutoff_name(row["parcel_id"])
        cutoff_value = self.checkpoints[cutoff_name]
        sample = {
            key: raw[key]
            for key in (
                "label", "source_label", "fold", "patch_id", "parcel_id",
                "instance_id", "pixel_count"
            )
        }
        sample["cutoff"] = cutoff_name
        sample["cutoff_date"] = -1 if cutoff_value is None else int(cutoff_value)
        for satellite in self.modalities:
            dates = raw["dates_{}".format(satellite)]
            keep = self._prefix_mask(dates, cutoff_value)
            if not torch.any(keep):
                raise ValueError(
                    "Empty {} prefix for parcel {} at {}".format(
                        satellite, row["parcel_id"], cutoff_name
                    )
                )
            sample["{}_features".format(satellite)] = raw[satellite][keep]
            sample["{}_dates".format(satellite)] = dates[keep]
        if "s2" in self.modalities:
            quality_dates = torch.from_numpy(quality["dates_s2"].astype(np.int64))
            if not torch.equal(quality_dates, raw["dates_s2"]):
                raise ValueError("Quality/raw S2 dates differ")
            keep = self._prefix_mask(quality_dates, cutoff_value)
            s2_quality = torch.from_numpy(
                quality["s2_quality"][index].astype(np.float32, copy=False)
            )[keep]
            if s2_quality.shape[0] != sample["s2_features"].shape[0]:
                raise ValueError("S2 feature/quality prefix lengths differ")
            sample["s2_quality"] = s2_quality
        return sample


def dataset_from_config(config, split, cutoff_mode, modalities):
    data = config["data"]
    return ProgressivePastisDataset(
        cache_root=resolve_project_path(data["cache_root"]),
        quality_root=resolve_project_path(data["quality_root"]),
        dataset_root=resolve_project_path(config["dataset"]["root"]),
        split_manifest=resolve_project_path(data["split_manifests"][split]),
        protocol_path=resolve_project_path(data["protocol"]),
        train_folds=data.get("train_folds", [2, 3, 4]),
        cutoff_mode=cutoff_mode,
        seed=config["seed"],
        modalities=modalities,
        patch_cache_size=data.get("patch_cache_size", 2500),
    )
