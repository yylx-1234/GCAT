"""Leakage-safe progressive Brandenburg 2018 parcel dataset."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import OrderedDict
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.config import resolve_project_path


CHECKPOINTS = OrderedDict((
    ("Apr30", 20180430),
    ("May31", 20180531),
    ("Jun30", 20180630),
    ("Jul31", 20180731),
    ("Aug31", 20180831),
    ("Sep30", 20180930),
    ("FullSeason", None),
))
CLASS_NAMES = (
    "Wheat", "Rye", "Barley", "Oats", "Corn", "Oil Seeds",
    "Root Crops", "Meadows", "Forage Crops",
)


def _date_int(values: np.ndarray) -> torch.Tensor:
    return torch.as_tensor([int(str(value)[:10].replace("-", "")) for value in values], dtype=torch.long)


def _block_integer(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:15], 16)


class ProgressiveBrandenburgDataset(Dataset):
    """2018-only dataset with frozen Train statistics and asynchronous prefixes."""

    def __init__(
        self,
        data_root,
        split_manifest,
        s2_stats_path,
        s1_stats_path,
        cutoff_mode="FullSeason",
        seed=42,
        modalities=("s2", "s1a", "s1d"),
    ):
        self.data_root = Path(data_root).expanduser().resolve()
        self.split_manifest = Path(split_manifest).expanduser().resolve()
        self.modalities = tuple(modalities)
        if not self.modalities or set(self.modalities).difference(("s2", "s1a", "s1d")):
            raise ValueError("Invalid Brandenburg modalities")
        self.seed = int(seed)
        self.epoch = 0
        with self.split_manifest.open("r", newline="", encoding="utf-8-sig") as stream:
            source_rows = list(csv.DictReader(stream))
        self.rows = []
        for row in source_rows:
            crop_id = int(row["crop_id"])
            if crop_id not in range(1, 10):
                raise ValueError("Expected Brandenburg crop_id 1..9")
            self.rows.append({
                "fid": int(row["fid"]),
                "label": crop_id - 1,
                "source_label": crop_id,
                "spatial_block_id": row["spatial_block_id"],
                "split": row["split"],
            })
        s2_stats = json.loads(Path(s2_stats_path).read_text(encoding="utf-8"))
        if s2_stats.get("source_parcel_count") != 1769 or s2_stats.get("normalization_source") != "2018 spatial train split only":
            raise ValueError("S2 statistics are not the frozen 2018 Train-only statistics")
        s2_record = s2_stats["statistics"]["s2_features"]
        self.s2_mean = np.asarray(s2_record["mean"], dtype=np.float32)
        self.s2_std = np.asarray(s2_record["std"], dtype=np.float32)
        if self.s2_mean.shape != (20,) or self.s2_std.shape != (20,) or np.any(self.s2_std <= 0):
            raise ValueError("Invalid frozen S2 statistics")
        s1_stats = json.loads(Path(s1_stats_path).read_text(encoding="utf-8"))
        if s1_stats.get("status") != "passed" or s1_stats.get("source_parcel_count") != 1769 or s1_stats.get("external_2019_used"):
            raise ValueError("S1 statistics are not the frozen 2018 Train-only statistics")
        self.s1_stats = {}
        for orbit in ("s1a", "s1d"):
            record = s1_stats["statistics"][orbit]
            mean = np.asarray(record["mean"], dtype=np.float32)
            std = np.asarray(record["std"], dtype=np.float32)
            if mean.shape != (4,) or std.shape != (4,) or np.any(std <= 0):
                raise ValueError("Invalid frozen {} statistics".format(orbit))
            self.s1_stats[orbit] = (mean, std)
        self._cache = {}
        self.set_cutoff(cutoff_mode)

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def set_cutoff(self, cutoff_mode):
        cutoff_mode = str(cutoff_mode)
        if cutoff_mode != "random" and cutoff_mode not in CHECKPOINTS:
            raise ValueError("Unknown cutoff: {}".format(cutoff_mode))
        self.cutoff_mode = cutoff_mode

    def _cutoff_name(self, fid):
        if self.cutoff_mode != "random":
            return self.cutoff_mode
        value = (int(fid) * 2654435761 + self.epoch * 2246822519 + self.seed * 3266489917) & 0xFFFFFFFF
        return tuple(CHECKPOINTS)[value % len(CHECKPOINTS)]

    @staticmethod
    def _keep(dates, cutoff):
        return torch.ones_like(dates, dtype=torch.bool) if cutoff is None else dates <= int(cutoff)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.data_root / "cache/parcel/train2018" / "parcel_{}.npz".format(row["fid"])
        cached = self._cache.get(row["fid"])
        if cached is None:
            with np.load(path, allow_pickle=False) as archive:
                if int(archive["year"]) != 2018 or int(archive["crop_id"]) != row["source_label"]:
                    raise ValueError("Manifest/cache identity mismatch: {}".format(path))
                cached = {"pixel_count": int(archive["pixel_count"])}
                if "s2" in self.modalities:
                    cached["s2_dates"] = _date_int(archive["s2_dates"])
                    features = (np.asarray(archive["s2_features"], dtype=np.float32) - self.s2_mean) / self.s2_std
                    cached["s2_features"] = torch.from_numpy(features)
                    cached["s2_quality"] = torch.from_numpy(np.asarray(archive["s2_quality"], dtype=np.float32))
                for orbit in ("s1a", "s1d"):
                    if orbit not in self.modalities:
                        continue
                    cached["{}_dates".format(orbit)] = _date_int(archive["{}_dates".format(orbit)])
                    raw = np.asarray(archive["{}_features".format(orbit)], dtype=np.float32)
                    if np.any(raw < 0):
                        raise ValueError("Negative linear-power parcel statistic")
                    mean, std = self.s1_stats[orbit]
                    cached["{}_features".format(orbit)] = torch.from_numpy((np.log1p(raw) - mean) / std)
            self._cache[row["fid"]] = cached
        cutoff_name = self._cutoff_name(row["fid"])
        cutoff = CHECKPOINTS[cutoff_name]
        output = {
            "label": row["label"], "source_label": row["source_label"], "fold": 2018,
            "patch_id": _block_integer(row["spatial_block_id"]), "parcel_id": row["fid"],
            "instance_id": row["fid"], "pixel_count": cached["pixel_count"],
            "cutoff": cutoff_name, "cutoff_date": -1 if cutoff is None else cutoff,
        }
        for modality in self.modalities:
            dates = cached["{}_dates".format(modality)]
            keep = self._keep(dates, cutoff)
            output["{}_features".format(modality)] = cached["{}_features".format(modality)][keep]
            output["{}_dates".format(modality)] = dates[keep]
        if "s2" in self.modalities:
            output["s2_quality"] = cached["s2_quality"][self._keep(cached["s2_dates"], cutoff)]
        for modality in self.modalities:
            features = output["{}_features".format(modality)]
            dates = output["{}_dates".format(modality)]
            if not len(features) or len(features) != len(dates):
                raise ValueError("Empty or inconsistent {} prefix".format(modality))
            if len(dates) > 1 and not bool(torch.all(dates[1:] > dates[:-1])):
                raise ValueError("Non-monotonic {} dates".format(modality))
            if cutoff is not None and int(dates.max()) > cutoff:
                raise ValueError("Future leakage in {}".format(modality))
            if not bool(torch.isfinite(features).all()):
                raise FloatingPointError("Non-finite normalized {} feature".format(modality))
        if "s2" in self.modalities and not bool(torch.isfinite(output["s2_quality"]).all()):
            raise FloatingPointError("Non-finite S2 quality")
        return output


def build_dataset(config, split, cutoff_mode, modalities):
    return ProgressiveBrandenburgDataset(
        data_root=resolve_project_path(config["data"]["root"]),
        split_manifest=resolve_project_path(config["data"]["splits"][split]),
        s2_stats_path=resolve_project_path(config["data"]["s2_stats"]),
        s1_stats_path=resolve_project_path(config["data"]["s1_stats"]),
        cutoff_mode=cutoff_mode,
        seed=config["training"]["seed"],
        modalities=modalities,
    )
