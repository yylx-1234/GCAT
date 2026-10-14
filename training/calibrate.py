"""Fit temperatures on Validation, then choose decisions on development calibration."""

import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation.calibration import fit_temperature, softmax_temperature
from evaluation.inference import predict_checkpoints
from evaluation.policies import (
    assign_quality_bin, quality_boundaries, select_brandenburg_development,
    select_global_src, select_primary_fold2,
)
from training.runner import CHECKPOINT_NAMES, build_dataset
from utils.config import resolve_project_path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logits(frame, num_classes):
    return frame[[f"logit_{i}" for i in range(num_classes)]].to_numpy(dtype=np.float64)


def _train_quality_boundaries(config, checkpoints):
    dataset = build_dataset(config, "train", "FullSeason")
    output = {}
    for checkpoint in checkpoints:
        dataset.set_cutoff(checkpoint)
        values = [float(dataset[i]["s2_quality"][:, 0].mean()) for i in range(len(dataset))]
        output[checkpoint] = quality_boundaries(values)
    return output


def calibrate(config, checkpoint_path, output_path, protocol, target_risk=0.05):
    dataset_name = config["dataset"]["name"]
    if protocol == "primary-fold2":
        if dataset_name != "PASTIS-R" or tuple(config["data"]["test_folds"]) != (2,):
            raise ValueError("The primary quality-aware protocol is Fold-2 PASTIS-R only")
    elif protocol == "global-src":
        if dataset_name != "PASTIS-R":
            raise ValueError("Global SRC is the separate PASTIS-R cross-fold analysis")
    elif protocol == "brandenburg-2018":
        if dataset_name != "Brandenburg":
            raise ValueError("Brandenburg policy uses 2018 development data only")
    else:
        raise ValueError("Unknown policy protocol")
    calibration_split = "calibration" if dataset_name == "Brandenburg" else "cal_b"
    checkpoints = CHECKPOINT_NAMES
    val = predict_checkpoints(config, checkpoint_path, "val", checkpoints)
    cal = predict_checkpoints(config, checkpoint_path, calibration_split, checkpoints)
    num_classes = 18 if dataset_name == "PASTIS-R" else 9
    temperatures = {
        name: fit_temperature(_logits(val[name], num_classes), val[name].label.to_numpy(dtype=np.int64))
        for name in checkpoints
    }
    boundaries = _train_quality_boundaries(config, checkpoints[:-1]) if protocol == "primary-fold2" else None
    frames, probabilities = [], []
    for checkpoint in checkpoints[:-1]:
        frame = cal[checkpoint][["patch_id", "parcel_id", "label", "mean_clear_fraction"]].copy()
        if boundaries is not None:
            frame["quality_bin"] = assign_quality_bin(frame.mean_clear_fraction, boundaries[checkpoint])
        frames.append(frame)
        probabilities.append(softmax_temperature(_logits(cal[checkpoint], num_classes), temperatures[checkpoint]))
    if protocol == "primary-fold2":
        selected = select_primary_fold2(frames, probabilities, target_risk)
    elif protocol == "global-src":
        selected = select_global_src(frames, probabilities, target_risk)
    else:
        selected = select_brandenburg_development(frames, probabilities, target_risk)
    report = {
        "dataset": dataset_name,
        "protocol": protocol,
        "model": "gcat_rp_t",
        "checkpoint_sha256": _sha256(resolve_project_path(checkpoint_path)),
        "temperature_source": "validation_only",
        "temperature_objective": "negative_log_likelihood",
        "temperatures": temperatures,
        "quality_boundaries_source": "train_only" if boundaries is not None else None,
        "quality_boundaries": boundaries,
        "policy_selection_source": calibration_split,
        "test_used_for_selection": False,
        "source_year": 2018 if dataset_name == "Brandenburg" else 2019,
        "operational_checkpoints": list(checkpoints[:-1]),
        "full_season_in_policy": False,
        "target_risk": float(target_risk),
        "selected_policy": selected,
        "formal_2019_guarantee": False if dataset_name == "Brandenburg" else None,
    }
    output = resolve_project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
