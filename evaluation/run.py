"""Apply a frozen development policy to PASTIS-R Test or Brandenburg 2019."""

import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation.calibration import softmax_temperature
from evaluation.inference import classification_by_checkpoint, predict_checkpoints
from evaluation.operational import decision_summary
from evaluation.policies import assign_quality_bin
from training.runner import CHECKPOINT_NAMES
from utils.config import resolve_project_path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(config, checkpoint_path, policy_path, output_dir):
    policy = json.loads(resolve_project_path(policy_path).read_text(encoding="utf-8"))
    if policy["checkpoint_sha256"] != _sha256(resolve_project_path(checkpoint_path)):
        raise ValueError("Policy was calibrated for a different checkpoint")
    if policy.get("test_used_for_selection") or policy.get("full_season_in_policy"):
        raise ValueError("Policy violates frozen Test/operational protocol")
    if policy["dataset"] != config["dataset"]["name"]:
        raise ValueError("Policy/dataset mismatch")
    if config["dataset"]["name"] == "Brandenburg":
        if policy["protocol"] != "brandenburg-2018" or policy["source_year"] != 2018:
            raise ValueError("Brandenburg 2019 needs the 2018 frozen policy")
    elif policy["protocol"] == "primary-fold2":
        if tuple(config["data"]["test_folds"]) != (2,):
            raise ValueError("Primary Fold-2 policy cannot be applied to another fold")
    elif policy["protocol"] != "global-src":
        raise ValueError("Unknown PASTIS-R policy")
    selected = policy.get("selected_policy")
    if selected is None:
        raise ValueError("No development-certified/selected policy is available")
    predictions = predict_checkpoints(config, checkpoint_path, "test", CHECKPOINT_NAMES)
    num_classes = 18 if config["dataset"]["name"] == "PASTIS-R" else 9
    classification = classification_by_checkpoint(predictions, num_classes)
    frames, probabilities = [], []
    for name in CHECKPOINT_NAMES[:-1]:
        source = predictions[name]
        frame = source[["patch_id", "parcel_id", "label", "checkpoint", "mean_clear_fraction"]].copy()
        if policy["protocol"] == "primary-fold2":
            frame["quality_bin"] = assign_quality_bin(
                frame.mean_clear_fraction, policy["quality_boundaries"][name]
            )
        frames.append(frame)
        logits = source[[f"logit_{i}" for i in range(num_classes)]].to_numpy(dtype=np.float64)
        probabilities.append(softmax_temperature(logits, policy["temperatures"][name]))
    threshold = selected["thresholds"] if policy["protocol"] == "primary-fold2" else selected["threshold"]
    outcome, decision = decision_summary(frames, probabilities, threshold)
    target = resolve_project_path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    records = frames[0][["patch_id", "parcel_id", "label"]].copy()
    records["first_accept_index"] = outcome["accepted_index"]
    records["first_accept_checkpoint"] = [
        CHECKPOINT_NAMES[index] if index >= 0 else "DEFER" for index in outcome["accepted_index"]
    ]
    records["locked_prediction"] = outcome["accepted_label"]
    records["locked_confidence"] = outcome["accepted_confidence"]
    records["correct_accept"] = outcome["correct"]
    records.to_csv(target / "first_acceptance.csv", index=False)
    for name, frame in predictions.items():
        frame.to_csv(target / f"{name}_predictions.csv", index=False)
    summary = {
        "dataset": policy["dataset"], "protocol": policy["protocol"],
        "model_checkpoint_sha256": policy["checkpoint_sha256"],
        "policy_source": policy["policy_selection_source"],
        "test_used_for_selection": False,
        "classification_by_checkpoint": classification,
        "operational": decision,
        "risk_interpretation": "external empirical only; not formal 2019 certification"
        if policy["dataset"] == "Brandenburg" else
        "conditional on the adopted parcel-level calibration assumptions",
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
