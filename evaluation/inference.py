"""Frozen checkpoint inference over the six operational and one full-season views."""

import numpy as np
import pandas as pd
import torch

from data.progressive_brandenburg_external import ProgressiveBrandenburgExternal2019Dataset
from evaluation.metrics import classification_metrics
from training.runner import (
    CHECKPOINT_NAMES, build_dataset, build_model, forward_model, make_loader,
)
from utils.checkpoint import load_checkpoint
from utils.config import resolve_project_path


def test_dataset(config, cutoff):
    if config["dataset"]["name"] == "PASTIS-R":
        return build_dataset(config, "test", cutoff)
    data = config["data"]
    return ProgressiveBrandenburgExternal2019Dataset(
        data_root=resolve_project_path(data["root"]),
        split_manifest=resolve_project_path(data["splits"]["test"]),
        s2_stats_path=resolve_project_path(data["s2_stats"]),
        s1_stats_path=resolve_project_path(data["s1_stats"]),
        cutoff_mode=cutoff,
        modalities=("s2", "s1a", "s1d"),
    )


def load_model(config, checkpoint_path, device):
    model = build_model(config).to(device)
    state = load_checkpoint(resolve_project_path(checkpoint_path), model, map_location=device)
    if state.get("test_used_for_selection"):
        raise RuntimeError("Checkpoint reports Test-based model selection")
    return model.eval(), state


def predict_checkpoints(config, checkpoint_path, split, checkpoints=CHECKPOINT_NAMES):
    """Return per-checkpoint logits, identities and observed-prefix quality."""
    if split not in ("train", "val", "cal_b", "calibration", "test"):
        raise ValueError("Unsupported split")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(config, checkpoint_path, device)
    if split == "test":
        dataset = test_dataset(config, "FullSeason")
    else:
        dataset = build_dataset(config, split, "FullSeason")
    seed = config.get("seed", config["training"].get("seed", 42))
    loader = make_loader(dataset, config, False, seed + 1)
    result = {}
    for checkpoint in checkpoints:
        dataset.set_cutoff(checkpoint)
        records, logits_chunks = [], []
        with torch.no_grad():
            for batch in loader:
                logits = forward_model(model, batch, device).detach().cpu().numpy()
                logits_chunks.append(logits)
                for i, (patch_id, parcel_id, label) in enumerate(zip(
                    batch["patch_id"].tolist(), batch["parcel_id"].tolist(), batch["label"].tolist()
                )):
                    length = int(batch["s2_lengths"][i])
                    quality = batch["s2_quality"][i, :length, 0].numpy()
                    clear_fraction = float(quality.mean()) if config["dataset"]["name"] == "PASTIS-R" else float(1.0 - quality.mean())
                    records.append((int(patch_id), int(parcel_id), int(label), clear_fraction))
        logits = np.concatenate(logits_chunks)
        frame = pd.DataFrame(records, columns=("patch_id", "parcel_id", "label", "mean_clear_fraction"))
        order = np.lexsort((frame.parcel_id.to_numpy(), frame.patch_id.to_numpy()))
        frame = frame.iloc[order].reset_index(drop=True)
        logits = logits[order]
        if frame[["patch_id", "parcel_id"]].duplicated().any() or not np.isfinite(logits).all():
            raise RuntimeError("Duplicate parcel identity or non-finite logits")
        frame["checkpoint"] = checkpoint
        for class_id in range(logits.shape[1]):
            frame[f"logit_{class_id}"] = logits[:, class_id]
        result[checkpoint] = frame
    reference = result[CHECKPOINT_NAMES[0]][["patch_id", "parcel_id", "label"]].to_numpy()
    if any(not np.array_equal(reference, frame[["patch_id", "parcel_id", "label"]].to_numpy()) for frame in result.values()):
        raise RuntimeError("Parcel ordering changed between checkpoints")
    return result


def classification_by_checkpoint(predictions, num_classes):
    output = {}
    for checkpoint, frame in predictions.items():
        logits = frame[[f"logit_{i}" for i in range(num_classes)]].to_numpy(dtype=np.float64)
        metrics = classification_metrics(frame.label.to_numpy(), logits.argmax(1), num_classes)
        output[checkpoint] = {key: metrics[key] for key in ("oa", "macro_f1", "kappa")}
    return output
