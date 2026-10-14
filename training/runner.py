"""GCAT-RP-T training with random prefixes and seven-point Validation selection."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data.progressive_brandenburg import CHECKPOINTS as BRANDENBURG_CHECKPOINTS
from data.progressive_brandenburg import build_dataset as build_brandenburg_dataset
from data.progressive_collate import progressive_collate_fn
from data.progressive_pastis import dataset_from_config, load_progressive_protocol
from evaluation.metrics import classification_metrics
from models.brandenburg_external import BrandenburgGCATRPTClassifier
from models.gcat_rp_t import GCATRPTClassifier
from training.losses import class_counts_from_rows, compute_class_weights
from utils.checkpoint import save_checkpoint
from utils.config import resolve_project_path
from utils.seed import seed_everything


CHECKPOINT_NAMES = ("Apr30", "May31", "Jun30", "Jul31", "Aug31", "Sep30", "FullSeason")


def build_model(config):
    if config["dataset"]["name"] == "PASTIS-R":
        values = dict(config["model"])
        if values.pop("name") != "gcat_rp_t":
            raise ValueError("This release trains GCAT-RP-T only")
        return GCATRPTClassifier(**values)
    if config["dataset"]["name"] == "Brandenburg":
        return BrandenburgGCATRPTClassifier()
    raise ValueError("Unknown dataset")


def build_dataset(config, split, cutoff):
    if config["dataset"]["name"] == "PASTIS-R":
        return dataset_from_config(config, split, cutoff, ("s2", "s1a", "s1d"))
    if split == "test" and cutoff == "random":
        raise ValueError("2019 Test cannot be used for random-prefix training")
    return build_brandenburg_dataset(config, split, cutoff, ("s2", "s1a", "s1d"))


def make_loader(dataset, config, shuffle, seed):
    generator = torch.Generator().manual_seed(int(seed))
    workers = int(config["data"].get("num_workers", 0))
    if workers != 0:
        raise ValueError("This release fixes num_workers=0 so checkpoint cutoffs reach the live Dataset")
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=progressive_collate_fn,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        persistent_workers=False,
    )


def forward_model(model, batch, device):
    """The source GCAT forward argument order; lengths stay on CPU for packing."""
    return model(
        batch["s2"].to(device, non_blocking=True), batch["s2_lengths"],
        batch["s2_dates"].to(device, non_blocking=True),
        batch["s1a"].to(device, non_blocking=True), batch["s1a_lengths"],
        batch["s1a_dates"].to(device, non_blocking=True),
        batch["s1d"].to(device, non_blocking=True), batch["s1d_lengths"],
        batch["s1d_dates"].to(device, non_blocking=True),
    )


def run_epoch(model, loader, criterion, device, num_classes, optimizer=None, collect=False):
    training = optimizer is not None
    model.train(training)
    loss_total, count = 0.0, 0
    labels_all, logits_all, parcel_ids, patch_ids = [], [], [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            labels = batch["label"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = forward_model(model, batch, device)
            loss = criterion(logits, labels)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError("Non-finite GCAT output/loss")
            if training:
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("Non-finite GCAT gradient")
                optimizer.step()
            size = len(labels)
            count += size
            loss_total += float(loss.detach()) * size
            labels_all.append(labels.detach().cpu().numpy())
            logits_all.append(logits.detach().cpu().numpy())
            if collect:
                parcel_ids.extend(batch["parcel_id"].tolist())
                patch_ids.extend(batch["patch_id"].tolist())
    labels_np = np.concatenate(labels_all)
    logits_np = np.concatenate(logits_all)
    result = classification_metrics(labels_np, logits_np.argmax(1), num_classes)
    result["loss"] = loss_total / count
    if collect:
        result.update(
            labels=labels_np, logits=logits_np,
            parcel_ids=np.asarray(parcel_ids, dtype=np.int64),
            patch_ids=np.asarray(patch_ids, dtype=np.int64),
        )
    return result


def protocol_checkpoints(config):
    if config["dataset"]["name"] == "PASTIS-R":
        _, checkpoints = load_progressive_protocol(resolve_project_path(config["data"]["protocol"]))
    else:
        checkpoints = BRANDENBURG_CHECKPOINTS
    if tuple(checkpoints) != CHECKPOINT_NAMES:
        raise ValueError("The seven checkpoints differ from the published protocol")
    return tuple(checkpoints)


def train(config):
    if config["dataset"]["name"] == "Brandenburg" and "2019" in str(config["data"]["splits"]["train"]):
        raise ValueError("Brandenburg 2019 is frozen Test, never Train")
    seed = int(config["seed"] if config["dataset"]["name"] == "PASTIS-R" else config["training"]["seed"])
    seed_everything(seed)
    train_data = build_dataset(config, "train", "random")
    val_data = build_dataset(config, "val", "FullSeason")
    expected = config["data"].get("expected_counts", {})
    for name, dataset in (("train", train_data), ("val", val_data)):
        if name in expected and len(dataset) != int(expected[name]):
            raise ValueError(f"{name} count differs from the configured protocol")
    train_loader = make_loader(train_data, config, True, seed)
    val_loader = make_loader(val_data, config, False, seed + 1)
    checkpoints = protocol_checkpoints(config)
    num_classes = 18 if config["dataset"]["name"] == "PASTIS-R" else 9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    counts = class_counts_from_rows(train_data.rows, num_classes)
    _, _, weights = compute_class_weights(counts, config["training"]["max_class_weight"])
    criterion = torch.nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float32, device=device))
    optimizer = AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]),
                      weight_decay=float(config["training"]["weight_decay"]))
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=float(config["training"]["scheduler_factor"]),
        patience=int(config["training"]["scheduler_patience"]),
    )
    output = resolve_project_path(config["output_root"]) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best, patience, history = -float("inf"), 0, []
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_data.set_epoch(epoch)
        trained = run_epoch(model, train_loader, criterion, device, num_classes, optimizer=optimizer)
        validation = {}
        for checkpoint in checkpoints:
            val_data.set_cutoff(checkpoint)
            validation[checkpoint] = run_epoch(model, val_loader, criterion, device, num_classes)
        mean_f1 = float(np.mean([validation[name]["macro_f1"] for name in checkpoints]))
        scheduler.step(mean_f1)
        history.append({"epoch": epoch, "train_macro_f1": trained["macro_f1"],
                        "val_mean_macro_f1": mean_f1, "lr": optimizer.param_groups[0]["lr"]})
        if mean_f1 > best:
            best, patience = mean_f1, 0
            save_checkpoint(output / "best.pt", {
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch, "best_mean_val_macro_f1": best,
                "selection_metric": "mean_validation_macro_f1_over_7_checkpoints",
                "config": config, "class_weights": weights.tolist(), "seed": seed,
                "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "calibration_used_for_selection": False, "test_used_for_selection": False,
            })
        else:
            patience += 1
        print(f"epoch={epoch} val_mean_macro_f1={mean_f1:.6f} best={best:.6f}", flush=True)
        if patience >= int(config["training"]["early_stopping_patience"]):
            break
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return output
