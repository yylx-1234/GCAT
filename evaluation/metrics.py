"""OA, macro metrics, Cohen's kappa and per-class scores."""

import numpy as np


def confusion_matrix(y_true, y_pred, num_classes=18):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred shapes differ")
    if y_true.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    encoded = y_true * int(num_classes) + y_pred
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def classification_metrics(y_true, y_pred, num_classes=18):
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    true_count = matrix.sum(axis=1).astype(np.float64)
    pred_count = matrix.sum(axis=0).astype(np.float64)
    correct = np.diag(matrix).astype(np.float64)
    precision = np.divide(
        correct, pred_count, out=np.zeros_like(correct), where=pred_count > 0
    )
    recall = np.divide(
        correct, true_count, out=np.zeros_like(correct), where=true_count > 0
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(correct),
        where=(precision + recall) > 0,
    )
    total = float(matrix.sum())
    oa = float(correct.sum() / total) if total else 0.0
    expected = float((true_count * pred_count).sum() / (total * total)) if total else 0.0
    kappa = float((oa - expected) / (1.0 - expected)) if expected < 1.0 else 0.0
    return {
        "oa": oa,
        "accuracy": oa,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "kappa": kappa,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": matrix,
    }
