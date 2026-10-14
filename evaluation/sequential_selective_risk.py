"""Finite-grid sequential selective-risk certification utilities."""

import math

import numpy as np
from scipy.stats import beta


def clopper_pearson_upper(errors, accepted, delta):
    """Exact one-sided (1-delta) binomial upper confidence bound."""
    errors, accepted, delta = int(errors), int(accepted), float(delta)
    if accepted <= 0:
        return 1.0
    if not 0 <= errors <= accepted or not 0 < delta < 1:
        raise ValueError("Invalid binomial-bound arguments")
    if errors == accepted:
        return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, accepted - errors))


def sequential_first_accept(frames, probabilities, thresholds):
    """Apply a global scalar or quality-bin threshold and lock first acceptance."""
    if len(frames) != len(probabilities):
        raise ValueError("Checkpoint frames/probabilities differ")
    parcel_ids = frames[0]["parcel_id"].to_numpy(dtype=np.int64)
    labels = frames[0]["label"].to_numpy(dtype=np.int64)
    n = len(parcel_ids)
    accepted_index = np.full(n, -1, dtype=np.int64)
    accepted_label = np.full(n, -1, dtype=np.int64)
    accepted_confidence = np.full(n, np.nan, dtype=np.float64)
    for checkpoint_index, (frame, current) in enumerate(zip(frames, probabilities)):
        if not np.array_equal(parcel_ids, frame["parcel_id"].to_numpy(dtype=np.int64)):
            raise ValueError("Parcel order differs across checkpoints")
        confidence = current.max(1)
        prediction = current.argmax(1)
        if isinstance(thresholds, dict):
            threshold = np.asarray([float(thresholds[value]) for value in frame["quality_bin"]])
        else:
            threshold = float(thresholds)
        newly = (accepted_index < 0) & (confidence >= threshold)
        accepted_index[newly] = checkpoint_index
        accepted_label[newly] = prediction[newly]
        accepted_confidence[newly] = confidence[newly]
    accepted = accepted_index >= 0
    correct = accepted & (accepted_label == labels)
    wrong = accepted & ~correct
    return {
        "parcel_ids": parcel_ids, "labels": labels, "accepted": accepted,
        "correct": correct, "wrong": wrong, "accepted_index": accepted_index,
        "accepted_label": accepted_label, "accepted_confidence": accepted_confidence,
    }


def policy_counts(outcome):
    accepted = int(outcome["accepted"].sum())
    errors = int(outcome["wrong"].sum())
    total = int(len(outcome["labels"]))
    return {
        "n_total": total, "n_accept": accepted, "n_error": errors,
        "n_correct": accepted - errors,
        "accepted_coverage": accepted / total,
        "empirical_selective_risk": errors / accepted if accepted else 0.0,
        "unresolved_rate": 1.0 - accepted / total,
    }


def certify_candidates(candidate_records, target_risk, family_delta=0.05):
    """Bonferroni-correct every fixed candidate, then maximize certified coverage."""
    candidate_records = [dict(record) for record in candidate_records]
    if not candidate_records:
        raise ValueError("Candidate family is empty")
    local_delta = float(family_delta) / len(candidate_records)
    for record in candidate_records:
        record["one_sided_confidence"] = 1.0 - local_delta
        record["family_delta"] = float(family_delta)
        record["family_size"] = len(candidate_records)
        record["risk_upper_bound"] = clopper_pearson_upper(
            record["n_error"], record["n_accept"], local_delta
        )
        record["certified"] = record["risk_upper_bound"] <= float(target_risk)
    certified = [record for record in candidate_records if record["certified"]]
    selected = None
    if certified:
        selected = max(certified, key=lambda record: (record["n_accept"], -record["risk_upper_bound"]))
    return candidate_records, selected

