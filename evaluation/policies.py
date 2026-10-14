"""Keep primary quality-aware SRC, secondary global SRC, and external policy distinct."""

import itertools

import numpy as np
from scipy.stats import beta

from evaluation.sequential_selective_risk import clopper_pearson_upper, sequential_first_accept


GRID = np.round(np.arange(0.50, 1.00, 0.01), 2)
QUALITY_BINS = ("Q1", "Q2", "Q3", "Q4")


def quality_boundaries(train_scores):
    """Compute checkpoint-specific quartiles from Train parcels only."""
    return np.quantile(np.asarray(train_scores, dtype=np.float64), [0.25, 0.5, 0.75]).tolist()


def assign_quality_bin(scores, boundaries):
    indices = np.searchsorted(np.asarray(boundaries), np.asarray(scores), side="left")
    return np.asarray(QUALITY_BINS, dtype=object)[indices]


def _add_rectangle(diff, low, high, value=1):
    if any(low[axis] > high[axis] for axis in range(4)):
        return
    for bits in itertools.product((0, 1), repeat=4):
        corner = tuple(high[axis] + 1 if bits[axis] else low[axis] for axis in range(4))
        diff[corner] += value * (-1 if sum(bits) % 2 else 1)


def exhaustive_quality_policy_counts(frames, probabilities):
    """The source experiment's exact 50^4 first-accept candidate counting."""
    size = len(GRID)
    shape = (size + 1,) * 4
    accepted_diff = np.zeros(shape, dtype=np.int32)
    error_diff = np.zeros(shape, dtype=np.int32)
    labels = frames[0]["label"].to_numpy(dtype=np.int64)
    bins = [frame["quality_bin"].map({name: i for i, name in enumerate(QUALITY_BINS)}).to_numpy(dtype=np.int8) for frame in frames]
    for row in range(len(labels)):
        prior_max = [-1, -1, -1, -1]
        for time_index in range(len(frames)):
            quality_index = int(bins[time_index][row])
            confidence = float(probabilities[time_index][row].max())
            grid_index = int(np.searchsorted(GRID, confidence, side="right") - 1)
            if grid_index >= 0:
                low = [value + 1 for value in prior_max]
                high = [size - 1] * 4
                high[quality_index] = min(high[quality_index], grid_index)
                _add_rectangle(accepted_diff, low, high)
                if int(probabilities[time_index][row].argmax()) != int(labels[row]):
                    _add_rectangle(error_diff, low, high)
            prior_max[quality_index] = max(prior_max[quality_index], grid_index)
    accepted, errors = accepted_diff, error_diff
    for axis in range(4):
        accepted = accepted.cumsum(axis=axis)
        errors = errors.cumsum(axis=axis)
    slices = (slice(0, size),) * 4
    accepted, errors = accepted[slices], errors[slices]
    if np.any(accepted < 0) or np.any(errors < 0) or np.any(errors > accepted):
        raise RuntimeError("Quality-policy candidate count construction failed")
    return accepted, errors


def select_primary_fold2(frames, probabilities, target=0.05, family_delta=0.05):
    """Bonferroni-correct the complete Cartesian family, not 50 global thresholds."""
    accepted, errors = exhaustive_quality_policy_counts(frames, probabilities)
    local_delta = float(family_delta) / int(accepted.size)
    pairs = np.unique(np.stack((accepted.ravel(), errors.ravel()), axis=1), axis=0)
    pairs = pairs[pairs[:, 0] > 0]
    upper = beta.ppf(1.0 - local_delta, pairs[:, 1].astype(np.float64) + 1.0,
                     pairs[:, 0].astype(np.float64) - pairs[:, 1].astype(np.float64))
    certified = upper <= float(target)
    if not np.any(certified):
        return None
    candidates, bounds = pairs[certified], upper[certified]
    keep = candidates[:, 0] == candidates[:, 0].max()
    candidate_pairs, candidate_bounds = candidates[keep], bounds[keep]
    position = int(np.argmin(candidate_bounds))
    best_accept, best_error = (int(value) for value in candidate_pairs[position])
    location = np.argwhere((accepted == best_accept) & (errors == best_error))[0]
    return {
        "protocol": "primary_fold2_quality_aware", "thresholds": {
            QUALITY_BINS[axis]: float(GRID[int(location[axis])]) for axis in range(4)
        },
        "n_accept": best_accept, "n_error": best_error,
        "empirical_calibration_risk": best_error / best_accept,
        "risk_upper_bound": float(candidate_bounds[position]),
        "family_delta": float(family_delta), "family_size": int(accepted.size),
        "calibration_split": "cal_b", "target_risk": float(target),
    }


def select_global_src(frames, probabilities, target=0.05, family_delta=0.05):
    """Secondary 50-policy scalar-threshold analysis on Calibration-B."""
    local_delta = float(family_delta) / len(GRID)
    candidates = []
    for tau in GRID:
        outcome = sequential_first_accept(frames, probabilities, float(tau))
        accepted, errors = int(outcome["accepted"].sum()), int(outcome["wrong"].sum())
        candidates.append({
            "threshold": float(tau), "n_accept": accepted, "n_error": errors,
            "risk_upper_bound": clopper_pearson_upper(errors, accepted, local_delta),
        })
    certified = [row for row in candidates if row["n_accept"] and row["risk_upper_bound"] <= target]
    if not certified:
        return None
    best = max(certified, key=lambda row: (row["n_accept"], -row["risk_upper_bound"]))
    return {
        "protocol": "secondary_global_src", "threshold": best["threshold"],
        "n_accept": best["n_accept"], "n_error": best["n_error"],
        "risk_upper_bound": best["risk_upper_bound"], "family_delta": float(family_delta),
        "family_size": len(GRID), "calibration_split": "cal_b", "target_risk": float(target),
    }


def select_brandenburg_development(frames, probabilities, target=0.05):
    """2018 empirical risk-targeted rule; *not* a formal 2019 certification."""
    candidates = []
    for tau in GRID:
        outcome = sequential_first_accept(frames, probabilities, float(tau))
        accepted, errors = int(outcome["accepted"].sum()), int(outcome["wrong"].sum())
        if accepted:
            candidates.append((float(tau), accepted, errors, errors / accepted))
    feasible = [row for row in candidates if row[3] <= target]
    if not feasible:
        return None
    tau, accepted, errors, risk = min(feasible, key=lambda row: row[0])
    return {
        "protocol": "brandenburg_2018_empirical_development", "threshold": tau,
        "n_accept": accepted, "n_error": errors, "empirical_calibration_risk": risk,
        "single_policy_upper_bound": clopper_pearson_upper(errors, accepted, 0.05),
        "bonferroni_upper_bound": clopper_pearson_upper(errors, accepted, 0.05 / len(GRID)),
        "formally_certified": False, "calibration_split": "calibration", "target_risk": float(target),
    }
