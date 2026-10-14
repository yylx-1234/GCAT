"""First-accept-and-lock, MCAC, empirical risk, and paired lead-time gain."""

from datetime import datetime

import numpy as np

from evaluation.sequential_selective_risk import policy_counts, sequential_first_accept


def decision_summary(frames, probabilities, threshold):
    """Evaluate six monthly prefixes; never include the full-season reference."""
    outcome = sequential_first_accept(frames, probabilities, threshold)
    monthly = []
    for month_index in range(len(frames)):
        accepted = (outcome["accepted_index"] >= 0) & (outcome["accepted_index"] <= month_index)
        errors = accepted & (outcome["accepted_label"] != outcome["labels"])
        monthly.append({
            "checkpoint": str(frames[month_index]["checkpoint"].iloc[0]),
            "accepted_count": int(accepted.sum()),
            "accepted_coverage": float(accepted.mean()),
            "empirical_selective_risk": float(errors.sum() / accepted.sum()) if accepted.any() else 0.0,
        })
    terminal = policy_counts(outcome)
    terminal["mcac"] = float(np.mean([row["accepted_coverage"] for row in monthly]))
    terminal["monthly"] = monthly
    return outcome, terminal


def lead_time_gain(s2_outcome, gcat_outcome, checkpoint_dates):
    """S2 first-accept date minus GCAT first-accept date on jointly accepted parcels.

    The caller must supply outcomes for the same ordered parcel IDs and separate
    frozen development policies. GCAT-only accepts are counted as SAR-rescued,
    not assigned a fabricated numerical lead time.
    """
    if not np.array_equal(s2_outcome["parcel_ids"], gcat_outcome["parcel_ids"]):
        raise ValueError("Comparator and GCAT parcel IDs are not aligned")
    dates = np.asarray([datetime.strptime(value, "%Y-%m-%d").date().toordinal() for value in checkpoint_dates])
    s2, gcat = s2_outcome["accepted"], gcat_outcome["accepted"]
    both = s2 & gcat
    gain = dates[s2_outcome["accepted_index"][both]] - dates[gcat_outcome["accepted_index"][both]]
    return {
        "paired_count": int(both.sum()),
        "mean_gain_days": float(gain.mean()) if len(gain) else None,
        "median_gain_days": float(np.median(gain)) if len(gain) else None,
        "positive_gain_rate": float((gain > 0).mean()) if len(gain) else None,
        "sar_rescued_count": int((gcat & ~s2).sum()),
        "sar_rescued_rate_all": float((gcat & ~s2).mean()),
    }
