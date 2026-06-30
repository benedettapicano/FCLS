from __future__ import annotations

import math
from typing import Any

import pandas as pd


def binary_selection_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "selected_count": int(y_pred.sum()),
        "true_relevant_count": int(y_true.sum()),
    }


CORE_REQUIREMENTS = {"water_quality", "weather", "wave", "current"}


def decision_from_subset(
    subset: pd.DataFrame,
    required: set[str] | None = None,
    current_time: float | None = None,
    unsafe_threshold: float = 0.56,
    safe_threshold: float = 0.46,
) -> str:

    if subset.empty:
        return "uncertain"
    if current_time is None:
        current_time = float(subset["current_time"].iloc[0]) if "current_time" in subset else 0.0
    valid = subset[(subset["tau_start"] <= current_time) & (subset["tau_end"] >= current_time)]
    if valid.empty:
        return "uncertain"

    core_seen = set(valid[valid["satisfies_requirement"].isin(CORE_REQUIREMENTS)]["satisfies_requirement"].dropna())
    confidence = float((valid["sigma_rel"] * valid["sigma_time"]).mean())
    risk = float(valid["hazard_evidence"].mean()) if "hazard_evidence" in valid else 0.5

    if len(core_seen) < 3 or confidence < 0.45:
        return "uncertain"
    if risk >= unsafe_threshold:
        return "unsafe"
    if risk <= safe_threshold:
        return "safe"
    return "uncertain"


def evaluate_method(df: pd.DataFrame, selected: pd.Series, method: str) -> dict[str, Any]:
    metrics = binary_selection_metrics(df["is_ground_truth_relevant"], selected)
    selected_df = df[selected]
    metrics["method"] = method
    metrics["contextual_relevance"] = float(selected_df["rel_score"].mean()) if not selected_df.empty else 0.0
    metrics["information_reduction"] = 1.0 - float(selected.sum()) / max(len(df), 1)
    metrics["downstream_load"] = int(selected.sum())
    pred = decision_from_subset(selected_df, current_time=float(df["current_time"].iloc[0]))
    truth = str(df["true_decision_state"].iloc[0])
    metrics["decision_accuracy"] = 1.0 if pred == truth else 0.0
    n = len(df)
    if method == "Observation-only":
        semantic_overhead = 0.0
    elif method == "Physical-semantic":
        semantic_overhead = 0.0015 * n
    else:
        semantic_overhead = 0.0025 * n * math.log(max(n, 2), 2)
    metrics["semantic_layer_overhead"] = semantic_overhead
    metrics["total_runtime"] = semantic_overhead + 0.018 * metrics["downstream_load"] + 0.10
    metrics["semantic_efficiency"] = (
        metrics["f1"]
        * metrics["contextual_relevance"]
        * (1.0 + metrics["information_reduction"])
        / max(metrics["total_runtime"], 1e-9)
    )
    metrics["predicted_decision_state"] = pred
    metrics["true_decision_state"] = truth
    return metrics
