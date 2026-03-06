"""Model evaluation and metric computation."""
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def evaluate_model(
    model: Any,
    X: Any,
    y: Any,
    prefix: str = "",
) -> Dict[str, float]:
    """Compute standard classification metrics and confusion matrix."""
    y_pred = model.predict(X)
    proba = getattr(model, "predict_proba", None)
    if proba is not None:
        try:
            y_proba = proba(X)[:, 1]
        except Exception:
            y_proba = None
    else:
        y_proba = None

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y, y_proba)
    else:
        metrics["roc_auc"] = 0.0

    cm = confusion_matrix(y, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    if prefix:
        metrics = {f"{prefix}_{k}" if k != "confusion_matrix" else k: v for k, v in metrics.items()}
    return metrics
