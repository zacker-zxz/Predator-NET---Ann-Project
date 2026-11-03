"""
Evaluation Metrics for PREDATOR NET.

Per Spec §17:
  Never report only accuracy.
  Cybersecurity Intrusion Detection requires:
    - Recall (Detection Rate)
    - Precision
    - F1 Score & Macro-F1
    - False Positive Rate (FPR)
    - False Negative Rate (FNR)
    - PR-AUC & ROC-AUC
    - Balanced Accuracy
    - Confusion Matrix breakdown
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute comprehensive IDS classification metrics.

    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class (attack).
        threshold: Classification cutoff (default 0.5).

    Returns:
        Dictionary of all metrics.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    # Basic metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # Curve AUCs (if both classes present)
    if len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = 0.5
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = 0.0
    else:
        roc_auc = 0.5
        pr_auc = 0.0

    # Confusion matrix breakdown
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Cybersecurity specific rates
    fpr = float(fp / max(fp + tn, 1))
    fnr = float(fn / max(fn + tp, 1))

    return {
        "threshold": float(threshold),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "total_samples": int(len(y_true)),
    }


def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    num_steps: int = 100,
) -> Tuple[float, float]:
    """
    Find optimal classification threshold on VALIDATION data (Spec §14).
    Never tune using test data.

    Args:
        y_true: True binary labels from validation set.
        y_prob: Predicted probabilities on validation set.
        metric: Optimization target ("f1", "balanced_accuracy", or "recall").
        num_steps: Grid search resolution between 0.05 and 0.95.

    Returns:
        Tuple of (best_threshold, best_metric_value).
    """
    thresholds = np.linspace(0.05, 0.95, num_steps)
    best_thresh = 0.5
    best_score = -1.0

    for t in thresholds:
        metrics = compute_metrics(y_true, y_prob, threshold=t)
        score = metrics.get(metric, metrics["f1"])
        if score > best_score:
            best_score = score
            best_thresh = float(t)

    return best_thresh, float(best_score)
