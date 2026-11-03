"""
Visualization and Plotting Utilities for PREDATOR NET.

Generates publication-quality figures:
  - Confusion matrix heatmap
  - Training and validation loss / F1 curves
  - ROC curve with AUC score
  - Precision-Recall curve with PR-AUC score
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Path,
    labels: List[str] = ["BENIGN", "ATTACK"],
    title: str = "Confusion Matrix",
) -> None:
    """Save formatted confusion matrix heatmap."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
    )
    plt.title(title, fontsize=14, pad=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Path,
    title: str = "Training & Validation Curves",
) -> None:
    """Plot Loss and Validation Metric trajectories over epochs."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color = "tab:red"
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", color=color, fontsize=12)
    if "train_loss" in history:
        ax1.plot(epochs, history["train_loss"], label="Train Loss", color=color, linestyle="-")
    if "val_loss" in history:
        ax1.plot(epochs, history["val_loss"], label="Val Loss", color=color, linestyle="--")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Validation F1 Score", color=color, fontsize=12)
    if "val_f1" in history:
        ax2.plot(epochs, history["val_f1"], label="Val F1", color=color, linewidth=2)
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(title, fontsize=14, pad=12)
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    roc_auc: float,
    save_path: Path,
) -> None:
    """Plot Receiver Operating Characteristic (ROC) curve."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=14, pad=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pr_auc: float,
    save_path: Path,
) -> None:
    """Plot Precision-Recall curve."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color="purple", lw=2, label=f"PR curve (AUC = {pr_auc:.4f})")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall (PR) Curve", fontsize=14, pad=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
