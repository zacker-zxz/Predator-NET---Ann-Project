"""
Model Trainer for PREDATOR NET.

Encapsulates the PyTorch training, validation, early stopping, and history logging.
Per Spec §16:
  - Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
  - Loss: BCEWithLogitsLoss with pos_weight
  - Checkpointing best model weights on validation F1
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from predator_net.evaluation.metrics import compute_metrics
from predator_net.training.callbacks import EarlyStopping

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Execute one training epoch."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(x_batch)
        total_samples += len(x_batch)

    return float(running_loss / max(total_samples, 1))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[float, Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Evaluate model performance on a dataset.

    Returns:
        Tuple of (average_loss, metrics_dict, y_true_array, y_prob_array).
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            probs = torch.sigmoid(logits)

            running_loss += loss.item() * len(x_batch)
            total_samples += len(x_batch)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    avg_loss = float(running_loss / max(total_samples, 1))
    y_prob = np.vstack(all_probs).ravel()
    y_true = np.vstack(all_targets).ravel()

    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    return avg_loss, metrics, y_true, y_prob


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int = 50,
    early_stopping: Optional[EarlyStopping] = None,
    device: str = "cpu",
    scheduler: Optional[Any] = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Full training pipeline across epochs with validation evaluation and early stopping.
    """
    dev = torch.device(device)
    model.to(dev)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_recall": [],
        "val_precision": [],
    }

    logger.info(f"Starting model training for up to {epochs} epochs on device '{device}'.")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, dev)
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, loss_fn, dev)

        val_f1 = val_metrics["f1"]
        val_rec = val_metrics["recall"]
        val_prec = val_metrics["precision"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_recall"].append(val_rec)
        history["val_precision"].append(val_prec)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Val Recall: {val_rec:.4f} | "
            f"Val Precision: {val_prec:.4f}"
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        if early_stopping is not None:
            stop_metric = val_f1 if early_stopping.monitor == "val_f1" else val_loss
            should_stop = early_stopping(stop_metric, model, epoch)
            if should_stop:
                break

    if early_stopping is not None and early_stopping.best_state_dict is not None:
        early_stopping.restore_best_weights(model)

    return model, history
