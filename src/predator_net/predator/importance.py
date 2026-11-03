"""
Neuron Importance Scoring Module for PREDATOR NET.

Implements Spec §20–§23:
  - First-order contribution: mean(|a_j * ∂L/∂a_j|) over calibration batches.
  - Secondary signal: outgoing weight magnitude ||U_j||_1, normalized.
  - Combined score: α * normalized_gradient + (1-α) * normalized_weight (α=0.7 default).
  - Weak/Strong candidate classification via configurable policies (percentile, relative, hybrid).
  - Exports neuron_importance.csv per Spec §22.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from predator_net.models.dynamic_mlp import DynamicMLP

logger = logging.getLogger(__name__)


def compute_layer_neuron_importance(
    model: DynamicMLP,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    alpha: float = 0.70,
    max_samples: Optional[int] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Compute activation-gradient and outgoing weight importance for all hidden layers.

    Args:
        model: DynamicMLP instance.
        loader: Calibration DataLoader.
        loss_fn: BCEWithLogitsLoss.
        device: Torch device.
        alpha: Weight for gradient importance (default 0.70).
        max_samples: Optional cap on calibration samples processed.

    Returns:
        Dict mapping layer_idx (0 to num_hidden_layers-1) to dict of:
          - "act_imp": raw activation-gradient importance array (H_dim,)
          - "weight_imp": raw outgoing weight magnitude array (H_dim,)
          - "combined_imp": combined normalized importance array (H_dim,)
    """
    model.eval()
    num_hidden = model.num_hidden_layers

    # Track cumulative |a_j * dL/da_j| per hidden layer
    act_grad_sums = [np.zeros(model.hidden_dims[i], dtype=np.float64) for i in range(num_hidden)]
    total_samples = 0

    for x_batch, y_batch in loader:
        if max_samples is not None and total_samples >= max_samples:
            break

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        batch_size = len(x_batch)

        # Forward pass tracking hidden activations with grad
        h = x_batch
        hidden_acts = []
        for i in range(num_hidden):
            lin = model.linears[i]
            z = lin(h)
            a = model.activation(z)
            a.retain_grad()
            hidden_acts.append(a)
            h = a

        logits = model.linears[-1](h)
        loss = loss_fn(logits, y_batch)

        model.zero_grad()
        loss.backward()

        # Compute |a_j * dL/da_j| for each hidden layer
        for i in range(num_hidden):
            a = hidden_acts[i]
            if a.grad is not None:
                contrib = (a * a.grad).abs().detach().cpu().numpy()
                act_grad_sums[i] += contrib.sum(axis=0)

        total_samples += batch_size

    # Average activation-gradient importance
    results = {}
    for i in range(num_hidden):
        h_dim = model.hidden_dims[i]
        act_imp = act_grad_sums[i] / max(total_samples, 1)

        # Outgoing weights: from next linear layer
        next_lin = model.linears[i + 1]
        next_weight = next_lin.weight.detach().cpu().numpy()  # shape: (out_features, h_dim)
        weight_imp = np.sum(np.abs(next_weight), axis=0)  # shape: (h_dim,)

        # Normalize both metrics to [0, 1] per layer
        def _minmax(arr: np.ndarray) -> np.ndarray:
            denom = arr.max() - arr.min()
            if denom > 1e-8:
                return (arr - arr.min()) / denom
            return np.ones_like(arr) * 0.5

        norm_act = _minmax(act_imp)
        norm_weight = _minmax(weight_imp)

        combined_imp = alpha * norm_act + (1.0 - alpha) * norm_weight

        results[i] = {
            "act_imp": act_imp.astype(np.float32),
            "weight_imp": weight_imp.astype(np.float32),
            "norm_act": norm_act.astype(np.float32),
            "norm_weight": norm_weight.astype(np.float32),
            "combined_imp": combined_imp.astype(np.float32),
        }

    return results


def classify_neurons(
    combined_scores: np.ndarray,
    policy: str = "percentile",
    weak_percentile: float = 15.0,
    strong_percentile: float = 30.0,
    weak_factor: float = 0.30,
) -> List[str]:
    """
    Classify neurons into 'weak', 'strong', or 'normal' according to policy (Spec §23).

    Args:
        combined_scores: 1D array of combined importance scores for a layer.
        policy: 'percentile', 'relative', or 'hybrid'.
        weak_percentile: Bottom X% are weak candidates (default 15).
        strong_percentile: Top X% are strong candidates (default 30).
        weak_factor: Factor for relative policy (score < median * factor).

    Returns:
        List of status strings ('weak', 'strong', 'normal') for each neuron.
    """
    n = len(combined_scores)
    statuses = ["normal"] * n

    if policy.lower() == "percentile":
        weak_cutoff = np.percentile(combined_scores, weak_percentile)
        strong_cutoff = np.percentile(combined_scores, 100.0 - strong_percentile)

        for i, score in enumerate(combined_scores):
            if score <= weak_cutoff:
                statuses[i] = "weak"
            elif score >= strong_cutoff:
                statuses[i] = "strong"

    elif policy.lower() == "relative":
        med = np.median(combined_scores)
        weak_cutoff = med * weak_factor
        strong_cutoff = np.percentile(combined_scores, 100.0 - strong_percentile)

        for i, score in enumerate(combined_scores):
            if score < weak_cutoff:
                statuses[i] = "weak"
            elif score >= strong_cutoff:
                statuses[i] = "strong"

    else:  # hybrid
        weak_cutoff = np.percentile(combined_scores, weak_percentile)
        med = np.median(combined_scores)
        effective_weak = min(weak_cutoff, med * weak_factor)
        strong_cutoff = np.percentile(combined_scores, 100.0 - strong_percentile)

        for i, score in enumerate(combined_scores):
            if score <= effective_weak:
                statuses[i] = "weak"
            elif score >= strong_cutoff:
                statuses[i] = "strong"

    return statuses


def generate_importance_report(
    model: DynamicMLP,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    alpha: float = 0.70,
    policy: str = "percentile",
    weak_percentile: float = 15.0,
    strong_percentile: float = 30.0,
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compute importance scores across all hidden layers and generate report DataFrame.
    Saves to artifacts/predator/neuron_importance.csv per Spec §22.
    """
    layer_results = compute_layer_neuron_importance(
        model=model,
        loader=loader,
        loss_fn=loss_fn,
        device=device,
        alpha=alpha,
    )

    records = []
    for layer_idx, data in layer_results.items():
        combined = data["combined_imp"]
        statuses = classify_neurons(
            combined,
            policy=policy,
            weak_percentile=weak_percentile,
            strong_percentile=strong_percentile,
        )

        # Ranks: 1 = highest importance
        order = np.argsort(-combined)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(combined) + 1)

        layer_name = f"layer_{layer_idx + 1}"
        for n_idx in range(len(combined)):
            records.append({
                "layer": layer_name,
                "neuron_index": int(n_idx),
                "activation_importance": float(data["act_imp"][n_idx]),
                "weight_importance": float(data["weight_imp"][n_idx]),
                "combined_importance": float(combined[n_idx]),
                "rank": int(ranks[n_idx]),
                "status": statuses[n_idx],
            })

    df = pd.DataFrame(records)

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved neuron importance report to {output_csv}")

    return df
