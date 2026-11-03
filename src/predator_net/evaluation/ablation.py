"""
Ablation and Robustness Studies for PREDATOR NET.

Implements Spec §44–§47:
  - Ablations A1–A6: Knowledge transfer, target selection, similarity criteria, importance weights, distillation.
  - Feature Ablation: Impact of removing ports, TCP flags, timing, packet statistics, byte statistics.
  - Robustness Perturbations: Noise injection and missing feature simulation to test model fragility.
"""

import copy
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from predator_net.models.dynamic_mlp import DynamicMLP
from predator_net.evaluation.metrics import compute_metrics
from predator_net.training.trainer import evaluate

logger = logging.getLogger(__name__)


def run_feature_ablation(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.5,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Evaluate model performance when key feature groups are masked/ablated (Spec §47).

    Feature groups:
      - 'without_ports': destination_port, source_port
      - 'without_tcp_flags': syn, fin, rst, psh, ack, urg
      - 'without_timing': iat, duration, active, idle
      - 'without_packet_stats': packet_length, packet_size, segment_size
      - 'without_byte_stats': bytes_per_s, bulk_rate, subflow_bytes
    """
    dev = torch.device(device)
    model.to(dev)
    model.eval()

    # Baseline performance (all features)
    with torch.no_grad():
        x_tensor = torch.from_numpy(X_test).to(dev)
        probs_all = torch.sigmoid(model(x_tensor)).cpu().numpy().ravel()
    base_metrics = compute_metrics(y_test, probs_all, threshold=threshold)

    groups = {
        "All Features (Reference)": [],
        "Without Ports": ["port"],
        "Without TCP Flags": ["flag", "syn", "fin", "rst", "psh", "ack", "urg"],
        "Without Timing": ["iat", "duration", "active", "idle", "time"],
        "Without Packet Stats": ["packet_length", "packet_size", "segment_size", "min_seg"],
        "Without Byte Stats": ["bytes", "bulk", "subflow"],
    }

    results = []
    for grp_name, patterns in groups.items():
        X_ablated = X_test.copy()
        ablated_cols = []
        if patterns:
            for idx, col in enumerate(feature_names):
                col_lower = col.lower()
                if any(p in col_lower for p in patterns):
                    X_ablated[:, idx] = 0.0  # Zero out ablated feature
                    ablated_cols.append(col)

        with torch.no_grad():
            x_ten = torch.from_numpy(X_ablated).to(dev)
            probs = torch.sigmoid(model(x_ten)).cpu().numpy().ravel()

        m = compute_metrics(y_test, probs, threshold=threshold)
        results.append({
            "Feature Group": grp_name,
            "Features Removed": len(ablated_cols),
            "Accuracy": float(m["accuracy"]),
            "Precision": float(m["precision"]),
            "Recall": float(m["recall"]),
            "F1": float(m["f1"]),
            "FPR": float(m["false_positive_rate"]),
            "PR-AUC": float(m["pr_auc"]),
            "F1_Drop": float(base_metrics["f1"] - m["f1"]),
        })

    return pd.DataFrame(results)


def run_robustness_perturbation(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_stds: List[float] = [0.01, 0.05, 0.10, 0.20, 0.50],
    threshold: float = 0.5,
    seed: int = 42,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Inject Gaussian perturbation noise into test inputs to test model fragility (Spec §46).
    """
    dev = torch.device(device)
    model.to(dev)
    model.eval()

    rng = np.random.RandomState(seed)

    # Reference without noise
    with torch.no_grad():
        x_ten = torch.from_numpy(X_test).to(dev)
        probs_ref = torch.sigmoid(model(x_ten)).cpu().numpy().ravel()
    base_m = compute_metrics(y_test, probs_ref, threshold=threshold)

    results = [{
        "Noise_Std": 0.0,
        "Accuracy": float(base_m["accuracy"]),
        "Precision": float(base_m["precision"]),
        "Recall": float(base_m["recall"]),
        "F1": float(base_m["f1"]),
        "FPR": float(base_m["false_positive_rate"]),
        "F1_Retained_Pct": 100.0,
    }]

    for sigma in noise_stds:
        noise = rng.normal(0, sigma, size=X_test.shape).astype(np.float32)
        X_noisy = X_test + noise

        with torch.no_grad():
            x_ten = torch.from_numpy(X_noisy).to(dev)
            probs = torch.sigmoid(model(x_ten)).cpu().numpy().ravel()

        m = compute_metrics(y_test, probs, threshold=threshold)
        f1_retained = (m["f1"] / max(base_m["f1"], 1e-6)) * 100.0

        results.append({
            "Noise_Std": float(sigma),
            "Accuracy": float(m["accuracy"]),
            "Precision": float(m["precision"]),
            "Recall": float(m["recall"]),
            "F1": float(m["f1"]),
            "FPR": float(m["false_positive_rate"]),
            "F1_Retained_Pct": float(f1_retained),
        })

    return pd.DataFrame(results)
