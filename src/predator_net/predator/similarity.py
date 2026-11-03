"""
Neuron Compatibility and Cosine Similarity System for PREDATOR NET.

Implements Spec §24, §25:
  - Activation profile similarity: cosine_similarity(H_w, H_s) across calibration samples.
  - Incoming weight similarity: cosine_similarity(V_w, V_s).
  - Combined compatibility: β * act_sim + (1-β) * weight_sim (β=0.7 default).
  - Gated candidate selection: requires compatibility >= minimum_similarity (default 0.70).
  - Explicit rejection status if no compatible neuron meets the threshold.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from predator_net.models.dynamic_mlp import DynamicMLP

logger = logging.getLogger(__name__)


def compute_activation_profiles(
    model: DynamicMLP,
    loader: DataLoader,
    layer_idx: int,
    device: torch.device,
    max_samples: Optional[int] = 5000,
) -> np.ndarray:
    """
    Extract post-ReLU activation profiles for all neurons in a specific hidden layer.

    Args:
        model: DynamicMLP instance.
        loader: DataLoader for calibration samples.
        layer_idx: Index of hidden layer (0 to num_hidden_layers-1).
        device: Torch device.
        max_samples: Maximum samples to accumulate.

    Returns:
        NumPy array of shape (N_samples, H_dim) containing neuron activations.
    """
    model.eval()
    activations_list = []
    total = 0

    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            _, hidden_acts = model.forward_with_activations(x_batch)
            act = hidden_acts[layer_idx].detach().cpu().numpy()
            activations_list.append(act)
            total += len(x_batch)
            if max_samples is not None and total >= max_samples:
                break

    all_acts = np.vstack(activations_list)
    if max_samples is not None and len(all_acts) > max_samples:
        all_acts = all_acts[:max_samples]
    return all_acts


def cosine_similarity_1d(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-8) -> float:
    """Compute cosine similarity between two 1D vectors."""
    dot = float(np.dot(v1, v2))
    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))
    denom = (norm1 * norm2) + eps
    sim = dot / denom
    return float(np.clip(sim, -1.0, 1.0))


def compute_compatibility(
    model: DynamicMLP,
    layer_idx: int,
    weak_idx: int,
    strong_idx: int,
    act_profiles: np.ndarray,
    beta: float = 0.70,
) -> Tuple[float, float, float]:
    """
    Compute compatibility score between a weak neuron and a strong neuron candidate.

    Args:
        model: DynamicMLP model.
        layer_idx: Index of target hidden layer.
        weak_idx: Index of weak neuron w in target layer.
        strong_idx: Index of strong neuron candidate s in target layer.
        act_profiles: Activation matrix (N_samples, H_dim).
        beta: Weight for activation similarity (default 0.70).

    Returns:
        Tuple of (combined_compatibility, act_similarity, weight_similarity).
    """
    # 1. Activation cosine similarity: columns from act_profiles
    h_w = act_profiles[:, weak_idx]
    h_s = act_profiles[:, strong_idx]
    act_sim = cosine_similarity_1d(h_w, h_s)

    # 2. Incoming weight cosine similarity: rows from current layer weight matrix
    # lin.weight has shape (out_features, in_features)
    lin = model.linears[layer_idx]
    v_w = lin.weight[weak_idx].detach().cpu().numpy()
    v_s = lin.weight[strong_idx].detach().cpu().numpy()
    weight_sim = cosine_similarity_1d(v_w, v_s)

    # Combined compatibility
    compatibility = beta * act_sim + (1.0 - beta) * weight_sim
    return float(compatibility), float(act_sim), float(weight_sim)


def find_best_compatible_neuron(
    model: DynamicMLP,
    layer_idx: int,
    weak_idx: int,
    strong_indices: List[int],
    act_profiles: np.ndarray,
    beta: float = 0.70,
    minimum_similarity: float = 0.70,
) -> Dict[str, Any]:
    """
    Find the most compatible strong neuron for weak neuron absorption (Spec §24, §25).

    Returns:
        Dictionary with:
          - "accepted": bool
          - "strong_idx": best strong index or None
          - "compatibility": score
          - "act_sim": score
          - "weight_sim": score
          - "reason": decision reason
    """
    if not strong_indices:
        return {
            "accepted": False,
            "strong_idx": None,
            "compatibility": 0.0,
            "act_sim": 0.0,
            "weight_sim": 0.0,
            "reason": "rejected_no_strong_candidates",
        }

    # Filter out weak_idx if accidentally present
    candidates = [s for s in strong_indices if s != weak_idx]
    if not candidates:
        return {
            "accepted": False,
            "strong_idx": None,
            "compatibility": 0.0,
            "act_sim": 0.0,
            "weight_sim": 0.0,
            "reason": "rejected_no_distinct_candidates",
        }

    best_score = -2.0
    best_candidate = None
    best_act_sim = 0.0
    best_weight_sim = 0.0

    for s_idx in candidates:
        comp, a_sim, w_sim = compute_compatibility(
            model=model,
            layer_idx=layer_idx,
            weak_idx=weak_idx,
            strong_idx=s_idx,
            act_profiles=act_profiles,
            beta=beta,
        )
        if comp > best_score:
            best_score = comp
            best_candidate = s_idx
            best_act_sim = a_sim
            best_weight_sim = w_sim

    if best_score < minimum_similarity:
        return {
            "accepted": False,
            "strong_idx": best_candidate,
            "compatibility": best_score,
            "act_sim": best_act_sim,
            "weight_sim": best_weight_sim,
            "reason": "rejected_no_compatible_neuron",
        }

    return {
        "accepted": True,
        "strong_idx": best_candidate,
        "compatibility": best_score,
        "act_sim": best_act_sim,
        "weight_sim": best_weight_sim,
        "reason": "compatible_neuron_found",
    }
