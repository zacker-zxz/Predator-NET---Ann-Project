"""
Neuron Absorption Operators for PREDATOR NET.

Implements Spec §26 (PREDATOR-LIGHT) and Spec §27 (PREDATOR-FULL):
  - Calculates absorption coefficient α = clip(compatibility * imp_w / (imp_s + ε), α_min, α_max).
  - Transfers outgoing weights from weak neuron w to compatible strong neuron s.
  - In FULL mode, also transfers incoming weights and biases with coefficient γ.
"""

from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn

from predator_net.models.dynamic_mlp import DynamicMLP


def compute_alpha(
    compatibility: float,
    imp_w: float,
    imp_s: float,
    alpha_min: float = 0.05,
    alpha_max: float = 0.50,
    epsilon: float = 1e-8,
) -> float:
    """
    Calculate outgoing weight absorption coefficient per Spec §26:
      α = clip(compatibility * imp_w / (imp_s + ε), α_min, α_max)
    """
    ratio = imp_w / (imp_s + epsilon)
    raw_alpha = compatibility * ratio
    alpha = float(np.clip(raw_alpha, alpha_min, alpha_max))
    return alpha


def apply_absorption_weights(
    model: DynamicMLP,
    layer_idx: int,
    weak_idx: int,
    strong_idx: int,
    alpha: float,
    mode: str = "light",
    gamma: float = 0.30,
) -> None:
    """
    Perform weight absorption in-place on the model parameters before structural removal.

    Args:
        model: DynamicMLP model.
        layer_idx: Index of hidden layer containing weak and strong neurons.
        weak_idx: Index of weak neuron w.
        strong_idx: Index of strong neuron s.
        alpha: Outgoing weight transfer coefficient.
        mode: "light" (outgoing only) or "full" (outgoing + incoming).
        gamma: Incoming weight absorption coefficient for PREDATOR-FULL.
    """
    with torch.no_grad():
        # 1. Outgoing weights: layer_idx + 1 (columns weak_idx and strong_idx)
        next_lin = model.linears[layer_idx + 1]
        # next_lin.weight shape: (out_features, in_features)
        # Column j represents outgoing weights of neuron j in layer_idx
        u_w = next_lin.weight[:, weak_idx]
        u_s = next_lin.weight[:, strong_idx]
        # U_s_new = U_s + α * U_w
        u_s_new = u_s + alpha * u_w
        next_lin.weight[:, strong_idx].copy_(u_s_new)

        # 2. PREDATOR-FULL: incoming weights & bias transfer (Spec §27)
        if mode.lower() == "full":
            curr_lin = model.linears[layer_idx]
            # curr_lin.weight shape: (out_features, in_features)
            # Row j represents incoming weights of neuron j in layer_idx
            v_w = curr_lin.weight[weak_idx, :]
            v_s = curr_lin.weight[strong_idx, :]
            # V_s_new = (1 - γ) * V_s + γ * V_w
            v_s_new = (1.0 - gamma) * v_s + gamma * v_w
            curr_lin.weight[strong_idx, :].copy_(v_s_new)

            # Bias transfer: b_s_new = (1 - γ) * b_s + γ * b_w
            if curr_lin.bias is not None:
                b_w = curr_lin.bias[weak_idx]
                b_s = curr_lin.bias[strong_idx]
                b_s_new = (1.0 - gamma) * b_s + gamma * b_w
                curr_lin.bias[strong_idx].copy_(b_s_new)
