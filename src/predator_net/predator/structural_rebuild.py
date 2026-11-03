"""
Physical Structural Rebuild Module for PREDATOR NET.

CRITICAL RULE (Spec §28, Rule #7, #8):
  Never simply zero weights. Physically rebuild the adjacent Linear layers.
  Removing neuron w from hidden layer L reduces:
    - Layer L: weight shape (N, In) -> (N - 1, In), bias shape (N,) -> (N - 1,)
    - Layer L+1: weight shape (Out, N) -> (Out, N - 1), bias unchanged
  Resulting in physically smaller linear modules and strictly decreased parameter count.
"""

import logging
from typing import Tuple
import torch
import torch.nn as nn

from predator_net.models.dynamic_mlp import DynamicMLP

logger = logging.getLogger(__name__)


def remove_neuron_structurally(
    model: DynamicMLP,
    layer_idx: int,
    neuron_idx: int,
) -> DynamicMLP:
    """
    Physically remove neuron at index neuron_idx from hidden layer at layer_idx.

    Args:
        model: DynamicMLP model to modify in-place or rebuild.
        layer_idx: Target hidden layer index (0 to num_hidden_layers-1).
        neuron_idx: Index of neuron to be physically removed.

    Returns:
        The structurally rebuilt DynamicMLP model.
    """
    if layer_idx < 0 or layer_idx >= model.num_hidden_layers:
        raise ValueError(
            f"Invalid layer_idx {layer_idx}. Must be in [0, {model.num_hidden_layers - 1}]."
        )

    curr_dim = model.hidden_dims[layer_idx]
    if curr_dim <= 1:
        raise ValueError(
            f"Cannot remove neuron from layer {layer_idx} with only {curr_dim} neuron(s)."
        )
    if neuron_idx < 0 or neuron_idx >= curr_dim:
        raise IndexError(
            f"Invalid neuron_idx {neuron_idx} for layer with width {curr_dim}."
        )

    device = model.linears[0].weight.device

    # 1. Rebuild Layer L (curr_lin): remove row neuron_idx
    curr_lin = model.linears[layer_idx]
    in_feat = curr_lin.in_features
    out_feat = curr_lin.out_features  # curr_dim
    new_out_feat = out_feat - 1

    # Keep all row indices except neuron_idx
    row_keep = [i for i in range(out_feat) if i != neuron_idx]
    row_keep_tensor = torch.tensor(row_keep, dtype=torch.long, device=device)

    new_curr_lin = nn.Linear(in_feat, new_out_feat, bias=(curr_lin.bias is not None)).to(device)

    with torch.no_grad():
        new_w = curr_lin.weight.data.index_select(0, row_keep_tensor)
        new_curr_lin.weight.data.copy_(new_w)

        if curr_lin.bias is not None:
            new_b = curr_lin.bias.data.index_select(0, row_keep_tensor)
            new_curr_lin.bias.data.copy_(new_b)

    # 2. Rebuild Layer L + 1 (next_lin): remove column neuron_idx
    next_lin = model.linears[layer_idx + 1]
    next_in_feat = next_lin.in_features  # curr_dim
    next_out_feat = next_lin.out_features
    new_next_in_feat = next_in_feat - 1

    col_keep = [i for i in range(next_in_feat) if i != neuron_idx]
    col_keep_tensor = torch.tensor(col_keep, dtype=torch.long, device=device)

    new_next_lin = nn.Linear(new_next_in_feat, next_out_feat, bias=(next_lin.bias is not None)).to(device)

    with torch.no_grad():
        new_next_w = next_lin.weight.data.index_select(1, col_keep_tensor)
        new_next_lin.weight.data.copy_(new_next_w)

        if next_lin.bias is not None:
            new_next_lin.bias.data.copy_(next_lin.bias.data)

    # 3. Update model layers and hidden dimension tracking
    model.linears[layer_idx] = new_curr_lin
    model.linears[layer_idx + 1] = new_next_lin
    model.hidden_dims[layer_idx] = new_out_feat

    logger.debug(
        f"Structurally removed neuron {neuron_idx} from layer {layer_idx}. "
        f"Layer width reduced: {curr_dim} -> {new_out_feat}."
    )
    return model
