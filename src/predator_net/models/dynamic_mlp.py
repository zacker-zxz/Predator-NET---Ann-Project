"""
Dynamic MLP Architecture for PREDATOR NET.

Per Spec §28, §29:
  - Uses nn.ModuleList to support dynamically changing hidden layer widths.
  - Can be initialized from a list of hidden dimensions or converted from BaselineMLP.
  - Enables physical neuron removal and structural rebuilding without changing external interfaces.
"""

from typing import List, Tuple, Dict, Any, Optional
import copy
import torch
import torch.nn as nn

from predator_net.models.baseline_mlp import BaselineMLP


class DynamicMLP(nn.Module):
    """
    Feed-Forward Neural Network with dynamically configurable hidden dimensions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim

        self.linears = nn.ModuleList()
        prev_dim = input_dim

        for h_dim in self.hidden_dims:
            self.linears.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim

        # Final classification head
        self.linears.append(nn.Linear(prev_dim, output_dim))
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all linear layers with ReLU activations."""
        h = x
        for i in range(len(self.linears) - 1):
            h = self.activation(self.linears[i](h))
        # Final layer: raw logit (no activation)
        logits = self.linears[-1](h)
        return logits

    def forward_with_activations(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass returning both final logits and hidden layer post-activation tensors.
        Useful for neuron importance and activation cosine similarity computation.
        """
        hidden_acts = []
        h = x
        for i in range(len(self.linears) - 1):
            h = self.activation(self.linears[i](h))
            hidden_acts.append(h)
        logits = self.linears[-1](h)
        return logits, hidden_acts

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute attack probabilities using sigmoid activation."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Binary predictions."""
        probs = self.predict_proba(x)
        return (probs >= threshold).long()

    @property
    def num_hidden_layers(self) -> int:
        """Number of hidden layers (excluding input and output)."""
        return len(self.hidden_dims)

    def get_layer_shape(self, layer_idx: int) -> Tuple[int, int]:
        """Get (in_features, out_features) for linear layer at index layer_idx."""
        l = self.linears[layer_idx]
        return l.in_features, l.out_features

    def clone(self) -> "DynamicMLP":
        """Deep copy the model architecture and current weights."""
        return copy.deepcopy(self)

    @classmethod
    def from_baseline(cls, baseline: BaselineMLP) -> "DynamicMLP":
        """
        Convert a BaselineMLP into an equivalent DynamicMLP preserving all weights.
        """
        dynamic = cls(
            input_dim=baseline.input_dim,
            hidden_dims=baseline.hidden_dims,
            output_dim=baseline.output_dim,
        )

        # Extract linears from baseline.net
        baseline_linears = [m for m in baseline.net if isinstance(m, nn.Linear)]
        assert len(baseline_linears) == len(dynamic.linears)

        for b_lin, d_lin in zip(baseline_linears, dynamic.linears):
            d_lin.weight.data.copy_(b_lin.weight.data)
            d_lin.bias.data.copy_(b_lin.bias.data)

        return dynamic
