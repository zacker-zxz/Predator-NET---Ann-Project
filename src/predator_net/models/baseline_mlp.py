"""
Baseline MLP Architecture for PREDATOR NET.

Per Spec §13, §14:
  - Input: input_dim (dynamically determined from features)
  - Hidden layers: Linear(input_dim, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, 1)
  - No BatchNorm or Dropout initially.
  - Output is raw logit. Sigmoid is applied during inference.
"""

from typing import List, Tuple
import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    """
    Standard Feed-Forward Artificial Neural Network for Network Intrusion Detection.
    Provides a stable reference architecture for evaluating neuron absorption.
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

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        # Final classification head outputting raw logit
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw logits of shape (batch_size, output_dim).
        """
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute probabilities using sigmoid activation on logits.
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Binary prediction based on probability threshold.
        1 = ATTACK, 0 = BENIGN.
        """
        probs = self.predict_proba(x)
        return (probs >= threshold).long()
