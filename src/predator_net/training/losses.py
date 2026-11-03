"""
Loss functions for PREDATOR NET.

Per Spec §14, §15:
  - Uses BCEWithLogitsLoss directly on raw model logits.
  - Supports configurable pos_weight calculated from training class imbalance:
      pos_weight = negative_count / positive_count
"""

from typing import Optional, Union
import torch
import torch.nn as nn


def get_loss_function(
    pos_weight: Optional[float] = None,
    device: Union[str, torch.device] = "cpu",
) -> nn.BCEWithLogitsLoss:
    """
    Construct BCEWithLogitsLoss with optional positive class weighting.

    Args:
        pos_weight: Optional weight for the positive (attack) class.
                    pos_weight = negative_count / positive_count
        device: Torch device where the loss tensor should reside.

    Returns:
        Configured nn.BCEWithLogitsLoss instance.
    """
    if pos_weight is not None and pos_weight > 0:
        weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    return nn.BCEWithLogitsLoss()
