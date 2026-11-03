"""
Knowledge Distillation Module for PREDATOR-D.

Per Spec §32:
  Loss function combining ground-truth binary cross-entropy with teacher soft predictions:
    L_total = (1 - λ) * L_classification + λ * L_distillation
  Temperature T scales teacher and student logits.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Combined classification and distillation loss for PREDATOR-D student models.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        alpha_distill: float = 0.30,
        pos_weight: Optional[float] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.alpha_distill = float(alpha_distill)

        if pos_weight is not None and pos_weight > 0:
            pw_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
            self.cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
        else:
            self.cls_loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined loss.
        """
        # Hard label classification loss
        cls_loss = self.cls_loss_fn(student_logits, targets)

        # Distillation loss on soft probabilities
        soft_student = torch.sigmoid(student_logits / self.temperature)
        soft_teacher = torch.sigmoid(teacher_logits / self.temperature).detach()

        distill_loss = F.binary_cross_entropy(
            soft_student,
            soft_teacher,
            reduction="mean",
        ) * (self.temperature ** 2)

        total_loss = (1.0 - self.alpha_distill) * cls_loss + self.alpha_distill * distill_loss
        return total_loss
