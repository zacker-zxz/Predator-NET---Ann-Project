"""
Training Callbacks for PREDATOR NET.

Includes EarlyStopping with configurable patience, minimum delta,
and automatic state checkpointing.
"""

import copy
import logging
from typing import Dict, Any, Optional
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping handler to terminate training when monitored metric stops improving.
    """

    def __init__(
        self,
        patience: int = 8,
        min_delta: float = 1e-4,
        monitor: str = "val_f1",
        mode: str = "max",
    ) -> None:
        """
        Args:
            patience: Number of epochs with no improvement before stopping.
            min_delta: Minimum change in monitored metric to qualify as improvement.
            monitor: Name of metric being monitored.
            mode: "max" (higher is better, e.g. F1) or "min" (lower is better, e.g. loss).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode.lower()
        if self.mode not in ["min", "max"]:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'min' or 'max'.")

        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.best_epoch: int = 0
        self.best_state_dict: Optional[Dict[str, Any]] = None

    def __call__(self, current_score: float, model: nn.Module, epoch: int) -> bool:
        """
        Evaluate current metric score and update early stopping state.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            self.best_state_dict = copy.deepcopy(model.state_dict())
            logger.info(f"Epoch {epoch}: Initial best {self.monitor} = {current_score:.4f}")
            return False

        if self.mode == "max":
            improved = current_score > (self.best_score + self.min_delta)
        else:
            improved = current_score < (self.best_score - self.min_delta)

        if improved:
            logger.info(
                f"Epoch {epoch}: {self.monitor} improved from {self.best_score:.4f} to {current_score:.4f}. Checkpointing model."
            )
            self.best_score = current_score
            self.best_epoch = epoch
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                f"Epoch {epoch}: {self.monitor} ({current_score:.4f}) did not improve over best ({self.best_score:.4f}). "
                f"Patience: {self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. Best epoch was {self.best_epoch} with {self.monitor} = {self.best_score:.4f}"
                )
                return True

        return False

    def restore_best_weights(self, model: nn.Module) -> None:
        """Restore model weights to the best recorded epoch."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            logger.info(f"Restored model weights from best epoch {self.best_epoch} ({self.monitor}={self.best_score:.4f})")
