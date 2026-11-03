"""
Deterministic seed management for reproducible experiments.

Sets seeds for Python, NumPy, and PyTorch (CPU + CUDA).
Spec §57: Every experiment should use fixed, recorded seeds.
"""

import os
import random
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """
    Set deterministic seeds across all relevant libraries.

    Args:
        seed: Integer seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enable deterministic algorithms where practical.
    # Note: Some operations may not have deterministic implementations,
    # so we set warn_only=True to avoid hard failures.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch >= 1.8 deterministic flag
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Older PyTorch versions don't support warn_only
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info(f"Global seed set to {seed}")


def get_generator(seed: int = DEFAULT_SEED) -> torch.Generator:
    """
    Create a seeded torch Generator for DataLoader workers, etc.

    Args:
        seed: Integer seed value.

    Returns:
        A torch.Generator initialized with the given seed.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
