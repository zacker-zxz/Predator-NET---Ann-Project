"""
Model utilities for parameter counting, architecture inspection, and integrity checks.
"""

from typing import Dict, List, Any, Tuple
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters in the model.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "non_trainable_parameters": int(non_trainable),
    }


def calculate_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Calculate physical model parameter size in kilobytes and megabytes (float32).
    """
    param_count = sum(p.numel() for p in model.parameters())
    bytes_size = param_count * 4  # 4 bytes per float32
    return {
        "size_bytes": int(bytes_size),
        "size_kb": float(bytes_size / 1024),
        "size_mb": float(bytes_size / (1024 * 1024)),
    }


def get_layer_dimensions(model: nn.Module) -> List[int]:
    """
    Extract the sequence of layer dimensions (input -> hidden... -> output).
    Works on BaselineMLP and DynamicMLP models.
    """
    dims = []
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linears:
        return dims

    dims.append(linears[0].in_features)
    for l in linears:
        dims.append(l.out_features)
    return dims


def verify_model_integrity(model: nn.Module, sample_input: torch.Tensor) -> Tuple[bool, str]:
    """
    Check for NaN, Inf, dimension mismatches, or exploding activations (Spec §60).
    """
    try:
        model.eval()
        with torch.no_grad():
            output = model(sample_input)

        if torch.isnan(output).any():
            return False, "Output contains NaN values"
        if torch.isinf(output).any():
            return False, "Output contains Infinite values"
        if output.shape[-1] != 1:
            return False, f"Unexpected output dimension: {output.shape}"
        if (output.abs() > 1e6).any():
            return False, "Exploding logits detected (> 1e6)"

        return True, "Model passed integrity check"
    except Exception as e:
        return False, f"Integrity check failed with error: {str(e)}"
