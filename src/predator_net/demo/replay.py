"""
SOC Live Flow Replay Simulator for PREDATOR NET.

Implements Spec §51:
  Streams test network flows one-by-one with simulated delays.
  Displays:
    - Protocol, Dst Port, Duration, Packets, Bytes
    - Prediction (⚠ ATTACK / ✓ BENIGN)
    - Confidence %
    - Active vs Absorbed Neurons
"""

import sys
import time
import logging
from typing import Optional
import numpy as np
import torch
import torch.nn as nn

from predator_net.models.dynamic_mlp import DynamicMLP

logger = logging.getLogger(__name__)

# Ensure stdout handles UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_soc_replay(
    model: DynamicMLP,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
    max_flows: int = 20,
    delay_sec: float = 0.3,
    initial_neurons: int = 224,
) -> None:
    """
    Stream and display simulated network flow events per Spec §51.
    """
    model.eval()
    device = model.linears[0].weight.device
    active_neurons = sum(model.hidden_dims)
    absorbed_neurons = max(0, initial_neurons - active_neurons)

    print("\n" + "=" * 52)
    print("      [PREDATOR NET] LIVE SOC FLOW MONITOR")
    print("        (Simulated Flow Stream Replay)")
    print("=" * 52)

    correct_predictions = 0

    for i in range(min(max_flows, len(X_test))):
        flow_features = X_test[i]
        true_label = int(y_test[i])

        # Inference
        x_tensor = torch.from_numpy(flow_features).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(x_tensor)
            prob = float(torch.sigmoid(logit).cpu().item())

        is_attack = prob >= threshold
        pred_label = 1 if is_attack else 0
        if pred_label == true_label:
            correct_predictions += 1

        confidence = prob * 100.0 if is_attack else (1.0 - prob) * 100.0
        status_icon = "[!] ATTACK" if is_attack else "[+] BENIGN"
        gt_icon = "ATTACK" if true_label == 1 else "BENIGN"

        print(f"\n+-- [FLOW #{i+1:04d}] " + "-" * 32)
        print(f"| Prediction:      {status_icon} (Confidence: {confidence:5.1f}%)")
        print(f"| Ground Truth:    {gt_icon}")
        print(f"| Active Neurons:  {active_neurons}")
        print(f"| Absorbed:        {absorbed_neurons}")
        print("+" + "-" * 45)

        time.sleep(delay_sec)

    accuracy = (correct_predictions / min(max_flows, len(X_test))) * 100.0
    print("\n" + "=" * 52)
    print(f" Replay Complete: {min(max_flows, len(X_test))} flows processed | Accuracy: {accuracy:.1f}%")
    print("=" * 52 + "\n")
