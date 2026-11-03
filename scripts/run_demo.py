"""
PHASE 11 SCRIPT: Launch PREDATOR NET Demo (Terminal Replay or Streamlit Dashboard).

Usage:
    python scripts/run_demo.py [--mode terminal|dashboard] [--flows 20]
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch

from predator_net.models.dynamic_mlp import DynamicMLP
from predator_net.demo.replay import run_soc_replay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_demo")


def main() -> None:
    parser = argparse.ArgumentParser(description="PREDATOR NET — Phase 11: Demo Launcher")
    parser.add_argument(
        "--mode",
        type=str,
        default="terminal",
        choices=["terminal", "dashboard"],
        help="Demo mode: 'terminal' for CLI SOC flow replay, 'dashboard' for Streamlit web UI",
    )
    parser.add_argument(
        "--flows",
        type=int,
        default=15,
        help="Number of network flows to simulate in terminal replay",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.15,
        help="Interval between flows in seconds",
    )
    args = parser.parse_args()

    if args.mode == "dashboard":
        logger.info("Launching Streamlit SOC Dashboard...")
        dashboard_path = PROJECT_ROOT / "src" / "predator_net" / "demo" / "dashboard.py"
        subprocess.run(["streamlit", "run", str(dashboard_path)])
    else:
        logger.info("Launching Terminal SOC Replay Simulator...")
        pred_dir = PROJECT_ROOT / "artifacts" / "predator"
        processed_dir = PROJECT_ROOT / "data" / "processed"

        comp_ckpt_path = pred_dir / "predator_compressed.pt"
        if not comp_ckpt_path.exists():
            comp_ckpt_path = PROJECT_ROOT / "artifacts" / "models" / "baseline_best.pt"

        test_x_path = processed_dir / "test_X.npy"
        test_y_path = processed_dir / "test_y.npy"

        if not (test_x_path.exists() and comp_ckpt_path.exists()):
            logger.error("Processed test data or model checkpoint missing.")
            sys.exit(1)

        ckpt = torch.load(comp_ckpt_path, map_location="cpu", weights_only=False)
        model = DynamicMLP(input_dim=ckpt["input_dim"], hidden_dims=ckpt["hidden_dims"])
        model.load_state_dict(ckpt["model_state_dict"])

        X_test = np.load(test_x_path)
        y_test = np.load(test_y_path)
        threshold = ckpt.get("threshold", 0.5)

        run_soc_replay(
            model=model,
            X_test=X_test,
            y_test=y_test,
            threshold=threshold,
            max_flows=args.flows,
            delay_sec=args.delay,
        )


if __name__ == "__main__":
    main()
