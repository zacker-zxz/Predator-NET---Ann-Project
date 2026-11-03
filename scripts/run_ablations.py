"""
PHASE 10 SCRIPT: Run Ablation Studies & Generate Research Visualizations.

Usage:
    python scripts/run_ablations.py [--config config/experiments.yaml]

This script:
  1. Executes Feature Ablation (Spec §47: without ports, without TCP flags, without timing, etc.)
  2. Executes Robustness Perturbations (Spec §46: Gaussian feature noise)
  3. Generates all 8 publication graphs from Spec §49 & §50
  4. Saves feature_ablation.csv, robustness_perturbation.csv, and all 8 figures
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml
import numpy as np
import pandas as pd
import torch

from predator_net.training.seed import set_global_seed
from predator_net.models.dynamic_mlp import DynamicMLP
from predator_net.evaluation.ablation import run_feature_ablation, run_robustness_perturbation
from predator_net.visualization.evolution import plot_evolution_graphs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_ablations")


def main() -> None:
    parser = argparse.ArgumentParser(description="PREDATOR NET — Phase 10: Ablation & Robustness Studies")
    parser.add_argument(
        "--config",
        type=str,
        default="config/baseline.yaml",
        help="Path to configuration YAML",
    )
    args = parser.parse_args()

    # Load config
    with open(PROJECT_ROOT / args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("project", {}).get("seed", 42)
    set_global_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 70)
    logger.info("PREDATOR NET — PHASE 10: ABLATION & ROBUSTNESS STUDIES")
    logger.info("=" * 70)

    # Directories
    eval_dir = PROJECT_ROOT / "artifacts" / "evaluation"
    figures_dir = PROJECT_ROOT / "artifacts" / "figures"
    pred_dir = PROJECT_ROOT / "artifacts" / "predator"
    features_dir = PROJECT_ROOT / "artifacts" / "features"
    processed_dir = PROJECT_ROOT / config.get("data", {}).get("processed_dir", "data/processed")

    eval_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load test data and feature names
    X_test = np.load(processed_dir / "test_X.npy")
    y_test = np.load(processed_dir / "test_y.npy")

    with open(features_dir / "selected_features.json", "r") as f:
        feat_meta = json.load(f)
    feature_names = feat_meta["selected_features"]

    # Load compressed model
    comp_ckpt_path = pred_dir / "predator_compressed.pt"
    if not comp_ckpt_path.exists():
        logger.warning("Compressed model not found, falling back to baseline checkpoint...")
        comp_ckpt_path = PROJECT_ROOT / "artifacts" / "models" / "baseline_best.pt"

    ckpt = torch.load(comp_ckpt_path, map_location=device, weights_only=False)
    model = DynamicMLP(
        input_dim=ckpt["input_dim"],
        hidden_dims=ckpt["hidden_dims"],
        output_dim=1,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # 1. Feature Ablation (Spec §47)
    logger.info("Running Feature Ablation experiments...")
    ablation_df = run_feature_ablation(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        threshold=0.5,
        device=device,
    )
    ablation_csv_path = eval_dir / "feature_ablation.csv"
    ablation_df.to_csv(ablation_csv_path, index=False)
    logger.info(f"Saved feature ablation results to {ablation_csv_path}")
    print("\n--- Feature Ablation Results (Spec §47) ---")
    print(ablation_df.to_string(index=False))

    # 2. Robustness Perturbation (Spec §46)
    logger.info("\nRunning Robustness Noise Perturbations...")
    robustness_df = run_robustness_perturbation(
        model=model,
        X_test=X_test,
        y_test=y_test,
        noise_stds=[0.01, 0.05, 0.10, 0.20, 0.50],
        threshold=0.5,
        seed=seed,
        device=device,
    )
    robustness_csv_path = eval_dir / "robustness_perturbation.csv"
    robustness_df.to_csv(robustness_csv_path, index=False)
    logger.info(f"Saved robustness results to {robustness_csv_path}")
    print("\n--- Robustness Noise Perturbation Results (Spec §46) ---")
    print(robustness_df.to_string(index=False))

    # 3. Evolution Graphs (Spec §49 & §50)
    logger.info("\nGenerating all 8 publication graphs from Spec §49 & §50...")
    history_path = pred_dir / "absorption_history.csv"
    importance_path = pred_dir / "neuron_importance.csv"

    history_df = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    importance_df = pd.read_csv(importance_path) if importance_path.exists() else None

    plot_evolution_graphs(history_df, importance_df, figures_dir)
    logger.info(f"Saved 8 publication figures to {figures_dir}")

    logger.info("=" * 70)
    logger.info("PHASE 10 COMPLETE!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
