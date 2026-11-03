"""
PHASE 9 SCRIPT: Run 6-Model Comparative Benchmark Suite.

Usage:
    python scripts/run_benchmarks.py [--config config/predator.yaml] [--baseline-config config/baseline.yaml]

This script:
  1. Loads baseline ANN checkpoint artifacts/models/baseline_best.pt
  2. Evaluates all 6 model variants under strictly identical conditions:
       - Model A: Dense ANN Baseline
       - Model B: Random Structured Removal
       - Model C: Magnitude-Based Structured Pruning (L1-norm)
       - Model D: PREDATOR-LIGHT (Outgoing weight absorption)
       - Model E: PREDATOR-FULL (Outgoing + incoming/bias absorption)
       - Model F: PREDATOR-D (Absorption + Knowledge Distillation)
  3. Measures Accuracy, Precision, Recall, F1, PR-AUC, FPR, Latency, and Throughput
  4. Generates the Core Research Table (Spec §41) in CSV and Markdown
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
import torch

from predator_net.training.seed import set_global_seed
from predator_net.models.baseline_mlp import BaselineMLP
from predator_net.models.dynamic_mlp import DynamicMLP
from predator_net.data.dataset import FlowDataset, create_dataloader
from predator_net.training.losses import get_loss_function
from predator_net.evaluation.benchmark import run_6_model_benchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_benchmarks")


def main() -> None:
    parser = argparse.ArgumentParser(description="PREDATOR NET — Phase 9: 6-Model Benchmarking")
    parser.add_argument(
        "--config",
        type=str,
        default="config/predator.yaml",
        help="Path to PREDATOR configuration YAML",
    )
    parser.add_argument(
        "--baseline-config",
        type=str,
        default="config/baseline.yaml",
        help="Path to baseline configuration YAML",
    )
    args = parser.parse_args()

    # Load configs
    with open(PROJECT_ROOT / args.config, "r") as f:
        pred_config = yaml.safe_load(f)

    with open(PROJECT_ROOT / args.baseline_config, "r") as f:
        base_config = yaml.safe_load(f)

    seed = base_config.get("project", {}).get("seed", 42)
    set_global_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 70)
    logger.info("PREDATOR NET — PHASE 9: 6-MODEL COMPARATIVE BENCHMARK")
    logger.info(f"Target Device: {device}")
    logger.info("=" * 70)

    # Output paths
    eval_dir = PROJECT_ROOT / "artifacts" / "evaluation"
    models_dir = PROJECT_ROOT / "artifacts" / "models"
    processed_dir = PROJECT_ROOT / base_config.get("data", {}).get("processed_dir", "data/processed")
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline model
    ckpt_path = models_dir / "baseline_best.pt"
    if not ckpt_path.exists():
        logger.error(f"Baseline checkpoint not found at {ckpt_path}. Run scripts/train_baseline.py first!")
        sys.exit(1)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    baseline_model = BaselineMLP(
        input_dim=checkpoint["input_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        output_dim=1,
    )
    baseline_model.load_state_dict(checkpoint["model_state_dict"])
    dynamic_baseline = DynamicMLP.from_baseline(baseline_model).to(device)

    # Load data arrays
    X_train = np.load(processed_dir / "train_X.npy")
    y_train = np.load(processed_dir / "train_y.npy")
    X_val = np.load(processed_dir / "val_X.npy")
    y_val = np.load(processed_dir / "val_y.npy")
    X_test = np.load(processed_dir / "test_X.npy")
    y_test = np.load(processed_dir / "test_y.npy")

    batch_size = base_config.get("training", {}).get("batch_size", 1024)
    train_dataset = FlowDataset(X_train, y_train)
    val_dataset = FlowDataset(X_val, y_val)
    test_dataset = FlowDataset(X_test, y_test)

    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    calib_samples = pred_config.get("predator", {}).get("importance", {}).get("calibration_samples", 50000)
    if len(train_dataset) > calib_samples:
        calib_idx = np.random.choice(len(train_dataset), size=calib_samples, replace=False)
        calib_dataset = FlowDataset(X_train[calib_idx], y_train[calib_idx])
    else:
        calib_dataset = train_dataset
    calib_loader = create_dataloader(calib_dataset, batch_size=batch_size, shuffle=False)

    metadata_path = PROJECT_ROOT / "artifacts" / "features" / "dataset_metadata.json"
    pos_weight = None
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            pos_weight = json.load(f).get("pos_weight", None)
    loss_fn = get_loss_function(pos_weight=pos_weight, device=device)

    # Run Benchmark
    df_table, _ = run_6_model_benchmark(
        baseline_model=dynamic_baseline,
        train_loader=train_loader,
        calib_loader=calib_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        predator_config=pred_config.get("predator", {}),
        device=device,
        output_dir=eval_dir,
    )

    logger.info("\n" + "=" * 90)
    logger.info("CORE RESEARCH TABLE (Spec §41)")
    logger.info("=" * 90)
    print(df_table.to_string(index=False))
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
