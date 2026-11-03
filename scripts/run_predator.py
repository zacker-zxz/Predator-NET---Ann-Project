"""
PHASE 8 SCRIPT: Run Full Iterative PREDATOR Absorption Loop.

Usage:
    python scripts/run_predator.py [--config config/predator.yaml] [--baseline-config config/baseline.yaml]

This script:
  1. Loads the trained baseline checkpoint artifacts/models/baseline_best.pt (Spec §16)
  2. Converts BaselineMLP to DynamicMLP with nn.ModuleList (Spec §29)
  3. Prepares calibration subset and validation DataLoaders (Spec §22)
  4. Runs the iterative PREDATOR loop with validation gating and rollback (Spec §30, §33)
  5. Logs all 17 event columns to artifacts/predator/absorption_history.csv (Spec §48)
  6. Evaluates final compressed network on unseen TEST set (Spec §17)
  7. Calculates exact structural compression metrics (neurons, params, size)
  8. Saves compressed model checkpoint artifacts/predator/predator_compressed.pt
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
from predator_net.models.model_utils import count_parameters, calculate_model_size, get_layer_dimensions
from predator_net.data.dataset import FlowDataset, create_dataloader
from predator_net.training.losses import get_loss_function
from predator_net.training.trainer import evaluate
from predator_net.evaluation.metrics import compute_metrics, tune_threshold
from predator_net.evaluation.latency import measure_inference_latency
from predator_net.predator.controller import run_predator_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_predator")


def main() -> None:
    parser = argparse.ArgumentParser(description="PREDATOR NET — Phase 8: Iterative Absorption Loop")
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
    pred_cfg_path = PROJECT_ROOT / args.config
    base_cfg_path = PROJECT_ROOT / args.baseline_config

    with open(pred_cfg_path, "r") as f:
        pred_config = yaml.safe_load(f)

    with open(base_cfg_path, "r") as f:
        base_config = yaml.safe_load(f)

    seed = base_config.get("project", {}).get("seed", 42)
    set_global_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 70)
    logger.info("PREDATOR NET — PHASE 8: ITERATIVE NEURON ABSORPTION LOOP")
    logger.info(f"Target Device: {device}")
    logger.info("=" * 70)

    # Output paths
    pred_artifacts_dir = PROJECT_ROOT / "artifacts" / "predator"
    models_dir = PROJECT_ROOT / "artifacts" / "models"
    processed_dir = PROJECT_ROOT / base_config.get("data", {}).get("processed_dir", "data/processed")
    pred_artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load baseline checkpoint
    ckpt_path = models_dir / "baseline_best.pt"
    if not ckpt_path.exists():
        logger.error(f"Baseline checkpoint not found at {ckpt_path}. Run scripts/train_baseline.py first!")
        sys.exit(1)

    logger.info(f"Loading trained baseline checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    input_dim = checkpoint["input_dim"]
    hidden_dims = checkpoint["hidden_dims"]

    baseline_model = BaselineMLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=1)
    baseline_model.load_state_dict(checkpoint["model_state_dict"])
    baseline_model.to(device)

    # Convert to DynamicMLP (Spec §29)
    dynamic_model = DynamicMLP.from_baseline(baseline_model)
    dynamic_model.to(device)

    init_params = count_parameters(dynamic_model)
    init_size = calculate_model_size(dynamic_model)
    init_layers = get_layer_dimensions(dynamic_model)

    logger.info(f"Converted to DynamicMLP. Initial hidden widths: {dynamic_model.hidden_dims}")
    logger.info(f"Initial Trainable Parameters: {init_params['trainable_parameters']:,} ({init_size['size_kb']:.2f} KB)")

    # 2. Load processed datasets
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

    # Calibration loader (subsample per Spec §22)
    calib_samples = pred_config.get("predator", {}).get("importance", {}).get("calibration_samples", 50000)
    if len(train_dataset) > calib_samples:
        calib_idx = np.random.choice(len(train_dataset), size=calib_samples, replace=False)
        calib_dataset = FlowDataset(X_train[calib_idx], y_train[calib_idx])
    else:
        calib_dataset = train_dataset
    calib_loader = create_dataloader(calib_dataset, batch_size=batch_size, shuffle=False)

    # Loss function with pos_weight
    metadata_path = PROJECT_ROOT / "artifacts" / "features" / "dataset_metadata.json"
    pos_weight = None
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            pos_weight = json.load(f).get("pos_weight", None)
    loss_fn = get_loss_function(pos_weight=pos_weight, device=device)

    # Read baseline validation threshold
    baseline_metrics_path = models_dir / "baseline_metrics.json"
    if baseline_metrics_path.exists():
        with open(baseline_metrics_path, "r") as f:
            base_rep = json.load(f)
        eval_threshold = base_rep.get("optimal_threshold", 0.5)
    else:
        eval_threshold = 0.5

    # 3. Run PREDATOR Iterative Loop
    pred_settings = pred_config.get("predator", {})
    compressed_model, history_df = run_predator_loop(
        model=dynamic_model,
        train_loader=train_loader,
        calibration_loader=calib_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        predator_config=pred_settings,
        output_dir=pred_artifacts_dir,
    )

    # 4. Final Evaluation on unseen TEST data
    logger.info("Evaluating compressed model on unseen TEST set...")
    # Re-tune threshold on validation for compressed model
    _, _, val_targets, val_probs = evaluate(compressed_model, val_loader, loss_fn, device=torch.device(device))
    best_threshold, _ = tune_threshold(val_targets, val_probs, metric="f1")

    test_loss, comp_metrics, test_targets, test_probs = evaluate(
        compressed_model, test_loader, loss_fn, device=torch.device(device), threshold=best_threshold
    )

    final_params = count_parameters(compressed_model)
    final_size = calculate_model_size(compressed_model)
    final_layers = get_layer_dimensions(compressed_model)

    latency_info = measure_inference_latency(compressed_model, input_dim=input_dim, device=device)

    # Calculate compression metrics
    initial_hidden_neurons = sum(checkpoint["hidden_dims"])
    final_hidden_neurons = sum(compressed_model.hidden_dims)
    neuron_reduction_pct = float((initial_hidden_neurons - final_hidden_neurons) / initial_hidden_neurons * 100)
    param_reduction_pct = float(
        (init_params["total_parameters"] - final_params["total_parameters"])
        / init_params["total_parameters"]
        * 100
    )

    # Save compressed checkpoint
    compressed_ckpt_path = pred_artifacts_dir / "predator_compressed.pt"
    torch.save(
        {
            "model_state_dict": compressed_model.state_dict(),
            "input_dim": input_dim,
            "hidden_dims": compressed_model.hidden_dims,
            "layer_dims": final_layers,
            "parameters": final_params,
            "threshold": best_threshold,
            "seed": seed,
        },
        compressed_ckpt_path,
    )
    logger.info(f"Saved compressed model checkpoint to {compressed_ckpt_path}")

    # Save summary report
    summary_report = {
        "model_name": "PREDATOR_Compressed_ANN",
        "initial_architecture": init_layers,
        "final_architecture": final_layers,
        "initial_hidden_neurons": initial_hidden_neurons,
        "final_hidden_neurons": final_hidden_neurons,
        "neuron_reduction_percent": neuron_reduction_pct,
        "initial_parameters": init_params,
        "final_parameters": final_params,
        "parameter_reduction_percent": param_reduction_pct,
        "initial_model_size": init_size,
        "final_model_size": final_size,
        "optimal_threshold": float(best_threshold),
        "test_loss": float(test_loss),
        "metrics": comp_metrics,
        "latency_and_throughput": latency_info,
        "absorption_events": {
            "total_attempts": len(history_df),
            "accepted_absorptions": int((history_df["decision"] == "ACCEPTED").sum()) if not history_df.empty else 0,
            "rejected_absorptions": int((history_df["decision"] == "REJECTED").sum()) if not history_df.empty else 0,
        },
    }

    summary_json_path = pred_artifacts_dir / "predator_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_report, f, indent=2)
    logger.info(f"Saved PREDATOR summary report to {summary_json_path}")

    logger.info("=" * 70)
    logger.info("PHASE 8 COMPLETE — PREDATOR COMPRESSION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Hidden Neurons:   {initial_hidden_neurons} → {final_hidden_neurons} (-{neuron_reduction_pct:.1f}%)")
    logger.info(f"Parameters:       {init_params['total_parameters']:,} → {final_params['total_parameters']:,} (-{param_reduction_pct:.1f}%)")
    logger.info(f"Test Accuracy:    {comp_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision:   {comp_metrics['precision']:.4f}")
    logger.info(f"Test Recall:      {comp_metrics['recall']:.4f}")
    logger.info(f"Test F1 Score:    {comp_metrics['f1']:.4f}")
    logger.info(f"Test FPR:         {comp_metrics['false_positive_rate']:.4f}")
    logger.info(f"Throughput:       {latency_info['throughput_samples_per_sec']:,.0f} samples/sec")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
