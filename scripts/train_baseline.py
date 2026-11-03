"""
PHASE 3 SCRIPT: Train Baseline Artificial Neural Network (ANN).

Usage:
    python scripts/train_baseline.py [--config config/baseline.yaml]

This script:
  1. Loads processed arrays from data/processed/ (Spec §12)
  2. Dynamically instantiates BaselineMLP with exact input dimension (Spec Rule #3, §13)
  3. Configures AdamW optimizer, BCEWithLogitsLoss with pos_weight, and EarlyStopping (Spec §15, §16)
  4. Trains baseline MLP and logs learning curve
  5. Tunes classification threshold strictly on VALIDATION set (Spec §14)
  6. Evaluates final model on TEST set (Accuracy, Prec, Recall, F1, FPR, FNR, PR-AUC, ROC-AUC) (Spec §17)
  7. Measures inference latency & throughput (Spec §43)
  8. Saves baseline_best.pt, baseline_metrics.json, and all publication figures (Spec §16)
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
import torch.optim as optim

from predator_net.training.seed import set_global_seed
from predator_net.models.baseline_mlp import BaselineMLP
from predator_net.models.model_utils import count_parameters, calculate_model_size, get_layer_dimensions
from predator_net.training.losses import get_loss_function
from predator_net.training.callbacks import EarlyStopping
from predator_net.training.trainer import train_model, evaluate
from predator_net.data.dataset import FlowDataset, create_dataloader
from predator_net.evaluation.metrics import compute_metrics, tune_threshold
from predator_net.evaluation.latency import measure_inference_latency
from predator_net.visualization.plots import (
    plot_confusion_matrix,
    plot_training_curves,
    plot_roc_curve,
    plot_pr_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_baseline")


def main() -> None:
    parser = argparse.ArgumentParser(description="PREDATOR NET — Phase 3: Train Baseline ANN")
    parser.add_argument(
        "--config",
        type=str,
        default="config/baseline.yaml",
        help="Path to baseline configuration YAML",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = PROJECT_ROOT / args.config
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")
        config = {}

    seed = config.get("project", {}).get("seed", 42)
    set_global_seed(seed)

    # Device selection
    cfg_device = config.get("project", {}).get("device", "auto")
    if cfg_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg_device

    logger.info("=" * 70)
    logger.info("PREDATOR NET — PHASE 3: BASELINE ANN TRAINING")
    logger.info(f"Target Device: {device}")
    logger.info("=" * 70)

    # Paths
    processed_dir = PROJECT_ROOT / config.get("data", {}).get("processed_dir", "data/processed")
    models_dir = PROJECT_ROOT / "artifacts" / "models"
    figures_dir = PROJECT_ROOT / "artifacts" / "figures"
    metadata_path = PROJECT_ROOT / "artifacts" / "features" / "dataset_metadata.json"

    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset arrays
    train_x_path = processed_dir / "train_X.npy"
    train_y_path = processed_dir / "train_y.npy"
    val_x_path = processed_dir / "val_X.npy"
    val_y_path = processed_dir / "val_y.npy"
    test_x_path = processed_dir / "test_X.npy"
    test_y_path = processed_dir / "test_y.npy"

    if not (train_x_path.exists() and val_x_path.exists() and test_x_path.exists()):
        logger.error(
            f"Processed data arrays not found in {processed_dir}. Run scripts/preprocess_data.py first!"
        )
        sys.exit(1)

    logger.info(f"Loading preprocessed arrays from {processed_dir}...")
    X_train = np.load(train_x_path)
    y_train = np.load(train_y_path)
    X_val = np.load(val_x_path)
    y_val = np.load(val_y_path)
    X_test = np.load(test_x_path)
    y_test = np.load(test_y_path)

    input_dim = X_train.shape[1]
    logger.info(f"Input features: {input_dim}")
    logger.info(f"Train / Val / Test samples: {len(y_train):,} / {len(y_val):,} / {len(y_test):,}")

    # Read class pos_weight from metadata if available
    pos_weight = None
    if config.get("class_weighting", {}).get("enabled", True):
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                meta = json.load(f)
            pos_weight = meta.get("pos_weight", None)
        else:
            neg_count = float((y_train == 0).sum())
            pos_count = float((y_train == 1).sum())
            pos_weight = neg_count / max(pos_count, 1.0)
        logger.info(f"Class weighting enabled. Computed pos_weight: {pos_weight:.4f}")

    # Build DataLoader
    batch_size = config.get("training", {}).get("batch_size", 1024)
    train_dataset = FlowDataset(X_train, y_train)
    val_dataset = FlowDataset(X_val, y_val)
    test_dataset = FlowDataset(X_test, y_test)

    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize BaselineMLP dynamically
    hidden_dims = config.get("model", {}).get("hidden_dims", [128, 64, 32])
    model = BaselineMLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=1)
    model.to(device)

    param_info = count_parameters(model)
    size_info = calculate_model_size(model)
    layer_dims = get_layer_dimensions(model)

    logger.info(f"Initialized BaselineMLP architecture: {' → '.join(map(str, layer_dims))}")
    logger.info(f"Trainable Parameters: {param_info['trainable_parameters']:,} ({size_info['size_kb']:.2f} KB)")

    # Loss, Optimizer, Callbacks
    loss_fn = get_loss_function(pos_weight=pos_weight, device=device)
    lr = float(config.get("training", {}).get("learning_rate", 0.001))
    wd = float(config.get("training", {}).get("weight_decay", 0.0001))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    es_cfg = config.get("training", {}).get("early_stopping", {})
    early_stopping = None
    if es_cfg.get("enabled", True):
        early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 8),
            monitor=es_cfg.get("monitor", "val_f1"),
            mode=es_cfg.get("mode", "max"),
        )

    # Train model
    epochs = config.get("training", {}).get("epochs", 50)
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        early_stopping=early_stopping,
        device=device,
    )

    # Save best checkpoint
    checkpoint_path = models_dir / "baseline_best.pt"
    torch.save(
        {
            "model_state_dict": trained_model.state_dict(),
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "layer_dims": layer_dims,
            "parameters": param_info,
            "seed": seed,
        },
        checkpoint_path,
    )
    logger.info(f"Saved best model checkpoint to {checkpoint_path}")

    # Plot training curves
    plot_training_curves(history, figures_dir / "baseline_training_curve.png")

    # Tune threshold strictly on VALIDATION data (Spec §14)
    logger.info("Evaluating on validation data for threshold tuning...")
    _, _, val_targets, val_probs = evaluate(trained_model, val_loader, loss_fn, device=torch.device(device))

    if config.get("threshold", {}).get("tune_on_validation", True):
        best_threshold, val_opt_f1 = tune_threshold(val_targets, val_probs, metric="f1")
        logger.info(f"Optimal validation threshold: {best_threshold:.3f} (Val F1: {val_opt_f1:.4f})")
    else:
        best_threshold = config.get("threshold", {}).get("default", 0.5)
        logger.info(f"Using default threshold: {best_threshold:.3f}")

    # Evaluate on TEST data using validation-tuned threshold (Spec §17)
    logger.info("Evaluating final model on unseen TEST data...")
    test_loss, test_metrics, test_targets, test_probs = evaluate(
        trained_model, test_loader, loss_fn, device=torch.device(device), threshold=best_threshold
    )

    # Measure latency & throughput
    latency_info = measure_inference_latency(trained_model, input_dim=input_dim, device=device)

    # Aggregate full baseline evaluation report
    cm = np.array([
        [test_metrics["confusion_matrix"]["tn"], test_metrics["confusion_matrix"]["fp"]],
        [test_metrics["confusion_matrix"]["fn"], test_metrics["confusion_matrix"]["tp"]],
    ])

    report = {
        "model_name": "Baseline_Dense_ANN",
        "architecture": layer_dims,
        "input_dim": input_dim,
        "parameters": param_info,
        "model_size": size_info,
        "optimal_threshold": float(best_threshold),
        "test_loss": float(test_loss),
        "metrics": test_metrics,
        "latency_and_throughput": latency_info,
    }

    metrics_json_path = models_dir / "baseline_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved full test metrics report to {metrics_json_path}")

    # Plot publication figures
    plot_confusion_matrix(cm, figures_dir / "baseline_confusion_matrix.png", title="Baseline ANN Confusion Matrix")
    plot_roc_curve(test_targets, test_probs, test_metrics["roc_auc"], figures_dir / "baseline_roc_curve.png")
    plot_pr_curve(test_targets, test_probs, test_metrics["pr_auc"], figures_dir / "baseline_pr_curve.png")

    logger.info("=" * 70)
    logger.info("PHASE 3 COMPLETE — BASELINE PERFORMANCE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Accuracy:         {test_metrics['accuracy']:.4f}")
    logger.info(f"Precision:        {test_metrics['precision']:.4f}")
    logger.info(f"Recall (Det Rate):{test_metrics['recall']:.4f}")
    logger.info(f"F1 Score:         {test_metrics['f1']:.4f}")
    logger.info(f"Macro F1:         {test_metrics['macro_f1']:.4f}")
    logger.info(f"FPR (False Alarm):{test_metrics['false_positive_rate']:.4f}")
    logger.info(f"PR-AUC:           {test_metrics['pr_auc']:.4f}")
    logger.info(f"ROC-AUC:          {test_metrics['roc_auc']:.4f}")
    logger.info(f"Throughput:       {latency_info['throughput_samples_per_sec']:,.0f} samples/sec")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
