"""
Benchmarking Engine for 6-Model Comparative Analysis.

Implements Spec §40–§42:
  - Model A: Dense ANN (Baseline)
  - Model B: Random Structured Removal
  - Model C: Magnitude-Based Structured Pruning (L1-norm)
  - Model D: PREDATOR-LIGHT (Outgoing weight absorption)
  - Model E: PREDATOR-FULL (Outgoing + incoming/bias absorption)
  - Model F: PREDATOR-D (Absorption + Knowledge Distillation)

Produces the Core Research Table (Spec §41) populated strictly with measured experimental values.
"""

import copy
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from predator_net.models.dynamic_mlp import DynamicMLP
from predator_net.models.model_utils import count_parameters, calculate_model_size
from predator_net.predator.structural_rebuild import remove_neuron_structurally
from predator_net.predator.distillation import DistillationLoss
from predator_net.predator.controller import run_predator_loop
from predator_net.evaluation.metrics import compute_metrics, tune_threshold
from predator_net.evaluation.latency import measure_inference_latency
from predator_net.training.trainer import evaluate, train_epoch

logger = logging.getLogger(__name__)


def create_random_pruned_model(
    baseline_model: DynamicMLP,
    target_hidden_dims: List[int],
    seed: int = 42,
) -> DynamicMLP:
    """
    Model B: Random structured neuron removal without knowledge transfer.
    Removes the exact number of neurons needed to match target_hidden_dims.
    """
    model = baseline_model.clone()
    rng = np.random.RandomState(seed)

    for l_idx in range(model.num_hidden_layers):
        target_dim = target_hidden_dims[l_idx]
        while model.hidden_dims[l_idx] > target_dim:
            curr_dim = model.hidden_dims[l_idx]
            random_neuron = rng.randint(0, curr_dim)
            remove_neuron_structurally(model, layer_idx=l_idx, neuron_idx=random_neuron)

    return model


def create_magnitude_pruned_model(
    baseline_model: DynamicMLP,
    target_hidden_dims: List[int],
) -> DynamicMLP:
    """
    Model C: Magnitude-based structured pruning (L1-norm).
    Removes neurons with smallest outgoing weight L1-norm without transfer.
    """
    model = baseline_model.clone()

    for l_idx in range(model.num_hidden_layers):
        target_dim = target_hidden_dims[l_idx]
        while model.hidden_dims[l_idx] > target_dim:
            # Outgoing weights are in linears[l_idx + 1].weight (shape: out_feat, in_feat)
            next_w = model.linears[l_idx + 1].weight.detach().cpu().numpy()
            l1_norms = np.sum(np.abs(next_w), axis=0)  # shape: (curr_dim,)
            smallest_neuron = int(np.argmin(l1_norms))
            remove_neuron_structurally(model, layer_idx=l_idx, neuron_idx=smallest_neuron)

    return model


def train_distilled_model(
    student_model: DynamicMLP,
    teacher_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-4,
    temperature: float = 2.0,
    lambda_distill: float = 0.30,
    pos_weight: Optional[float] = None,
    device: str = "cpu",
) -> DynamicMLP:
    """
    Model F: Knowledge Distillation for PREDATOR-D (Spec §32).
    """
    dev = torch.device(device)
    student_model.to(dev)
    teacher_model.to(dev)
    teacher_model.eval()

    loss_fn = DistillationLoss(
        temperature=temperature,
        alpha_distill=lambda_distill,
        pos_weight=pos_weight,
        device=device,
    )
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr, weight_decay=1e-4)

    logger.info(f"Training PREDATOR-D student with KD (epochs={epochs}, T={temperature}, λ={lambda_distill})...")

    for ep in range(1, epochs + 1):
        student_model.train()
        for x_b, y_b in train_loader:
            x_b = x_b.to(dev)
            y_b = y_b.to(dev)

            with torch.no_grad():
                teacher_logits = teacher_model(x_b)

            optimizer.zero_grad()
            student_logits = student_model(x_b)
            loss = loss_fn(student_logits, teacher_logits, y_b)
            loss.backward()
            optimizer.step()

    return student_model


def run_6_model_benchmark(
    baseline_model: DynamicMLP,
    train_loader: DataLoader,
    calib_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    predator_config: Dict[str, Any],
    device: str = "cpu",
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute all 6 model evaluations under identical conditions (Spec §40–§42).
    """
    dev = torch.device(device)
    baseline_model.to(dev)
    input_dim = baseline_model.input_dim

    results: Dict[str, Any] = {}
    table_rows: List[Dict[str, Any]] = []

    def evaluate_model_pipeline(model: DynamicMLP, name: str) -> Dict[str, Any]:
        """Tune threshold on validation, evaluate on test, measure latency."""
        _, _, v_targets, v_probs = evaluate(model, val_loader, loss_fn, device=dev)
        best_t, _ = tune_threshold(v_targets, v_probs, metric="f1")

        _, test_m, _, _ = evaluate(model, test_loader, loss_fn, device=dev, threshold=best_t)
        lat = measure_inference_latency(model, input_dim=input_dim, device=device)
        p_count = count_parameters(model)["total_parameters"]
        total_neurons = sum(model.hidden_dims)

        row = {
            "Model": name,
            "Neurons": int(total_neurons),
            "Parameters": int(p_count),
            "Accuracy": float(test_m["accuracy"]),
            "Precision": float(test_m["precision"]),
            "Recall": float(test_m["recall"]),
            "F1": float(test_m["f1"]),
            "PR-AUC": float(test_m["pr_auc"]),
            "FPR": float(test_m["false_positive_rate"]),
            "Latency (ms)": float(lat["single_sample_latency_ms"]["median"]),
            "Throughput (samples/s)": float(lat["throughput_samples_per_sec"]),
        }
        table_rows.append(row)
        return {"metrics": test_m, "latency": lat, "threshold": best_t, "hidden_dims": model.hidden_dims}

    # 1. Model A: Dense ANN
    logger.info("Evaluating Model A: Dense ANN Baseline...")
    results["Model A (Dense ANN)"] = evaluate_model_pipeline(baseline_model, "Dense ANN")

    # 2. Model D: PREDATOR-LIGHT
    logger.info("Running Model D: PREDATOR-LIGHT absorption...")
    cfg_light = copy.deepcopy(predator_config)
    cfg_light["absorption"]["mode"] = "light"
    pred_dir_light = (output_dir / "light") if output_dir else Path("temp_light")
    model_light, _ = run_predator_loop(
        model=baseline_model.clone(),
        train_loader=train_loader,
        calibration_loader=calib_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        predator_config=cfg_light,
        output_dir=pred_dir_light,
    )
    results["Model D (PREDATOR-Light)"] = evaluate_model_pipeline(model_light, "PREDATOR-Light")
    target_dims = model_light.hidden_dims

    # 3. Model B: Random Pruning matching target_dims
    logger.info(f"Evaluating Model B: Random Pruned to {target_dims}...")
    model_random = create_random_pruned_model(baseline_model, target_hidden_dims=target_dims)
    results["Model B (Random Pruning)"] = evaluate_model_pipeline(model_random, "Random Pruning")

    # 4. Model C: Magnitude-based Pruning matching target_dims
    logger.info(f"Evaluating Model C: Magnitude Pruned to {target_dims}...")
    model_magnitude = create_magnitude_pruned_model(baseline_model, target_hidden_dims=target_dims)
    results["Model C (Structured Pruning)"] = evaluate_model_pipeline(model_magnitude, "Structured Pruning")

    # 5. Model E: PREDATOR-FULL
    logger.info("Running Model E: PREDATOR-FULL absorption...")
    cfg_full = copy.deepcopy(predator_config)
    cfg_full["absorption"]["mode"] = "full"
    pred_dir_full = (output_dir / "full") if output_dir else Path("temp_full")
    model_full, _ = run_predator_loop(
        model=baseline_model.clone(),
        train_loader=train_loader,
        calibration_loader=calib_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        predator_config=cfg_full,
        output_dir=pred_dir_full,
    )
    results["Model E (PREDATOR-Full)"] = evaluate_model_pipeline(model_full, "PREDATOR-Full")

    # 6. Model F: PREDATOR-D (Absorption + Distillation)
    logger.info("Running Model F: PREDATOR-D Distillation...")
    model_distilled = train_distilled_model(
        student_model=model_light.clone(),
        teacher_model=baseline_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        device=device,
    )
    results["Model F (PREDATOR-D)"] = evaluate_model_pipeline(model_distilled, "PREDATOR-D")

    # Construct DataFrame for Core Research Table (Spec §41)
    df_table = pd.DataFrame(table_rows)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "core_research_table.csv"
        md_path = output_dir / "core_research_table.md"
        json_path = output_dir / "all_models_metrics.json"

        df_table.to_csv(csv_path, index=False)
        with open(md_path, "w") as f:
            f.write("# PREDATOR NET — Core Research Table (Spec §41)\n\n")
            f.write(df_table.to_markdown(index=False))
            f.write("\n")

        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Saved Core Research Table to {csv_path} and {md_path}")

    return df_table, results
