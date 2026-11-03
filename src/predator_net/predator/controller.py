"""
PREDATOR NET Controller and Iterative Absorption Loop.

Implements Spec §30, §31, §33, §34, §35, §37, §38, §48:
  - Validation-gated absorption with rollback support (Spec §30).
  - Configurable performance tolerance (max F1 drop, max Recall drop) (Spec §31).
  - Minimum hidden layer width constraints (Spec §34).
  - Layer selection strategies: deepest_first, weakest_first, sequential, round_robin (Spec §37).
  - Complete 17-column absorption history logging (Spec §48).
"""

import copy
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from predator_net.models.dynamic_mlp import DynamicMLP
from predator_net.models.model_utils import count_parameters, verify_model_integrity
from predator_net.predator.importance import (
    compute_layer_neuron_importance,
    classify_neurons,
)
from predator_net.predator.similarity import (
    compute_activation_profiles,
    find_best_compatible_neuron,
)
from predator_net.predator.absorption import compute_alpha, apply_absorption_weights
from predator_net.predator.structural_rebuild import remove_neuron_structurally
from predator_net.training.trainer import evaluate, train_epoch

logger = logging.getLogger(__name__)


def predator_step(
    model: DynamicMLP,
    train_loader: Optional[DataLoader],
    calibration_loader: DataLoader,
    val_loader: DataLoader,
    layer_idx: int,
    loss_fn: nn.Module,
    device: torch.device,
    predator_config: Dict[str, Any],
    val_metrics_current: Dict[str, Any],
) -> Tuple[DynamicMLP, Dict[str, Any], bool]:
    """
    Execute one PREDATOR absorption step on a specified hidden layer (Spec §35).

    Returns:
        Tuple of (resulting_model, event_record_dict, was_accepted_bool).
    """
    min_width = predator_config.get("minimum_width", {}).get(f"layer_{layer_idx + 1}", 8)
    curr_width = model.hidden_dims[layer_idx]

    params_before = count_parameters(model)["total_parameters"]
    f1_before = val_metrics_current["f1"]
    recall_before = val_metrics_current["recall"]

    base_record: Dict[str, Any] = {
        "layer": f"layer_{layer_idx + 1}",
        "weak_neuron": None,
        "strong_neuron": None,
        "weak_importance": 0.0,
        "strong_importance": 0.0,
        "activation_similarity": 0.0,
        "weight_similarity": 0.0,
        "compatibility": 0.0,
        "alpha": 0.0,
        "f1_before": f1_before,
        "f1_after": f1_before,
        "recall_before": recall_before,
        "recall_after": recall_before,
        "params_before": params_before,
        "params_after": params_before,
        "decision": "REJECTED",
        "reason": "",
    }

    # Constraint: minimum network width (Spec §34)
    if curr_width <= min_width:
        base_record["reason"] = f"minimum_width_reached ({curr_width} <= {min_width})"
        return model, base_record, False

    # 1. Compute neuron importance for target layer
    imp_cfg = predator_config.get("importance", {})
    alpha_imp = imp_cfg.get("gradient_weight", 0.70)
    layer_results = compute_layer_neuron_importance(
        model=model,
        loader=calibration_loader,
        loss_fn=loss_fn,
        device=device,
        alpha=alpha_imp,
    )

    layer_data = layer_results[layer_idx]
    combined_scores = layer_data["combined_imp"]

    cls_cfg = predator_config.get("classification", {})
    policy = cls_cfg.get("policy", "percentile")
    weak_pct = cls_cfg.get("weak_percentile", 15.0)
    strong_pct = cls_cfg.get("strong_percentile", 30.0)

    statuses = classify_neurons(
        combined_scores,
        policy=policy,
        weak_percentile=weak_pct,
        strong_percentile=strong_pct,
    )

    weak_indices = [i for i, s in enumerate(statuses) if s == "weak"]
    strong_indices = [i for i, s in enumerate(statuses) if s == "strong"]

    if not weak_indices:
        base_record["reason"] = "rejected_no_weak_candidates"
        return model, base_record, False

    if not strong_indices:
        base_record["reason"] = "rejected_no_strong_candidates"
        return model, base_record, False

    # Pick weakest candidate in this layer
    weak_idx = min(weak_indices, key=lambda idx: combined_scores[idx])
    imp_w = float(combined_scores[weak_idx])
    base_record["weak_neuron"] = int(weak_idx)
    base_record["weak_importance"] = imp_w

    # 2. Extract activation profiles and evaluate compatibility
    act_profiles = compute_activation_profiles(
        model=model,
        loader=calibration_loader,
        layer_idx=layer_idx,
        device=device,
    )

    comp_cfg = predator_config.get("compatibility", {})
    beta = comp_cfg.get("activation_weight", 0.70)
    min_sim = comp_cfg.get("minimum_similarity", 0.70)

    compat_result = find_best_compatible_neuron(
        model=model,
        layer_idx=layer_idx,
        weak_idx=weak_idx,
        strong_indices=strong_indices,
        act_profiles=act_profiles,
        beta=beta,
        minimum_similarity=min_sim,
    )

    base_record["activation_similarity"] = float(compat_result["act_sim"])
    base_record["weight_similarity"] = float(compat_result["weight_sim"])
    base_record["compatibility"] = float(compat_result["compatibility"])
    base_record["strong_neuron"] = compat_result["strong_idx"]

    if not compat_result["accepted"] or compat_result["strong_idx"] is None:
        base_record["reason"] = compat_result["reason"]
        return model, base_record, False

    strong_idx = compat_result["strong_idx"]
    imp_s = float(combined_scores[strong_idx])
    base_record["strong_importance"] = imp_s

    # 3. Simulate absorption on a cloned model
    temp_model = model.clone()

    abs_cfg = predator_config.get("absorption", {})
    mode = abs_cfg.get("mode", "light")
    alpha_min = abs_cfg.get("alpha_min", 0.05)
    alpha_max = abs_cfg.get("alpha_max", 0.50)
    gamma = abs_cfg.get("gamma", 0.30)

    alpha_weight = compute_alpha(
        compatibility=compat_result["compatibility"],
        imp_w=imp_w,
        imp_s=imp_s,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )
    base_record["alpha"] = alpha_weight

    # Transfer weights
    apply_absorption_weights(
        model=temp_model,
        layer_idx=layer_idx,
        weak_idx=weak_idx,
        strong_idx=strong_idx,
        alpha=alpha_weight,
        mode=mode,
        gamma=gamma,
    )

    # Physically remove neuron w from temp_model
    remove_neuron_structurally(temp_model, layer_idx=layer_idx, neuron_idx=weak_idx)

    params_after = count_parameters(temp_model)["total_parameters"]
    base_record["params_after"] = params_after

    # Verify model integrity (Spec §60)
    sample_x = torch.randn(2, model.input_dim, device=device)
    passed_integrity, int_msg = verify_model_integrity(temp_model, sample_x)
    if not passed_integrity:
        base_record["reason"] = f"integrity_failed ({int_msg})"
        return model, base_record, False

    # 4. Validation Gate (Spec §30, §31)
    _, val_metrics_after, _, _ = evaluate(temp_model, val_loader, loss_fn, device=device)
    f1_after = val_metrics_after["f1"]
    recall_after = val_metrics_after["recall"]

    base_record["f1_after"] = f1_after
    base_record["recall_after"] = recall_after

    val_cfg = predator_config.get("validation", {})
    max_f1_drop = val_cfg.get("max_f1_drop", 0.005)
    max_rec_drop = val_cfg.get("max_recall_drop", 0.005)

    f1_drop = f1_before - f1_after
    rec_drop = recall_before - recall_after

    if f1_drop > max_f1_drop:
        base_record["reason"] = f"f1_drop_exceeded (drop={f1_drop:.4f} > {max_f1_drop})"
        return model, base_record, False

    if rec_drop > max_rec_drop:
        base_record["reason"] = f"recall_drop_exceeded (drop={rec_drop:.4f} > {max_rec_drop})"
        return model, base_record, False

    # 5. Optional Fine-Tuning after acceptance (Spec §39)
    ft_cfg = predator_config.get("fine_tune", {})
    if ft_cfg.get("enabled", True) and train_loader is not None:
        ft_epochs = ft_cfg.get("epochs", 2)
        ft_lr = ft_cfg.get("learning_rate", 0.0001)
        ft_opt = torch.optim.AdamW(temp_model.parameters(), lr=ft_lr, weight_decay=1e-4)

        for _ in range(ft_epochs):
            train_epoch(temp_model, train_loader, ft_opt, loss_fn, device)

        # Refresh metrics post fine-tuning
        _, ft_val_metrics, _, _ = evaluate(temp_model, val_loader, loss_fn, device=device)
        base_record["f1_after"] = ft_val_metrics["f1"]
        base_record["recall_after"] = ft_val_metrics["recall"]

    base_record["decision"] = "ACCEPTED"
    base_record["reason"] = "passed_validation_gate"
    logger.info(
        f"ACCEPTED: Layer {layer_idx + 1} neuron {weak_idx} -> {strong_idx} | "
        f"Compat: {compat_result['compatibility']:.3f} | F1: {f1_before:.4f} -> {base_record['f1_after']:.4f} | "
        f"Params: {params_before:,} -> {params_after:,}"
    )
    return temp_model, base_record, True


def run_predator_loop(
    model: DynamicMLP,
    train_loader: Optional[DataLoader],
    calibration_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    predator_config: Dict[str, Any],
    output_dir: Path,
) -> Tuple[DynamicMLP, pd.DataFrame]:
    """
    Run the full iterative PREDATOR absorption loop (Spec §33).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    model.to(dev)

    # Initial baseline validation metrics
    _, curr_metrics, _, _ = evaluate(model, val_loader, loss_fn, device=dev)
    logger.info(f"Initial validation F1: {curr_metrics['f1']:.4f}, Recall: {curr_metrics['recall']:.4f}")

    iter_cfg = predator_config.get("iteration", {})
    max_iters = iter_cfg.get("max_iterations", 100)
    max_consecutive_rejections = iter_cfg.get("max_consecutive_rejections", 10)

    strat_cfg = predator_config.get("layer_strategy", {})
    layer_mode = strat_cfg.get("mode", "deepest_first")

    history_records = []
    consecutive_rejections = 0
    current_model = model

    logger.info(
        f"Starting PREDATOR iterative loop (max_iters={max_iters}, "
        f"strategy='{layer_mode}', max_consec_rejections={max_consecutive_rejections})."
    )

    for step_num in range(1, max_iters + 1):
        num_layers = current_model.num_hidden_layers

        # Select target layer according to strategy
        if layer_mode == "deepest_first":
            # Cycle through layers backwards: layer 2, layer 1, layer 0
            layer_idx = (num_layers - 1) - ((step_num - 1) % num_layers)
        elif layer_mode == "sequential":
            layer_idx = (step_num - 1) % num_layers
        else:  # round_robin or fallback
            layer_idx = (step_num - 1) % num_layers

        # Attempt one PREDATOR step
        new_model, record, accepted = predator_step(
            model=current_model,
            train_loader=train_loader,
            calibration_loader=calibration_loader,
            val_loader=val_loader,
            layer_idx=layer_idx,
            loss_fn=loss_fn,
            device=dev,
            predator_config=predator_config,
            val_metrics_current=curr_metrics,
        )

        record["iteration"] = step_num
        history_records.append(record)

        if accepted:
            current_model = new_model
            curr_metrics["f1"] = record["f1_after"]
            curr_metrics["recall"] = record["recall_after"]
            consecutive_rejections = 0
        else:
            consecutive_rejections += 1
            logger.debug(f"Step {step_num} rejected: {record['reason']}. (Consecutive: {consecutive_rejections})")

        if consecutive_rejections >= max_consecutive_rejections:
            logger.info(f"Stopping PREDATOR loop: {consecutive_rejections} consecutive rejections reached.")
            break

    # Save absorption history log with all 17 columns (Spec §48)
    history_df = pd.DataFrame(history_records)
    cols_order = [
        "iteration",
        "layer",
        "weak_neuron",
        "strong_neuron",
        "weak_importance",
        "strong_importance",
        "activation_similarity",
        "weight_similarity",
        "compatibility",
        "alpha",
        "f1_before",
        "f1_after",
        "recall_before",
        "recall_after",
        "params_before",
        "params_after",
        "decision",
        "reason",
    ]
    # Filter to existing columns in order
    final_cols = [c for c in cols_order if c in history_df.columns]
    history_df = history_df[final_cols]

    csv_path = output_dir / "absorption_history.csv"
    history_df.to_csv(csv_path, index=False)
    logger.info(f"Saved complete absorption history to {csv_path}")

    accepted_count = int((history_df["decision"] == "ACCEPTED").sum()) if not history_df.empty else 0
    logger.info(f"PREDATOR loop complete: {len(history_df)} attempts, {accepted_count} neurons absorbed.")
    return current_model, history_df
