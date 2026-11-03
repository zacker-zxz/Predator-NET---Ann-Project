"""
Unit tests for BaselineMLP, Model Utils, Losses, Callbacks, and Metrics.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from predator_net.models.baseline_mlp import BaselineMLP
from predator_net.models.model_utils import (
    count_parameters,
    calculate_model_size,
    get_layer_dimensions,
    verify_model_integrity,
)
from predator_net.training.losses import get_loss_function
from predator_net.training.callbacks import EarlyStopping
from predator_net.evaluation.metrics import compute_metrics, tune_threshold
from predator_net.evaluation.latency import measure_inference_latency


class TestBaselineMLP:
    def test_forward_pass_output_shape(self):
        input_dim = 25
        model = BaselineMLP(input_dim=input_dim, hidden_dims=[128, 64, 32], output_dim=1)

        # Test single sample
        x1 = torch.randn(1, input_dim)
        out1 = model(x1)
        assert out1.shape == (1, 1)

        # Test batch
        x_batch = torch.randn(32, input_dim)
        out_batch = model(x_batch)
        assert out_batch.shape == (32, 1)

    def test_predict_proba_range(self):
        model = BaselineMLP(input_dim=10)
        x = torch.randn(20, 10)
        probs = model.predict_proba(x)

        assert probs.shape == (20, 1)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_predict_binary_values(self):
        model = BaselineMLP(input_dim=10)
        x = torch.randn(20, 10)
        preds = model.predict(x, threshold=0.5)

        assert preds.shape == (20, 1)
        unique_vals = set(preds.numpy().flatten())
        assert unique_vals.issubset({0, 1})

    def test_parameter_count_formula(self):
        # 10 -> 128 -> 64 -> 32 -> 1
        # L1: 10*128 + 128 = 1408
        # L2: 128*64 + 64 = 8256
        # L3: 64*32 + 32 = 2080
        # L4: 32*1 + 1 = 33
        # Total = 1408 + 8256 + 2080 + 33 = 11777
        model = BaselineMLP(input_dim=10, hidden_dims=[128, 64, 32], output_dim=1)
        counts = count_parameters(model)
        assert counts["total_parameters"] == 11777
        assert counts["trainable_parameters"] == 11777

    def test_layer_dimensions(self):
        model = BaselineMLP(input_dim=15, hidden_dims=[128, 64, 32], output_dim=1)
        dims = get_layer_dimensions(model)
        assert dims == [15, 128, 64, 32, 1]

    def test_model_integrity_check(self):
        model = BaselineMLP(input_dim=10)
        x = torch.randn(5, 10)
        passed, msg = verify_model_integrity(model, x)
        assert passed is True


class TestLossesAndCallbacks:
    def test_loss_function_with_pos_weight(self):
        loss_fn = get_loss_function(pos_weight=4.0)
        assert isinstance(loss_fn, nn.BCEWithLogitsLoss)
        assert loss_fn.pos_weight is not None
        assert abs(loss_fn.pos_weight.item() - 4.0) < 1e-5

    def test_early_stopping_trigger(self):
        model = BaselineMLP(input_dim=5)
        es = EarlyStopping(patience=3, monitor="val_f1", mode="max")

        # Monotonically worsening scores
        assert es(0.90, model, 1) is False
        assert es(0.85, model, 2) is False
        assert es(0.80, model, 3) is False
        assert es(0.75, model, 4) is True  # 3 consecutive non-improvements
        assert es.early_stop is True
        assert es.best_epoch == 1


class TestEvaluationMetrics:
    def test_compute_metrics_perfect_prediction(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.05, 0.10, 0.95, 0.90])

        metrics = compute_metrics(y_true, y_prob, threshold=0.5)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["false_positive_rate"] == 0.0
        assert metrics["false_negative_rate"] == 0.0
        assert metrics["roc_auc"] == 1.0
        assert metrics["pr_auc"] == 1.0

    def test_tune_threshold(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.35, 0.40, 0.8, 0.9])

        # At threshold 0.5, recall would be 2/3. At threshold 0.38, recall and F1 are 1.0
        best_t, best_f1 = tune_threshold(y_true, y_prob, metric="f1")
        assert 0.35 <= best_t <= 0.40
        assert best_f1 == 1.0

    def test_measure_latency(self):
        model = BaselineMLP(input_dim=16)
        bench = measure_inference_latency(model, input_dim=16, batch_size=128, warmup_runs=2, benchmark_runs=5)
        assert "single_sample_latency_ms" in bench
        assert "throughput_samples_per_sec" in bench
        assert bench["throughput_samples_per_sec"] > 0
