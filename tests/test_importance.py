"""
Unit tests for Neuron Importance Scoring and Classification.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from predator_net.models.dynamic_mlp import DynamicMLP
from predator_net.predator.importance import (
    compute_layer_neuron_importance,
    classify_neurons,
    generate_importance_report,
)


@pytest.fixture
def toy_setup():
    model = DynamicMLP(input_dim=8, hidden_dims=[16, 8], output_dim=1)
    X = torch.randn(64, 8)
    y = torch.randint(0, 2, (64, 1)).float()
    loader = DataLoader(TensorDataset(X, y), batch_size=16)
    loss_fn = nn.BCEWithLogitsLoss()
    return model, loader, loss_fn


class TestNeuronImportance:
    def test_compute_importance_shapes_and_values(self, toy_setup):
        model, loader, loss_fn = toy_setup
        res = compute_layer_neuron_importance(model, loader, loss_fn, device=torch.device("cpu"))

        assert len(res) == 2  # 2 hidden layers
        assert len(res[0]["act_imp"]) == 16
        assert len(res[0]["weight_imp"]) == 16
        assert len(res[0]["combined_imp"]) == 16

        assert len(res[1]["act_imp"]) == 8
        assert len(res[1]["weight_imp"]) == 8

        # Combined importance should be normalized in [0, 1]
        assert (res[0]["combined_imp"] >= 0.0).all()
        assert (res[0]["combined_imp"] <= 1.0).all()

    def test_classify_neurons_policies(self):
        scores = np.array([0.05, 0.12, 0.35, 0.40, 0.50, 0.65, 0.85, 0.95])

        # Percentile policy
        p_statuses = classify_neurons(scores, policy="percentile", weak_percentile=25, strong_percentile=25)
        assert p_statuses[0] == "weak"
        assert p_statuses[-1] == "strong"

        # Relative policy
        r_statuses = classify_neurons(scores, policy="relative", weak_factor=0.3, strong_percentile=25)
        assert r_statuses[0] == "weak"
        assert r_statuses[-1] == "strong"

    def test_generate_importance_report_csv(self, toy_setup):
        model, loader, loss_fn = toy_setup
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)

        try:
            df = generate_importance_report(
                model=model,
                loader=loader,
                loss_fn=loss_fn,
                device=torch.device("cpu"),
                output_csv=csv_path,
            )
            assert csv_path.exists()
            assert len(df) == 16 + 8  # 24 neurons across both hidden layers
            expected_cols = [
                "layer",
                "neuron_index",
                "activation_importance",
                "weight_importance",
                "combined_importance",
                "rank",
                "status",
            ]
            for col in expected_cols:
                assert col in df.columns
        finally:
            if csv_path.exists():
                csv_path.unlink()
