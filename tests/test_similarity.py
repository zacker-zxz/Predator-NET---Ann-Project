"""
Unit tests for Neuron Cosine Similarity and Compatibility system.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from predator_net.models.dynamic_mlp import DynamicMLP
from predator_net.predator.similarity import (
    cosine_similarity_1d,
    compute_activation_profiles,
    compute_compatibility,
    find_best_compatible_neuron,
)


class TestNeuronSimilarity:
    def test_cosine_similarity_values(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        v3 = np.array([0.0, 1.0, 0.0])
        v4 = np.array([-1.0, 0.0, 0.0])

        assert abs(cosine_similarity_1d(v1, v2) - 1.0) < 1e-5
        assert abs(cosine_similarity_1d(v1, v3) - 0.0) < 1e-5
        assert abs(cosine_similarity_1d(v1, v4) - (-1.0)) < 1e-5

    def test_activation_profiles_shape(self):
        model = DynamicMLP(input_dim=6, hidden_dims=[10, 5], output_dim=1)
        X = torch.randn(40, 6)
        y = torch.zeros(40, 1)
        loader = DataLoader(TensorDataset(X, y), batch_size=10)

        profiles = compute_activation_profiles(model, loader, layer_idx=0, device=torch.device("cpu"), max_samples=40)
        assert profiles.shape == (40, 10)

    def test_find_best_compatible_neuron_gating(self):
        model = DynamicMLP(input_dim=4, hidden_dims=[5, 3], output_dim=1)
        act_profiles = np.random.randn(30, 5)

        # High threshold -> expected rejection
        res_strict = find_best_compatible_neuron(
            model=model,
            layer_idx=0,
            weak_idx=0,
            strong_indices=[1, 2, 3],
            act_profiles=act_profiles,
            minimum_similarity=0.9999,
        )
        assert res_strict["accepted"] is False
        assert "rejected" in res_strict["reason"]

        # Permissive threshold -> accepted
        res_permissive = find_best_compatible_neuron(
            model=model,
            layer_idx=0,
            weak_idx=0,
            strong_indices=[1, 2, 3],
            act_profiles=act_profiles,
            minimum_similarity=-1.0,
        )
        assert res_permissive["accepted"] is True
        assert res_permissive["strong_idx"] in [1, 2, 3]
