"""
Unit tests for Structural Rebuild module.

Spec Rule #19: Test structural modification on a tiny toy network (e.g. 4 -> 3 -> 2 -> 1)
BEFORE applying to CIC-IDS2017.
"""

import pytest
import torch

from predator_net.models.dynamic_mlp import DynamicMLP
from predator_net.models.model_utils import count_parameters
from predator_net.predator.structural_rebuild import remove_neuron_structurally


class TestStructuralRebuildToyNetwork:
    def test_toy_network_4_3_2_1_neuron_removal(self):
        """
        Architecture: 4 -> 3 -> 2 -> 1
        Total params = (4*3 + 3) + (3*2 + 2) + (2*1 + 1) = 15 + 8 + 3 = 26 params.
        Removing neuron 1 from hidden layer 0:
          New Layer 0: 4 -> 2 (10 params)
          New Layer 1: 2 -> 2 (6 params)
          Layer 2: 2 -> 1 (3 params)
          New Total params = 19 params (delta = -7).
        """
        model = DynamicMLP(input_dim=4, hidden_dims=[3, 2], output_dim=1)
        initial_params = count_parameters(model)["total_parameters"]
        assert initial_params == 26

        # Forward pass before modification
        x = torch.randn(8, 4)
        out_before = model(x)
        assert out_before.shape == (8, 1)

        # Structurally remove neuron 1 from layer 0
        remove_neuron_structurally(model, layer_idx=0, neuron_idx=1)

        # Verify new hidden dimensions
        assert model.hidden_dims == [2, 2]
        assert model.linears[0].weight.shape == (2, 4)
        assert model.linears[0].bias.shape == (2,)
        assert model.linears[1].weight.shape == (2, 2)
        assert model.linears[1].bias.shape == (2,)
        assert model.linears[2].weight.shape == (1, 2)

        # Verify strict parameter reduction
        new_params = count_parameters(model)["total_parameters"]
        assert new_params == 19
        assert new_params < initial_params

        # Forward pass after structural rebuild
        out_after = model(x)
        assert out_after.shape == (8, 1)
        assert not torch.isnan(out_after).any()
        assert not torch.isinf(out_after).any()

    def test_removal_from_second_hidden_layer(self):
        """
        Architecture: 4 -> 3 -> 2 -> 1
        Removing neuron 0 from hidden layer 1 (layer_idx=1):
          Layer 0: 4 -> 3 (untouched)
          Layer 1: 3 -> 1 (width reduced from 2 to 1)
          Layer 2: 1 -> 1
        """
        model = DynamicMLP(input_dim=4, hidden_dims=[3, 2], output_dim=1)
        remove_neuron_structurally(model, layer_idx=1, neuron_idx=0)

        assert model.hidden_dims == [3, 1]
        assert model.linears[0].weight.shape == (3, 4)
        assert model.linears[1].weight.shape == (1, 3)
        assert model.linears[2].weight.shape == (1, 1)

        x = torch.randn(4, 4)
        out = model(x)
        assert out.shape == (4, 1)

    def test_invalid_neuron_index_raises(self):
        model = DynamicMLP(input_dim=4, hidden_dims=[3, 2], output_dim=1)
        with pytest.raises(IndexError):
            remove_neuron_structurally(model, layer_idx=0, neuron_idx=99)
