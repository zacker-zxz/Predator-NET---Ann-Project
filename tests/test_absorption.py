"""
Unit tests for PREDATOR Neuron Absorption operators (Light & Full).
"""

import pytest
import torch
import numpy as np

from predator_net.models.dynamic_mlp import DynamicMLP
from predator_net.predator.absorption import compute_alpha, apply_absorption_weights


class TestNeuronAbsorption:
    def test_compute_alpha_clipping(self):
        # Very high ratio -> should clip to alpha_max
        a_max = compute_alpha(compatibility=1.0, imp_w=100.0, imp_s=1.0, alpha_min=0.05, alpha_max=0.50)
        assert abs(a_max - 0.50) < 1e-5

        # Very low ratio -> should clip to alpha_min
        a_min = compute_alpha(compatibility=0.5, imp_w=0.001, imp_s=10.0, alpha_min=0.05, alpha_max=0.50)
        assert abs(a_min - 0.05) < 1e-5

    def test_predator_light_weight_transfer(self):
        """Spec §26: U_s_new = U_s + α * U_w on next layer's weights."""
        model = DynamicMLP(input_dim=4, hidden_dims=[3, 2], output_dim=1)

        # Set known outgoing weights in layer 1 (shape: (2, 3))
        # Column 0 = neuron 0 (weak), Column 1 = neuron 1 (strong)
        with torch.no_grad():
            model.linears[1].weight.data.copy_(
                torch.tensor([[1.0, 2.0, 0.5],
                              [3.0, 4.0, 0.5]])
            )

        u_w_initial = model.linears[1].weight[:, 0].clone()
        u_s_initial = model.linears[1].weight[:, 1].clone()
        alpha = 0.25

        apply_absorption_weights(
            model=model,
            layer_idx=0,
            weak_idx=0,
            strong_idx=1,
            alpha=alpha,
            mode="light",
        )

        expected_u_s = u_s_initial + alpha * u_w_initial
        torch.testing.assert_close(model.linears[1].weight[:, 1], expected_u_s)

    def test_predator_full_incoming_and_bias_transfer(self):
        """Spec §27: V_s_new = (1-γ)*V_s + γ*V_w and b_s_new = (1-γ)*b_s + γ*b_w."""
        model = DynamicMLP(input_dim=2, hidden_dims=[3, 2], output_dim=1)

        with torch.no_grad():
            model.linears[0].weight.data.copy_(
                torch.tensor([[10.0, 20.0],   # weak row 0
                              [30.0, 40.0],   # strong row 1
                              [50.0, 60.0]])
            )
            model.linears[0].bias.data.copy_(torch.tensor([1.0, 5.0, 9.0]))

        gamma = 0.30
        apply_absorption_weights(
            model=model,
            layer_idx=0,
            weak_idx=0,
            strong_idx=1,
            alpha=0.10,
            mode="full",
            gamma=gamma,
        )

        # Incoming weight expected: (1 - 0.3)*30 + 0.3*10 = 21 + 3 = 24
        # (1 - 0.3)*40 + 0.3*20 = 28 + 6 = 34
        expected_v_s = torch.tensor([24.0, 34.0])
        torch.testing.assert_close(model.linears[0].weight[1], expected_v_s)

        # Bias expected: (1 - 0.3)*5.0 + 0.3*1.0 = 3.5 + 0.3 = 3.8
        expected_b_s = torch.tensor(3.8)
        torch.testing.assert_close(model.linears[0].bias[1], expected_b_s)
