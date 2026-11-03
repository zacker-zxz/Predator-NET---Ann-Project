"""PREDATOR neuron importance, compatibility, absorption, and controller modules."""

from predator_net.predator.importance import (
    compute_layer_neuron_importance,
    classify_neurons,
    generate_importance_report,
)
from predator_net.predator.similarity import (
    compute_activation_profiles,
    compute_compatibility,
    find_best_compatible_neuron,
)
from predator_net.predator.absorption import compute_alpha, apply_absorption_weights
from predator_net.predator.structural_rebuild import remove_neuron_structurally
from predator_net.predator.distillation import DistillationLoss
from predator_net.predator.controller import predator_step, run_predator_loop

__all__ = [
    "compute_layer_neuron_importance",
    "classify_neurons",
    "generate_importance_report",
    "compute_activation_profiles",
    "compute_compatibility",
    "find_best_compatible_neuron",
    "compute_alpha",
    "apply_absorption_weights",
    "remove_neuron_structurally",
    "DistillationLoss",
    "predator_step",
    "run_predator_loop",
]
