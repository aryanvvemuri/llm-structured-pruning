"""Reversible channel masking and export-time structural pruning."""
from __future__ import annotations

from contextlib import contextmanager
from numbers import Integral
import torch

from .modeling import mlp_layers


def validate_keep_indices(model, keep_indices):
    mlps = mlp_layers(model)
    if getattr(model, "_prune_lab_compacted", False) or getattr(model.config, "prune_lab_compacted", False):
        raise ValueError("Model has already been compacted; indices refer to the original model")
    if set(keep_indices) != set(range(len(mlps))):
        raise ValueError("Keep indices must cover every decoder layer exactly")
    normalized = {}
    for index, mlp in enumerate(mlps):
        values = list(keep_indices[index])
        width = mlp.gate_proj.out_features
        if not values or any(not isinstance(i, Integral) or isinstance(i, bool) for i in values):
            raise ValueError("Each layer needs nonempty integer neuron indices")
        if len(values) != len(set(values)) or any(i < 0 or i >= width for i in values):
            raise ValueError("Neuron indices must be unique and in the original layer range")
        if mlp.up_proj.out_features != width or mlp.down_proj.in_features != width:
            raise ValueError("Inconsistent MLP projection widths")
        normalized[index] = sorted(int(i) for i in values)
    return normalized


@contextmanager
def masked_neurons(model, keep_indices):
    keep_indices = validate_keep_indices(model, keep_indices)
    handles = []
    try:
        for index, mlp in enumerate(mlp_layers(model)):
            mask = mlp.down_proj.weight.new_zeros(mlp.down_proj.in_features)
            mask[keep_indices[index]] = 1

            def mask_input(module, args, mask=mask):
                return (args[0] * mask, *args[1:])

            handles.append(mlp.down_proj.register_forward_pre_hook(mask_input))
        yield model
    finally:
        for handle in handles:
            handle.remove()


def _select_linear(linear, indices, dimension):
    indices = torch.tensor(indices, device=linear.weight.device, dtype=torch.long)
    weight = linear.weight.detach().index_select(dimension, indices)
    replacement = torch.nn.Linear(weight.shape[1], weight.shape[0],
                                  bias=linear.bias is not None,
                                  device="meta", dtype=weight.dtype)
    replacement.weight = torch.nn.Parameter(weight, requires_grad=False)
    if linear.bias is not None:
        bias = linear.bias.detach()
        if dimension == 0:
            bias = bias.index_select(0, indices)
        replacement.bias = torch.nn.Parameter(bias.clone(), requires_grad=False)
    replacement.train(linear.training)
    return replacement


def compact_model(model, keep_indices):
    keep_indices = validate_keep_indices(model, keep_indices)
    # Build all replacements first so validation/allocation errors cannot half-prune a model.
    replacements = []
    for index, mlp in enumerate(mlp_layers(model)):
        keep = keep_indices[index]
        replacements.append((mlp, _select_linear(mlp.gate_proj, keep, 0),
                             _select_linear(mlp.up_proj, keep, 0),
                             _select_linear(mlp.down_proj, keep, 1)))
    original_widths = [mlp.gate_proj.out_features for mlp in mlp_layers(model)]
    widths = [len(keep_indices[i]) for i in range(len(replacements))]
    for width, (mlp, gate, up, down) in zip(widths, replacements):
        mlp.gate_proj, mlp.up_proj, mlp.down_proj = gate, up, down
        mlp.intermediate_size = width
    # Uniform exports remain directly readable by the stock HF Qwen2 class.
    model.config.intermediate_size = max(widths)
    model.config.prune_lab_layer_widths = widths
    model.config.prune_lab_original_widths = original_widths
    model.config.prune_lab_keep_indices = [keep_indices[i] for i in range(len(widths))]
    model.config.prune_lab_compacted = True
    model._prune_lab_compacted = True
    return model
