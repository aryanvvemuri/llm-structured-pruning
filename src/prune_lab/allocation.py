"""Calibration-only layer sensitivity and total-parameter budget allocation."""
from __future__ import annotations

import heapq
import math
from numbers import Integral

import torch

from .modeling import count_parameters, mlp_layers
from .pruning import masked_neurons, validate_keep_indices
from .scoring import score_neurons


def _validated_scores(model, scores):
    mlps = mlp_layers(model)
    validate_keep_indices(model, {i: list(range(m.down_proj.in_features))
                                  for i, m in enumerate(mlps)})
    if set(scores) != set(range(len(mlps))):
        raise ValueError("Scores must cover every layer")
    result = {}
    for i, mlp in enumerate(mlps):
        value = torch.as_tensor(scores[i]).detach().cpu().double()
        if (value.shape != (mlp.down_proj.in_features,) or
                not torch.isfinite(value).all() or (value < 0).any()):
            raise ValueError("Scores must be finite nonnegative vectors matching layer widths")
        result[i] = value
    return result


def measure_sensitivity(model, batches, probe_fraction=0.10, scores=None):
    """Change in token-weighted answer CE from probing each layer separately.

    Batches are retained on CPU for repeated passes, never activation graphs.
    Negative deltas are returned faithfully; allocation clamps their penalty.
    """
    if not math.isfinite(probe_fraction) or not 0 < probe_fraction < 1:
        raise ValueError("probe_fraction must be between zero and one")
    batches = [{k: v.detach().cpu() for k, v in batch.items()} for batch in batches]
    if not batches:
        raise ValueError("No calibration batches supplied")
    scores = _validated_scores(model, scores if scores is not None else
                               score_neurons(model, batches, "taylor"))
    training = [(m, m.training) for m in model.modules()]
    cache = model.config.use_cache
    device = next(model.parameters()).device
    full = {i: list(range(len(s))) for i, s in scores.items()}

    def loss():
        total, tokens = 0.0, 0
        for batch in batches:
            if "labels" not in batch:
                raise ValueError("Sensitivity requires answer-only labels")
            inputs = {k: batch[k].to(device) for k in ("input_ids", "attention_mask", "labels") if k in batch}
            labels = inputs["labels"].clone()
            if "attention_mask" in inputs:
                labels.masked_fill_(~inputs["attention_mask"].bool(), -100)
            n = int((labels[:, 1:] != -100).sum().item())
            if not n:
                raise ValueError("Sensitivity requires shifted supervised tokens")
            inputs["labels"] = labels
            value = float(model(**inputs, use_cache=False).loss.item())
            if not math.isfinite(value):
                raise ValueError("Nonfinite sensitivity loss")
            total += value * n
            tokens += n
        return total / tokens

    try:
        model.eval()
        model.config.use_cache = False
        with torch.no_grad():
            baseline = loss()
            result = {}
            for i, values in scores.items():
                if len(values) < 2:
                    raise ValueError("Sensitivity probe requires at least two neurons per layer")
                drop = min(len(values) - 1, max(1, round(len(values) * probe_fraction)))
                keep = dict(full)
                keep[i] = values.argsort(stable=True)[drop:].tolist()
                with masked_neurons(model, keep):
                    result[i] = loss() - baseline
            return result
    finally:
        model.config.use_cache = cache
        for module, was_training in training:
            module.training = was_training


def allocate_neurons(model, scores, sensitivities, target_reduction,
                     group_size=64, min_keep_ratio=0.25, uniform=False):
    """Allocate a fraction of ALL original parameters, using only FFN neurons.

    Cross-layer priorities use measured CE sensitivity, not raw Taylor scales.
    Within each layer stable Taylor ranks select removals; increasing percentile
    penalties discourage concentrating all removals in one insensitive layer.
    The nearest reachable budget is used (within one group, or one uniform step).
    """
    scores = _validated_scores(model, scores)
    if (not isinstance(group_size, Integral) or isinstance(group_size, bool) or group_size < 1
            or not math.isfinite(min_keep_ratio) or not 0 < min_keep_ratio <= 1
            or not math.isfinite(target_reduction) or not 0 <= target_reduction < 1):
        raise ValueError("Invalid group size, keep ratio, or target reduction")
    if set(sensitivities) != set(scores) or any(not math.isfinite(float(v)) for v in sensitivities.values()):
        raise ValueError("Sensitivities must be finite and cover every layer")
    mlps = mlp_layers(model)
    widths = [len(scores[i]) for i in scores]
    costs = [m.gate_proj.in_features + m.up_proj.in_features + m.down_proj.out_features
             + int(m.gate_proj.bias is not None) + int(m.up_proj.bias is not None) for m in mlps]
    # Keep aligned widths where possible, never crossing the per-layer floor.
    choices = []
    for width in widths:
        floor = max(1, math.ceil(width * min_keep_ratio))
        choices.append([width] + list(range(((width - 1) // group_size) * group_size,
                                           floor - 1, -group_size)))
    target = count_parameters(model) * target_reduction
    maximum = sum((width - options[-1]) * cost for width, options, cost in zip(widths, choices, costs))
    if target > maximum + 1e-8:
        raise ValueError(f"Infeasible total-parameter reduction: requested {target_reduction:.4f}, maximum {maximum/count_parameters(model):.4f}")
    selected = [0] * len(widths)
    if uniform:
        if len(set(widths)) != 1:
            raise ValueError("Uniform allocation requires equal original widths")
        step = min(range(len(choices[0])), key=lambda j: abs(sum(costs) * (widths[0] - choices[0][j]) - target))
        selected = [step] * len(widths)
    else:
        scale = max([max(0.0, float(v)) for v in sensitivities.values()] + [1e-8])
        penalties = [max(float(sensitivities[i]), scale * 0.01) for i in scores]
        queue = []

        def push(i):
            j = selected[i]
            if j + 1 < len(choices[i]):
                removed = widths[i] - choices[i][j]
                count = choices[i][j] - choices[i][j + 1]
                percentile = (removed + count / 2) / widths[i]
                priority = penalties[i] * (1 + percentile) ** 2 / (widths[i] * costs[i])
                heapq.heappush(queue, (priority, i, count * costs[i]))

        for i in scores:
            push(i)
        achieved = 0
        while queue and achieved < target:
            _, i, cost = heapq.heappop(queue)
            if abs(achieved + cost - target) >= abs(achieved - target):
                continue
            achieved += cost
            selected[i] += 1
            push(i)
    return {i: sorted(scores[i].argsort(stable=True)[widths[i] - choices[i][selected[i]]:].tolist())
            for i in scores}


# --- Improvement 1 (opt-in): sensitivity curves and alternative allocators ---------------------

CURVE_FRACTIONS = (0.10, 0.25, 0.50)
NEW_MIN_KEEP_RATIO = 0.5


def masked_calibration_loss(model, batches, keep=None):
    """Token-weighted answer CE on calibration batches, optionally under a reversible neuron mask."""
    batches = [{k: v.detach().cpu() for k, v in batch.items()} for batch in batches]
    if not batches:
        raise ValueError("No calibration batches supplied")
    device = next(model.parameters()).device
    training = [(m, m.training) for m in model.modules()]
    cache = model.config.use_cache

    def loss():
        total, tokens = 0.0, 0
        for batch in batches:
            inputs = {k: batch[k].to(device) for k in ("input_ids", "attention_mask", "labels") if k in batch}
            if "labels" not in inputs:
                raise ValueError("Calibration loss requires answer-only labels")
            labels = inputs["labels"].clone()
            if "attention_mask" in inputs:
                labels.masked_fill_(~inputs["attention_mask"].bool(), -100)
            n = int((labels[:, 1:] != -100).sum().item())
            if not n:
                raise ValueError("Calibration loss requires shifted supervised tokens")
            inputs["labels"] = labels
            value = float(model(**inputs, use_cache=False).loss.item())
            if not math.isfinite(value):
                raise ValueError("Nonfinite calibration loss")
            total += value * n
            tokens += n
        return total / tokens

    try:
        model.eval()
        model.config.use_cache = False
        with torch.no_grad():
            if keep is None:
                return loss()
            with masked_neurons(model, keep):
                return loss()
    finally:
        model.config.use_cache = cache
        for module, was_training in training:
            module.training = was_training


def measure_sensitivity_curves(model, batches, scores, fractions=CURVE_FRACTIONS):
    """Raw loss increase per layer when removing each fraction of its lowest-Taylor neurons.

    Returns {"baseline": loss, "raw": {layer: [increase per fraction]}}; nothing is clamped here.
    """
    fractions = tuple(float(f) for f in fractions)
    if not fractions or any(not 0 < f < 1 for f in fractions) or list(fractions) != sorted(set(fractions)):
        raise ValueError("Probe fractions must be strictly increasing values in (0, 1)")
    scores = _validated_scores(model, scores)
    full = {i: list(range(len(s))) for i, s in scores.items()}
    baseline = masked_calibration_loss(model, batches)
    raw = {}
    for i, values in scores.items():
        order = values.argsort(stable=True)
        raw[i] = []
        for fraction in fractions:
            drop = min(len(values) - 1, max(1, round(len(values) * fraction)))
            keep = dict(full)
            keep[i] = order[drop:].tolist()
            raw[i].append(masked_calibration_loss(model, batches, keep) - baseline)
    return {"baseline": baseline, "raw": raw}


def process_curve(raw):
    """Clamp negative increases to zero, then take the running maximum (non-decreasing)."""
    result, running = [], 0.0
    for value in raw:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Sensitivity curve values must be finite")
        running = max(running, max(0.0, value))
        result.append(running)
    return result


def curve_value(curve, fractions, x):
    """Piecewise-linear interpolation through (0, 0) and (fraction_k, curve_k)."""
    xs, ys = [0.0] + list(fractions), [0.0] + list(curve)
    if x < -1e-12 or x > xs[-1] + 1e-12:
        raise ValueError("Removal fraction outside the probed range")
    for k in range(1, len(xs)):
        if x <= xs[k] + 1e-12:
            t = (x - xs[k - 1]) / (xs[k] - xs[k - 1])
            return ys[k - 1] + min(1.0, max(0.0, t)) * (ys[k] - ys[k - 1])
    return ys[-1]


def _layout(model, scores, group_size, min_keep_ratio):
    if (not isinstance(group_size, Integral) or isinstance(group_size, bool) or group_size < 1
            or not math.isfinite(min_keep_ratio) or not 0 < min_keep_ratio <= 1):
        raise ValueError("Invalid group size or keep ratio")
    mlps = mlp_layers(model)
    widths = [len(scores[i]) for i in scores]
    costs = [m.gate_proj.in_features + m.up_proj.in_features + m.down_proj.out_features
             + int(m.gate_proj.bias is not None) + int(m.up_proj.bias is not None) for m in mlps]
    choices = []
    for width in widths:
        floor = max(1, math.ceil(width * min_keep_ratio))
        choices.append([width] + list(range(((width - 1) // group_size) * group_size, floor - 1, -group_size)))
    return widths, costs, choices


def _greedy_allocate(model, scores, target_reduction, group_size, min_keep_ratio, marginal):
    """Greedy over aligned groups by (marginal(i, removed, count) per parameter, fraction removed, layer).

    Uses the legacy budget rule: a group that would move the total further from the target is
    skipped and its layer leaves the queue. Within a layer, neurons go in stable Taylor order.
    """
    if not math.isfinite(target_reduction) or not 0 <= target_reduction < 1:
        raise ValueError("Invalid target reduction")
    widths, costs, choices = _layout(model, scores, group_size, min_keep_ratio)
    target = count_parameters(model) * target_reduction
    maximum = sum((w - c[-1]) * cost for w, c, cost in zip(widths, choices, costs))
    if target > maximum + 1e-8:
        raise ValueError(f"Infeasible total-parameter reduction: requested {target_reduction:.4f}, "
                         f"maximum {maximum / count_parameters(model):.4f}")
    selected = [0] * len(widths)
    queue = []

    def push(i):
        j = selected[i]
        if j + 1 < len(choices[i]):
            removed = widths[i] - choices[i][j]
            count = choices[i][j] - choices[i][j + 1]
            priority = marginal(i, removed, count) / (count * costs[i])
            heapq.heappush(queue, (priority, removed / widths[i], i, count * costs[i]))

    for i in range(len(widths)):
        push(i)
    achieved = 0
    while queue and achieved < target:
        _, _, i, cost = heapq.heappop(queue)
        if abs(achieved + cost - target) >= abs(achieved - target):
            continue
        achieved += cost
        selected[i] += 1
        push(i)
    keys = list(scores)
    return {keys[i]: sorted(scores[keys[i]].argsort(stable=True)[widths[i] - choices[i][selected[i]]:].tolist())
            for i in range(len(widths))}


def allocate_curve(model, scores, curves, target_reduction, group_size=64, min_keep_ratio=NEW_MIN_KEEP_RATIO,
                   fractions=CURVE_FRACTIONS):
    """1a: smallest marginal loss increase per removed parameter, read from processed curves."""
    scores = _validated_scores(model, scores)
    fractions = tuple(float(f) for f in fractions)
    if set(curves) != set(scores) or any(len(curves[i]) != len(fractions) for i in scores):
        raise ValueError("Curves must cover every layer with one value per probe fraction")
    if 1 - min_keep_ratio > fractions[-1] + 1e-12:
        raise ValueError("The per-layer removal cap exceeds the largest probed fraction")
    processed = [process_curve(curves[i]) for i in scores]
    widths = [len(scores[i]) for i in scores]

    def marginal(i, removed, count):
        return (curve_value(processed[i], fractions, (removed + count) / widths[i])
                - curve_value(processed[i], fractions, removed / widths[i]))
    return _greedy_allocate(model, scores, target_reduction, group_size, min_keep_ratio, marginal)


def standardized_scores(scores):
    """Per-layer (score - mean) / population sd; a constant layer standardizes to zeros."""
    result = {}
    for i, value in scores.items():
        value = torch.as_tensor(value).detach().cpu().double()
        sd = float(value.std(unbiased=False))
        result[i] = (value - value.mean()) / sd if sd > 0 else torch.zeros_like(value)
    return result


def allocate_global_standardized(model, scores, target_reduction, group_size=64, min_keep_ratio=NEW_MIN_KEEP_RATIO):
    """1b (FLAP-style): rank groups of the lowest standardized Taylor scores globally.

    A layer's next group (its next-lowest neurons in Taylor order, which standardization keeps)
    is ranked by the mean standardized score of the neurons in that group.
    """
    scores = _validated_scores(model, scores)
    z = standardized_scores(scores)
    ordered = [z[i][scores[i].argsort(stable=True)].tolist() for i in scores]
    widths, costs, _ = _layout(model, scores, group_size, min_keep_ratio)

    def marginal(i, removed, count):
        # Scaled by the group's parameter count so _greedy_allocate's per-parameter division
        # leaves the mean standardized score as the priority.
        return sum(ordered[i][removed:removed + count]) / count * count * costs[i]
    return _greedy_allocate(model, scores, target_reduction, group_size, min_keep_ratio, marginal)


def removal_fractions(scores, keep):
    return [1 - len(keep[i]) / len(scores[i]) for i in sorted(scores)]
