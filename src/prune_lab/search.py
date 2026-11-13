"""Reversible, uniform-width greedy search; this module never accepts test data."""
import math
import random
from .pruning import masked_neurons
from .scoring import score_neurons


def _validate_metrics(metrics):
    if any(not math.isfinite(metrics[key]) for key in ("accuracy", "answer_token_loss")):
        raise ValueError("Development metrics must be finite")
    if not 0 <= metrics["accuracy"] <= 1 or metrics["answer_token_loss"] < 0:
        raise ValueError("Development accuracy/loss outside valid range")


def passes_guardrail(candidate, baseline, max_drop_pp, loss_tolerance=None):
    _validate_metrics(baseline)
    _validate_metrics(candidate)
    accuracy_ok = candidate["accuracy"] >= baseline["accuracy"] - max_drop_pp / 100.0 - 1e-12
    loss_ok = loss_tolerance is None or candidate["answer_token_loss"] <= baseline["answer_token_loss"] + loss_tolerance
    return accuracy_ok and loss_ok


def search_masks(model, calibration_batches, evaluate_dev, method="taylor", initial_step=0.1, min_step=1, max_rounds=5, max_drop_pp=0.0, loss_tolerance=None, min_keep_ratio=0.5, seed=17):
    if method not in {"activation", "contribution", "taylor", "random"}:
        raise ValueError("Unknown scoring method")
    bounds = [initial_step, min_step, max_rounds, max_drop_pp, min_keep_ratio]
    if loss_tolerance is not None:
        bounds.append(loss_tolerance)
    if not all(math.isfinite(value) for value in bounds):
        raise ValueError("Search bounds/tolerances must be finite")
    if not 0 < initial_step < 1 or not 0 < min_keep_ratio <= 1 or min_step < 1 or max_rounds < 1 or max_drop_pp < 0 or (loss_tolerance is not None and loss_tolerance < 0):
        raise ValueError("Invalid search bounds/tolerances")
    width, layers = model.config.intermediate_size, model.config.num_hidden_layers
    keep = {layer: list(range(width)) for layer in range(layers)}
    minimum = max(1, math.ceil(width * min_keep_ratio))
    step = max(min_step, int(width * initial_step))
    baseline = evaluate_dev()
    _validate_metrics(baseline)
    current, history, scores = baseline, [], None
    rng = random.Random(seed)
    for round_index in range(max_rounds):
        current_width = len(keep[0])
        remove = min(step, current_width - minimum)
        if remove < min_step:
            break
        if scores is None:
            if method == "random":
                scores = {layer: [rng.random() for _ in range(width)] for layer in keep}
            else:
                with masked_neurons(model, keep):
                    scores = score_neurons(model, calibration_batches, method=method)
        candidate = {layer: sorted(sorted(indices, key=lambda i: (float(scores[layer][i]), i))[remove:]) for layer, indices in keep.items()}
        with masked_neurons(model, candidate):
            metrics = evaluate_dev()
        accepted = passes_guardrail(metrics, baseline, max_drop_pp, loss_tolerance)
        history.append({"round": round_index + 1, "retained_width": len(candidate[0]), "removed_per_layer": remove, "accepted": accepted, "dev": metrics})
        if accepted:
            keep, current, scores = candidate, metrics, None
        else:
            if step <= min_step:
                break
            step = max(min_step, step // 2)
    return {"keep_indices": keep, "baseline_dev": baseline, "selected_dev": current, "history": history, "method": method, "max_drop_pp": max_drop_pp, "loss_tolerance": loss_tolerance, "original_width": width, "retained_width": len(keep[0]), "test_used_for_search": False}
