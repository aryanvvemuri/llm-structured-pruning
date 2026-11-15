"""Layer dropping (improvement 8): calibration-loss layer scan, greedy selection, physical removal.

Dropped decoder layers are removed from ``model.model.layers``; ``num_hidden_layers``,
``layer_types`` and every kept layer's ``layer_idx`` are renumbered, so a ``layers-N`` export is
an ordinary smaller Qwen2 checkpoint that the stock Hugging Face class loads. All measurements use
calibration examples only.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import time

LAYER_DROP_KEYS = {"examples", "candidates"}
CANDIDATE_KEYS = {"id", "drop", "total_target"}


def layer_drop_options(config):
    """Validated ``layer_drop`` block (shape-independent checks), or None when absent."""
    options = config.get("layer_drop")
    if options is None:
        return None
    if not isinstance(options, dict) or set(options) - LAYER_DROP_KEYS or "candidates" not in options:
        raise ValueError(f"layer_drop must be an object with candidates (and optional examples): keys {sorted(LAYER_DROP_KEYS)}")
    examples = options.get("examples")
    if examples is not None and (isinstance(examples, bool) or not isinstance(examples, int) or examples < 1):
        raise ValueError("layer_drop.examples must be a positive integer")
    calibration = (config.get("counts") or {}).get("calibration")
    if examples is not None and isinstance(calibration, int) and examples > calibration:
        raise ValueError(f"layer_drop.examples ({examples}) exceeds counts.calibration ({calibration})")
    candidates = options["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("layer_drop.candidates must be a nonempty list")
    ids = []
    for item in candidates:
        if not isinstance(item, dict) or set(item) - CANDIDATE_KEYS or not {"id", "drop"} <= set(item):
            raise ValueError(f"Each layer_drop candidate needs id and drop (optional total_target): {item}")
        if not isinstance(item["id"], str) or not item["id"]:
            raise ValueError("layer_drop candidate ids must be nonempty strings")
        if isinstance(item["drop"], bool) or not isinstance(item["drop"], int) or item["drop"] < 1:
            raise ValueError(f"layer_drop {item['id']}: drop must be an integer >= 1")
        target = item.get("total_target")
        if target is not None and (isinstance(target, bool) or not isinstance(target, (int, float))
                                   or not math.isfinite(target) or not 0 < target < 1):
            raise ValueError(f"layer_drop {item['id']}: total_target must be a fraction in (0, 1)")
        ids.append(item["id"])
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate layer_drop candidate ids: {ids}")
    return {"examples": examples, "candidates": candidates}


def layer_parameters(model):
    """Parameters of each decoder layer (attention, MLP and norms)."""
    return [sum(p.numel() for p in layer.parameters()) for layer in model.model.layers]


def check_shape(config, model):
    """Shape-dependent validation: drop < layers and total_target beyond what dropping removes."""
    from .modeling import count_parameters
    options = layer_drop_options(config)
    if options is None:
        return {}
    layers = len(model.model.layers)
    total = count_parameters(model)
    per_layer = max(layer_parameters(model))
    result = {}
    for item in options["candidates"]:
        if item["drop"] >= layers:
            raise ValueError(f"layer_drop {item['id']}: drop {item['drop']} must be smaller than the {layers} layers")
        target = item.get("total_target")
        if target is not None and target * total <= item["drop"] * per_layer:
            raise ValueError(f"layer_drop {item['id']}: total_target {target} does not exceed the "
                             f"{item['drop'] * per_layer / total:.4f} already removed by dropping {item['drop']} layers")
        if target is not None:  # The remaining uniform MLP cut must respect the per-layer keep floor.
            widths = [layer.mlp.down_proj.in_features for layer in model.model.layers]
            per_neuron = [3 * layer.mlp.gate_proj.in_features for layer in model.model.layers]
            floor = [math.ceil(w * config["min_keep_ratio"]) for w in widths]
            removable = sum(sorted((w - f) * c for w, f, c in zip(widths, floor, per_neuron))[item["drop"]:])
            if target * total - item["drop"] * per_layer > removable:
                raise ValueError(f"layer_drop {item['id']}: total_target {target} is not reachable after dropping "
                                 f"{item['drop']} layers with min_keep_ratio {config['min_keep_ratio']}")
        result[item["id"]] = {"layers": layers, "drop": item["drop"]}
    return result


def check_shape_from_config(config):
    """``check_shape`` on a weightless meta model built from the source config; None if unreadable offline."""
    import torch
    from transformers import AutoConfig
    from .modeling import CompactQwen2ForCausalLM
    if config.get("layer_drop") is None:
        return {}
    try:
        source = AutoConfig.from_pretrained(config["model"], cache_dir=config.get("cache_dir"),
                                            local_files_only=True, trust_remote_code=False)
    except (OSError, ValueError):
        return None
    with torch.device("meta"):
        model = CompactQwen2ForCausalLM(source)
    return check_shape(config, model)


def _renumber(model, kept):
    """Make ``model`` contain exactly the original layers in ``kept`` (original indices), renumbered."""
    import torch
    layers = [model.model.layers[i] for i in kept]
    for new, layer in enumerate(layers):
        layer.self_attn.layer_idx = new
    model.model.layers = torch.nn.ModuleList(layers)
    model.config.num_hidden_layers = len(layers)
    if getattr(model.config, "layer_types", None) is not None:
        model.config.layer_types = [layer.attention_type for layer in layers]


def drop_layers(model, dropped):
    """Physically remove decoder layers (original indices) and record provenance in the config."""
    total = len(model.model.layers)
    dropped = sorted(int(i) for i in dropped)
    if not dropped or len(set(dropped)) != len(dropped) or dropped[0] < 0 or dropped[-1] >= total or len(dropped) >= total:
        raise ValueError("Dropped layers must be distinct, in range and leave at least one layer")
    if getattr(model.config, "prune_lab_layer_widths", None) is not None or getattr(model.config, "prune_lab_dropped_layers", None):
        raise ValueError("Drop layers on the original model, before MLP pruning or a previous drop")
    kept = [i for i in range(total) if i not in dropped]
    _renumber(model, kept)
    model.config.prune_lab_dropped_layers = dropped
    model.config.prune_lab_kept_layers = kept
    model.config.prune_lab_original_num_hidden_layers = total
    return kept


@contextmanager
def without_layers(model, dropped):
    """Temporarily run ``model`` without the given layers (current indices); fully restored after."""
    original_layers = model.model.layers
    original = (model.config.num_hidden_layers, getattr(model.config, "layer_types", None))
    indices = [layer.self_attn.layer_idx for layer in original_layers]
    kept = [i for i in range(len(original_layers)) if i not in set(dropped)]
    try:
        _renumber(model, kept)
        yield model
    finally:
        model.model.layers = original_layers
        for layer, index in zip(original_layers, indices):
            layer.self_attn.layer_idx = index
        model.config.num_hidden_layers, layer_types = original
        if layer_types is not None:
            model.config.layer_types = layer_types


def calibration_loss(model, batches):
    """Token-weighted answer-token cross entropy over ``batches`` (no gradients, eval mode)."""
    import torch
    device = next(model.parameters()).device
    training = [(m, m.training) for m in model.modules()]
    total = tokens = 0.0
    try:
        model.eval()
        with torch.no_grad():
            for batch in batches:
                inputs = {k: batch[k].to(device) for k in ("input_ids", "attention_mask", "labels") if k in batch}
                labels = inputs["labels"].clone()
                if "attention_mask" in inputs:
                    labels.masked_fill_(~inputs["attention_mask"].bool(), -100)
                n = int((labels[:, 1:] != -100).sum().item())
                if not n:
                    raise ValueError("Layer scan requires shifted supervised tokens")
                inputs["labels"] = labels
                value = float(model(**inputs, use_cache=False).loss.item())
                if not math.isfinite(value):
                    raise ValueError("Nonfinite calibration loss in the layer scan")
                total += value * n
                tokens += n
    finally:
        for module, mode in training:
            module.training = mode
    return total / tokens


def scan_and_select(model, batches, steps):
    """Single-layer scan on the intact model, then greedy removal for ``steps`` layers.

    Each greedy step removes, among the remaining layers, the one whose removal gives the lowest
    calibration loss in the already-reduced network (ties toward the higher original index). The
    first greedy step is the single-layer scan itself. Returns a JSON-safe record (lists only).
    """
    start = time.perf_counter()
    layers = len(model.model.layers)
    if not 1 <= steps < layers:
        raise ValueError("Greedy layer selection needs 1 <= steps < number of layers")
    baseline = calibration_loss(model, batches)
    removed, greedy, single = [], [], None
    for step in range(steps):
        remaining = [i for i in range(layers) if i not in removed]
        losses = []
        for layer in remaining:
            with without_layers(model, removed + [layer]):
                losses.append(calibration_loss(model, batches))
        if step == 0:
            single = [{"layer": layer, "loss": loss, "increase": loss - baseline} for layer, loss in zip(remaining, losses)]
        best = min(range(len(remaining)), key=lambda k: (losses[k], -remaining[k]))
        removed.append(remaining[best])
        greedy.append({"step": step + 1, "removed_layer": remaining[best], "loss": losses[best],
                       "increase": losses[best] - baseline, "evaluated": len(remaining)})
    return {"examples": len(batches), "baseline_loss": baseline, "single_layer": single, "greedy": greedy,
            "order": removed, "seconds": time.perf_counter() - start,
            "note": "Calibration split only; original pretrained model; ties broken toward the higher layer index."}
