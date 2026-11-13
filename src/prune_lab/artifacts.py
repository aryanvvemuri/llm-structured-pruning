"""Portable Hugging Face checkpoints and JSON provenance."""
from __future__ import annotations

import json
from pathlib import Path

from .modeling import layer_widths, mlp_layers


def save_checkpoint(model, tokenizer, output_dir, manifest: dict | None = None):
    """Save safetensors plus architecture provenance.

    Compaction stages all selected projection tensors before swapping modules,
    so its peak includes the dense model plus retained FFN weights. Export on
    CPU, when the caller has placed the model there, avoids additional MPS
    serialization copies. This function leaves the device unchanged and makes
    no full state-dict clone. Heterogeneous exports require the project loader.
    """
    output = Path(output_dir)
    manifest = dict(manifest or {})
    widths = layer_widths(model.config)
    if widths is not None:
        actual = [m.gate_proj.out_features for m in mlp_layers(model)]
        if actual != widths or any(m.up_proj.out_features != w or m.down_proj.in_features != w
                                   for m, w in zip(mlp_layers(model), widths)):
            raise ValueError("Checkpoint MLP shapes do not match saved layer widths")
        manifest["architecture"] = {
            "layer_widths": widths,
            "original_widths": getattr(model.config, "prune_lab_original_widths", None),
            "keep_indices": getattr(model.config, "prune_lab_keep_indices", None),
            "loader": "prune_lab.modeling.load_model" if len(set(widths)) > 1 else "transformers.AutoModelForCausalLM",
        }
    # Validate serialization before writing any weights.
    payload = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    (output / "manifest.json").write_text(payload, encoding="utf-8")
    return output
