"""Verify a real exported checkpoint against its reversible mask, one model at a time."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

from prune_lab.cli import experiment_lock, prepare, read_config
from prune_lab.data import load_splits, make_batches, prompt_ids
from prune_lab.modeling import count_parameters, load_model
from prune_lab.pruning import masked_neurons
from prune_lab.reporting import write_json


def parameter_hashes(model, keep=None):
    """Hash one tensor at a time; derive expected slices directly from the source."""
    hashes = {}
    for name, parameter in model.named_parameters():
        value = parameter.detach()
        if keep is not None and ".mlp." in name:
            layer = int(name.split(".layers.")[1].split(".")[0])
            indices = torch.tensor(keep[layer], device=value.device)
            if name.endswith(("gate_proj.weight", "up_proj.weight")):
                value = value.index_select(0, indices)
            elif name.endswith("down_proj.weight"):
                value = value.index_select(1, indices)
        value = value.cpu().contiguous()
        hashes[name] = {"shape": list(value.shape), "sha256": hashlib.sha256(memoryview(value.numpy())).hexdigest()}
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--float64", action="store_true", help="CPU-only higher precision diagnostic for matrix reduction rounding")
    args = parser.parse_args()
    config = read_config(args.config)
    result = json.loads(Path(args.result).read_text())
    keep = {int(k): v for k, v in result["search"]["keep_indices"].items()}
    with experiment_lock():
        model, tokenizer = prepare(config)
        expected_hashes = parameter_hashes(model, keep)
        if args.float64:
            if next(model.parameters()).device.type != "cpu":
                raise ValueError("float64 diagnostic requires CPU")
            model.double()
        examples = load_splits(config)["calibration"][:3]
        batches = make_batches(examples, tokenizer, config["max_length"])
        expected_tokens = [prompt_ids(tokenizer, ex.prompt) for ex in examples]
        device = next(model.parameters()).device
        original_parameters = count_parameters(model)
        expected = []
        with masked_neurons(model, keep), torch.inference_mode():
            for batch in batches:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                expected.append(model(**inputs, use_cache=False).logits.cpu())
        del model, inputs
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        compact, compact_tokenizer = load_model(args.checkpoint, config["device"], config["dtype"])
        actual_hashes = parameter_hashes(compact)
        if expected_hashes != actual_hashes:
            raise AssertionError("Exported parameter tensors do not match expected original slices")
        if args.float64:
            compact.double()
        assert expected_tokens == [prompt_ids(compact_tokenizer, ex.prompt) for ex in examples]
        maximum_error = 0.0
        with torch.inference_mode():
            for batch, target in zip(batches, expected):
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                actual = compact(**inputs, use_cache=False).logits.cpu()
                maximum_error = max(maximum_error, (actual - target).abs().max().item())
                torch.testing.assert_close(actual, target, atol=1e-4, rtol=1e-4)
        compact_parameters = count_parameters(compact)
        payload = {
            "passed": True,
            "examples_checked": len(examples),
            "tokenization_preserved": True,
            "exact_parameter_hashes_match": True,
            "parameter_tensors_checked": len(actual_hashes),
            "masked_vs_reloaded_max_abs_logit_difference": maximum_error,
            "atol": 1e-4,
            "rtol": 1e-4,
            "original_parameters": original_parameters,
            "compact_parameters": compact_parameters,
            "parameter_reduction_percent": 100 * (1 - compact_parameters / original_parameters),
            "device": str(device),
            "verification_dtype": str(next(compact.parameters()).dtype),
        }
        write_json(args.output, payload)
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
