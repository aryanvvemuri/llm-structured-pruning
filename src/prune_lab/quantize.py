"""Optional MLX conversion, quantization and evaluation of exported candidates (improvement 4).

Requires the optional ``mlx-lm`` package; importing this module (and ``prune_lab``) works without
it, and every entry point raises a clear error when it is missing. Generation reproduces the
PyTorch support path: same prompt template and token ids (Hugging Face tokenizer), greedy
decoding, the same ``max_new_tokens``, the export's EOS ids and repetition penalty (applied by
``transformers.generate`` even under greedy decoding), and the same scoring code. Speed numbers
are MLX-to-MLX only and must not be compared with PyTorch-MPS timings.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import time

from .reporting import write_json

PRECISIONS = {"fp32": None, "fp16": None, "8bit": 8, "4bit": 4}
GROUP_SIZE = 64  # mlx-lm default group size for affine quantization
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json",
                   "merges.txt", "added_tokens.json", "chat_template.jinja", "generation_config.json")
# Generation-config keys whose processors transformers would apply under greedy decoding but this
# MLX loop does not implement; their presence is refused rather than silently ignored.
UNSUPPORTED_GENERATION_KEYS = ("no_repeat_ngram_size", "bad_words_ids", "min_new_tokens", "min_length",
                               "suppress_tokens", "begin_suppress_tokens", "forced_bos_token_id",
                               "forced_eos_token_id", "encoder_repetition_penalty", "sequence_bias",
                               "exponential_decay_length_penalty")


def _mlx():
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx_lm  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised by tests that hide mlx
        raise RuntimeError("prune-lab quantize needs the optional mlx-lm package (pip install mlx-lm)") from exc
    return mx, nn


def mlx_versions():
    import importlib.metadata as metadata
    _mlx()
    return {"mlx": metadata.version("mlx"), "mlx-lm": metadata.version("mlx-lm")}


def _read_config(directory):
    config = json.loads((Path(directory) / "config.json").read_text())
    if config.get("model_type") != "qwen2":
        raise ValueError("Only Qwen2 exports are supported")
    return config


def build_model(config):
    """mlx-lm Qwen2 model with per-layer MLP widths from ``prune_lab_layer_widths`` when present."""
    _mlx()
    from mlx_lm.models import qwen2
    args = qwen2.ModelArgs.from_dict(config)
    model = qwen2.Model(args)
    widths = config.get("prune_lab_layer_widths")
    if widths is not None:
        if len(widths) != args.num_hidden_layers or any(type(w) is not int or w < 1 for w in widths):
            raise ValueError("Invalid prune_lab_layer_widths")
        for layer, width in zip(model.layers, widths):
            if width != args.intermediate_size:
                layer.mlp = qwen2.MLP(args.hidden_size, width)
    return model


def _quantized_paths(model):
    _, nn = _mlx()
    return sorted(name for name, module in model.named_modules()
                  if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)))


def load_mlx(directory):
    """Load a converted model (any precision) written by :func:`convert`."""
    mx, nn = _mlx()
    directory = Path(directory)
    config = _read_config(directory)
    model = build_model(config)
    weights = {}
    for file in sorted(directory.glob("model*.safetensors")):
        weights.update(mx.load(str(file)))
    weights = model.sanitize(weights)
    quantization = config.get("quantization")
    if quantization is not None:
        nn.quantize(model, group_size=quantization["group_size"], bits=quantization["bits"],
                    class_predicate=lambda path, module: hasattr(module, "to_quantized") and f"{path}.scales" in weights)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()
    return model, config


def convert(source, destination, precision):
    """Convert a float32 safetensors export to MLX at ``precision`` and record what was done."""
    mx, nn = _mlx()
    from mlx.utils import tree_flatten, tree_map
    from mlx_lm.utils import compute_bits_per_weight, quantize_model, save_config, save_model
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {tuple(PRECISIONS)}")
    source, destination = Path(source), Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Conversion output exists and is not empty: {destination}")
    config = _read_config(source)
    model = build_model(config)
    weights = {}
    files = sorted(source.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors export in {source}")
    for file in files:
        weights.update(mx.load(str(file)))
    model.load_weights(list(model.sanitize(weights).items()), strict=True)
    dtype = mx.float32 if precision == "fp32" else mx.float16
    model.update(tree_map(lambda x: x.astype(dtype) if mx.issubdtype(x.dtype, mx.floating) else x, model.parameters()))
    config = dict(config, torch_dtype="float32" if precision == "fp32" else "float16")
    config.pop("quantization", None)
    config.pop("quantization_config", None)
    bits = PRECISIONS[precision]
    if bits is not None:
        model, config = quantize_model(model, config, GROUP_SIZE, bits)
    mx.eval(model.parameters())
    quantized = _quantized_paths(model)
    unquantized_linear = sorted(name for name, module in model.named_modules()
                                if isinstance(module, (nn.Linear, nn.Embedding))
                                and not isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)))
    bits_per_weight = compute_bits_per_weight(model)
    destination.mkdir(parents=True, exist_ok=True)
    save_model(destination, model)
    save_config(dict(config), destination / "config.json")
    for name in TOKENIZER_FILES:
        if (source / name).exists():
            shutil.copyfile(source / name, destination / name)
    record = {"precision": precision, "bits": bits, "group_size": GROUP_SIZE if bits else None,
              "mode": "affine" if bits else None, "dtype": str(dtype),
              "quantized_modules": quantized, "unquantized_weight_modules": unquantized_linear,
              "effective_bits_per_weight": bits_per_weight,
              "weight_bytes": sum(p.stat().st_size for p in destination.glob("*.safetensors")),
              "source_weight_bytes": sum(p.stat().st_size for p in files),
              "layer_widths": config.get("prune_lab_layer_widths"), "versions": mlx_versions()}
    write_json(destination / "conversion.json", record)
    return record


def generation_settings(directory, max_new_tokens):
    """EOS ids and repetition penalty that transformers' greedy ``generate`` would use."""
    path = Path(directory) / "generation_config.json"
    config = json.loads(path.read_text()) if path.exists() else {}
    for key in UNSUPPORTED_GENERATION_KEYS:
        value = config.get(key)
        neutral = (None, [], 0) + ((1.0,) if key.endswith("penalty") else ())
        if value not in neutral:
            raise ValueError(f"generation_config.{key}={value!r} is not supported by the MLX evaluation loop")
    eos = config.get("eos_token_id")
    eos = [] if eos is None else eos if isinstance(eos, list) else [eos]
    penalty = float(config.get("repetition_penalty", 1.0))
    if not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("Invalid repetition_penalty")
    return {"do_sample": False, "max_new_tokens": max_new_tokens, "eos_token_id": eos,
            "repetition_penalty": penalty, "batch_size": 1}


def greedy_generate(model, prompt, settings):
    """Greedy decoding with a KV cache, HF-style repetition penalty and stop at the first EOS (kept)."""
    mx, _ = _mlx()
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    penalty, eos = settings["repetition_penalty"], set(settings["eos_token_id"])
    seen = list(dict.fromkeys(prompt))
    logits = model(mx.array([prompt]), cache=cache)[:, -1, :]
    generated = []
    for _ in range(settings["max_new_tokens"]):
        scores = logits.astype(mx.float32)[0]
        if penalty != 1.0 and seen:
            index = mx.array(seen)
            picked = scores[index]
            scores[index] = mx.where(picked < 0, picked * penalty, picked / penalty)
        token = int(mx.argmax(scores).item())
        generated.append(token)
        if token in eos:
            break
        if token not in seen:
            seen.append(token)
        logits = model(mx.array([[token]]), cache=cache)[:, -1, :]
    return generated


def answer_loss(model, tokenizer, example):
    """Token-weighted answer CE exactly as the PyTorch batches define it (answer + EOS supervised)."""
    mx, _ = _mlx()
    from .data import prompt_ids
    prompt = prompt_ids(tokenizer, example.prompt)
    answer = tokenizer.encode(example.answer, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        answer = answer + [tokenizer.eos_token_id]
    ids = prompt + answer
    logits = model(mx.array([ids[:-1]])).astype(mx.float32)[0, len(prompt) - 1:]
    targets = mx.array(answer)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    loss = -mx.take_along_axis(logprobs, targets[:, None], axis=-1).sum()
    return float(loss.item()), len(answer)


def evaluate_mlx(model, tokenizer, examples, settings):
    """Per-example records in the same schema as ``support_evaluation.evaluate_support``."""
    mx, _ = _mlx()
    from .data import prompt_ids
    from .support_evaluation import normalize_reply, reply_metrics
    examples = list(examples)
    if not examples:
        raise ValueError("No support evaluation examples")
    reset = getattr(mx, "reset_peak_memory", None)
    if reset:
        reset()
    loss_sum, supervised = 0.0, 0
    for ex in examples:
        if not normalize_reply(ex.answer):
            raise ValueError(f"Empty normalized support reference: {ex.key}")
        loss, n = answer_loss(model, tokenizer, ex)
        if not math.isfinite(loss) or loss < 0:
            raise ValueError("Nonfinite or negative support answer-token loss")
        loss_sum += loss
        supervised += n
    greedy_generate(model, prompt_ids(tokenizer, examples[0].prompt), settings)  # unmeasured warmup
    rows, elapsed, tokens = [], 0.0, 0
    for ex in examples:
        start = time.perf_counter()
        generated = greedy_generate(model, prompt_ids(tokenizer, ex.prompt), settings)
        seconds = time.perf_counter() - start
        prediction = tokenizer.decode(generated, skip_special_tokens=True)
        metrics = reply_metrics(prediction, ex.answer)
        row = {"key": ex.key, "prediction": prediction, "expected": ex.answer, "score": metrics["token_f1"],
               **metrics, "seconds": seconds, "generated_tokens": len(generated)}
        for field in ("doc_id", "dialogue_id", "domain", "group_id"):
            value = getattr(ex, field, None)
            if value is not None:
                row[field] = value
        rows.append(row)
        elapsed += seconds
        tokens += len(generated)
    aggregate = {name: sum(row[name] for row in rows) / len(rows) for name in ("token_f1", "rouge_l", "exact_match")}
    peak = getattr(mx, "get_peak_memory", lambda: None)()
    return {"selection_score": aggregate["token_f1"], "score_name": "token_f1", **aggregate,
            "count": len(rows), "answer_token_loss": loss_sum / supervised, "supervised_tokens": supervised,
            "generation_seconds": elapsed, "generated_tokens": tokens,
            "tokens_per_second": tokens / elapsed if elapsed else 0.0,
            "generation_limit_fraction": sum(r["generated_tokens"] >= settings["max_new_tokens"] for r in rows) / len(rows),
            "generation_limit_note": "Fraction reaching the token cap; not confirmed truncation because EOS may occur on the final token.",
            "mean_example_seconds": elapsed / len(rows), "warmup_examples": 1, "small_sample_warning": len(rows) < 100,
            "memory": {"mlx_peak_bytes": peak}, "generation_config": dict(settings, backend="mlx"),
            "metric_note": "Lexical overlap proxies; not factual grounding, policy correctness, or human quality.",
            "normalization": "Casefold; Unicode punctuation to spaces; remove English a/an/the; collapse whitespace.",
            "predictions": rows}


def speed(model, tokenizer, examples, settings, repeats=3):
    """Tokens/s over a fixed slice after one warm-up pass; MLX-to-MLX comparisons only."""
    mx, _ = _mlx()
    from .data import prompt_ids
    prompts = [prompt_ids(tokenizer, ex.prompt) for ex in examples]
    for prompt in prompts[:1]:
        greedy_generate(model, prompt, settings)
    runs = []
    for _ in range(repeats):
        start, tokens = time.perf_counter(), 0
        for prompt in prompts:
            tokens += len(greedy_generate(model, prompt, settings))
        seconds = time.perf_counter() - start
        runs.append({"seconds": seconds, "generated_tokens": tokens, "tokens_per_second": tokens / seconds})
    peak = getattr(mx, "get_peak_memory", lambda: None)()
    return {"examples": len(prompts), "repeats": repeats, "runs": runs,
            "median_tokens_per_second": sorted(r["tokens_per_second"] for r in runs)[len(runs) // 2],
            "mlx_peak_bytes": peak,
            "note": "MLX greedy loop, batch one; not comparable with PyTorch-MPS timings."}


def logit_check(source, model, tokenizer, examples, device="cpu"):
    """Compare MLX logits with the PyTorch export on a few prompts (all prompt positions)."""
    mx, _ = _mlx()
    import torch
    from .data import prompt_ids
    from .modeling import load_model
    reference, _ = load_model(str(source), device=device, dtype="float32")
    results = []
    try:
        for ex in examples:
            ids = prompt_ids(tokenizer, ex.prompt)
            with torch.inference_mode():
                expected = reference(input_ids=torch.tensor([ids], device=device), use_cache=False).logits[0].float().cpu()
            actual = torch.tensor(model(mx.array([ids])).astype(mx.float32)[0].tolist())
            results.append({"key": ex.key, "positions": len(ids),
                            "max_abs_difference": float((actual - expected).abs().max()),
                            "next_token_argmax_equal": int(actual[-1].argmax()) == int(expected[-1].argmax()),
                            "position_argmax_agreement": float((actual.argmax(-1) == expected.argmax(-1)).float().mean())})
    finally:
        del reference
    return {"prompts": results, "passed": all(r["next_token_argmax_equal"] for r in results),
            "max_abs_difference": max(r["max_abs_difference"] for r in results),
            "rule": "Passes when the next-token argmax equals the PyTorch export on every prompt."}


def _seed_directories(source, seeds):
    directories = sorted((p for p in Path(source).glob("seed-*") if p.is_dir() and p.name[5:].isdigit()),
                         key=lambda p: int(p.name[5:]))
    if seeds:
        wanted = {f"seed-{s}" for s in seeds}
        missing = wanted - {p.name for p in directories}
        if missing:
            raise ValueError(f"Seeds not found in {source}: {sorted(missing)}")
        directories = [p for p in directories if p.name in wanted]
    if not directories:
        raise ValueError(f"No seed-* campaign directories under {source}")
    return directories


def quantize_run(source, output, variants, precisions, seeds=None, check_prompts=4, speed_examples=32,
                 speed_repeats=3, discard_models=False, check_device="cpu"):
    """Convert and evaluate exports of ``variants`` for every seed of a multi-seed run.

    Writes ``<output>/seed-<s>/<variant>-<precision>/`` (MLX model + result.json) and a campaign-shaped
    seed directory (config, state, selection-dev, selection + lock) whose candidates are the MLX
    variants plus the PyTorch source records as ``<variant>-pytorch``, so ``multi-seed aggregate`` and
    ``compare`` work unchanged. Only dev is evaluated; the test split is never read.
    """
    from .campaign import _fingerprint, _selection, checkpoint_identity, read_campaign_config
    from .data import validate_splits
    from .modeling import load_model  # noqa: F401  (fails early if the PyTorch stack is broken)
    from .support_data import read_support_jsonl
    from transformers import AutoTokenizer
    _mlx()
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("Provide distinct variants")
    if not precisions or len(set(precisions)) != len(precisions) or any(p not in PRECISIONS for p in precisions):
        raise ValueError(f"precisions must be distinct values from {tuple(PRECISIONS)}")
    output = Path(output)
    summary, failures = [], []
    for seed_dir in _seed_directories(source, seeds):
        selection = _selection(seed_dir)
        config = read_campaign_config(seed_dir / "config.json")
        if config.get("task") != "support":
            raise ValueError("Quantization evaluation supports support campaigns only")
        dev = read_support_jsonl(seed_dir / "data" / "dev.jsonl")
        calibration = read_support_jsonl(seed_dir / "data" / "calibration.jsonl")
        validate_splits({"dev": dev, "calibration": calibration})
        records = {c["id"]: c for c in selection["candidates"] if c.get("status") == "complete"}
        target = output / seed_dir.name
        target.mkdir(parents=True, exist_ok=True)
        candidates = []
        for variant in variants:
            if variant not in records:
                raise ValueError(f"{seed_dir.name}: variant {variant} has no complete export record")
            record = records[variant]
            export = Path(record["checkpoint"])
            identity = checkpoint_identity(export)
            if identity != record["checkpoint_sha256"]:
                raise ValueError(f"{seed_dir.name}/{variant}: export hash differs from its record")
            tokenizer = AutoTokenizer.from_pretrained(str(export), trust_remote_code=False, fix_mistral_regex=False)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            settings = generation_settings(export, config["max_new_tokens"])
            pytorch = {k: v for k, v in record.items() if k not in ("checkpoint",)}
            candidates.append(dict(pytorch, id=f"{variant}-pytorch", checkpoint=str(export), backend="pytorch"))
            for precision in precisions:
                name = f"{variant}-{precision}"
                directory = target / name
                result_path = directory / "result.json"
                if result_path.exists():
                    done = json.loads(result_path.read_text())
                    if precision in ("fp32", "fp16") and done.get("logit_check") and not done["logit_check"]["passed"]:
                        failures.append(f"{seed_dir.name}/{name}")
                    candidates.append(done)
                    continue
                print(f"Quantize {seed_dir.name}/{name}", flush=True)
                start = time.perf_counter()
                model_dir = directory / "model"
                if model_dir.exists():
                    shutil.rmtree(model_dir)  # An interrupted conversion is redone from scratch.
                conversion = convert(export, model_dir, precision)
                model, _ = load_mlx(model_dir)
                check = logit_check(export, model, tokenizer, calibration[:check_prompts], check_device) if check_prompts else None
                if precision in ("fp32", "fp16") and check is not None and not check["passed"]:
                    # Recorded and reported at the end rather than aborting a long multi-seed job.
                    failures.append(f"{seed_dir.name}/{name}")
                    print(f"WARNING: {name}: MLX {precision} next-token argmax differs from the PyTorch export", flush=True)
                metrics = evaluate_mlx(model, tokenizer, dev, settings)
                timing = speed(model, tokenizer, dev[:speed_examples], settings, speed_repeats)
                del model
                result = {"id": name, "status": "complete", "variant": variant, "precision": precision,
                          "backend": "mlx", "source_checkpoint": str(export), "source_checkpoint_sha256": identity,
                          "parameters": record["parameters"], "actual_reduction": record["actual_reduction"],
                          "target": record.get("target"), "control": record.get("control", False),
                          "conversion": conversion, "logit_check": check, "dev": metrics, "speed": timing,
                          "on_disk_weight_bytes": conversion["weight_bytes"],
                          "elapsed_seconds": time.perf_counter() - start, "versions": mlx_versions()}
                if discard_models:
                    for file in model_dir.glob("*.safetensors"):
                        file.unlink()
                    result["model_weights_deleted"] = True
                write_json(result_path, result)
                candidates.append(result)
        for name in ("config.json", "state.json", "selection-dev.json"):
            if (seed_dir / name).exists():
                shutil.copyfile(seed_dir / name, target / name)
        assembled = {"selected_on": "development_only", "test_used_for_selection": False,
                     "baseline": selection["baseline"], "candidates": candidates, "reference_id": "original",
                     "selection_score_name": "token_f1", "tiers": [],
                     "source_run": str(Path(source).resolve() / seed_dir.name),
                     "note": "Assembled by prune-lab quantize: MLX conversions of saved exports plus their PyTorch records."}
        write_json(target / "selection.json", assembled)
        write_json(target / "selection-lock.json", {"sha256": _fingerprint(assembled)})
        summary.append({"seed_dir": str(target), "candidates": [c["id"] for c in candidates]})
    write_json(output / "quantize-run.json", {"source": str(Path(source).resolve()), "variants": variants,
                                             "precisions": precisions, "versions": mlx_versions(), "seeds": summary,
                                             "logit_check_failures": failures})
    if failures:
        raise ValueError(f"MLX full-precision logit check failed for {failures}; all other results were written")
    return summary
