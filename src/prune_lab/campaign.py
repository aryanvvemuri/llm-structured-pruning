"""Bounded per-layer pruning/recovery campaign with development-only selection.

Stages persist independently. Once final-test evaluation starts, fitting is locked.
No model is loaded alongside another model; recovery keeps only small adapter snapshots.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import shutil
import statistics
import time

from .data import dataset_manifest, jsonl_examples, make_batches, validate_splits
from .evaluation import evaluate_model, memory_snapshot
from .reporting import write_json
from .metrics import selection_score, score_name


DEFAULT_CAMPAIGN = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct", "device": "mps", "dtype": "float32",
    "cache_dir": ".cache/huggingface/hub", "seed": 20261113, "cpu_threads": 4,
    "dataset": "synthetic", "counts": {"recovery": 1000, "calibration": 150, "dev": 200, "test": 300},
    "max_length": 256, "max_new_tokens": 32,
    "targets": [0.05, 0.10, 0.15, 0.20], "tiers_pp": [0.0, 5.0, 10.0],
    "group_size": 64, "min_keep_ratio": 0.25, "probe_fraction": 0.10,
    "sensitivity_examples": 16, "benchmark_examples": 64, "benchmark_repeats": 3,
    "include_controls": True, "exclude_legacy_seeds": [17],
    "exclude_legacy_counts": {"calibration": 32, "dev": 64, "test": 128},
    "exclusion_jsonl": [],
    "recovery": {"rank": 8, "alpha": 16, "learning_rate": 0.0002, "max_steps": 200,
                 "gradient_accumulation": 4, "checkpoints": [50, 100, 200],
                 "seed": 20261113, "gradient_checkpointing": False},
}


def read_campaign_config(path=None):
    config = json.loads(json.dumps(DEFAULT_CAMPAIGN))
    if path:
        supplied = json.loads(Path(path).read_text())
        for key in ("recovery", "counts"):
            if key in supplied:
                config[key].update(supplied.pop(key))
        config.update(supplied)
    return config


def _fingerprint(value):
    """SHA-256 of the JSON form of ``value``, stable across a write/read round trip.

    The value is normalised through JSON first, so integer dict keys hash as the strings they
    become on disk (sorted as strings). Values without integer keys hash exactly as before.
    """
    normalised = json.loads(json.dumps(value, allow_nan=False))
    return hashlib.sha256(json.dumps(normalised, sort_keys=True, allow_nan=False).encode()).hexdigest()


def legacy_fingerprint(value):
    """The pre-fix fingerprint (no JSON normalisation), kept only to verify old locks."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_config(config):
    if config.get("task", "arithmetic") not in {"arithmetic", "support"}:
        raise ValueError("Unknown campaign task")
    if (config.get("task") == "support") != (config.get("dataset") == "doc2dial"):
        raise ValueError("Support task requires the Doc2Dial dataset adapter")
    if config.get("selection_reference", "original") not in {"original", "unpruned-recovery"}:
        raise ValueError("Unknown selection reference")
    if config.get("selection_reference") == "unpruned-recovery" and not config["include_controls"]:
        raise ValueError("Recovery reference requires the unpruned recovery control")
    for key in ("max_length", "max_new_tokens", "cpu_threads", "group_size", "sensitivity_examples", "benchmark_examples", "benchmark_repeats"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not config["targets"] or len(set(config["targets"])) != len(config["targets"]):
        raise ValueError("targets must be nonempty and unique")
    for value in config["targets"]:
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError("targets are fractions of TOTAL parameters, strictly between 0 and 1")
    _validate_tier_mode(config.get("tiers_pp"), config.get("retention_targets"))
    for key in ("min_keep_ratio", "probe_fraction"):
        if not math.isfinite(config[key]) or not 0 < config[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    if config["probe_fraction"] == 1:
        raise ValueError("probe_fraction must be strictly less than one")
    subset = config.get("selection_dev_examples")
    if subset is not None and (isinstance(subset, bool) or not isinstance(subset, int) or subset < 1):
        raise ValueError("selection_dev_examples must be a positive integer or null")
    dev_count = (config.get("counts") or {}).get("dev")
    if subset is not None and isinstance(dev_count, int) and subset > dev_count:
        raise ValueError(f"selection_dev_examples ({subset}) exceeds counts.dev ({dev_count})")
    batch = config.get("eval_batch_size", 1)
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("eval_batch_size must be a positive integer")
    if batch > 1 and config.get("task") != "support":
        raise ValueError("eval_batch_size > 1 is implemented only for the support task")
    if not isinstance(config.get("skip_full_merged_dev", False), bool):
        raise ValueError("skip_full_merged_dev must be true or false")
    if not isinstance(config.get("store_example_losses", False), bool):
        raise ValueError("store_example_losses must be true or false")
    from .recipes import validate_recipe
    validate_recipe(config["recovery"])
    if not isinstance(config.get("uniform_per_target", False), bool):
        raise ValueError("uniform_per_target must be true or false")
    if config.get("uniform_per_target") and not config["include_controls"]:
        raise ValueError("uniform_per_target needs include_controls")
    chosen = config.get("train_candidates")
    if chosen is not None:
        if not isinstance(chosen, list) or not chosen or any(not isinstance(c, str) for c in chosen) or len(set(chosen)) != len(chosen):
            raise ValueError("train_candidates must be a nonempty list of distinct candidate names")
        reference = config.get("selection_reference", "original")
        if reference != "original" and reference not in chosen:
            raise ValueError(f"train_candidates must include the selection reference {reference}")
    specs = candidate_specs(config)  # Validates allocation options, names and train_candidates.
    iterative_stage_widths(config)  # Feasibility and strictly shrinking widths, from the model shape only.
    if config.get("allocation") is not None and allocation_options(config)["curve_sensitivity_examples"] is not None:
        calibration = (config.get("counts") or {}).get("calibration")
        if isinstance(calibration, int) and allocation_options(config)["curve_sensitivity_examples"] > calibration:
            raise ValueError("curve_sensitivity_examples exceeds counts.calibration")
    if not specs:
        raise ValueError("No candidates to train")


def selection_dev_subset(dev, count, seed):
    """Deterministic domain-stratified dev subset for in-training checkpoint selection.

    Depends only on the dev examples and the top-level seed (never the recovery seed), so
    every candidate, recovery seed and run directory with the same data selects the same
    rows. Domains get largest-remainder proportional shares; rows keep dev order.
    """
    if type(count) is not int or not 1 <= count <= len(dev):
        raise ValueError(f"selection_dev_examples must be between 1 and {len(dev)}")
    strata = {}
    for row in sorted(dev, key=lambda r: r.key):
        strata.setdefault(getattr(row, "domain", None) or "", []).append(row)
    exact = {d: count * len(rows) / len(dev) for d, rows in strata.items()}
    shares = {d: int(v) for d, v in exact.items()}
    for d in sorted(strata, key=lambda d: (-(exact[d] - shares[d]), d))[:count - sum(shares.values())]:
        shares[d] += 1
    chosen = set()
    for d in sorted(strata):
        rows = list(strata[d])
        random.Random(f"{seed}:selection-dev:{d}").shuffle(rows)
        chosen.update(r.key for r in rows[:shares[d]])
    return [row for row in dev if row.key in chosen]


def _selection_dev(config, output, dev):
    """Return the selection subset (or all dev) and persist/verify its ids in the run."""
    count = config.get("selection_dev_examples")
    if count is None:
        return dev
    rows = selection_dev_subset(dev, count, config["seed"])
    keys = [r.key for r in rows]
    record = {"count": len(keys), "seed": config["seed"], "keys_sha256": _fingerprint(keys), "keys": keys,
              "domains": dict(sorted(Counter(getattr(r, "domain", None) or "" for r in rows).items())),
              "note": "Used only for in-training checkpoint selection (and the merge identity check); "
                      "it is part of the full dev set on which every candidate is reported."}
    path = output / "selection-dev.json"
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError("Persisted selection dev subset differs; refusing to resume")
    write_json(path, record)
    return rows


def _initialize(config, output):
    from .data import load_campaign_splits
    validate_config(config)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "state.json"
    fingerprint = _fingerprint(config)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["config_sha256"] != fingerprint:
            raise ValueError("Run configuration changed; use a new output directory")
        reader = jsonl_examples
        if config.get("task") == "support":
            from .support_data import read_support_jsonl
            reader = read_support_jsonl
        splits = {name: reader(output / "data" / f"{name}.jsonl") for name in config["counts"]}
        validate_splits(splits)
        if dataset_manifest(splits) != state["dataset"]:
            raise ValueError("Persisted datasets changed; refusing to resume")
        return state, splits
    if any(output.iterdir()):
        raise ValueError("Output directory is nonempty without campaign state; choose a new directory")
    splits = load_campaign_splits(config)
    (output / "data").mkdir()
    for name, examples in splits.items():
        (output / "data" / f"{name}.jsonl").write_text("".join(json.dumps(asdict(x)) + "\n" for x in examples))
    state = {"config_sha256": fingerprint, "dataset": dataset_manifest(splits), "test_started": False}
    write_json(output / "config.json", config)
    write_json(state_path, state)
    if config.get("task") == "support":
        manifest = Path(config["support_data_dir"]) / "manifest.json"
        write_json(output / "data-provenance.json", json.loads(manifest.read_text()))
    return state, splits


def _load(config, checkpoint=None):
    import torch
    from .modeling import load_model
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    return load_model(checkpoint or config["model"], config["device"], config["dtype"], config.get("cache_dir"))


def _release():
    import torch
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _evaluate(model, tokenizer, examples, config):
    if config.get("task") == "support":
        from .support_evaluation import evaluate_support
        batch_size = config.get("eval_batch_size", 1)
        if config.get("store_example_losses", False):
            return evaluate_support(model, tokenizer, examples, config["max_length"], config["max_new_tokens"],
                                    batch_size, example_losses=True)
        if batch_size == 1:
            return evaluate_support(model, tokenizer, examples, config["max_length"], config["max_new_tokens"])
        return evaluate_support(model, tokenizer, examples, config["max_length"], config["max_new_tokens"], batch_size)
    return evaluate_model(model, tokenizer, examples, config["max_length"], config["max_new_tokens"])


def _provenance(model):
    import torch
    import transformers
    return {"model_revision": getattr(model.config, "_commit_hash", None),
            "device": str(next(model.parameters()).device),
            "versions": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__}}


def checkpoint_identity(path):
    """Stream file hashes without allocating another model or state dictionary."""
    directory = Path(path)
    if not (directory / "config.json").is_file() or not list(directory.glob("*.safetensors")):
        raise ValueError(f"Incomplete safetensors checkpoint: {directory}")
    files = {}
    for file in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        files[str(file.relative_to(directory))] = digest.hexdigest()
    return _fingerprint(files)


def _source_identity(config, model):
    if Path(config["model"]).is_dir():
        return {"local_checkpoint_sha256": checkpoint_identity(config["model"])}
    revision = getattr(model.config, "_commit_hash", None)
    if not revision:
        raise ValueError("Remote source has no resolved revision; use a local safetensors checkpoint")
    return {"model_id": config["model"], "revision": revision}


def _verify_checkpoint(record):
    if checkpoint_identity(record["checkpoint"]) != record["checkpoint_sha256"]:
        raise ValueError(f"Saved checkpoint changed: {record['id']}")


def _verify_source(config, model, baseline):
    if _source_identity(config, model) != baseline["source_identity"]:
        raise ValueError("Original model changed during campaign; use the recorded source snapshot")


def preflight(config, output, splits):
    """Two optimizer steps on training data; no checkpoint or test-set selection."""
    from .recovery import recover_model
    model, tokenizer = _load(config)
    try:
        print("Preflight: measuring two recovery steps on the selected device", flush=True)
        recovery = dict(config["recovery"], max_steps=2, checkpoints=[2])
        if "warmup_steps" in recovery:
            # Two-update timing probe: keep the configured schedule valid.
            recovery["warmup_steps"] = min(recovery["warmup_steps"], 1)
        n = 2 * recovery["gradient_accumulation"]
        batches = make_batches(splits["recovery"][:n], tokenizer, config["max_length"])
        device = next(model.parameters()).device
        before = memory_snapshot(device)
        start = time.perf_counter()
        result = recover_model(model, batches,
                               lambda m: _evaluate(m, tokenizer, splits["dev"][:8], config), recovery)
        record = {"passed": True, **_provenance(model), "elapsed_seconds": time.perf_counter() - start,
                  "memory_before": before, "memory_after": memory_snapshot(device), "recovery": result,
                  "memory_note": "Snapshots are not peaks. This short run does not establish quality."}
        write_json(output / "preflight.json", record)
    finally:
        del model, tokenizer
        _release()


ALLOCATION_METHODS = {"legacy": "per-layer", "curve": "curve", "global-standardized": "global-std"}
ALLOCATION_KEYS = {"method", "methods", "curve_sensitivity_examples", "probe_fractions", "min_keep_ratio"}


def allocation_options(config):
    """Opt-in improvement-1 allocators; None (absent) keeps the legacy allocator only."""
    from .allocation import CURVE_FRACTIONS, NEW_MIN_KEEP_RATIO
    options = config.get("allocation")
    if options is None:
        return None
    if not isinstance(options, dict) or set(options) - ALLOCATION_KEYS or ("method" in options) == ("methods" in options):
        raise ValueError(f"allocation must be an object with exactly one of method/methods and keys from {sorted(ALLOCATION_KEYS)}")
    methods = options["methods"] if "methods" in options else [options["method"]]
    if (not isinstance(methods, list) or not methods or len(set(map(str, methods))) != len(methods)
            or any(m not in ALLOCATION_METHODS for m in methods)):
        raise ValueError(f"allocation methods must be distinct values from {sorted(ALLOCATION_METHODS)}")
    examples = options.get("curve_sensitivity_examples")
    if examples is not None and (isinstance(examples, bool) or not isinstance(examples, int) or examples < 1):
        raise ValueError("curve_sensitivity_examples must be a positive integer or null (all calibration examples)")
    fractions = options.get("probe_fractions", list(CURVE_FRACTIONS))
    if (not isinstance(fractions, list) or not fractions
            or any(isinstance(f, bool) or not isinstance(f, (int, float)) or not 0 < f < 1 for f in fractions)
            or fractions != sorted(set(fractions))):
        raise ValueError("probe_fractions must be strictly increasing values in (0, 1)")
    keep = options.get("min_keep_ratio", NEW_MIN_KEEP_RATIO)
    if isinstance(keep, bool) or not isinstance(keep, (int, float)) or not math.isfinite(keep) or not 0 < keep <= 1:
        raise ValueError("allocation min_keep_ratio must be in (0, 1]")
    if "curve" in methods and 1 - keep > fractions[-1] + 1e-12:
        raise ValueError("The curve allocator's removal cap exceeds the largest probed fraction")
    return {"methods": methods, "curve_sensitivity_examples": examples, "probe_fractions": fractions,
            "min_keep_ratio": keep}


ITERATIVE_KEYS = {"id", "stages", "steps", "checkpoints"}


def iterative_options(config):
    """Opt-in iterative candidates: uniform prune -> recover -> re-score -> prune -> recover ..."""
    items = config.get("iterative")
    if items is None:
        return []
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list) or not items:
        raise ValueError("iterative must be an object or a nonempty list of objects")
    recovery = config.get("recovery", {})
    for item in items:
        if not isinstance(item, dict) or set(item) - ITERATIVE_KEYS or not {"id", "stages", "steps"} <= set(item):
            raise ValueError(f"Each iterative candidate needs id, stages and steps (optional checkpoints): {item}")
        stages, steps = item["stages"], item["steps"]
        if (not isinstance(item["id"], str) or not item["id"] or not isinstance(stages, list) or len(stages) < 2
                or any(isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 < t < 1 for t in stages)
                or stages != sorted(set(stages))):
            raise ValueError("iterative stages must be at least two strictly increasing fractions in (0, 1)")
        if not isinstance(steps, list) or len(steps) != len(stages) or any(type(n) is not int or n < 1 for n in steps):
            raise ValueError("iterative steps must give one positive integer update count per stage")
        if sum(steps) != recovery.get("max_steps"):
            raise ValueError(f"iterative steps {steps} must sum to recovery.max_steps ({recovery.get('max_steps')}): equal budget")
        if any(recovery.get("warmup_steps", 0) >= n for n in steps):
            raise ValueError("Every iterative stage needs more updates than recovery.warmup_steps")
        checkpoints = item.get("checkpoints")
        if checkpoints is not None and (not isinstance(checkpoints, list) or len(checkpoints) != len(stages) or any(
                not isinstance(c, list) or not c or any(type(v) is not int or not 1 <= v <= n for v in c)
                for c, n in zip(checkpoints, steps))):
            raise ValueError("iterative checkpoints must be one list of update numbers within each stage")
    return items


def iterative_stage_widths(config, model=None):
    """Uniform per-layer widths of every iterative stage; rejects infeasible or non-shrinking stages.

    Uses ``model`` when given, otherwise a weightless (meta-device) model built from the source
    config. Uniform allocation depends only on shapes, so placeholder scores are exact here.
    Returns None when the model config cannot be read offline (fit re-checks with the real model).
    """
    import torch
    from .allocation import allocate_neurons
    from .modeling import mlp_layers
    items = iterative_options(config)
    if not items:
        return {}
    if model is None:
        from transformers import AutoConfig
        from .modeling import CompactQwen2ForCausalLM
        try:
            source = AutoConfig.from_pretrained(config["model"], cache_dir=config.get("cache_dir"),
                                                local_files_only=True, trust_remote_code=False)
        except (OSError, ValueError):
            return None
        with torch.device("meta"):
            model = CompactQwen2ForCausalLM(source)
    widths = [m.down_proj.in_features for m in mlp_layers(model)]
    scores = {i: torch.arange(w, dtype=torch.float32) for i, w in enumerate(widths)}
    result = {}
    for item in items:
        stage_widths = []
        for target in item["stages"]:
            try:
                keep = allocate_neurons(model, scores, {i: 1.0 for i in scores}, target, config["group_size"],
                                        config["min_keep_ratio"], True)
            except ValueError as exc:
                raise ValueError(f"Iterative candidate {item['id']}: stage target {target} is infeasible: {exc}") from exc
            stage_widths.append(len(keep[0]))
        if any(a <= b for a, b in zip([widths[0]] + stage_widths, stage_widths)):
            raise ValueError(f"Iterative candidate {item['id']}: stages {item['stages']} give uniform widths {stage_widths}; "
                             "every stage must prune further (strictly smaller width)")
        result[item["id"]] = stage_widths
    return result


def stage_checkpoints(spec, recovery, stage):
    """Checkpoints of one stage: explicit, or the recipe's checkpoints scaled to the stage length."""
    steps = spec["steps"][stage]
    if spec.get("stage_checkpoints"):
        return sorted(set(spec["stage_checkpoints"][stage]) | {steps})
    total = recovery["max_steps"]
    return sorted({max(1, round(c * steps / total)) for c in recovery.get("checkpoints", [])} | {steps})


def _percent(target):
    return f"{round(100 * target):d}"


def candidate_specs(config):
    options = allocation_options(config)
    methods = options["methods"] if options else ["legacy"]
    result = []
    if "legacy" in methods:
        result += [{"id": f"per-layer-{i:02d}", "target": target, "uniform": False, "control": False}
                   for i, target in enumerate(config["targets"], 1)]
    for method in methods:
        if method != "legacy":
            result += [{"id": f"{ALLOCATION_METHODS[method]}-{_percent(target)}", "target": target, "uniform": False,
                        "control": False, "method": method} for target in config["targets"]]
    if config["include_controls"]:
        result += [{"id": "unpruned-recovery", "target": 0.0, "uniform": False, "control": True},
                   {"id": "uniform-10", "target": 0.10, "uniform": True, "control": True,
                    "selection_eligible": config.get("select_uniform", False)}]
        if config.get("uniform_per_target", False):
            result += [{"id": f"uniform-{_percent(target)}", "target": target, "uniform": True, "control": True,
                        "selection_eligible": config.get("select_uniform", False)}
                       for target in config["targets"] if f"uniform-{_percent(target)}" != "uniform-10"]
    for item in iterative_options(config):
        result.append({"id": item["id"], "target": item["stages"][-1], "uniform": True, "control": False,
                       "kind": "iterative", "stages": item["stages"], "steps": item["steps"],
                       **({"stage_checkpoints": item["checkpoints"]} if item.get("checkpoints") else {})})
    ids = [spec["id"] for spec in result]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Candidate names collide: {ids}")
    chosen = config.get("train_candidates")
    if chosen is not None:
        unknown = sorted(set(chosen) - set(ids))
        if unknown:
            raise ValueError(f"train_candidates not produced by this configuration: {unknown}; available {ids}")
        result = [spec for spec in result if spec["id"] in chosen]
    return result


def _valid_metrics(metrics):
    return (math.isfinite(selection_score(metrics)) and 0 <= selection_score(metrics) <= 1
            and math.isfinite(metrics["answer_token_loss"]) and metrics["answer_token_loss"] >= 0)


def _validate_tier_mode(tolerances, retention_targets):
    if tolerances is not None and retention_targets is not None:
        raise ValueError("Specify only one of tiers_pp or retention_targets; set the other to null")
    if retention_targets is not None:
        if (not isinstance(retention_targets, list) or not retention_targets
                or any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not math.isfinite(x) or not 0 < x <= 1 for x in retention_targets)):
            raise ValueError("retention_targets must be a nonempty list of finite ratios in (0, 1]")
    elif (not isinstance(tolerances, (list, tuple)) or not tolerances
          or any(isinstance(x, bool) or not isinstance(x, (int, float))
                 or not math.isfinite(x) or not 0 <= x <= 100 for x in tolerances)):
        raise ValueError("tiers_pp must contain finite percentage-point tolerances")


def select_tiers(baseline, candidates, tolerances=None, retention_targets=None):
    """Select the smallest eligible candidate using only reloaded development scores."""
    if not _valid_metrics(baseline["dev"]):
        raise ValueError("Invalid baseline development metrics")
    _validate_tier_mode(tolerances, retention_targets)
    baseline_score = selection_score(baseline["dev"])
    if retention_targets is not None and baseline_score == 0:
        raise ValueError("Relative retention requires a nonzero baseline development score")
    eligible = [c for c in candidates if c.get("status") == "complete"
                and (not c.get("control") or c.get("selection_eligible", False))
                and 0 < c["parameters"] < baseline["parameters"] and _valid_metrics(c["dev"])
                and score_name(c["dev"]) == score_name(baseline["dev"])]
    tiers = []
    for tolerance in retention_targets if retention_targets is not None else tolerances:
        relative = retention_targets is not None
        floor = baseline_score * tolerance if relative else max(0.0, baseline_score - tolerance / 100)
        matches = [c for c in eligible if selection_score(c["dev"]) + 1e-12 >= floor]
        selected = min(matches, key=lambda c: (c["parameters"], -selection_score(c["dev"]), c["dev"]["answer_token_loss"], c["id"])) if matches else None
        policy = ({"mode": "relative_retention", "min_retained_score": tolerance,
                   "baseline_dev_score": baseline_score} if relative else {"max_drop_pp": tolerance})
        tiers.append({**policy, "development_floor": floor,
                      "candidate_id": selected["id"] if selected else None,
                      "checkpoint": selected["checkpoint"] if selected else None,
                      "status": "selected" if selected else "no_qualifying_pruned_model"})
    return tiers


def fit(config, output, state, splits):
    from .allocation import allocate_neurons, measure_sensitivity
    from . import allocation as allocators
    from .artifacts import save_checkpoint
    from .modeling import count_parameters, mlp_layers
    from .pruning import compact_model
    from .recovery import recover_model
    from .scoring import score_neurons
    if state["test_started"]:
        raise ValueError("Final test has been opened: fitting/reselection is locked")
    if not (output / "preflight.json").exists():
        raise ValueError("Run the preflight stage before fitting")
    if not (output / "allocation.json").exists():
        model, tokenizer = _load(config)
        try:
            iterative_stage_widths(config, model)  # Before the baseline and Taylor scoring.
            print("Measuring baseline development score and calibration importance", flush=True)
            baseline = {"dev": _evaluate(model, tokenizer, splits["dev"], config),
                        "parameters": count_parameters(model), "source_identity": _source_identity(config, model),
                        **_provenance(model)}
            if not _valid_metrics(baseline["dev"]):
                raise ValueError("Invalid baseline metrics")
            calibration = make_batches(splits["calibration"], tokenizer, config["max_length"])
            start = time.perf_counter()
            scores = score_neurons(model, calibration, "taylor")
            sensitivity = measure_sensitivity(model, calibration[:config["sensitivity_examples"]],
                                              probe_fraction=config["probe_fraction"], scores=scores)
            options = allocation_options(config)
            specs = candidate_specs(config)
            extra = {}
            if any(spec.get("method") == "curve" for spec in specs):
                curve_start = time.perf_counter()
                count = options["curve_sensitivity_examples"] or len(calibration)
                probes = allocators.measure_sensitivity_curves(model, calibration[:count], scores, options["probe_fractions"])
                extra["curve_probes"] = {"examples": len(calibration[:count]), "fractions": options["probe_fractions"],
                                         "baseline_loss": probes["baseline"], "raw": probes["raw"],
                                         "curves": {i: allocators.process_curve(v) for i, v in probes["raw"].items()},
                                         "seconds": time.perf_counter() - curve_start}
            allocations = []
            for spec in specs:
                if spec.get("kind") == "iterative":
                    stage_keeps = [allocate_neurons(model, scores, sensitivity, target, config["group_size"],
                                                    config["min_keep_ratio"], True) for target in spec["stages"]]
                    allocations.append(dict(spec, keep_indices=stage_keeps[0], oneshot_keep_indices=stage_keeps[-1],
                                            stage_widths=[len(keep[0]) for keep in stage_keeps],
                                            stage1_scores_sha256=_scores_sha256(scores),
                                            original_widths=[m.down_proj.in_features for m in mlp_layers(model)]))
                    continue
                if spec["target"] == 0:
                    keep = {i: list(range(m.down_proj.in_features)) for i, m in enumerate(mlp_layers(model))}
                elif spec.get("method") == "curve":
                    keep = allocators.allocate_curve(model, scores, extra["curve_probes"]["raw"], spec["target"],
                                                     config["group_size"], options["min_keep_ratio"], options["probe_fractions"])
                elif spec.get("method") == "global-standardized":
                    keep = allocators.allocate_global_standardized(model, scores, spec["target"], config["group_size"],
                                                                   options["min_keep_ratio"])
                else:
                    keep = allocate_neurons(model, scores, sensitivity, spec["target"],
                                            config["group_size"], config["min_keep_ratio"], spec["uniform"])
                allocations.append(dict(spec, keep_indices=keep))
            if options is not None:
                extra["allocation_options"] = options
                # Per-neuron Taylor scores, so every mask can be recomputed from the record alone.
                extra["taylor_scores"] = {i: scores[i].detach().cpu().tolist() for i in sorted(scores)}
                extra["removal_fractions"] = {a["id"]: allocators.removal_fractions(scores, a["keep_indices"]) for a in allocations}
                # Calibration loss right after pruning (masked, before any recovery), per candidate.
                extra["calibration_loss_after_pruning"] = {
                    a["id"]: allocators.masked_calibration_loss(model, calibration, {int(k): v for k, v in a["keep_indices"].items()})
                    for a in allocations}
            write_json(output / "baseline.json", baseline)
            write_json(output / "allocation.json", {"candidates": allocations, "sensitivity": sensitivity,
                       "elapsed_seconds": time.perf_counter() - start, "baseline_sha256": _fingerprint(baseline),
                       **extra})
        finally:
            del model, tokenizer
            _release()
    baseline = json.loads((output / "baseline.json").read_text())
    allocation = json.loads((output / "allocation.json").read_text())
    if allocation["baseline_sha256"] != _fingerprint(baseline):
        raise ValueError("Baseline changed after allocation")
    selection_dev = _selection_dev(config, output, splits.get("dev"))
    skip_merged = config.get("skip_full_merged_dev", False)
    candidates = []
    for spec in allocation["candidates"]:
        directory = output / "candidates" / spec["id"]
        record_path = directory / "result.json"
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record.get("status") == "complete":
                _verify_checkpoint(record)
                candidates.append(record)
                continue
        print(f"Recovering {spec['id']}: target {spec['target']:.0%} total parameters", flush=True)
        start = time.perf_counter()
        model = tokenizer = None
        try:
            final_keep = spec["keep_indices"]
            if spec.get("kind") == "iterative":
                model, tokenizer, recovered, final_keep = _iterative_recovery(config, splits, spec, baseline, selection_dev, directory)
            else:
                model, tokenizer = _load(config)
                _verify_source(config, model, baseline)
                if spec["target"]:
                    compact_model(model, {int(k): v for k, v in spec["keep_indices"].items()})
                batches = make_batches(splits["recovery"], tokenizer, config["max_length"])
                recovered = recover_model(model, batches, lambda m: _evaluate(m, tokenizer, selection_dev, config), config["recovery"])
            parameters = count_parameters(model)
            # Selection uses the actual merged weights, not an unmerged adapter's score.
            merged_dev = _evaluate(model, tokenizer, selection_dev if skip_merged else splits["dev"], config)
            if not _valid_metrics(merged_dev):
                raise ValueError("Invalid post-merge development metrics")
            checkpoint = directory / "checkpoint"
            record = {**{k: v for k, v in spec.items() if k != "keep_indices"}, "status": "complete",
                      "checkpoint": str(checkpoint), "parameters": parameters,
                      "actual_reduction": 1 - parameters / baseline["parameters"],
                      "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                      "layer_widths": [m.down_proj.in_features for m in mlp_layers(model)],
                      "dev": None if skip_merged else merged_dev, "recovery": recovered, "elapsed_seconds": time.perf_counter() - start,
                      "memory_after": memory_snapshot(next(model.parameters()).device), **_provenance(model)}
            manifest = {"config": config, "dataset": state["dataset"], "selection_split": "dev",
                        "keep_indices": final_keep, "candidate": record}
            save_checkpoint(model, tokenizer, checkpoint, manifest)
            # Reopen the actual export before selection, with only one model resident.
            del model, tokenizer
            model = tokenizer = None
            _release()
            model, tokenizer = _load(config, str(checkpoint))
            if count_parameters(model) != parameters:
                raise ValueError("Reloaded parameter count differs from exported model")
            reloaded_dev = _evaluate(model, tokenizer, splits["dev"], config)
            if not _valid_metrics(reloaded_dev):
                raise ValueError("Invalid reloaded development metrics")
            if skip_merged:
                reloaded = {r["key"]: r["prediction"] for r in reloaded_dev["predictions"]}
                same = sum(r["prediction"] == reloaded[r["key"]] for r in merged_dev["predictions"])
                record["merged_selection_dev_before_export"] = merged_dev
                record["merge_check"] = {"scope": "selection_dev_subset", "examples": len(merged_dev["predictions"]),
                                         "identical_predictions_after_reload": same,
                                         "note": "Full after-merge dev evaluation skipped; merged-model predictions on the selection "
                                                 "subset are compared with the full after-reload evaluation."}
            else:
                record["merged_dev_before_export"] = merged_dev
            if len(selection_dev) != len(splits["dev"]):
                record["recovery_selection_dev"] = {"examples": len(selection_dev),
                                                    "keys_sha256": _fingerprint([r.key for r in selection_dev])}
            record["dev"] = reloaded_dev
            record["export_reload_verified"] = True
            record["checkpoint_sha256"] = checkpoint_identity(checkpoint)
            record["elapsed_seconds"] = time.perf_counter() - start
            record["checkpoint_bytes"] = sum(p.stat().st_size for p in checkpoint.rglob("*") if p.is_file())
            if spec.get("kind") == "iterative":
                record["iterative"] = _iterative_summary(spec, final_keep, recovered)
            write_json(record_path, record)
            if spec.get("kind") == "iterative":
                # Intermediate stage exports only serve resume; the final export is verified above.
                for weights in directory.glob("stage-*/checkpoint/*.safetensors"):
                    weights.unlink()
            candidates.append(record)
        except Exception as exc:
            failed = {**{k: v for k, v in spec.items() if k != "keep_indices"}, "status": "failed",
                      "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.perf_counter() - start}
            write_json(record_path, failed)
            candidates.append(failed)
            print(f"Candidate failed: {failed['error']}", flush=True)
        finally:
            del model, tokenizer
            _release()
    reference_id = config.get("selection_reference", "original")
    reference = baseline
    if reference_id != "original":
        reference = next((c for c in candidates if c["id"] == reference_id and c["status"] == "complete"), None)
        if reference is None:
            raise ValueError("The recovery reference failed; refusing to select against an easier baseline")
    selection = {"selected_on": "development_only", "test_used_for_selection": False,
                 "baseline": baseline, "candidates": candidates,
                 "reference_id": reference_id, "selection_score_name": score_name(reference["dev"]),
                 "evaluate_all_candidates": config.get("evaluate_all_candidates", False),
                 "tiers": select_tiers(reference, candidates, config.get("tiers_pp"),
                                       retention_targets=config.get("retention_targets"))}
    write_json(output / "selection.json", selection)
    write_json(output / "selection-lock.json", {"sha256": _fingerprint(selection)})
    return selection


def _scores_sha256(scores):
    digest = hashlib.sha256()
    for i in sorted(scores):
        digest.update(scores[i].detach().cpu().float().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _stage_seed(seed, stage):
    """Stage-specific recovery seed derived from the candidate's recovery seed (stage counted from 1)."""
    return int(hashlib.sha256(f"{seed}:iterative-stage-{stage}".encode()).hexdigest()[:8], 16)


def _iterative_recovery(config, splits, spec, baseline, selection_dev, directory):
    """Run every stage of an iterative candidate; returns the final merged model before export.

    The data order is the recovery seed's one-epoch permutation (the order a one-shot candidate
    with the same seed uses), split into consecutive per-stage slices so each example is seen once.
    Stage k > 1 re-scores (Taylor, calibration) the merged stage k-1 model, keeps the highest-scoring
    neurons per layer down to the stage's uniform width, and trains fresh adapters. Intermediate
    stages are exported under ``stage-<k>/`` so a restart resumes after the last finished stage.
    """
    from .artifacts import save_checkpoint
    from .modeling import mlp_layers
    from .pruning import compact_model
    from .recovery import recover_model
    from .scoring import score_neurons
    recovery = config["recovery"]
    seed, accumulation = recovery["seed"], recovery["gradient_accumulation"]
    widths = spec["stage_widths"]
    model, tokenizer = _load(config)
    _verify_source(config, model, baseline)
    batches = make_batches(splits["recovery"], tokenizer, config["max_length"])
    order = list(range(len(batches)))
    random.Random(seed).shuffle(order)
    needed = sum(spec["steps"]) * accumulation
    if needed > len(order):
        raise ValueError(f"Iterative stages need {needed} microbatches but recovery has {len(order)}")
    keep = {int(k): v for k, v in spec["keep_indices"].items()}
    stages, start_stage = [], 0
    # Resume after the last finished intermediate stage whose export is intact.
    for stage in range(len(spec["stages"]) - 1):
        path = directory / f"stage-{stage + 1}" / "stage.json"
        if not path.exists():
            break
        done = json.loads(path.read_text())
        try:
            identity_ok = checkpoint_identity(done["checkpoint"]) == done["checkpoint_sha256"]
        except ValueError:
            identity_ok = False
        if not identity_ok:
            break
        stages.append(done)
        start_stage = stage + 1
    if start_stage:
        del model
        _release()
        model, tokenizer = _load(config, stages[-1]["checkpoint"])
        keep = {int(k): v for k, v in stages[-1]["keep_indices"].items()}
    offset = sum(spec["steps"][:start_stage]) * accumulation
    for stage in range(start_stage, len(spec["stages"])):
        current = [m.down_proj.in_features for m in mlp_layers(model)]
        scores_hash = spec["stage1_scores_sha256"]
        if stage == 0:
            compact_model(model, keep)
        else:
            calibration = make_batches(splits["calibration"], tokenizer, config["max_length"])
            scores = score_neurons(model, calibration, "taylor")  # on the merged previous-stage model
            scores_hash = _scores_sha256(scores)
            local = {i: sorted(scores[i].argsort(stable=True)[current[i] - widths[stage]:].tolist()) for i in scores}
            keep = {i: [keep[i][j] for j in local[i]] for i in local}
            # A second compaction on purpose: indices are local to the stage-(k-1) widths here, and the
            # config's provenance is rewritten to original indices right after.
            model._prune_lab_compacted = False
            model.config.prune_lab_compacted = False
            compact_model(model, local)
            model.config.prune_lab_keep_indices = [keep[i] for i in range(len(keep))]
            model.config.prune_lab_original_widths = spec["original_widths"]
        steps = spec["steps"][stage]
        stage_order = order[offset:offset + steps * accumulation]
        offset += steps * accumulation
        recipe = dict(recovery, max_steps=steps, checkpoints=stage_checkpoints(spec, recovery, stage),
                      seed=_stage_seed(seed, stage + 1), shuffle_batches=False)
        print(f"Iterative {spec['id']} stage {stage + 1}/{len(spec['stages'])}: target {spec['stages'][stage]:.0%}, "
              f"{steps} updates", flush=True)
        result = recover_model(model, [batches[i] for i in stage_order], lambda m: _evaluate(m, tokenizer, selection_dev, config), recipe)
        summary = {"stage": stage + 1, "target": spec["stages"][stage], "width": widths[stage], "updates": steps,
                   "checkpoints": recipe["checkpoints"], "seed": recipe["seed"], "data_slice": [offset - steps * accumulation, offset],
                   "data_keys_sha256": _fingerprint([splits["recovery"][i].key for i in stage_order]),
                   "keep_indices": keep, "scores_sha256": scores_hash, "best_step": result["best_step"],
                   "pre_recovery_selection_dev": result["initial_dev"], "recovery": result}
        if stage < len(spec["stages"]) - 1:
            stage_dir = directory / f"stage-{stage + 1}"
            if stage_dir.exists():
                shutil.rmtree(stage_dir)
            checkpoint = stage_dir / "checkpoint"
            save_checkpoint(model, tokenizer, checkpoint, {"config": config, "keep_indices": keep, "stage": summary})
            summary.update(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_identity(checkpoint))
            write_json(stage_dir / "stage.json", summary)
        stages.append(summary)
    return model, tokenizer, dict(stages[-1]["recovery"], stages=stages), keep


def _iterative_summary(spec, final_keep, recovered):
    oneshot = {int(k): set(v) for k, v in spec["oneshot_keep_indices"].items()}
    final = {int(k): set(v) for k, v in final_keep.items()}
    differ = {i: len(final[i] - oneshot[i]) for i in sorted(final)}
    return {"stages": [{k: v for k, v in s.items() if k not in ("recovery", "keep_indices")} for s in recovered["stages"]],
            "stage_keep_indices": [s["keep_indices"] for s in recovered["stages"]],
            "final_widths": [len(final[i]) for i in sorted(final)],
            "oneshot_widths": [len(oneshot[i]) for i in sorted(oneshot)],
            "kept_neurons_differing_from_oneshot": sum(differ.values()), "per_layer_differing": differ,
            "total_updates": sum(s["updates"] for s in recovered["stages"])}


def _selection(output):
    selected = json.loads((output / "selection.json").read_text())
    lock = json.loads((output / "selection-lock.json").read_text())
    if _fingerprint(selected) != lock["sha256"]:
        raise ValueError("Frozen development selection changed")
    return selected


def paired_accuracy_delta(baseline, candidate, seed=17, samples=2000):
    """Paired bootstrap interval in percentage points; descriptive, not equivalence proof."""
    base_rows, candidate_rows = baseline["predictions"], candidate["predictions"]
    if type(samples) is not int or samples < 2:
        raise ValueError("Bootstrap needs at least two resamples")
    if not base_rows or [x["key"] for x in base_rows] != [x["key"] for x in candidate_rows]:
        raise ValueError("Paired comparison requires identical ordered example keys")
    differences = [int(b["correct"]) - int(a["correct"]) for a, b in zip(base_rows, candidate_rows)]
    rng = random.Random(seed)
    n = len(differences)
    bootstrap = sorted(100 * sum(rng.choices(differences, k=n)) / n for _ in range(samples))
    return {"delta_pp": 100 * sum(differences) / n,
            "bootstrap_95_percent_interval_pp": [bootstrap[int(samples * .025)], bootstrap[min(samples - 1, int(samples * .975))]],
            "regressions": sum(x == -1 for x in differences), "improvements": sum(x == 1 for x in differences),
            "note": "Paired bootstrap descriptive interval; equal sample accuracy does not prove population equivalence."}


def paired_score_delta(baseline, candidate, seed=17, samples=2000):
    """Document-cluster bootstrap for paired continuous support response scores."""
    a, b = baseline["predictions"], candidate["predictions"]
    if type(samples) is not int or samples < 2:
        raise ValueError("Bootstrap needs at least two resamples")
    if not a or [r["key"] for r in a] != [r["key"] for r in b]:
        raise ValueError("Paired comparison requires identical ordered example keys")
    if score_name(baseline) != score_name(candidate):
        raise ValueError("Cannot compare different score definitions")
    groups = {}
    differences = []
    for original, compact in zip(a, b):
        group = original.get("group_id") or original.get("doc_id")
        other_group = compact.get("group_id") or compact.get("doc_id")
        if not group or group != other_group:
            raise ValueError("Support score comparison requires matching document groups")
        first, second = float(original["score"]), float(compact["score"])
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (first, second)):
            raise ValueError("Invalid example score")
        difference = second - first
        differences.append(difference)
        groups.setdefault(group, []).append(difference)
    totals = [(sum(values), len(values)) for values in groups.values()]
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(samples):
        selected = rng.choices(totals, k=len(totals))
        bootstrap.append(100 * sum(s for s, n in selected) / sum(n for s, n in selected))
    bootstrap.sort()
    return {"delta_points": 100 * sum(differences) / len(differences),
            "bootstrap_95_interval_points": [bootstrap[int(samples * .025)], bootstrap[min(samples - 1, int(samples * .975))]],
            "bootstrap_unit": "document_group", "document_groups": len(groups),
            "score_name": score_name(baseline),
            "lower_score_examples": sum(x < -1e-12 for x in differences),
            "higher_score_examples": sum(x > 1e-12 for x in differences),
            "note": "Descriptive document-cluster interval for lexical overlap; not a correctness or equivalence guarantee."}


def evaluation_targets(selection):
    ids = {t["candidate_id"] for t in selection["tiers"] if t["candidate_id"]}
    ids.update(c["id"] for c in selection["candidates"] if c.get("control") and c["status"] == "complete")
    if selection.get("evaluate_all_candidates", False):
        ids.update(c["id"] for c in selection["candidates"] if c["status"] == "complete")
    return [{"id": "original", "checkpoint": None}] + [c for c in selection["candidates"] if c["id"] in ids]


def finalize(config, output, state, splits):
    selection = _selection(output)
    for spec in evaluation_targets(selection):
        if spec["checkpoint"]:
            _verify_checkpoint(spec)
    state["test_started"] = True
    write_json(output / "state.json", state)
    tests = {}
    for spec in evaluation_targets(selection):
        destination = output / "final-test" / f"{spec['id']}.json"
        if destination.exists():
            tests[spec["id"]] = json.loads(destination.read_text())
            continue
        print(f"Final test: {spec['id']} ({len(splits['test'])} questions)", flush=True)
        model, tokenizer = _load(config, spec["checkpoint"])
        try:
            if spec["checkpoint"] is None:
                _verify_source(config, model, selection["baseline"])
            metrics = _evaluate(model, tokenizer, splits["test"], config)
            record = {"metrics": metrics, **_provenance(model)}
            write_json(destination, record)
            tests[spec["id"]] = record
        finally:
            del model, tokenizer
            _release()
    tiers = []
    for tier in selection["tiers"]:
        result = dict(tier)
        if tier["candidate_id"]:
            metrics = tests[tier["candidate_id"]]["metrics"]
            reference_metrics = tests[selection.get("reference_id", "original")]["metrics"]
            if "selection_score" in metrics:
                comparison = paired_score_delta(reference_metrics, metrics, config["seed"])
                delta = comparison["delta_points"]
            else:
                comparison = paired_accuracy_delta(reference_metrics, metrics, config["seed"])
                delta = comparison["delta_pp"]
            if tier.get("mode") == "relative_retention":
                if score_name(reference_metrics) != score_name(metrics):
                    raise ValueError("Relative retention requires matching test score metrics")
                reference_score, candidate_score = selection_score(reference_metrics), selection_score(metrics)
                if any(not math.isfinite(x) or not 0 <= x <= 1 for x in (reference_score, candidate_score)):
                    raise ValueError("Invalid test scores for relative retention")
                floor = reference_score * tier["min_retained_score"]
                passed = reference_score > 0 and candidate_score + 1e-12 >= floor
                result.update(test=comparison, test_reference_score=reference_score,
                              test_score=candidate_score, test_floor=floor,
                              retained_score_ratio=candidate_score / reference_score if reference_score else None,
                              test_threshold_met=passed,
                              test_retention_status=("unassessable_zero_reference" if reference_score == 0
                                                     else "met" if passed else "below_threshold"))
            else:
                result.update(test=comparison, test_threshold_met=delta + 1e-12 >= -tier["max_drop_pp"])
        tiers.append(result)
    result = {"config": config, "dataset": state["dataset"], "selection": selection,
              "final_test": tests, "tiers": tiers,
              "test_selection_policy": "Selection frozen before test; failures are reported without reselection."}
    write_json(output / "summary.json", result)
    if config.get("task") == "support":
        from .support_reporting import write_blind_review
        write_blind_review(result, splits["test"], output)
    return result


def benchmark(config, output, splits):
    selection = _selection(output)
    if not (output / "summary.json").exists():
        raise ValueError("Finalize before benchmarking")
    targets = evaluation_targets(selection)
    for spec in targets:
        if spec["checkpoint"]:
            _verify_checkpoint(spec)
    # Benchmark development prompts, so repeated timing doesn't add test-driven selection.
    examples = splits["dev"][:config["benchmark_examples"]]
    measurements = {s["id"]: [] for s in targets}
    for repeat in range(config["benchmark_repeats"]):
        ordered = targets if repeat % 2 == 0 else list(reversed(targets))
        for spec in ordered:
            destination = output / "benchmark" / f"{spec['id']}-{repeat + 1}.json"
            if destination.exists():
                record = json.loads(destination.read_text())
            else:
                print(f"Benchmark {repeat + 1}: {spec['id']}", flush=True)
                model, tokenizer = _load(config, spec["checkpoint"])
                try:
                    if spec["checkpoint"] is None:
                        _verify_source(config, model, selection["baseline"])
                    # Latency benchmark stays batch-one so per-reply timings remain comparable.
                    record = {"metrics": _evaluate(model, tokenizer, examples, dict(config, eval_batch_size=1)), **_provenance(model)}
                    write_json(destination, record)
                finally:
                    del model, tokenizer
                    _release()
            measurements[spec["id"]].append(record)
    summary = {"split": "dev", "examples": len(examples), "repeats": config["benchmark_repeats"],
               "models": {}, "note": "Synchronized generation after warmup; inspect generated token counts and timing variation. Memory snapshots are not peaks."}
    for name, records in measurements.items():
        metrics = [r["metrics"] for r in records]
        latencies = [m["mean_example_seconds"] * 1000 for m in metrics]
        summary["models"][name] = {"mean_example_ms_per_run": latencies,
            "median_mean_example_ms": statistics.median(latencies),
            "tokens_per_second_per_run": [m["tokens_per_second"] for m in metrics],
            "generated_tokens_per_run": [m["generated_tokens"] for m in metrics],
            "memory_snapshots": [m["memory"] for m in metrics], "device": records[0]["device"]}
    write_json(output / "benchmark.json", summary)
    return summary


def run_campaign(config, output, stage="all"):
    from .cli import experiment_lock
    from .reporting import render_campaign_report
    output = Path(output).resolve()
    with experiment_lock():
        state, splits = _initialize(config, output)
        stages = ["preflight", "fit", "finalize", "benchmark"] if stage == "all" else [stage]
        for current in stages:
            if current == "preflight":
                if not (output / "preflight.json").exists():
                    preflight(config, output, splits)
            elif current == "fit":
                if not (output / "selection-lock.json").exists():
                    fit(config, output, state, splits)
            elif current == "finalize":
                finalize(config, output, state, splits)
            elif current == "benchmark":
                benchmark(config, output, splits)
            else:
                raise ValueError(f"Unknown campaign stage: {current}")
        if (output / "summary.json").exists():
            summary = json.loads((output / "summary.json").read_text())
            timings = json.loads((output / "benchmark.json").read_text()) if (output / "benchmark.json").exists() else None
            (output / "report.md").write_text(render_campaign_report(summary, timings))
    print(f"Campaign stage {stage} complete: {output}")
