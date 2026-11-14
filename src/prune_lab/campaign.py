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
    from .recipes import validate_recipe, validate_distillation
    validate_recipe(config["recovery"])
    if validate_distillation(config["recovery"]) is not None and not config["include_controls"]:
        raise ValueError("Distillation needs the unpruned-recovery control as its teacher")


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


def candidate_specs(config):
    result = [{"id": f"per-layer-{i:02d}", "target": target, "uniform": False, "control": False}
              for i, target in enumerate(config["targets"], 1)]
    if config["include_controls"]:
        result += [{"id": "unpruned-recovery", "target": 0.0, "uniform": False, "control": True},
                   {"id": "uniform-10", "target": 0.10, "uniform": True, "control": True,
                    "selection_eligible": config.get("select_uniform", False)}]
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
            allocations = []
            for spec in candidate_specs(config):
                if spec["target"] == 0:
                    keep = {i: list(range(m.down_proj.in_features)) for i, m in enumerate(mlp_layers(model))}
                else:
                    keep = allocate_neurons(model, scores, sensitivity, spec["target"],
                                            config["group_size"], config["min_keep_ratio"], spec["uniform"])
                allocations.append(dict(spec, keep_indices=keep))
            write_json(output / "baseline.json", baseline)
            write_json(output / "allocation.json", {"candidates": allocations, "sensitivity": sensitivity,
                       "elapsed_seconds": time.perf_counter() - start, "baseline_sha256": _fingerprint(baseline)})
        finally:
            del model, tokenizer
            _release()
    baseline = json.loads((output / "baseline.json").read_text())
    allocation = json.loads((output / "allocation.json").read_text())
    if allocation["baseline_sha256"] != _fingerprint(baseline):
        raise ValueError("Baseline changed after allocation")
    selection_dev = _selection_dev(config, output, splits.get("dev"))
    skip_merged = config.get("skip_full_merged_dev", False)
    distill = config.get("recovery", {}).get("distillation")
    specs = allocation["candidates"]
    if distill is not None:
        # The teacher (this seed's unpruned-recovery export) must be trained before any student.
        specs = sorted(specs, key=lambda spec: spec["id"] != TEACHER_ID)
    candidates = []
    for spec in specs:
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
        model = tokenizer = teacher = None
        try:
            model, tokenizer = _load(config)
            _verify_source(config, model, baseline)
            if spec["target"]:
                compact_model(model, {int(k): v for k, v in spec["keep_indices"].items()})
            parameters = count_parameters(model)
            batches = make_batches(splits["recovery"], tokenizer, config["max_length"])
            teaching = None
            if distill is not None and spec["target"]:
                teacher, teaching = _load_teacher(config, candidates)
            recovered = recover_model(model, batches, lambda m: _evaluate(m, tokenizer, selection_dev, config), config["recovery"],
                                      **({"teacher": teacher} if teacher is not None else {}))
            if teacher is not None:
                del teacher
                teacher = None
                _release()
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
                      "memory_after": memory_snapshot(next(model.parameters()).device), **_provenance(model),
                      **({"distillation": dict(teaching, alpha=distill["alpha"], temperature=distill["temperature"])}
                         if teaching else {})}
            manifest = {"config": config, "dataset": state["dataset"], "selection_split": "dev",
                        "keep_indices": spec["keep_indices"], "candidate": record}
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
            write_json(record_path, record)
            candidates.append(record)
        except Exception as exc:
            failed = {**{k: v for k, v in spec.items() if k != "keep_indices"}, "status": "failed",
                      "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.perf_counter() - start}
            write_json(record_path, failed)
            candidates.append(failed)
            print(f"Candidate failed: {failed['error']}", flush=True)
        finally:
            del model, tokenizer, teacher
            _release()
    order = [spec["id"] for spec in allocation["candidates"]]
    candidates.sort(key=lambda record: order.index(record["id"]))
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


TEACHER_ID = "unpruned-recovery"


def _load_teacher(config, candidates):
    """Load this seed's recovered full model from its export, hash-checked, frozen and in eval mode."""
    record = next((c for c in candidates if c["id"] == TEACHER_ID), None)
    if record is None or record.get("status") != "complete":
        raise ValueError("Distillation teacher (unpruned-recovery) is not complete for this seed")
    _verify_checkpoint(record)
    teacher, _ = _load(config, record["checkpoint"])
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher, {"teacher_id": TEACHER_ID, "teacher_checkpoint": record["checkpoint"],
                     "teacher_checkpoint_sha256": record["checkpoint_sha256"],
                     "teacher_memory_after_load": memory_snapshot(next(teacher.parameters()).device)}


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
