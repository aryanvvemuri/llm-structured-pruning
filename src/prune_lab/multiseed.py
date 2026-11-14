"""Multi-seed development campaigns and seed x document-family hierarchical bootstrap.

Each recovery seed is an ordinary campaign under ``<root>/seed-<s>/`` whose configuration
differs only in ``recovery.seed`` (LoRA initialization and training-data order; generation is
greedy). Data, calibration scores, sensitivities and pruning masks are computed once, by the
first seed, and copied byte-for-byte to the others. Only preflight and fit run, so the test
split is never evaluated unless ``open_test`` is requested.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import random
import shutil
import statistics

from .reporting import write_json

METRICS = ("token_f1", "rouge_l")
# Settings that change generated replies; compared runs and seeds must share them.
EVALUATION_KEYS = ("max_length", "max_new_tokens", "eval_batch_size", "dtype", "device")


def evaluation_protocol(config):
    return {key: config.get(key, 1 if key == "eval_batch_size" else None) for key in EVALUATION_KEYS}
SHARED_FILES = ("baseline.json", "allocation.json")


def recovery_recipe(config):
    """Every recovery setting except the seed (schedule, adapters, distillation, ...)."""
    return {k: v for k, v in config.get("recovery", {}).items() if k != "seed"}


def recovery_schedule(config):
    recovery = config.get("recovery", {})
    return {"max_steps": recovery.get("max_steps"), "checkpoints": recovery.get("checkpoints")}
MIN_SEEDS = 5
DECISION_NOTE = (
    "Decision rule: a difference is 'confident' only when the hierarchical bootstrap interval AND a seed-level "
    "Student-t interval (per-seed mean differences; paired t with n-1 df within a run or with --paired-seeds, "
    "Welch otherwise) both exclude zero with the same sign; otherwise 'not established'. With 3 seeds the bootstrap "
    "alone rejected a true null in about 30% of seed-dominated simulations; the t interval restores small-seed calibration.")
INTERVAL_NOTE = (
    "Percentile interval from a hierarchical bootstrap that resamples recovery seeds and whole dev "
    "document families (examples paired by id). It covers recovery-seed variation (LoRA init and data "
    "order) as far as the observed seeds show it, and sampling of document families like those in dev. "
    "It does not cover: other data splits or recovery subsets, calibration/pruning-mask variation (fixed), "
    "hyperparameter or hardware changes, the test distribution, multiple comparisons, or optimism from "
    "choosing among many variants on the same dev set. With few seeds the seed component is poorly "
    "estimated and tends to be too narrow. Token F1/ROUGE-L are lexical proxies, not correctness.")


def seed_config(config, seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Recovery seeds must be non-negative integers")
    result = json.loads(json.dumps(config))
    result["recovery"]["seed"] = seed
    return result


def _base(config):
    base = json.loads(json.dumps(config))
    base["recovery"].pop("seed", None)
    return base


def _read(path):
    return json.loads(Path(path).read_text())


def _share_allocation(source, target):
    """Copy baseline/allocation from the first seed after checking the runs are identical."""
    if not all((source / name).exists() for name in SHARED_FILES):
        raise ValueError(f"First seed has no allocation yet: {source}")
    if _base(_read(source / "config.json")) != _base(_read(target / "config.json")):
        raise ValueError("Seed configurations differ beyond recovery.seed; refusing to share pruning masks")
    if _read(source / "state.json")["dataset"] != _read(target / "state.json")["dataset"]:
        raise ValueError("Seed datasets differ; refusing to share pruning masks")
    for name in SHARED_FILES:
        shutil.copyfile(source / name, target / name)
    write_json(target / "shared-allocation.json", {"source": str(source), "files": list(SHARED_FILES),
               "note": "Original-model dev baseline, calibration scores and pruning masks copied from the first seed."})


def discard_checkpoint_weights(seed_dir):
    """Delete *.safetensors of every exported candidate after verifying its recorded hash.

    Per-example dev predictions stay in result.json/selection.json; config, tokenizer and
    manifest files stay in place. Finalize/benchmark need the weights and will then refuse.
    """
    from .campaign import _selection, checkpoint_identity
    seed_dir = Path(seed_dir)
    selection = _selection(seed_dir)
    log_path = seed_dir / "checkpoints-deleted.json"
    log = _read(log_path) if log_path.exists() else {}
    for record in selection["candidates"]:
        if record.get("status") != "complete":
            continue
        checkpoint = Path(record["checkpoint"])
        weights = sorted(checkpoint.glob("*.safetensors"))
        if not weights:
            if record["id"] not in log:
                raise ValueError(f"Checkpoint weights missing without a deletion record: {checkpoint}")
            continue
        if checkpoint_identity(checkpoint) != record["checkpoint_sha256"]:
            raise ValueError(f"Saved checkpoint changed before deletion: {record['id']}")
        log[record["id"]] = {"checkpoint": str(checkpoint), "checkpoint_sha256": record["checkpoint_sha256"],
                             "deleted_files": [p.name for p in weights],
                             "deleted_bytes": sum(p.stat().st_size for p in weights)}
        write_json(log_path, log)  # Record intent before deleting, so a crash stays explainable.
        for path in weights:
            path.unlink()
    return log


def run_seeds(config, root, seeds, open_test=False, discard_checkpoints=False):
    from .campaign import run_campaign, validate_config
    from .cli import experiment_lock
    validate_config(config)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide one or more distinct recovery seeds")
    configs = [seed_config(config, s) for s in seeds]
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "multi-seed.json"
    plan = _read(plan_path) if plan_path.exists() else {"base_config": _base(config), "seeds": [],
                                                       "allocation_source_seed": seeds[0]}
    if plan["base_config"] != _base(config):
        raise ValueError("Configuration differs from this multi-seed run beyond recovery.seed; use a new directory")
    plan["seeds"] = plan["seeds"] + [s for s in seeds if s not in plan["seeds"]]
    write_json(plan_path, plan)
    source = root / f"seed-{plan['allocation_source_seed']}"
    order = sorted(seeds, key=lambda s: s != plan["allocation_source_seed"])
    if open_test:
        # Check every requested seed before opening the test for any of them.
        blocked = [f"seed-{s}" for s in order if (root / f"seed-{s}" / "checkpoints-deleted.json").exists()
                   and not (root / f"seed-{s}" / "summary.json").exists()]
        if blocked:
            raise ValueError(f"Checkpoint weights were deleted before the test was opened for {', '.join(blocked)}; "
                             "no seed was finalized")
    for seed in order:
        directory = root / f"seed-{seed}"
        seed_cfg = configs[seeds.index(seed)]
        print(f"Multi-seed: recovery seed {seed} -> {directory}", flush=True)
        run_campaign(seed_cfg, directory, "preflight")
        if seed != plan["allocation_source_seed"] and not (directory / "allocation.json").exists():
            if not (source / "allocation.json").exists():
                raise ValueError(f"Run the allocation source seed {plan['allocation_source_seed']} first")
            with experiment_lock():
                _share_allocation(source, directory)
        run_campaign(seed_cfg, directory, "fit")
        if open_test:
            run_campaign(seed_cfg, directory, "finalize")
        if discard_checkpoints:
            with experiment_lock():
                discard_checkpoint_weights(directory)
    return plan


def _variant(metrics, parameters, extra=None):
    rows = metrics["predictions"]
    examples = {}
    for row in rows:
        if row["key"] in examples or not row.get("group_id"):
            raise ValueError("Per-example predictions need unique keys and document group ids")
        values = {"token_f1": float(row["score"]), "rouge_l": float(row["rouge_l"])}
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in values.values()):
            raise ValueError(f"Invalid example score: {row['key']}")
        examples[row["key"]] = (row["group_id"], values)
    return {"examples": examples, "parameters": parameters, **(extra or {}),
            "aggregate": {"token_f1": 100 * metrics["token_f1"], "rouge_l": 100 * metrics["rouge_l"],
                          "answer_token_loss": metrics["answer_token_loss"],
                          "generation_limit_fraction": metrics.get("generation_limit_fraction")}}


def collect_run(root, split="dev"):
    """Read per-seed per-example scores from saved records; checkpoints are not needed."""
    from .campaign import _selection, evaluation_targets
    if split not in ("dev", "test"):
        raise ValueError("split must be dev or test")
    root = Path(root)
    directories = sorted((p for p in root.glob("seed-*") if p.is_dir() and p.name[5:].isdigit()), key=lambda p: int(p.name[5:]))
    if not directories and (root / "selection.json").exists():
        directories = [root]  # A single ordinary campaign counts as one seed.
    if not directories:
        raise ValueError(f"No seed-* campaign directories under {root}")
    seeds, incomplete, dataset, subsets, protocol, schedules, selection_keys = {}, [], None, set(), None, set(), None
    for directory in directories:
        if not (directory / "selection-lock.json").exists():
            incomplete.append(directory.name)
            continue
        selection = _selection(directory)
        config, state = _read(directory / "config.json"), _read(directory / "state.json")
        seed = config["recovery"]["seed"]
        if seed in seeds:
            raise ValueError(f"Duplicate recovery seed {seed} in {root}")
        if protocol is None:
            protocol = evaluation_protocol(config)
        elif evaluation_protocol(config) != protocol:
            raise ValueError(f"Seeds in {root} used different evaluation settings ({', '.join(EVALUATION_KEYS)})")
        if dataset is None:
            dataset = state["dataset"]
        elif state["dataset"] != dataset:
            raise ValueError(f"Seed datasets differ within {root}")
        if split == "test":
            # finalize sets test_started before evaluating; only a written summary marks completion.
            if not state.get("test_started") or not (directory / "summary.json").exists():
                incomplete.append(directory.name)
                continue
            records = {p.stem: _read(p) for p in (directory / "final-test").glob("*.json")}
            missing = [t["id"] for t in evaluation_targets(selection) if t["id"] not in records]
            if missing:
                raise ValueError(f"{directory.name}: finalized but missing test results for {', '.join(missing)}")
        subset_path = directory / "selection-dev.json"
        subset = _read(subset_path) if subset_path.exists() else None
        subsets.add(None if subset is None else (subset["count"], subset["keys_sha256"]))
        if subset is not None:
            if len(subset["keys"]) != subset["count"] or len(set(subset["keys"])) != subset["count"]:
                raise ValueError(f"{directory.name}: selection-dev.json keys do not match its count")
            selection_keys = sorted(subset["keys"])
        schedules.add(json.dumps(recovery_schedule(config), sort_keys=True))
        recipe = recovery_recipe(config)
        variants = {}
        baseline = selection["baseline"]
        if split == "dev":
            variants["original"] = _variant(baseline["dev"], baseline["parameters"], {"actual_reduction": 0.0})
        elif "original" in records:
            variants["original"] = _variant(records["original"]["metrics"], baseline["parameters"], {"actual_reduction": 0.0})
        for record in selection["candidates"]:
            if record.get("status") != "complete":
                continue
            metrics = record["dev"] if split == "dev" else records.get(record["id"], {}).get("metrics")
            if metrics is not None:
                variants[record["id"]] = _variant(metrics, record["parameters"],
                                                  {"actual_reduction": record["actual_reduction"]})
        expected = state["dataset"][split]["count"]
        for name, variant in variants.items():
            if len(variant["examples"]) != expected:
                raise ValueError(f"{directory.name}/{name}: reported {split} metrics cover {len(variant['examples'])} of {expected} examples")
            if split == "dev" and subset is not None and not set(subset["keys"]) <= set(variant["examples"]):
                raise ValueError(f"{directory.name}/{name}: selection-dev.json ids are not all in the reported dev records")
        seeds[seed] = variants
    if not seeds:
        raise ValueError(f"No completed seeds under {root}")
    if len(subsets) > 1:
        raise ValueError(f"Seeds in {root} used different checkpoint-selection dev subsets")
    if len(schedules) > 1:
        raise ValueError(f"Seeds in {root} used different recovery schedules")
    subset = next(iter(subsets))
    return {"root": str(root), "split": split, "dataset": dataset, "seeds": seeds, "incomplete": incomplete,
            "evaluation_protocol": protocol, "recovery_schedule": json.loads(next(iter(schedules))),
            "recovery_recipe": recipe,
            "selection_keys": selection_keys,
            "selection_dev_subset": None if subset is None else {"count": subset[0], "keys_sha256": subset[1]}}


def _matrix(per_seed, metric):
    """Per-seed family sums for one variant; also checks example/group pairing."""
    reference = None
    sums = {}
    for seed, variant in per_seed.items():
        groups = {key: group for key, (group, _) in variant["examples"].items()}
        if reference is None:
            reference = groups
        elif groups != reference:
            raise ValueError("Paired comparison requires identical example ids and document groups")
        total = Counter()
        for key, (group, values) in variant["examples"].items():
            total[group] += values[metric]
        sums[seed] = total
    return reference, sums


def _incomplete_beta(x, a, b):
    """Regularized incomplete beta I_x(a, b) by Lentz's continued fraction."""
    if x <= 0 or x >= 1:
        return 0.0 if x <= 0 else 1.0
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _incomplete_beta(1 - x, b, a)
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)) / a
    tiny, c, d = 1e-300, 1.0, 0.0
    f = None
    for i in range(400):
        m = i // 2
        if i == 0:
            numerator = 1.0
        elif i % 2:
            numerator = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        else:
            numerator = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + numerator * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + numerator / c
        c = c if abs(c) > tiny else tiny
        f = c * d if f is None else f * c * d
        if abs(1.0 - c * d) < 1e-15:
            break
    return front * (f - 1.0)


def t_cdf(t, df):
    if not df > 0:
        raise ValueError("Student t needs positive degrees of freedom")
    tail = 0.5 * _incomplete_beta(df / (df + t * t), df / 2, 0.5)
    return 1.0 - tail if t > 0 else tail


def t_quantile(p, df):
    """Student-t quantile by bisection on the exact CDF (df may be fractional, e.g. Welch)."""
    if not 0 < p < 1:
        raise ValueError("Quantile probability must be in (0, 1)")
    if p < 0.5:
        return -t_quantile(1 - p, df)
    low, high = 0.0, 1.0
    while t_cdf(high, df) < p:
        high *= 2
    for _ in range(200):
        middle = (low + high) / 2
        low, high = (middle, high) if t_cdf(middle, df) < p else (low, middle)
    return (low + high) / 2


def seed_t_interval(means_a, means_b, paired):
    """95% Student-t interval on per-seed mean scores (points), ignoring family sampling.

    Paired: t on per-seed differences with n-1 df. Otherwise Welch. ``means_a=None`` gives
    a one-sample interval for b. Returns interval None when fewer than two seeds (per side).
    """
    if means_a is not None and paired:
        values = [means_b[s] - means_a[s] for s in sorted(means_b)]
        method = "paired t over seeds"
    elif means_a is None:
        values = [means_b[s] for s in sorted(means_b)]
        method = "one-sample t over seeds"
    else:
        values = None
        method = "Welch t over seeds"
    if values is not None:
        n = len(values)
        estimate = statistics.fmean(values)
        if n < 2:
            return {"method": method, "estimate_points": estimate, "interval_95_points": None, "df": None, "excludes_zero": False}
        df, se = n - 1, statistics.stdev(values) / math.sqrt(n)
    else:
        xa, xb = list(means_a.values()), list(means_b.values())
        estimate = statistics.fmean(xb) - statistics.fmean(xa)
        if len(xa) < 2 or len(xb) < 2:
            return {"method": method, "estimate_points": estimate, "interval_95_points": None, "df": None, "excludes_zero": False}
        va, vb = statistics.variance(xa) / len(xa), statistics.variance(xb) / len(xb)
        se = math.sqrt(va + vb)
        df = (va + vb) ** 2 / ((va ** 2 / (len(xa) - 1)) + (vb ** 2 / (len(xb) - 1))) if va + vb > 0 else len(xa) + len(xb) - 2
    half = t_quantile(0.975, df) * se
    low, high = estimate - half, estimate + half
    return {"method": method, "estimate_points": estimate, "interval_95_points": [low, high], "df": df,
            "excludes_zero": low > 0 or high < 0}


def hierarchical_bootstrap(a, b, metric="token_f1", paired_seeds=True, samples=4000, seed=17):
    """Interval for mean(b) - mean(a) in points; ``a=None`` gives b's own mean.

    ``a``/``b`` map recovery seed -> collected variant. Each replicate draws seeds with
    replacement (jointly when paired, independently per side otherwise) and one common
    with-replacement draw of document families; the statistic is the seed-mean of the
    family-resampled example means. Components fixing one level are reported too.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    if type(samples) is not int or samples < 2:
        raise ValueError("Bootstrap needs at least two resamples")
    groups_b, sums_b = _matrix(b, metric)
    if a is not None:
        groups_a, sums_a = _matrix(a, metric)
        if groups_a != groups_b:
            raise ValueError("Paired comparison requires identical example ids and document groups")
    families = sorted(set(groups_b.values()))
    sizes = Counter(groups_b.values())
    seeds_b = sorted(sums_b)
    seeds_a = sorted(sums_a) if a is not None else []
    if a is not None and paired_seeds:
        common = sorted(set(seeds_a) & set(seeds_b))
        if not common:
            raise ValueError("Seed-paired comparison needs shared recovery seeds")
        if set(seeds_a) != set(seeds_b):
            raise ValueError(f"Seed-paired comparison needs identical seed sets, got {seeds_a} and {seeds_b}")

    def side(sums, chosen_seeds, weights, count):
        return sum(sum(w * sums[s][f] for f, w in weights.items()) for s in chosen_seeds) / (len(chosen_seeds) * count)

    def statistic(draw_a, draw_b, weights):
        count = sum(w * sizes[f] for f, w in weights.items())
        value = side(sums_b, draw_b, weights, count)
        return 100 * (value - (side(sums_a, draw_a, weights, count) if a is not None else 0.0))

    everything = {f: 1 for f in families}
    estimate = statistic(seeds_a, seeds_b, everything)
    rng = random.Random(seed)

    def interval(resample_seeds, resample_families):
        values = []
        for _ in range(samples):
            weights = Counter(rng.choices(families, k=len(families))) if resample_families else everything
            if not resample_seeds:
                draw_a, draw_b = seeds_a, seeds_b
            elif a is not None and paired_seeds:
                draw_b = rng.choices(seeds_b, k=len(seeds_b))
                draw_a = draw_b
            else:
                draw_b = rng.choices(seeds_b, k=len(seeds_b))
                draw_a = rng.choices(seeds_a, k=len(seeds_a)) if a is not None else []
            values.append(statistic(draw_a, draw_b, weights))
        values.sort()
        return [values[int(samples * .025)], values[min(samples - 1, int(samples * .975))]]

    low, high = interval(True, True)
    total = sum(sizes.values())
    means = lambda sums: {s: 100 * sum(sums[s].values()) / total for s in sums}
    t_result = seed_t_interval(means(sums_a) if a is not None else None, means(sums_b), paired_seeds)
    bootstrap_excludes = a is not None and (low > 0 or high < 0)
    t_interval = t_result["interval_95_points"]
    confident = (bootstrap_excludes and t_result["excludes_zero"]
                 and (low > 0) == (t_interval[0] > 0))
    return {"metric": metric, "estimate_points": estimate, "interval_95_points": [low, high],
            "excludes_zero": bootstrap_excludes, "seed_t_interval": t_result,
            "decision": None if a is None else ("confident" if confident else "not established"),
            "decision_rule": "confident only if the hierarchical bootstrap AND the seed-level t interval exclude zero with the same sign",
            "seeds_only_interval": interval(True, False), "families_only_interval": interval(False, True),
            "seeds_a": seeds_a, "seeds_b": seeds_b, "seed_pairing": "paired" if a is not None and paired_seeds else "independent",
            "document_families": len(families), "examples": len(groups_b), "bootstrap_samples": samples, "bootstrap_seed": seed}


def _variant_order(names):
    fixed = ["original", "unpruned-recovery", "uniform-10"]
    return [n for n in fixed if n in names] + sorted(n for n in names if n not in fixed)


def _per_variant(run):
    names = _variant_order({name for variants in run["seeds"].values() for name in variants})
    result = {}
    for name in names:
        seeds = {seed: variants[name] for seed, variants in sorted(run["seeds"].items()) if name in variants}
        parameters = sorted({v["parameters"] for v in seeds.values()})
        if len(parameters) != 1:
            raise ValueError(f"Parameter count of {name} differs across seeds")
        entry = {"seeds": sorted(seeds), "parameters": parameters[0],
                 "actual_reduction": next(iter(seeds.values()))["actual_reduction"], "metrics": {}}
        for metric in ("token_f1", "rouge_l", "answer_token_loss", "generation_limit_fraction"):
            values = [v["aggregate"][metric] for v in seeds.values()]
            if any(x is None for x in values):
                continue
            entry["metrics"][metric] = {"per_seed": dict(zip(map(str, sorted(seeds)), values)),
                                        "mean": statistics.fmean(values),
                                        "sd": statistics.stdev(values) if len(values) > 1 else None}
        result[name] = (entry, seeds)
    return result


def _comparison(label_a, seeds_a, label_b, seeds_b, paired, samples, seed):
    result = {"a": label_a, "b": label_b, "difference": "b minus a"}
    for metric in METRICS:
        result[metric] = hierarchical_bootstrap(seeds_a, seeds_b, metric, paired, samples, seed)
    used_a, used_b = result["token_f1"]["seeds_a"], result["token_f1"]["seeds_b"]
    result["answer_token_loss"] = {"mean_difference": statistics.fmean(seeds_b[s]["aggregate"]["answer_token_loss"] for s in used_b)
                                   - statistics.fmean(seeds_a[s]["aggregate"]["answer_token_loss"] for s in used_a),
                                   "note": "Seed means only: per-example losses are not stored, so no interval."}
    return result


def _restrict(run, excluded):
    """Copy of a collected run without ``excluded`` examples; F1/ROUGE-L means are recomputed."""
    excluded = set(excluded)
    seeds = {}
    for seed, variants in run["seeds"].items():
        seeds[seed] = {}
        for name, variant in variants.items():
            examples = {k: v for k, v in variant["examples"].items() if k not in excluded}
            if not examples:
                raise ValueError("No examples remain after removing the selection subset")
            aggregate = dict(variant["aggregate"])
            for metric in METRICS:
                aggregate[metric] = 100 * statistics.fmean(v[metric] for _, v in examples.values())
            seeds[seed][name] = dict(variant, examples=examples, aggregate=aggregate)
    return dict(run, seeds=seeds)


def _example_sets(run, excluded):
    """Headline first: dev-holdout when a selection subset exists, otherwise the full split."""
    count = len(next(iter(next(iter(run["seeds"].values())).values()))["examples"])
    if run["split"] != "dev" or not excluded:
        return [(run["split"] if run["split"] == "test" else "dev-full", run, count)]
    holdout = _restrict(run, excluded)
    return [("dev-holdout", holdout, count - len(set(excluded))), ("dev-full", run, count)]


def _within_run(run, samples, seed):
    variants = _per_variant(run)
    names = list(variants)
    comparisons = []
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            seeds_a, seeds_b = variants[first][1], variants[second][1]
            common = sorted(set(seeds_a) & set(seeds_b))
            if not common:
                continue
            comparison = _comparison(first, {s: seeds_a[s] for s in common}, second, {s: seeds_b[s] for s in common},
                                     True, samples, seed)
            excluded = sorted(set(seeds_a) ^ set(seeds_b))
            if excluded:
                comparison["seeds_excluded"] = excluded
                comparison["seeds_excluded_note"] = "Seeds where only one variant completed are left out of this paired comparison."
            comparisons.append(comparison)
    entries = {name: dict(entry, token_f1_interval=hierarchical_bootstrap(None, per_seed, "token_f1", True, samples, seed))
               for name, (entry, per_seed) in variants.items()}
    return entries, comparisons


def aggregate_run(root, split="dev", samples=4000, seed=17):
    run = collect_run(root, split)
    sets = _example_sets(run, run["selection_keys"])
    results = {name: (_within_run(view, samples, seed), count) for name, view, count in sets}
    headline, secondary = sets[0][0], sets[1:]
    (variants, comparisons), count = results[headline]
    summary = {"root": run["root"], "split": split, "dataset": run["dataset"],
               "seeds": sorted(run["seeds"]), "incomplete_seed_directories": run["incomplete"],
               "example_set": headline, "example_count": count,
               "variants": variants, "comparisons": comparisons,
               "secondary": {name: {"example_count": results[name][1], "variants": results[name][0][0],
                                    "comparisons": results[name][0][1]} for name, _, _ in secondary},
               "interval_note": INTERVAL_NOTE, "decision_note": DECISION_NOTE,
               "selection_dev_subset": run["selection_dev_subset"],
               "selection_note": _selection_note(run["selection_dev_subset"], run["dataset"]["dev"]["count"]),
               "recovery_schedule": run["recovery_schedule"], "recovery_recipe": run["recovery_recipe"],
               "within_run_seed_pairing": "Variants from the same seed directory are paired by seed."}
    suffix = "" if split == "dev" else "-test"
    write_json(Path(root) / f"multi-seed-summary{suffix}.json", summary)
    (Path(root) / f"multi-seed-report{suffix}.md").write_text(render_summary(summary))
    return summary


def _cross_run(a, b, pairs, paired_seeds, samples, seed):
    va, vb = _per_variant(a), _per_variant(b)
    pairs = pairs or [(n, n) for n in va if n in vb]
    if not pairs:
        raise ValueError("No common variants to compare")
    comparisons = []
    for name_a, name_b in pairs:
        if name_a not in va or name_b not in vb:
            raise ValueError(f"Unknown variant pair {name_a}:{name_b}")
        sa, sb = va[name_a][1], vb[name_b][1]
        paired = set(sa) == set(sb) if paired_seeds is None else paired_seeds
        comparisons.append(_comparison(f"{a['root']}:{name_a}", sa, f"{b['root']}:{name_b}", sb, paired, samples, seed))
    return {"baseline": {n: e for n, (e, _) in va.items()}, "candidate": {n: e for n, (e, _) in vb.items()}}, comparisons


def compare_runs(baseline_root, candidate_root, pairs=None, split="dev", paired_seeds=None, samples=4000, seed=17,
                 allow_different_schedules=False):
    """Compare variants across two multi-seed runs on the same split data (e.g. new method vs main).

    ``paired_seeds=None`` pairs by seed value whenever a pair's seed sets are identical (the
    documented protocol) and resamples seeds independently otherwise; True/False force it.
    On dev with a selection subset, the headline uses dev-holdout (examples outside both runs'
    selection subsets). Dev-full decisions need identical recovery schedules unless allowed.
    """
    a, b = collect_run(baseline_root, split), collect_run(candidate_root, split)
    if a["dataset"][split] != b["dataset"][split]:
        raise ValueError(f"Runs use different {split} data; comparison would be unpaired")
    if a["evaluation_protocol"] != b["evaluation_protocol"]:
        raise ValueError(f"Runs use different evaluation settings: {a['evaluation_protocol']} vs {b['evaluation_protocol']}")
    excluded = sorted(set(a["selection_keys"] or []) | set(b["selection_keys"] or []))
    same_schedule = a["recovery_schedule"] == b["recovery_schedule"]
    sets = [(name, view) for name, view, _ in _example_sets(a, excluded)]
    sets_b = {name: view for name, view, _ in _example_sets(b, excluded)}
    withheld = None
    if split == "dev" and not same_schedule and not allow_different_schedules:
        if sets[0][0] == "dev-full":
            raise ValueError("Runs use different recovery schedules (checkpoints/max_steps) and in-training selection used the "
                             "reported dev examples, so a dev-full decision is biased; pass --allow-different-schedules to override")
        withheld = "dev-full omitted: recovery schedules differ and selection optimism would differ between runs."
        sets = [item for item in sets if item[0] != "dev-full"]
    results = {}
    for name, view in sets:
        variants, comparisons = _cross_run(view, sets_b[name], pairs, paired_seeds, samples, seed)
        results[name] = {"example_count": len(next(iter(next(iter(view["seeds"].values())).values()))["examples"]),
                         "variants": variants, "comparisons": comparisons}
    headline = sets[0][0]
    pairing = ("auto: paired by seed value when seed sets are identical, otherwise independent" if paired_seeds is None
               else "paired by seed value" if paired_seeds else "independent seed resampling per run")
    return {"baseline_root": a["root"], "candidate_root": b["root"], "split": split,
            "example_set": headline, "example_count": results[headline]["example_count"],
            "variants": results[headline]["variants"], "comparisons": results[headline]["comparisons"],
            "secondary": {name: value for name, value in results.items() if name != headline},
            "secondary_withheld": withheld,
            "recovery_recipes": {"baseline": a["recovery_recipe"], "candidate": b["recovery_recipe"]},
            "recovery_schedules": {"baseline": a["recovery_schedule"], "candidate": b["recovery_schedule"],
                                   "identical": same_schedule},
            "allow_different_schedules": allow_different_schedules,
            "selection_dev_subsets": {"baseline": a["selection_dev_subset"], "candidate": b["selection_dev_subset"],
                                      "identical": a["selection_dev_subset"] == b["selection_dev_subset"],
                                      "excluded_from_holdout": len(excluded)},
            "selection_note": _selection_note(b["selection_dev_subset"], b["dataset"]["dev"]["count"]),
            "seed_pairing": pairing, "interval_note": INTERVAL_NOTE, "decision_note": DECISION_NOTE}


def _selection_note(subset, dev_count):
    if subset is None:
        return f"In-training checkpoint selection used all {dev_count} dev examples, the same set that is reported."
    return (f"In-training checkpoint selection used a fixed {subset['count']}-example domain-stratified dev subset "
            f"(keys sha256 {subset['keys_sha256'][:12]}…), which is part of the {dev_count}-example dev set. Headline numbers and "
            f"decisions use dev-holdout (the {dev_count - subset['count']} dev examples outside it); dev-full is secondary and "
            "slightly optimistic, by amounts that can differ between methods. Answer loss is only stored for full dev.")


def _fmt(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def _interval(values):
    return "—" if values is None else f"[{values[0]:+.2f}, {values[1]:+.2f}]"


def _comparison_rows(comparisons):
    lines = ["| Comparison (b − a) | F1 decision | ΔF1 | Bootstrap 95% | Seed t 95% | Seeds a / b | ΔROUGE-L (decision) | Δ answer loss |",
             "| --- | --- | ---: | --- | --- | --- | --- | ---: |"]
    for c in comparisons:
        f1, rl = c["token_f1"], c["rouge_l"]
        lines.append(f"| {c['b']} − {c['a']} | **{f1['decision']}** | {f1['estimate_points']:+.2f} | "
                     f"{_interval(f1['interval_95_points'])} | {_interval(f1['seed_t_interval']['interval_95_points'])} | "
                     f"{len(f1['seeds_a'])} / {len(f1['seeds_b'])} ({f1['seed_pairing']}) | "
                     f"{rl['estimate_points']:+.2f} ({rl['decision']}) | {c['answer_token_loss']['mean_difference']:+.4f} |")
    return lines


def _decision_lines(comparisons):
    confident = [f"{c['b']} − {c['a']} ({c['token_f1']['estimate_points']:+.2f} F1)" for c in comparisons
                 if c["token_f1"]["decision"] == "confident"]
    lines = ["## Decision", "", "Confident token-F1 differences: " + ("; ".join(confident) if confident else "none") + ".",
             "", DECISION_NOTE]
    fewest = min((min(len(c["token_f1"]["seeds_a"]), len(c["token_f1"]["seeds_b"])) for c in comparisons), default=0)
    if fewest < MIN_SEEDS:
        lines += ["", f"**Warning:** only {fewest} recovery seed(s) per side; fewer than {MIN_SEEDS} seeds makes the "
                  "seed component weakly estimated and the decision low-powered."]
    return lines


def _secondary_lines(result):
    lines = []
    for name, value in result.get("secondary", {}).items():
        lines += ["", f"## Secondary: {name} ({value['example_count']} examples; not used for decisions)", ""]
        lines += _comparison_rows(value["comparisons"])
    return lines


def render_summary(summary):
    lines = [f"# Multi-seed {summary['split']} summary", "",
             f"Run: `{summary['root']}`. Recovery seeds: {', '.join(map(str, summary['seeds']))}."
             + (f" Incomplete (ignored): {', '.join(summary['incomplete_seed_directories'])}." if summary['incomplete_seed_directories'] else ""),
             "", f"Headline example set: **{summary['example_set']}** ({summary['example_count']} examples).", ""] + _decision_lines(summary["comparisons"]) + [
             "", "## Variants", "",
             "F1 and ROUGE-L are 0–100 word-overlap scores, **not accuracy**. Original Qwen is not retrained, so its sd is 0.", "",
             "| Variant | Parameters | Removed | Seeds | F1 mean ± sd | F1 95% interval | Per-seed F1 | ROUGE-L mean ± sd | Answer loss mean ± sd |",
             "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |"]
    for name, v in summary["variants"].items():
        m = v["metrics"]
        interval = v["token_f1_interval"]["interval_95_points"]
        lines.append(f"| {name} | {v['parameters']:,} | {100 * v['actual_reduction']:.3f}% | {len(v['seeds'])} | "
                     f"{_fmt(m['token_f1']['mean'])} ± {_fmt(m['token_f1']['sd'])} | [{interval[0]:.2f}, {interval[1]:.2f}] | "
                     f"{', '.join(_fmt(x) for x in m['token_f1']['per_seed'].values())} | "
                     f"{_fmt(m['rouge_l']['mean'])} ± {_fmt(m['rouge_l']['sd'])} | "
                     f"{_fmt(m['answer_token_loss']['mean'], 4)} ± {_fmt(m['answer_token_loss']['sd'], 4)} |")
    lines += ["", "## Paired differences", ""] + _comparison_rows(summary["comparisons"])
    lines += _secondary_lines(summary)
    lines += ["", "## What the interval covers", "", summary["interval_note"], "",
              summary["within_run_seed_pairing"] + " `seeds_only_interval` and `families_only_interval` in the JSON "
              "show each source of variation alone. Accept a change only if a predeclared comparison is 'confident'; "
              "scanning every pair inflates false positives.", "", summary["selection_note"]]
    return "\n".join(lines) + "\n"


def render_comparison(result):
    lines = [f"# Cross-run {result['split']} comparison", "",
             f"Baseline (a): `{result['baseline_root']}`. Candidate (b): `{result['candidate_root']}`. "
             f"Seeds: {result['seed_pairing']}. Examples are paired by id; document families are resampled jointly.", "",
             f"Headline example set: **{result['example_set']}** ({result['example_count']} examples). Recovery schedules: "
             f"baseline {json.dumps(result['recovery_schedules']['baseline'])}, candidate {json.dumps(result['recovery_schedules']['candidate'])}"
             + (" (different schedules explicitly allowed)." if result["allow_different_schedules"] else "."), ""]
    lines += _decision_lines(result["comparisons"]) + ["", "## Differences", ""]
    lines += _comparison_rows(result["comparisons"])
    lines += _secondary_lines(result)
    if result.get("secondary_withheld"):
        lines += ["", result["secondary_withheld"]]
    if not result["selection_dev_subsets"]["identical"]:
        lines += ["", "**Warning:** the two runs used different checkpoint-selection dev subsets."]
    lines += ["", "## What the interval covers", "", result["interval_note"], "", result["selection_note"]]
    return "\n".join(lines) + "\n"
