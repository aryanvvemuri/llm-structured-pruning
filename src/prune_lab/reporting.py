import json
import os
from pathlib import Path
import tempfile


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def tier_requirement(tier):
    if tier.get("mode") == "relative_retention":
        return f"At least {100 * tier['min_retained_score']:g}% of baseline"
    return f"At most {tier['max_drop_pp']:g}-point drop"


def tier_test_status(tier):
    if tier.get("test_retention_status") == "unassessable_zero_reference":
        return "Unassessable (zero baseline)"
    return "Yes" if tier["test_threshold_met"] else "No"


def tier_retained_score(tier):
    ratio = tier.get("retained_score_ratio")
    return f"{100 * ratio:.2f}%" if ratio is not None else "—"


def render_report(result):
    search = result.get("search", {})
    lines = ["# Pruning experiment", "", f"Model: `{result.get('config', {}).get('model', 'unknown')}`", "", "Metrics use numeric exact match on generated completion only. Small samples are smoke checks, not quality evidence.", ""]
    if "parameters" in result:
        lines += [f"Parameters: {result['parameters']:,}; parameter tensor bytes: {result.get('parameter_bytes', 'unavailable')}. Device: {result.get('resolved_device', 'unknown')}. Checkpoint file size and runtime RSS are separate measurements.", ""]
    if search:
        lines += [f"Method: {search['method']}; MLP width: {search['original_width']} → {search['retained_width']}.", f"Allowed accuracy drop: {search['max_drop_pp']} percentage points relative to original development score.", ""]
    if "search_seconds" in result:
        lines += [f"Search elapsed time: {result['search_seconds']:.3f}s (includes baseline development evaluation and calibration scoring; excludes final test and model loading).", ""]
    for name, metrics in [("Original development", search.get("baseline_dev")), ("Selected development", search.get("selected_dev")), ("Final test", result.get("final_test")), ("Evaluation", result.get("metrics"))]:
        if metrics:
            lines += [f"## {name}", "", f"Accuracy: {metrics['correct']}/{metrics['count']} ({metrics['accuracy']:.1%}); answer-token loss: {metrics['answer_token_loss']:.4f}.", f"Generation: {metrics['tokens_per_second']:.2f} tokens/s; mean example latency: {metrics['mean_example_seconds']:.3f}s after one warmup.", f"Memory snapshot: `{json.dumps(metrics['memory'], sort_keys=True)}`. RSS is process memory; MPS counters, when present, are separate and are not additive to RSS.", ""]
    lines += ["Masked evaluation does not reduce model storage or dense matrix dimensions. Only structural export reduces parameters; benchmark the reloaded export separately. Timing is local to this run and does not establish a speedup without a matching baseline. Final test is evaluated once after selection and is never consulted by search.", ""]
    return "\n".join(lines)


def render_campaign_report(summary, timings=None):
    if summary.get("config", {}).get("task") == "support":
        from .support_reporting import render_support_report
        return render_support_report(summary, timings)
    selection = summary["selection"]
    baseline = selection["baseline"]
    original = summary["final_test"]["original"]["metrics"]
    candidates = {row["id"]: row for row in selection["candidates"]}
    reference_id = selection.get("reference_id", "original")
    relative = any(tier.get("mode") == "relative_retention" for tier in selection["tiers"])
    requirement_note = ("Requirements are fractions of baseline accuracy: 95% retention multiplies baseline accuracy by 0.95. "
                        "It does not mean 95% absolute accuracy or a five-percentage-point drop. " if relative else
                        "Tolerances are absolute percentage-point drops from baseline accuracy. ")
    counts = ", ".join(f"{name}: {item['count']}" for name, item in summary["dataset"].items())
    lines = ["# Per-layer pruning and recovery", "",
             f"Model: `{summary['config']['model']}`. Dataset counts — {counts}.", "",
             f"Original parameters: {baseline['parameters']:,}. Original development accuracy: {baseline['dev']['accuracy']:.2%}. "
             f"Original final-test accuracy: {original['correct']}/{original['count']} ({original['accuracy']:.2%}).", "",
             f"Selection reference: **{reference_id}**. Targets define candidate parameter reductions, not a required final size. "
             + requirement_note +
             "Selection uses reloaded checkpoints on development data and is frozen before the final test.", "",
             "| Quality requirement | Selected candidate | Parameters removed | Final-test accuracy | Change vs reference | Test score retained | Test threshold met |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for tier in summary["tiers"]:
        name = tier["candidate_id"]
        if name is None:
            lines.append(f"| {tier_requirement(tier)} | No qualifying pruned model | — | — | — | — | — |")
            continue
        record = candidates[name]
        metrics = summary["final_test"][name]["metrics"]
        lines.append(f"| {tier_requirement(tier)} | {name} | {record['actual_reduction']:.2%} | "
                     f"{metrics['correct']}/{metrics['count']} ({metrics['accuracy']:.2%}) | "
                     f"{tier['test']['delta_pp']:+.2f} pp | {tier_retained_score(tier)} | {tier_test_status(tier)} |")
    lines += ["", "A checkpoint can qualify for several tiers. Missing tiers and failed held-out thresholds are reported without reselection. "
              "Equal sample accuracy does not establish equal population accuracy. Retention targets are checked against the reference's test score, "
              "not its development score; an observed threshold pass is not a statistical guarantee. Selection finds the smallest qualifying tested candidate, not a global optimum.", "", "## Candidates and controls", "",
              "| Candidate | Target reduction | Actual reduction | Development accuracy | Recovery step selected | Parameters |",
              "| --- | --- | --- | --- | --- | --- |"]
    for record in selection["candidates"]:
        name = record["id"] + (" (control)" if record.get("control") else "")
        if record["status"] != "complete":
            error = record.get("error", "unknown error").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {name} | {record['target']:.0%} | Failed: {error} | — | — | — |")
            continue
        lines.append(f"| {name} | {record['target']:.0%} | {record['actual_reduction']:.2%} | "
                     f"{record['dev']['accuracy']:.2%} | {record['recovery']['best_step']} | {record['parameters']:,} |")
    lines += ["", "The unpruned recovery control tests whether training alone accounts for gains. The uniform 10% control tests the per-layer allocation at a similar size. "
              "Controls compete only when explicitly marked selection-eligible.", "",
              "## Held-out comparisons", ""]
    reported = set()
    for tier in summary["tiers"]:
        name = tier["candidate_id"]
        if name is None or name in reported:
            continue
        reported.add(name)
        change = tier["test"]
        low, high = change["bootstrap_95_percent_interval_pp"]
        lines += [f"- {name}: {change['delta_pp']:+.2f} pp; paired bootstrap 95% interval [{low:+.2f}, {high:+.2f}] pp. "
                  f"{change['regressions']} regressions; {change['improvements']} improvements."]
    for record in selection["candidates"]:
        if record.get("control") and record["id"] in summary["final_test"]:
            metrics = summary["final_test"][record["id"]]["metrics"]
            lines += [f"- {record['id']}: {metrics['correct']}/{metrics['count']} ({metrics['accuracy']:.2%})."]
    lines += ["", "Intervals are descriptive paired bootstrap estimates, not formal evidence of equivalence. "
              "The arithmetic task includes simple word templates and does not establish general math capability.", "",
              "## Checkpoints", ""]
    for record in selection["candidates"]:
        if record["status"] == "complete":
            lines += [f"- {record['id']}: `{record['checkpoint']}`; file size {record['checkpoint_bytes'] / 2**20:.1f} MiB; "
                      f"parameter tensors {record['parameter_bytes'] / 2**20:.1f} MiB; fitting and reload validation {record['elapsed_seconds']:.1f}s. "
                      f"Layer widths: `{record['layer_widths']}`."]
    lines += ["", "Use `prune_lab.modeling.load_model` for heterogeneous checkpoints. Recovery adapters are merged before export. "
              "Weights, tokenizer, configuration and manifest are fingerprinted before selection.", ""]
    if timings:
        lines += ["## Local timing", "", f"{timings['repeats']} repeats on {timings['examples']} development prompts, with synchronized generation after warmup.", "",
                  "| Model | Median mean latency (ms) | Run means (ms) | Tokens/s per run | Generated tokens per run | Device |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for name, timing in timings["models"].items():
            runs = ", ".join(f"{x:.2f}" for x in timing["mean_example_ms_per_run"])
            rates = ", ".join(f"{x:.2f}" for x in timing["tokens_per_second_per_run"])
            lines.append(f"| {name} | {timing['median_mean_example_ms']:.2f} | {runs} | {rates} | {timing['generated_tokens_per_run']} | {timing['device']} |")
        lines += ["", "Models run one at a time in the same process with cache cleanup between loads. Compare variation and generated token counts before claiming a speedup. "
                  "Memory snapshots are recorded in benchmark.json; they are not peaks, and RSS and MPS counters are not additive.", ""]
    else:
        lines += ["Timing benchmark has not run. Fewer parameters alone does not establish a latency or memory improvement.", ""]
    return "\n".join(lines)
