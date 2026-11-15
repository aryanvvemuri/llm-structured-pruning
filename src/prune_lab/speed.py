"""Ratio-to-reference arithmetic for repeat-based speed benchmarks (improvement 8d)."""
from __future__ import annotations

import math
import random
import statistics

# Higher is faster for throughput; lower is faster for times. Ratios are always model / reference.
METRICS = {"batch_decode_tokens_per_second": "higher", "decode_ms_per_token": "lower",
           "prefill_ms_per_prompt": "lower", "batch_prefill_ms_per_prompt": "lower"}
PRIMARY = "batch_decode_tokens_per_second"


def ratio_interval(reference, model, samples=2000, seed=17):
    """median(model) / median(reference) with a 95% percentile bootstrap over repeats.

    ``reference[i]`` and ``model[i]`` are the same repeat (models are interleaved within each repeat),
    so repeats are resampled jointly. Returns the point ratio, the interval and both medians.
    """
    if len(reference) != len(model) or len(reference) < 2:
        raise ValueError("Need the same number (>= 2) of repeats for the model and the reference")
    values = [float(v) for v in list(reference) + list(model)]
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Timings must be finite and positive")
    if type(samples) is not int or samples < 2:
        raise ValueError("Bootstrap needs at least two resamples")
    n = len(reference)
    rng = random.Random(seed)
    ratios = []
    for _ in range(samples):
        picks = [rng.randrange(n) for _ in range(n)]
        ratios.append(statistics.median(model[i] for i in picks) / statistics.median(reference[i] for i in picks))
    ratios.sort()
    return {"ratio": statistics.median(model) / statistics.median(reference),
            "interval_95": [ratios[int(samples * .025)], ratios[min(samples - 1, int(samples * .975))]],
            "model_median": statistics.median(model), "reference_median": statistics.median(reference),
            "repeats": n, "bootstrap_samples": samples, "bootstrap_seed": seed}


def verdict(metric, interval):
    """'faster' / 'slower' when the whole interval is on one side of 1, else 'not established'."""
    low, high = interval
    higher_is_faster = METRICS[metric] == "higher"
    if low > 1:
        return "faster" if higher_is_faster else "slower"
    if high < 1:
        return "slower" if higher_is_faster else "faster"
    return "not established"


def summarize(runs, reference, samples=2000, seed=17):
    """``runs[model]`` is a list of per-repeat dicts of METRICS. Per-model medians and ratios to ``reference``."""
    if reference not in runs:
        raise ValueError(f"Reference {reference!r} is not among the benchmarked models")
    result = {}
    for name, repeats in runs.items():
        entry = {"medians": {m: statistics.median(r[m] for r in repeats) for m in METRICS if m in repeats[0]}}
        if name != reference:
            entry["vs_reference"] = {}
            for metric in entry["medians"]:
                ratio = ratio_interval([r[metric] for r in runs[reference]], [r[metric] for r in repeats], samples, seed)
                entry["vs_reference"][metric] = dict(ratio, confidently=verdict(metric, ratio["interval_95"]))
        result[name] = entry
    return result
