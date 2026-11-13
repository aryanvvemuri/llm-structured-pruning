"""Run the saved original/compact comparison from a GPU-enabled macOS Terminal."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    # Fail visibly on unsupported operations instead of silently timing CPU work.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import torch
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable here. Run this script in your normal macOS Terminal; no CPU fallback was used.")
    probe = torch.ones(1, device="mps")
    assert (probe + probe).item() == 2
    del probe
    torch.mps.synchronize()
    torch.mps.empty_cache()

    project = Path(__file__).resolve().parents[1]
    config = project / "runs/adaptive/config.json"
    checkpoint = project / "runs/adaptive/compact"
    if not config.is_file() or not (checkpoint / "config.json").is_file():
        raise SystemExit("The saved adaptive configuration and compact checkpoint are required.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = project / "runs" / ("mps-benchmark-" + stamp)
    output.mkdir(parents=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_HOME=str(project / ".cache/huggingface"))
    results = {"original": [], "compact": []}
    for repeat in range(args.repeats):
        order = ("original", "compact") if repeat % 2 == 0 else ("compact", "original")
        for name in order:
            destination = output / f"{name}-{repeat + 1}.json"
            command = [sys.executable, "-m", "prune_lab", "evaluate", "--config", str(config),
                       "--device", "mps", "--split", "test", "--output", str(destination)]
            if name == "compact":
                command.extend(["--model", str(checkpoint)])
            print(f"Repeat {repeat + 1}/{args.repeats}: {name} on MPS", flush=True)
            # Fresh process for each model; the evaluator warms up and synchronizes.
            subprocess.run(command, cwd=project, env=env, check=True)
            result = json.loads(destination.read_text())
            if torch.device(result["resolved_device"]).type != "mps":
                raise RuntimeError("Benchmark did not run on MPS")
            results[name].append(result)

    reference = results["original"][0]
    for runs in results.values():
        for run in runs:
            if run["dataset"] != reference["dataset"] or run["metrics"]["generation_config"] != reference["metrics"]["generation_config"]:
                raise RuntimeError("Dataset or generation configuration differs between runs")
    summary = {"device": "mps", "repeats": args.repeats, "dtype": reference["config"]["dtype"],
               "versions": reference["versions"], "dataset": reference["dataset"],
               "note": "End-to-end generation on identical prompts; output lengths can differ. Inspect token counts. Memory values are snapshots, not peaks. Repeated timings do not establish statistical significance."}
    for name, runs in results.items():
        metrics = [run["metrics"] for run in runs]
        latencies = [m["mean_example_seconds"] * 1000 for m in metrics]
        summary[name] = {
            "parameters": runs[0]["parameters"],
            "accuracy_per_run": [m["accuracy"] for m in metrics],
            "correct_per_run": [m["correct"] for m in metrics],
            "examples": metrics[0]["count"],
            "generated_tokens_per_run": [m["generated_tokens"] for m in metrics],
            "mean_example_ms_per_run": latencies,
            "median_mean_example_ms": statistics.median(latencies),
            "tokens_per_second_per_run": [m["tokens_per_second"] for m in metrics],
            "memory_snapshots": [m["memory"] for m in metrics],
        }
    summary["original_over_compact_median_latency_ratio"] = (
        summary["original"]["median_mean_example_ms"] / summary["compact"]["median_mean_example_ms"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
