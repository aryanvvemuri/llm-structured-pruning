"""Supplement frozen campaign results with equal-output-length generation timings."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from prune_lab import campaign
from prune_lab.cli import experiment_lock
from prune_lab.data import prompt_ids
from prune_lab.evaluation import memory_snapshot, synchronize
from prune_lab.reporting import write_json


def measure_fixed(model, tokenizer, examples, new_tokens=16):
    """Time generation only, with tokenization excluded and a single warmup."""
    if not examples or type(new_tokens) is not int or new_tokens < 1:
        raise ValueError("Nonempty prompts and a positive fixed token count required")
    device = next(model.parameters()).device
    model.eval()
    inputs = [torch.tensor([prompt_ids(tokenizer, ex.prompt)], device=device) for ex in examples]
    masks = [torch.ones_like(ids) for ids in inputs]
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id

    def generate(ids, mask):
        result = model.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                                min_new_tokens=new_tokens, max_new_tokens=new_tokens,
                                pad_token_id=pad, use_cache=True)
        actual = result.shape[1] - ids.shape[1]
        if actual != new_tokens:
            raise AssertionError(f"Expected {new_tokens} new tokens, got {actual}")

    rows = []
    before = memory_snapshot(device)
    with torch.inference_mode():
        generate(inputs[0], masks[0])
        synchronize(device)
        for ex, ids, mask in zip(examples, inputs, masks):
            synchronize(device)
            start = time.perf_counter()
            generate(ids, mask)
            synchronize(device)
            seconds = time.perf_counter() - start
            rows.append({"key": ex.key, "prompt_tokens": ids.shape[1],
                         "generated_tokens": new_tokens, "seconds": seconds})
    latencies = [row["seconds"] for row in rows]
    throughputs = [new_tokens / seconds for seconds in latencies]
    return {"count": len(rows), "fixed_new_tokens": new_tokens, "warmup_examples": 1,
            "generated_tokens": len(rows) * new_tokens,
            "mean_example_ms": statistics.mean(latencies) * 1000,
            "median_example_ms": statistics.median(latencies) * 1000,
            "tokens_per_second": len(rows) * new_tokens / sum(latencies),
            "mean_example_tokens_per_second": statistics.mean(throughputs),
            "median_example_tokens_per_second": statistics.median(throughputs),
            "memory_before": before, "memory_after": memory_snapshot(device),
            "predictions": rows}


def benchmark(run, examples=64):
    if type(examples) is not int or examples < 1:
        raise ValueError("examples must be positive")
    run = Path(run).resolve()
    with experiment_lock():
        config = json.loads((run / "config.json").read_text())
        state, splits = campaign._initialize(config, run)
        if not state["test_started"] or not (run / "summary.json").is_file():
            raise ValueError("Complete campaign finalization before fixed-token benchmarking")
        selection = campaign._selection(run)
        finalized = json.loads((run / "summary.json").read_text())
        if finalized["selection"] != selection:
            raise ValueError("Finalized selection differs from frozen selection")
        targets = campaign.evaluation_targets(selection)
        for target in targets:
            if target["checkpoint"]:
                campaign._verify_checkpoint(target)
        prompts = splits["dev"][:examples]
        destination = run / "fixed-benchmark"
        records = {target["id"]: [] for target in targets}
        orders = []
        for repeat in range(3):
            ordered = targets if repeat % 2 == 0 else list(reversed(targets))
            orders.append([target["id"] for target in ordered])
            for target in ordered:
                print(f"Fixed 16-token benchmark {repeat + 1}/3: {target['id']}", flush=True)
                model = tokenizer = None
                try:
                    model, tokenizer = campaign._load(config, target["checkpoint"])
                    if target["checkpoint"] is None:
                        campaign._verify_source(config, model, selection["baseline"])
                    record = {"id": target["id"], "repeat": repeat + 1,
                              "metrics": measure_fixed(model, tokenizer, prompts),
                              **campaign._provenance(model)}
                    write_json(destination / f"{target['id']}-{repeat + 1}.json", record)
                    records[target["id"]].append(record)
                finally:
                    del model, tokenizer
                    campaign._release()
        summary = {"split": "dev", "examples": len(prompts), "repeats": 3,
                   "generation_config": {"min_new_tokens": 16, "max_new_tokens": 16,
                                         "do_sample": False, "use_cache": True, "batch_size": 1},
                   "config_sha256": state["config_sha256"], "dataset": state["dataset"],
                   "selection_sha256": campaign._fingerprint(selection),
                   "device": config["device"], "dtype": config["dtype"],
                   "cpu_threads": config["cpu_threads"], "run_orders": orders,
                   "models": {},
                   "note": "Equal generated token counts isolate output-length effects. Timings include prompt prefill and generation; tokenization and loading excluded. Memory snapshots are not peaks. No quality selection is performed."}
        for name, rows in records.items():
            metrics = [row["metrics"] for row in rows]
            summary["models"][name] = {
                "mean_example_ms_per_run": [m["mean_example_ms"] for m in metrics],
                "median_example_ms_per_run": [m["median_example_ms"] for m in metrics],
                "tokens_per_second_per_run": [m["tokens_per_second"] for m in metrics],
                "median_mean_example_ms": statistics.median(m["mean_example_ms"] for m in metrics),
                "median_tokens_per_second": statistics.median(m["tokens_per_second"] for m in metrics),
                "generated_tokens_per_run": [m["generated_tokens"] for m in metrics],
                "memory_snapshots": [m["memory_after"] for m in metrics]}
        write_json(destination / "summary.json", summary)
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--examples", type=int, default=64)
    args = parser.parse_args()
    benchmark(args.run, args.examples)
