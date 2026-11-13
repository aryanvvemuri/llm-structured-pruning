"""Command-line orchestration. All expensive commands explicitly load one model."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

from .data import load_splits, make_batches, dataset_manifest
from .evaluation import evaluate_model
from .reporting import write_json, render_report

DEFAULT = {"model": "Qwen/Qwen2.5-0.5B-Instruct", "device": "auto", "dtype": "float32", "cache_dir": ".cache/huggingface/hub", "dataset": "synthetic", "seed": 17, "cpu_threads": 4, "counts": {"calibration": 4, "dev": 4, "test": 4}, "max_length": 256, "max_new_tokens": 32, "batch_size": 1, "method": "taylor", "search": {"initial_step": 0.05, "min_step": 8, "max_rounds": 2, "max_drop_pp": 0.0, "loss_tolerance": None, "min_keep_ratio": 0.8}}


@contextmanager
def experiment_lock():
    """OS releases flock even on failure; one experiment per project."""
    import fcntl
    Path("runs").mkdir(exist_ok=True)
    with open("runs/.experiment.lock", "w") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another experiment is running in this project")
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def read_config(path):
    config = dict(DEFAULT)
    if path:
        config.update(json.loads(Path(path).read_text()))
    return config


def prepare(config):
    import torch
    from .modeling import load_model
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    model, tokenizer = load_model(config["model"], device=config["device"], dtype=config["dtype"], cache_dir=config.get("cache_dir"))
    return model, tokenizer


def run(args):
    if args.command == "campaign":
        from .campaign import read_campaign_config, run_campaign
        config = read_campaign_config(args.config)
        if args.model:
            config["model"] = args.model
        if args.device:
            config["device"] = args.device
        run_campaign(config, args.output, args.stage)
        return
    if args.command == "report":
        result = json.loads(Path(args.result).read_text())
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(render_report(result))
        print(args.output)
        return
    config = read_config(args.config)
    if args.model:
        config["model"] = args.model
    if args.device:
        config["device"] = args.device
    if args.command == "doctor":
        import torch
        import transformers
        from .modeling import resolve_device
        print(json.dumps({"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "transformers": transformers.__version__, "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(), "resolved_device": resolve_device(config["device"]), "config": config}, indent=2))
        return
    with experiment_lock():
        model, tokenizer = prepare(config)
        from .modeling import count_parameters
        import torch
        import transformers
        result = {"config": config, "parameters": count_parameters(model), "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()), "resolved_device": str(next(model.parameters()).device), "model_revision": getattr(model.config, "_commit_hash", None), "versions": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__}}
        if args.command == "export":
            from .pruning import compact_model
            from .artifacts import save_checkpoint
            previous = json.loads(Path(args.result).read_text())
            if previous["config"]["model"] != config["model"]:
                raise ValueError("Export model must match search model; pass original search config")
            keep = {int(k): v for k, v in previous["search"]["keep_indices"].items()}
            compact_model(model, keep)
            manifest = {"config": config, "keep_indices": keep, "source_result": str(Path(args.result).resolve()), "dataset": previous.get("dataset"), "parameters_before": result["parameters"], "parameters_after": count_parameters(model), "parameter_bytes_before": result["parameter_bytes"], "parameter_bytes_after": sum(p.numel() * p.element_size() for p in model.parameters()), "model_revision": result["model_revision"]}
            save_checkpoint(model, tokenizer, args.output, manifest=manifest)
            print(args.output)
            return
        splits = load_splits(config)
        result["dataset"] = dataset_manifest(splits)
        def evaluate(rows):
            return evaluate_model(model, tokenizer, rows, max_length=config["max_length"], max_new_tokens=config["max_new_tokens"])
        if args.command == "evaluate":
            result["split"] = args.split
            result["metrics"] = evaluate(splits[args.split])
        elif args.command == "score":
            import torch
            from .scoring import score_neurons
            batches = make_batches(splits["calibration"], tokenizer, config["max_length"], config["batch_size"])
            scores = score_neurons(model, batches, method=args.method or config["method"])
            result["method"] = args.method or config["method"]
            result["scores"] = {int(k): v.tolist() for k, v in scores.items()}
        elif args.command == "search":
            from .search import search_masks
            from .pruning import masked_neurons
            batches = make_batches(splits["calibration"], tokenizer, config["max_length"], config["batch_size"])
            search_start = time.perf_counter()
            result["search"] = search_masks(model, batches, lambda: evaluate(splits["dev"]), method=args.method or config["method"], seed=config["seed"], **config["search"])
            result["search_seconds"] = time.perf_counter() - search_start
            # Only after selection is frozen can held-out test affect this result.
            with masked_neurons(model, result["search"]["keep_indices"]):
                result["final_test"] = evaluate(splits["test"])
        write_json(args.output, result)
        print(args.output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    campaign = sub.add_parser("campaign", help="Per-layer pruning, recovery and frozen accuracy-tier selection")
    campaign.add_argument("--config")
    campaign.add_argument("--model")
    campaign.add_argument("--device", choices=["auto", "cpu", "mps"])
    campaign.add_argument("--output", default="runs/campaign-v2")
    campaign.add_argument("--stage", choices=["preflight", "fit", "finalize", "benchmark", "all"], default="all")
    for command in ("doctor", "evaluate", "score", "search", "export", "report"):
        p = sub.add_parser(command)
        if command != "report":
            p.add_argument("--config")
            p.add_argument("--model")
            p.add_argument("--device", choices=["auto", "cpu", "mps"])
        if command in ("score", "search"):
            p.add_argument("--method", choices=["activation", "contribution", "taylor"] + (["random"] if command == "search" else []))
        if command == "evaluate":
            p.add_argument("--split", choices=["calibration", "dev", "test"], default="dev")
        if command in ("export", "report"):
            p.add_argument("--result", required=True)
        if command != "doctor":
            p.add_argument("--output", default={"export": "runs/compact", "report": "runs/report.md"}.get(command, f"runs/{command}.json"))
    args = parser.parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(f"prune-lab failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
