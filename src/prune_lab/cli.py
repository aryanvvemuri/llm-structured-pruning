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
    if args.command == "multi-seed":
        run_multi_seed(args)
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


def run_multi_seed(args):
    from . import multiseed
    if args.action == "run":
        from .campaign import read_campaign_config
        config = read_campaign_config(args.config)
        if args.model:
            config["model"] = args.model
        if args.device:
            config["device"] = args.device
        if args.num_seeds < 1:
            raise ValueError("--num-seeds must be positive")
        seeds = args.seeds or [config["recovery"]["seed"] + i for i in range(args.num_seeds)]
        multiseed.run_seeds(config, args.output, seeds, open_test=args.open_test,
                            discard_checkpoints=args.discard_checkpoints)
        summary = multiseed.aggregate_run(args.output, "dev", args.samples, args.bootstrap_seed)
        print(f"Multi-seed dev summary: {Path(args.output) / 'multi-seed-report.md'} ({len(summary['seeds'])} seeds)")
    elif args.action == "aggregate":
        multiseed.aggregate_run(args.run, args.split, args.samples, args.bootstrap_seed)
        print(Path(args.run) / ("multi-seed-report.md" if args.split == "dev" else "multi-seed-report-test.md"))
    else:
        pairs = None
        if args.pair:
            pairs = [tuple(item.split(":", 1)) for item in args.pair]
            if any(len(p) != 2 or not all(p) for p in pairs):
                raise ValueError("--pair expects BASELINE_VARIANT:CANDIDATE_VARIANT")
        paired = True if args.paired_seeds else False if args.unpaired_seeds else None
        result = multiseed.compare_runs(args.baseline, args.candidate, pairs, args.split, paired,
                                        args.samples, args.bootstrap_seed, args.allow_different_schedules)
        output = Path(args.output or Path(args.candidate) / f"compare-vs-{Path(args.baseline).resolve().name}.json")
        write_json(output, result)
        output.with_suffix(".md").write_text(multiseed.render_comparison(result))
        print(output.with_suffix(".md"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    campaign = sub.add_parser("campaign", help="Per-layer pruning, recovery and frozen accuracy-tier selection")
    campaign.add_argument("--config")
    campaign.add_argument("--model")
    campaign.add_argument("--device", choices=["auto", "cpu", "mps"])
    campaign.add_argument("--output", default="runs/campaign-v2")
    campaign.add_argument("--stage", choices=["preflight", "fit", "finalize", "benchmark", "all"], default="all")
    multi = sub.add_parser("multi-seed", help="Repeat recovery over seeds (dev only by default) and aggregate with a seed x family bootstrap")
    actions = multi.add_subparsers(dest="action", required=True)
    seeds_run = actions.add_parser("run", help="Run preflight+fit for every recovery seed into OUTPUT/seed-<s>/")
    seeds_run.add_argument("--config", required=True)
    seeds_run.add_argument("--output", required=True)
    seeds_run.add_argument("--model")
    seeds_run.add_argument("--device", choices=["auto", "cpu", "mps"])
    chosen = seeds_run.add_mutually_exclusive_group()
    chosen.add_argument("--seeds", type=int, nargs="+", help="Explicit recovery seeds")
    chosen.add_argument("--num-seeds", type=int, default=3, help="Use config recovery.seed + 0..N-1")
    seeds_run.add_argument("--open-test", action="store_true", help="Also run finalize (final TEST evaluation) per seed")
    seeds_run.add_argument("--discard-checkpoints", action="store_true",
                           help="Delete each seed's exported *.safetensors after its evaluations are saved")
    seeds_aggregate = actions.add_parser("aggregate", help="Write multi-seed-summary.json and multi-seed-report.md")
    seeds_aggregate.add_argument("--run", required=True)
    seeds_aggregate.add_argument("--split", choices=["dev", "test"], default="dev")
    seeds_compare = actions.add_parser("compare", help="Paired difference of variants across two runs (b minus a)")
    seeds_compare.add_argument("--baseline", required=True, help="Run directory a, e.g. main")
    seeds_compare.add_argument("--candidate", required=True, help="Run directory b, e.g. the new method")
    seeds_compare.add_argument("--pair", action="append", metavar="A_VARIANT:B_VARIANT",
                               help="Variants to compare (default: every variant present in both)")
    seeds_compare.add_argument("--split", choices=["dev", "test"], default="dev")
    pairing = seeds_compare.add_mutually_exclusive_group()
    pairing.add_argument("--paired-seeds", action="store_true",
                         help="Force pairing by seed value (default: paired whenever both seed sets are identical)")
    pairing.add_argument("--unpaired-seeds", action="store_true", help="Force independent seed resampling per run")
    seeds_compare.add_argument("--allow-different-schedules", action="store_true",
                               help="Permit dev-full decisions between runs with different recovery checkpoints/max_steps")
    seeds_compare.add_argument("--output")
    for p in (seeds_run, seeds_aggregate, seeds_compare):
        p.add_argument("--samples", type=int, default=4000, help="Bootstrap resamples")
        p.add_argument("--bootstrap-seed", type=int, default=17)
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
