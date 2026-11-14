#!/usr/bin/env python3
"""Timing/memory preflight for a campaign config on MPS, without any dev or test score.

Trains LoRA recovery for a few updates on recovery examples only (the dev callback is a dummy),
then times batched reply generation on recovery prompts. Reports seconds per update, seconds per
reply, peak MPS driver memory, and a run-time estimate with the improvement-5 seed rule
(five seeds if the whole run is estimated at <= 10 hours, otherwise three). Writes no run directory.

  HF_HUB_OFFLINE=1 .venv/bin/python scripts/preflight_model.py --config configs/support-v2-1p5b.json \
      --output /tmp/preflight-1p5b.json
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True, help='JSON report path (outside runs/)')
    parser.add_argument('--updates', type=int, default=30)
    parser.add_argument('--replies', type=int, default=48)
    parser.add_argument('--prune-uniform', type=float, default=None,
                        help='Also time a uniformly pruned copy at this fraction of all parameters (first-k neurons kept; timing only)')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    if any(part == 'runs' for part in Path(args.output).resolve().parts):
        parser.error('Write the preflight report outside runs/')
    import torch
    from prune_lab import campaign
    from prune_lab.allocation import allocate_neurons
    from prune_lab.data import make_batches
    from prune_lab.modeling import count_parameters, mlp_layers
    from prune_lab.pruning import compact_model
    from prune_lab.recovery import recover_model
    from prune_lab.support_data import read_support_jsonl
    from prune_lab.support_evaluation import evaluate_support

    config = campaign.read_campaign_config(args.config)
    if args.device:
        config['device'] = args.device
    campaign.validate_config(config)
    recovery_rows = read_support_jsonl(Path(config['support_data_dir']) / 'recovery.jsonl')
    replies = recovery_rows[-args.replies:]  # Recovery prompts only: no dev/test example is touched.

    def measure(prune):
        model, tokenizer = campaign._load(config)
        try:
            total = count_parameters(model)
            if prune:
                widths = [m.down_proj.in_features for m in mlp_layers(model)]
                scores = {i: torch.arange(w, 0, -1).float() for i, w in enumerate(widths)}
                keep = allocate_neurons(model, scores, {i: 1.0 for i in range(len(widths))}, prune,
                                        config['group_size'], config['min_keep_ratio'], uniform=True)
                compact_model(model, keep)
            parameters = count_parameters(model)
            batches = make_batches(recovery_rows, tokenizer, config['max_length'])
            recipe = dict(config['recovery'], max_steps=args.updates, checkpoints=[args.updates])
            if 'warmup_steps' in recipe:
                recipe['warmup_steps'] = min(recipe['warmup_steps'], args.updates - 1)
            start = time.perf_counter()
            result = recover_model(model, batches, lambda m: {'selection_score': 0.0, 'score_name': 'token_f1',
                                                              'answer_token_loss': 1.0}, recipe)
            wall = time.perf_counter() - start
            start = time.perf_counter()
            metrics = evaluate_support(model, tokenizer, replies, config['max_length'], config['max_new_tokens'],
                                       config.get('eval_batch_size', 1))
            evaluation = time.perf_counter() - start
            driver = torch.mps.driver_allocated_memory() if torch.backends.mps.is_available() else None
            return {'parameters': parameters, 'original_parameters': total, 'updates': args.updates,
                    'seconds_per_update': result['training_seconds'] / args.updates, 'training_wall_seconds': wall,
                    'peak_mps_driver_bytes_training': result['peak_mps_driver_bytes'],
                    'mps_driver_bytes_after_generation': driver,
                    'replies': len(replies), 'seconds_per_reply_generation': metrics['generation_seconds'] / len(replies),
                    'seconds_per_reply_including_loss_pass': evaluation / len(replies),
                    'generated_tokens': metrics['generated_tokens'], 'device': str(next(model.parameters()).device)}
        finally:
            del model, tokenizer
            campaign._release()

    report = {'config': args.config, 'model': config['model'], 'unpruned': measure(None)}
    if args.prune_uniform:
        report[f'uniform-{args.prune_uniform}'] = measure(args.prune_uniform)
    unpruned = report['unpruned']
    # Per candidate: recovery updates plus evaluations (assessments at 0 and each checkpoint on the
    # selection subset, merge check on the subset, full-dev reload), as in campaign.fit.
    steps = config['recovery']['max_steps']
    checks = len(set(config['recovery']['checkpoints']) | {steps}) + 1
    subset = config.get('selection_dev_examples') or config['counts']['dev']
    replies_per_candidate = checks * subset + (subset if config.get('skip_full_merged_dev') else config['counts']['dev']) + config['counts']['dev']
    candidates = len(campaign.candidate_specs(config))
    candidate_hours = (steps * unpruned['seconds_per_update']
                       + replies_per_candidate * unpruned['seconds_per_reply_including_loss_pass'] + 120) / 3600
    baseline_hours = (config['counts']['dev'] * unpruned['seconds_per_reply_including_loss_pass'] + 600) / 3600
    seed_hours = candidates * candidate_hours
    five = baseline_hours + 5 * seed_hours
    report['estimate'] = {'candidates_per_seed': candidates, 'replies_per_candidate': replies_per_candidate,
                          'hours_per_candidate_upper': candidate_hours, 'hours_per_seed': seed_hours,
                          'hours_five_seeds': five, 'hours_three_seeds': baseline_hours + 3 * seed_hours,
                          'seed_rule': 'five seeds if the five-seed estimate is at most 10 hours, otherwise three',
                          'seeds': 5 if five <= 10 else 3,
                          'note': 'Uses unpruned timings for every candidate (pruned ones are faster); +120 s per candidate '
                                  'for load/export/reload and +600 s once for the baseline, Taylor scoring and probes.'}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
