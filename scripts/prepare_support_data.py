#!/usr/bin/env python3
"""Prepare local Doc2Dial data; cached tokenizer only, no model or downloads.

Defaults rebuild datasets/doc2dial_v1 byte-for-byte. The v2 benchmark is built with:
  --output datasets/doc2dial_v2 --recovery 1800 --calibration 150 --dev 600 --test 224
  --exclude-dialogues-from datasets/doc2dial_v1/test.jsonl --family-cap dev=15
  --domain-quotas calibration=balanced --domain-quotas recovery=balanced
  --family-order dev=random --family-order calibration=random
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from prune_lab.support_data import prepare_support_dataset, DEFAULT_COUNTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', default='.cache/datasets/doc2dial/doc2dial_v1.0.1.zip')
    parser.add_argument('--output', default='datasets/doc2dial_v1')
    parser.add_argument('--tokenizer', default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--cache-dir', default='.cache/huggingface/hub')
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--max-answer-tokens', type=int, default=64)
    parser.add_argument('--max-history-turns', type=int, default=4)
    parser.add_argument('--max-per-dialogue', type=int, choices=(1, 2), default=1)
    for name, count in DEFAULT_COUNTS.items():
        parser.add_argument('--' + name, type=int, default=count)
    parser.add_argument('--family-cap', action='append', default=[], metavar='SPLIT=N',
                        help='Override the per-family cap max(10, ceil(count/20)) for one split')
    parser.add_argument('--domain-quotas', action='append', default=[], metavar='SPLIT=equal|balanced')
    parser.add_argument('--family-order', action='append', default=[], metavar='SPLIT=size|random')
    parser.add_argument('--exclude-dialogues-from', action='append', default=[], metavar='JSONL',
                        help='Drop every dialogue_id listed in this prepared JSONL (only that field is read)')
    args = parser.parse_args()

    def mapping(values, convert=str):
        result = {}
        for item in values:
            name, sep, value = item.partition('=')
            if not sep or name not in DEFAULT_COUNTS or name in result:
                parser.error(f'Expected one SPLIT=VALUE per split, got {item!r}')
            result[name] = convert(value)
        return result or None
    excluded, sources = None, []
    for path in args.exclude_dialogues_from:
        with open(path) as stream:
            ids = {json.loads(line)['dialogue_id'] for line in stream if line.strip()}
        excluded = (excluded or set()) | ids
        sources.append({'path': path, 'dialogues': len(ids)})
    start = time.perf_counter()
    from transformers import AutoTokenizer
    # Resolve a local snapshot first: transformers 4.57's Mistral-regex probe
    # can otherwise request model metadata even with local_files_only=True.
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.is_dir():
        cache = Path(args.cache_dir) / ('models--' + args.tokenizer.replace('/', '--'))
        revision = (cache / 'refs' / 'main').read_text().strip()
        tokenizer_path = cache / 'snapshots' / revision
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), cache_dir=args.cache_dir,
                                             local_files_only=True, trust_remote_code=False,
                                             fix_mistral_regex=False)
    manifest = prepare_support_dataset(args.archive, args.output, tokenizer,
        {name: getattr(args, name) for name in DEFAULT_COUNTS}, seed=args.seed,
        max_length=args.max_length, max_answer_tokens=args.max_answer_tokens,
        max_history_turns=args.max_history_turns, max_per_dialogue=args.max_per_dialogue,
        family_caps=mapping(args.family_cap, int), domain_quotas=mapping(args.domain_quotas),
        family_order=mapping(args.family_order), exclude_dialogue_ids=excluded,
        exclusion_source=sources or None)
    keys = ('split_counts', 'split_dialogue_counts', 'split_document_family_counts', 'split_domains', 'filters',
            'cross_split_collision_drops', 'group_overlaps', 'near_match_audit', 'dialogue_exclusion')
    print(json.dumps({k: manifest[k] for k in keys if k in manifest}, indent=2))
    print(f'Prepared in {time.perf_counter() - start:.1f} s', file=sys.stderr)


if __name__ == '__main__':
    main()
