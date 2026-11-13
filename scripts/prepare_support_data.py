#!/usr/bin/env python3
"""Prepare local Doc2Dial data; cached tokenizer only, no model or downloads."""
import argparse
import json
from pathlib import Path
import sys

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
    args = parser.parse_args()
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
        max_history_turns=args.max_history_turns, max_per_dialogue=args.max_per_dialogue)
    print(json.dumps({k: manifest[k] for k in ('split_counts', 'split_dialogue_counts', 'split_document_family_counts', 'filters', 'group_overlaps', 'near_match_audit')}, indent=2))


if __name__ == '__main__':
    main()
