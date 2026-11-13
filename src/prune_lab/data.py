"""Disjoint datasets and strictly answer-only supervised batches."""
from dataclasses import dataclass, asdict
import hashlib
import json
import random
import re
from pathlib import Path


@dataclass(frozen=True)
class Example:
    prompt: str
    answer: str
    key: str


def canonical_prompt(text):
    return re.sub(r"\s+", " ", text.strip()).casefold()


def synthetic_splits(calibration=32, dev=48, test=64, seed=17):
    """Sample without replacement from one operation/operand universe."""
    counts = (calibration, dev, test)
    if any(n <= 0 for n in counts):
        raise ValueError("Every split must contain at least one example")
    pool = [(op, a, b) for op in ("+", "-", "*") for a in range(-30, 31) for b in range(-30, 31)]
    rng = random.Random(seed)
    rng.shuffle(pool)
    if sum(counts) > len(pool):
        raise ValueError("Requested more unique arithmetic examples than available")
    examples = []
    for op, a, b in pool[:sum(counts)]:
        answer = a + b if op == "+" else a - b if op == "-" else a * b
        examples.append(Example(f"Calculate ({a}) {op} ({b}). Respond with only the final numeric answer.", str(answer), f"{op}:{a}:{b}"))
    i, j = calibration, calibration + dev
    return {"calibration": examples[:i], "dev": examples[i:j], "test": examples[j:]}


def jsonl_examples(path):
    result = []
    with open(path) as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt, answer = str(row["prompt"]).strip(), str(row["answer"]).strip()
            if not prompt or not answer:
                raise ValueError(f"{path}:{line_number}: empty prompt or answer")
            result.append(Example(prompt, answer, str(row.get("key", canonical_prompt(prompt)))))
    if not result:
        raise ValueError(f"Empty dataset: {path}")
    return result


def validate_splits(splits):
    seen_keys, seen_prompts = {}, {}
    for split, examples in splits.items():
        if not examples:
            raise ValueError(f"Empty {split} split")
        for ex in examples:
            key, prompt = ex.key, canonical_prompt(ex.prompt)
            if key in seen_keys or prompt in seen_prompts:
                raise ValueError(f"Duplicate/leaked example in {split}: {key}")
            seen_keys[key] = split
            seen_prompts[prompt] = split
    return splits


def load_splits(config):
    source = config.get("dataset", "synthetic")
    counts = config.get("counts", {"calibration": 8, "dev": 12, "test": 16})
    if source == "synthetic":
        splits = synthetic_splits(**counts, seed=config.get("seed", 17))
    elif source == "jsonl":
        splits = {split: jsonl_examples(path) for split, path in config["data_paths"].items()}
        if set(splits) != {"calibration", "dev", "test"}:
            raise ValueError("data_paths must specify calibration, dev, test")
    elif source == "gsm8k":
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", cache_dir=config.get("cache_dir"))
        # Official test is never used to choose calibration or development data.
        train = list(ds["train"])
        random.Random(config.get("seed", 17)).shuffle(train)
        c, d, t = (counts[k] for k in ("calibration", "dev", "test"))
        if min(c, d, t) <= 0 or c+d > len(train) or t > len(ds["test"]):
            raise ValueError("Invalid GSM8K split sizes")
        def convert(rows):
            return [Example(r["question"], r["answer"], canonical_prompt(r["question"])) for r in rows]
        splits = {"calibration": convert(train[:c]), "dev": convert(train[c:c+d]), "test": convert(list(ds["test"])[:t])}
    else:
        raise ValueError(f"Unknown dataset: {source}")
    return validate_splits(splits)


def dataset_manifest(splits):
    return {name: {"count": len(rows), "sha256": hashlib.sha256(json.dumps([asdict(e) for e in rows], sort_keys=True).encode()).hexdigest()} for name, rows in splits.items()}


def prompt_ids(tokenizer, prompt):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
    return tokenizer.encode(prompt + "\nAnswer:", add_special_tokens=True)


def make_batches(examples, tokenizer, max_length=256, batch_size=1):
    """Reject overlength examples explicitly; never silently erase answer labels."""
    import torch
    if max_length < 2 or batch_size < 1:
        raise ValueError("max_length >= 2 and batch_size >= 1 required")
    encoded = []
    for ex in examples:
        prompt = prompt_ids(tokenizer, ex.prompt)
        answer = tokenizer.encode(ex.answer, add_special_tokens=False)
        if not prompt or not answer:
            raise ValueError(f"No prompt or answer tokens: {ex.key}")
        if tokenizer.eos_token_id is not None:
            answer = answer + [tokenizer.eos_token_id]
        if len(prompt) + len(answer) > max_length:
            raise ValueError(f"Example {ex.key} requires {len(prompt)+len(answer)} tokens > max_length={max_length}; increase max_length or shorten data (no truncation performed)")
        encoded.append((prompt + answer, [-100]*len(prompt) + answer))
    if not encoded:
        raise ValueError("No supervised examples")
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError("Tokenizer needs a pad or EOS token")
    batches = []
    for start in range(0, len(encoded), batch_size):
        rows = encoded[start:start+batch_size]
        width = max(len(ids) for ids, _ in rows)
        batches.append({"input_ids": torch.tensor([ids + [pad]*(width-len(ids)) for ids, _ in rows]), "attention_mask": torch.tensor([[1]*len(ids)+[0]*(width-len(ids)) for ids, _ in rows]), "labels": torch.tensor([labels+[-100]*(width-len(labels)) for ids, labels in rows])})
    return batches


def semantic_problem_key(operation, left, right):
    """Commutative arithmetic and all surface templates share one problem key."""
    if operation not in ('+', '-', '*'):
        raise ValueError('Unsupported arithmetic operation')
    left, right = int(left), int(right)
    if operation in ('+', '*'):
        left, right = sorted((left, right))
    return f'{operation}:{left}:{right}'


def _campaign_key(example):
    key_match = re.fullmatch(r'([+*\-]):(-?\d+):(-?\d+)', example.key)
    if key_match:
        return semantic_problem_key(*key_match.groups())
    expression = re.search(r'\((-?\d+)\)\s*([+*\-])\s*\((-?\d+)\)', example.prompt)
    if expression:
        left, operation, right = expression.groups()
        return semantic_problem_key(operation, left, right)
    raise ValueError(f'Arithmetic campaign requires an explicit semantic key: {example.key}')


def _campaign_example(operation, left, right, template):
    answer = left + right if operation == '+' else left - right if operation == '-' else left * right
    if template == 0:
        text = f'Calculate ({left}) {operation} ({right}).'
    elif operation == '+':
        text = f'A game score is {left} points. A signed change of {right} points is added. What is the new score?'
    elif operation == '-':
        text = f'A temperature reading is {left} degrees. Subtract a signed correction of {right} degrees. What is the corrected reading?'
    else:
        text = f'A machine multiplies an input of {left} by a signed factor of {right}. What is its output?'
    return Example(text + ' Respond with only the final numeric answer.', str(answer),
                   semantic_problem_key(operation, left, right))


def load_campaign_splits(config):
    """Generate four semantic-disjoint arithmetic splits with prior-run exclusions.

    Balanced across operation, operand signs and magnitude (<=20 or <=99).
    Alternate plain expressions and short signed-arithmetic word problems.
    ``exclusion_jsonl`` supplements automatic ``runs/adaptive/*.jsonl`` exclusions.
    ``exclude_legacy_seeds`` regenerates old synthetic data with
    ``exclude_legacy_counts`` (default 32/48/64). No prior rows are reused.
    Explicit JSONL input requires arithmetic semantic keys for word problems.
    """
    if config.get('dataset') == 'doc2dial':
        from .support_data import load_support_splits
        return load_support_splits(config)
    names = ('recovery', 'calibration', 'dev', 'test')
    counts = {'recovery': 1000, 'calibration': 150, 'dev': 200, 'test': 300}
    counts.update(config.get('counts', {}))
    if set(counts) != set(names) or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError('Campaign counts require four positive integers: recovery/calibration/dev/test')
    excluded_keys, excluded_prompts = set(), set()
    exclusion_paths = set(Path('runs/adaptive').glob('*.jsonl'))
    exclusion_paths.update(Path(p) for p in config.get('exclusion_jsonl', []))
    for path in sorted(exclusion_paths):
        for example in jsonl_examples(path):
            excluded_prompts.add(canonical_prompt(example.prompt))
            # Non-arithmetic prior prompts are still excluded verbatim.
            try:
                excluded_keys.add(_campaign_key(example))
            except ValueError:
                pass
    legacy_counts = config.get('exclude_legacy_counts', {'calibration': 32, 'dev': 48, 'test': 64})
    for seed in config.get('exclude_legacy_seeds', [17]):
        for rows in synthetic_splits(**legacy_counts, seed=seed).values():
            excluded_keys.update(_campaign_key(ex) for ex in rows)
    if config.get('dataset', 'synthetic') == 'jsonl':
        paths = config['data_paths']
        if set(paths) != set(names):
            raise ValueError('Campaign JSONL requires recovery, calibration, dev and test paths')
        splits = {name: [Example(ex.prompt, ex.answer, _campaign_key(ex))
                         for ex in jsonl_examples(paths[name])] for name in names}
        for name, rows in splits.items():
            if len(rows) != counts[name]:
                raise ValueError(f'{name} JSONL count differs from requested count')
            for ex in rows:
                if ex.key in excluded_keys or canonical_prompt(ex.prompt) in excluded_prompts:
                    raise ValueError(f'Previously exposed campaign example: {ex.key}')
                op, a, b = ex.key.split(':')
                expected = _campaign_example(op, int(a), int(b), 0).answer
                if ex.answer != expected:
                    raise ValueError(f'Incorrect supervised arithmetic answer: {ex.key}')
        return validate_splits(splits)
    if config.get('dataset', 'synthetic') != 'synthetic':
        raise ValueError('Campaign supports synthetic arithmetic or keyed arithmetic JSONL')
    rng = random.Random(config.get('seed', 20261113))
    buckets = {}
    for operation in ('+', '-', '*'):
        for left in range(-99, 100):
            for right in range(-99, 100):
                if operation in ('+', '*') and left > right:
                    continue
                key = semantic_problem_key(operation, left, right)
                if key in excluded_keys:
                    continue
                sign = 'negative' if left < 0 and right < 0 else 'nonnegative' if left >= 0 and right >= 0 else 'mixed'
                difficulty = 'small' if max(abs(left), abs(right)) <= 20 else 'large'
                buckets.setdefault((operation, sign, difficulty), []).append((operation, left, right))
    for rows in buckets.values():
        rng.shuffle(rows)
    splits = {}
    for name in names:
        rows = []
        bucket_names = sorted(buckets)
        rng.shuffle(bucket_names)
        while len(rows) < counts[name]:
            progress = False
            for bucket in bucket_names:
                while buckets[bucket]:
                    operation, left, right = buckets[bucket].pop()
                    example = _campaign_example(operation, left, right, len(rows) % 2)
                    if canonical_prompt(example.prompt) in excluded_prompts:
                        continue
                    rows.append(example)
                    progress = True
                    break
                if len(rows) == counts[name]:
                    break
            if not progress:
                raise ValueError('Requested more unseen semantic arithmetic problems than available')
        rng.shuffle(rows)
        splits[name] = rows
    return validate_splits(splits)
