"""Lexical reply metrics and answer-only CE for grounded customer support.

Token F1 and ROUGE-L are lexical proxies, not measures of factual grounding,
policy correctness, semantic equivalence, or human-rated reply quality.
"""
from collections import Counter
import math
import time
import unicodedata

from .data import make_batches, prompt_ids
from .evaluation import memory_snapshot, synchronize


def normalize_reply(text):
    """Casefold, replace Unicode punctuation with spaces, remove a/an/the,
    and collapse whitespace. Numbers and answer markers receive no special
    interpretation. This English normalization is used by all three metrics.
    """
    text = ''.join(' ' if unicodedata.category(c).startswith('P') else c
                   for c in text.casefold())
    return ' '.join(token for token in text.split() if token not in {'a', 'an', 'the'})


def reply_metrics(prediction, reference):
    """Multiset token F1, normalized-token LCS F1, and exact-match diagnostic.

    Empty predictions score zero, including when normalization erases them.
    References must contain at least one normalized token.
    """
    predicted = normalize_reply(prediction).split()
    expected = normalize_reply(reference).split()
    if not expected:
        raise ValueError('Support reference must contain nonempty normalized text')
    if not predicted:
        return {'token_f1': 0.0, 'rouge_l': 0.0, 'exact_match': 0.0}
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    # Two-row LCS uses O(reference tokens) memory, without extra dependencies.
    previous = [0] * (len(expected) + 1)
    for token in predicted:
        current = [0]
        for j, other in enumerate(expected, 1):
            current.append(previous[j - 1] + 1 if token == other
                           else max(previous[j], current[-1]))
        previous = current
    denominator = len(predicted) + len(expected)
    return {'token_f1': 2 * overlap / denominator,
            'rouge_l': 2 * previous[-1] / denominator,
            'exact_match': float(predicted == expected)}


def evaluate_support(model, tokenizer, examples, max_length=512, max_new_tokens=64):
    """Evaluate batch-one CE and greedy replies after one unmeasured warmup.

    Like evaluate_model, leaves the model in eval mode. Loss is weighted by
    supervised shifted answer tokens (including EOS); prompt labels are ignored.
    """
    import torch
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError('max_new_tokens must be a positive integer')
    examples = list(examples)
    if not examples:
        raise ValueError('No support evaluation examples')
    for ex in examples:
        if not normalize_reply(ex.answer):
            raise ValueError(f'Empty normalized support reference: {ex.key}')
    batches = make_batches(examples, tokenizer, max_length=max_length, batch_size=1)
    device = next(model.parameters()).device
    model.eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    loss_sum, supervised = 0.0, 0
    rows, elapsed, tokens = [], 0.0, 0
    with torch.inference_mode():
        for batch in batches:
            batch = {k: v.to(device) for k, v in batch.items()}
            n = int((batch['labels'][:, 1:] != -100).sum().item())
            if not n:
                raise ValueError('Support evaluation requires shifted supervised tokens')
            loss = float(model(**batch, use_cache=False).loss.item())
            if not math.isfinite(loss) or loss < 0:
                raise ValueError('Nonfinite or negative support answer-token loss')
            loss_sum += loss * n
            supervised += n

        def generate(ex):
            ids = torch.tensor([prompt_ids(tokenizer, ex.prompt)], device=device, dtype=torch.long)
            result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    do_sample=False, max_new_tokens=max_new_tokens,
                                    pad_token_id=pad_id, use_cache=True)
            return result[0, ids.shape[1]:]

        generate(examples[0])
        synchronize(device)
        for ex in examples:
            synchronize(device)
            start = time.perf_counter()
            generated = generate(ex)
            synchronize(device)
            seconds = time.perf_counter() - start
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError('Invalid generation duration')
            prediction = tokenizer.decode(generated, skip_special_tokens=True)
            metrics = reply_metrics(prediction, ex.answer)
            row = {'key': ex.key, 'prediction': prediction, 'expected': ex.answer,
                   'score': metrics['token_f1'], **metrics, 'seconds': seconds,
                   'generated_tokens': len(generated)}
            for field in ('doc_id', 'dialogue_id', 'domain', 'group_id'):
                value = getattr(ex, field, None)
                if value is not None:
                    row[field] = value
            rows.append(row)
            elapsed += seconds
            tokens += len(generated)
            if len(examples) >= 100 and len(rows) % 50 == 0:
                print(f"Support evaluation: {len(rows)}/{len(examples)} replies", flush=True)
    aggregate = {name: sum(row[name] for row in rows) / len(rows)
                 for name in ('token_f1', 'rouge_l', 'exact_match')}
    return {'selection_score': aggregate['token_f1'], 'score_name': 'token_f1', **aggregate,
            'count': len(rows), 'answer_token_loss': loss_sum / supervised,
            'supervised_tokens': supervised, 'generation_seconds': elapsed,
            'generated_tokens': tokens, 'tokens_per_second': tokens / elapsed if elapsed else 0.0,
            'generation_limit_fraction': sum(row['generated_tokens'] >= max_new_tokens for row in rows) / len(rows),
            'generation_limit_note': 'Fraction reaching the token cap; not confirmed truncation because EOS may occur on the final token.',
            'mean_example_seconds': elapsed / len(rows), 'warmup_examples': 1,
            'small_sample_warning': len(rows) < 100, 'memory': memory_snapshot(device),
            'generation_config': {'do_sample': False, 'max_new_tokens': max_new_tokens, 'batch_size': 1},
            'metric_note': 'Lexical overlap proxies; not factual grounding, policy correctness, or human quality.',
            'normalization': 'Casefold; Unicode punctuation to spaces; remove English a/an/the; collapse whitespace.',
            'predictions': rows}
