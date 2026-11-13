"""Batched support generation must reproduce the batch-one path on a tiny CPU model."""
import random
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from prune_lab import campaign, support_evaluation
from prune_lab.support_evaluation import evaluate_support, generation_batches

WORDS = [f'w{i}' for i in range(17)]


@pytest.fixture(scope='module')
def setup():
    torch.set_num_threads(1)
    vocab = {'[PAD]': 0, '[UNK]': 1, '[EOS]': 2, **{w: i + 3 for i, w in enumerate(WORDS)}}
    backend = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]', eos_token='[EOS]')
    torch.manual_seed(4)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=20, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                                         pad_token_id=0, eos_token_id=2))
    with torch.no_grad():
        model.lm_head.weight[2] *= 3  # Makes EOS win at varied steps: mixed-length outputs.
    rng = random.Random(0)
    examples = [SimpleNamespace(key=f'k{i:02d}', prompt=' '.join(rng.choice(WORDS) for _ in range(rng.randint(2, 12))),
                                answer='w1 w2 w3', group_id=f'g{i % 3}', domain='d', doc_id='doc', dialogue_id=f'dl{i}')
                for i in range(11)]
    single = evaluate_support(model, tokenizer, examples, 128, 8)
    return model, tokenizer, examples, single


def strip(metrics):
    return [{k: v for k, v in row.items() if k != 'seconds'} for row in metrics['predictions']]


@pytest.mark.parametrize('batch_size', [2, 4, 5, 11, 16])
def test_batched_predictions_equal_single(setup, batch_size):
    model, tokenizer, examples, single = setup
    lengths = {row['generated_tokens'] for row in single['predictions']}
    assert len(lengths) > 2 and 8 in lengths and min(lengths) < 8  # precondition: EOS at mixed steps and cap hits
    batched = evaluate_support(model, tokenizer, examples, 128, 8, batch_size=batch_size)
    assert strip(batched) == strip(single)
    for key in ('token_f1', 'rouge_l', 'exact_match', 'answer_token_loss', 'generation_limit_fraction',
                'generated_tokens', 'count', 'supervised_tokens'):
        assert batched[key] == single[key], key
    assert set(batched) == set(single) and batched['generation_config']['batch_size'] == batch_size
    assert tokenizer.padding_side == 'right'  # Tokenizer state untouched.


def test_order_independence_and_deterministic_buckets(setup):
    model, tokenizer, examples, single = setup
    shuffled = list(examples)
    random.Random(3).shuffle(shuffled)
    batched = evaluate_support(model, tokenizer, shuffled, 128, 8, batch_size=4)
    assert [r['key'] for r in batched['predictions']] == [e.key for e in shuffled]
    by_key = {r['key']: r for r in strip(single)}
    assert all(r == by_key[r['key']] for r in strip(batched))
    keys = lambda rows: [[rows[i].key for i, _ in batch] for batch in generation_batches(rows, tokenizer, 4)]
    assert keys(examples) == keys(shuffled)


def test_tokens_after_eos_are_discarded(setup):
    model, tokenizer, examples, _ = setup
    batched = evaluate_support(model, tokenizer, examples, 128, 8, batch_size=11)
    for row in batched['predictions']:
        assert '[EOS]' not in row['prediction'] and '[PAD]' not in row['prediction']
        assert 1 <= row['generated_tokens'] <= 8


def test_default_path_untouched(monkeypatch, setup):
    model, tokenizer, examples, single = setup
    monkeypatch.setattr(support_evaluation, '_generate_batched', lambda *a, **k: pytest.fail('batched path used'))
    assert strip(evaluate_support(model, tokenizer, examples, 128, 8)) == strip(single)
    seen = []
    monkeypatch.setattr(support_evaluation, 'evaluate_support', lambda *a, **k: seen.append((a[5:], k)) or {})
    config = {'task': 'support', 'max_length': 128, 'max_new_tokens': 8}
    campaign._evaluate(model, tokenizer, examples, config)
    campaign._evaluate(model, tokenizer, examples, dict(config, eval_batch_size=8))
    assert seen == [((), {}), ((8,), {})]
    with pytest.raises(ValueError, match='batch_size'):
        evaluate_support(model, tokenizer, examples, 128, 8, batch_size=0)


def test_eval_batch_size_validation():
    config = campaign.read_campaign_config()
    before = campaign._fingerprint(config)
    campaign.validate_config(config)
    assert 'eval_batch_size' not in config and campaign._fingerprint(config) == before
    for bad in (0, 1.5, True):
        with pytest.raises(ValueError, match='eval_batch_size'):
            campaign.validate_config(dict(config, eval_batch_size=bad))
    with pytest.raises(ValueError, match='support task'):
        campaign.validate_config(dict(config, eval_batch_size=4))
