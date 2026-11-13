"""Independent support integration and continuous-metric boundary checks."""
import json
import random

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from prune_lab import campaign, support_data
from prune_lab.metrics import selection_score
from test_support_data import fixture_archive


def test_support_real_cpu_campaign_and_review_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    torch.set_num_threads(1)
    torch.manual_seed(913)
    backend = Tokenizer(models.WordLevel({'[PAD]': 0, '[UNK]': 1, '[EOS]': 2,
                                         'helpful': 3, 'response': 4}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]',
                                       unk_token='[UNK]', eos_token='[EOS]')
    source = tmp_path / 'source'
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=5, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=256, pad_token_id=0, eos_token_id=2))
    model.save_pretrained(source, safe_serialization=True)
    tokenizer.save_pretrained(source)
    del model
    counts = {name: 1 for name in support_data.DEFAULT_COUNTS}
    prepared = tmp_path / 'prepared'
    manifest = support_data.prepare_support_dataset(fixture_archive(tmp_path, monkeypatch),
        prepared, tokenizer, counts=counts, max_length=256, max_answer_tokens=16)
    config = campaign.read_campaign_config()
    config.update(task='support', dataset='doc2dial', support_data_dir=str(prepared),
                  model=str(source), device='cpu', cpu_threads=1, counts=counts,
                  targets=[.15], group_size=1, sensitivity_examples=1,
                  benchmark_examples=1, benchmark_repeats=1, max_length=256,
                  max_new_tokens=16, selection_reference='unpruned-recovery',
                  select_uniform=True, evaluate_all_candidates=True)
    config['recovery'].update(rank=2, max_steps=1, checkpoints=[1], gradient_accumulation=1)
    output = tmp_path / 'run'
    observed = []
    evaluate = campaign._evaluate
    def track(model, tokenizer, examples, config):
        observed.extend(ex.key for ex in examples)
        return evaluate(model, tokenizer, examples, config)
    monkeypatch.setattr(campaign, '_evaluate', track)
    campaign.run_campaign(config, output, 'preflight')
    campaign.run_campaign(config, output, 'fit')
    state, splits = campaign._initialize(config, output)
    assert not set(observed).intersection(ex.key for ex in splits['test'])
    assert isinstance(splits['test'][0], support_data.SupportExample)
    assert json.loads((output / 'data-provenance.json').read_text()) == manifest
    selection = campaign._selection(output)
    assert all(c['status'] == 'complete' for c in selection['candidates'])
    assert selection['reference_id'] == 'unpruned-recovery'
    ref = next(c for c in selection['candidates'] if c['id'] == 'unpruned-recovery')
    assert selection['tiers'] == campaign.select_tiers(ref, selection['candidates'], config['tiers_pp'])
    campaign.run_campaign(config, output, 'finalize')
    summary = json.loads((output / 'summary.json').read_text())
    assert set(summary['final_test']) == {'original', 'per-layer-01', 'unpruned-recovery', 'uniform-10'}
    for row in summary['final_test'].values():
        metrics = row['metrics']
        assert metrics['score_name'] == 'token_f1'
        assert 'accuracy' not in metrics and 'correct' not in metrics
        assert metrics['predictions'][0]['group_id'] == splits['test'][0].group_id
    for tier in summary['tiers']:
        if tier['candidate_id']:
            assert tier['test']['bootstrap_unit'] == 'document_group'
            assert 'delta_pp' not in tier['test']
    report = (output / 'report.md').read_text()
    assert 'not factual correctness' in report and 'unpruned-recovery' in report
    worksheet = output / 'blind-review.jsonl'
    review = json.loads(worksheet.read_text())
    assert len(review['responses']) == 4
    alias = next(iter(review['review']))
    review['review'][alias]['notes'] = 'Human grading must survive resume.'
    worksheet.write_text(json.dumps(review) + '\n')
    calls = len(observed)
    campaign.run_campaign(config, output, 'finalize')
    assert len(observed) == calls
    assert json.loads(worksheet.read_text()) == review
    state, splits = campaign._initialize(config, output)
    with pytest.raises(ValueError, match='locked'):
        campaign.fit(config, output, state, splits)


def test_lexical_selection_against_recovered_reference_not_original_accuracy():
    def metrics(value):
        return {'selection_score': value, 'score_name': 'token_f1', 'answer_token_loss': 1}
    reference = {'parameters': 1000, 'dev': metrics(.8)}
    rows = [{'id': name, 'status': 'complete', 'parameters': n, 'checkpoint': name,
             'control': control, 'selection_eligible': control, 'dev': metrics(score)}
            for name, n, score, control in [('small', 700, .4, False),
                                           ('close', 800, .75, False),
                                           ('uniform', 900, .8, True)]]
    assert [t['candidate_id'] for t in campaign.select_tiers(reference, rows, [0, 5, 10])] == ['uniform', 'close', 'close']
    assert selection_score(dict(metrics(.7), accuracy=.1)) == .7
    assert selection_score({'accuracy': .3}) == .3
    wrong_metric = dict(rows[0], parameters=500,
                        dev={'accuracy': 1.0, 'answer_token_loss': 0})
    assert campaign.select_tiers(reference, [wrong_metric], [0])[0]['candidate_id'] is None


def test_document_cluster_bootstrap_matches_independent_resampling():
    groups = ['large', 'large', 'large', 'small']
    first = {'score_name': 'token_f1', 'predictions': [
        {'key': str(i), 'group_id': g, 'score': .5} for i, g in enumerate(groups)]}
    second = {'score_name': 'token_f1', 'predictions': [
        dict(row, score=.7 if row['group_id'] == 'large' else .1) for row in first['predictions']]}
    result = campaign.paired_score_delta(first, second, seed=13, samples=200)
    rng = random.Random(13)
    cluster_totals = [(3 * (.7 - .5), 3), (.1 - .5, 1)]
    values = []
    for _ in range(200):
        draw = rng.choices(cluster_totals, k=2)
        values.append(100 * sum(s for s, n in draw) / sum(n for s, n in draw))
    values.sort()
    assert result['delta_points'] == pytest.approx(5)
    assert result['document_groups'] == 2
    assert result['bootstrap_95_interval_points'] == pytest.approx([values[5], values[195]])
    second['predictions'][0]['group_id'] = 'mismatched'
    with pytest.raises(ValueError, match='matching document groups'):
        campaign.paired_score_delta(first, second)
