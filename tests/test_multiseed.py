"""Minimal checks for multi-seed running, checkpoint discarding and hierarchical bootstrap."""
import json

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from prune_lab import campaign, multiseed, support_data
from prune_lab.cli import main
from test_support_data import fixture_archive


def variant(scores, groups=None, loss=1.0):
    groups = groups or [f'g{i % 3}' for i in range(len(scores))]
    examples = {f'k{i}': (g, {'token_f1': s, 'rouge_l': s}) for i, (g, s) in enumerate(zip(groups, scores))}
    return {'examples': examples, 'parameters': 10, 'actual_reduction': 0.0,
            'aggregate': {'token_f1': 100 * sum(scores) / len(scores), 'rouge_l': 100 * sum(scores) / len(scores),
                          'answer_token_loss': loss, 'generation_limit_fraction': 0.0}}


def test_constant_difference_collapses_interval_and_is_deterministic():
    a = {1: variant([.2, .4, .6, .1, .3, .5]), 2: variant([.3, .5, .2, .2, .1, .4])}
    b = {s: variant([x + .1 for x in (v['examples'][k][1]['token_f1'] for k in v['examples'])]) for s, v in a.items()}
    result = multiseed.hierarchical_bootstrap(a, b, samples=200, seed=3)
    assert result['estimate_points'] == pytest.approx(10)
    assert result['interval_95_points'] == pytest.approx([10, 10])
    assert result['excludes_zero'] and result['document_families'] == 3 and result['seed_pairing'] == 'paired'
    assert result == multiseed.hierarchical_bootstrap(a, b, samples=200, seed=3)
    independent = multiseed.hierarchical_bootstrap(a, b, paired_seeds=False, samples=200, seed=3)
    assert independent['interval_95_points'][0] < 10 < independent['interval_95_points'][1]


def test_seed_variation_widens_interval_and_pairing_is_checked():
    a = {s: variant([.5] * 6) for s in (1, 2, 3)}
    b = {1: variant([.5] * 6), 2: variant([.6] * 6), 3: variant([.7] * 6)}
    result = multiseed.hierarchical_bootstrap(a, b, samples=500, seed=5)
    assert result['estimate_points'] == pytest.approx(10)
    assert result['families_only_interval'] == pytest.approx([10, 10])
    assert result['seeds_only_interval'][0] < 10 < result['seeds_only_interval'][1]
    own = multiseed.hierarchical_bootstrap(None, b, samples=100)
    assert own['estimate_points'] == pytest.approx(60) and not own['excludes_zero']
    shifted = {1: variant([.5] * 6, groups=['x'] * 6)}
    with pytest.raises(ValueError, match='identical example ids'):
        multiseed.hierarchical_bootstrap({1: a[1]}, shifted)
    with pytest.raises(ValueError, match='shared recovery seeds'):
        multiseed.hierarchical_bootstrap({9: a[1]}, {1: b[1]})


def tiny_support_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    torch.set_num_threads(1)
    backend = Tokenizer(models.WordLevel({'[PAD]': 0, '[UNK]': 1, '[EOS]': 2, 'helpful': 3, 'response': 4}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]', eos_token='[EOS]')
    source = tmp_path / 'source'
    torch.manual_seed(913)
    Qwen2ForCausalLM(Qwen2Config(vocab_size=5, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                                 num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
                                 pad_token_id=0, eos_token_id=2)).save_pretrained(source, safe_serialization=True)
    tokenizer.save_pretrained(source)
    counts = {'recovery': 2, 'calibration': 1, 'dev': 3, 'test': 1}
    prepared = tmp_path / 'prepared'
    support_data.prepare_support_dataset(fixture_archive(tmp_path, monkeypatch), prepared, tokenizer,
                                         counts=counts, max_length=256, max_answer_tokens=16)
    config = campaign.read_campaign_config()
    config.update(task='support', dataset='doc2dial', support_data_dir=str(prepared), model=str(source),
                  device='cpu', cpu_threads=1, counts=counts, targets=[.15], group_size=1,
                  sensitivity_examples=1, benchmark_examples=1, benchmark_repeats=1, max_length=256,
                  max_new_tokens=16, selection_reference='unpruned-recovery', select_uniform=True,
                  evaluate_all_candidates=True)
    config['recovery'].update(rank=2, max_steps=1, checkpoints=[1], gradient_accumulation=1)
    return config


def test_multi_seed_cpu_run_discard_aggregate_and_compare(tmp_path, monkeypatch):
    config = tiny_support_config(tmp_path, monkeypatch)
    observed = []
    evaluate = campaign._evaluate
    monkeypatch.setattr(campaign, '_evaluate', lambda m, t, ex, c: observed.extend(e.key for e in ex) or evaluate(m, t, ex, c))
    root = tmp_path / 'runs' / 'ms'
    multiseed.run_seeds(config, root, [5, 6], discard_checkpoints=True)
    test_keys = {json.loads(line)['key'] for line in (tmp_path / 'prepared' / 'test.jsonl').read_text().splitlines()}
    assert observed and not test_keys & set(observed)
    for name in multiseed.SHARED_FILES:
        assert (root / 'seed-5' / name).read_bytes() == (root / 'seed-6' / name).read_bytes()
    assert json.loads((root / 'seed-6' / 'config.json').read_text())['recovery']['seed'] == 6
    assert not list((root).rglob('*.safetensors'))
    deleted = json.loads((root / 'seed-5' / 'checkpoints-deleted.json').read_text())
    assert set(deleted) == {'per-layer-01', 'unpruned-recovery', 'uniform-10'}
    assert not (root / 'seed-5' / 'final-test').exists()
    calls = len(observed)
    multiseed.run_seeds(config, root, [5, 6], discard_checkpoints=True)  # Resume is a no-op.
    assert len(observed) == calls
    with pytest.raises(ValueError, match='deleted before the test'):
        multiseed.run_seeds(config, root, [5], open_test=True)
    other = dict(config, max_new_tokens=17)
    with pytest.raises(ValueError, match='beyond recovery.seed'):
        multiseed.run_seeds(other, root, [7])
    summary = multiseed.aggregate_run(root, samples=50)
    assert summary['seeds'] == [5, 6]
    assert list(summary['variants']) == ['original', 'unpruned-recovery', 'uniform-10', 'per-layer-01']
    assert summary['variants']['original']['metrics']['token_f1']['sd'] == 0
    assert len(summary['comparisons']) == 6
    assert json.loads((root / 'multi-seed-summary.json').read_text()) == summary
    assert 'does not cover' in (root / 'multi-seed-report.md').read_text()
    main(['multi-seed', 'compare', '--baseline', str(root), '--candidate', str(root), '--samples', '20',
          '--pair', 'uniform-10:per-layer-01'])
    result = json.loads((root / 'compare-vs-ms.json').read_text())
    assert result['comparisons'][0]['token_f1']['seed_pairing'] == 'paired'  # identical seed sets pair by default
    main(['multi-seed', 'compare', '--baseline', str(root), '--candidate', str(root), '--samples', '20',
          '--unpaired-seeds', '--output', str(tmp_path / 'unpaired.json')])
    unpaired = json.loads((tmp_path / 'unpaired.json').read_text())
    assert unpaired['comparisons'][0]['token_f1']['seed_pairing'] == 'independent'
    assert result['example_set'] == 'dev-full' and result['secondary'] == {}
    same = multiseed.compare_runs(root, root, samples=20, paired_seeds=True)
    assert all(c['token_f1']['interval_95_points'] == [0, 0] for c in same['comparisons'])
    import shutil
    other_root = tmp_path / 'runs' / 'other'
    shutil.copytree(root, other_root)
    for path in other_root.glob('seed-*/config.json'):
        path.write_text(json.dumps(dict(json.loads(path.read_text()), eval_batch_size=8)))
    with pytest.raises(ValueError, match='different evaluation settings'):
        multiseed.compare_runs(root, other_root, samples=20)
    next(other_root.glob('seed-5/config.json')).write_text((root / 'seed-5' / 'config.json').read_text())
    with pytest.raises(ValueError, match='different evaluation settings'):
        multiseed.collect_run(other_root)


def test_selection_dev_subset_is_stratified_and_independent_of_recovery_seed():
    from types import SimpleNamespace
    dev = [SimpleNamespace(key=f'{d}{i}', domain=d) for d, n in (('a', 30), ('b', 60), ('c', 10)) for i in range(n)]
    rows = campaign.selection_dev_subset(dev, 20, 7)
    assert [r.domain for r in rows].count('a') == 6 and [r.domain for r in rows].count('b') == 12
    assert {r.key for r in rows} == {r.key for r in campaign.selection_dev_subset(list(reversed(dev)), 20, 7)}
    assert {r.key for r in rows} != {r.key for r in campaign.selection_dev_subset(dev, 20, 8)}
    with pytest.raises(ValueError, match='between 1 and 100'):
        campaign.selection_dev_subset(dev, 101, 7)


def test_multi_seed_with_selection_subset_and_skipped_full_merge_eval(tmp_path, monkeypatch):
    config = tiny_support_config(tmp_path, monkeypatch)
    config.update(selection_dev_examples=2, skip_full_merged_dev=True)
    sizes = []
    evaluate = campaign._evaluate
    monkeypatch.setattr(campaign, '_evaluate', lambda m, t, ex, c: sizes.append(len(ex)) or evaluate(m, t, ex, c))
    root = tmp_path / 'runs' / 'subset'
    multiseed.run_seeds(config, root, [5, 6])
    subsets = [(root / f'seed-{s}' / 'selection-dev.json').read_text() for s in (5, 6)]
    assert subsets[0] == subsets[1] and json.loads(subsets[0])['count'] == 2
    # Full 3-example dev: preflight's dev[:8] (2 checks x 2 seeds), one shared baseline, one reload per candidate per seed.
    assert sizes.count(3) == 4 + 1 + 2 * 3
    assert sizes.count(2) == 2 * 3 * 3  # per candidate: steps 0 and 1 plus the merge identity check
    for record in campaign._selection(root / 'seed-6')['candidates']:
        assert record['dev']['count'] == 3 and 'merged_dev_before_export' not in record
        assert record['merge_check']['examples'] == 2 and record['recovery_selection_dev']['examples'] == 2
        assert all(h['dev']['count'] == 2 for h in record['recovery']['history'])
    summary = multiseed.aggregate_run(root, samples=20)
    assert summary['selection_dev_subset']['count'] == 2
    assert 'part of the 3-example dev set' in (root / 'multi-seed-report.md').read_text()


@pytest.mark.parametrize('p, df, expected', [(.975, 1, 12.7062), (.975, 2, 4.3027), (.975, 4, 2.7764),
                                             (.975, 10, 2.2281), (.975, 30, 2.0423), (.95, 5, 2.0150),
                                             (.995, 2, 9.9248), (.025, 3, -3.1824)])
def test_t_quantile_matches_tables(p, df, expected):
    assert multiseed.t_quantile(p, df) == pytest.approx(expected, abs=1e-4)


def test_decision_requires_bootstrap_and_seed_t_to_agree():
    a = {s: variant([.5] * 6) for s in (1, 2, 3)}
    b = {1: variant([.5] * 6), 2: variant([.6] * 6), 3: variant([.7] * 6)}
    result = multiseed.hierarchical_bootstrap(a, b, samples=300, seed=5)
    t = result['seed_t_interval']
    assert t['method'] == 'paired t over seeds' and t['df'] == 2
    assert t['interval_95_points'] == pytest.approx([10 - 4.3027 * 10 / 3 ** .5, 10 + 4.3027 * 10 / 3 ** .5], abs=1e-3)
    assert not t['excludes_zero'] and result['decision'] == 'not established'
    steady = {s: variant([.6] * 6) for s in (1, 2, 3)}
    assert multiseed.hierarchical_bootstrap(a, steady, samples=100)['decision'] == 'confident'
    welch = multiseed.hierarchical_bootstrap(a, b, paired_seeds=False, samples=100)['seed_t_interval']
    assert welch['method'] == 'Welch t over seeds' and welch['df'] == pytest.approx(2)
    assert multiseed.hierarchical_bootstrap(None, b, samples=20)['decision'] is None



def write_seed(root, seed, keys, subset=None, schedule=(1, [1]), shift=0.0):
    """Synthetic completed seed directory with per-example dev records."""
    from prune_lab import campaign as c
    directory = root / f'seed-{seed}'
    directory.mkdir(parents=True)
    config = {'seed': 1, 'recovery': {'seed': seed, 'max_steps': schedule[0], 'checkpoints': schedule[1]}}
    (directory / 'config.json').write_text(json.dumps(config))
    (directory / 'state.json').write_text(json.dumps({'dataset': {'dev': {'count': len(keys), 'sha256': 'd'}}, 'test_started': False}))

    def metrics(scores):
        return {'token_f1': sum(scores.values()) / len(scores), 'rouge_l': sum(scores.values()) / len(scores),
                'answer_token_loss': 1.0, 'predictions': [{'key': k, 'group_id': f'g{i % 5}', 'score': v, 'rouge_l': v}
                                                          for i, (k, v) in enumerate(scores.items())]}
    base = {k: .3 for k in keys}
    # The candidate looks much better on the selection subset only: optimism the holdout must remove.
    cand = {k: (.9 if subset and k in subset else .3 + shift + .01 * (seed % 3)) for k in keys}
    selection = {'baseline': {'dev': metrics(base), 'parameters': 10},
                 'candidates': [{'id': 'per-layer-01', 'status': 'complete', 'parameters': 8, 'actual_reduction': .2,
                                 'checkpoint': 'x', 'dev': metrics(cand)}], 'tiers': []}
    (directory / 'selection.json').write_text(json.dumps(selection))
    (directory / 'selection-lock.json').write_text(json.dumps({'sha256': c._fingerprint(selection)}))
    if subset is not None:
        (directory / 'selection-dev.json').write_text(json.dumps({'count': len(subset), 'keys_sha256': 'h', 'keys': sorted(subset)}))
    return directory


def test_holdout_headline_excludes_selection_subset(tmp_path):
    keys = [f'k{i:02d}' for i in range(20)]
    subset = set(keys[:5])
    for seed in (1, 2, 3):
        write_seed(tmp_path / 'run', seed, keys, subset)
    summary = multiseed.aggregate_run(tmp_path / 'run', samples=50)
    assert summary['example_set'] == 'dev-holdout' and summary['example_count'] == 15
    holdout = summary['variants']['per-layer-01']['metrics']['token_f1']['mean']
    full = summary['secondary']['dev-full']['variants']['per-layer-01']['metrics']['token_f1']['mean']
    assert holdout == pytest.approx(31) and full == pytest.approx((5 * 90 + 15 * 31) / 20)
    assert summary['comparisons'][0]['token_f1']['estimate_points'] == pytest.approx(1)
    assert 'Headline example set: **dev-holdout**' in (tmp_path / 'run' / 'multi-seed-report.md').read_text()
    bad = write_seed(tmp_path / 'bad', 1, keys, subset)
    (bad / 'selection-dev.json').write_text(json.dumps({'count': 1, 'keys_sha256': 'h', 'keys': ['missing']}))
    with pytest.raises(ValueError, match='not all in the reported dev records'):
        multiseed.collect_run(tmp_path / 'bad')


def test_compare_schedules_holdout_and_pairing(tmp_path):
    keys = [f'k{i:02d}' for i in range(20)]
    subset = set(keys[:5])
    for seed in (1, 2, 3):
        write_seed(tmp_path / 'main', seed, keys, subset)
        write_seed(tmp_path / 'long', seed, keys, subset, schedule=(2, [1, 2]), shift=.05)
        write_seed(tmp_path / 'main-nosub', seed, keys)
        write_seed(tmp_path / 'long-nosub', seed, keys, schedule=(2, [1, 2]), shift=.05)
    result = multiseed.compare_runs(tmp_path / 'main', tmp_path / 'long', samples=50)
    assert result['example_set'] == 'dev-holdout' and result['secondary'] == {} and result['secondary_withheld']
    assert result['recovery_schedules']['candidate'] == {'max_steps': 2, 'checkpoints': [1, 2]}
    assert result['comparisons'][1]['token_f1']['seed_pairing'] == 'paired'
    assert result['comparisons'][1]['token_f1']['estimate_points'] == pytest.approx(5)
    allowed = multiseed.compare_runs(tmp_path / 'main', tmp_path / 'long', samples=50, allow_different_schedules=True)
    assert allowed['allow_different_schedules'] and set(allowed['secondary']) == {'dev-full'}
    with pytest.raises(ValueError, match='allow-different-schedules'):
        multiseed.compare_runs(tmp_path / 'main-nosub', tmp_path / 'long-nosub', samples=20)
    forced = multiseed.compare_runs(tmp_path / 'main-nosub', tmp_path / 'long-nosub', samples=20,
                                    paired_seeds=False, allow_different_schedules=True)
    assert forced['example_set'] == 'dev-full' and forced['comparisons'][0]['token_f1']['seed_pairing'] == 'independent'
    write_seed(tmp_path / 'fewer', 1, keys, subset)
    auto = multiseed.compare_runs(tmp_path / 'main', tmp_path / 'fewer', samples=20)
    assert auto['comparisons'][0]['token_f1']['seed_pairing'] == 'independent'
