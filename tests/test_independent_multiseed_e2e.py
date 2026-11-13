"""Independent CPU end-to-end checks of multi-seed runs and the selection dev subset.

Tiny randomly initialised Qwen2 model, synthetic three-domain Doc2Dial-like archive.
"""
import json
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from prune_lab import campaign, multiseed, recovery, scoring, support_data
from test_independent_support_v2 import LAYOUT, build_archive

COUNTS = {'recovery': 3, 'calibration': 1, 'dev': 6, 'test': 3}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    torch.set_num_threads(1)
    backend = Tokenizer(models.WordLevel({'[PAD]': 0, '[UNK]': 1, '[EOS]': 2, 'support': 3, 'reply': 4, ':': 5},
                                         unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]', eos_token='[EOS]')
    source = tmp_path / 'model'
    torch.manual_seed(5)
    Qwen2ForCausalLM(Qwen2Config(vocab_size=6, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                                 num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
                                 pad_token_id=0, eos_token_id=2)).save_pretrained(source, safe_serialization=True)
    tokenizer.save_pretrained(source)
    archive = build_archive(tmp_path, monkeypatch, LAYOUT)
    prepared = tmp_path / 'prepared'
    support_data.prepare_support_dataset(archive, prepared, tokenizer, counts=COUNTS, max_length=256,
                                         max_answer_tokens=12)
    config = campaign.read_campaign_config()
    config.update(task='support', dataset='doc2dial', support_data_dir=str(prepared), model=str(source),
                  device='cpu', cpu_threads=1, counts=COUNTS, targets=[.15], group_size=1,
                  sensitivity_examples=1, benchmark_examples=1, benchmark_repeats=1, max_length=256,
                  max_new_tokens=12, selection_reference='unpruned-recovery', select_uniform=True,
                  evaluate_all_candidates=True, selection_dev_examples=3, skip_full_merged_dev=True)
    config['recovery'].update(rank=2, max_steps=2, checkpoints=[1, 2], gradient_accumulation=1)
    evaluated = []
    evaluate = campaign._evaluate
    monkeypatch.setattr(campaign, '_evaluate', lambda m, t, ex, c: evaluated.extend(e.key for e in ex) or evaluate(m, t, ex, c))
    scored = []
    score = scoring.score_neurons
    monkeypatch.setattr(scoring, 'score_neurons', lambda *a, **k: scored.append(1) or score(*a, **k))
    keys = {name: [json.loads(l)['key'] for l in (prepared / f'{name}.jsonl').read_text().splitlines()]
            for name in COUNTS}
    domains = {json.loads(l)['key']: json.loads(l)['domain'] for l in (prepared / 'dev.jsonl').read_text().splitlines()}
    return SimpleNamespace(config=config, evaluated=evaluated, scored=scored, keys=keys, dev_domains=domains,
                           tmp=tmp_path)


def records(seed_dir):
    return {c['id']: c for c in campaign._selection(seed_dir)['candidates']}


def test_multi_seed_end_to_end_sealing_sharing_subset_and_resume(setup, monkeypatch):
    s = setup
    root = s.tmp / 'runs' / 'a'
    multiseed.run_seeds(s.config, root, [11, 12])
    # Sealing: no test example was ever evaluated, nothing recorded for test.
    assert s.evaluated and not set(s.keys['test']) & set(s.evaluated)
    for seed in (11, 12):
        d = root / f'seed-{seed}'
        assert not json.loads((d / 'state.json').read_text())['test_started']
        assert not (d / 'final-test').exists() and not (d / 'summary.json').exists()
    # Baseline/calibration/masks computed once and copied byte-for-byte.
    assert len(s.scored) == 1
    for name in multiseed.SHARED_FILES:
        assert (root / 'seed-11' / name).read_bytes() == (root / 'seed-12' / name).read_bytes()
    # The selection subset: identical across seeds, stratified, recorded in each candidate.
    subset = json.loads((root / 'seed-11' / 'selection-dev.json').read_text())
    assert subset == json.loads((root / 'seed-12' / 'selection-dev.json').read_text())
    assert subset['count'] == 3 and sorted(s.dev_domains[k] for k in subset['keys']) == ['alpha', 'beta', 'short']
    for seed in (11, 12):
        for record in records(root / f'seed-{seed}').values():
            assert record['recovery_selection_dev']['keys_sha256'] == subset['keys_sha256']
            assert all([p['key'] for p in h['dev']['predictions']] == subset['keys'] for h in record['recovery']['history'])
            # Reported dev metrics are always the full dev set, after reload.
            assert sorted(p['key'] for p in record['dev']['predictions']) == sorted(s.keys['dev'])
            assert record['dev']['count'] == COUNTS['dev'] and record['export_reload_verified']
            assert record['merge_check']['examples'] == 3
    baseline = campaign._selection(root / 'seed-11')['baseline']
    assert sorted(p['key'] for p in baseline['dev']['predictions']) == sorted(s.keys['dev'])
    # Different recovery seeds really train differently.
    r11, r12 = records(root / 'seed-11'), records(root / 'seed-12')
    assert r11['per-layer-01']['recovery']['seed'] == 11 and r12['per-layer-01']['recovery']['seed'] == 12
    assert any(r11[c]['recovery']['train_losses'] != r12[c]['recovery']['train_losses'] for c in r11)
    # Aggregated means equal the per-example means of the saved full-dev predictions.
    summary = multiseed.aggregate_run(root, samples=50)
    for name, variant in summary['variants'].items():
        assert variant['metrics']['token_f1']['mean'] == pytest.approx(variant['token_f1_interval']['estimate_points'])
    assert summary['selection_dev_subset'] == {'count': 3, 'keys_sha256': subset['keys_sha256']}

    # A second run directory, interrupted mid-fit in the second seed, then resumed.
    other = s.tmp / 'runs' / 'b'
    calls = []
    real = recovery.recover_model

    def interrupting(*args, **kwargs):
        calls.append(args[3]['seed'])
        if args[3]['seed'] == 12 and calls.count(12) == 3:  # preflight, per-layer-01, then interrupt
            raise KeyboardInterrupt
        return real(*args, **kwargs)
    monkeypatch.setattr(recovery, 'recover_model', interrupting)
    with pytest.raises(KeyboardInterrupt):
        multiseed.run_seeds(s.config, other, [11, 12])
    seed11 = {p: p.read_bytes() for p in (other / 'seed-11').rglob('*') if p.is_file()}
    first12 = (other / 'seed-12' / 'candidates' / 'per-layer-01' / 'result.json').read_bytes()
    calls.clear()
    monkeypatch.setattr(recovery, 'recover_model', lambda *a, **k: calls.append(a[3]['seed']) or real(*a, **k))
    multiseed.run_seeds(s.config, other, [11, 12])
    assert calls == [12, 12]  # only the two unfinished seed-12 candidates are trained
    assert {p: p.read_bytes() for p in seed11} == seed11
    assert (other / 'seed-12' / 'candidates' / 'per-layer-01' / 'result.json').read_bytes() == first12
    assert (other / 'seed-12' / 'selection-dev.json').read_bytes() == (root / 'seed-12' / 'selection-dev.json').read_bytes()
    # Resumed results equal the uninterrupted run's (deterministic CPU, same seed): nothing mixed.
    for name, record in records(other / 'seed-12').items():
        strip = lambda rows: [(r['key'], r['prediction'], r['score'], r['rouge_l']) for r in rows]
        assert strip(record['dev']['predictions']) == strip(r12[name]['dev']['predictions'])
        assert record['recovery']['train_losses'] == r12[name]['recovery']['train_losses']
    assert len(s.scored) == 2  # one baseline per run directory

    # Discard, then the test cannot be opened and remains sealed.
    multiseed.discard_checkpoint_weights(other / 'seed-12')
    with pytest.raises(ValueError, match='Incomplete safetensors checkpoint'):
        campaign.run_campaign(dict(s.config, recovery=dict(s.config['recovery'], seed=12)), other / 'seed-12', 'finalize')
    assert not json.loads((other / 'seed-12' / 'state.json').read_text())['test_started']
    assert not set(s.keys['test']) & set(s.evaluated)


# --- selection_dev_subset units -----------------------------------------------------------

def dev_rows(spec):
    return [SimpleNamespace(key=f'{d}-{i:03d}', domain=d) for d, n in spec for i in range(n)]


def test_selection_subset_proportional_order_free_and_recovery_seed_free(tmp_path):
    dev = dev_rows([('dmv', 250), ('ssa', 150), ('va', 140), ('studentaid', 60)])
    rows = campaign.selection_dev_subset(dev, 150, 20261114)
    counts = {d: sum(r.domain == d for r in rows) for d in ('dmv', 'ssa', 'va', 'studentaid')}
    assert sum(counts.values()) == 150 and len({r.key for r in rows}) == 150
    exact = {'dmv': 62.5, 'ssa': 37.5, 'va': 35.0, 'studentaid': 15.0}
    assert all(abs(counts[d] - exact[d]) < 1 for d in exact)
    # Same rows whatever the input order; output keeps the caller's dev order.
    reversed_rows = campaign.selection_dev_subset(list(reversed(dev)), 150, 20261114)
    assert {r.key for r in reversed_rows} == {r.key for r in rows}
    assert [r.key for r in rows] == [r.key for r in dev if r.key in {x.key for x in rows}]
    # Fresh objects (as reloaded in another run directory) give the same selection.
    clones = [SimpleNamespace(key=r.key, domain=r.domain) for r in dev]
    assert [r.key for r in campaign.selection_dev_subset(clones, 150, 20261114)] == [r.key for r in rows]
    # Persisted record depends on the top-level seed only, never recovery.seed.
    config = campaign.read_campaign_config()
    config.update(selection_dev_examples=150, seed=20261114)
    one, two = tmp_path / 'one', tmp_path / 'two'
    one.mkdir(), two.mkdir()
    campaign._selection_dev(config, one, dev)
    campaign._selection_dev(dict(config, recovery=dict(config['recovery'], seed=999)), two, clones)
    assert (one / 'selection-dev.json').read_bytes() == (two / 'selection-dev.json').read_bytes()
    assert campaign._selection_dev(dict(config, selection_dev_examples=None), tmp_path, dev) is dev
    # Tampering with the persisted subset is refused on resume.
    record = json.loads((one / 'selection-dev.json').read_text())
    record['keys'][0] = 'other'
    (one / 'selection-dev.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='differs'):
        campaign._selection_dev(config, one, dev)
    # Full-size subset is all of dev.
    assert {r.key for r in campaign.selection_dev_subset(dev, len(dev), 1)} == {r.key for r in dev}
    with pytest.raises(ValueError):
        campaign.selection_dev_subset(dev, len(dev) + 1, 1)
    with pytest.raises(ValueError):
        campaign.selection_dev_subset(dev, 0, 1)


@pytest.mark.parametrize('value', [0, -3, True, 2.0, '150', [150]])
def test_invalid_selection_dev_examples_rejected_by_config_validation(value):
    config = campaign.read_campaign_config()
    config['selection_dev_examples'] = value
    with pytest.raises(ValueError, match='selection_dev_examples'):
        campaign.validate_config(config)


@pytest.mark.parametrize('value', [1, 'yes', None])
def test_invalid_skip_full_merged_dev_rejected(value):
    config = campaign.read_campaign_config()
    config['skip_full_merged_dev'] = value
    with pytest.raises(ValueError, match='skip_full_merged_dev'):
        campaign.validate_config(config)


def test_selection_dev_examples_above_dev_count_rejected_up_front():
    config = campaign.read_campaign_config()
    config['selection_dev_examples'] = config['counts']['dev'] + 1
    with pytest.raises(ValueError):
        campaign.validate_config(config)


def test_new_options_absent_leave_v1_fingerprint_unchanged():
    config = campaign.read_campaign_config()
    before = campaign._fingerprint(config)
    campaign.validate_config(config)
    assert campaign._fingerprint(config) == before
    assert 'selection_dev_examples' not in config and 'skip_full_merged_dev' not in config
