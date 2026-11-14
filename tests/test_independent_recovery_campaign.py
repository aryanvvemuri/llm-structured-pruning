"""Independent campaign-level checks of improvement 2: fit order, teacher identity, the
unpruned-recovery sanity check, resume mid-distillation, discard ordering, option
combinations, per-example losses and protocol compatibility with runs/support-v2-main.
"""
import json
import math
from pathlib import Path
import shutil

import pytest

from prune_lab import campaign, multiseed, recovery, support_data
from prune_lab.support_evaluation import evaluate_support
from test_independent_multiseed_e2e import setup  # noqa: F401  (pytest fixture)
from test_independent_multiseed_run import FakeCampaign, base_config as fake_base_config

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'runs' / 'support-v2-main'
DISTILL = {'alpha': .5, 'temperature': 2.0}


def with_recovery(config, **recovery_updates):
    result = json.loads(json.dumps(config))
    result['recovery'].update(recovery_updates)
    return result


def fit(config, output):
    campaign.run_campaign(config, output, 'preflight')
    campaign.run_campaign(config, output, 'fit')
    return {c['id']: c for c in campaign._selection(output)['candidates']}


def weights(record):
    return {p.name: p.read_bytes() for p in sorted(Path(record['checkpoint']).glob('*.safetensors'))}


def strip(rows):
    return [(r['key'], r['prediction'], r['score'], r['rouge_l']) for r in rows]


@pytest.fixture
def spy(monkeypatch):
    calls, loads = [], []
    real_recover, real_load = recovery.recover_model, campaign._load

    def recover(model, batches, evaluate_dev, config, teacher=None):
        calls.append({'teacher': teacher, 'widths': [l.mlp.down_proj.in_features for l in model.model.layers],
                      'max_steps': config['max_steps']})
        kwargs = {} if teacher is None else {'teacher': teacher}
        return real_recover(model, batches, evaluate_dev, config, **kwargs)

    def load(config, checkpoint=None):
        loads.append(checkpoint)
        return real_load(config, checkpoint)
    monkeypatch.setattr(recovery, 'recover_model', recover)
    monkeypatch.setattr(campaign, '_load', load)
    return calls, loads


def test_distillation_fit_order_teacher_identity_and_unpruned_sanity_check(setup, spy):
    calls, loads = spy
    plain = fit(setup.config, setup.tmp / 'plain')
    calls.clear(), loads.clear()
    out = setup.tmp / 'distill'
    distilled = fit(with_recovery(setup.config, distillation=DISTILL), out)
    fit_calls = calls[1:]  # calls[0] is preflight
    assert [c['teacher'] is None for c in fit_calls] == [True, False, False]
    assert fit_calls[0]['widths'] == [24, 24]  # unpruned-recovery first
    assert {tuple(c['widths']) for c in fit_calls[1:]} == {tuple(distilled['per-layer-01']['layer_widths']),
                                                          tuple(distilled['uniform-10']['layer_widths'])}
    assert distilled['per-layer-01']['layer_widths'] != distilled['uniform-10']['layer_widths']
    teacher = distilled['unpruned-recovery']
    teacher_dir = str(out.resolve() / 'candidates' / 'unpruned-recovery' / 'checkpoint')
    assert teacher['checkpoint'] == teacher_dir
    assert loads.count(teacher_dir) == 1 + 2  # export reload + one teacher load per student
    assert 'distillation' not in teacher and 'distillation' not in teacher['recovery']
    for name in ('per-layer-01', 'uniform-10'):
        info = distilled[name]['distillation']
        assert info['teacher_id'] == 'unpruned-recovery' and info['teacher_checkpoint'] == teacher_dir
        assert info['teacher_checkpoint_sha256'] == teacher['checkpoint_sha256']
        assert info['alpha'] == .5 and info['temperature'] == 2.0
        assert len(distilled[name]['recovery']['distillation']['train_kl_losses']) == 2
        assert weights(distilled[name]) != weights(plain[name])  # distillation changed the students
    # Record order is the allocation order, regardless of training order.
    assert list(distilled) == list(plain) == ['per-layer-01', 'unpruned-recovery', 'uniform-10']
    # Sanity check of the plan: the teacher is bit-identical to the same candidate without distillation.
    assert weights(teacher) == weights(plain['unpruned-recovery'])
    assert teacher['recovery']['train_losses'] == plain['unpruned-recovery']['recovery']['train_losses']
    assert strip(teacher['dev']['predictions']) == strip(plain['unpruned-recovery']['dev']['predictions'])
    # Default config stores no per-example losses.
    assert all('answer_token_loss' not in row for row in plain['uniform-10']['dev']['predictions'])
    # Test split stays sealed.
    assert not set(setup.keys['test']) & set(setup.evaluated)


def test_resume_mid_distillation_reproduces_uninterrupted_and_discard_waits(setup, spy, monkeypatch):
    calls, _ = spy
    config = with_recovery(setup.config, distillation=DISTILL)
    full = multiseed.run_seeds(config, setup.tmp / 'runs' / 'full', [11]) and \
        {c['id']: c for c in campaign._selection(setup.tmp / 'runs' / 'full' / 'seed-11')['candidates']}
    calls.clear()
    spy_recover = recovery.recover_model

    def interrupt(*args, **kwargs):
        if len(calls) == 2:  # preflight, teacher done; interrupt the first student
            raise KeyboardInterrupt
        return spy_recover(*args, **kwargs)
    monkeypatch.setattr(recovery, 'recover_model', interrupt)
    root = setup.tmp / 'runs' / 'resumed'
    with pytest.raises(KeyboardInterrupt):
        multiseed.run_seeds(config, root, [11], discard_checkpoints=True)
    seed_dir = root / 'seed-11'
    teacher_weights = list((seed_dir / 'candidates' / 'unpruned-recovery' / 'checkpoint').glob('*.safetensors'))
    assert teacher_weights and not (seed_dir / 'checkpoints-deleted.json').exists()
    monkeypatch.setattr(recovery, 'recover_model', spy_recover)
    calls.clear()
    multiseed.run_seeds(config, root, [11], discard_checkpoints=True)
    assert [c['teacher'] is not None for c in calls] == [True, True]  # only the two students; teacher reused
    resumed = {c['id']: c for c in campaign._selection(seed_dir)['candidates']}
    for name, record in resumed.items():
        assert record['status'] == 'complete'
        assert strip(record['dev']['predictions']) == strip(full[name]['dev']['predictions'])
        assert record['recovery']['train_losses'] == full[name]['recovery']['train_losses']
        assert record['recovery'].get('distillation', {}).get('train_kl_losses') == \
            full[name]['recovery'].get('distillation', {}).get('train_kl_losses')
    assert not list(root.rglob('*.safetensors'))
    assert set(json.loads((seed_dir / 'checkpoints-deleted.json').read_text())) == set(resumed)


def test_tampered_or_failed_teacher_is_refused(setup, spy, monkeypatch):
    calls, _ = spy
    config = with_recovery(setup.config, distillation=DISTILL)
    spy_recover = recovery.recover_model

    def interrupt(*args, **kwargs):
        if len(calls) == 2:
            raise KeyboardInterrupt
        return spy_recover(*args, **kwargs)
    monkeypatch.setattr(recovery, 'recover_model', interrupt)
    out = setup.tmp / 'tampered'
    campaign.run_campaign(config, out, 'preflight')
    with pytest.raises(KeyboardInterrupt):
        campaign.run_campaign(config, out, 'fit')
    monkeypatch.setattr(recovery, 'recover_model', spy_recover)
    weights_file = next((out / 'candidates' / 'unpruned-recovery' / 'checkpoint').glob('*.safetensors'))
    data = bytearray(weights_file.read_bytes())
    data[-1] ^= 1
    weights_file.write_bytes(bytes(data))
    calls.clear()
    with pytest.raises(ValueError, match='Saved checkpoint changed'):
        campaign.run_campaign(config, out, 'fit')
    assert calls == [] and not (out / 'candidates' / 'per-layer-01' / 'result.json').exists()

    # A teacher whose training fails makes every student fail explicitly (no silent CE-only fallback).
    seen = []

    def failing(model, batches, evaluate_dev, cfg, teacher=None):
        seen.append(teacher is None)
        if len(seen) == 2:  # preflight passes; the teacher's training (first in fit) fails
            raise RuntimeError('teacher training failed')
        return spy_recover(model, batches, evaluate_dev, cfg, **({'teacher': teacher} if teacher is not None else {}))
    monkeypatch.setattr(recovery, 'recover_model', failing)
    out2 = setup.tmp / 'failed-teacher'
    campaign.run_campaign(config, out2, 'preflight')
    with pytest.raises(ValueError, match='recovery reference failed'):
        campaign.run_campaign(config, out2, 'fit')
    records = {p.parent.name: json.loads(p.read_text()) for p in (out2 / 'candidates').glob('*/result.json')}
    assert records['unpruned-recovery']['status'] == 'failed' and seen == [True, True]
    for name in ('per-layer-01', 'uniform-10'):
        assert records[name]['status'] == 'failed' and 'teacher' in records[name]['error']
    assert not (out2 / 'selection-lock.json').exists()


def test_all_three_options_combine(setup):
    config = with_recovery(setup.config, distillation=DISTILL, lora_targets='mlp+attention', lr_schedule='cosine',
                           warmup_steps=1, min_lr_ratio=.1, max_steps=3, checkpoints=[1, 3])
    records = fit(config, setup.tmp / 'combo')
    peak = config['recovery']['learning_rate']
    for name, record in records.items():
        assert record['status'] == 'complete' and record['export_reload_verified']
        rec = record['recovery']
        assert rec['recipe']['lora_targets'] == 'mlp+attention' and rec['recipe']['lr_schedule'] == 'cosine'
        assert rec['learning_rates'] == pytest.approx([peak, peak, peak * (.1 + .9 * .5 * (1 + math.cos(math.pi / 2)))])
        assert ('distillation' in record) == (name != 'unpruned-recovery')
        assert record['merge_check']['identical_predictions_after_reload'] == record['merge_check']['examples']
    summary_root = setup.tmp / 'runs' / 'combo-ms'
    multiseed.run_seeds(config, summary_root, [11, 12], discard_checkpoints=True)
    for seed in (11, 12):  # each seed distils from its own teacher export
        seed_records = {c['id']: c for c in campaign._selection(summary_root / f'seed-{seed}')['candidates']}
        own = seed_records['unpruned-recovery']
        for name in ('per-layer-01', 'uniform-10'):
            assert seed_records[name]['distillation']['teacher_checkpoint'] == own['checkpoint']
            assert seed_records[name]['distillation']['teacher_checkpoint_sha256'] == own['checkpoint_sha256']
            assert f'seed-{seed}' in own['checkpoint']
    summary = multiseed.aggregate_run(summary_root, samples=20)
    assert summary['recovery_recipe']['distillation'] == DISTILL
    assert summary['recovery_recipe']['lora_targets'] == 'mlp+attention'


@pytest.mark.parametrize('batch_size', [1, 3])
def test_store_example_losses_schema_and_token_weighted_mean(setup, batch_size):
    config = setup.config
    model, tokenizer = campaign._load(config)
    dev = support_data.read_support_jsonl(Path(config['support_data_dir']) / 'dev.jsonl')
    plain = evaluate_support(model, tokenizer, dev, 256, 12, batch_size)
    extra = evaluate_support(model, tokenizer, dev, 256, 12, batch_size, example_losses=True)
    drop = lambda m: {k: v for k, v in m.items() if k not in ('predictions', 'memory', 'generation_seconds',
                                                             'tokens_per_second', 'mean_example_seconds')}
    assert drop(plain) == drop(extra)
    for a, b in zip(plain['predictions'], extra['predictions']):
        assert set(b) - set(a) == {'answer_token_loss', 'answer_tokens'}
        assert {k: v for k, v in a.items() if k != 'seconds'} == {k: v for k, v in b.items() if k not in ('seconds', 'answer_token_loss', 'answer_tokens')}
    rows = extra['predictions']
    weighted = sum(r['answer_token_loss'] * r['answer_tokens'] for r in rows) / sum(r['answer_tokens'] for r in rows)
    assert weighted == pytest.approx(extra['answer_token_loss'], rel=1e-12)
    assert sum(r['answer_tokens'] for r in rows) == extra['supervised_tokens']
    for row, example in zip(rows, dev):  # each row carries its own example's loss (alignment)
        alone = evaluate_support(model, tokenizer, [example], 256, 12)
        assert row['key'] == example.key and row['answer_token_loss'] == pytest.approx(alone['answer_token_loss'], rel=1e-6)


# --- protocol ------------------------------------------------------------------------------------

def test_reference_config_fingerprints_unchanged():
    for name, run in (('support.json', 'support-v1'), ('support-v2.json', 'support-v2-main')):
        config = campaign.read_campaign_config(ROOT / 'configs' / name)
        campaign.validate_config(config)
        assert not set(config['recovery']) & {'lr_schedule', 'warmup_steps', 'min_lr_ratio', 'distillation', 'lora_targets'}
        assert 'store_example_losses' not in config
        run_dir = ROOT / 'runs' / run
        if not run_dir.exists():
            continue
        states = [run_dir / 'state.json'] if (run_dir / 'state.json').exists() else sorted(run_dir.glob('seed-*/state.json'))
        assert states
        for state in states:
            seed = json.loads((state.parent / 'config.json').read_text())['recovery']['seed']
            expected = json.loads(state.read_text())['config_sha256']
            assert campaign._fingerprint(multiseed.seed_config(config, seed)) == expected, state


def copy_run(source, target, edit=None, only=None):
    for seed_dir in sorted(source.glob('seed-*')):
        if only is not None and seed_dir.name not in only:
            continue
        destination = target / seed_dir.name
        destination.mkdir(parents=True)
        for name in ('config.json', 'state.json', 'selection.json', 'selection-lock.json', 'selection-dev.json'):
            if (seed_dir / name).exists():
                shutil.copyfile(seed_dir / name, destination / name)
        if edit is not None:
            config = json.loads((destination / 'config.json').read_text())
            edit(seed_dir.name, config)
            (destination / 'config.json').write_text(json.dumps(config))


needs_main = pytest.mark.skipif(not (MAIN / 'multi-seed.json').exists(), reason='runs/support-v2-main absent')


@needs_main
def test_main_reference_still_compares_against_a_new_recipe(tmp_path):
    before = {p: p.stat().st_mtime_ns for p in MAIN.rglob('*.json')}
    candidate = tmp_path / 'longer'

    def longer(_, config):
        config['recovery'].update(lr_schedule='cosine', warmup_steps=20, min_lr_ratio=.1, max_steps=450,
                                  checkpoints=[150, 300, 450])
    copy_run(MAIN, candidate, longer)
    result = multiseed.compare_runs(MAIN, candidate, pairs=[('uniform-10', 'uniform-10')], samples=200)
    assert result['example_set'] == 'dev-holdout' and result['example_count'] == 450
    assert result['recovery_recipes']['baseline'].get('lr_schedule') is None
    assert result['recovery_recipes']['candidate']['lr_schedule'] == 'cosine'
    assert result['recovery_schedules']['identical'] is False and result['secondary_withheld']
    (comparison,) = result['comparisons']
    assert comparison['token_f1']['estimate_points'] == pytest.approx(0, abs=1e-9)  # same records copied
    assert comparison['token_f1']['seed_pairing'] == 'paired'
    assert {p: p.stat().st_mtime_ns for p in MAIN.rglob('*.json')} == before  # read-only


@needs_main
@pytest.mark.xfail(strict=True, reason='BUG (low): collect_run checks max_steps/checkpoints across seeds but not the '
                   'rest of the recovery recipe; seeds trained with different schedules/distillation/adapter targets '
                   'are aggregated together and the run reports only the last seed\'s recipe.')
def test_mixed_recipes_within_one_run_are_refused(tmp_path):
    mixed = tmp_path / 'mixed'

    def edit(name, config):
        if name.endswith('5'):
            config['recovery']['distillation'] = DISTILL
    copy_run(MAIN, mixed, edit)
    with pytest.raises(ValueError):
        multiseed.collect_run(mixed)


@pytest.mark.parametrize('change', [
    {'lr_schedule': 'cosine', 'warmup_steps': 2, 'min_lr_ratio': .1},
    {'distillation': DISTILL}, {'lora_targets': 'mlp+attention'}, {'max_steps': 300},
])
def test_multi_seed_sharing_rejects_recipe_changes(tmp_path, monkeypatch, change):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(campaign, 'run_campaign', FakeCampaign())
    config = fake_base_config()
    multiseed.run_seeds(config, tmp_path / 'ms', [5])
    other = with_recovery(config, **change)
    with pytest.raises(ValueError, match='beyond recovery.seed'):
        multiseed.run_seeds(other, tmp_path / 'ms', [6])


def test_preflight_accepts_a_warmup_longer_than_its_two_probe_updates(setup):
    """support-v2-longer uses warmup_steps 20 while preflight trains only 2 updates."""
    config = with_recovery(setup.config, lr_schedule='cosine', warmup_steps=3, min_lr_ratio=.1, max_steps=4,
                           checkpoints=[2, 4])
    campaign.validate_config(config)
    campaign.run_campaign(config, setup.tmp / 'warm', 'preflight')
    assert json.loads((setup.tmp / 'warm' / 'preflight.json').read_text())['passed']
