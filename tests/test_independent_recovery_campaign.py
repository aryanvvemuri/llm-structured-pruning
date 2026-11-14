"""Independent campaign-level checks of improvement 2a: per-example losses, the preflight probe,
recipe consistency across seeds and protocol compatibility with runs/support-v2-main.
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
        assert not set(config['recovery']) & {'lr_schedule', 'warmup_steps', 'min_lr_ratio'}
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
def test_mixed_recipes_within_one_run_are_refused(tmp_path):
    mixed = tmp_path / 'mixed'

    def edit(name, config):
        if name.endswith('5'):  # same max_steps/checkpoints, different learning-rate schedule
            config['recovery'].update(lr_schedule='cosine', warmup_steps=5, min_lr_ratio=.1)
    copy_run(MAIN, mixed, edit)
    with pytest.raises(ValueError):
        multiseed.collect_run(mixed)


@pytest.mark.parametrize('change', [
    {'lr_schedule': 'cosine', 'warmup_steps': 2, 'min_lr_ratio': .1}, {'max_steps': 300},
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
