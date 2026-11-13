"""Independent orchestration checks for multi-seed runs using a recording fake campaign.

The fake stands in for run_campaign so sealing, seed sharing, resume and checkpoint
discarding can be checked exactly and cheaply. A real tiny-model run is in
test_independent_multiseed_e2e.py.
"""
import json
from pathlib import Path

import pytest

from prune_lab import campaign, cli, multiseed
from test_independent_multiseed_stats import DEV, TEST, predictions


class FakeCampaign:
    def __init__(self, dataset_for=None):
        self.calls = []
        self.dataset_for = dataset_for or (lambda seed: {'dev': {'count': len(DEV), 'sha256': 'dev'},
                                                         'test': {'count': len(TEST), 'sha256': 'test'}})

    def __call__(self, config, output, stage='all'):
        output = Path(output)
        seed = config['recovery']['seed']
        self.calls.append((seed, stage))
        output.mkdir(parents=True, exist_ok=True)
        if stage == 'preflight':
            if not (output / 'state.json').exists():
                (output / 'config.json').write_text(json.dumps(config))
                (output / 'state.json').write_text(json.dumps({'config_sha256': campaign._fingerprint(config),
                                                               'dataset': self.dataset_for(seed), 'test_started': False}))
            elif json.loads((output / 'config.json').read_text()) != config:
                raise ValueError('Run configuration changed; use a new output directory')
            (output / 'preflight.json').write_text('{}')
        elif stage == 'fit':
            if (output / 'selection-lock.json').exists():
                return
            if not (output / 'allocation.json').exists():
                self.calls.append((seed, 'baseline'))
                (output / 'baseline.json').write_text(json.dumps({'computed_by_seed': seed}))
                (output / 'allocation.json').write_text(json.dumps({'computed_by_seed': seed}))
            candidates = []
            for name in ('per-layer-01', 'uniform-10'):
                checkpoint = output / 'candidates' / name / 'checkpoint'
                checkpoint.mkdir(parents=True)
                (checkpoint / 'config.json').write_text('{}')
                (checkpoint / 'tokenizer.json').write_text('{}')
                (checkpoint / 'prune_lab_manifest.json').write_text('{}')
                (checkpoint / 'model.safetensors').write_bytes(b'w' * (100 + seed))
                (checkpoint / 'model-00002.safetensors').write_bytes(b'v' * 7)
                rows = [(k, g, min(1, f + .01 * seed), r) for k, g, f, r in DEV]
                candidates.append({'id': name, 'status': 'complete', 'parameters': 80, 'actual_reduction': .2,
                                   'checkpoint': str(checkpoint), 'checkpoint_sha256': campaign.checkpoint_identity(checkpoint),
                                   'dev': predictions(rows)})
            candidates.append({'id': 'per-layer-02', 'status': 'failed', 'error': 'x'})
            selection = {'baseline': {'dev': predictions(DEV), 'parameters': 100}, 'candidates': candidates, 'tiers': []}
            (output / 'selection.json').write_text(json.dumps(selection))
            (output / 'selection-lock.json').write_text(json.dumps({'sha256': campaign._fingerprint(selection)}))
        elif stage == 'finalize':
            selection = campaign._selection(output)
            for record in selection['candidates']:
                if record['status'] == 'complete':
                    campaign._verify_checkpoint(record)
            state = json.loads((output / 'state.json').read_text())
            state['test_started'] = True
            (output / 'state.json').write_text(json.dumps(state))
            (output / 'final-test').mkdir(exist_ok=True)
            for name in ('original', 'per-layer-01', 'uniform-10'):
                (output / 'final-test' / f'{name}.json').write_text(json.dumps({'metrics': predictions(TEST)}))
            (output / 'summary.json').write_text('{}')
        else:
            raise AssertionError(stage)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # experiment_lock() uses ./runs relative to cwd
    instance = FakeCampaign()
    monkeypatch.setattr(campaign, 'run_campaign', instance)
    return instance


def base_config():
    config = campaign.read_campaign_config()
    config['device'] = 'cpu'
    return config


def stages(fake, stage):
    return [s for s, st in fake.calls if st == stage]


def test_dev_only_by_default_source_first_and_shared_allocation(fake, tmp_path):
    root = tmp_path / 'ms'
    multiseed.run_seeds(base_config(), root, [6, 5, 9])
    assert stages(fake, 'finalize') == []
    assert {st for _, st in fake.calls} == {'preflight', 'fit', 'baseline'}
    assert stages(fake, 'baseline') == [6]  # first listed seed is the allocation source
    assert stages(fake, 'preflight')[0] == 6
    for seed in (5, 9):
        for name in multiseed.SHARED_FILES:
            assert (root / f'seed-{seed}' / name).read_bytes() == (root / 'seed-6' / name).read_bytes()
        assert json.loads((root / f'seed-{seed}' / 'config.json').read_text())['recovery']['seed'] == seed
    assert not (root / 'seed-6' / 'shared-allocation.json').exists()
    assert not any(root.rglob('final-test'))
    # Weights are kept unless explicitly discarded.
    assert len(list(root.rglob('*.safetensors'))) == 3 * 2 * 2
    plan = json.loads((root / 'multi-seed.json').read_text())
    assert plan['seeds'] == [6, 5, 9] and plan['allocation_source_seed'] == 6


def test_resume_and_extension_do_not_recompute_or_change_source(fake, tmp_path):
    root = tmp_path / 'ms'
    config = base_config()
    multiseed.run_seeds(config, root, [5, 6])
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file() and p.name != 'multi-seed.json'}
    fake.calls.clear()
    multiseed.run_seeds(config, root, [5, 6])
    assert stages(fake, 'baseline') == []
    multiseed.run_seeds(config, root, [7])
    assert stages(fake, 'baseline') == []
    assert json.loads((root / 'seed-7' / 'baseline.json').read_text()) == {'computed_by_seed': 5}
    after = {p: p.read_bytes() for p in before}
    assert after == before
    assert json.loads((root / 'multi-seed.json').read_text())['seeds'] == [5, 6, 7]


@pytest.mark.parametrize('change', [
    lambda c: c.update(max_new_tokens=c['max_new_tokens'] + 1),
    lambda c: c['recovery'].update(learning_rate=1e-3),
    lambda c: c.update(seed=c['seed'] + 1),
    lambda c: c.update(selection_dev_examples=3),
])
def test_config_differing_beyond_recovery_seed_is_rejected(fake, tmp_path, change):
    root = tmp_path / 'ms'
    config = base_config()
    multiseed.run_seeds(config, root, [5])
    other = json.loads(json.dumps(config))
    change(other)
    with pytest.raises(ValueError, match='beyond recovery.seed'):
        multiseed.run_seeds(other, root, [6])
    assert not (root / 'seed-6').exists()
    only_seed = json.loads(json.dumps(config))
    only_seed['recovery']['seed'] = 999  # the base recovery.seed itself is irrelevant
    multiseed.run_seeds(only_seed, root, [6])
    assert (root / 'seed-6' / 'allocation.json').exists()


def test_different_dataset_or_config_in_seed_directory_is_not_shared(fake, tmp_path):
    fake.dataset_for = lambda seed: {'dev': {'count': len(DEV), 'sha256': 'dev' if seed == 5 else 'other'},
                                     'test': {'count': len(TEST), 'sha256': 'test'}}
    root = tmp_path / 'ms'
    with pytest.raises(ValueError, match='datasets differ'):
        multiseed.run_seeds(base_config(), root, [5, 6])
    assert not (root / 'seed-6' / 'allocation.json').exists()
    # A seed directory whose own config was edited behind the plan's back is refused too.
    fake.dataset_for = FakeCampaign().dataset_for
    root2 = tmp_path / 'ms2'
    multiseed.run_seeds(base_config(), root2, [5])
    rogue = base_config()
    rogue['recovery'].update(seed=6, rank=4)
    (root2 / 'seed-6').mkdir()
    (root2 / 'seed-6' / 'config.json').write_text(json.dumps(rogue))
    (root2 / 'seed-6' / 'state.json').write_text(json.dumps({'dataset': fake.dataset_for(6), 'test_started': False}))
    with pytest.raises(ValueError):
        multiseed.run_seeds(base_config(), root2, [6])
    assert not (root2 / 'seed-6' / 'allocation.json').exists()


@pytest.mark.parametrize('seeds', [[], [5, 5], [-1], [True], [1.5], ['3']])
def test_invalid_seed_lists(fake, tmp_path, seeds):
    with pytest.raises(ValueError):
        multiseed.run_seeds(base_config(), tmp_path / 'ms', seeds)
    assert fake.calls == []


def test_discard_deletes_only_weights_logs_and_aggregation_still_works(fake, tmp_path):
    root = tmp_path / 'ms'
    multiseed.run_seeds(base_config(), root, [5, 6], discard_checkpoints=True)
    assert not list(root.rglob('*.safetensors'))
    for seed in (5, 6):
        log = json.loads((root / f'seed-{seed}' / 'checkpoints-deleted.json').read_text())
        assert set(log) == {'per-layer-01', 'uniform-10'}  # failed candidate untouched
        for name, entry in log.items():
            checkpoint = Path(entry['checkpoint'])
            assert sorted(entry['deleted_files']) == ['model-00002.safetensors', 'model.safetensors']
            assert entry['deleted_bytes'] == 100 + seed + 7
            assert sorted(p.name for p in checkpoint.iterdir()) == ['config.json', 'prune_lab_manifest.json', 'tokenizer.json']
        assert (root / f'seed-{seed}' / 'selection.json').exists()
    # Idempotent.
    assert multiseed.discard_checkpoint_weights(root / 'seed-5') == json.loads((root / 'seed-5' / 'checkpoints-deleted.json').read_text())
    summary = multiseed.aggregate_run(root, samples=30)
    assert summary['seeds'] == [5, 6] and set(summary['variants']) == {'original', 'uniform-10', 'per-layer-01'}
    # The test can no longer be opened, and nothing is finalized.
    with pytest.raises(ValueError, match='deleted before the test'):
        multiseed.run_seeds(base_config(), root, [5, 6], open_test=True)
    assert stages(fake, 'finalize') == []
    assert not json.loads((root / 'seed-5' / 'state.json').read_text())['test_started']


def test_discard_refuses_changed_or_unexplained_missing_weights(fake, tmp_path):
    root = tmp_path / 'ms'
    multiseed.run_seeds(base_config(), root, [5])
    seed_dir = root / 'seed-5'
    weights = seed_dir / 'candidates' / 'per-layer-01' / 'checkpoint' / 'model.safetensors'
    weights.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='changed before deletion'):
        multiseed.discard_checkpoint_weights(seed_dir)
    assert weights.exists()
    for path in (seed_dir / 'candidates' / 'per-layer-01' / 'checkpoint').glob('*.safetensors'):
        path.unlink()
    assert not (seed_dir / 'checkpoints-deleted.json').exists() or 'per-layer-01' not in json.loads(
        (seed_dir / 'checkpoints-deleted.json').read_text())
    with pytest.raises(ValueError, match='missing without a deletion record'):
        multiseed.discard_checkpoint_weights(seed_dir)


def test_open_test_finalizes_every_seed_once_and_test_aggregate_requires_it(fake, tmp_path):
    root = tmp_path / 'ms'
    multiseed.run_seeds(base_config(), root, [5, 6])
    with pytest.raises(ValueError, match='No completed seeds'):
        multiseed.aggregate_run(root, split='test', samples=10)
    assert not (root / 'multi-seed-summary-test.json').exists()
    multiseed.run_seeds(base_config(), root, [5, 6], open_test=True, discard_checkpoints=True)
    assert stages(fake, 'finalize') == [5, 6]
    summary = multiseed.aggregate_run(root, split='test', samples=10)
    assert summary['seeds'] == [5, 6] and (root / 'multi-seed-report-test.md').exists()
    assert all(v['metrics']['token_f1']['mean'] == pytest.approx(30) for v in summary['variants'].values())


def test_open_test_refuses_before_opening_any_seed_if_one_cannot_be_finalized(fake, tmp_path):
    root = tmp_path / 'ms'
    multiseed.run_seeds(base_config(), root, [5])
    multiseed.run_seeds(base_config(), root, [6], discard_checkpoints=True)
    with pytest.raises(ValueError, match='deleted before the test'):
        multiseed.run_seeds(base_config(), root, [5, 6], open_test=True)
    assert stages(fake, 'finalize') == []


def test_cli_seed_derivation_and_validation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    captured = {}
    monkeypatch.setattr(multiseed, 'run_seeds', lambda config, root, seeds, **kw: captured.update(seeds=seeds, kw=kw))
    monkeypatch.setattr(multiseed, 'aggregate_run', lambda *a, **k: {'seeds': captured['seeds']})
    config_path = tmp_path / 'c.json'
    config_path.write_text(json.dumps({'recovery': {'seed': 40}, 'device': 'cpu'}))
    cli.main(['multi-seed', 'run', '--config', str(config_path), '--output', str(tmp_path / 'o'), '--num-seeds', '3'])
    assert captured['seeds'] == [40, 41, 42] and captured['kw'] == {'open_test': False, 'discard_checkpoints': False}
    cli.main(['multi-seed', 'run', '--config', str(config_path), '--output', str(tmp_path / 'o'), '--seeds', '9', '2'])
    assert captured['seeds'] == [9, 2]
    with pytest.raises(SystemExit):
        cli.main(['multi-seed', 'run', '--config', str(config_path), '--output', str(tmp_path / 'o'), '--num-seeds', '0'])
