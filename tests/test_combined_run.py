"""The combined allocator + iterative run trains each candidate exactly as the two separate runs would."""
import json
from pathlib import Path

from prune_lab import campaign
from test_multiseed import tiny_support_config

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'


def load(name):
    return campaign.read_campaign_config(CONFIGS / name)


def test_combined_config_specs_and_settings_equal_the_separate_configs():
    combined, allocator, iterative = (load(n) for n in ('support-v2-allocator-iterative.json', 'support-v2-allocator.json',
                                                        'support-v2-iterative.json'))
    campaign.validate_config(combined)
    specs = {s['id']: s for s in campaign.candidate_specs(combined)}
    assert set(specs) == {'unpruned-recovery', 'uniform-20', 'per-layer-02', 'curve-20', 'global-std-20', 'iter-20'}
    for separate in (allocator, iterative):
        for spec in campaign.candidate_specs(separate):
            assert specs[spec['id']] == spec  # same target, method/kind, stages and steps
        same = {k for k in set(combined) | set(separate) if k not in ('train_candidates', 'allocation', 'iterative')}
        assert {k: combined.get(k) for k in same} == {k: separate.get(k) for k in same}  # recipe, data, seeds, eval
    assert combined['allocation'] == allocator['allocation'] and combined['iterative'] == iterative['iterative']


def variant(base, train, allocation=True, iterative=True):
    config = json.loads(json.dumps(base))
    config.update(targets=[.10, .20], uniform_per_target=True, train_candidates=train)
    if allocation:
        config['allocation'] = {'methods': ['legacy', 'curve', 'global-standardized']}
    if iterative:
        config['iterative'] = [{'id': 'iter-20', 'stages': [.10, .20], 'steps': [1, 1]}]
    config['recovery'].update(max_steps=2, checkpoints=[1, 2], gradient_accumulation=1)
    campaign.validate_config(config)
    return config


def fit(config, output):
    campaign.run_campaign(config, output, 'preflight')
    campaign.run_campaign(config, output, 'fit')
    allocation = {c['id']: c['keep_indices'] for c in json.loads((output / 'allocation.json').read_text())['candidates']}
    return {c['id']: c for c in campaign._selection(output)['candidates']}, allocation


def weights(record):
    return [p.read_bytes() for p in sorted(Path(record['checkpoint']).glob('*.safetensors'))]


def test_sharing_a_run_and_training_order_do_not_change_any_candidate(tmp_path, monkeypatch):
    allocator = ['unpruned-recovery', 'uniform-20', 'per-layer-02', 'curve-20', 'global-std-20']
    iterative = ['unpruned-recovery', 'uniform-20', 'iter-20']
    base = tiny_support_config(tmp_path, monkeypatch)
    combined, combined_masks = fit(variant(base, allocator + ['iter-20']), tmp_path / 'runs' / 'combined')
    alone_a, masks_a = fit(variant(base, allocator, iterative=False), tmp_path / 'runs' / 'allocator')
    alone_i, masks_i = fit(variant(base, iterative, allocation=False), tmp_path / 'runs' / 'iterative')
    # The separate runs train candidates in different orders and subsets; every candidate's mask,
    # weights and dev predictions are nevertheless identical (fresh load + seeded recovery per candidate).
    for alone, masks in ((alone_a, masks_a), (alone_i, masks_i)):
        for name, record in alone.items():
            assert masks[name] == combined_masks[name], name
            assert weights(record) == weights(combined[name]), name
            assert [r['prediction'] for r in record['dev']['predictions']] == \
                [r['prediction'] for r in combined[name]['dev']['predictions']], name
            assert record['recovery']['train_losses'] == combined[name]['recovery']['train_losses'], name
