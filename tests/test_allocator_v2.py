"""Improvement 1: curve allocator (1a), global-standardized allocator (1b), per-target uniform."""
import json

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from prune_lab import allocation as alloc, campaign
from prune_lab.modeling import count_parameters


def tiny(layers=2):
    torch.manual_seed(7)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=layers,
                                        num_attention_heads=2, num_key_value_heads=1,
                                        max_position_embeddings=32)).eval().requires_grad_(False)


def removed(keep):
    return [24 - len(v) for _, v in sorted(keep.items())]


def test_curve_processing_and_interpolation():
    assert alloc.process_curve([-0.5, -0.2, 0.1]) == [0.0, 0.0, 0.1]
    assert alloc.process_curve([0.3, 0.1, 0.2]) == [0.3, 0.3, 0.3]
    curve = [0.1, 0.4, 1.0]
    assert alloc.curve_value(curve, (.1, .25, .5), 0) == 0
    assert alloc.curve_value(curve, (.1, .25, .5), .05) == pytest.approx(.05)
    assert alloc.curve_value(curve, (.1, .25, .5), .175) == pytest.approx(.25)
    assert alloc.curve_value(curve, (.1, .25, .5), .5) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        alloc.curve_value(curve, (.1, .25, .5), .6)


def test_curve_prefers_insensitive_layer_and_respects_cap():
    model = tiny()
    scores = {i: torch.arange(24).float() for i in range(2)}
    curves = {0: [0.0, 0.0, 0.0], 1: [1.0, 2.0, 4.0]}
    per_group = 4 * 48  # four neurons x (gate + up + down) parameters
    total = count_parameters(model)
    keep = alloc.allocate_curve(model, scores, curves, 2 * per_group / total, group_size=4)
    assert removed(keep) == [8, 0]
    assert keep[0] == list(range(8, 24))  # lowest Taylor scores removed first
    keep = alloc.allocate_curve(model, scores, curves, 4 * per_group / total, group_size=4)
    assert removed(keep) == [12, 4]  # layer 0 stops at its 50% cap
    with pytest.raises(ValueError, match='Infeasible'):
        alloc.allocate_curve(model, scores, curves, 7 * per_group / total, group_size=4)


def test_zero_signal_layers_are_pruned_evenly():
    model = tiny(3)
    scores = {i: torch.arange(24).float() for i in range(3)}
    curves = {i: [-0.01, 0.0, 0.0] for i in range(3)}  # noise: no evidence of harm anywhere
    total = count_parameters(model)
    keep = alloc.allocate_curve(model, scores, curves, 5 * 4 * 48 / total, group_size=4)
    assert sorted(removed(keep)) == [4, 8, 8]


def test_global_standardized_ranks_by_standardized_scores():
    model = tiny()
    total = count_parameters(model)
    scale_free = {0: torch.arange(24).float(), 1: 1000 * torch.arange(24).float()}
    keep = alloc.allocate_global_standardized(model, scale_free, 2 * 4 * 48 / total, group_size=4)
    assert removed(keep) == [4, 4]  # raw scales differ by 1000x; standardized ranks are equal
    skewed = {0: torch.arange(24).float(), 1: torch.arange(24).float() ** 3}
    z0 = alloc.standardized_scores(skewed)[0][:4].mean()
    z1 = alloc.standardized_scores(skewed)[1][:4].mean()
    assert z0 < z1  # hand check: the uniform layer's lowest group is further below its mean
    keep = alloc.allocate_global_standardized(model, skewed, 4 * 48 / total, group_size=4)
    assert removed(keep) == [4, 0]
    full = alloc.allocate_global_standardized(model, skewed, 6 * 4 * 48 / total, group_size=4)
    assert removed(full) == [12, 12]  # both capped at 50%


def config_with(**extra):
    config = campaign.read_campaign_config('configs/support-v2-longer.json')
    config.update(extra)
    return config


def test_candidate_specs_naming_filtering_and_defaults():
    default = campaign.read_campaign_config('configs/support-v2-longer.json')
    assert [s['id'] for s in campaign.candidate_specs(default)] == ['per-layer-01', 'per-layer-02', 'unpruned-recovery', 'uniform-10']
    assert all('method' not in s for s in campaign.candidate_specs(default))
    full = config_with(allocation={'methods': ['legacy', 'curve', 'global-standardized']}, uniform_per_target=True)
    specs = {s['id']: s for s in campaign.candidate_specs(full)}
    assert set(specs) == {'per-layer-01', 'per-layer-02', 'curve-10', 'curve-20', 'global-std-10', 'global-std-20',
                          'unpruned-recovery', 'uniform-10', 'uniform-20'}
    assert specs['uniform-20']['uniform'] and specs['uniform-20']['control'] and specs['curve-20']['method'] == 'curve'
    allocator = campaign.read_campaign_config('configs/support-v2-allocator.json')
    campaign.validate_config(allocator)
    assert [s['id'] for s in campaign.candidate_specs(allocator)] == ['per-layer-02', 'curve-20', 'global-std-20',
                                                                      'unpruned-recovery', 'uniform-20']
    assert {k: v for k, v in allocator.items() if k not in ('allocation', 'uniform_per_target', 'train_candidates')} == default


@pytest.mark.parametrize('extra, message', [
    ({'allocation': {'method': 'flap'}}, 'allocation methods'),
    ({'allocation': {'methods': ['curve'], 'method': 'curve'}}, 'exactly one'),
    ({'allocation': {'methods': ['curve'], 'min_keep_ratio': .25}}, 'removal cap'),
    ({'allocation': {'methods': ['curve'], 'probe_fractions': [.5, .25]}}, 'probe_fractions'),
    ({'train_candidates': ['unpruned-recovery', 'uniform-20']}, 'not produced'),
    ({'train_candidates': ['per-layer-02']}, 'selection reference'),
    ({'uniform_per_target': 'yes'}, 'uniform_per_target'),
])
def test_config_validation(extra, message):
    with pytest.raises(ValueError, match=message):
        campaign.validate_config(config_with(**extra))


def test_tiny_campaign_records_allocators(tmp_path, monkeypatch):
    from test_multiseed import tiny_support_config
    config = tiny_support_config(tmp_path, monkeypatch)
    config.update(targets=[.15], allocation={'methods': ['legacy', 'curve', 'global-standardized']},
                  uniform_per_target=True, train_candidates=['unpruned-recovery', 'uniform-15', 'per-layer-01',
                                                              'curve-15', 'global-std-15'])
    campaign.validate_config(config)
    output = tmp_path / 'runs' / 'alloc'
    campaign.run_campaign(config, output, 'preflight')
    campaign.run_campaign(config, output, 'fit')
    record = json.loads((output / 'allocation.json').read_text())
    ids = [c['id'] for c in record['candidates']]
    assert ids == ['per-layer-01', 'curve-15', 'global-std-15', 'unpruned-recovery', 'uniform-15']
    probes = record['curve_probes']
    assert probes['fractions'] == [.1, .25, .5] and probes['examples'] == 1 and set(probes['raw']) == {'0', '1'}
    assert all(c == alloc.process_curve(r) for c, r in zip(probes['curves'].values(), probes['raw'].values()))
    assert set(record['removal_fractions']) == set(ids) and set(record['calibration_loss_after_pruning']) == set(ids)
    for name in ('curve-15', 'global-std-15'):
        assert max(record['removal_fractions'][name]) <= .5
    assert all(c['status'] == 'complete' for c in campaign._selection(output)['candidates'])


def test_masks_recompute_from_recorded_scores_and_curves(tmp_path, monkeypatch):
    from test_multiseed import tiny_support_config
    config = tiny_support_config(tmp_path, monkeypatch)
    config.update(targets=[.15], allocation={'methods': ['legacy', 'curve', 'global-standardized']},
                  train_candidates=['unpruned-recovery', 'curve-15', 'global-std-15'])
    output = tmp_path / 'runs' / 'recompute'
    campaign.run_campaign(config, output, 'preflight')
    from prune_lab import recovery
    monkeypatch.setattr(recovery, 'recover_model', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('stop after allocation')))
    with pytest.raises(ValueError, match='recovery reference failed'):  # Only the allocation stage is needed here.
        campaign.run_campaign(config, output, 'fit')
    record = json.loads((output / 'allocation.json').read_text())
    scores = {int(i): torch.tensor(v) for i, v in record['taylor_scores'].items()}
    raw = {int(i): v for i, v in record['curve_probes']['raw'].items()}
    model, _ = campaign._load(config)
    specs = {c['id']: {int(k): v for k, v in c['keep_indices'].items()} for c in record['candidates']}
    assert specs['curve-15'] == alloc.allocate_curve(model, scores, raw, .15, config['group_size'])
    assert specs['global-std-15'] == alloc.allocate_global_standardized(model, scores, .15, config['group_size'])
