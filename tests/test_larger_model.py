"""Improvement 5: Qwen2.5-1.5B configuration, candidate sizes and cross-model comparison opt-in."""
import json
import runpy
import sys

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from prune_lab import campaign, multiseed
from prune_lab.allocation import allocate_neurons
from prune_lab.modeling import count_parameters, mlp_layers
from test_multiseed import tiny_support_config, write_seed

from pathlib import Path
SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'preflight_model.py'
QWEN_1P5B = dict(vocab_size=151936, hidden_size=1536, intermediate_size=8960, num_hidden_layers=28,
                 num_attention_heads=12, num_key_value_heads=2, tie_word_embeddings=True)


def test_1p5b_candidate_sizes_on_meta_device():
    with torch.device('meta'):
        model = Qwen2ForCausalLM(Qwen2Config(**QWEN_1P5B))
    total = count_parameters(model)
    assert total == 1_543_714_304
    per_neuron, layers, width = 3 * 1536, 28, 8960
    assert sum(p.numel() for m in mlp_layers(model) for p in m.parameters()) == layers * width * per_neuron == 1_156_055_040
    floor = width // 4  # 25% kept, 64-aligned
    assert layers * (width - floor) * per_neuron == 867_041_280  # true maximum removable (56.17%)
    scores = {i: torch.arange(width).float() for i in range(layers)}
    sizes = {}
    for target in (0.25, 0.50):
        keep = allocate_neurons(model, scores, {i: 1.0 for i in range(layers)}, target, 64, 0.25, uniform=True)
        kept = {len(v) for v in keep.values()}
        assert len(kept) == 1 and min(kept) >= floor
        sizes[target] = total - layers * (width - kept.pop()) * per_neuron
    assert sizes == {0.25: 1_155_610_112, 0.50: 775_763_456}
    with pytest.raises(ValueError, match='Infeasible'):
        allocate_neurons(model, scores, {i: 1.0 for i in range(layers)}, 0.57, 64, 0.25, uniform=True)


def test_1p5b_config_is_longer_recipe_with_new_model_and_targets():
    config = campaign.read_campaign_config('configs/support-v2-1p5b.json')
    longer = campaign.read_campaign_config('configs/support-v2-longer.json')
    campaign.validate_config(config)
    assert config['model'] == 'Qwen/Qwen2.5-1.5B-Instruct' and config['targets'] == [0.25, 0.50]
    assert config['recovery'] == longer['recovery']
    assert {k: v for k, v in config.items() if k not in ('model', 'targets', 'uniform_per_target', 'train_candidates')} == \
        {k: v for k, v in longer.items() if k not in ('model', 'targets')}
    assert [s['id'] for s in campaign.candidate_specs(config)] == ['unpruned-recovery', 'uniform-25', 'uniform-50']
    assert all('method' not in s for s in campaign.candidate_specs(config))  # no allocator dependence


def model_run(root, model, seeds=(1, 2, 3)):
    keys = [f'k{i:02d}' for i in range(20)]
    for seed in seeds:
        directory = write_seed(root, seed, keys)
        config = json.loads((directory / 'config.json').read_text())
        (directory / 'config.json').write_text(json.dumps(dict(config, model=model)))


def test_cross_model_compare_requires_explicit_opt_in_and_is_unpaired(tmp_path):
    model_run(tmp_path / 'small', 'Qwen/Qwen2.5-0.5B-Instruct')
    model_run(tmp_path / 'large', 'Qwen/Qwen2.5-1.5B-Instruct')
    with pytest.raises(ValueError, match='allow-different-models'):
        multiseed.compare_runs(tmp_path / 'small', tmp_path / 'large', samples=20)
    result = multiseed.compare_runs(tmp_path / 'small', tmp_path / 'large', pairs=[('per-layer-01', 'per-layer-01')],
                                    samples=20, allow_different_models=True)
    assert result['base_models'] == {'baseline': 'Qwen/Qwen2.5-0.5B-Instruct', 'candidate': 'Qwen/Qwen2.5-1.5B-Instruct',
                                     'identical': False}
    assert result['comparisons'][0]['token_f1']['seed_pairing'] == 'independent'
    assert 'Different base models' in multiseed.render_comparison(result)
    same = multiseed.compare_runs(tmp_path / 'small', tmp_path / 'small', samples=20)
    assert same['base_models']['identical'] and same['comparisons'][0]['token_f1']['seed_pairing'] == 'paired'


def test_mixed_models_within_one_run_are_refused(tmp_path):
    model_run(tmp_path / 'mixed', 'Qwen/Qwen2.5-0.5B-Instruct', seeds=(1, 2))
    model_run(tmp_path / 'mixed', 'Qwen/Qwen2.5-1.5B-Instruct', seeds=(3,))
    with pytest.raises(ValueError, match='different base models'):
        multiseed.collect_run(tmp_path / 'mixed')


def test_preflight_script_reports_timing_without_dev_scores(tmp_path, monkeypatch):
    config = tiny_support_config(tmp_path, monkeypatch)
    path = tmp_path / 'tiny.json'
    path.write_text(json.dumps(config))
    seen = []
    from prune_lab import support_evaluation
    original = support_evaluation.evaluate_support
    monkeypatch.setattr(support_evaluation, 'evaluate_support', lambda m, t, ex, *a, **k: seen.extend(e.key for e in ex) or original(m, t, ex, *a, **k))
    out = tmp_path / 'preflight.json'
    monkeypatch.setattr(sys, 'argv', ['preflight_model.py', '--config', str(path), '--output', str(out),
                                      '--updates', '2', '--replies', '1', '--device', 'cpu', '--prune-uniform', '0.15'])
    runpy.run_path(str(SCRIPT), run_name='__main__')
    report = json.loads(out.read_text())
    recovery = {json.loads(l)['key'] for l in (tmp_path / 'prepared' / 'recovery.jsonl').read_text().splitlines()}
    assert seen and set(seen) <= recovery
    assert report['unpruned']['seconds_per_update'] > 0 and report['estimate']['seeds'] in (3, 5)
    assert report['uniform-0.15']['parameters'] < report['unpruned']['parameters']
    with pytest.raises(SystemExit):
        monkeypatch.setattr(sys, 'argv', ['preflight_model.py', '--config', str(path), '--output', 'runs/x.json'])
        runpy.run_path(str(SCRIPT), run_name='__main__')
