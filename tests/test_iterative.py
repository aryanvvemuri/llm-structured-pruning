"""Improvement 7: iterative prune -> recover -> re-score -> prune -> recover candidates (tiny CPU)."""
import json
from pathlib import Path
import random

import pytest
import torch

from prune_lab import campaign, recovery as recovery_module, scoring
from test_multiseed import tiny_support_config


def iterative_config(tmp_path, monkeypatch):
    config = tiny_support_config(tmp_path, monkeypatch)
    config.update(targets=[.10, .20], uniform_per_target=True,
                  iterative=[{'id': 'iter-20', 'stages': [.10, .20], 'steps': [1, 1]}],
                  train_candidates=['unpruned-recovery', 'uniform-20', 'iter-20'])
    config['recovery'].update(max_steps=2, checkpoints=[1, 2], gradient_accumulation=1)
    campaign.validate_config(config)
    return config


def fit(config, output):
    campaign.run_campaign(config, output, 'preflight')
    campaign.run_campaign(config, output, 'fit')
    return {c['id']: c for c in campaign._selection(output)['candidates']}


def test_iterative_candidate_stages_widths_data_and_records(tmp_path, monkeypatch):
    config = iterative_config(tmp_path, monkeypatch)
    calls = []
    real_score = scoring.score_neurons

    def spy_score(model, batches, method='taylor'):
        widths = [l.mlp.down_proj.in_features for l in model.model.layers]
        calls.append({'widths': widths, 'gate0': model.model.layers[0].mlp.gate_proj.weight.detach().clone()})
        return real_score(model, batches, method)
    monkeypatch.setattr(scoring, 'score_neurons', spy_score)
    seen = []
    real_recover = recovery_module.recover_model
    monkeypatch.setattr(recovery_module, 'recover_model', lambda m, b, e, c, **k: seen.append((c, [x['input_ids'].tolist() for x in b])) or real_recover(m, b, e, c, **k))
    output = tmp_path / 'runs' / 'iter'
    records = fit(config, output)
    allocation = json.loads((output / 'allocation.json').read_text())
    spec = next(c for c in allocation['candidates'] if c['id'] == 'iter-20')
    oneshot = next(c for c in allocation['candidates'] if c['id'] == 'uniform-20')
    assert spec['oneshot_keep_indices'] == oneshot['keep_indices']
    record = records['iter-20']
    info = record['iterative']
    assert record['layer_widths'] == records['uniform-20']['layer_widths'] == info['oneshot_widths'] == info['final_widths']
    assert record['parameters'] == records['uniform-20']['parameters'] and info['total_updates'] == 2
    # Stage 2 Taylor scores were computed on the merged stage-1 model (stage-1 widths, trained weights).
    stage_calls = [c for c in calls if c['widths'] == [spec['stage_widths'][0]] * 2]
    assert len(stage_calls) == 1
    assert spec['stage_widths'][0] > spec['stage_widths'][1]
    # Each stage: own schedule length, derived seed, no reshuffle; together one pass in one-shot order.
    stage_configs = [c for c, _ in seen if c.get('shuffle_batches') is False]
    assert [c['max_steps'] for c in stage_configs] == [1, 1] and len({c['seed'] for c in stage_configs}) == 2
    order = list(range(2))
    random.Random(config['recovery']['seed']).shuffle(order)
    slices = [s['data_slice'] for s in info['stages']]
    assert slices == [[0, 1], [1, 2]]
    stage_inputs = [b for c, b in seen if c.get('shuffle_batches') is False]
    oneshot_batches = [b for c, b in seen if c['max_steps'] == 2][0]  # given in data order; recover_model shuffles by seed
    assert stage_inputs[0] + stage_inputs[1] == [oneshot_batches[i] for i in order]  # one-shot's epoch order, continued
    assert [s['best_step'] in (0, 1) for s in info['stages']] == [True, True]
    assert info['kept_neurons_differing_from_oneshot'] == sum(info['per_layer_differing'].values()) >= 0
    assert len(info['stage_keep_indices']) == 2 and info['stages'][0]['scores_sha256'] == spec['stage1_scores_sha256']
    assert info['stages'][1]['scores_sha256'] != spec['stage1_scores_sha256']
    assert not list((output / 'candidates' / 'iter-20').glob('stage-*/checkpoint/*.safetensors'))
    assert record['export_reload_verified'] and 'merged_dev_before_export' in record


def test_resume_between_stages_matches_uninterrupted_run(tmp_path, monkeypatch):
    config = iterative_config(tmp_path, monkeypatch)
    full = fit(config, tmp_path / 'runs' / 'full')['iter-20']
    calls = []
    real_recover = recovery_module.recover_model

    def crash_in_stage_two(model, batches, evaluate, cfg, **kwargs):
        calls.append(cfg.get('shuffle_batches'))
        if cfg.get('shuffle_batches') is False and calls.count(False) == 2:
            raise RuntimeError('interrupted during stage 2')
        return real_recover(model, batches, evaluate, cfg, **kwargs)
    monkeypatch.setattr(recovery_module, 'recover_model', crash_in_stage_two)
    output = tmp_path / 'runs' / 'resumed'
    campaign.run_campaign(config, output, 'preflight')
    campaign.run_campaign(config, output, 'fit')  # the candidate fails; the others complete
    failed = json.loads((output / 'candidates' / 'iter-20' / 'result.json').read_text())
    assert failed['status'] == 'failed' and (output / 'candidates' / 'iter-20' / 'stage-1' / 'stage.json').exists()
    (output / 'selection.json').unlink(missing_ok=True)
    (output / 'selection-lock.json').unlink(missing_ok=True)
    monkeypatch.setattr(recovery_module, 'recover_model', lambda m, b, e, c, **k: calls.append(('resume', c.get('seed'))) or real_recover(m, b, e, c, **k))
    campaign.run_campaign(config, output, 'fit')
    resumed = {c['id']: c for c in campaign._selection(output)['candidates']}['iter-20']
    assert [c for c in calls if isinstance(c, tuple)] == [('resume', full['iterative']['stages'][1]['seed'])]  # stage 1 not redone
    weights = lambda r: [f.read_bytes() for f in sorted(Path(r['checkpoint']).glob('*.safetensors'))]
    assert weights(resumed) == weights(full)  # manifests differ only in timing fields
    assert [x['prediction'] for x in resumed['dev']['predictions']] == [x['prediction'] for x in full['dev']['predictions']]
    assert resumed['iterative']['stage_keep_indices'] == full['iterative']['stage_keep_indices']


@pytest.mark.parametrize('item, message', [
    ({'id': 'it', 'stages': [.2, .1], 'steps': [225, 225]}, 'strictly increasing'),
    ({'id': 'it', 'stages': [.1, .2], 'steps': [200, 200]}, 'equal budget'),
    ({'id': 'it', 'stages': [.1, .2], 'steps': [440, 10]}, 'warmup'),
    ({'id': 'it', 'stages': [.1], 'steps': [450]}, 'at least two'),
    ({'id': 'it', 'stages': [.1, .2], 'steps': [225, 225], 'mode': 'x'}, 'needs id'),
])
def test_iterative_validation(item, message):
    config = campaign.read_campaign_config('configs/support-v2-longer.json')
    config['iterative'] = [item]
    with pytest.raises(ValueError, match=message):
        campaign.validate_config(config)


def test_config_and_defaults():
    config = campaign.read_campaign_config('configs/support-v2-iterative.json')
    longer = campaign.read_campaign_config('configs/support-v2-longer.json')
    campaign.validate_config(config)
    assert config['recovery'] == longer['recovery'] and config['targets'] == longer['targets']
    assert [s['id'] for s in campaign.candidate_specs(config)] == ['unpruned-recovery', 'uniform-20', 'iter-20']
    spec = campaign.candidate_specs(config)[-1]
    assert [campaign.stage_checkpoints(spec, config['recovery'], k) for k in (0, 1)] == [[75, 150, 225]] * 2
    assert all('kind' not in s for s in campaign.candidate_specs(longer))
