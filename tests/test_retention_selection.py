"""Relative retention uses frozen development selection and a fresh test floor."""
import json
from types import SimpleNamespace

import pytest

from prune_lab import campaign
from prune_lab.reporting import write_json


def metrics(value, name='accuracy'):
    if name == 'accuracy':
        return {'accuracy': value, 'answer_token_loss': 1.0,
                'predictions': [{'key': str(i), 'correct': i < round(value * 100)} for i in range(100)]}
    return {'selection_score': value, 'score_name': name, 'answer_token_loss': 1.0,
            'predictions': [{'key': 'a', 'group_id': 'doc', 'score': value}]}


def candidate(name, size, value, metric='accuracy'):
    return {'id': name, 'parameters': size, 'status': 'complete', 'control': False,
            'checkpoint': '/checkpoints/' + name, 'dev': metrics(value, metric)}


@pytest.mark.parametrize('metric', ['accuracy', 'token_f1'])
def test_relative_selects_smallest_tested_passing_model(metric):
    baseline = {'parameters': 1000, 'dev': metrics(.8, metric)}
    rows = [candidate('large', 900, .8, metric), candidate('pass', 800, .76, metric),
            candidate('fail', 700, .759, metric), candidate('wrong_metric', 600, .99, 'rouge_l')]
    tier = campaign.select_tiers(baseline, rows, retention_targets=[.95])[0]
    assert tier['candidate_id'] == 'pass'
    assert tier['development_floor'] == pytest.approx(.76)
    assert tier['baseline_dev_score'] == .8
    assert tier['min_retained_score'] == .95
    assert tier['mode'] == 'relative_retention'
    assert 'max_drop_pp' not in tier
    assert campaign.select_tiers(baseline, rows, [5])[0]['candidate_id'] == 'fail'


@pytest.mark.parametrize('value', [[], [True], [False], [0], [-.1], [1.01], [float('nan')],
                                   [float('inf')], ['.95'], .95])
def test_invalid_retention_configuration(value):
    config = campaign.read_campaign_config()
    config.update(tiers_pp=None, retention_targets=value)
    with pytest.raises(ValueError, match='retention_targets'):
        campaign.validate_config(config)


def test_modes_and_zero_development_reference():
    config = campaign.read_campaign_config()
    assert 'retention_targets' not in config
    config['retention_targets'] = [.95]
    with pytest.raises(ValueError, match='only one'):
        campaign.validate_config(config)
    config['tiers_pp'] = None
    campaign.validate_config(config)
    with pytest.raises(ValueError, match='nonzero baseline'):
        campaign.select_tiers({'parameters': 1000, 'dev': metrics(0)}, [], retention_targets=[.95])
    assert campaign.select_tiers({'parameters': 1000, 'dev': metrics(.8)}, [], retention_targets=[1])[0]['status'] == 'no_qualifying_pruned_model'


@pytest.mark.parametrize('metric', ['accuracy', 'token_f1'])
@pytest.mark.parametrize('test_reference,test_candidate,passed,status', [
    (.6, .58, True, 'met'), (.9, .80, False, 'below_threshold'),
    (0, .8, False, 'unassessable_zero_reference')])
def test_finalize_recomputes_floor_without_reselection(tmp_path, monkeypatch, metric,
                                                       test_reference, test_candidate, passed, status):
    chosen = candidate('chosen', 800, .76, metric)
    unused = candidate('unused', 900, .8, metric)
    baseline = {'parameters': 1000, 'dev': metrics(.8, metric)}
    selection = {'baseline': baseline, 'candidates': [chosen, unused],
                 'tiers': campaign.select_tiers(baseline, [chosen, unused], retention_targets=[.95])}
    write_json(tmp_path / 'selection.json', selection)
    lock = {'sha256': campaign._fingerprint(selection)}
    write_json(tmp_path / 'selection-lock.json', lock)
    monkeypatch.setattr(campaign, '_verify_checkpoint', lambda *args: None)
    monkeypatch.setattr(campaign, '_verify_source', lambda *args: None)
    monkeypatch.setattr(campaign, '_provenance', lambda *args: {})
    monkeypatch.setattr(campaign, '_release', lambda: None)
    calls = []

    def load(config, checkpoint):
        calls.append(checkpoint)
        return SimpleNamespace(checkpoint=checkpoint), None

    monkeypatch.setattr(campaign, '_load', load)
    monkeypatch.setattr(campaign, '_evaluate', lambda model, *args: metrics(
        test_reference if model.checkpoint is None else test_candidate, metric))
    state = {'test_started': False, 'dataset': {}}
    result = campaign.finalize({'seed': 17}, tmp_path, state, {'test': [1]})
    tier = result['tiers'][0]
    assert tier['candidate_id'] == 'chosen'
    assert tier['test_reference_score'] == test_reference
    assert tier['test_score'] == test_candidate
    assert tier['test_floor'] == pytest.approx(test_reference * .95)
    assert tier['development_floor'] == pytest.approx(.76)
    assert tier['test_threshold_met'] is passed
    assert tier['test_retention_status'] == status
    assert tier['retained_score_ratio'] == (pytest.approx(test_candidate / test_reference) if test_reference else None)
    assert ('delta_points' if metric == 'token_f1' else 'delta_pp') in tier['test']
    assert calls == [None, chosen['checkpoint']]
    assert json.loads((tmp_path / 'selection-lock.json').read_text()) == lock
    campaign.finalize({'seed': 17}, tmp_path, state, {'test': [1]})
    assert len(calls) == 2
