"""Independent report semantics and persisted-fit retention wiring checks."""
import copy
import json

import pytest

from prune_lab import campaign
from prune_lab.reporting import render_campaign_report, write_json


def summary_fixture(task):
    def metric(value):
        if task == 'support':
            return {'selection_score': value, 'token_f1': value, 'score_name': 'token_f1',
                    'rouge_l': value, 'exact_match': 0, 'answer_token_loss': 1}
        return {'accuracy': value, 'correct': round(value * 100), 'count': 100,
                'answer_token_loss': 1}
    def candidate(name, size, score, control=False):
        return {'id': name, 'parameters': size, 'status': 'complete', 'control': control,
                'target': 1 - size / 1000, 'actual_reduction': 1 - size / 1000,
                'dev': metric(score), 'checkpoint': '/unused/' + name,
                'checkpoint_bytes': size * 4, 'parameter_bytes': size * 4,
                'elapsed_seconds': 1, 'layer_widths': [10, 8], 'recovery': {'best_step': 1}}
    rows = [candidate('compact', 800, .76), candidate('unpruned-recovery', 1000, .8, True)]
    tiers = campaign.select_tiers(rows[1], rows, retention_targets=[.95])
    delta = {'delta_points': -3} if task == 'support' else {
        'delta_pp': -3, 'bootstrap_95_percent_interval_pp': [-6, 0],
        'regressions': 3, 'improvements': 0}
    final_tier = dict(tiers[0], test=delta, test_reference_score=.6, test_score=.57,
                      test_floor=.57, retained_score_ratio=.95, test_threshold_met=True,
                      test_retention_status='met')
    return {'config': {'task': task, 'model': 'tiny'}, 'dataset': {},
            'selection': {'reference_id': 'unpruned-recovery', 'baseline': {
                'parameters': 1000, 'dev': metric(.2)}, 'candidates': rows, 'tiers': tiers},
            'tiers': [final_tier], 'final_test': {name: {'metrics': metric(value)} for name, value in [
                ('original', .1), ('unpruned-recovery', .6), ('compact', .57)]}}


@pytest.mark.parametrize('task', ['arithmetic', 'support'])
@pytest.mark.parametrize('case', ['relative', 'zero_reference', 'no_candidate', 'legacy'])
def test_report_relative_zero_missing_and_absolute_compatibility(task, case):
    summary = summary_fixture(task)
    if case == 'zero_reference':
        summary['tiers'][0].update(retained_score_ratio=None, test_reference_score=0,
                                   test_floor=0, test_threshold_met=False,
                                   test_retention_status='unassessable_zero_reference')
    elif case == 'no_candidate':
        for tiers in (summary['tiers'], summary['selection']['tiers']):
            tiers[0].update(candidate_id=None, checkpoint=None, status='no_qualifying_pruned_model')
    elif case == 'legacy':
        for tiers in (summary['tiers'], summary['selection']['tiers']):
            for key in ['mode', 'min_retained_score', 'baseline_dev_score']:
                tiers[0].pop(key, None)
            tiers[0]['max_drop_pp'] = 5
        for key in ['retained_score_ratio', 'test_retention_status', 'test_reference_score', 'test_score', 'test_floor']:
            summary['tiers'][0].pop(key, None)
    before = copy.deepcopy(summary)
    rendered = render_campaign_report(summary)
    assert summary == before  # Rendering cannot mutate frozen evidence.
    assert 'Selection reference: **unpruned-recovery**' in rendered
    if case == 'legacy':
        assert 'At most 5-point drop' in rendered
        assert 'At least 95% of baseline' not in rendered
    else:
        assert 'At least 95% of baseline' in rendered
        assert 'five-point F1 drop' in rendered if task == 'support' else 'five-percentage-point drop' in rendered
    if case == 'relative':
        assert '| 95.00% | Yes |' in rendered
        assert '0.95' in rendered
    elif case == 'zero_reference':
        assert '| — | Unassessable (zero baseline) |' in rendered
        assert 'nan%' not in rendered.lower() and 'inf%' not in rendered.lower()
    elif case == 'no_candidate':
        assert 'No qualifying' in rendered


def test_fit_propagates_retention_configuration_against_recovered_reference(tmp_path, monkeypatch):
    fixture = summary_fixture('support')
    baseline = fixture['selection']['baseline']
    rows = fixture['selection']['candidates']
    write_json(tmp_path / 'preflight.json', {'passed': True})
    write_json(tmp_path / 'baseline.json', baseline)
    write_json(tmp_path / 'allocation.json', {
        'baseline_sha256': campaign._fingerprint(baseline),
        'candidates': [{'id': row['id']} for row in rows]})
    for row in rows:
        write_json(tmp_path / 'candidates' / row['id'] / 'result.json', row)
    monkeypatch.setattr(campaign, '_verify_checkpoint', lambda record: None)
    monkeypatch.setattr(campaign, '_load', lambda *args: pytest.fail('Completed fit should not load a model'))
    config = {'selection_reference': 'unpruned-recovery', 'tiers_pp': None,
              'retention_targets': [.95], 'evaluate_all_candidates': True}
    selected = campaign.fit(config, tmp_path, {'test_started': False}, {})
    tier = selected['tiers'][0]
    assert tier['mode'] == 'relative_retention'
    assert tier['baseline_dev_score'] == .8  # Original .2 is deliberately easier.
    assert tier['development_floor'] == pytest.approx(.76)
    assert tier['candidate_id'] == 'compact'
    assert campaign._selection(tmp_path) == selected
    assert json.loads((tmp_path / 'selection-lock.json').read_text())['sha256'] == campaign._fingerprint(selected)
