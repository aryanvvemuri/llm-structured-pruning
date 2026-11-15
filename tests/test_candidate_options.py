"""Method-neutral campaign infrastructure: per-target uniform candidates, train_candidates, stable locks."""
import json

import pytest

from prune_lab import campaign


def config_with(**extra):
    config = campaign.read_campaign_config('configs/support-v2-longer.json')
    config.update(extra)
    return config


def test_defaults_unchanged():
    config = campaign.read_campaign_config('configs/support-v2-longer.json')
    assert [s['id'] for s in campaign.candidate_specs(config)] == ['per-layer-01', 'per-layer-02', 'unpruned-recovery', 'uniform-10']


def test_uniform_per_target_and_train_candidates():
    config = config_with(targets=[.10, .20, .25], uniform_per_target=True,
                         train_candidates=['unpruned-recovery', 'uniform-20', 'uniform-25', 'per-layer-02'])
    campaign.validate_config(config)
    specs = {s['id']: s for s in campaign.candidate_specs(config)}
    assert list(specs) == ['per-layer-02', 'unpruned-recovery', 'uniform-20', 'uniform-25']
    assert specs['uniform-25'] == {'id': 'uniform-25', 'target': .25, 'uniform': True, 'control': True, 'selection_eligible': True}
    every = campaign.candidate_specs(config_with(targets=[.10, .20], uniform_per_target=True))
    assert [s['id'] for s in every] == ['per-layer-01', 'per-layer-02', 'unpruned-recovery', 'uniform-10', 'uniform-20']


@pytest.mark.parametrize('extra, message', [
    ({'uniform_per_target': 'yes'}, 'uniform_per_target'),
    ({'uniform_per_target': True, 'include_controls': False, 'selection_reference': 'original'}, 'include_controls'),
    ({'train_candidates': ['per-layer-02']}, 'selection reference'),
    ({'train_candidates': ['unpruned-recovery', 'uniform-20']}, 'not produced'),
    ({'train_candidates': []}, 'nonempty'),
    ({'targets': [.20, .204], 'uniform_per_target': True}, 'collide'),
])
def test_validation(extra, message):
    with pytest.raises(ValueError, match=message):
        campaign.validate_config(config_with(**extra))


def test_fingerprint_round_trip_stable_and_unchanged_without_int_keys():
    import hashlib
    legacy = lambda v: hashlib.sha256(json.dumps(v, sort_keys=True, allow_nan=False).encode()).hexdigest()
    value = {'layers': {i: [i, 0.1 * i] for i in range(12)}}
    assert campaign._fingerprint(value) == campaign._fingerprint(json.loads(json.dumps(value)))
    plain = {'b': [1, 2.5, None, True, {'z': 1e-17, 'a': 'é'}], 'a': (3, 4)}
    assert campaign._fingerprint(plain) == legacy(plain)
