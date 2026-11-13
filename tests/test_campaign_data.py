import json
from dataclasses import asdict

import pytest

from prune_lab.data import (load_campaign_splits, semantic_problem_key,
                            dataset_manifest, synthetic_splits)


def test_campaign_default_counts_disjoint_and_reproducible(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    splits = load_campaign_splits({})
    assert {name: len(rows) for name, rows in splits.items()} == {'recovery': 1000, 'calibration': 150, 'dev': 200, 'test': 300}
    keys = [ex.key for rows in splits.values() for ex in rows]
    assert len(keys) == len(set(keys))
    assert dataset_manifest(splits) == dataset_manifest(load_campaign_splits({}))
    for rows in splits.values():
        operation_counts = [sum(ex.key.startswith(op + ':') for ex in rows) for op in ('+', '-', '*')]
        assert max(operation_counts) - min(operation_counts) <= 6
        assert any(ex.prompt.startswith('Calculate') for ex in rows)
        assert any(not ex.prompt.startswith('Calculate') for ex in rows)
    test_keys = {ex.key for ex in splits['test']}
    assert not test_keys.intersection(ex.key for name in ('recovery', 'calibration', 'dev') for ex in splits[name])


def test_commutative_semantic_key():
    assert semantic_problem_key('+', 4, -2) == semantic_problem_key('+', -2, 4)
    assert semantic_problem_key('*', 4, -2) == semantic_problem_key('*', -2, 4)
    assert semantic_problem_key('-', 4, -2) != semantic_problem_key('-', -2, 4)


def test_excludes_all_legacy_and_adaptive_splits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    baseline = load_campaign_splits({})
    adaptive = tmp_path / 'runs' / 'adaptive'
    adaptive.mkdir(parents=True)
    excluded = baseline['recovery'][:3] + baseline['dev'][:3] + baseline['test'][:3]
    with (adaptive / 'calibration.jsonl').open('w') as stream:
        for ex in excluded:
            stream.write(json.dumps(asdict(ex)) + '\n')
    splits = load_campaign_splits({})
    keys = {ex.key for rows in splits.values() for ex in rows}
    assert not keys.intersection(ex.key for ex in excluded)
    old_keys = {semantic_problem_key(*ex.key.split(':')) for rows in synthetic_splits(seed=17).values() for ex in rows}
    assert not keys.intersection(old_keys)


@pytest.mark.parametrize('count', [0, -1, True, 1.5])
def test_campaign_counts_strict(count):
    with pytest.raises(ValueError, match='positive integers'):
        load_campaign_splits({'counts': {'test': count}})
