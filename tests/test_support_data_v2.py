"""Minimal checks for the optional v2 preparation controls; defaults must stay v1."""
import pytest

from prune_lab import support_data as support
from test_support_data import TinyTokenizer, fixture_archive

V2_KEYS = ('dialogue_exclusion',)
V2_PREPARATION_KEYS = ('per_family_cap_effective', 'domain_quotas', 'family_order')


def test_balanced_quotas_water_fill():
    assert support._balanced_quotas(10, {'a': 1, 'b': 9, 'c': 9}) == {'a': 1, 'b': 5, 'c': 4}
    assert support._balanced_quotas(9, {'a': 3, 'b': 3, 'c': 3}) == {'a': 3, 'b': 3, 'c': 3}
    assert support._balanced_quotas(20, {'a': 2, 'b': 3}) == {'a': 2, 'b': 3}


def test_default_manifest_has_no_v2_fields_and_exclusion_removes_dialogues(tmp_path, monkeypatch):
    archive = fixture_archive(tmp_path, monkeypatch)
    counts = {name: 1 for name in support.DEFAULT_COUNTS}
    default = support.prepare_support_dataset(archive, tmp_path / 'default', TinyTokenizer(), counts=counts)
    assert not any(k in default for k in V2_KEYS)
    assert not any(k in default['preparation'] for k in V2_PREPARATION_KEYS)
    used = {row.dialogue_id for row in support.read_support_jsonl(tmp_path / 'default' / 'recovery.jsonl')}
    manifest = support.prepare_support_dataset(
        archive, tmp_path / 'v2', TinyTokenizer(), counts=counts, exclude_dialogue_ids=used,
        exclusion_source='recovery of default', family_caps={'dev': 3}, domain_quotas={'recovery': 'balanced'},
        family_order={'dev': 'random'})
    splits = support.load_support_splits({'support_data_dir': str(tmp_path / 'v2'), 'counts': counts})
    assert not used & {row.dialogue_id for rows in splits.values() for row in rows}
    assert manifest['dialogue_exclusion']['eligible_dialogues_excluded'] == len(used)
    assert manifest['preparation']['per_family_cap_effective']['dev'] == 3
    assert manifest['preparation']['family_order']['dev'] == 'random'
    assert manifest['preparation']['domain_quotas']['recovery'] == 'balanced'
    with pytest.raises(ValueError, match='family_caps'):
        support.prepare_support_dataset(archive, tmp_path / 'bad', TinyTokenizer(), counts=counts, family_caps={'dev': 0})
    with pytest.raises(ValueError, match='domain_quotas'):
        support.prepare_support_dataset(archive, tmp_path / 'bad', TinyTokenizer(), counts=counts, domain_quotas='uneven')
