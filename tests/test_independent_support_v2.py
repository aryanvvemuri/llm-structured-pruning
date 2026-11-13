"""Independent checks of the v2 data-preparation options and of the real v1/v2 artefacts.

Synthetic archives with several domains and multi-dialogue documents exercise exclusions,
family caps, balanced quotas and random family order. Real-file checks use only ids,
hashes and counts (no example text is printed or inspected).
"""
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from prune_lab import support_data as support
from prune_lab.data import dataset_manifest

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / '.cache' / 'datasets' / 'doc2dial' / 'doc2dial_v1.0.1.zip'
SPLITS = ('recovery', 'calibration', 'dev', 'test')


class WordTokenizer:
    eos_token_id = 0
    chat_template = None
    name_or_path = 'word-tokenizer'

    def encode(self, text, add_special_tokens=True):
        return list(range(len(text.split())))


def build_archive(tmp_path, monkeypatch, layout, name='archive.zip'):
    """layout: {domain: [(train_dialogues, test_dialogues), ...] per document}."""
    counter = itertools.count()
    word = lambda prefix: f'{prefix}{next(counter)}q'
    docs, train, test = {}, {}, {}
    for domain, documents in layout.items():
        docs[domain], train[domain], test[domain] = {}, {}, {}
        for d, (n_train, n_test) in enumerate(documents):
            doc_id = f'{domain}-doc{d}'
            text = ' '.join(word('t') for _ in range(25))
            docs[domain][doc_id] = {'title': f'{domain} title {word("h")}', 'doc_text': text,
                                    'spans': {'1': {'start_sec': 0, 'id_sec': 's1', 'text_sec': text}}}
            for source, n in (('train', n_train), ('test', n_test)):
                conversations = []
                for c in range(n):
                    conversations.append({'dial_id': f'{source}-{doc_id}-dial{c}', 'turns': [
                        {'turn_id': 1, 'role': 'user', 'da': 'query_solution',
                         'utterance': ' '.join(word('u') for _ in range(6)), 'references': []},
                        {'turn_id': 2, 'role': 'agent', 'da': 'respond_solution',
                         'utterance': ' '.join(word('a') for _ in range(8)), 'references': [{'sp_id': '1'}]}]})
                if conversations:
                    (train if source == 'train' else test)[domain][doc_id] = conversations
    archive = tmp_path / name
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('doc2dial_doc.json', json.dumps({'doc_data': docs}))
        for source, data in (('train', train), ('validation', {}), ('test', test)):
            bundle.writestr(f'doc2dial_dial_{source}.json', json.dumps({'dial_data': data}))
        bundle.writestr('DATA_README.md', 'synthetic')
    monkeypatch.setattr(support, 'SOURCE_SHA256', hashlib.sha256(archive.read_bytes()).hexdigest())
    return archive


# Three domains; 'short' has few documents/dialogues so equal quotas cannot be met.
LAYOUT = {
    'alpha': [(6, 2), (5, 0), (4, 3), (4, 0), (3, 0), (3, 0), (2, 0), (2, 0)],
    'beta': [(6, 0), (5, 2), (4, 0), (4, 0), (3, 2), (3, 0), (2, 0), (2, 0)],
    'short': [(3, 2), (3, 0), (3, 0), (2, 0), (2, 0), (2, 0), (1, 0), (1, 0)],
}
COUNTS = {'recovery': 20, 'calibration': 6, 'dev': 9, 'test': 6}


def rows_of(directory):
    return {name: support.read_support_jsonl(Path(directory) / f'{name}.jsonl') for name in SPLITS}


def assert_invariants(splits, counts):
    assert {n: len(r) for n, r in splits.items()} == counts
    support.validate_support_splits(splits)
    for field in ('dialogue_id', 'doc_family', 'doc_id', 'prompt_sha256', 'response_sha256', 'key', 'group_id'):
        owners = {}
        for name, rows in splits.items():
            for row in rows:
                assert owners.setdefault(getattr(row, field), name) == name, (field, name)
    for name, rows in splits.items():
        assert len({r.prompt_sha256 for r in rows}) == len(rows)
        assert all((r.source_split == 'test') == (name == 'test') for r in rows)


OPTIONS = [
    {},
    {'family_caps': 2},
    {'family_caps': {'dev': 1, 'recovery': 3}},
    {'domain_quotas': 'balanced'},
    {'family_order': 'random'},
    {'family_order': {'dev': 'random', 'calibration': 'random'}, 'domain_quotas': {'recovery': 'balanced'},
     'family_caps': {'dev': 2}},
]


@pytest.mark.parametrize('options', OPTIONS, ids=[json.dumps(o, sort_keys=True) for o in OPTIONS])
def test_invariants_and_caps_hold_under_each_option(tmp_path, monkeypatch, options):
    archive = build_archive(tmp_path, monkeypatch, LAYOUT)
    counts = dict(COUNTS, recovery=9)
    manifest = support.prepare_support_dataset(archive, tmp_path / 'out', WordTokenizer(), counts=counts, **options)
    splits = rows_of(tmp_path / 'out')
    assert_invariants(splits, counts)
    assert manifest['dataset'] == dataset_manifest(splits)
    caps = options.get('family_caps')
    for name, rows in splits.items():
        cap = caps.get(name) if isinstance(caps, dict) else caps
        cap = cap or support.default_family_cap(counts[name])
        assert max(Counter(r.doc_family for r in rows).values()) <= cap
    repeat = support.prepare_support_dataset(archive, tmp_path / 'again', WordTokenizer(), counts=counts, **options)
    assert repeat == manifest
    for name in SPLITS:
        assert (tmp_path / 'out' / f'{name}.jsonl').read_bytes() == (tmp_path / 'again' / f'{name}.jsonl').read_bytes()


def test_family_cap_one_spreads_dev_over_distinct_families(tmp_path, monkeypatch):
    archive = build_archive(tmp_path, monkeypatch, LAYOUT)
    counts = dict(COUNTS, recovery=10)
    manifest = support.prepare_support_dataset(archive, tmp_path / 'out', WordTokenizer(), counts=counts,
                                               family_caps={'dev': 1})
    dev = rows_of(tmp_path / 'out')['dev']
    assert len({r.doc_family for r in dev}) == len(dev) == counts['dev']
    assert manifest['preparation']['per_family_cap_effective'] == {
        'recovery': 10, 'calibration': 10, 'dev': 1, 'test': 10}


def test_exclusions_remove_dialogues_from_every_split(tmp_path, monkeypatch):
    archive = build_archive(tmp_path, monkeypatch, LAYOUT)
    counts = dict(COUNTS, recovery=10)
    first = rows_of_built(archive, tmp_path / 'first', counts)
    excluded = {rows[0].dialogue_id for name, rows in first.items() if name != 'test'}
    excluded |= {next(r.dialogue_id for r in first['test'] if r.domain == 'alpha'), 'not-a-real-dialogue'}
    manifest = support.prepare_support_dataset(archive, tmp_path / 'excl', WordTokenizer(), counts=counts,
                                               exclude_dialogue_ids=excluded, exclusion_source=[{'path': 'x'}])
    splits = rows_of(tmp_path / 'excl')
    assert_invariants(splits, counts)
    assert not excluded & {r.dialogue_id for rows in splits.values() for r in rows}
    info = manifest['dialogue_exclusion']
    assert info['requested_dialogues'] == 5 and info['eligible_dialogues_excluded'] == 4
    assert info['candidates_excluded'] == 4 and info['excluded_source_splits'] == {'train': 3, 'test': 1}
    assert info['requested_sha256'] == hashlib.sha256('\n'.join(sorted(excluded)).encode()).hexdigest()
    assert manifest['eligible_candidates'] == sum(n for docs in LAYOUT.values() for t in docs for n in t) - 4
    for bad in (set(), {''}, {1}):
        with pytest.raises(ValueError, match='exclude_dialogue_ids'):
            support.prepare_support_dataset(archive, tmp_path / f'bad{len(bad)}{bad}', WordTokenizer(), counts=counts,
                                            exclude_dialogue_ids=bad)


def rows_of_built(archive, output, counts, **options):
    support.prepare_support_dataset(archive, output, WordTokenizer(), counts=counts, **options)
    return rows_of(output)


def test_exclusion_rejects_bare_string(tmp_path, monkeypatch):
    archive = build_archive(tmp_path, monkeypatch, LAYOUT)
    with pytest.raises(ValueError):
        support.prepare_support_dataset(archive, tmp_path / 'o', WordTokenizer(), counts=dict(COUNTS, recovery=10),
                                        exclude_dialogue_ids='train-alpha-doc0-dial0')


def test_balanced_quotas_fill_a_short_domain_where_equal_quotas_fail(tmp_path, monkeypatch):
    archive = build_archive(tmp_path, monkeypatch, LAYOUT)
    counts = dict(COUNTS, recovery=36)
    with pytest.raises(ValueError, match='Insufficient document-disjoint recovery'):
        support.prepare_support_dataset(archive, tmp_path / 'equal', WordTokenizer(), counts=counts)
    manifest = support.prepare_support_dataset(archive, tmp_path / 'bal', WordTokenizer(), counts=counts,
                                               domain_quotas={'recovery': 'balanced'})
    splits = rows_of(tmp_path / 'bal')
    assert_invariants(splits, counts)
    used = {r.doc_family for n, rows in splits.items() if n != 'recovery' for r in rows}
    families = {r.doc_family: r.domain for rows in splits.values() for r in rows}
    domains = Counter(r.domain for r in splits['recovery'])
    # The short domain contributes everything it has left after earlier splits; the others share the rest.
    assert manifest['split_domains']['recovery'] == dict(domains)
    assert domains == {'short': 8, 'alpha': 14, 'beta': 14}  # short exhausted; the rest water-filled
    assert manifest['preparation']['domain_quotas'] == {'recovery': 'balanced', 'calibration': 'equal',
                                                        'dev': 'equal', 'test': 'equal'}
    # Unchanged splits are identical to the equal-quota build with the same smaller recovery count
    # only for splits selected before recovery (test, dev, calibration are selected first).
    reference = rows_of_built(archive, tmp_path / 'ref', dict(counts, recovery=10))
    for name in ('test', 'dev', 'calibration'):
        assert [r.key for r in reference[name]] == [r.key for r in splits[name]]
    assert used and families


def test_balanced_quota_water_fill_unit():
    q = support._balanced_quotas
    assert q(12, {'a': 2, 'b': 10, 'c': 10}) == {'a': 2, 'b': 5, 'c': 5}
    assert q(13, {'a': 2, 'b': 10, 'c': 10}) == {'a': 2, 'b': 6, 'c': 5}
    assert q(9, {'a': 5, 'b': 5, 'c': 5}) == {'a': 3, 'b': 3, 'c': 3}
    assert q(5, {'a': 0, 'b': 9}) == {'a': 0, 'b': 5}
    for count, available in [(7, {'a': 1, 'b': 2, 'c': 30}), (40, {'a': 9, 'b': 9}), (1, {'a': 3, 'b': 3})]:
        quotas = q(count, available)
        assert sum(quotas.values()) == min(count, sum(available.values()))
        assert all(0 <= quotas[d] <= available[d] for d in available)


def test_random_family_order_is_seeded_and_changes_with_seed(tmp_path, monkeypatch):
    archive = build_archive(tmp_path, monkeypatch, LAYOUT)
    counts = dict(COUNTS, recovery=10)
    a = rows_of_built(archive, tmp_path / 'a', counts, family_order='random')
    b = rows_of_built(archive, tmp_path / 'b', counts, family_order='random')
    assert {n: [r.key for r in rows] for n, rows in a.items()} == {n: [r.key for r in rows] for n, rows in b.items()}
    seeds = set()
    for seed in range(5):
        support.prepare_support_dataset(archive, tmp_path / f's{seed}', WordTokenizer(), counts=counts, seed=seed,
                                        family_order='random')
        seeds.add(tuple(sorted(r.doc_family for r in rows_of(tmp_path / f's{seed}')['dev'])))
    assert len(seeds) > 1


def test_explicit_default_options_reproduce_default_bytes(tmp_path, monkeypatch):
    archive = build_archive(tmp_path, monkeypatch, LAYOUT)
    counts = dict(COUNTS, recovery=12)
    support.prepare_support_dataset(archive, tmp_path / 'default', WordTokenizer(), counts=counts)
    explicit = dict(family_caps=None, domain_quotas='equal', family_order='size')
    support.prepare_support_dataset(archive, tmp_path / 'explicit', WordTokenizer(), counts=counts, **explicit)
    support.prepare_support_dataset(archive, tmp_path / 'mapping', WordTokenizer(), counts=counts,
                                    domain_quotas={'dev': 'equal'}, family_order={'test': 'size', 'dev': None})
    for other in ('explicit', 'mapping'):
        for name in [*(f'{s}.jsonl' for s in SPLITS), 'manifest.json']:
            assert (tmp_path / other / name).read_bytes() == (tmp_path / 'default' / name).read_bytes(), (other, name)
    # Setting each split's cap to its default value changes only the manifest, not the data.
    caps = {n: support.default_family_cap(c) for n, c in counts.items()}
    support.prepare_support_dataset(archive, tmp_path / 'caps', WordTokenizer(), counts=counts, family_caps=caps)
    for s in SPLITS:
        assert (tmp_path / 'caps' / f'{s}.jsonl').read_bytes() == (tmp_path / 'default' / f'{s}.jsonl').read_bytes()
    for bad in ({'family_caps': {'devv': 2}}, {'family_caps': 1.5}, {'family_caps': True}, {'family_caps': '3'},
                {'domain_quotas': {'dev': 'uneven'}}, {'family_order': 'shuffled'}, {'family_order': {'x': 'random'}}):
        with pytest.raises(ValueError):
            support.prepare_support_dataset(archive, tmp_path / 'bad', WordTokenizer(), counts=counts, **bad)


# --- real artefacts (ids / hashes / counts only) -------------------------------------------

ID_FIELDS = ('key', 'dialogue_id', 'doc_id', 'doc_family', 'group_id', 'source_split', 'prompt_sha256',
             'response_sha256', 'dialogue_context_sha256', 'source_context_sha256')


def id_rows(path):
    with open(path) as stream:
        return [{k: row[k] for k in ID_FIELDS} for row in map(json.loads, filter(str.strip, stream))]


V2 = ROOT / 'datasets' / 'doc2dial_v2'
V1 = ROOT / 'datasets' / 'doc2dial_v1'
needs_v2 = pytest.mark.skipif(not (V2 / 'manifest.json').exists(), reason='datasets/doc2dial_v2 absent')


@needs_v2
def test_real_v2_structural_invariants():
    manifest = json.loads((V2 / 'manifest.json').read_text())
    rows = {name: id_rows(V2 / f'{name}.jsonl') for name in SPLITS}
    expected = {'recovery': 1800, 'calibration': 150, 'dev': 600, 'test': 224}
    assert {n: len(r) for n, r in rows.items()} == expected == manifest['split_counts']
    assert json.loads((ROOT / 'configs' / 'support-v2.json').read_text())['counts'] == expected
    for name, data in rows.items():
        digest = manifest['dataset'][name]
        assert digest['count'] == len(data)
    for field in ('dialogue_id', 'doc_family', 'group_id', 'doc_id', 'prompt_sha256', 'response_sha256',
                  'dialogue_context_sha256', 'source_context_sha256', 'key'):
        owners = {}
        for name, data in rows.items():
            for row in data:
                assert owners.setdefault(row[field], name) == name, (field, name)
    for name, data in rows.items():
        assert len({r['prompt_sha256'] for r in data}) == len(data)
        assert all(r['group_id'] == r['doc_family'] for r in data)
        assert all((r['source_split'] == 'test') == (name == 'test') for r in data)
        assert len({r['dialogue_id'] for r in data}) == len(data)  # one target per dialogue
    assert max(Counter(r['doc_family'] for r in rows['dev']).values()) <= 15
    assert {n: len({r['doc_family'] for r in d}) for n, d in rows.items()} == manifest['split_document_family_counts']
    assert manifest['near_match_audit']['matched_pairs'] == 0 and manifest['near_match_audit']['exhaustive']
    assert manifest['near_match_audit']['checked_pairs'] == sum(
        len(rows[a]) * len(rows[b]) for a, b in itertools.combinations(SPLITS, 2))
    v1_test = id_rows(V1 / 'test.jsonl')
    v1_dialogues = {r['dialogue_id'] for r in v1_test}
    assert len(v1_dialogues) == 300
    assert not v1_dialogues & {r['dialogue_id'] for d in rows.values() for r in d}
    exclusion = manifest['dialogue_exclusion']
    assert exclusion['requested_dialogues'] == 300 and exclusion['eligible_dialogues_excluded'] == 300
    assert exclusion['requested_sha256'] == hashlib.sha256('\n'.join(sorted(v1_dialogues)).encode()).hexdigest()


@needs_v2
@pytest.mark.xfail(strict=True, reason='Known, documented limitation: v1-test exclusion is by dialogue id only; '
                   'one v2 test example shares a canonical prompt hash with an inspected v1 test example '
                   '(see README).')
def test_real_v2_test_shares_no_prompt_with_inspected_v1_test():
    v1 = {r['prompt_sha256'] for r in id_rows(V1 / 'test.jsonl')}
    v1 |= {r['dialogue_context_sha256'] for r in id_rows(V1 / 'test.jsonl')}
    v2 = id_rows(V2 / 'test.jsonl')
    assert not {r['prompt_sha256'] for r in v2} & v1
    assert not {r['dialogue_context_sha256'] for r in v2} & v1


def test_v1_config_fingerprint_unchanged():
    state_path = ROOT / 'runs' / 'support-v1' / 'state.json'
    if not state_path.exists():
        pytest.skip('runs/support-v1 absent')
    from prune_lab import campaign
    config = campaign.read_campaign_config(ROOT / 'configs' / 'support.json')
    assert 'selection_dev_examples' not in config and 'skip_full_merged_dev' not in config
    campaign.validate_config(config)
    assert campaign._fingerprint(config) == json.loads(state_path.read_text())['config_sha256']


def tokenizer_available():
    return (Path.home() / '.cache' / 'huggingface').exists() or (ROOT / '.cache' / 'huggingface').exists()


@pytest.mark.skipif(not ARCHIVE.exists() or not tokenizer_available(), reason='Doc2Dial archive or tokenizer cache absent')
@pytest.mark.parametrize('version', ['v1', 'v2'])
def test_real_rebuild_is_byte_identical(tmp_path, version):
    args = [sys.executable, str(ROOT / 'scripts' / 'prepare_support_data.py'), '--output', str(tmp_path / version)]
    if version == 'v2':
        args += ['--recovery', '1800', '--calibration', '150', '--dev', '600', '--test', '224',
                 '--exclude-dialogues-from', 'datasets/doc2dial_v1/test.jsonl', '--family-cap', 'dev=15',
                 '--domain-quotas', 'calibration=balanced', '--domain-quotas', 'recovery=balanced',
                 '--family-order', 'dev=random', '--family-order', 'calibration=random']
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=600,
                            env={'HF_HUB_OFFLINE': '1', 'PATH': '/usr/bin:/bin', 'HOME': str(Path.home())})
    assert result.returncode == 0, result.stderr[-2000:]
    reference = ROOT / 'datasets' / f'doc2dial_{version}'
    for name in ['manifest.json', 'DATA_README.md', *(f'{s}.jsonl' for s in SPLITS)]:
        assert (tmp_path / version / name).read_bytes() == (reference / name).read_bytes(), name
