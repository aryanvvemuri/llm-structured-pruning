from dataclasses import replace
import hashlib
import json
import zipfile

import pytest

from prune_lab import support_data as support


class TinyTokenizer:
    eos_token_id = 0
    chat_template = None
    name_or_path = 'test-tokenizer'

    def encode(self, text, add_special_tokens=True):
        return list(range(len(text.split())))


def fixture_archive(tmp_path, monkeypatch):
    docs, train, test = {}, {}, {}
    for i in range(8):
        doc_id = f'doc{i}'
        text = f'Policy{i} condition{i} action{i} service{i} eligibility{i} evidence{i}'
        docs[doc_id] = {'title': f'Unique document {i}', 'doc_text': text,
                       'spans': {'1': {'start_sec': 0, 'id_sec': 'section1', 'text_sec': text}}}
        conversation = {'dial_id': f'dialogue{i}', 'turns': [
            {'turn_id': 1, 'role': 'user', 'da': 'query_solution', 'utterance': f'Please explain policy number {i}', 'references': []},
            {'turn_id': 2, 'role': 'agent', 'da': 'respond_solution', 'utterance': f'Unique response {i} with useful helpful information', 'references': [{'sp_id': '1'}]},
            {'turn_id': 3, 'role': 'user', 'da': 'query_solution', 'utterance': 'FUTURE TURN MUST NOT APPEAR', 'references': []}]}
        (test if i == 0 else train)[doc_id] = [conversation]
    archive = tmp_path / 'data.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('doc2dial_doc.json', json.dumps({'doc_data': {'domain': docs}}))
        for name, data in [('train', train), ('validation', {}), ('test', test)]:
            bundle.writestr(f'doc2dial_dial_{name}.json', json.dumps({'dial_data': {'domain': data}}))
        bundle.writestr('DATA_README.md', 'Original attribution')
        bundle.writestr('__MACOSX/doc2dial_doc.json', 'invalid ignored')
    monkeypatch.setattr(support, 'SOURCE_SHA256', hashlib.sha256(archive.read_bytes()).hexdigest())
    return archive


def test_prepare_roundtrip_and_no_target_future_context(tmp_path, monkeypatch):
    archive = fixture_archive(tmp_path, monkeypatch)
    output = tmp_path / 'prepared'
    counts = {name: 1 for name in support.DEFAULT_COUNTS}
    manifest = support.prepare_support_dataset(archive, output, TinyTokenizer(), counts=counts)
    splits = support.load_support_splits({'support_data_dir': str(output), 'counts': counts})
    assert all(n == 0 for n in manifest['group_overlaps'].values())
    assert (output / 'DATA_README.md').read_text() == 'Original attribution'
    for name, rows in splits.items():
        row = rows[0]
        assert row.answer not in row.prompt
        assert 'FUTURE TURN' not in row.prompt
        assert row.history_turn_ids == [1]
        assert row.source_section_ids == ['section1']
        assert row.group_id == row.doc_family
        assert row.source_split == ('test' if name == 'test' else 'train')
    repeat = support.prepare_support_dataset(archive, tmp_path / 'repeat', TinyTokenizer(), counts=counts)
    assert manifest['dataset'] == repeat['dataset']


def test_changed_prepared_data_is_rejected(tmp_path, monkeypatch):
    archive = fixture_archive(tmp_path, monkeypatch)
    output = tmp_path / 'prepared'
    support.prepare_support_dataset(archive, output, TinyTokenizer(), counts={n: 1 for n in support.DEFAULT_COUNTS})
    rows = support.read_support_jsonl(output / 'recovery.jsonl')
    support.write_support_jsonl(output / 'recovery.jsonl', [replace(rows[0], answer='modified response')])
    with pytest.raises(ValueError, match='fingerprints'):
        support.load_support_splits({'support_data_dir': str(output)})


def test_document_families_title_exact_and_near_duplicates():
    original = ' '.join(f'word{i}' for i in range(100))
    docs = {
        'a': {'title': 'Policy#1', 'doc_text': original},
        'b': {'title': 'Policy#2', 'doc_text': 'entirely different second page'},
        'c': {'title': 'Other', 'doc_text': original.upper()},
        'd': {'title': 'Near', 'doc_text': original + ' trailing'},
        'e': {'title': 'Distinct', 'doc_text': 'unrelated subject has separate terms'},
    }
    families, edges = support._document_families(docs, .8)
    assert families['a'] == families['b'] == families['c'] == families['d']
    assert families['a'] != families['e']
    assert edges


def test_duplicate_responses_cannot_cross_splits(tmp_path, monkeypatch):
    archive = fixture_archive(tmp_path, monkeypatch)
    output = tmp_path / 'prepared'
    support.prepare_support_dataset(archive, output, TinyTokenizer(), counts={n: 1 for n in support.DEFAULT_COUNTS})
    splits = support.load_support_splits({'support_data_dir': str(output)})
    response = splits['test'][0].answer
    splits['recovery'][0] = replace(splits['recovery'][0], answer=response, response_sha256=support.digest(response))
    with pytest.raises(ValueError, match='response_sha256 overlap'):
        support.validate_support_splits(splits)


def test_limits_skip_answers_without_truncation(tmp_path, monkeypatch):
    archive = fixture_archive(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='Insufficient'):
        support.prepare_support_dataset(archive, tmp_path / 'prepared', TinyTokenizer(),
                                        counts={n: 1 for n in support.DEFAULT_COUNTS}, max_answer_tokens=2)


def test_archive_hash_is_verified(tmp_path):
    archive = tmp_path / 'bad.zip'
    archive.write_bytes(b'wrong archive')
    with pytest.raises(ValueError, match='SHA256'):
        support.prepare_support_dataset(archive, tmp_path / 'output', TinyTokenizer())


@pytest.mark.parametrize('reply', [
    'I can help you with that information.', 'I can help you with that.',
    'Sure, I would be happy to assist you with this request!',
    'Thank you for your question.',
])
def test_pure_generic_reply_filter(reply):
    assert support.is_generic_reply(reply)


@pytest.mark.parametrize('reply', [
    'I can help you with that. You can renew your license online.',
    'Sure, I can help you with that information: complete Form 1040.',
    'Thank you for your question. You must apply before July 1.',
    'You can renew your license online today.',
])
def test_generic_prefix_does_not_remove_substantive_answer(reply):
    assert not support.is_generic_reply(reply)


def test_near_duplicate_selection_guard_and_exhaustive_audit(tmp_path, monkeypatch):
    archive = fixture_archive(tmp_path, monkeypatch)
    with zipfile.ZipFile(archive) as bundle:
        docs = json.loads(bundle.read('doc2dial_doc.json'))['doc_data']['domain']
        dialogues = {source: json.loads(bundle.read(f'doc2dial_dial_{source}.json'))['dial_data']
                     for source in ('train', 'validation', 'test')}
    documents = {'domain:' + key: value for key, value in docs.items()}
    families, _ = support._document_families(documents, .8)
    rows, _ = support._build_candidates(dialogues, documents, families, TinyTokenizer(), 512, 64, 4)
    test = next(row for row in rows if row.source_split == 'test')
    answer = 'You can renew your license online today.'
    test = replace(test, answer=answer, response_sha256=support.digest(answer))
    near = next(row for row in rows if row.source_split == 'train')
    near_answer = 'You can renew your license online today please.'
    near = replace(near, answer=near_answer, response_sha256=support.digest(near_answer))
    candidates = [test, near] + [row for row in rows if row.key not in (test.key, near.key)]
    # Require all remaining non-test examples: the near-duplicate must be rejected,
    # producing an explicit shortage rather than quietly leaking across splits.
    with pytest.raises(ValueError, match='Insufficient'):
        support._select(candidates, {'test': 1, 'dev': 1, 'calibration': 1, 'recovery': 5}, 7)
    audit = support._near_match_audit({'test': [test], 'recovery': [near]})
    assert audit['exhaustive'] and audit['checked_pairs'] == 1 and audit['matched_pairs'] == 1
    assert support._near_duplicate_kind(support._representations(test), support._representations(near)) == 'response'


def test_context_near_duplicate_guard():
    a = (support.shingles('completely different answer words', 3), support.shingles('Please tell me how to renew my license online', 3))
    b = (support.shingles('another unrelated factual statement', 3), support.shingles('Please tell me how to renew my license online today', 3))
    assert support._near_duplicate_kind(a, b) == 'dialogue_context'
