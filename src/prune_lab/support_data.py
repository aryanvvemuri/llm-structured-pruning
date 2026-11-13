"""Prepare Doc2Dial oracle-document support reply examples without target leakage.

Gold turn references select source sections: this is deliberately NOT a retrieval
benchmark. Only preceding turns enter the conversation context. Original human
responses may contain annotation errors; heuristic deduplication is not semantic
identity detection or an assertion of factual correctness.
"""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import zipfile

from .data import dataset_manifest, prompt_ids, validate_splits

SOURCE_SHA256 = '94499fa5259f69018d2458cb948552e7a05f424711a95a562bb9e816a515dc23'
DEFAULT_COUNTS = {'recovery': 1000, 'calibration': 150, 'dev': 200, 'test': 300}


def normalize(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def digest(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def shingles(text, width=5):
    words = normalize(text).split()
    return {tuple(words[i:i + width]) for i in range(max(1, len(words) - width + 1))}


def jaccard(left, right):
    return len(left & right) / len(left | right) if left or right else 1.0


def is_generic_reply(text):
    """Reject only complete generic acknowledgements/help offers, never prefixes."""
    normalized = normalize(text)
    prefix = r'(?:(?:yes|sure|certainly|absolutely|okay|ok|of course|hello|hi) )*'
    offer = (r'(?:i (?:can|will|would be happy to|am happy to|would be glad to) '
             r'(?:help|assist) you(?: with (?:that|this)(?: (?:information|request|question|issue))?)?'
             r'|(?:thank you|thanks)(?: for (?:that|the information|your question))?'
             r'|you are welcome|no problem|happy to help)')
    return bool(re.fullmatch(prefix + offer, normalized))


def _representations(row):
    context = row.prompt.split('Conversation:\n', 1)[1].rsplit('\n\nNext agent reply:', 1)[0]
    return shingles(row.answer, 3), shingles(context, 3)


def _near_duplicate_kind(left, right, threshold=.8):
    for name, a, b in zip(('response', 'dialogue_context'), left, right):
        # Jaccard cannot exceed min(size)/max(size). This is an exact bound.
        if min(len(a), len(b)) >= threshold * max(len(a), len(b)) and jaccard(a, b) >= threshold:
            return name
    return None


@dataclass(frozen=True)
class SupportExample:
    prompt: str
    answer: str
    key: str
    doc_id: str
    dialogue_id: str
    domain: str
    doc_family: str
    group_id: str
    doc_text_sha256: str
    source_split: str
    target_turn_id: int
    source_section_ids: list
    history_turn_ids: list
    source_context_sha256: str
    dialogue_context_sha256: str
    response_sha256: str
    prompt_sha256: str
    prompt_tokens: int
    answer_tokens: int


def read_support_jsonl(path):
    with Path(path).open() as stream:
        rows = [SupportExample(**json.loads(line)) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f'Empty support dataset: {path}')
    return rows


def write_support_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(asdict(row), ensure_ascii=False) + '\n' for row in rows))


GROUP_FIELDS = ('group_id', 'doc_id', 'dialogue_id', 'doc_family', 'doc_text_sha256',
                'source_context_sha256', 'dialogue_context_sha256', 'response_sha256', 'prompt_sha256')


def validate_support_splits(splits):
    if set(splits) != set(DEFAULT_COUNTS):
        raise ValueError('Support data requires recovery/calibration/dev/test')
    validate_splits(splits)
    owners = {field: {} for field in GROUP_FIELDS}
    for name, rows in splits.items():
        for row in rows:
            if row.group_id != row.doc_family:
                raise ValueError('Support group_id must equal the document-family cluster')
            if not row.answer.strip() or not row.prompt.strip():
                raise ValueError('Empty support prompt or response')
            if row.response_sha256 != digest(row.answer) or row.prompt_sha256 != digest(row.prompt):
                raise ValueError('Support prompt/response digest mismatch')
            if any(turn >= row.target_turn_id for turn in row.history_turn_ids):
                raise ValueError('Future/target turn appears in support history')
            if name == 'test' and row.source_split != 'test':
                raise ValueError('Final test must originate in the official test dialogues')
            if name != 'test' and row.source_split not in ('train', 'validation'):
                raise ValueError('Official test dialogue used outside final test')
            for field in GROUP_FIELDS:
                value = getattr(row, field)
                previous = owners[field].setdefault(value, name)
                if previous != name:
                    raise ValueError(f'Cross-split support {field} overlap: {value}')
    return {field: 0 for field in GROUP_FIELDS}


def load_support_splits(config):
    directory = Path(config['support_data_dir'])
    manifest = json.loads((directory / 'manifest.json').read_text())
    splits = {name: read_support_jsonl(directory / f'{name}.jsonl') for name in DEFAULT_COUNTS}
    if dataset_manifest(splits) != manifest['dataset']:
        raise ValueError('Prepared support dataset fingerprints changed')
    validate_support_splits(splits)
    if config.get('counts') is not None and {k: len(v) for k, v in splits.items()} != config['counts']:
        raise ValueError('Configured counts differ from prepared support counts')
    limits = manifest['preparation']
    if config.get('max_length', limits['max_length']) < max(row.prompt_tokens + row.answer_tokens for rows in splits.values() for row in rows):
        raise ValueError('Configured max_length cannot fit prepared support examples')
    if config.get('max_new_tokens', limits['max_answer_tokens']) < max(row.answer_tokens for rows in splits.values() for row in rows):
        raise ValueError('Configured generation cap cannot fit prepared reference responses')
    return splits


def _document_families(documents, threshold):
    ids = sorted(documents)
    parents = {key: key for key in ids}
    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    def union(a, b):
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)
    titles, texts = {}, {}
    sets = {}
    for key in ids:
        doc = documents[key]
        title = normalize(doc['title'].split('#', 1)[0])
        text = digest(doc['doc_text'])
        for value, seen in ((title, titles), (text, texts)):
            if value in seen:
                union(key, seen[value])
            else:
                seen[value] = key
        sets[key] = shingles(doc['doc_text'])
    near_edges = []
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            a, b = sets[left], sets[right]
            if min(len(a), len(b)) < threshold * max(len(a), len(b)):
                continue
            similarity = jaccard(a, b)
            if similarity >= threshold:
                union(left, right)
                near_edges.append({'left': left, 'right': right, 'jaccard': similarity})
    return {key: hashlib.sha256(find(key).encode()).hexdigest() for key in ids}, near_edges


def _build_candidates(dialogues, documents, families, tokenizer, max_length, max_answer_tokens, max_history_turns):
    candidates, filters = [], Counter()
    for source, domains in dialogues.items():
        for domain, docs in domains.items():
            for doc_id, conversations in docs.items():
                doc_key = domain + ':' + doc_id
                doc = documents[doc_key]
                for conversation in conversations:
                    turns = conversation['turns']
                    for index, turn in enumerate(turns):
                        if turn['role'] != 'agent' or turn['da'] != 'respond_solution':
                            filters['not_agent_solution'] += 1
                            continue
                        answer = turn['utterance'].strip()
                        if is_generic_reply(answer):
                            filters['pure_generic_acknowledgement_or_help_offer'] += 1
                            continue
                        if len(normalize(answer).split()) < 5:
                            filters['short_or_greeting_response'] += 1
                            continue
                        refs = turn.get('references', [])
                        if not refs or any(ref['sp_id'] not in doc['spans'] for ref in refs):
                            filters['missing_source_reference'] += 1
                            continue
                        sections = {}
                        for ref in refs:
                            span = doc['spans'][ref['sp_id']]
                            sections[(int(span['start_sec']), str(span['id_sec']))] = span['text_sec'].strip()
                        section_keys = sorted(sections)
                        source_text = '\n\n'.join(sections[k] for k in section_keys)
                        history_turns = turns[max(0, index - max_history_turns):index]
                        if not source_text or not history_turns or history_turns[-1]['role'] != 'user':
                            filters['missing_source_or_latest_user'] += 1
                            continue
                        history = '\n'.join(('Customer' if t['role'] == 'user' else 'Agent') + ': ' + t['utterance'].strip() for t in history_turns)
                        if normalize(answer) in normalize(history):
                            filters['reference_response_already_in_history'] += 1
                            continue
                        prompt = ('Write the next helpful customer-support reply using only the supplied documentation and conversation. '
                                  'Do not invent unsupported details. Answer the latest customer question concisely.\n\n'
                                  'Supporting documentation:\n'
                                  + source_text + '\n\nConversation:\n' + history + '\n\nNext agent reply:')
                        answer_count = len(tokenizer.encode(answer, add_special_tokens=False)) + int(tokenizer.eos_token_id is not None)
                        if answer_count > max_answer_tokens:
                            filters['answer_over_token_cap'] += 1
                            continue
                        prompt_count = len(prompt_ids(tokenizer, prompt))
                        if prompt_count + answer_count > max_length:
                            filters['example_over_total_token_cap'] += 1
                            continue
                        candidates.append(SupportExample(
                            prompt, answer, source + ':' + conversation['dial_id'] + ':' + str(turn['turn_id']),
                            doc_id, conversation['dial_id'], domain, families[doc_key], families[doc_key], digest(doc['doc_text']), source,
                            int(turn['turn_id']), [k[1] for k in section_keys], [int(t['turn_id']) for t in history_turns],
                            digest(source_text), digest(history), digest(answer), digest(prompt), prompt_count, answer_count))
    return candidates, dict(filters)


def _select(candidates, counts, seed, max_per_dialogue=1):
    rng = random.Random(seed)
    candidates = list(candidates)
    rng.shuffle(candidates)
    by_source = {'test': defaultdict(list), 'other': defaultdict(list)}
    for row in candidates:
        by_source['test' if row.source_split == 'test' else 'other'][row.doc_family].append(row)
    used_families, signatures, used_keys = set(), {f: {} for f in GROUP_FIELDS}, set()
    splits, drops = {}, Counter()
    prior_split_representations = []
    # Maximize held-out examples per document family while retaining training data.
    # Tie-breaking is seeded. Source choice never uses model predictions/scores.
    for name in ('test', 'dev', 'calibration', 'recovery'):
        groups = by_source['test' if name == 'test' else 'other']
        families = list(groups)
        rng.shuffle(families)
        if name == 'test':
            families.sort(key=lambda f: (-len({r.dialogue_id for r in groups[f]}) /
                                        max(1, len({r.dialogue_id for r in by_source['other'].get(f, [])})),
                                        -len({r.dialogue_id for r in groups[f]})))
        else:
            families.sort(key=lambda f: -len({r.dialogue_id for r in groups[f]}))
        selected, dialogue_counts, domain_counts = [], Counter(), Counter()
        domains = sorted({r.domain for rows in groups.values() for r in rows})
        quotas = {domain: counts[name] // len(domains) + int(i < counts[name] % len(domains)) for i, domain in enumerate(domains)}
        family_cap = max(10, math.ceil(counts[name] / 20))
        for family in families:
            if family in used_families:
                continue
            added = False
            family_count = 0
            for row in groups[family]:
                if domain_counts[row.domain] >= quotas[row.domain]:
                    continue
                if family_count >= family_cap:
                    break
                if row.key in used_keys or dialogue_counts[row.dialogue_id] >= max_per_dialogue:
                    continue
                collision = next((field for field in GROUP_FIELDS if getattr(row, field) in signatures[field]
                                  and signatures[field][getattr(row, field)] != name), None)
                if collision:
                    drops[collision] += 1
                    continue
                # validate_splits also disallows identical prompts within one split.
                if any(r.prompt_sha256 == row.prompt_sha256 for r in selected):
                    drops['within_split_prompt'] += 1
                    continue
                representation = _representations(row)
                near_kind = next((kind for prior in prior_split_representations
                                  if (kind := _near_duplicate_kind(representation, prior)) is not None), None)
                if near_kind is not None:
                    drops['near_duplicate_' + near_kind] += 1
                    continue
                selected.append(row)
                used_keys.add(row.key)
                dialogue_counts[row.dialogue_id] += 1
                domain_counts[row.domain] += 1
                family_count += 1
                for field in GROUP_FIELDS:
                    signatures[field][getattr(row, field)] = name
                added = True
                if len(selected) == counts[name]:
                    break
            if added:
                used_families.add(family)
            if len(selected) == counts[name]:
                break
        if len(selected) < counts[name]:
            raise ValueError(f'Insufficient document-disjoint {name}: {len(selected)}/{counts[name]} with max {max_per_dialogue} example/dialogue')
        rng.shuffle(selected)
        splits[name] = selected
        prior_split_representations.extend(_representations(row) for row in selected)
    return splits, dict(drops)


def _near_match_audit(splits, seed=None):
    """Exhaustively inspect every selected cross-split pair with exact pruning."""
    representations = {row.key: _representations(row) for rows in splits.values() for row in rows}
    matches, checked = [], 0
    names = list(splits)
    for i, left_split in enumerate(names):
        for right_split in names[i + 1:]:
            for left in splits[left_split]:
                a = representations[left.key]
                for right in splits[right_split]:
                    checked += 1
                    b = representations[right.key]
                    if _near_duplicate_kind(a, b) is not None:
                        matches.append({'left': left.key, 'right': right.key,
                                        'left_split': left_split, 'right_split': right_split,
                                        'response_jaccard': jaccard(a[0], b[0]),
                                        'dialogue_context_jaccard': jaccard(a[1], b[1])})
    matches.sort(key=lambda x: -max(x['response_jaccard'], x['dialogue_context_jaccard']))
    return {'method': 'Exhaustive cross-split normalized word 3-shingle Jaccard >=0.8 on responses or dialogue context; exact length-ratio bound skips impossible matches',
            'exhaustive': True, 'checked_pairs': checked, 'matched_pairs': len(matches), 'examples': matches[:20],
            'limitation': 'Exhaustive lexical heuristic only; does not establish semantic independence.'}


def prepare_support_dataset(archive, output, tokenizer, counts=None, seed=20261113,
                            max_length=512, max_answer_tokens=64, max_history_turns=4,
                            max_per_dialogue=1, doc_similarity_threshold=.8):
    counts = dict(DEFAULT_COUNTS if counts is None else counts)
    if set(counts) != set(DEFAULT_COUNTS) or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError('Four strictly positive integer split counts required')
    if any(type(n) is not int or n < 1 for n in (max_length, max_answer_tokens, max_history_turns, max_per_dialogue)) or max_per_dialogue > 2:
        raise ValueError('Positive token/history limits and max_per_dialogue of 1 or 2 required')
    if not 0 < doc_similarity_threshold <= 1:
        raise ValueError('Document similarity threshold must be in (0, 1]')
    archive = Path(archive)
    actual_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual_hash != SOURCE_SHA256:
        raise ValueError('Unexpected Doc2Dial archive SHA256')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Prepared output must be absent or empty; do not overwrite an existing benchmark')
    with zipfile.ZipFile(archive) as bundle:
        doc_data = json.loads(bundle.read('doc2dial_doc.json'))['doc_data']
        dialogues = {source: json.loads(bundle.read(f'doc2dial_dial_{source}.json'))['dial_data']
                     for source in ('train', 'validation', 'test')}
        readme = bundle.read('DATA_README.md')
    documents = {domain + ':' + key: doc for domain, docs in doc_data.items() for key, doc in docs.items()}
    families, near_edges = _document_families(documents, doc_similarity_threshold)
    candidates, filters = _build_candidates(dialogues, documents, families, tokenizer, max_length, max_answer_tokens, max_history_turns)
    splits, collisions = _select(candidates, counts, seed, max_per_dialogue)
    overlaps = validate_support_splits(splits)
    audit = _near_match_audit(splits)
    if audit['matched_pairs']:
        raise ValueError('Near-duplicate guard failed exhaustive cross-split audit')
    manifest = {
        'source': {'name': 'Doc2Dial', 'version': '1.0.1', 'sha256': actual_hash,
                   'url': 'https://doc2dial.github.io/data.html',
                   'dataset_card': 'https://huggingface.co/datasets/IBM/doc2dial',
                   'license': 'CC-BY-3.0 (as specified by the IBM dataset card)',
                   'license_url': 'https://creativecommons.org/licenses/by/3.0/',
                   'citation': 'Feng et al. (2020), doc2dial: A Goal-Oriented Document-Grounded Dialogue Dataset, EMNLP; https://aclanthology.org/2020.emnlp-main.652/'},
        'task': 'Oracle supporting-document customer-support reply generation; not a retrieval benchmark.',
        'preparation': {'seed': seed, 'max_length': max_length, 'max_answer_tokens': max_answer_tokens,
                        'max_history_turns': max_history_turns, 'max_per_dialogue': max_per_dialogue,
                        'doc_similarity_threshold': doc_similarity_threshold,
                        'tokenizer': getattr(tokenizer, 'name_or_path', type(tokenizer).__name__),
                        'domain_allocation': 'Equal domain quotas, remainder assigned alphabetically',
                        'per_family_cap': 'max(10, ceil(split_count / 20))',
                        'answer_tokens_include_eos': tokenizer.eos_token_id is not None},
        'split_policy': 'Official test dialogue source only for final test; train+validation for remaining splits. Entire document families and conversations are split-disjoint. Exact canonical response/source-context/dialogue-context/prompt cross-split collisions and response/dialogue-context word 3-shingle Jaccard >=0.8 near-duplicates are dropped before acceptance.',
        'filters': filters, 'cross_split_collision_drops': collisions,
        'eligible_candidates': len(candidates), 'document_count': len(documents),
        'document_family_count': len(set(families.values())),
        'document_grouping': 'Transitive union of normalized title before #, canonical exact full text, or word 5-shingle Jaccard >= threshold',
        'near_duplicate_document_edges': near_edges, 'group_overlaps': overlaps,
        'split_counts': {name: len(rows) for name, rows in splits.items()},
        'split_dialogue_counts': {name: len({r.dialogue_id for r in rows}) for name, rows in splits.items()},
        'split_document_family_counts': {name: len({r.doc_family for r in rows}) for name, rows in splits.items()},
        'split_domains': {name: dict(Counter(r.domain for r in rows)) for name, rows in splits.items()},
        'dataset': dataset_manifest(splits), 'near_match_audit': audit,
        'limitations': ['Oracle gold references choose source sections and reveal relevant evidence.',
                       'Length filtering favors short responses and compact supporting sections; long or complex support cases are underrepresented.',
                       'Human-written references can contain errors or unsupported statements; they are not verified factual answers.',
                       'No target or future turn is inserted into dialogue context; document text may naturally overlap the response.',
                       'Lexical grouping/audits cannot rule out paraphrase or semantic overlap; document disjointness changes the official benchmark distribution.']}
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        write_support_jsonl(output / f'{name}.jsonl', rows)
    (output / 'DATA_README.md').write_bytes(readme)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    return manifest
