"""Independent tests of improvement 6 (vocabulary trimming), from plans/improvement-6-vocabulary.md.

Uses the REAL Qwen2.5 tokenizer (cached files, no weights) with tiny random Qwen2 models that
have the real 151,936-row vocabulary, so ids, special tokens and the chat template are real.
"""
from dataclasses import asdict
import glob
import json
import math
import os
from pathlib import Path
import random

import pytest
import torch
from transformers import LogitsProcessor, LogitsProcessorList, Qwen2Config, Qwen2ForCausalLM

from prune_lab import vocab

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = glob.glob('/Users/aryan/llm_pruning_optimization/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*')
DATASET = Path('/Users/aryan/llm_pruning_optimization/datasets/doc2dial_v2')
pytestmark = pytest.mark.skipif(not SNAPSHOTS, reason='Qwen2.5 tokenizer not cached')

ENGLISH = ['You can renew your driver license online or at a local office.',
           'Please bring two proofs of residency and your social security card.',
           'Student aid applications open on October first each year.',
           'Veterans may qualify for disability compensation and health care benefits.',
           'The office is open Monday through Friday from nine to five.',
           'Contact the benefits hotline if your payment is late.']

ADVERSARIAL = [
    '我们的政策规定，申请人必须在三十天内提交。日本語のテキストも含みます。한국어 문장입니다.',
    '👩‍👩‍👧‍👦 🏳️‍🌈 🤖🫠 emoji ZWJ sequences and skin tones 👍🏽👍🏿',
    'Combining: é ä Z̤̥͍͈̮̖̯ͅa̝l̦g̲o̺ naïve café façade Ångström',
    ' the  The\tthe\n the\r\n  THE   tHe',
    'inline special <|im_start|>user hello<|im_end|> and <|endoftext|> inside text',
    '1234567890' * 40 + ' 3.14159265358979323846264338327950288419716939937510',
    'مرحبا بكم في الخدمة — שלום עולם — Привет, мир! — γειά σου κόσμε',
    'zero​width‍join﻿bom \x00nul \x07bell  line sep',
    '“curly quotes” ‘single’ — en–dash … ellipsis ™ © ® ° ± µ ¶ § ¿¡',
    '    indented code:\n\tdef f(x):\n\t\treturn {"a": [1, 2, 3]}  # comment\n',
    'https://studentaid.gov/apply-for-aid/fafsa?year=2026&lang=en#section-3',
    '\U0001F600\U0001F4A9\U00010348\U0001D11E astral plane characters',
]


def qwen_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(SNAPSHOTS[0])


class Row:
    def __init__(self, key, prompt, answer):
        self.key, self.prompt, self.answer = key, prompt, answer


@pytest.fixture(scope='module')
def tok():
    return qwen_tokenizer()


@pytest.fixture(scope='module')
def kept(tok):
    rows = [Row(f'r{i}', ENGLISH[i], ENGLISH[(i + 1) % len(ENGLISH)]) for i in range(len(ENGLISH))]
    ids, _ = vocab.build_kept_set(tok, {'recovery': rows[:4], 'calibration': rows[4:]})
    return ids


def test_worktree_package_is_under_test():
    assert Path(vocab.__file__).resolve().is_relative_to(ROOT / 'src')


# --- tokenizer fallback on the real Qwen tokenizer --------------------------------------------------

def test_kept_set_contents_and_sources(tok, kept):
    structure = vocab.BPEStructure(tok)
    s = set(kept)
    assert set(structure.byte_ids) <= s and len(structure.byte_ids) == 256
    assert set(int(i) for i in tok.added_tokens_decoder) <= s  # all 22 added/control tokens
    assert {151643, 151644, 151645} <= s  # endoftext (pad), im_start, im_end (eos)
    rendered = tok.apply_chat_template([{'role': 'user', 'content': 'x'}], tokenize=True, add_generation_prompt=True)
    assert set(rendered) - set(tok.encode('x')) <= s  # chat-template tokens of the rendering
    for name in ('dev', 'test', 'validation'):
        with pytest.raises(ValueError, match='recovery and calibration'):
            vocab.build_kept_set(tok, {'recovery': [Row('a', 'b c', 'd e')], name: [Row('z', 'q', 'r')]})
    with pytest.raises(ValueError):
        vocab.build_kept_set(tok, {})


def test_every_vocabulary_token_splits_into_kept_pieces(tok, kept):
    wrapped = vocab.TrimmedTokenizer(tok, kept)
    structure = vocab.BPEStructure(tok)
    s = set(kept)
    removed = 0
    for token_id in range(len(tok)):
        pieces = wrapped._pieces(token_id)
        assert set(pieces) <= s, token_id
        if token_id in structure.tokens:
            assert ''.join(structure.tokens[p] for p in pieces) == structure.tokens[token_id], token_id
        if token_id not in s:
            removed += 1
            assert len(pieces) >= 2
    assert removed > 140_000


@pytest.mark.parametrize('text', ADVERSARIAL + ENGLISH, ids=range(len(ADVERSARIAL) + len(ENGLISH)))
def test_round_trip_and_pieces_on_adversarial_text(tok, kept, text):
    wrapped = vocab.TrimmedTokenizer(tok, kept)
    original = tok.encode(text, add_special_tokens=False)
    new = wrapped.encode(text, add_special_tokens=False)
    assert all(0 <= i < len(kept) for i in new)
    old = wrapped.unmap_ids(new)
    assert set(old) <= set(kept)
    structure = vocab.BPEStructure(tok)
    # Same byte-level string, piece by piece: concatenated pieces equal the original tokens.
    assert ''.join(structure.tokens.get(i, tok.convert_ids_to_tokens(i)) for i in old) == \
        ''.join(structure.tokens.get(i, tok.convert_ids_to_tokens(i)) for i in original)
    for skip in (False, True):
        assert wrapped.decode(new, skip_special_tokens=skip) == tok.decode(original, skip_special_tokens=skip)
    if tok.decode(original) == text:
        assert wrapped.decode(new) == text
    # Prefix decodes (partial UTF-8 at the cut) agree byte-for-byte with decoding the same old pieces.
    for cut in range(0, len(new), max(1, len(new) // 7)):
        assert wrapped.decode(new[:cut]) == tok.decode(old[:cut])
    # Special tokens written inside text stay single kept specials.
    for special in ('<|im_start|>', '<|im_end|>', '<|endoftext|>'):
        if special in text:
            assert wrapped._new[tok.convert_tokens_to_ids(special)] in new


def test_fallback_is_exercised_and_chat_template_round_trips(tok, kept):
    wrapped = vocab.TrimmedTokenizer(tok, kept)
    wrapped.fallback_tokens = 0
    for text in ADVERSARIAL:
        wrapped.encode(text, add_special_tokens=False)
    assert wrapped.fallback_tokens > 50
    messages = [{'role': 'user', 'content': ADVERSARIAL[0] + ' ' + ENGLISH[0]}]
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = wrapped.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    assert wrapped.decode(ids) == rendered
    assert wrapped.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) == rendered
    from prune_lab.data import prompt_ids
    assert prompt_ids(wrapped, ENGLISH[1]) == wrapped.map_ids(prompt_ids(tok, ENGLISH[1]))
    assert wrapped.eos_token_id == wrapped._new[151645] and wrapped.pad_token_id == wrapped._new[151643]
    assert len(wrapped) == len(kept)


@pytest.mark.skipif(not (DATASET / 'recovery.jsonl').exists(), reason='v2 data not available read-only')
def test_real_kept_set_size_and_parameter_counts(tok):
    from prune_lab.support_data import read_support_jsonl
    splits = {name: read_support_jsonl(DATASET / f'{name}.jsonl') for name in ('recovery', 'calibration')}
    kept, provenance = vocab.build_kept_set(tok, splits)
    assert len(kept) == 7764 and provenance['from_splits'] == ['calibration', 'recovery']
    config = {'hidden_size': 896, 'vocab_size': 151936, 'tie_word_embeddings': True}
    assert vocab.trimmed_parameters(config, len(kept), 494_032_768) == 364_854_656
    uniform_20 = 494_032_768 - 24 * 64 * 24 * 3 * 896  # 24 uniform steps of 64 neurons per layer
    assert uniform_20 == 394_942_336
    assert vocab.trimmed_parameters(config, len(kept), uniform_20) == 265_764_224
    assert 364_854_656 <= uniform_20  # the primary comparison is size-valid


# --- trimmed model -------------------------------------------------------------------------------

def tiny_model(seed=0, tie=True, eos_boost=0.0, width=64):
    torch.manual_seed(seed)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=151936, hidden_size=32, intermediate_size=width, num_hidden_layers=2,
                                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=1024,
                                         bos_token_id=151643, eos_token_id=151645, pad_token_id=None,
                                         tie_word_embeddings=tie)).eval()
    with torch.no_grad():
        model.lm_head.weight[151645] *= 1 + eos_boost
    model.generation_config.update(bos_token_id=151643, pad_token_id=151643, eos_token_id=[151645, 151643],
                                   repetition_penalty=1.1, do_sample=True, temperature=.7, top_p=.8, top_k=20)
    return model


def save_export(model, tok, path, widths=None):
    from prune_lab.artifacts import save_checkpoint
    if widths is not None:
        from prune_lab.pruning import compact_model
        compact_model(model, {i: list(range(w)) for i, w in enumerate(widths)})
    save_checkpoint(model, tok, path, {'test': True})
    return path


def on_disk_parameters(directory):
    from safetensors import safe_open
    total = 0
    for file in Path(directory).glob('*.safetensors'):
        with safe_open(str(file), 'pt') as f:
            total += sum(math.prod(f.get_slice(k).get_shape()) for k in f.keys())
    return total


@pytest.mark.parametrize('tie', [True, False])
def test_trim_export_ids_tying_and_sizes(tok, kept, tmp_path, tie):
    from prune_lab.modeling import count_parameters
    source = save_export(tiny_model(tie=tie), tok, tmp_path / 'src')
    info = vocab.trim_export(source, tmp_path / 'trim', kept, {'kept_set': {'x': 1}})
    new = {old: i for i, old in enumerate(kept)}
    config = json.loads((tmp_path / 'trim' / 'config.json').read_text())
    generation = json.loads((tmp_path / 'trim' / 'generation_config.json').read_text())
    assert config['vocab_size'] == len(kept)
    assert config['eos_token_id'] == new[151645] and config['bos_token_id'] == new[151643]
    assert generation['eos_token_id'] == [new[151645], new[151643]]
    assert generation['pad_token_id'] == new[151643] and generation['bos_token_id'] == new[151643]
    assert generation['repetition_penalty'] == 1.1
    model, wrapped = vocab.load_trimmed(tmp_path / 'trim')
    assert model.model.embed_tokens.weight.shape == (len(kept), 32)
    assert (model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()) == tie
    original = load_original(source)
    assert torch.equal(model.model.embed_tokens.weight, original.model.embed_tokens.weight[kept])
    assert torch.equal(model.lm_head.weight, original.lm_head.weight[kept])
    removed = (151936 - len(kept)) * 32 * (1 if tie else 2)
    assert info['parameters_after'] == count_parameters(model) == count_parameters(original) - removed
    assert on_disk_parameters(tmp_path / 'trim') == info['parameters_after']
    mapping = json.loads((tmp_path / 'trim' / vocab.MAP_FILE).read_text())
    from prune_lab.campaign import checkpoint_identity
    assert mapping['kept_old_ids'] == kept and mapping['source_checkpoint_sha256'] == checkpoint_identity(source)
    with pytest.raises(ValueError, match='not empty'):
        vocab.trim_export(source, tmp_path / 'trim', kept)
    with pytest.raises(ValueError):
        vocab.trim_export(source, tmp_path / 'bad', kept + [151936])


def test_logits_bit_identical_at_real_head_shape(tok, tmp_path):
    """Hidden 896 and the real 7.7k/152k head shapes: kept-id logits must be bit-identical on CPU."""
    torch.manual_seed(1)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=151936, hidden_size=896, intermediate_size=128, num_hidden_layers=1,
                                         num_attention_heads=14, num_key_value_heads=2, tie_word_embeddings=True)).eval()
    kept = sorted(set(range(256)) | set(random.Random(0).sample(range(151936), 7764 - 256)))
    source = save_export(model, tok, tmp_path / 'src')
    vocab.trim_export(source, tmp_path / 'trim', kept)
    trimmed, _ = vocab.load_trimmed(tmp_path / 'trim')
    model = load_original(source)
    new = {old: i for i, old in enumerate(kept)}
    ids = [k for k in kept[300:400]]
    with torch.inference_mode():
        expected = model(input_ids=torch.tensor([ids])).logits[0][:, torch.tensor(kept)]
        actual = trimmed(input_ids=torch.tensor([[new[i] for i in ids]])).logits[0]
    assert torch.equal(actual, expected)


def load_original(path):
    from prune_lab.modeling import load_model
    return load_model(str(path), device='cpu', dtype='float32')[0]  # same (eager) loader as the trimmed path


class MaskRemoved(LogitsProcessor):
    def __init__(self, kept, size=151936):
        self.removed = torch.ones(size, dtype=torch.bool)
        self.removed[torch.tensor(kept)] = False

    def __call__(self, input_ids, scores):
        return scores.masked_fill(self.removed.to(scores.device), float('-inf'))


def masked_generate(model, rows, pad_id, max_new_tokens, kept):
    size = model.config.vocab_size
    width = max(len(r) for r in rows)
    ids = torch.tensor([[pad_id] * (width - len(r)) + r for r in rows])
    mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows])
    out = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=max_new_tokens,
                         pad_token_id=pad_id, use_cache=True, logits_processor=LogitsProcessorList([MaskRemoved(kept, size)]))
    return [row[width:] for row in out.tolist()]


def eos_competing_model(tok, kept, seed=3):
    """Untied model whose EOS output row is twice the row of the token greedy would pick first for
    ENGLISH[0], so EOS wins at least there (even after the repetition penalty) and replies stop."""
    from prune_lab.data import prompt_ids
    model = tiny_model(seed=seed, tie=False)
    first = masked_generate(model, [prompt_ids(tok, ENGLISH[0])], 151643, 1, kept)[0][0]
    with torch.no_grad():
        model.lm_head.weight[151645] = 2 * model.lm_head.weight[first]
    return model


@pytest.mark.parametrize('eos_boost', [0.0, 'stop'])
def test_generation_equals_original_with_removed_logits_masked(tok, kept, tmp_path, eos_boost):
    """Batch one and left-padded batches: remapped pad/EOS/penalty must reproduce masked-original greedy."""
    from prune_lab.data import prompt_ids
    model = eos_competing_model(tok, kept) if eos_boost == 'stop' else tiny_model(seed=3)
    source = save_export(model, tok, tmp_path / 'src')
    vocab.trim_export(source, tmp_path / 'trim', kept)
    original = load_original(source)
    trimmed, wrapped = vocab.load_trimmed(tmp_path / 'trim')
    prompts = [prompt_ids(tok, text) for text in ENGLISH]  # every token kept: no fallback
    assert all(set(p) <= set(kept) for p in prompts)
    eos = {151645, 151643}
    lengths = set()
    for batch in ([p] for p in prompts):
        expected = masked_generate(original, batch, 151643, 10, kept)
        got = masked_generate(trimmed, [wrapped.map_ids(p) for p in batch], wrapped.pad_token_id, 10, list(range(len(kept))))
        assert [wrapped.unmap_ids(g) for g in got] == expected
        lengths.add(next((j + 1 for j, t in enumerate(expected[0]) if t in eos), len(expected[0])))
    batched = masked_generate(original, prompts, 151643, 10, kept)
    got = masked_generate(trimmed, [wrapped.map_ids(p) for p in prompts], wrapped.pad_token_id, 10, list(range(len(kept))))
    assert [wrapped.unmap_ids(g) for g in got] == batched
    if eos_boost:
        assert min(lengths) < 10  # EOS actually stops some replies
    # The unmasked original often picks a removed token here, so the masking above is doing real work.
    plain = original.generate(input_ids=torch.tensor([prompts[0]]), do_sample=False, max_new_tokens=10, pad_token_id=151643)
    assert eos_boost or set(plain[0, len(prompts[0]):].tolist()) - set(kept)


def test_evaluate_support_records_match_masked_original(tok, kept, tmp_path):
    from prune_lab.support_evaluation import evaluate_support, reply_metrics
    from prune_lab.data import prompt_ids
    source = save_export(eos_competing_model(tok, kept, seed=4), tok, tmp_path / 'src')
    vocab.trim_export(source, tmp_path / 'trim', kept)
    original = load_original(source)
    trimmed, wrapped = vocab.load_trimmed(tmp_path / 'trim')
    from prune_lab.support_evaluation import generation_batches
    rows = support_rows(ENGLISH, 'dev')

    def cut(tokens):
        stop = next((j for j, t in enumerate(tokens) if t in (151645, 151643)), None)
        return tokens if stop is None else tokens[:stop + 1]
    single = evaluate_support(trimmed, wrapped, rows, 512, 8, 1)
    lengths = []
    for row, ex in zip(single['predictions'], rows):
        expected = cut(masked_generate(original, [prompt_ids(tok, ex.prompt)], 151643, 8, kept)[0])
        assert row['generated_tokens'] == len(expected)
        assert row['prediction'] == tok.decode(expected, skip_special_tokens=True)
        assert row['score'] == reply_metrics(row['prediction'], ex.answer)['token_f1']
        lengths.append(len(expected))
    assert min(lengths) < 8  # EOS (remapped) really stops replies
    # Batched path: same length buckets (no fallback, so lengths agree), left padding with the pad id.
    batched = evaluate_support(trimmed, wrapped, rows, 512, 8, 3)
    expected = {}
    for batch in generation_batches(rows, tok, 3):
        outputs = masked_generate(original, [ids for _, ids in batch], 151643, 8, kept)
        for (index, _), out in zip(batch, outputs):
            expected[index] = cut(out)
    for index, row in enumerate(batched['predictions']):
        assert row['generated_tokens'] == len(expected[index])
        assert row['prediction'] == tok.decode(expected[index], skip_special_tokens=True)


# --- trim_run on a hand-built source run -----------------------------------------------------------

def support_rows(texts, prefix, answers=None):
    from prune_lab.support_data import SupportExample, digest
    rows = []
    for i, text in enumerate(texts):
        answer = (answers or texts)[(i + 1) % len(texts)]
        rows.append(SupportExample(prompt=text, answer=answer, key=f'{prefix}{i}', doc_id=f'doc{i % 2}',
                                   dialogue_id=f'{prefix}d{i}', domain='a', doc_family=f'fam{i % 3}', group_id=f'fam{i % 3}',
                                   doc_text_sha256='x', source_split='train', target_turn_id=2, source_section_ids=['s'],
                                   history_turn_ids=[1], source_context_sha256='c', dialogue_context_sha256=f'{prefix}{i}',
                                   response_sha256=digest(answer), prompt_sha256=digest(text), prompt_tokens=1, answer_tokens=1))
    return rows


DEV_TEXTS = ENGLISH[:3] + ['Une phrase en français avec des accents: été, déjà, garçon.',
                           ADVERSARIAL[0], 'Call 1-800-555-0199 about your 1099-G form, case #A7Z.']


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(asdict(r), ensure_ascii=False) + '\n' for r in rows))


@pytest.fixture
def source_run(tok, tmp_path, monkeypatch):
    from prune_lab import campaign
    from prune_lab.modeling import count_parameters, load_model
    from prune_lab.support_evaluation import evaluate_support
    monkeypatch.chdir(tmp_path)
    torch.set_num_threads(1)
    recovery = support_rows([f'{t} Case {i}.' for i, t in enumerate(ENGLISH * 2)], 'rec')
    calibration = support_rows(ENGLISH[3:], 'cal')
    dev = support_rows(DEV_TEXTS, 'dev')
    root = tmp_path / 'source'
    for seed in (5, 6):
        seed_dir = root / f'seed-{seed}'
        (seed_dir / 'data').mkdir(parents=True)
        write_jsonl(seed_dir / 'data' / 'recovery.jsonl', recovery)
        write_jsonl(seed_dir / 'data' / 'calibration.jsonl', calibration)
        write_jsonl(seed_dir / 'data' / 'dev.jsonl', dev)
        test = seed_dir / 'data' / 'test.jsonl'
        test.write_text('POISON\n')
        test.chmod(0)
        config = campaign.read_campaign_config()
        config.update(task='support', dataset='doc2dial', support_data_dir='unused', device='cpu', dtype='float32',
                      max_length=64, max_new_tokens=6, eval_batch_size=1, selection_dev_examples=2)
        config['recovery']['seed'] = seed
        (seed_dir / 'config.json').write_text(json.dumps(config))
        (seed_dir / 'state.json').write_text(json.dumps({'test_started': False, 'dataset': {'dev': {'count': len(dev), 'sha256': 'd'}}}))
        keys = [r.key for r in dev[:2]]
        (seed_dir / 'selection-dev.json').write_text(json.dumps({'count': 2, 'keys': keys, 'keys_sha256': 'h', 'seed': 1}))
        candidates = []
        for k, (name, widths) in enumerate((('unpruned-recovery', None), ('uniform-20', [32, 32]))):
            export = save_export(tiny_model(seed=seed * 10 + k, eos_boost=3.0), tok, seed_dir / 'candidates' / name / 'checkpoint', widths)
            model, tokenizer = load_model(str(export), device='cpu', dtype='float32')
            candidates.append({'id': name, 'status': 'complete', 'checkpoint': str(export),
                               'checkpoint_sha256': campaign.checkpoint_identity(export), 'parameters': count_parameters(model),
                               'actual_reduction': .2 if widths else 0.0, 'control': True,
                               'dev': evaluate_support(model, tokenizer, dev, 512, 6)})
        selection = {'baseline': {'dev': candidates[0]['dev'], 'parameters': candidates[0]['parameters']},
                     'candidates': candidates + [{'id': 'per-layer-02', 'status': 'failed', 'error': 'x'}], 'tiers': []}
        (seed_dir / 'selection.json').write_text(json.dumps(selection))
        (seed_dir / 'selection-lock.json').write_text(json.dumps({'sha256': campaign._fingerprint(selection)}))
    yield root
    for seed in (5, 6):
        (root / f'seed-{seed}' / 'data' / 'test.jsonl').chmod(0o600)


def test_trim_run_end_to_end(source_run, tmp_path, monkeypatch):
    from prune_lab import campaign, multiseed
    opened, real_open = [], open
    monkeypatch.setattr('builtins.open', lambda f, *a, **k: opened.append(str(f)) or real_open(f, *a, **k))
    output = tmp_path / 'vocab'
    vocab.trim_run(source_run, output, ['unpruned-recovery', 'uniform-20'], check_prompts=2)
    monkeypatch.setattr('builtins.open', real_open)
    assert not [p for p in opened if p.endswith('test.jsonl')]
    maps = []
    for seed in (5, 6):
        src, dst = source_run / f'seed-{seed}', output / f'seed-{seed}'
        for name in ('config.json', 'state.json', 'selection-dev.json'):
            assert (dst / name).read_bytes() == (src / name).read_bytes()
        source_records = {c['id']: c for c in campaign._selection(src)['candidates']}
        assembled = {c['id']: c for c in campaign._selection(dst)['candidates']}
        assert set(assembled) == {'unpruned-recovery', 'uniform-20', 'vocab-unpruned', 'vocab-uniform-20'}
        for name in ('unpruned-recovery', 'uniform-20'):
            assert json.dumps(assembled[name], sort_keys=True) == json.dumps(source_records[name], sort_keys=True)
        for name, variant in (('vocab-unpruned', 'unpruned-recovery'), ('vocab-uniform-20', 'uniform-20')):
            result = assembled[name]
            assert result['source_checkpoint_sha256'] == source_records[variant]['checkpoint_sha256']
            assert result['logit_check']['passed'] and result['logit_check']['checked'] == 2
            assert on_disk_parameters(result['checkpoint']) == result['parameters'] == result['trim']['parameters_after']
            assert result['parameters'] == source_records[variant]['parameters'] - (151936 - result['trim']['kept_vocab_size']) * 32
            assert [r['key'] for r in result['dev']['predictions']] == [r.key for r in support_rows(DEV_TEXTS, 'dev')]
            assert result['dev_fallback']['examples_with_prompt_fallback'] >= 2  # French, CJK
            assert result['kept_set']['from_splits'] == ['calibration', 'recovery']
            maps.append(json.loads((Path(result['checkpoint']) / vocab.MAP_FILE).read_text())['kept_old_ids'])
            assert 'not comparable' in result['loss_note']
            assert result['evaluation_max_length'] >= 64 and json.loads((dst / 'config.json').read_text())['max_length'] == 64
    assert all(m == maps[0] for m in maps)  # same recovery/calibration data -> same kept set
    a, b = multiseed.collect_run(source_run), multiseed.collect_run(output)
    assert a['selection_keys'] == b['selection_keys']
    summary = multiseed.aggregate_run(output, samples=20)
    assert {'vocab-unpruned', 'vocab-uniform-20', 'uniform-20', 'unpruned-recovery'} <= set(summary['variants'])
    same = multiseed.compare_runs(source_run, output, pairs=[('uniform-20', 'uniform-20')], samples=20)
    assert same['comparisons'][0]['token_f1']['estimate_points'] == 0
    primary = multiseed.compare_runs(output, output, pairs=[('uniform-20', 'vocab-unpruned')], samples=20)
    assert primary['example_set'] == 'dev-holdout'


def test_kept_set_independent_of_dev(source_run, tmp_path):
    vocab.trim_run(source_run, tmp_path / 'a', ['uniform-20'], seeds=[5], check_prompts=1)
    dev = source_run / 'seed-5' / 'data' / 'dev.jsonl'
    rows = [json.loads(l) for l in dev.read_text().splitlines()]
    for row in rows:
        row['prompt'] = ADVERSARIAL[1] + ' ' + ADVERSARIAL[6]
        row['answer'] = ADVERSARIAL[0]
    dev.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    vocab.trim_run(source_run, tmp_path / 'b', ['uniform-20'], seeds=[5], check_prompts=1)
    a = json.loads((tmp_path / 'a' / 'seed-5' / 'vocab-uniform-20' / 'export' / vocab.MAP_FILE).read_text())
    b = json.loads((tmp_path / 'b' / 'seed-5' / 'vocab-uniform-20' / 'export' / vocab.MAP_FILE).read_text())
    assert a['kept_old_ids'] == b['kept_old_ids']


def test_hash_missing_variant_and_stale_resume(source_run, tmp_path, monkeypatch):
    from prune_lab import campaign
    with pytest.raises(ValueError, match='per-layer-02'):
        vocab.trim_run(source_run, tmp_path / 'm', ['uniform-20', 'per-layer-02'])
    assert not (tmp_path / 'm').exists()
    output = tmp_path / 'vocab'
    vocab.trim_run(source_run, output, ['uniform-20'], seeds=[5], check_prompts=1)
    saved = (output / 'seed-5' / 'vocab-uniform-20' / 'result.json').read_bytes()
    monkeypatch.setattr(vocab, 'trim_export', lambda *a, **k: (_ for _ in ()).throw(AssertionError('retrimmed')))
    vocab.trim_run(source_run, output, ['uniform-20'], seeds=[5], check_prompts=1)
    assert (output / 'seed-5' / 'vocab-uniform-20' / 'result.json').read_bytes() == saved
    monkeypatch.undo()
    # Replace the source export consistently (a retrained candidate): the stale result must be redone.
    seed_dir = source_run / 'seed-5'
    export = seed_dir / 'candidates' / 'uniform-20' / 'checkpoint'
    for p in export.iterdir():
        p.unlink()
    save_export(tiny_model(seed=999), qwen_tokenizer(), export, [32, 32])
    selection = campaign._selection(seed_dir)
    for c in selection['candidates']:
        if c['id'] == 'uniform-20':
            c['checkpoint_sha256'] = campaign.checkpoint_identity(export)
    (seed_dir / 'selection.json').write_text(json.dumps(selection))
    (seed_dir / 'selection-lock.json').write_text(json.dumps({'sha256': campaign._fingerprint(selection)}))
    monkeypatch.chdir(tmp_path)
    vocab.trim_run(source_run, output, ['uniform-20'], seeds=[5], check_prompts=1)
    result = json.loads((output / 'seed-5' / 'vocab-uniform-20' / 'result.json').read_text())
    assert result['source_checkpoint_sha256'] == campaign.checkpoint_identity(export)
    # Corrupted export bytes are refused.
    weights = next(export.glob('*.safetensors'))
    data = bytearray(weights.read_bytes())
    data[-5] ^= 0xFF
    weights.write_bytes(bytes(data))
    with pytest.raises(ValueError, match='hash'):
        vocab.trim_run(source_run, tmp_path / 'c', ['uniform-20'], seeds=[5])


@pytest.mark.xfail(strict=True, reason='Medium: the plan calls the logit exactness check "required", but trim_run only '
                   'records a failed check in result.json and finishes normally (no warning, no non-zero exit), so a '
                   'broken trim can be aggregated and compared silently.')
def test_failed_exactness_check_is_not_silent(source_run, tmp_path, monkeypatch):
    real = vocab.logit_check
    monkeypatch.setattr(vocab, 'logit_check', lambda *a, **k: dict(real(*a, **k), passed=False))
    with pytest.raises(Exception):
        vocab.trim_run(source_run, tmp_path / 'v', ['uniform-20'], seeds=[5], check_prompts=1)


def test_evaluation_max_length_raised_only_for_fallback(source_run, tmp_path):
    from prune_lab.data import prompt_ids
    vocab.trim_run(source_run, tmp_path / 'v', ['uniform-20'], seeds=[5], check_prompts=1)
    result = json.loads((tmp_path / 'v' / 'seed-5' / 'vocab-uniform-20' / 'result.json').read_text())
    tok = qwen_tokenizer()
    _, wrapped = vocab.load_trimmed(result['checkpoint'])
    dev = support_rows(DEV_TEXTS, 'dev')
    needed = max(len(prompt_ids(wrapped, ex.prompt)) + len(wrapped.encode(ex.answer, add_special_tokens=False)) + 1 for ex in dev)
    original = max(len(prompt_ids(tok, ex.prompt)) + len(tok.encode(ex.answer, add_special_tokens=False)) + 1 for ex in dev)
    assert needed > original  # fallback lengthened something
    assert result['evaluation_max_length'] == max(64, needed)
    assert result['dev_examples_over_config_max_length'] == sum(
        len(prompt_ids(wrapped, ex.prompt)) + len(wrapped.encode(ex.answer, add_special_tokens=False)) + 1 > 64 for ex in dev)
