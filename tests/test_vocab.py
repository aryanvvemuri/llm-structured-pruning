"""Improvement 6: vocabulary trimming on tiny byte-level BPE tokenizers and models."""
import json

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from prune_lab import vocab

CORPUS = ['Please explain policy number one and the renewal rules.', 'You can renew your license online today.',
          'Unique response with useful helpful information about eligibility evidence.',
          'Customer asks about student aid and veterans benefits.'] * 20


def byte_tokenizer(size=320):
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(CORPUS, trainers.BpeTrainer(vocab_size=size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                                            special_tokens=['<|endoftext|>', '<|im_end|>']))
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='<|im_end|>', pad_token='<|endoftext|>')


class Example:
    def __init__(self, key, prompt, answer):
        self.key, self.prompt, self.answer = key, prompt, answer
        self.group_id = self.doc_id = 'g'
        self.domain, self.dialogue_id = 'd', key


def test_round_trip_for_arbitrary_unicode_and_pieces():
    tokenizer = byte_tokenizer()
    structure = vocab.BPEStructure(tokenizer)
    for token, (left, right) in list(structure.parents.items())[:50]:
        assert left + right == token
    kept, provenance = vocab.build_kept_set(tokenizer, {'recovery': [Example('a', 'renew', 'You can renew online.')]})
    wrapped = vocab.TrimmedTokenizer(tokenizer, kept)
    for text in ['You can renew your license online today.', 'Ünïcödé — 漢字 and emoji 🙂 \t\n tabs', 'policy policy evidence']:
        ids = wrapped.encode(text, add_special_tokens=False)
        assert all(0 <= i < len(kept) for i in ids)
        assert wrapped.decode(ids) == tokenizer.decode(tokenizer.encode(text, add_special_tokens=False))
    assert wrapped.fallback_tokens > 0
    assert wrapped.eos_token_id == kept.index(tokenizer.eos_token_id) and wrapped.pad_token_id == kept.index(tokenizer.pad_token_id)
    assert provenance['from_splits'] == ['recovery']
    with pytest.raises(ValueError, match='recovery and calibration'):
        vocab.build_kept_set(tokenizer, {'dev': []})


def tiny_model(tokenizer, tied=True):
    torch.manual_seed(3)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tokenizer) + 8, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                                        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
                                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                                        tie_word_embeddings=tied)).eval()


@pytest.mark.parametrize('tied', [True, False])
def test_trimmed_logits_equal_restricted_original_and_ids_remapped(tmp_path, tied):
    from prune_lab.artifacts import save_checkpoint
    from prune_lab.campaign import checkpoint_identity
    from prune_lab.modeling import count_parameters
    tokenizer = byte_tokenizer()
    model = tiny_model(tokenizer, tied)
    save_checkpoint(model, tokenizer, tmp_path / 'export', {})
    kept, _ = vocab.build_kept_set(tokenizer, {'calibration': [Example('c', CORPUS[0], CORPUS[1])]})
    summary = vocab.trim_export(tmp_path / 'export', tmp_path / 'trimmed', kept)
    assert summary['source_checkpoint_sha256'] == checkpoint_identity(tmp_path / 'export')
    removed = (len(tokenizer) + 8 - len(kept)) * 16 * (1 if tied else 2)
    assert summary['parameters_after'] == count_parameters(model) - removed
    trimmed, wrapped = vocab.load_trimmed(tmp_path / 'trimmed')
    assert trimmed.config.vocab_size == len(kept)
    assert (trimmed.lm_head.weight.data_ptr() == trimmed.model.embed_tokens.weight.data_ptr()) == tied
    assert trimmed.config.eos_token_id == kept.index(tokenizer.eos_token_id)
    assert trimmed.generation_config.eos_token_id == kept.index(tokenizer.eos_token_id)
    check = vocab.logit_check(tmp_path / 'export', tmp_path / 'trimmed', [Example('c', CORPUS[0], CORPUS[1])])
    assert check['checked'] == 1 and check['passed'] and check['max_abs_difference'] == 0
    saved = json.loads((tmp_path / 'trimmed' / vocab.MAP_FILE).read_text())
    assert saved['kept_old_ids'] == kept and json.loads((tmp_path / 'trimmed' / 'manifest.json').read_text())['vocab_trim']['tied'] == tied


def test_generation_identical_when_no_removed_token_is_involved(tmp_path):
    from prune_lab.artifacts import save_checkpoint
    from prune_lab.support_evaluation import evaluate_support
    tokenizer = byte_tokenizer()
    model = tiny_model(tokenizer)
    data = [Example(f'k{i}', text, CORPUS[1]) for i, text in enumerate(CORPUS[:3])]
    original = evaluate_support(model, tokenizer, data, 256, 6)
    used = set()
    for row, ex in zip(original['predictions'], data):
        used.update(tokenizer.encode(ex.prompt + '\nAnswer:') + tokenizer.encode(ex.answer, add_special_tokens=False))
    emitted = set()
    for ex in data:
        ids = torch.tensor([tokenizer.encode(ex.prompt + '\nAnswer:')])
        emitted.update(model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=6,
                                      pad_token_id=tokenizer.pad_token_id)[0].tolist())
    special = vocab.special_ids(tokenizer, vocab.BPEStructure(tokenizer))
    kept = sorted(used | emitted | special)
    assert len(kept) < len(tokenizer)
    save_checkpoint(model, tokenizer, tmp_path / 'export', {})
    vocab.trim_export(tmp_path / 'export', tmp_path / 'trimmed', kept)
    trimmed, wrapped = vocab.load_trimmed(tmp_path / 'trimmed')
    for batch in (1, 2):
        result = evaluate_support(trimmed, wrapped, data, 256, 6, batch)
        strip = lambda rows: [(r['key'], r['prediction'], r['generated_tokens']) for r in rows]
        assert strip(result['predictions']) == strip(original['predictions'])
    # Answer CE is normalized over the kept vocabulary only, so it can only drop: not comparable across trimming.
    assert result['answer_token_loss'] <= original['answer_token_loss'] + 1e-6


def test_fallback_statistics():
    tokenizer = byte_tokenizer()
    kept, _ = vocab.build_kept_set(tokenizer, {'recovery': [Example('a', 'renew', 'renew')]})
    stats = vocab.fallback_statistics(tokenizer, kept, [Example('d', 'renew', 'renew'), Example('e', 'eligibility evidence', 'veterans')])
    assert stats['examples'] == 2 and stats['examples_with_prompt_fallback'] == 1
    assert 0 < stats['prompt_fallback_share'] < 1 and stats['reference_tokens_not_emittable'] > 0


def test_trim_run_assembles_records_for_aggregation(tmp_path, monkeypatch):
    from prune_lab import campaign, multiseed, support_data
    from prune_lab.artifacts import save_checkpoint
    from test_support_data import fixture_archive
    monkeypatch.chdir(tmp_path)
    torch.set_num_threads(1)
    tokenizer = byte_tokenizer()
    save_checkpoint(tiny_model(tokenizer), tokenizer, tmp_path / 'source', {})
    counts = {'recovery': 2, 'calibration': 1, 'dev': 3, 'test': 1}
    support_data.prepare_support_dataset(fixture_archive(tmp_path, monkeypatch), tmp_path / 'prepared', tokenizer,
                                         counts=counts, max_length=1024, max_answer_tokens=60)
    config = campaign.read_campaign_config()
    config.update(task='support', dataset='doc2dial', support_data_dir=str(tmp_path / 'prepared'), model=str(tmp_path / 'source'),
                  device='cpu', cpu_threads=1, counts=counts, targets=[.15], group_size=1, sensitivity_examples=1,
                  max_length=1024, max_new_tokens=60, selection_reference='unpruned-recovery', select_uniform=True,
                  evaluate_all_candidates=True)
    config['recovery'].update(rank=2, max_steps=1, checkpoints=[1], gradient_accumulation=1)
    multiseed.run_seeds(config, tmp_path / 'runs' / 'ms', [5])
    test_keys = {json.loads(l)['key'] for l in (tmp_path / 'prepared' / 'test.jsonl').read_text().splitlines()}
    output = tmp_path / 'runs' / 'vocab'
    vocab.trim_run(tmp_path / 'runs' / 'ms', output, ['unpruned-recovery', 'uniform-10'], check_prompts=1)
    result = json.loads((output / 'seed-5' / 'vocab-unpruned' / 'result.json').read_text())
    assert result['kept_set']['from_splits'] == ['calibration', 'recovery']
    assert result['parameters'] < result['source_parameters'] and result['logit_check']['passed']
    assert not test_keys & {r['key'] for r in result['dev']['predictions']}
    summary = multiseed.aggregate_run(output, samples=10)
    assert {'vocab-unpruned', 'vocab-uniform-10', 'unpruned-recovery', 'uniform-10'} <= set(summary['variants'])
    with pytest.raises(ValueError, match='per-layer-99'):
        vocab.trim_run(tmp_path / 'runs' / 'ms', output, ['per-layer-99'])
    real = vocab.logit_check
    monkeypatch.setattr(vocab, 'logit_check', lambda *a, **k: dict(real(*a, **k), passed=False))
    failing = tmp_path / 'runs' / 'vocab-failing'
    with pytest.raises(ValueError, match='seed-5/vocab-unpruned.*seed-5/vocab-uniform-10'):
        vocab.trim_run(tmp_path / 'runs' / 'ms', failing, ['unpruned-recovery', 'uniform-10'], check_prompts=1)
    assert (failing / 'seed-5' / 'vocab-uniform-10' / 'result.json').exists()  # other variants still finished
    monkeypatch.setattr(vocab, 'trim_export', lambda *a, **k: pytest.fail('resume must not redo work'))
    with pytest.raises(ValueError, match='logit exactness'):  # remembered on resume
        vocab.trim_run(tmp_path / 'runs' / 'ms', failing, ['unpruned-recovery', 'uniform-10'], check_prompts=1)
    assert json.loads((failing / 'trim-vocab-run.json').read_text())['logit_check_failures'] == [
        'seed-5/vocab-unpruned', 'seed-5/vocab-uniform-10']
