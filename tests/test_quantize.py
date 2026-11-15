"""Improvement 4: MLX conversion, per-layer widths, generation parity and record format (tiny models)."""
import json
import random
import sys
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GenerationConfig, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from prune_lab import quantize

WORDS = [f'w{i}' for i in range(37)]


def test_module_imports_and_fails_clearly_without_mlx(monkeypatch):
    for name in ('mlx', 'mlx.core', 'mlx.nn', 'mlx_lm'):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(RuntimeError, match='mlx-lm'):
        quantize.convert('x', 'y', 'fp16')
    from prune_lab import cli
    with pytest.raises(SystemExit):
        cli.main(['quantize', '--source', 'x', '--output', 'y', '--variants', 'unpruned-recovery'])


def test_generation_settings_follow_the_export(tmp_path):
    (tmp_path / 'generation_config.json').write_text(json.dumps(
        {'do_sample': True, 'eos_token_id': [5, 6], 'repetition_penalty': 1.1, 'temperature': .7, 'top_k': 20}))
    settings = quantize.generation_settings(tmp_path, 64)
    assert settings == {'do_sample': False, 'max_new_tokens': 64, 'eos_token_id': [5, 6],
                        'repetition_penalty': 1.1, 'batch_size': 1}
    (tmp_path / 'generation_config.json').write_text(json.dumps({'no_repeat_ngram_size': 3}))
    with pytest.raises(ValueError, match='no_repeat_ngram_size'):
        quantize.generation_settings(tmp_path, 64)


mlx = pytest.importorskip('mlx.core', reason='mlx-lm not installed')
pytest.importorskip('mlx_lm', reason='mlx-lm not installed')


@pytest.fixture(autouse=True)
def mlx_cpu():
    """MLX CPU kernels are exact float32 (Metal fp32 GEMMs differ from PyTorch by ~1e-3) and keep
    these tiny tests off the GPU while a PyTorch run may be using it."""
    previous = mlx.default_device()
    mlx.set_default_device(mlx.cpu)
    yield
    mlx.set_default_device(previous)


def tiny_export(tmp_path, widths=None, penalty=1.3):
    torch.manual_seed(4)
    vocab = {'[PAD]': 0, '[UNK]': 1, '[EOS]': 2, **{w: i + 3 for i, w in enumerate(WORDS)}}
    backend = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]', eos_token='[EOS]')
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=40, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
                                         pad_token_id=0, eos_token_id=2, tie_word_embeddings=True)).eval()
    with torch.no_grad():
        model.lm_head.weight[2] *= 3  # mixed-length replies
    model.generation_config = GenerationConfig(eos_token_id=2, pad_token_id=0, repetition_penalty=penalty)
    if widths is not None:
        from prune_lab.pruning import compact_model
        compact_model(model, {i: list(range(w)) for i, w in enumerate(widths)})
    from prune_lab.artifacts import save_checkpoint
    path = tmp_path / 'export'
    save_checkpoint(model, tokenizer, path, {'test': True})
    return model, tokenizer, path


def examples(n=6):
    rng = random.Random(0)
    return [SimpleNamespace(key=f'k{i}', prompt=' '.join(rng.choice(WORDS) for _ in range(rng.randint(2, 9))),
                            answer='w1 w2 w3', group_id=f'g{i % 2}', domain='d', doc_id='doc', dialogue_id=f'dl{i}')
            for i in range(n)]


@pytest.mark.parametrize('widths', [None, [64, 128]])
def test_fp32_conversion_reproduces_pytorch_logits_and_widths(tmp_path, widths):
    model, tokenizer, export = tiny_export(tmp_path, widths)
    record = quantize.convert(export, tmp_path / 'fp32', 'fp32')
    assert record['layer_widths'] == widths and record['quantized_modules'] == []
    mlx_model, config = quantize.load_mlx(tmp_path / 'fp32')
    if widths:
        assert [layer.mlp.gate_proj.weight.shape[0] for layer in mlx_model.layers] == widths
    ids = [3, 7, 9, 11, 4]
    with torch.no_grad():
        expected = model(input_ids=torch.tensor([ids])).logits[0]
    actual = torch.tensor(mlx_model(mlx.array([ids]))[0].tolist())
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_quantized_files_are_smaller_and_low_bit(tmp_path):
    _, _, export = tiny_export(tmp_path, [64, 128])
    records = {p: quantize.convert(export, tmp_path / p, p) for p in ('fp16', '8bit', '4bit')}
    assert records['4bit']['weight_bytes'] < records['8bit']['weight_bytes'] < records['fp16']['weight_bytes'] < records['fp16']['source_weight_bytes']
    assert records['4bit']['bits'] == 4 and records['4bit']['group_size'] == 64
    assert 'model.layers.0.mlp.gate_proj' in records['4bit']['quantized_modules']
    assert records['4bit']['effective_bits_per_weight'] < 6 < records['8bit']['effective_bits_per_weight'] < 10
    model, config = quantize.load_mlx(tmp_path / '4bit')
    assert config['quantization']['bits'] == 4 and quantize._quantized_paths(model) == records['4bit']['quantized_modules']
    assert isinstance(model(mlx.array([[3, 4, 5]])), mlx.array)


def test_mlx_greedy_equals_pytorch_support_evaluation(tmp_path):
    from prune_lab.support_evaluation import evaluate_support
    model, tokenizer, export = tiny_export(tmp_path, [64, 128], penalty=1.3)
    data = examples()
    reference = evaluate_support(model, tokenizer, data, 128, 8)
    quantize.convert(export, tmp_path / 'fp32', 'fp32')
    mlx_model, _ = quantize.load_mlx(tmp_path / 'fp32')
    settings = quantize.generation_settings(export, 8)
    assert settings['repetition_penalty'] == pytest.approx(1.3)
    result = quantize.evaluate_mlx(mlx_model, tokenizer, data, settings)
    lengths = {r['generated_tokens'] for r in reference['predictions']}
    assert len(lengths) > 1  # precondition: EOS stops at different steps
    drop = ('seconds',)
    assert [{k: v for k, v in r.items() if k not in drop} for r in result['predictions']] == \
        [{k: v for k, v in r.items() if k not in drop} for r in reference['predictions']]
    for key in ('token_f1', 'rouge_l', 'exact_match', 'count', 'supervised_tokens', 'generation_limit_fraction', 'generated_tokens'):
        assert result[key] == reference[key], key
    assert result['answer_token_loss'] == pytest.approx(reference['answer_token_loss'], rel=1e-5)
    assert set(reference) <= set(result)
    timing = quantize.speed(mlx_model, tokenizer, data[:2], settings, repeats=2)
    assert len(timing['runs']) == 2 and timing['runs'][0]['generated_tokens'] == timing['runs'][1]['generated_tokens']


def test_quantize_run_assembles_records_for_aggregation(tmp_path, monkeypatch):
    from prune_lab import campaign, multiseed, support_data
    from test_support_data import fixture_archive
    monkeypatch.chdir(tmp_path)
    torch.set_num_threads(1)
    _, tokenizer, export = tiny_export(tmp_path, None, penalty=1.0)
    counts = {'recovery': 2, 'calibration': 1, 'dev': 3, 'test': 1}
    support_data.prepare_support_dataset(fixture_archive(tmp_path, monkeypatch), tmp_path / 'prepared', tokenizer,
                                         counts=counts, max_length=256, max_answer_tokens=16)
    config = campaign.read_campaign_config()
    config.update(task='support', dataset='doc2dial', support_data_dir=str(tmp_path / 'prepared'), model=str(export),
                  device='cpu', cpu_threads=1, counts=counts, targets=[.15], group_size=64, min_keep_ratio=.5,
                  sensitivity_examples=1, max_length=256, max_new_tokens=8, selection_reference='unpruned-recovery',
                  select_uniform=True, evaluate_all_candidates=True)
    config['recovery'].update(rank=2, max_steps=1, checkpoints=[1], gradient_accumulation=1)
    source = tmp_path / 'runs' / 'ms'
    multiseed.run_seeds(config, source, [5])
    test_keys = {json.loads(l)['key'] for l in (tmp_path / 'prepared' / 'test.jsonl').read_text().splitlines()}
    output = tmp_path / 'runs' / 'quant'
    quantize.quantize_run(source, output, ['unpruned-recovery', 'uniform-10'], ['fp32', '4bit'],
                          check_prompts=1, speed_examples=1, speed_repeats=1)
    seed = output / 'seed-5'
    assert (seed / 'uniform-10-4bit' / 'model' / 'config.json').exists()
    result = json.loads((seed / 'uniform-10-4bit' / 'result.json').read_text())
    assert result['source_checkpoint_sha256'] == campaign.checkpoint_identity(result['source_checkpoint'])
    fp32 = json.loads((seed / 'uniform-10-fp32' / 'result.json').read_text())
    assert fp32['logit_check']['passed'] and fp32['logit_check']['max_abs_difference'] < 1e-3
    assert not test_keys & {r['key'] for r in fp32['dev']['predictions']}
    summary = multiseed.aggregate_run(output, samples=10)
    assert set(summary['variants']) >= {'original', 'uniform-10-pytorch', 'uniform-10-fp32', 'uniform-10-4bit',
                                        'unpruned-recovery-pytorch'}
    pytorch = summary['variants']['uniform-10-pytorch']['metrics']['token_f1']['mean']
    assert summary['variants']['uniform-10-fp32']['metrics']['token_f1']['mean'] == pytest.approx(pytorch)
    calls = []
    monkeypatch.setattr(quantize, 'convert', lambda *a, **k: calls.append(a))
    quantize.quantize_run(source, output, ['unpruned-recovery', 'uniform-10'], ['fp32', '4bit'],
                          check_prompts=1, speed_examples=1, speed_repeats=1)
    assert calls == []  # resumed from result.json
    # A pre-v2 record with a next-token flip and no margin is re-checked from its weights on resume.
    path = seed / 'uniform-10-fp32' / 'result.json'
    record = json.loads(path.read_text())
    legacy = {'prompts': [{k: v for k, v in pr.items() if k not in ('agreeing_positions', 'next_token_margin')}
                          | {'next_token_argmax_equal': False} for pr in record['logit_check']['prompts']],
              'passed': False, 'max_abs_difference': record['logit_check']['max_abs_difference']}
    path.write_text(json.dumps(dict(record, logit_check=legacy)))
    quantize.quantize_run(source, output, ['uniform-10'], ['fp32'], check_prompts=1, speed_examples=1, speed_repeats=1)
    rechecked = json.loads(path.read_text())
    assert rechecked['logit_check']['gate_version'] == 2 and rechecked['logit_check']['passed']
    assert rechecked['legacy_logit_check'] == legacy
    # Without the converted weights it cannot be re-checked and the run fails instead of accepting it.
    path.write_text(json.dumps(dict(record, logit_check=legacy)))
    for weights in (seed / 'uniform-10-fp32' / 'model').glob('*.safetensors'):
        weights.unlink()
    with pytest.raises(ValueError, match='logit check failed'):
        quantize.quantize_run(source, output, ['uniform-10'], ['fp32'], check_prompts=1, speed_examples=1, speed_repeats=1)
    with pytest.raises(ValueError, match='check_prompts'):
        quantize.quantize_run(source, output, ['uniform-10'], ['fp32'], check_prompts=0)


def prompt(key, margin, equal, positions=100, agreeing=100, diff=0.5):
    return {'key': key, 'next_token_margin': margin, 'next_token_argmax_equal': equal, 'positions': positions,
            'agreeing_positions': agreeing, 'max_abs_difference': diff}


def test_logit_gate_tolerates_near_ties_only():
    near_tie = quantize.logit_gate([prompt('a', 0.0015, False), prompt('b', 3.0, True)])
    assert near_tie['passed'] and near_tie['tolerated_near_tie_flips'] == ['a']
    assert near_tie['thresholds'] == {'margin_tolerance': 0.1, 'min_position_agreement': 0.97, 'max_abs_difference': 2.0}
    assert not quantize.logit_gate([prompt('a', 0.5, False)])['decisive_next_token_flips'] == []
    assert not quantize.logit_gate([prompt('a', 0.5, False)])['passed']
    assert not quantize.logit_gate([prompt('a', 3.0, True, agreeing=96)])['passed']
    assert not quantize.logit_gate([prompt('a', 3.0, True, diff=2.5)])['passed']
    assert quantize.logit_gate([prompt('a', 0.5, False)], margin_tolerance=1.0)['passed']
    legacy = {'prompts': [{'positions': 100, 'position_argmax_agreement': 0.99, 'next_token_argmax_equal': False,
                           'max_abs_difference': 0.68}], 'passed': False, 'max_abs_difference': 0.68}
    assert quantize.recorded_gate_failed(legacy) and quantize.legacy_needs_recheck(legacy)  # flip without margin
    no_flip = {'prompts': [dict(legacy['prompts'][0], next_token_argmax_equal=True)], 'max_abs_difference': 0.68}
    assert not quantize.recorded_gate_failed(no_flip)  # pre-margin, no flip: agreement and difference only
    assert quantize.recorded_gate_failed(dict(no_flip, max_abs_difference=3.0))
    assert quantize.recorded_gate_failed({'gate_version': 2, 'passed': False})


def test_logit_check_passes_conversion_and_catches_a_real_bug(tmp_path):
    _, tokenizer, export = tiny_export(tmp_path, None)
    data = examples(3)
    quantize.convert(export, tmp_path / 'fp16', 'fp16')
    model, _ = quantize.load_mlx(tmp_path / 'fp16')
    good = quantize.logit_check(export, model, tokenizer, data)
    assert good['passed'] and good['gate_version'] == 2 and all('next_token_margin' in p for p in good['prompts'])
    # A conversion bug that loads the wrong tensors: layer 1's down_proj gets another matrix of the right shape.
    weight = model.layers[1].mlp.down_proj.weight
    model.layers[1].mlp.down_proj.weight = mlx.random.normal(weight.shape, key=mlx.random.key(0)).astype(weight.dtype)
    bad = quantize.logit_check(export, model, tokenizer, data)
    assert not bad['passed'] and (bad['max_abs_difference'] > 2.0 or bad['position_argmax_agreement'] < .97
                                  or bad['decisive_next_token_flips'])
