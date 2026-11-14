"""Independent tests of improvement 4 (MLX quantization baselines), from plans/improvement-4-quantization.md.

Tiny random Qwen2 exports only; MLX pinned to its CPU device. A hand-built source run
(real exports, real PyTorch dev records, poisoned test split) drives quantize_run.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GenerationConfig, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from prune_lab import quantize

ROOT = Path(__file__).resolve().parents[1]
WORDS = [f'v{i}' for i in range(45)]
VOCAB = 48  # [PAD] [UNK] [EOS] + 45 words


def test_worktree_package_is_under_test():
    assert Path(quantize.__file__).resolve().is_relative_to(ROOT / 'src')


def test_behaviour_without_mlx():
    """prune_lab and the CLI import without mlx; quantize fails clearly before touching the source."""
    code = ("import sys\n"
            "for n in ('mlx','mlx.core','mlx.nn','mlx.utils','mlx_lm','mlx_lm.utils','mlx_lm.models'): sys.modules[n]=None\n"
            "import prune_lab, prune_lab.quantize as q, prune_lab.cli as cli, prune_lab.campaign, prune_lab.multiseed\n"
            "assert q.PRECISIONS and q.generation_settings\n"
            "try:\n  q.quantize_run('/nonexistent-source', '/nonexistent-out', ['a'], ['fp16'])\n"
            "except RuntimeError as e:\n  assert 'mlx-lm' in str(e); print('OK1')\n"
            "try:\n  cli.main(['quantize','--source','/nonexistent','--output','/nonexistent-out','--variants','a'])\n"
            "except SystemExit as e:\n  assert e.code != 0; print('OK2')\n")
    out = subprocess.run([sys.executable, '-c', code], env=dict(os.environ, PYTHONPATH=str(ROOT / 'src')),
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert 'OK1' in out.stdout and 'OK2' in out.stdout
    assert not Path('/nonexistent-out').exists()


mx = pytest.importorskip('mlx.core', reason='mlx-lm not installed')
pytest.importorskip('mlx_lm', reason='mlx-lm not installed')


@pytest.fixture(autouse=True)
def mlx_cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def tokenizer_():
    vocab = {'[PAD]': 0, '[UNK]': 1, '[EOS]': 2, **{w: i + 3 for i, w in enumerate(WORDS)}}
    backend = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]', eos_token='[EOS]')


def make_export(path, widths=None, penalty=1.1, eos=(2,), seed=0, tie=True, eos_boost=2.5, generation_config=True):
    from prune_lab.artifacts import save_checkpoint
    from prune_lab.pruning import compact_model
    torch.manual_seed(seed)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
                                         pad_token_id=0, eos_token_id=2, tie_word_embeddings=tie)).eval()
    with torch.no_grad():
        model.lm_head.weight[2] *= eos_boost
    model.generation_config = GenerationConfig(eos_token_id=list(eos) if len(eos) > 1 else eos[0], pad_token_id=0,
                                               repetition_penalty=penalty)
    if widths is not None:
        rng = random.Random(seed)
        compact_model(model, {i: sorted(rng.sample(range(128), w)) for i, w in enumerate(widths)})
    tokenizer = tokenizer_()
    save_checkpoint(model, tokenizer, path, {'test': True})
    if not generation_config:
        (Path(path) / 'generation_config.json').unlink()
    return model, tokenizer


def logits_mlx(model, ids):
    return torch.tensor(model(mx.array([ids])).astype(mx.float32)[0].tolist())


PROMPTS = [[3, 9, 17, 4], [5, 5, 5, 30, 12, 8, 41], [44], [7, 20, 33, 21, 9, 6, 10, 11, 3, 2]]


# --- per-layer widths and logits ---------------------------------------------------------------------

@pytest.mark.parametrize('widths', [None, [64, 64, 64], [128, 64, 96], [32, 128, 1]])
def test_mlx_logits_equal_pytorch_for_uniform_and_per_layer_widths(tmp_path, widths):
    model, _ = make_export(tmp_path / 'export', widths)
    config = json.loads((tmp_path / 'export' / 'config.json').read_text())
    if widths is not None:
        assert config['prune_lab_layer_widths'] == widths and config['intermediate_size'] == max(widths)
    quantize.convert(tmp_path / 'export', tmp_path / 'fp32', 'fp32')
    mlx_model, _ = quantize.load_mlx(tmp_path / 'fp32')
    got = [layer.mlp.down_proj.weight.shape[1] for layer in mlx_model.layers]
    assert got == (widths or [128] * 3)
    for ids in PROMPTS:
        with torch.no_grad():
            expected = model(input_ids=torch.tensor([ids])).logits[0]
        torch.testing.assert_close(logits_mlx(mlx_model, ids), expected, rtol=1e-5, atol=1e-5)
    quantize.convert(tmp_path / 'export', tmp_path / 'fp16', 'fp16')
    half, _ = quantize.load_mlx(tmp_path / 'fp16')
    for ids in PROMPTS:
        with torch.no_grad():
            expected = model(input_ids=torch.tensor([ids])).logits[0]
        actual = logits_mlx(half, ids)
        assert (actual - expected).abs().max() < 5e-2
        assert int(actual[-1].argmax()) == int(expected[-1].argmax())


def test_wrong_widths_are_refused(tmp_path):
    make_export(tmp_path / 'export', [128, 64, 96])
    config = json.loads((tmp_path / 'export' / 'config.json').read_text())
    for bad in ([128, 64], [128, 64, 0], [128, 96, 64]):
        (tmp_path / 'export' / 'config.json').write_text(json.dumps(dict(config, prune_lab_layer_widths=bad)))
        target = tmp_path / f'out-{"-".join(map(str, bad))}'
        with pytest.raises(Exception):
            quantize.convert(tmp_path / 'export', target, 'fp16')


# --- low-bit files ------------------------------------------------------------------------------------

def mx_nn():
    import mlx.nn as nn
    return nn


@pytest.mark.parametrize('bits', [4, 8])
def test_quantized_files_are_really_low_bit(tmp_path, bits):
    make_export(tmp_path / 'export', [128, 64, 96])
    fp16 = quantize.convert(tmp_path / 'export', tmp_path / 'fp16', 'fp16')
    record = quantize.convert(tmp_path / 'export', tmp_path / 'q', f'{bits}bit')
    stored = {}
    for file in (tmp_path / 'q').glob('*.safetensors'):
        stored.update(mx.load(str(file)))
    source = {}
    for file in (tmp_path / 'export').glob('*.safetensors'):
        source.update(mx.load(str(file)))
    assert record['weight_bytes'] == sum(p.stat().st_size for p in (tmp_path / 'q').glob('*.safetensors'))
    assert record['weight_bytes'] < fp16['weight_bytes'] < fp16['source_weight_bytes']
    assert fp16['source_weight_bytes'] == sum(p.stat().st_size for p in (tmp_path / 'export').glob('*.safetensors'))
    assert all(v.dtype == mx.float16 for v in mx.load(str(next((tmp_path / 'fp16').glob('*.safetensors')))).values())
    # Every 2-D weight whose input dim is a multiple of 64 is packed uint32 with fp16 scales/biases.
    quantized, plain = set(), set()
    for name, value in source.items():
        if value.ndim != 2:
            continue
        module = name[:-len('.weight')]
        if value.shape[-1] % 64 == 0:
            packed = stored[name]
            assert packed.dtype == mx.uint32 and packed.shape == (value.shape[0], value.shape[1] * bits // 32), name
            assert stored[f'{module}.scales'].shape == (value.shape[0], value.shape[1] // 64)
            assert stored[f'{module}.scales'].dtype == mx.float16 and stored[f'{module}.biases'].dtype == mx.float16
            quantized.add(module)
        else:
            assert stored[name].dtype == mx.float16 and f'{module}.scales' not in stored
            plain.add(module)
    assert plain == {'model.layers.2.mlp.down_proj'}  # 96-wide input is not a multiple of the group
    assert set(record['quantized_modules']) == quantized
    assert set(record['unquantized_weight_modules']) == plain
    assert record['bits'] == bits and record['group_size'] == 64 and record['mode'] == 'affine'
    # Effective bits per weight: total stored bits over logical parameters (all tensors, norms included).
    logical = sum(v.size for v in source.values() if not v.dtype == mx.uint32)
    stored_bits = 8 * sum(v.nbytes for v in stored.values())
    assert record['effective_bits_per_weight'] == pytest.approx(stored_bits / logical, rel=1e-9)
    assert bits + 0.4 < record['effective_bits_per_weight'] < 16
    # Dequantized weights are close to the originals (the format really is low-bit affine).
    model, config = quantize.load_mlx(tmp_path / 'q')
    assert config['quantization'] == {'group_size': 64, 'bits': bits, 'mode': 'affine'}
    layer = model.layers[0].mlp.gate_proj
    restored = mx.dequantize(layer.weight, layer.scales, layer.biases, 64, bits)
    error = float(mx.abs(restored - source['model.layers.0.mlp.gate_proj.weight']).max())
    spread = float(mx.abs(source['model.layers.0.mlp.gate_proj.weight']).max())
    assert 0 < error < spread * (0.2 if bits == 4 else 0.02)


# --- generation parity -----------------------------------------------------------------------------

def dev_examples(n=8, seed=0, offset=0):
    from prune_lab.support_data import SupportExample, digest
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        prompt = ' '.join(rng.choice(WORDS) for _ in range(rng.randint(3, 12))) + f' v{(i + offset) % 45}'
        answer = ' '.join(rng.choice(WORDS) for _ in range(3))
        rows.append(SupportExample(prompt=prompt, answer=answer, key=f'ex{offset + i}', doc_id=f'doc{i % 3}',
                                   dialogue_id=f'd{offset + i}', domain=['a', 'b'][i % 2], doc_family=f'fam{i % 3}',
                                   group_id=f'fam{i % 3}', doc_text_sha256='x', source_split='train', target_turn_id=2,
                                   source_section_ids=['s'], history_turn_ids=[1], source_context_sha256=f'c{i}',
                                   dialogue_context_sha256=f'dc{offset + i}', response_sha256=digest(answer),
                                   prompt_sha256=digest(prompt), prompt_tokens=10, answer_tokens=4))
    return rows


def hf_generate(model, ids, max_new_tokens):
    out = model.generate(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                         do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=0, use_cache=True)
    return out[0, len(ids):].tolist()


@pytest.mark.parametrize('penalty,eos,boost', [(1.1, (2,), 2.5), (1.3, (2, 7), 2.0), (1.0, (2,), 1.0), (1.1, (2,), 40.0)])
def test_token_level_parity_with_transformers_generate(tmp_path, penalty, eos, boost):
    model, tokenizer = make_export(tmp_path / 'export', [128, 64, 96], penalty=penalty, eos=eos, eos_boost=boost)
    from prune_lab.modeling import load_model
    reference, _ = load_model(str(tmp_path / 'export'), device='cpu', dtype='float32')
    quantize.convert(tmp_path / 'export', tmp_path / 'fp32', 'fp32')
    mlx_model, _ = quantize.load_mlx(tmp_path / 'fp32')
    for max_new in (1, 2, 5, 12):
        settings = quantize.generation_settings(tmp_path / 'export', max_new)
        assert settings['eos_token_id'] == list(eos) and settings['repetition_penalty'] == pytest.approx(penalty)
        lengths = set()
        for ids in PROMPTS + [[3 + (k * 7) % 45 for k in range(9)]]:
            expected = hf_generate(reference, ids, max_new)
            got = quantize.greedy_generate(mlx_model, ids, settings)
            assert got == expected, (max_new, ids)
            assert len(got) <= max_new
            lengths.add(len(got))
    if boost == 40.0:
        assert lengths == {1}  # EOS first: empty reply


@pytest.mark.parametrize('boost', [8.0, 40.0])
def test_record_for_record_parity_with_support_evaluation(tmp_path, boost):
    from prune_lab.modeling import load_model
    from prune_lab.support_evaluation import evaluate_support
    make_export(tmp_path / 'export', [128, 64, 96], penalty=1.1, eos_boost=boost)  # boost 8: mixed lengths
    reference, tokenizer = load_model(str(tmp_path / 'export'), device='cpu', dtype='float32')
    data = dev_examples()
    expected = evaluate_support(reference, tokenizer, data, 256, 6)
    quantize.convert(tmp_path / 'export', tmp_path / 'fp32', 'fp32')
    mlx_model, _ = quantize.load_mlx(tmp_path / 'fp32')
    got = quantize.evaluate_mlx(mlx_model, tokenizer, data, quantize.generation_settings(tmp_path / 'export', 6))
    strip = lambda rows: [{k: v for k, v in r.items() if k != 'seconds'} for r in rows]
    assert strip(got['predictions']) == strip(expected['predictions'])
    if boost == 40.0:
        empty = [r for r in got['predictions'] if r['generated_tokens'] == 1]
        assert empty and all(r['prediction'] == '' and r['token_f1'] == 0 for r in empty)  # EOS-only reply
    else:
        assert len({r['generated_tokens'] for r in got['predictions']}) > 2  # EOS at several steps and the cap
    for key in ('token_f1', 'rouge_l', 'exact_match', 'count', 'supervised_tokens', 'generated_tokens',
                'generation_limit_fraction', 'selection_score', 'score_name'):
        assert got[key] == expected[key], key
    assert got['answer_token_loss'] == pytest.approx(expected['answer_token_loss'], rel=1e-5)


@pytest.mark.xfail(strict=True, reason='DOUBTFUL (low): with no generation_config.json the MLX loop has no EOS ids '
                   'and never stops, while transformers falls back to config.eos_token_id. Real exports from '
                   'save_checkpoint always write generation_config.json, so the planned run is unaffected.')
def test_missing_generation_config_stops_like_transformers(tmp_path):
    make_export(tmp_path / 'export', None, penalty=1.0, eos_boost=40.0, generation_config=False)
    from prune_lab.modeling import load_model
    reference, _ = load_model(str(tmp_path / 'export'), device='cpu', dtype='float32')
    quantize.convert(tmp_path / 'export', tmp_path / 'fp32', 'fp32')
    mlx_model, _ = quantize.load_mlx(tmp_path / 'fp32')
    settings = quantize.generation_settings(tmp_path / 'export', 4)
    assert quantize.greedy_generate(mlx_model, PROMPTS[0], settings) == hf_generate(reference, PROMPTS[0], 4)


@pytest.mark.xfail(strict=True, reason='PRE-EXISTING (support_evaluation, medium for interpreting "fp16 MLX - PyTorch"): '
                   'batched PyTorch generation (eval_batch_size > 1, used for the v2 source records) left-pads with '
                   'pad_token_id and transformers\' repetition penalty then also penalizes the pad token (for Qwen2.5 '
                   '<|endoftext|>, an EOS id). Replies differ from batch-one greedy, which the MLX loop reproduces, so '
                   'MLX-vs-PyTorch differences are not purely MLX effects.')
def test_batched_pytorch_records_equal_batch_one_under_repetition_penalty(tmp_path):
    from prune_lab.modeling import load_model
    from prune_lab.support_evaluation import evaluate_support
    make_export(tmp_path / 'export', None, eos=(2, 0), eos_boost=1.0, tie=False, penalty=1.3)
    model, tokenizer = load_model(str(tmp_path / 'export'), device='cpu', dtype='float32')
    with torch.no_grad():
        model.lm_head.weight[0] *= 3  # make the pad/EOS id competitive
    data = dev_examples(12)
    one = evaluate_support(model, tokenizer, data, 256, 8)
    batched = evaluate_support(model, tokenizer, data, 256, 8, 4)
    strip = lambda rows: [(r['key'], r['prediction'], r['generated_tokens']) for r in rows]
    assert strip(batched['predictions']) == strip(one['predictions'])


def test_batched_equals_batch_one_without_penalty(tmp_path):
    """Control for the xfail above: with repetition_penalty 1.0 batching changes nothing here."""
    from prune_lab.modeling import load_model
    from prune_lab.support_evaluation import evaluate_support
    make_export(tmp_path / 'export', None, eos=(2, 0), eos_boost=1.0, tie=False, penalty=1.0)
    model, tokenizer = load_model(str(tmp_path / 'export'), device='cpu', dtype='float32')
    with torch.no_grad():
        model.lm_head.weight[0] *= 3
    data = dev_examples(12)
    strip = lambda rows: [(r['key'], r['prediction'], r['generated_tokens']) for r in rows]
    assert strip(evaluate_support(model, tokenizer, data, 256, 8, 4)['predictions']) == \
        strip(evaluate_support(model, tokenizer, data, 256, 8)['predictions'])


# --- quantize_run on a hand-built source run --------------------------------------------------------

def write_jsonl(path, rows):
    from dataclasses import asdict
    path.write_text(''.join(json.dumps(asdict(r)) + '\n' for r in rows))


@pytest.fixture
def source_run(tmp_path, monkeypatch):
    """Two seeds with real exports and real PyTorch dev records; test.jsonl unreadable and poisoned."""
    from prune_lab import campaign
    from prune_lab.data import dataset_manifest
    from prune_lab.modeling import count_parameters, load_model
    from prune_lab.support_evaluation import evaluate_support
    monkeypatch.chdir(tmp_path)
    torch.set_num_threads(1)
    dev, calibration = dev_examples(8, 0, 0), dev_examples(3, 1, 100)
    root = tmp_path / 'source'
    variants = {'unpruned-recovery': None, 'uniform-20': [64, 64, 64], 'per-layer-02': [128, 64, 96]}
    for seed in (5, 6):
        seed_dir = root / f'seed-{seed}'
        (seed_dir / 'data').mkdir(parents=True)
        write_jsonl(seed_dir / 'data' / 'dev.jsonl', dev)
        write_jsonl(seed_dir / 'data' / 'calibration.jsonl', calibration)
        poisoned = seed_dir / 'data' / 'test.jsonl'
        poisoned.write_text('POISON: the test split must never be read\n')
        poisoned.chmod(0)
        config = campaign.read_campaign_config()
        config.update(task='support', dataset='doc2dial', support_data_dir='unused', max_new_tokens=6, max_length=256,
                      selection_dev_examples=3, eval_batch_size=1)
        config['recovery']['seed'] = seed
        (seed_dir / 'config.json').write_text(json.dumps(config))
        state = {'config_sha256': campaign._fingerprint(config), 'test_started': False,
                 'dataset': dict(dataset_manifest({'dev': dev, 'calibration': calibration}),
                                 test={'count': 4, 'sha256': 'sealed'})}
        (seed_dir / 'state.json').write_text(json.dumps(state))
        subset = [r.key for r in dev[:3]]
        (seed_dir / 'selection-dev.json').write_text(json.dumps({'count': 3, 'keys': subset,
                                                                 'keys_sha256': campaign._fingerprint(subset), 'seed': 1}))
        candidates = []
        for k, (name, widths) in enumerate(variants.items()):
            export = seed_dir / 'candidates' / name / 'checkpoint'
            make_export(export, widths, seed=seed * 10 + k)
            model, tokenizer = load_model(str(export), device='cpu', dtype='float32')
            candidates.append({'id': name, 'status': 'complete', 'checkpoint': str(export),
                               'checkpoint_sha256': campaign.checkpoint_identity(export),
                               'parameters': count_parameters(model), 'actual_reduction': .2 if widths else 0.0,
                               'target': .2 if widths else 0.0, 'control': name != 'per-layer-02',
                               'dev': evaluate_support(model, tokenizer, dev, 256, 6)})
        base_model, base_tok = load_model(str(seed_dir / 'candidates' / 'unpruned-recovery' / 'checkpoint'), device='cpu', dtype='float32')
        selection = {'baseline': {'dev': evaluate_support(base_model, base_tok, dev, 256, 6), 'parameters': candidates[0]['parameters']},
                     'candidates': candidates + [{'id': 'uniform-10', 'status': 'failed', 'error': 'x'}], 'tiers': []}
        (seed_dir / 'selection.json').write_text(json.dumps(selection))
        (seed_dir / 'selection-lock.json').write_text(json.dumps({'sha256': campaign._fingerprint(selection)}))
    yield root
    for seed in (5, 6):
        (root / f'seed-{seed}' / 'data' / 'test.jsonl').chmod(0o600)


def run(source, output, variants=('unpruned-recovery', 'per-layer-02'), precisions=('fp16', '4bit'), **kw):
    kw = dict(dict(check_prompts=2, speed_examples=2, speed_repeats=1), **kw)
    return quantize.quantize_run(source, output, list(variants), list(precisions), **kw)


def file_hashes(root):
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and os.access(p, os.R_OK)}


def test_quantize_run_end_to_end_records_membership_and_sealing(source_run, tmp_path, monkeypatch):
    from prune_lab import campaign, multiseed
    opened = []
    real_open = open

    def watch(file, *a, **k):
        opened.append(str(file))
        return real_open(file, *a, **k)
    monkeypatch.setattr('builtins.open', watch)
    before = file_hashes(source_run)
    output = tmp_path / 'quant'
    run(source_run, output, precisions=('fp32', 'fp16', '8bit', '4bit'))
    monkeypatch.setattr('builtins.open', real_open)
    assert not [p for p in opened if p.endswith('test.jsonl')]
    assert file_hashes(source_run) == before  # source untouched
    for seed in (5, 6):
        src, dst = source_run / f'seed-{seed}', output / f'seed-{seed}'
        for name in ('config.json', 'state.json', 'selection-dev.json'):
            assert (dst / name).read_bytes() == (src / name).read_bytes()
        assert not (dst / 'data').exists() and not list(dst.rglob('test.jsonl'))
        source_records = {c['id']: c for c in campaign._selection(src)['candidates']}
        assembled = {c['id']: c for c in campaign._selection(dst)['candidates']}
        assert campaign._selection(dst)['baseline'] == campaign._selection(src)['baseline']
        for variant in ('unpruned-recovery', 'per-layer-02'):
            copy = dict(assembled[f'{variant}-pytorch'])
            assert copy.pop('backend') == 'pytorch' and copy.pop('id') == f'{variant}-pytorch'
            original = dict(source_records[variant])
            original.pop('id')
            assert json.dumps(copy, sort_keys=True) == json.dumps(original, sort_keys=True)
            for precision in ('fp32', 'fp16', '8bit', '4bit'):
                result = assembled[f'{variant}-{precision}']
                assert result['source_checkpoint'] == source_records[variant]['checkpoint']
                assert result['source_checkpoint_sha256'] == source_records[variant]['checkpoint_sha256']
                assert [r['key'] for r in result['dev']['predictions']] == [r['key'] for r in copy['dev']['predictions']]
                assert result['dev']['generation_config']['backend'] == 'mlx'
            # fp32 MLX reproduces the PyTorch records exactly (batch-one source, CPU).
            strip = lambda rows: [{k: v for k, v in r.items() if k != 'seconds'} for r in rows]
            assert strip(assembled[f'{variant}-fp32']['dev']['predictions']) == strip(copy['dev']['predictions'])
            assert assembled[f'{variant}-fp16']['logit_check']['passed']
    a, b = multiseed.collect_run(source_run), multiseed.collect_run(output)
    assert a['selection_keys'] == b['selection_keys'] and a['dataset'] == b['dataset']
    summary = multiseed.aggregate_run(output, samples=20)
    assert {'per-layer-02-pytorch', 'per-layer-02-4bit', 'unpruned-recovery-fp16'} <= set(summary['variants'])
    result = multiseed.compare_runs(source_run, output, pairs=[('per-layer-02', 'per-layer-02-pytorch'),
                                                               ('per-layer-02', 'per-layer-02-fp32')], samples=50)
    assert result['example_set'] == 'dev-holdout' and result['example_count'] == 5
    for comparison in result['comparisons']:
        assert comparison['token_f1']['estimate_points'] == pytest.approx(0, abs=1e-12)
    log = json.loads((output / 'quantize-run.json').read_text())
    assert log['logit_check_failures'] == [] and set(log['versions']) == {'mlx', 'mlx-lm'}


def test_export_hash_and_missing_variant_refused(source_run, tmp_path):
    output = tmp_path / 'q1'
    with pytest.raises(ValueError, match='per-layer-07'):
        run(source_run, output, variants=('per-layer-07',))
    with pytest.raises(ValueError, match='uniform-10'):  # failed in the source: no complete export
        run(source_run, tmp_path / 'q2', variants=('uniform-10',))
    weights = next((source_run / 'seed-5' / 'candidates' / 'uniform-20' / 'checkpoint').glob('*.safetensors'))
    data = bytearray(weights.read_bytes())
    data[-3] ^= 0xFF
    weights.write_bytes(bytes(data))
    with pytest.raises(ValueError, match='hash'):
        run(source_run, tmp_path / 'q3', variants=('uniform-20',))
    assert not list((tmp_path / 'q3').rglob('*.safetensors'))
    with pytest.raises(ValueError):
        run(source_run, tmp_path / 'q4', variants=('uniform-20', 'uniform-20'))
    with pytest.raises(ValueError):
        run(source_run, tmp_path / 'q5', precisions=('3bit',))


@pytest.mark.xfail(strict=True, reason='Low: a variant missing from a later seed is only reported after earlier '
                   'seeds were fully converted and evaluated (no up-front check of every seed).')
def test_missing_variant_detected_before_any_work(source_run, tmp_path, monkeypatch):
    from prune_lab import campaign
    seed_dir = source_run / 'seed-6'
    selection = campaign._selection(seed_dir)
    selection['candidates'] = [c for c in selection['candidates'] if c['id'] != 'per-layer-02']
    (seed_dir / 'selection.json').write_text(json.dumps(selection))
    (seed_dir / 'selection-lock.json').write_text(json.dumps({'sha256': campaign._fingerprint(selection)}))
    calls = []
    monkeypatch.setattr(quantize, 'convert', lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(RuntimeError('converted')))
    with pytest.raises(ValueError, match='per-layer-02'):
        run(source_run, tmp_path / 'q', variants=('per-layer-02',))
    assert calls == []


def test_resume_reuses_results_and_discard_models(source_run, tmp_path, monkeypatch):
    from prune_lab import multiseed
    output = tmp_path / 'quant'
    run(source_run, output, variants=('uniform-20',), precisions=('4bit',), discard_models=True)
    assert not list(output.rglob('*.safetensors'))
    for seed in (5, 6):
        result = json.loads((output / f'seed-{seed}' / 'uniform-20-4bit' / 'result.json').read_text())
        assert result['model_weights_deleted'] and result['on_disk_weight_bytes'] > 0
        assert (output / f'seed-{seed}' / 'uniform-20-4bit' / 'model' / 'config.json').exists()
    saved = (output / 'seed-5' / 'uniform-20-4bit' / 'result.json').read_bytes()
    monkeypatch.setattr(quantize, 'convert', lambda *a, **k: (_ for _ in ()).throw(AssertionError('reconverted')))
    run(source_run, output, variants=('uniform-20',), precisions=('4bit',))
    assert (output / 'seed-5' / 'uniform-20-4bit' / 'result.json').read_bytes() == saved
    assert multiseed.aggregate_run(output, samples=10)['seeds'] == [5, 6]


@pytest.mark.xfail(strict=True, reason='BUG (medium): resume reuses an existing result.json without checking that '
                   'its source_checkpoint_sha256 matches the current export, so re-running into the same output '
                   'after the source export changed silently mixes MLX results of the old export with the '
                   '<variant>-pytorch record of the new one.')
def test_resume_refuses_result_built_from_a_different_export(source_run, tmp_path):
    from prune_lab import campaign
    output = tmp_path / 'quant'
    run(source_run, output, variants=('uniform-20',), precisions=('4bit',), seeds=[5])
    # Replace the source export with a different model (a retrained candidate), records consistent.
    seed_dir = source_run / 'seed-5'
    export = seed_dir / 'candidates' / 'uniform-20' / 'checkpoint'
    for p in export.iterdir():
        p.unlink()
    make_export(export, [64, 64, 64], seed=999)
    selection = campaign._selection(seed_dir)
    for c in selection['candidates']:
        if c['id'] == 'uniform-20':
            c['checkpoint_sha256'] = campaign.checkpoint_identity(export)
    (seed_dir / 'selection.json').write_text(json.dumps(selection))
    (seed_dir / 'selection-lock.json').write_text(json.dumps({'sha256': campaign._fingerprint(selection)}))
    try:
        run(source_run, output, variants=('uniform-20',), precisions=('4bit',), seeds=[5])
    except ValueError:
        return
    result = json.loads((output / 'seed-5' / 'uniform-20-4bit' / 'result.json').read_text())
    assert result['source_checkpoint_sha256'] == campaign.checkpoint_identity(export)


def test_logit_check_failure_recorded_then_nonzero_exit(source_run, tmp_path, monkeypatch):
    from prune_lab import cli
    real = quantize.logit_check
    used = []

    def failing(source, model, tokenizer, examples, device='cpu'):
        used.append([e.key for e in examples])
        result = real(source, model, tokenizer, examples, device)
        return dict(result, passed=False)
    monkeypatch.setattr(quantize, 'logit_check', failing)
    output = tmp_path / 'quant'
    with pytest.raises(ValueError, match='logit check failed'):
        run(source_run, output, variants=('uniform-20',), precisions=('fp16', '4bit'))
    # Checks use calibration prompts only, and every result was still written.
    calibration_keys = {f'ex{100 + i}' for i in range(3)}
    assert used and all(set(keys) <= calibration_keys for keys in used)
    for seed in (5, 6):
        for precision in ('fp16', '4bit'):
            assert (output / f'seed-{seed}' / f'uniform-20-{precision}' / 'result.json').exists()
    log = json.loads((output / 'quantize-run.json').read_text())
    assert log['logit_check_failures'] == ['seed-5/uniform-20-fp16', 'seed-6/uniform-20-fp16']
    monkeypatch.setattr(quantize, 'logit_check', real)
    with pytest.raises(SystemExit) as exit_info:  # resume still reports the recorded failure
        cli.main(['quantize', '--source', str(source_run), '--output', str(output), '--variants', 'uniform-20',
                  '--precisions', 'fp16', '4bit', '--check-prompts', '2', '--speed-examples', '1', '--speed-repeats', '1'])
    assert exit_info.value.code != 0
