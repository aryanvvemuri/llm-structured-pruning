"""Improvement 8d: speed ratio arithmetic and the benchmark script on tiny CPU exports."""
import json
from pathlib import Path
import random
import runpy
import statistics
import sys

import pytest
import torch

from prune_lab import speed

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'speed_benchmark.py'


def test_ratio_interval_matches_a_hand_bootstrap():
    reference = [100.0, 104.0, 98.0, 101.0, 97.0]
    model = [120.0, 126.0, 119.0, 118.0, 121.0]
    result = speed.ratio_interval(reference, model, samples=300, seed=5)
    assert result['ratio'] == statistics.median(model) / statistics.median(reference)
    rng = random.Random(5)
    ratios = []
    for _ in range(300):
        picks = [rng.randrange(5) for _ in range(5)]
        ratios.append(statistics.median(model[i] for i in picks) / statistics.median(reference[i] for i in picks))
    ratios.sort()
    assert result['interval_95'] == [ratios[7], ratios[292]]
    same = speed.ratio_interval(reference, reference, samples=50)
    assert same['ratio'] == 1 and same['interval_95'] == [1, 1]
    double = speed.ratio_interval(reference, [2 * v for v in reference], samples=50)
    assert double['ratio'] == pytest.approx(2) and double['interval_95'] == pytest.approx([2, 2])
    for bad in (([1.0], [1.0]), ([1.0, 2.0], [1.0]), ([1.0, -1.0], [1.0, 1.0]), ([1.0, float('nan')], [1.0, 1.0])):
        with pytest.raises(ValueError):
            speed.ratio_interval(*bad)


def test_verdict_directions_and_summary():
    assert speed.verdict('batch_decode_tokens_per_second', [1.05, 1.2]) == 'faster'
    assert speed.verdict('decode_ms_per_token', [1.05, 1.2]) == 'slower'
    assert speed.verdict('decode_ms_per_token', [0.8, 0.9]) == 'faster'
    assert speed.verdict('batch_decode_tokens_per_second', [0.9, 1.1]) == 'not established'
    runs = {'ref': [{'batch_decode_tokens_per_second': v, 'decode_ms_per_token': 10.0} for v in (100, 101, 99)],
            'fast': [{'batch_decode_tokens_per_second': v, 'decode_ms_per_token': 8.0} for v in (130, 131, 129)]}
    summary = speed.summarize(runs, 'ref', samples=50)
    assert 'vs_reference' not in summary['ref']
    fast = summary['fast']['vs_reference']
    assert fast['batch_decode_tokens_per_second']['confidently'] == 'faster'
    assert fast['decode_ms_per_token']['ratio'] == pytest.approx(.8) and fast['decode_ms_per_token']['confidently'] == 'faster'
    with pytest.raises(ValueError):
        speed.summarize(runs, 'missing')


def exports(tmp_path, monkeypatch):
    """Plain, MLP-pruned, reduced-depth and reduced-depth + pruned exports of one tiny model."""
    from prune_lab.artifacts import save_checkpoint
    from prune_lab.layerdrop import drop_layers
    from prune_lab.modeling import load_model
    from prune_lab.pruning import compact_model
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from test_multiseed import tiny_support_config
    config = tiny_support_config(tmp_path, monkeypatch)
    torch.manual_seed(5)
    Qwen2ForCausalLM(Qwen2Config(vocab_size=5, hidden_size=16, intermediate_size=24, num_hidden_layers=4, num_attention_heads=2,
                                 num_key_value_heads=1, max_position_embeddings=256, pad_token_id=0,
                                 eos_token_id=2)).save_pretrained(config['model'], safe_serialization=True)
    paths = {}
    for name, drop, width in (('plain', None, None), ('pruned', None, 16), ('depth', [1], None), ('depth-pruned', [1, 2], 12)):
        model, tokenizer = load_model(config['model'], device='cpu')
        if drop:
            drop_layers(model, drop)
        if width:
            compact_model(model, {i: list(range(width)) for i in range(len(model.model.layers))})
        paths[name] = tmp_path / 'exports' / name
        save_checkpoint(model, tokenizer, paths[name], {'test': True})
    return paths, Path(config['support_data_dir']) / 'dev.jsonl'


def test_benchmark_script_on_tiny_cpu_exports(tmp_path, monkeypatch):
    paths, dev = exports(tmp_path, monkeypatch)
    output = tmp_path / 'speed.json'
    monkeypatch.setattr(sys, 'argv', ['speed_benchmark.py', '--reference', 'plain', '--dev', str(dev), '--output', str(output),
                                      '--repeats', '3', '--prompts', '2', '--steps', '2', '--batch', '2', '--device', 'cpu',
                                      '--dtypes', 'float32', '--bootstrap-samples', '20',
                                      *[f'{n}={p}' for n, p in paths.items()]])
    runpy.run_path(str(SCRIPT), run_name='__main__')
    report = json.loads(output.read_text())
    block = report['results']['pytorch-float32']
    assert [block['models'][n]['layers'] for n in paths] == [4, 4, 3, 2]
    assert block['models']['depth-pruned']['mlp_widths'] == [12]
    assert all(len(block['runs'][n]) == 3 for n in paths)
    assert set(block['summary']['depth']['vs_reference']) == set(speed.METRICS)
    assert 'Speed benchmark' in output.with_suffix('.md').read_text()
    with pytest.raises(SystemExit):
        monkeypatch.setattr(sys, 'argv', ['speed_benchmark.py', '--reference', 'nope', '--dev', str(dev),
                                          '--output', str(output), f'plain={paths["plain"]}'])
        runpy.run_path(str(SCRIPT), run_name='__main__')


mx = pytest.importorskip('mlx.core', reason='mlx-lm not installed')
pytest.importorskip('mlx_lm', reason='mlx-lm not installed')


@pytest.fixture
def mlx_cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.mark.parametrize('drop, width', [([1], None), ([0, 2], 64)])
def test_reduced_depth_exports_convert_and_pass_the_logit_gate(tmp_path, mlx_cpu, drop, width):
    from prune_lab import quantize
    from prune_lab.artifacts import save_checkpoint
    from prune_lab.layerdrop import drop_layers
    from prune_lab.pruning import compact_model
    from test_quantize import examples, tiny_export
    model, tokenizer, _ = tiny_export(tmp_path / 'base', None)
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(4)
    deep = Qwen2ForCausalLM(Qwen2Config(vocab_size=40, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
                                        pad_token_id=0, eos_token_id=2, tie_word_embeddings=True)).eval()
    deep.generation_config = model.generation_config
    drop_layers(deep, drop)
    if width:
        compact_model(deep, {i: list(range(width)) for i in range(len(deep.model.layers))})
    save_checkpoint(deep, tokenizer, tmp_path / 'export', {'test': True})
    for precision in ('fp16', '8bit'):
        record = quantize.convert(tmp_path / 'export', tmp_path / precision, precision)
        converted, config = quantize.load_mlx(tmp_path / precision)
        assert len(converted.layers) == 4 - len(drop) and config['num_hidden_layers'] == 4 - len(drop)
        if width:
            assert record['layer_widths'] == [width] * (4 - len(drop))
    converted, _ = quantize.load_mlx(tmp_path / 'fp16')
    check = quantize.logit_check(tmp_path / 'export', converted, tokenizer, examples(3))
    assert check['passed'] and check['gate_version'] == 2


def test_benchmark_script_mlx_8bit_mode(tmp_path, monkeypatch, mlx_cpu):
    paths, dev = exports(tmp_path, monkeypatch)
    output = tmp_path / 'speed-mlx.json'
    monkeypatch.setattr(sys, 'argv', ['speed_benchmark.py', '--reference', 'plain', '--dev', str(dev), '--output', str(output),
                                      '--repeats', '2', '--prompts', '2', '--steps', '2', '--batch', '2', '--device', 'cpu',
                                      '--dtypes', '', '--mlx-8bit', '--mlx-dir', str(tmp_path / 'mlx'), '--bootstrap-samples', '20',
                                      f"plain={paths['plain']}", f"depth-pruned={paths['depth-pruned']}"])
    runpy.run_path(str(SCRIPT), run_name='__main__')
    block = json.loads(output.read_text())['results']['mlx-8bit']
    assert set(block['summary']['depth-pruned']['vs_reference']) == {'prefill_ms_per_prompt', 'decode_ms_per_token'}
    assert (tmp_path / 'mlx' / 'depth-pruned' / 'conversion.json').exists()
