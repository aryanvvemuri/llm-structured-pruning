"""Independent tests of the improvement-8 revision (4c63ccd): layers-3, greedy noise check,
speed ratio arithmetic and benchmark script, MLX conversion of reduced-depth exports."""
import copy
import json
import math
from pathlib import Path
import random
import statistics

import pytest
import torch

from prune_lab import campaign, layerdrop, speed
from prune_lab.modeling import count_parameters
from test_independent_layer_drop import (LAYER_DROP, base, batches, fit, make_identity, meta_qwen, tiny,  # noqa: F401
                                         with_drop)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'speed_benchmark.py'


# --- (1) layers-3 -------------------------------------------------------------------------------

def test_layers_3_real_shape_and_shipped_config():
    model = meta_qwen()
    three = copy.deepcopy(model)
    layerdrop.drop_layers(three, [5, 9, 17])
    assert count_parameters(three) == 449_295_616
    config = campaign.read_campaign_config(ROOT / 'configs' / 'support-v2-layer-drop.json')
    campaign.validate_config(config)
    assert [s['id'] for s in campaign.candidate_specs(config)] == ['unpruned-recovery', 'layers-3', 'layers-3-uniform-20', 'layers-7']
    specs = {s['id']: s for s in campaign.candidate_specs(config)}
    assert specs['layers-3']['drop'] == 3 and 'total_target' not in specs['layers-3']
    assert set(layerdrop.check_shape(config, model)) == {'layers-3', 'layers-3-uniform-20', 'layers-7'}


def test_layers_n_share_one_scan_and_prefixes(base, monkeypatch):
    config, tmp = base
    drop = {'examples': 3, 'candidates': [{'id': 'layers-1', 'drop': 1},
                                          {'id': 'layers-1-uniform-30', 'drop': 1, 'total_target': .3},
                                          {'id': 'layers-2', 'drop': 2}]}
    cfg = with_drop(config, drop, train=('unpruned-recovery', 'layers-1', 'layers-1-uniform-30', 'layers-2'))
    scans = []
    real = layerdrop.scan_and_select
    monkeypatch.setattr(layerdrop, 'scan_and_select', lambda m, b, s: scans.append(s) or real(m, b, s))
    records = fit(cfg, tmp / 'run')
    assert scans == [2]
    allocation = json.loads((tmp / 'run' / 'allocation.json').read_text())
    order = allocation['layer_scan']['order']
    specs = {c['id']: c for c in allocation['candidates']}
    assert specs['layers-1']['dropped_layers'] == specs['layers-1-uniform-30']['dropped_layers'] == order[:1]
    assert specs['layers-2']['dropped_layers'] == sorted(order[:2])
    assert records['layers-1']['layer_widths'] == [64] * 4 and specs['layers-1']['keep_indices'] is None
    assert records['layers-1']['parameters'] > records['layers-1-uniform-30']['parameters']


# --- (2) greedy noise check -----------------------------------------------------------------------

def per_example(model, data):
    out = []
    with torch.no_grad():
        for b in data:
            out.append(float(model(**b, use_cache=False).loss))
    return out


def token_weighted(model, data):
    total = tokens = 0.0
    with torch.no_grad():
        for b in data:
            n = int((b['labels'][:, 1:] != -100).sum())
            total += float(model(**b, use_cache=False).loss) * n
            tokens += n
    return total / tokens


@pytest.mark.parametrize('seed', [0, 3])
def test_noise_fields_match_recomputation_and_never_change_selection(seed, monkeypatch):
    model = tiny(layers=6, seed=seed)
    data = batches(5, seed=seed + 20)
    record = layerdrop.scan_and_select(model, data, 3)
    removed = []
    for step, entry in enumerate(record['greedy']):
        rows = []
        for layer in range(6):
            if layer in removed:
                continue
            candidate = copy.deepcopy(model)
            layerdrop.drop_layers(candidate, removed + [layer])
            rows.append((token_weighted(candidate, data), -layer, layer, per_example(candidate, data)))
        rows.sort()
        best, runner = rows[0], rows[1]
        assert entry['removed_layer'] == best[2] and entry['runner_up_layer'] == runner[2]
        assert entry['runner_up_loss'] == pytest.approx(runner[0], rel=1e-6)
        assert entry['gap'] == pytest.approx(runner[0] - best[0], rel=1e-5, abs=1e-9)
        diffs = [a - b for a, b in zip(runner[3], best[3])]
        se = statistics.stdev(diffs) / math.sqrt(len(diffs))
        assert entry['paired_standard_error'] == pytest.approx(se, rel=1e-5)
        assert entry['distinguishable'] == (entry['gap'] >= 2 * entry['paired_standard_error'])
        removed.append(best[2])
    # The noise check is purely descriptive: a scan without it (forcing NaN SE) selects the same order.
    monkeypatch.setattr(layerdrop, 'paired_standard_error', lambda a, b: math.nan)
    again = layerdrop.scan_and_select(model, data, 3)
    assert again['order'] == record['order']
    assert all(g['paired_standard_error'] is None and g['distinguishable'] is False for g in again['greedy'])


def test_single_example_gives_null_se_and_round_trips():
    model = tiny(layers=4, seed=1)
    record = layerdrop.scan_and_select(model, batches(1), 2)
    assert all(g['paired_standard_error'] is None and g['distinguishable'] is False for g in record['greedy'])
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert campaign._fingerprint(json.loads(json.dumps(record))) == campaign._fingerprint(record)


@pytest.mark.xfail(strict=True, reason='BUG (low, descriptive field only): an exact tie between the chosen layer and the '
                   'runner-up (gap 0, every per-example difference 0, so SE 0) is flagged distinguishable=True because '
                   'the rule is gap >= 2*SE; a tie can never be distinguishable.')
def test_exact_tie_is_not_distinguishable():
    model = tiny(layers=5, seed=4)
    for layer in range(5):
        make_identity(model, layer)
    record = layerdrop.scan_and_select(model, batches(3), 1)
    assert record['greedy'][0]['gap'] == 0
    assert record['greedy'][0]['distinguishable'] is False


def test_campaign_lock_stable_with_noise_fields(base):
    config, tmp = base
    fit(with_drop(config), tmp / 'run')
    allocation = json.loads((tmp / 'run' / 'allocation.json').read_text())
    for g in allocation['layer_scan']['greedy']:
        assert {'runner_up_layer', 'runner_up_loss', 'gap', 'paired_standard_error', 'distinguishable'} <= set(g)
    selection = json.loads((tmp / 'run' / 'selection.json').read_text())
    assert campaign._fingerprint(json.loads(json.dumps(selection))) == \
        json.loads((tmp / 'run' / 'selection-lock.json').read_text())['sha256']


# --- (3) speed arithmetic --------------------------------------------------------------------------

def my_bootstrap(reference, model, samples, seed):
    rng = random.Random(seed)
    n = len(reference)
    ratios = []
    for _ in range(samples):
        picks = [rng.randrange(n) for _ in range(n)]
        ratios.append(statistics.median([model[i] for i in picks]) / statistics.median([reference[i] for i in picks]))
    ratios.sort()
    return ratios[int(samples * .025)], ratios[min(samples - 1, int(samples * .975))]


@pytest.mark.parametrize('n', [2, 3, 7, 20])
def test_ratio_interval_against_own_bootstrap(n):
    rng = random.Random(n)
    reference = [100 + rng.gauss(0, 3) for _ in range(n)]
    model = [130 + rng.gauss(0, 3) for _ in range(n)]
    result = speed.ratio_interval(reference, model, samples=500, seed=9)
    assert result['ratio'] == statistics.median(model) / statistics.median(reference)
    assert tuple(result['interval_95']) == my_bootstrap(reference, model, 500, 9)
    assert result['interval_95'][0] <= result['ratio'] <= result['interval_95'][1]
    assert result['repeats'] == n


def test_ratio_pairs_repeats_jointly():
    # A shared per-repeat slowdown cancels in a joint resample; an unpaired resample would not.
    drift = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    reference = [100 * d for d in drift]
    model = [80 * d for d in drift]
    result = speed.ratio_interval(reference, model, samples=400, seed=1)
    assert result['interval_95'] == pytest.approx([.8, .8])


def test_degenerate_inputs():
    same = speed.ratio_interval([5.0, 5.0, 5.0], [5.0, 5.0, 5.0], samples=50)
    assert same['ratio'] == 1 and same['interval_95'] == [1, 1]
    assert speed.verdict('batch_decode_tokens_per_second', same['interval_95']) == 'not established'
    for bad in (([5.0], [5.0]), ([0.0, 1.0], [1.0, 1.0]), ([1.0, 1.0], [0.0, 1.0]), ([1.0, float('inf')], [1.0, 1.0]),
                ([1.0, 2.0], [1.0, 2.0, 3.0]), ([], [])):
        with pytest.raises(ValueError):
            speed.ratio_interval(*bad)
    for samples in (0, 1, 2.5, True):
        with pytest.raises(ValueError):
            speed.ratio_interval([1.0, 2.0], [1.0, 2.0], samples=samples)


def test_verdict_directions():
    for metric, direction in speed.METRICS.items():
        faster_interval = [1.1, 1.3] if direction == 'higher' else [.7, .9]
        slower_interval = [.7, .9] if direction == 'higher' else [1.1, 1.3]
        assert speed.verdict(metric, faster_interval) == 'faster'
        assert speed.verdict(metric, slower_interval) == 'slower'
        assert speed.verdict(metric, [.95, 1.05]) == 'not established'
        assert speed.verdict(metric, [1.0, 1.2]) == 'not established'  # touching 1 is not confident
    assert speed.METRICS[speed.PRIMARY] == 'higher'


def test_summarize_directions_end_to_end():
    def runs(tps, ms, k):
        return [{'batch_decode_tokens_per_second': tps * (1 + .01 * i), 'decode_ms_per_token': ms * (1 + .01 * i),
                 'prefill_ms_per_prompt': 10.0 + i, 'batch_prefill_ms_per_prompt': 5.0 + i} for i in range(k)]
    data = {'ref': runs(100, 20, 6), 'fast': runs(150, 13, 6), 'slow': runs(70, 30, 6)}
    summary = speed.summarize(data, 'ref', samples=200)
    assert 'vs_reference' not in summary['ref']
    assert summary['fast']['vs_reference']['batch_decode_tokens_per_second']['confidently'] == 'faster'
    assert summary['fast']['vs_reference']['decode_ms_per_token']['confidently'] == 'faster'
    assert summary['slow']['vs_reference']['batch_decode_tokens_per_second']['confidently'] == 'slower'
    assert summary['slow']['vs_reference']['decode_ms_per_token']['confidently'] == 'slower'
    assert summary['fast']['vs_reference']['prefill_ms_per_prompt']['confidently'] == 'not established'
    assert summary['fast']['medians']['decode_ms_per_token'] == statistics.median(r['decode_ms_per_token'] for r in data['fast'])
    with pytest.raises(ValueError):
        speed.summarize(data, 'missing')


# --- (3) benchmark script ---------------------------------------------------------------------------

def exports(tmp_path, hidden=64):
    """Three tiny exports with the same tokenizer: intact, one layer dropped, dropped + pruned."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    from prune_lab.artifacts import save_checkpoint
    from prune_lab.pruning import compact_model
    vocab = {'[PAD]': 0, '[UNK]': 1, '[EOS]': 2, **{f'w{i}': i + 3 for i in range(60)}}
    backend = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]', eos_token='[EOS]')
    paths = {}
    for name, drop, keep in (('intact', None, None), ('drop1', [1], None), ('drop1-pruned', [2], [96, 64, 128])):
        torch.manual_seed(3)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=hidden, intermediate_size=128, num_hidden_layers=4,
                                             num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
                                             pad_token_id=0, eos_token_id=2, tie_word_embeddings=True)).eval()
        if drop:
            layerdrop.drop_layers(model, drop)
        if keep:
            compact_model(model, {i: list(range(w)) for i, w in enumerate(keep)})
        save_checkpoint(model, tokenizer, tmp_path / name, {'test': True})
        paths[name] = tmp_path / name
    return paths, tokenizer


def dev_file(tmp_path, n=5):
    from dataclasses import asdict
    from prune_lab.support_data import SupportExample, digest
    rng = random.Random(0)
    rows = []
    for i in range(n):
        prompt = ' '.join(f'w{rng.randrange(60)}' for _ in range(3 + 2 * i))
        rows.append(SupportExample(prompt=prompt, answer='w1 w2', key=f'k{n - i:02d}', doc_id='d', dialogue_id=f'd{i}', domain='a',
                                   doc_family='f', group_id='f', doc_text_sha256='x', source_split='train', target_turn_id=2,
                                   source_section_ids=['s'], history_turn_ids=[1], source_context_sha256='c',
                                   dialogue_context_sha256=f'c{i}', response_sha256=digest('w1 w2'), prompt_sha256=digest(prompt),
                                   prompt_tokens=1, answer_tokens=1))
    path = tmp_path / 'dev.jsonl'
    path.write_text(''.join(json.dumps(asdict(r)) + '\n' for r in rows))
    return path, rows


def load_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location('speed_benchmark_under_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_interleaves_rotated_with_same_prompts_and_steps(tmp_path):
    paths, tokenizer = exports(tmp_path)
    dev, rows = dev_file(tmp_path, 5)
    script = load_script()
    calls = []
    real = script.measure_once

    def spy(timers, model, prompts, pad, steps, batch):
        calls.append({'model': id(model), 'layers': len(model.model.layers), 'prompts': [list(p) for p in prompts],
                      'pad': pad, 'steps': steps, 'batch': batch})
        return real(timers, model, prompts, pad, steps, batch)
    script.measure_once = spy
    out = tmp_path / 'report' / 'speed.json'
    out.parent.mkdir()
    script.main(['intact=' + str(paths['intact']), 'drop1=' + str(paths['drop1']), 'drop1-pruned=' + str(paths['drop1-pruned']),
                '--reference', 'intact', '--dev', str(dev), '--output', str(out), '--repeats', '3', '--prompts', '3',
                '--steps', '4', '--batch', '2', '--device', 'cpu', '--dtypes', 'float32', '--bootstrap-samples', '50'])
    assert len(calls) == 9
    layers = [c['layers'] for c in calls]
    assert layers == [4, 3, 3, 3, 3, 4, 3, 4, 3]  # repeat r visits models rotated by r
    order = [c['model'] for c in calls]
    assert order[3:6] == order[1:3] + order[:1] and order[6:9] == order[2:3] + order[:2]
    assert len({json.dumps(c['prompts']) for c in calls}) == 1 and {c['steps'] for c in calls} == {4}
    assert {c['batch'] for c in calls} == {2} and {c['pad'] for c in calls} == {0}
    first_three = sorted(rows, key=lambda r: r.key)[:3]
    from prune_lab.data import prompt_ids
    assert calls[0]['prompts'] == [list(prompt_ids(tokenizer, r.prompt)) for r in first_three]
    report = json.loads(out.read_text())
    assert report['prompt_keys'] == [r.key for r in first_three]
    block = report['results']['pytorch-float32']
    assert all(len(v) == 3 for v in block['runs'].values())
    assert set(block['summary']['drop1']['vs_reference']) == set(speed.METRICS)
    assert (tmp_path / 'report' / 'speed.md').exists()
    with pytest.raises(SystemExit):
        script.main(['intact=' + str(paths['intact']), '--reference', 'intact', '--dev', str(dev),
                    '--output', str(tmp_path / 'runs' / 'x.json'), '--device', 'cpu'])


class Recorder:
    """Delegating model wrapper that records the greedy next token of every forward call."""

    def __init__(self, model):
        self.model, self.tokens = model, []

    def __call__(self, **kwargs):
        out = self.model(**kwargs)
        self.tokens.append(out.logits[:, -1].argmax(-1).tolist())
        return out


@pytest.mark.parametrize('name', ['intact', 'drop1', 'drop1-pruned'])
def test_left_padded_batch_decoding_matches_batch_one(tmp_path, name):
    """The batched timer must compute the same tokens as batch one (else it times a different computation)."""
    paths, tokenizer = exports(tmp_path)
    _, rows = dev_file(tmp_path, 6)
    script = load_script()
    from prune_lab.data import prompt_ids
    from prune_lab.modeling import load_model
    model, tok = load_model(str(paths[name]), device='cpu', dtype='float32')
    prompts = [list(prompt_ids(tok, r.prompt)) for r in rows]
    assert len({len(p) for p in prompts}) > 1  # padding really happens
    single, batched = script.make_timers(torch, 'cpu')
    with torch.inference_mode():
        one = []
        for ids in prompts:
            rec = Recorder(model)
            single(rec, ids, 6)
            one.append([t[0] for t in rec.tokens])
        rec = Recorder(model)
        batched(rec, prompts, tok.pad_token_id, 6)
    many = [[step[row] for step in rec.tokens] for row in range(len(prompts))]
    assert many == one


# --- (4) MLX conversion of reduced-depth exports ----------------------------------------------------

mx = pytest.importorskip('mlx.core', reason='mlx-lm not installed')
pytest.importorskip('mlx_lm', reason='mlx-lm not installed')


@pytest.fixture
def mlx_cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.mark.parametrize('name', ['drop1', 'drop1-pruned'])
@pytest.mark.parametrize('precision', ['fp16', '8bit'])
def test_reduced_depth_mlx_conversion_passes_gate(tmp_path, mlx_cpu, name, precision):
    from prune_lab import quantize
    paths, tokenizer = exports(tmp_path)
    _, rows = dev_file(tmp_path, 4)
    record = quantize.convert(paths[name], tmp_path / f'{name}-{precision}', precision)
    model, config = quantize.load_mlx(tmp_path / f'{name}-{precision}')
    assert len(model.layers) == 3 == config['num_hidden_layers']
    if name == 'drop1-pruned':
        assert [l.mlp.down_proj.weight.shape[-1] if precision == 'fp16' else l.mlp.gate_proj.weight.shape[0]
                for l in model.layers] == [96, 64, 128]
    check = quantize.logit_check(paths[name], model, tokenizer, rows, 'cpu')
    assert check['passed'], check
    if precision == 'fp16':
        assert check['max_abs_difference'] < 5e-2
    assert record['weight_bytes'] > 0


def test_mlx_conversion_of_wrong_depth_is_caught(tmp_path, mlx_cpu):
    """Sanity: an MLX model that kept the dropped layer's slot would not load the reduced weights."""
    from prune_lab import quantize
    paths, _ = exports(tmp_path)
    quantize.convert(paths['drop1'], tmp_path / 'fp16', 'fp16')
    config = json.loads((tmp_path / 'fp16' / 'config.json').read_text())
    config['num_hidden_layers'] = 4
    (tmp_path / 'fp16' / 'config.json').write_text(json.dumps(config))
    with pytest.raises(Exception):
        quantize.load_mlx(tmp_path / 'fp16')
