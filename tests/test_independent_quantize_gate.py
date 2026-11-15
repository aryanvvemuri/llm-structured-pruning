"""Independent tests of the margin-aware MLX logit gate (quantize.logit_gate, gate_version 2)."""
import json
import math
from pathlib import Path

import pytest

from prune_lab import quantize


def prompt(key='p', positions=100, agreeing=100, diff=0.1, margin=1.0, equal=True):
    return {'key': key, 'positions': positions, 'agreeing_positions': agreeing, 'max_abs_difference': diff,
            'next_token_margin': margin, 'next_token_argmax_equal': equal,
            'position_argmax_agreement': agreeing / positions}


def gate(prompts, **kw):
    return quantize.logit_gate(prompts, **kw)


# --- thresholds and boundaries --------------------------------------------------------------

def test_defaults_and_recorded_thresholds():
    result = gate([prompt()])
    assert result['passed'] and result['thresholds'] == {'margin_tolerance': .1, 'min_position_agreement': .97,
                                                         'max_abs_difference': 2.0}
    custom = gate([prompt()], margin_tolerance=.5, min_position_agreement=.9, max_abs_difference=3.0)
    assert custom['thresholds'] == {'margin_tolerance': .5, 'min_position_agreement': .9, 'max_abs_difference': 3.0}


def test_margin_boundary():
    assert gate([prompt(margin=.1, equal=False)])['passed']  # exactly at tolerance: tolerated
    assert gate([prompt(margin=.1, equal=False)])['tolerated_near_tie_flips'] == ['p']
    over = gate([prompt(margin=math.nextafter(.1, 1), equal=False)])
    assert not over['passed'] and over['decisive_next_token_flips'] == ['p']
    assert gate([prompt(margin=0.0015, equal=False)])['passed']  # the real near-tie


def test_agreement_and_difference_boundaries():
    assert gate([prompt(positions=100, agreeing=97)])['passed']
    assert gate([prompt(positions=200, agreeing=194)])['passed']
    assert not gate([prompt(positions=100, agreeing=96)])['passed']
    assert not gate([prompt(positions=1000, agreeing=969)])['passed']
    pooled = gate([prompt('a', 50, 50), prompt('b', 50, 47)])  # pooled 97/100
    assert pooled['passed'] and pooled['position_argmax_agreement'] == pytest.approx(.97)
    assert not gate([prompt('a', 10, 10), prompt('b', 90, 86)])['passed']  # pooled 96/100
    assert gate([prompt(diff=2.0)])['passed']
    assert not gate([prompt(diff=math.nextafter(2.0, 3))])['passed']


def test_decisive_flip_fails_when_everything_else_passes():
    result = gate([prompt('a', 1000, 999, .01, 5.0, False), prompt('b')])
    assert not result['passed'] and result['decisive_next_token_flips'] == ['a']
    assert result['position_argmax_agreement'] > .99 and result['max_abs_difference'] < 2


# --- non-finite and empty inputs -----------------------------------------------------------------

@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_difference_in_first_prompt_fails(bad):
    assert not gate([prompt('a', diff=bad), prompt('b')])['passed']


def test_nan_difference_in_later_prompt_fails():
    assert not gate([prompt('a'), prompt('b', diff=float('nan'))])['passed']


def test_inf_difference_in_later_prompt_fails():
    assert not gate([prompt('a'), prompt('b', diff=float('inf'))])['passed']


def test_nan_margin_flip_fails():
    assert not gate([prompt(margin=float('nan'), equal=False)])['passed']


def test_empty_prompt_list_does_not_pass():
    try:
        result = gate([])
    except ValueError:
        return
    assert not result['passed']


def test_zero_check_prompts_rejected():
    with pytest.raises(ValueError, match='check'):
        quantize.quantize_run('/nonexistent-source', '/nonexistent-out', ['a'], ['fp16'], check_prompts=0)


# --- old-record path on resume ----------------------------------------------------------------------

def old_check(prompts, passed):
    return {'prompts': [{k: v for k, v in p.items() if k not in ('agreeing_positions', 'next_token_margin')} for p in prompts],
            'passed': passed, 'max_abs_difference': max(p['max_abs_difference'] for p in prompts)}


def test_gate_v2_records_use_their_stored_outcome():
    assert quantize.recorded_gate_failed({'gate_version': 2, 'passed': False, 'prompts': []})
    assert not quantize.recorded_gate_failed({'gate_version': 2, 'passed': True, 'prompts': []})
    assert not quantize.recorded_gate_failed(None)


def test_old_records_judged_on_agreement_and_difference():
    assert quantize.recorded_gate_failed(old_check([prompt(agreeing=90)], True))
    assert quantize.recorded_gate_failed(old_check([prompt(diff=2.5)], True))
    # A pre-v2 flip carries no margin, so it cannot be judged and is no longer accepted on resume: quantize_run
    # re-checks it from the converted weights (see test_old_record_with_unknown_margin_flip_is_not_silently_accepted).
    assert quantize.recorded_gate_failed(old_check([prompt(margin=.0015, equal=False)], False))
    assert not quantize.recorded_gate_failed(old_check([prompt()], True))  # no flip: agreement/difference only


def test_old_record_with_unknown_margin_flip_is_not_silently_accepted():
    bad = old_check([prompt('a', 1000, 999, .5, 5.0, False)], False)
    assert quantize.recorded_gate_failed(bad)


def test_old_record_with_nan_difference_fails():
    assert quantize.recorded_gate_failed(old_check([prompt(diff=float('nan'))], False))


# --- CLI validation ---------------------------------------------------------------------------------

@pytest.mark.parametrize('flag,value', [('--margin-tolerance', '-1'), ('--margin-tolerance', 'nan'),
                                        ('--margin-tolerance', '1e9'), ('--min-position-agreement', '1.5'),
                                        ('--min-position-agreement', '-0.1'), ('--max-logit-difference', '-1'),
                                        ('--max-logit-difference', 'inf')])
def test_cli_rejects_absurd_thresholds(flag, value, monkeypatch):
    from prune_lab import cli
    called = []
    monkeypatch.setattr(quantize, 'quantize_run', lambda *a, **k: called.append(k) or [])
    with pytest.raises(SystemExit):
        cli.main(['quantize', '--source', 'x', '--output', 'y', '--variants', 'v', flag, value])
    assert not called


def test_cli_passes_thresholds_through(monkeypatch):
    from prune_lab import cli
    called = []
    monkeypatch.setattr(quantize, 'quantize_run', lambda *a, **k: called.append(k) or [])
    cli.main(['quantize', '--source', 'x', '--output', 'y', '--variants', 'v', '--margin-tolerance', '.2',
              '--min-position-agreement', '.95', '--max-logit-difference', '1.5'])
    assert called[0]['gate'] == {'margin_tolerance': .2, 'min_position_agreement': .95, 'max_abs_difference': 1.5}


# --- real conversion bugs on a tiny model ------------------------------------------------------------

mx = pytest.importorskip('mlx.core', reason='mlx-lm not installed')
pytest.importorskip('mlx_lm', reason='mlx-lm not installed')


@pytest.fixture(autouse=True)
def mlx_cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def build(tmp_path, block_scale):
    import torch
    from test_independent_quantize import dev_examples, make_export
    model, tokenizer = make_export(tmp_path / 'export', [128, 64, 96], eos_boost=1.0, seed=3)
    with torch.no_grad():
        model.lm_head.weight.mul_(8)  # logit spreads of several units
        for layer in model.model.layers:  # make the blocks, not just the residual embedding, drive the output
            layer.mlp.down_proj.weight.mul_(block_scale)
            layer.self_attn.o_proj.weight.mul_(block_scale)
    from prune_lab.artifacts import save_checkpoint
    import shutil
    shutil.rmtree(tmp_path / 'export')
    save_checkpoint(model, tokenizer, tmp_path / 'export', {'test': True})
    quantize.convert(tmp_path / 'export', tmp_path / 'fp16', 'fp16')
    return tmp_path / 'export', tokenizer, dev_examples(6)


@pytest.fixture
def converted(tmp_path):
    return build(tmp_path, 20)


@pytest.mark.xfail(strict=True, reason='DOUBTFUL (low): when a layer matters less, a wrong-index bug (gate rows of '
                   'one layer reversed) changes logits by 0.37, about 60x the correct fp16 error (0.006), but flips '
                   'no argmax and stays under the 2.0 absolute limit, so the gate passes it.')
def test_weak_layer_permutation_bug_fails(tmp_path):
    export, tokenizer, examples = build(tmp_path, 5)
    model, _ = quantize.load_mlx(tmp_path / 'fp16')
    mlp = model.layers[1].mlp
    mlp.gate_proj.weight = mlp.gate_proj.weight[mx.array(list(range(63, -1, -1)))]
    mx.eval(model.parameters())
    assert not check(export, model, tokenizer, examples)['passed']


def check(export, model, tokenizer, examples):
    return quantize.logit_check(export, model, tokenizer, examples, 'cpu')


def test_correct_fp16_conversion_passes_and_records_version(converted, tmp_path):
    export, tokenizer, examples = converted
    model, _ = quantize.load_mlx(tmp_path / 'fp16')
    result = check(export, model, tokenizer, examples)
    assert result['passed'] and result['gate_version'] == 2 and result['thresholds']['margin_tolerance'] == .1
    assert all('next_token_margin' in p and 'agreeing_positions' in p for p in result['prompts'])


@pytest.mark.parametrize('bug', ['permute_gate_rows', 'zero_layer', 'scale_embedding', 'swap_layers'])
def test_real_conversion_bugs_fail(converted, tmp_path, bug):
    export, tokenizer, examples = converted
    model, _ = quantize.load_mlx(tmp_path / 'fp16')
    mlp = model.layers[1].mlp
    if bug == 'permute_gate_rows':  # gate permuted without up/down: a wrong-index bug in one layer
        perm = mx.array(list(range(63, -1, -1)))
        mlp.gate_proj.weight = mlp.gate_proj.weight[perm]
    elif bug == 'zero_layer':
        mlp.down_proj.weight = mx.zeros_like(mlp.down_proj.weight)
        model.layers[1].self_attn.o_proj.weight = mx.zeros_like(model.layers[1].self_attn.o_proj.weight)
    elif bug == 'scale_embedding':  # wrong tied embedding (e.g. a dtype/scale slip)
        model.model.embed_tokens.weight = model.model.embed_tokens.weight * 0.5
    else:
        model.layers[0], model.layers[2] = model.layers[2], model.layers[0]
    mx.eval(model.parameters())
    assert not check(export, model, tokenizer, examples)['passed'], bug


def test_wrong_per_layer_width_fails_at_load(converted, tmp_path):
    config_path = tmp_path / 'fp16' / 'config.json'
    config = json.loads(config_path.read_text())
    config['prune_lab_layer_widths'] = [128, 96, 64]
    config_path.write_text(json.dumps(config))
    with pytest.raises(Exception):
        quantize.load_mlx(tmp_path / 'fp16')
