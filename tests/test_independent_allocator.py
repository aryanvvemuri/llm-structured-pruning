"""Independent tests of improvement 1 (allocator fix), written from plans/improvement-1-allocator.md.

Covers: legacy masks/fingerprints unchanged, curve construction, hand-worked greedy
examples, an independent reference implementation compared on
randomized inputs (cap, budget, nearest-reachable), per-target uniform candidates and
matched size, allocation.json contents, multi-seed reuse and config validation.
"""
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from prune_lab import allocation as alloc, campaign, multiseed, recovery
from prune_lab.modeling import count_parameters, mlp_layers
from prune_lab.pruning import compact_model

ROOT = Path(__file__).resolve().parents[1]
GIT_ENV = dict(os.environ, DEVELOPER_DIR='/Library/Developer/CommandLineTools')
OLD = '94c879e'  # main before improvement 1


def model_of(layers=3, hidden=8, width=8, seed=0):
    torch.manual_seed(seed)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=19, hidden_size=hidden, intermediate_size=width,
                                        num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=1,
                                        max_position_embeddings=32)).eval().requires_grad_(False)


def group_target(model, groups, group=2):
    """Exact target fraction for `groups` equal-cost groups of `group` neurons (bias-free MLP: 3*hidden/neuron)."""
    hidden = model.config.hidden_size
    return groups * group * 3 * hidden / count_parameters(model)


def removed(keep, widths):
    return [w - len(keep[i]) for i, w in enumerate(widths)]


def old_module(tmp_path, name):
    try:
        source = subprocess.run(['git', 'show', f'{OLD}:src/prune_lab/{name}.py'], cwd=ROOT, env=GIT_ENV,
                                capture_output=True, text=True, check=True, timeout=30).stdout
    except Exception as exc:  # pragma: no cover
        pytest.skip(f'git history unavailable: {exc}')
    source = source.replace('from .', 'from prune_lab.')
    path = tmp_path / f'old_{name}.py'
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(f'old_{name}_{OLD}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worktree_package_is_under_test():
    assert Path(alloc.__file__).resolve().is_relative_to(ROOT / 'src')


# --- legacy path unchanged ----------------------------------------------------------------------

def test_legacy_allocator_identical_to_pre_change_on_random_inputs(tmp_path):
    old = old_module(tmp_path, 'allocation')
    rng = random.Random(3)
    for trial in range(40):
        model = model_of(layers=4, hidden=8, width=48, seed=trial)
        if trial % 2:
            compact_model(model, {i: sorted(rng.sample(range(48), rng.randint(30, 48))) for i in range(4)})
        widths = [m.down_proj.in_features for m in mlp_layers(model)]
        scores = {i: torch.rand(w, generator=torch.Generator().manual_seed(trial * 10 + i)) for i, w in enumerate(widths)}
        sens = {i: rng.uniform(-.2, 1) for i in range(4)}
        target, group, keep = rng.uniform(.01, .08), rng.choice([1, 4, 8]), rng.choice([.25, .5])
        uniform = trial % 2 == 0
        args = (model, scores, sens, target, group, keep, uniform)
        try:
            expected = old.allocate_neurons(*args)
        except ValueError as exc:
            with pytest.raises(ValueError, match=str(exc)[:20]):
                alloc.allocate_neurons(*args)
            continue
        assert alloc.allocate_neurons(*args) == expected


def test_reference_config_fingerprints_and_specs_unchanged(tmp_path):
    names = ['support.json', 'support-v2.json', 'support-v2-longer.json', 'campaign.json', 'campaign-smoke.json']
    names = [n for n in names if (ROOT / 'configs' / n).exists()]
    archive = tmp_path / 'old'
    archive.mkdir()
    try:
        tar = subprocess.run(['git', 'archive', OLD, 'src'], cwd=ROOT, env=GIT_ENV, capture_output=True, check=True, timeout=60).stdout
        subprocess.run(['tar', '-x', '-C', str(archive)], input=tar, check=True)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f'git history unavailable: {exc}')
    script = ('import json,sys; from prune_lab import campaign as c; out={}\n'
              'for n in sys.argv[1:]:\n'
              '  cfg=c.read_campaign_config(n); out[n]=[c._fingerprint(cfg), c.candidate_specs(cfg)]\n'
              'print(json.dumps(out))')
    paths = [str(ROOT / 'configs' / n) for n in names]
    old = json.loads(subprocess.run([sys.executable, '-c', script, *paths], env=dict(os.environ, PYTHONPATH=str(archive / 'src')),
                                    capture_output=True, text=True, check=True).stdout)
    for path in paths:
        config = campaign.read_campaign_config(path)
        campaign.validate_config(config)
        assert [campaign._fingerprint(config), campaign.candidate_specs(config)] == old[path], path


# --- curve construction ---------------------------------------------------------------------------

def test_process_curve_clamps_then_running_max():
    assert alloc.process_curve([-1, -2, -3]) == [0, 0, 0]
    assert alloc.process_curve([.2, -.1, .1]) == [.2, .2, .2]
    assert alloc.process_curve([-.1, .3, .2, .5]) == [0, .3, .3, .5]
    for bad in ([float('nan')], [0, float('inf')]):
        with pytest.raises(ValueError):
            alloc.process_curve(bad)
    rng = random.Random(1)
    for _ in range(200):
        raw = [rng.uniform(-1, 1) for _ in range(3)]
        out = alloc.process_curve(raw)
        assert all(b >= a >= 0 for a, b in zip([0] + out, out))
        assert all(o >= max(0, r) for o, r in zip(out, raw))
        assert all(o == max([0] + raw[:k + 1]) for k, o in enumerate(out))


def test_curve_value_piecewise_linear_through_origin():
    f = (.1, .25, .5)
    c = [.2, .5, 1.1]
    for x, y in [(0, 0), (.05, .1), (.1, .2), (.175, .35), (.25, .5), (.375, .8), (.5, 1.1)]:
        assert alloc.curve_value(c, f, x) == pytest.approx(y)
    with pytest.raises(ValueError):
        alloc.curve_value(c, f, .51)


def test_probes_use_lowest_taylor_neurons_and_are_reversible(monkeypatch):
    model = model_of(layers=2, hidden=8, width=24)
    g = torch.Generator().manual_seed(0)
    batches = [{'input_ids': torch.randint(3, 19, (1, 9), generator=g)} for _ in range(3)]
    for b in batches:
        b['labels'] = b['input_ids'].clone()
        b['labels'][0, :5] = -100
        b['attention_mask'] = torch.ones_like(b['input_ids'])
    scores = {i: torch.rand(24, generator=g) for i in range(2)}
    before = {k: v.clone() for k, v in model.state_dict().items()}
    seen = []
    real = alloc.masked_calibration_loss
    monkeypatch.setattr(alloc, 'masked_calibration_loss', lambda m, b, keep=None: seen.append(keep) or real(m, b, keep))
    result = alloc.measure_sensitivity_curves(model, batches, scores)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in before.items())
    assert all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules())
    assert seen[0] is None and len(seen) == 1 + 2 * 3
    for layer in range(2):
        order = scores[layer].argsort(stable=True).tolist()
        for k, drop in enumerate((2, 6, 12)):  # round(24 * fraction)
            keep = seen[1 + 3 * layer + k]
            assert sorted(keep[layer]) == sorted(order[drop:])
            assert keep[1 - layer] == list(range(24))
            # Masked probe equals actually removing those neurons.
            pruned = copy.deepcopy(model)
            compact_model(pruned, keep)
            assert result['raw'][layer][k] == pytest.approx(real(pruned, batches) - result['baseline'], abs=1e-5)


# --- hand-worked greedy examples -----------------------------------------------------------------

def test_zero_signal_layers_pruned_evenly_and_sensitive_layer_last():
    model = model_of(layers=3, width=8)
    scores = {i: torch.arange(8, dtype=torch.float) + i for i in range(3)}
    curves = {0: [0, 0, 0], 1: [-.3, -.1, 0], 2: [.1, .2, .4]}
    expected = {1: [2, 0, 0], 2: [2, 2, 0], 3: [4, 2, 0], 4: [4, 4, 0], 5: [4, 4, 2], 6: [4, 4, 4]}
    for groups, want in expected.items():
        keep = alloc.allocate_curve(model, scores, curves, group_target(model, groups), group_size=2)
        assert removed(keep, [8] * 3) == want, groups
        for i in range(3):
            assert keep[i] == list(range(want[i], 8))  # lowest Taylor scores removed first
    with pytest.raises(ValueError, match='Infeasible'):
        alloc.allocate_curve(model, scores, curves, group_target(model, 7), group_size=2)  # 50% cap


def test_greedy_uses_processed_curve_marginals_per_parameter():
    model = model_of(layers=2, width=8)
    scores = {i: torch.rand(8, generator=torch.Generator().manual_seed(i)) for i in range(2)}
    # A: raw [-.05, .3, .1] -> [0, .3, .3]: groups (0->.25) cost .3, (.25->.5) cost 0.
    # B: [.02, .04, .5]: groups cost .04, .46.  Order: B1, A1, A2, B2.
    curves = {0: [-.05, .3, .1], 1: [.02, .04, .5]}
    for groups, want in {1: [0, 2], 2: [2, 2], 3: [4, 2], 4: [4, 4]}.items():
        keep = alloc.allocate_curve(model, scores, curves, group_target(model, groups), group_size=2)
        assert removed(keep, [8, 8]) == want, groups


# --- independent reference implementation, randomized ----------------------------------------------

def reference_greedy(widths, costs, target, group, keep_ratio, marginal):
    choices = []
    for w in widths:
        floor = max(1, math.ceil(w * keep_ratio))
        choices.append([w] + list(range(((w - 1) // group) * group, floor - 1, -group)))
    sel, active, achieved = [0] * len(widths), set(range(len(widths))), 0
    while achieved < target:
        options = []
        for i in active:
            if sel[i] + 1 < len(choices[i]):
                r, n = widths[i] - choices[i][sel[i]], choices[i][sel[i]] - choices[i][sel[i] + 1]
                options.append((marginal(i, r, n) / (n * costs[i]), r / widths[i], i, n * costs[i]))
        if not options:
            break
        _, _, i, cost = min(options)
        if abs(achieved + cost - target) >= abs(achieved - target):
            active.discard(i)
            continue
        achieved += cost
        sel[i] += 1
    return [widths[i] - choices[i][sel[i]] for i in range(len(widths))], choices, sel, achieved


def random_case(rng, trial):
    # Allocators operate on the original (uncompacted) model, so layer widths are equal; widths that are
    # not a multiple of the group make the first group smaller than the rest.
    model = model_of(layers=4, hidden=8, width=rng.choice([48, 60, 62, 64]), seed=trial)
    widths = [m.down_proj.in_features for m in mlp_layers(model)]
    g = torch.Generator().manual_seed(trial)
    scores = {i: torch.rand(w, generator=g) ** rng.choice([1, 3]) for i, w in enumerate(widths)}
    if trial % 5 == 0:
        scores[1] = torch.zeros(widths[1])
    curves = {i: [rng.choice([0.0, rng.uniform(-.2, 1)]) for _ in range(3)] for i in range(4)}
    if trial % 4 == 0:
        curves[0] = curves[2] = [0.0, 0.0, 0.0]
    costs = [3 * 8] * 4
    group = rng.choice([1, 4, 8])
    return model, widths, scores, curves, costs, group, None


@pytest.mark.parametrize('method', ['curve'])
def test_matches_independent_reference_and_cap_budget_properties(method):
    rng = random.Random(11 if method == 'curve' else 12)
    checked = 0
    for trial in range(150):
        model, widths, scores, curves, costs, group, _ = random_case(rng, trial)
        total = count_parameters(model)
        target_fraction = rng.uniform(0, .5 * sum(widths) * costs[0] / total)
        target = total * target_fraction
        if method == 'curve':
            processed = [alloc.process_curve(curves[i]) for i in range(4)]
            marginal = lambda i, r, n: (alloc.curve_value(processed[i], (.1, .25, .5), (r + n) / widths[i])
                                        - alloc.curve_value(processed[i], (.1, .25, .5), r / widths[i]))
            run = lambda: alloc.allocate_curve(model, scores, curves, target_fraction, group)
        try:
            keep = run()
        except ValueError as exc:
            assert 'Infeasible' in str(exc)
            continue
        checked += 1
        want, choices, sel, achieved = reference_greedy(widths, costs, target, group, .5, marginal)
        got = removed(keep, widths)
        assert got == want, (trial, got, want)
        for i, w in enumerate(widths):
            assert len(keep[i]) >= math.ceil(w * .5)  # 50% cap
            assert keep[i] == sorted(scores[i].argsort(stable=True)[got[i]:].tolist())  # Taylor order
        achieved = sum(r * c for r, c in zip(got, costs))
        group_costs = [(choices[i][s] - choices[i][s + 1]) * costs[i] for i, s in enumerate(sel) if s + 1 < len(choices[i])]
        # Nearest reachable: no single further group under the cap gets closer to the target.
        assert all(abs(achieved + c - target) >= abs(achieved - target) - 1e-9 for c in group_costs)
        assert abs(achieved - target) <= max(group * costs[0], max(group_costs, default=0)) / 2 + 1e-9
        # Model size after compaction equals the allocation's accounting.
        pruned = copy.deepcopy(model)
        compact_model(pruned, keep)
        assert total - count_parameters(pruned) == achieved
    assert checked > 100


# --- per-target uniform, matched size ---------------------------------------------------------

def support_config(**extra):
    config = campaign.read_campaign_config()
    config.update(extra)
    return config


def test_uniform_per_target_and_candidate_naming():
    ids = lambda c: [s['id'] for s in campaign.candidate_specs(c)]
    base = support_config(targets=[.1, .2])
    assert ids(base) == ['per-layer-01', 'per-layer-02', 'unpruned-recovery', 'uniform-10']
    assert ids(dict(base, uniform_per_target=True)) == ['per-layer-01', 'per-layer-02', 'unpruned-recovery', 'uniform-10', 'uniform-20']
    full = dict(base, uniform_per_target=True, allocation={'methods': ['legacy', 'curve']})
    assert ids(full) == ['per-layer-01', 'per-layer-02', 'curve-10', 'curve-20',
                         'unpruned-recovery', 'uniform-10', 'uniform-20']
    spec = {s['id']: s for s in campaign.candidate_specs(full)}
    assert spec['uniform-20'] == {'id': 'uniform-20', 'target': .2, 'uniform': True, 'control': True, 'selection_eligible': False}
    assert spec['curve-20']['method'] == 'curve'
    assert 'method' not in spec['per-layer-02']
    only = dict(full, train_candidates=['unpruned-recovery', 'uniform-20', 'curve-20'], selection_reference='unpruned-recovery')
    assert ids(only) == ['curve-20', 'unpruned-recovery', 'uniform-20']
    assert ids(dict(support_config(targets=[.15]), uniform_per_target=True))[-2:] == ['uniform-10', 'uniform-15']
    with pytest.raises(ValueError, match='collide'):
        campaign.candidate_specs(dict(base, targets=[.2, .201], allocation={'method': 'curve'}))


QWEN = dict(vocab_size=151936, hidden_size=896, intermediate_size=4864, num_hidden_layers=24, num_attention_heads=14,
            num_key_value_heads=2, tie_word_embeddings=True)


def qwen_meta():
    with torch.device('meta'):
        return Qwen2ForCausalLM(Qwen2Config(**QWEN))


def sizes_at_20(model):
    g = torch.Generator().manual_seed(0)
    n = len(mlp_layers(model))
    width = mlp_layers(model)[0].down_proj.in_features
    scores = {i: torch.rand(width, generator=g) for i in range(n)}
    sens = {i: float(torch.rand(1, generator=g)) for i in range(n)}
    curves = {i: sorted(torch.rand(3, generator=g).tolist()) for i in range(n)}
    keeps = {'uniform-20': alloc.allocate_neurons(model, scores, sens, .2, 64, .25, True),
             'per-layer-02': alloc.allocate_neurons(model, scores, sens, .2, 64, .25, False),
             'curve-20': alloc.allocate_curve(model, scores, curves, .2, 64)}
    cost = 3 * model.config.hidden_size
    total = count_parameters(model)
    return {k: (sum(width - len(v) for v in keep.values()) * cost - .2 * total) / (64 * cost) for k, keep in keeps.items()}


def test_matched_size_of_allocators_at_qwen_scale():
    off = sizes_at_20(qwen_meta())
    for name in ('per-layer-02', 'curve-20'):
        assert abs(off[name]) <= .5, (name, off[name])
    # Uniform is within half a uniform step (24 groups), by construction.
    assert abs(off['uniform-20']) <= 12


@pytest.mark.xfail(strict=True, reason='Accepted (inherent 24-group uniform step): uniform-20 removes 1.65 groups '
                   '(~0.28M params) more than the target on Qwen2.5-0.5B (20.057% vs curve/global/per-layer 19.988%); '
                   'parameter counts are reported. See plans/improvement-1-allocator.md.')
def test_uniform_20_within_one_group_of_target_at_qwen_scale():
    assert abs(sizes_at_20(qwen_meta())['uniform-20']) <= 1


# --- config validation ------------------------------------------------------------------------------

@pytest.mark.parametrize('bad', [
    {'allocation': []}, {'allocation': {}}, {'allocation': {'method': 'curve', 'methods': ['curve']}},
    {'allocation': {'method': 'flap'}}, {'allocation': {'method': 'global-standardized'}}, {'allocation': {'methods': []}}, {'allocation': {'methods': ['curve', 'curve']}},
    {'allocation': {'methods': 'curve'}}, {'allocation': {'method': 'curve', 'extra': 1}},
    {'allocation': {'method': 'curve', 'curve_sensitivity_examples': 0}},
    {'allocation': {'method': 'curve', 'curve_sensitivity_examples': True}},
    {'allocation': {'method': 'curve', 'curve_sensitivity_examples': '150'}},
    {'allocation': {'method': 'curve', 'curve_sensitivity_examples': 151}},
    {'allocation': {'method': 'curve', 'probe_fractions': [.25, .1, .5]}},
    {'allocation': {'method': 'curve', 'probe_fractions': [.1, .1, .5]}},
    {'allocation': {'method': 'curve', 'probe_fractions': [0, .5]}},
    {'allocation': {'method': 'curve', 'probe_fractions': [.5, 1]}},
    {'allocation': {'method': 'curve', 'probe_fractions': []}},
    {'allocation': {'method': 'curve', 'min_keep_ratio': 0}},
    {'allocation': {'method': 'curve', 'min_keep_ratio': 1.5}},
    {'allocation': {'method': 'curve', 'min_keep_ratio': True}},
    {'allocation': {'method': 'curve', 'min_keep_ratio': .4}},  # 60% cap > largest probe 50%
    {'allocation': {'method': 'curve', 'probe_fractions': [.1, .25]}},  # 50% cap > 25%
    {'uniform_per_target': 'yes'}, {'uniform_per_target': True, 'include_controls': False},
    {'train_candidates': []}, {'train_candidates': 'uniform-10'}, {'train_candidates': ['uniform-10', 'uniform-10']},
    {'train_candidates': ['unpruned-recovery', 'curve-20']},  # no curve method configured
    {'train_candidates': ['uniform-10'], 'selection_reference': 'unpruned-recovery'},
    {'train_candidates': [1]},
])
def test_invalid_configs_rejected(bad):
    config = support_config(targets=[.1, .2], counts={'recovery': 1000, 'calibration': 150, 'dev': 200, 'test': 300})
    config.update(bad)
    with pytest.raises(ValueError):
        campaign.validate_config(config)


def test_valid_option_combinations_accepted():
    for extra in ({'allocation': None}, {'allocation': {'method': 'legacy'}},
                  {'allocation': {'method': 'curve', 'curve_sensitivity_examples': 150}},
                  {'allocation': {'method': 'curve', 'probe_fractions': [.1, .3, .6], 'min_keep_ratio': .4}}):
        campaign.validate_config(dict(support_config(targets=[.1, .2]), **extra))


def test_allocator_retest_config_matches_plan():
    path = ROOT / 'configs' / 'support-v2-allocator-retest.json'
    config = campaign.read_campaign_config(path)
    campaign.validate_config(config)
    assert [s['id'] for s in campaign.candidate_specs(config)] == ['curve-20', 'unpruned-recovery', 'uniform-20']
    longer = json.loads((ROOT / 'configs' / 'support-v2-longer.json').read_text())
    raw = json.loads(path.read_text())
    assert {k: v for k, v in raw.items() if k not in ('allocation', 'uniform_per_target', 'train_candidates')} == longer
    options = campaign.allocation_options(config)
    assert options['curve_sensitivity_examples'] is None and options['probe_fractions'] == [.1, .25, .5]
    assert options['min_keep_ratio'] == .5 and config['min_keep_ratio'] == .25 and config['sensitivity_examples'] == 16


# --- campaign: allocation.json, legacy identity, multi-seed reuse -------------------------------------

@pytest.fixture
def tiny_support(tmp_path, monkeypatch):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from prune_lab import support_data
    from test_independent_support_v2 import LAYOUT, build_archive
    monkeypatch.chdir(tmp_path)
    torch.set_num_threads(1)
    backend = Tokenizer(models.WordLevel({'[PAD]': 0, '[UNK]': 1, '[EOS]': 2, 'support': 3, 'reply': 4}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]', eos_token='[EOS]')
    torch.manual_seed(5)
    Qwen2ForCausalLM(Qwen2Config(vocab_size=5, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                                 num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
                                 pad_token_id=0, eos_token_id=2)).save_pretrained(tmp_path / 'model', safe_serialization=True)
    tokenizer.save_pretrained(tmp_path / 'model')
    counts = {'recovery': 3, 'calibration': 3, 'dev': 3, 'test': 3}
    support_data.prepare_support_dataset(build_archive(tmp_path, monkeypatch, LAYOUT), tmp_path / 'prepared', tokenizer,
                                         counts=counts, max_length=256, max_answer_tokens=12)
    config = campaign.read_campaign_config()
    config.update(task='support', dataset='doc2dial', support_data_dir=str(tmp_path / 'prepared'), model=str(tmp_path / 'model'),
                  device='cpu', cpu_threads=1, counts=counts, targets=[.1, .2], group_size=4, sensitivity_examples=2,
                  benchmark_examples=1, benchmark_repeats=1, max_length=256, max_new_tokens=12,
                  selection_reference='unpruned-recovery', select_uniform=True, evaluate_all_candidates=True)
    config['recovery'].update(rank=2, max_steps=1, checkpoints=[1], gradient_accumulation=1)
    return config, tmp_path


def allocation_only(config, output, monkeypatch):
    """Run preflight and fit up to the written allocation (training is interrupted)."""
    campaign.run_campaign(config, output, 'preflight')
    monkeypatch.setattr(recovery, 'recover_model', lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        campaign.run_campaign(config, output, 'fit')
    monkeypatch.undo()
    return json.loads((output / 'allocation.json').read_text())


def test_allocation_record_contents_and_legacy_masks_identical(tiny_support, monkeypatch):
    config, tmp = tiny_support
    legacy = allocation_only(config, tmp / 'legacy', monkeypatch)
    monkeypatch.chdir(tmp)
    new_config = dict(config, uniform_per_target=True, allocation={'methods': ['legacy', 'curve']})
    new = allocation_only(new_config, tmp / 'new', monkeypatch)
    old_keep = {c['id']: c['keep_indices'] for c in legacy['candidates']}
    new_keep = {c['id']: c['keep_indices'] for c in new['candidates']}
    for name in old_keep:  # per-layer-01/02, unpruned-recovery, uniform-10
        assert new_keep[name] == old_keep[name], name
    assert new['sensitivity'] == legacy['sensitivity']
    assert set(legacy) == {'candidates', 'sensitivity', 'elapsed_seconds', 'baseline_sha256'}
    assert (tmp / 'legacy' / 'baseline.json').exists()
    probes = new['curve_probes']
    assert probes['examples'] == 3 and probes['fractions'] == [.1, .25, .5]
    assert set(probes['raw']) == {'0', '1', '2'} and all(len(v) == 3 for v in probes['raw'].values())
    assert probes['curves'] == {k: alloc.process_curve(v) for k, v in probes['raw'].items()}
    assert new['allocation_options']['methods'] == ['legacy', 'curve']
    ids = [c['id'] for c in new['candidates']]
    assert set(new['removal_fractions']) == set(ids) == set(new['calibration_loss_after_pruning'])
    for c in new['candidates']:
        fractions = new['removal_fractions'][c['id']]
        assert fractions == [1 - len(c['keep_indices'][str(i)]) / 32 for i in range(3)]
        if c.get('method'):
            assert max(fractions) <= .5
    assert new['calibration_loss_after_pruning']['unpruned-recovery'] == pytest.approx(probes['baseline_loss'], rel=1e-9)
    assert all(v >= 0 for v in new['removal_fractions']['uniform-20'])


def test_allocation_record_allows_recomputing_masks(tiny_support, monkeypatch):
    config, tmp = tiny_support
    record = allocation_only(dict(config, allocation={'methods': ['curve']}), tmp / 'r', monkeypatch)
    text = json.dumps(record)
    assert 'scores' in record or 'taylor' in text.lower()


def test_multi_seed_reuses_one_allocation(tiny_support, monkeypatch):
    config, tmp = tiny_support
    config = dict(config, uniform_per_target=True, allocation={'methods': ['curve']},
                  train_candidates=['unpruned-recovery', 'curve-20', 'uniform-20'], targets=[.2])
    probes = []
    real = alloc.measure_sensitivity_curves
    monkeypatch.setattr(alloc, 'measure_sensitivity_curves', lambda *a, **k: probes.append(1) or real(*a, **k))
    root = tmp / 'runs' / 'ms'
    multiseed.run_seeds(config, root, [3, 4], discard_checkpoints=True)
    assert len(probes) == 1
    for name in ('allocation.json', 'baseline.json'):
        assert (root / 'seed-3' / name).read_bytes() == (root / 'seed-4' / name).read_bytes()
    for seed in (3, 4):
        records = {c['id']: c for c in campaign._selection(root / f'seed-{seed}')['candidates']}
        assert set(records) == {'unpruned-recovery', 'curve-20', 'uniform-20'}
        assert all(r['status'] == 'complete' for r in records.values())
        widths = {k: r['layer_widths'] for k, r in records.items()}
        allocation = json.loads((root / f'seed-{seed}' / 'allocation.json').read_text())
        for c in allocation['candidates']:
            assert widths[c['id']] == [len(c['keep_indices'][str(i)]) for i in range(3)]
    summary = multiseed.aggregate_run(root, samples=20)
    assert set(summary['variants']) >= {'curve-20', 'uniform-20'}
