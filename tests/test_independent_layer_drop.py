"""Independent tests of improvement 8 (layer dropping), from plans/improvement-8-layer-drop.md.

Tiny random Qwen2 models on CPU; the real 0.5B shape only on the meta device.
"""
import copy
import glob
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from prune_lab import campaign, layerdrop, recovery, scoring
from prune_lab.modeling import count_parameters

ROOT = Path(__file__).resolve().parents[1]
GIT_ENV = dict(os.environ, DEVELOPER_DIR='/Library/Developer/CommandLineTools')
MAIN = '02d34e3'
HUB = Path('/Users/aryan/llm_pruning_optimization/.cache/huggingface/hub')


def test_worktree_package_is_under_test():
    assert Path(layerdrop.__file__).resolve().is_relative_to(ROOT / 'src')


def tiny(layers=5, vocab=50, seed=0, hidden=16, width=64):
    torch.manual_seed(seed)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=vocab, hidden_size=hidden, intermediate_size=width,
                                        num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=1,
                                        max_position_embeddings=256, pad_token_id=0, eos_token_id=2)).eval()


def batches(n=4, vocab=50, seed=1):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        ids = torch.randint(3, vocab, (1, 9 + i), generator=g)
        labels = ids.clone()
        labels[0, :5] = -100
        out.append({'input_ids': ids, 'attention_mask': torch.ones_like(ids), 'labels': labels})
    return out


def my_loss(model, data):
    total = tokens = 0.0
    with torch.no_grad():
        for b in data:
            n = int((b['labels'][:, 1:] != -100).sum())
            total += float(model(**b, use_cache=False).loss) * n
            tokens += n
    return total / tokens


def make_identity(model, layer):
    with torch.no_grad():
        model.model.layers[layer].self_attn.o_proj.weight.zero_()
        model.model.layers[layer].mlp.down_proj.weight.zero_()


# --- (2) a dropped layer is really gone; KV cache renumbering -------------------------------------

def rebuilt_small(model, kept):
    """Independent construction: a fresh (n-k)-layer model with the kept layers' weights copied in."""
    config = copy.deepcopy(model.config)
    config.num_hidden_layers = len(kept)
    config.layer_types = None
    for key in ('prune_lab_dropped_layers', 'prune_lab_kept_layers', 'prune_lab_original_num_hidden_layers'):
        if hasattr(config, key):
            delattr(config, key)
    small = Qwen2ForCausalLM(Qwen2Config(**{k: v for k, v in config.to_dict().items() if k not in ('layer_types',)})).eval()
    state = model.state_dict() if not hasattr(model, '_original_state') else model._original_state
    new = {}
    for key, value in state.items():
        if key.startswith('model.layers.'):
            index = int(key.split('.')[2])
            if index in kept:
                new[key.replace(f'model.layers.{index}.', f'model.layers.{kept.index(index)}.', 1)] = value
        else:
            new[key] = value
    small.load_state_dict(new, strict=True)
    return small


@pytest.mark.parametrize('dropped', [[2], [0], [4], [0, 4], [1, 2, 3]])
def test_dropped_layers_absent_and_generation_matches(dropped):
    model = tiny()
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    per_layer = [sum(p.numel() for p in l.parameters()) for l in model.model.layers]
    before = count_parameters(model)
    kept = layerdrop.drop_layers(model, dropped)
    assert kept == [i for i in range(5) if i not in dropped]
    assert len(model.model.layers) == model.config.num_hidden_layers == 5 - len(dropped)
    assert model.config.layer_types == ['full_attention'] * (5 - len(dropped))
    assert [l.self_attn.layer_idx for l in model.model.layers] == list(range(5 - len(dropped)))
    assert count_parameters(model) == before - sum(per_layer[i] for i in dropped)
    layer_keys = {k.split('.')[2] for k in model.state_dict() if k.startswith('model.layers.')}
    assert layer_keys == {str(i) for i in range(5 - len(dropped))}
    assert model.config.prune_lab_dropped_layers == sorted(dropped) and model.config.prune_lab_kept_layers == kept
    holder = tiny()
    holder.load_state_dict(original_state)
    holder._original_state = original_state
    small = rebuilt_small(holder, kept)
    ids = torch.randint(3, 50, (1, 12), generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        assert torch.equal(model(input_ids=ids).logits, small(input_ids=ids).logits)
        cached = model.generate(input_ids=ids, max_new_tokens=10, do_sample=False, use_cache=True, pad_token_id=0)
        uncached = model.generate(input_ids=ids, max_new_tokens=10, do_sample=False, use_cache=False, pad_token_id=0)
        reference = small.generate(input_ids=ids, max_new_tokens=10, do_sample=False, use_cache=True, pad_token_id=0)
    assert torch.equal(cached, uncached) and torch.equal(cached, reference)
    # Cache step logits agree with full recomputation at every step.
    with torch.no_grad():
        out = model(input_ids=ids[:, :8], use_cache=True)
        step = model(input_ids=ids[:, 8:9], past_key_values=out.past_key_values, use_cache=True).logits[0, -1]
        full = model(input_ids=ids[:, :9], use_cache=False).logits[0, -1]
    assert torch.allclose(step, full, atol=1e-5)


def test_drop_layers_refuses_bad_input():
    for bad in ([], [5], [-1], [1, 1], [0, 1, 2, 3, 4]):
        with pytest.raises(ValueError):
            layerdrop.drop_layers(tiny(), bad)
    model = tiny()
    layerdrop.drop_layers(model, [1])
    with pytest.raises(ValueError):
        layerdrop.drop_layers(model, [0])  # a second drop would use renumbered indices


def test_without_layers_fully_restores():
    model = tiny()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    ids = torch.randint(3, 50, (1, 7))
    with torch.no_grad():
        before = model(input_ids=ids).logits
    with layerdrop.without_layers(model, [1, 3]):
        assert len(model.model.layers) == 3 and model.config.num_hidden_layers == 3
    assert len(model.model.layers) == 5 and model.config.num_hidden_layers == 5
    assert [l.self_attn.layer_idx for l in model.model.layers] == list(range(5))
    assert len(model.config.layer_types) == 5
    with torch.no_grad():
        assert torch.equal(model(input_ids=ids).logits, before)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in state.items())


# --- (3) scan and greedy against brute force ----------------------------------------------------

def brute_greedy(model, data, steps):
    removed, losses = [], []
    for _ in range(steps):
        best = None
        for layer in range(len(model.model.layers)):
            if layer in removed:
                continue
            candidate = copy.deepcopy(model)
            layerdrop.drop_layers(candidate, removed + [layer])
            loss = my_loss(candidate, data)
            if best is None or loss < best[0] or (loss == best[0] and layer > best[1]):
                best = (loss, layer)
        removed.append(best[1])
        losses.append(best[0])
    return removed, losses


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_greedy_matches_brute_force(seed):
    model = tiny(layers=6, seed=seed)
    data = batches(4, seed=seed + 10)
    record = layerdrop.scan_and_select(model, data, 4)
    order, losses = brute_greedy(model, data, 4)
    assert record['order'] == order
    assert [g['loss'] for g in record['greedy']] == pytest.approx(losses, rel=1e-6)
    assert [g['evaluated'] for g in record['greedy']] == [6, 5, 4, 3]
    baseline = my_loss(model, data)
    assert record['baseline_loss'] == pytest.approx(baseline, rel=1e-6)
    single = {s['layer']: s['loss'] for s in record['single_layer']}
    assert sorted(single) == list(range(6))
    for layer, loss in single.items():
        candidate = copy.deepcopy(model)
        layerdrop.drop_layers(candidate, [layer])
        assert loss == pytest.approx(my_loss(candidate, data), rel=1e-6)
    assert record['greedy'][0]['removed_layer'] == min(single, key=lambda l: (single[l], -l))
    assert len(model.model.layers) == 6  # scan leaves the model intact


def test_ties_break_toward_higher_index():
    model = tiny(layers=5, seed=4)
    for layer in range(5):
        make_identity(model, layer)  # every removal leaves the function unchanged: exact ties everywhere
    record = layerdrop.scan_and_select(model, batches(3), 3)
    assert record['order'] == [4, 3, 2]
    assert all(g['increase'] == 0 for g in record['greedy'])
    assert len({s['loss'] for s in record['single_layer']}) == 1


def test_scan_rejects_bad_steps():
    for steps in (0, 5, 6):
        with pytest.raises(ValueError):
            layerdrop.scan_and_select(tiny(), batches(2), steps)


# --- (4) real 0.5B shape on the meta device ---------------------------------------------------------

def meta_qwen():
    paths = glob.glob(str(HUB / 'models--Qwen--Qwen2.5-0.5B-Instruct' / 'snapshots' / '*' / 'config.json'))
    if not paths:
        pytest.skip('0.5B config not cached')
    from transformers import AutoConfig
    from prune_lab.modeling import CompactQwen2ForCausalLM
    config = AutoConfig.from_pretrained(str(Path(paths[0]).parent))
    with torch.device('meta'):
        return CompactQwen2ForCausalLM(config)


def test_planned_sizes_at_real_shape():
    from prune_lab.allocation import allocate_neurons
    from prune_lab.pruning import compact_model
    model = meta_qwen()
    assert count_parameters(model) == 494_032_768
    assert set(layerdrop.layer_parameters(model)) == {14_912_384}
    seven = copy.deepcopy(model)
    layerdrop.drop_layers(seven, [3, 5, 8, 11, 14, 17, 20])
    assert count_parameters(seven) == 389_646_080 <= 394_942_336
    three = copy.deepcopy(model)
    layerdrop.drop_layers(three, [4, 12, 19])
    remaining = count_parameters(three)
    target = (.2 * 494_032_768 - (494_032_768 - remaining)) / remaining
    scores = {i: torch.rand(4864, generator=torch.Generator().manual_seed(i)) for i in range(21)}
    keep = allocate_neurons(three, scores, {i: 1.0 for i in range(21)}, target, 64, .25, True)
    assert {len(v) for v in keep.values()} == {4864 - 15 * 64}
    try:
        compact_model(three, keep)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f'meta compaction unsupported: {exc}')
    assert count_parameters(three) == 395_105_536
    config = campaign.read_campaign_config(ROOT / 'configs' / 'support-v2-layer-drop.json')
    assert layerdrop.check_shape(config, model) == {'layers-7': {'layers': 24, 'drop': 7},
                                                    'layers-3-uniform-20': {'layers': 24, 'drop': 3}}


# --- campaign fixture -----------------------------------------------------------------------------

COUNTS = {'recovery': 3, 'calibration': 3, 'dev': 3, 'test': 3}
LAYER_DROP = {'examples': 3, 'candidates': [{'id': 'layers-2', 'drop': 2},
                                            {'id': 'layers-1-uniform-30', 'drop': 1, 'total_target': .3}]}


@pytest.fixture
def base(tmp_path, monkeypatch):
    from prune_lab import support_data
    from test_independent_support_v2 import LAYOUT, build_archive
    monkeypatch.chdir(tmp_path)
    torch.set_num_threads(1)
    backend = Tokenizer(models.WordLevel({'[PAD]': 0, '[UNK]': 1, '[EOS]': 2, 'support': 3, 'reply': 4}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]', eos_token='[EOS]')
    model = tiny(layers=5, vocab=5, seed=9)
    model.save_pretrained(tmp_path / 'model', safe_serialization=True)
    tokenizer.save_pretrained(tmp_path / 'model')
    support_data.prepare_support_dataset(build_archive(tmp_path, monkeypatch, LAYOUT), tmp_path / 'prepared', tokenizer,
                                         counts=COUNTS, max_length=256, max_answer_tokens=12)
    config = campaign.read_campaign_config()
    config.update(task='support', dataset='doc2dial', support_data_dir=str(tmp_path / 'prepared'), model=str(tmp_path / 'model'),
                  device='cpu', cpu_threads=1, counts=COUNTS, targets=[.1, .2], group_size=4, sensitivity_examples=2,
                  benchmark_examples=1, benchmark_repeats=1, max_length=256, max_new_tokens=12,
                  selection_reference='unpruned-recovery', select_uniform=True, evaluate_all_candidates=True,
                  uniform_per_target=True, selection_dev_examples=2, skip_full_merged_dev=True,
                  train_candidates=['unpruned-recovery', 'uniform-20'])
    config['recovery'].update(rank=2, learning_rate=2e-2, max_steps=2, checkpoints=[1, 2], gradient_accumulation=1)
    return config, tmp_path


def with_drop(config, layer_drop=None, train=('unpruned-recovery', 'uniform-20', 'layers-2', 'layers-1-uniform-30')):
    result = json.loads(json.dumps(config))
    result['layer_drop'] = json.loads(json.dumps(layer_drop or LAYER_DROP))
    result['train_candidates'] = list(train)
    campaign.validate_config(result)
    return result


def fit(config, output):
    campaign.run_campaign(config, output, 'preflight')
    campaign.run_campaign(config, output, 'fit')
    return {c['id']: c for c in campaign._selection(output)['candidates']}


def weights(record):
    return {p.name: p.read_bytes() for p in sorted(Path(record['checkpoint']).glob('*.safetensors'))}


def strip(rows):
    return [(r['key'], r['prediction'], r['score']) for r in rows]


def no_training(model, batches, evaluate_dev, cfg, **kwargs):
    dev = evaluate_dev(model)
    return {'initial_dev': dev, 'best_dev': dev, 'best_step': 0, 'history': [], 'optimizer_steps': 0,
            'train_losses': [], 'merged': True, 'seed': cfg['seed']}


# --- (1) defaults unchanged --------------------------------------------------------------------------

def old_src(tmp_path):
    target = tmp_path / 'main-src'
    target.mkdir(exist_ok=True)
    try:
        tar = subprocess.run(['git', 'archive', MAIN, 'src'], cwd=ROOT, env=GIT_ENV, capture_output=True, check=True, timeout=60).stdout
        subprocess.run(['tar', '-x', '-C', str(target)], input=tar, check=True)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f'git history unavailable: {exc}')
    return target / 'src'


def test_fingerprints_and_specs_unchanged_from_main(tmp_path):
    names = [p.name for p in sorted((ROOT / 'configs').glob('*.json')) if p.name != 'support-v2-layer-drop.json']
    script = ('import json,sys; from prune_lab import campaign as c; out={}\n'
              'for n in sys.argv[1:]:\n  cfg=c.read_campaign_config(n); out[n]=[c._fingerprint(cfg), c.candidate_specs(cfg)]\n'
              'print(json.dumps(out))')
    paths = [str(ROOT / 'configs' / n) for n in names]
    old = json.loads(subprocess.run([sys.executable, '-c', script, *paths], env=dict(os.environ, PYTHONPATH=str(old_src(tmp_path))),
                                    capture_output=True, text=True, check=True).stdout)
    for path in paths:
        config = campaign.read_campaign_config(path)
        assert [campaign._fingerprint(config), campaign.candidate_specs(config)] == old[path], path


def test_campaign_outputs_identical_to_main_without_layer_drop(base, tmp_path):
    config, tmp = base
    new = fit(config, tmp / 'new')
    (tmp / 'cfg.json').write_text(json.dumps(config))
    script = ('import json,sys; from prune_lab import campaign as c\n'
              'cfg=json.load(open(sys.argv[1])); c.run_campaign(cfg, sys.argv[2], "preflight"); c.run_campaign(cfg, sys.argv[2], "fit")')
    subprocess.run([sys.executable, '-c', script, str(tmp / 'cfg.json'), str(tmp / 'old')], cwd=tmp,
                   env=dict(os.environ, PYTHONPATH=str(old_src(tmp_path)), HF_HUB_OFFLINE='1'), check=True, capture_output=True)
    old = {c['id']: c for c in campaign._selection(tmp / 'old')['candidates']}
    assert list(old) == list(new)
    for name in old:
        assert weights(new[name]) == weights(old[name])
        assert strip(new[name]['dev']['predictions']) == strip(old[name]['dev']['predictions'])
    a, b = (json.loads((tmp / d / 'allocation.json').read_text()) for d in ('old', 'new'))
    assert set(a) == set(b) and a['candidates'] == b['candidates'] and a['sensitivity'] == b['sensitivity']
    assert not (tmp / 'new' / 'layer-scan.json').exists()


def test_existing_candidates_unaffected_by_layer_drop(base):
    config, tmp = base
    plain = fit(config, tmp / 'plain')
    dropped = fit(with_drop(config), tmp / 'drop')
    for name in plain:
        assert weights(dropped[name]) == weights(plain[name]), name
        assert strip(dropped[name]['dev']['predictions']) == strip(plain[name]['dev']['predictions'])


# --- campaign behaviour ---------------------------------------------------------------------------

def test_scan_uses_calibration_only_and_combined_scores_reduced_network(base, monkeypatch):
    config, tmp = base
    cfg = with_drop(config)
    made = {}
    real_make = campaign.make_batches

    def make(examples, tokenizer, *a, **k):
        examples = list(examples)
        out = real_make(examples, tokenizer, *a, **k)
        for b in out:
            made[id(b)] = examples[0].key.split(':')[0] + ':' + ','.join(sorted(e.key for e in examples))
        made.setdefault('lists', []).append((tuple(e.key for e in examples), out))
        return out
    monkeypatch.setattr(campaign, 'make_batches', make)
    scans, scored = [], []
    real_scan, real_score = layerdrop.scan_and_select, scoring.score_neurons
    monkeypatch.setattr(layerdrop, 'scan_and_select', lambda m, b, s: scans.append((list(b), s)) or real_scan(m, b, s))
    monkeypatch.setattr(scoring, 'score_neurons', lambda m, b, method='taylor': scored.append((len(m.model.layers), list(b))) or real_score(m, b, method))
    records = fit(cfg, tmp / 'run')
    from prune_lab.support_data import read_support_jsonl
    calibration_keys = tuple(e.key for e in read_support_jsonl(Path(cfg['support_data_dir']) / 'calibration.jsonl'))
    calibration_batches = next(out for keys, out in made['lists'] if keys == calibration_keys)
    (scan_batches, steps), = scans
    assert steps == 2 and len(scan_batches) == 3
    assert all(any(b is c for c in calibration_batches) for b in scan_batches)
    assert [n for n, _ in scored] == [5, 4]  # intact network first, reduced network for the combined candidate
    assert all(any(b is c for c in calibration_batches) for b in scored[1][1])
    allocation = json.loads((tmp / 'run' / 'allocation.json').read_text())
    scan = allocation['layer_scan']
    assert json.loads((tmp / 'run' / 'layer-scan.json').read_text()) == scan
    specs = {c['id']: c for c in allocation['candidates']}
    assert specs['layers-2']['dropped_layers'] == sorted(scan['order'][:2])
    assert specs['layers-1-uniform-30']['dropped_layers'] == scan['order'][:1]
    # Recompute the combined mask independently on the reduced network.
    from prune_lab.allocation import allocate_neurons
    from prune_lab.modeling import load_model
    model, tokenizer = load_model(cfg['model'], device='cpu', dtype='float32')
    original = count_parameters(model)
    layerdrop.drop_layers(model, specs['layers-1-uniform-30']['dropped_layers'])
    scores = real_score(model, real_make(read_support_jsonl(Path(cfg['support_data_dir']) / 'calibration.jsonl'), tokenizer, cfg['max_length']), 'taylor')
    digest = hashlib.sha256()
    for i in sorted(scores):
        digest.update(scores[i].detach().cpu().float().contiguous().numpy().tobytes())
    assert specs['layers-1-uniform-30']['scores_sha256'] == digest.hexdigest()
    remaining = count_parameters(model)
    keep = allocate_neurons(model, scores, {i: 1.0 for i in scores}, (.3 * original - (original - remaining)) / remaining, 4, .25, True)
    assert specs['layers-1-uniform-30']['keep_indices'] == {str(k): v for k, v in keep.items()}
    # Records: provenance and parameter counts.
    for name, spec in specs.items():
        if spec.get('kind') != 'layer-drop':
            continue
        record = records[name]
        assert record['dropped_layers'] == spec['dropped_layers'] and record['kept_layers'] == spec['kept_layers']
        assert record['parameters'] == spec['planned_parameters']
        assert len(record['layer_widths']) == 5 - spec['drop']
    assert abs(records['layers-1-uniform-30']['actual_reduction'] - .3) < .05
    assert records['layers-2']['layer_widths'] == [64, 64, 64]


def test_exports_load_and_match_in_memory(base, monkeypatch):
    config, tmp = base
    cfg = with_drop(config)
    monkeypatch.setattr(recovery, 'recover_model', no_training)
    records = fit(cfg, tmp / 'run')
    allocation = {c['id']: c for c in json.loads((tmp / 'run' / 'allocation.json').read_text())['candidates']}
    from prune_lab.modeling import load_model
    from prune_lab.pruning import compact_model
    ids = torch.tensor([[3, 4, 3, 4, 1, 2, 3]])
    for name in ('layers-2', 'layers-1-uniform-30'):
        reference, _ = load_model(cfg['model'], device='cpu', dtype='float32')
        layerdrop.drop_layers(reference, allocation[name]['dropped_layers'])
        if allocation[name].get('keep_indices'):
            compact_model(reference, {int(k): v for k, v in allocation[name]['keep_indices'].items()})
        project, _ = load_model(records[name]['checkpoint'], device='cpu', dtype='float32')
        with torch.no_grad():
            expected = reference(input_ids=ids).logits
            assert torch.equal(project(input_ids=ids).logits, expected)
        if name == 'layers-2':
            stock = AutoModelForCausalLM.from_pretrained(records[name]['checkpoint'], attn_implementation='eager').eval()
            assert type(stock).__name__ == 'Qwen2ForCausalLM' and stock.config.num_hidden_layers == 3
            with torch.no_grad():
                assert torch.equal(stock(input_ids=ids).logits, expected)
            assert count_parameters(stock) == records[name]['parameters']


def test_resume_mid_seed_identical(base, monkeypatch):
    config, tmp = base
    cfg = with_drop(config)
    full = fit(cfg, tmp / 'full')
    out = tmp / 'resumed'
    campaign.run_campaign(cfg, out, 'preflight')
    real = recovery.recover_model
    calls = []

    def interrupt(*a, **k):
        calls.append(1)
        if len(calls) == 3:  # after two candidates, during the third
            raise KeyboardInterrupt
        return real(*a, **k)
    monkeypatch.setattr(recovery, 'recover_model', interrupt)
    with pytest.raises(KeyboardInterrupt):
        campaign.run_campaign(cfg, out, 'fit')
    monkeypatch.setattr(recovery, 'recover_model', real)
    campaign.run_campaign(cfg, out, 'fit')
    resumed = {c['id']: c for c in campaign._selection(out)['candidates']}
    assert list(resumed) == list(full)
    for name in full:
        assert weights(resumed[name]) == weights(full[name]), name
        assert strip(resumed[name]['dev']['predictions']) == strip(full[name]['dev']['predictions'])
    assert (out / 'allocation.json').read_text().count('layer_scan') == 1
    a = json.loads((out / 'allocation.json').read_text())['layer_scan']
    b = json.loads((tmp / 'full' / 'allocation.json').read_text())['layer_scan']
    assert a['order'] == b['order'] and a['greedy'] == b['greedy']


def test_records_survive_json_round_trip(base):
    config, tmp = base
    fit(with_drop(config), tmp / 'run')
    selection = json.loads((tmp / 'run' / 'selection.json').read_text())
    again = json.loads(json.dumps(selection))
    lock = json.loads((tmp / 'run' / 'selection-lock.json').read_text())['sha256']
    assert campaign._fingerprint(again) == campaign._fingerprint(selection) == lock
    assert campaign._selection(tmp / 'run') == selection


def test_stale_scan_cannot_be_reused(base):
    config, tmp = base
    cfg = with_drop(config)
    fit(cfg, tmp / 'run')
    for change in ({'examples': 2, 'candidates': LAYER_DROP['candidates']},
                   {'examples': 3, 'candidates': [{'id': 'layers-2', 'drop': 3}, LAYER_DROP['candidates'][1]]},
                   {'examples': 3, 'candidates': LAYER_DROP['candidates'] + [{'id': 'layers-4', 'drop': 4}]}):
        other = with_drop(config, change, train=('unpruned-recovery', 'uniform-20', 'layers-2'))
        with pytest.raises(ValueError, match='configuration changed'):
            campaign.run_campaign(other, tmp / 'run', 'fit')
    # Multi-seed sharing also refuses a different layer_drop block.
    from prune_lab import multiseed
    from test_independent_multiseed_run import FakeCampaign
    import prune_lab.campaign as c
    original = c.run_campaign
    c.run_campaign = FakeCampaign()
    try:
        multiseed.run_seeds(cfg, tmp / 'ms', [1])
        with pytest.raises(ValueError, match='beyond recovery.seed'):
            multiseed.run_seeds(with_drop(config, {'examples': 2, 'candidates': LAYER_DROP['candidates']}), tmp / 'ms', [2])
    finally:
        c.run_campaign = original


# --- (6) validation -------------------------------------------------------------------------------

@pytest.mark.parametrize('bad', [
    {'candidates': [{'id': 'x', 'drop': 0}]}, {'candidates': [{'id': 'x', 'drop': 5}]}, {'candidates': [{'id': 'x', 'drop': 7}]},
    {'candidates': [{'id': 'x', 'drop': True}]}, {'candidates': [{'id': 'x', 'drop': 1.0}]},
    {'candidates': [{'id': 'x', 'drop': 1}, {'id': 'x', 'drop': 2}]}, {'candidates': [{'id': 'uniform-20', 'drop': 1}]},
    {'candidates': [{'id': 'unpruned-recovery', 'drop': 1}]}, {'candidates': [{'id': '', 'drop': 1}]},
    {'candidates': [{'id': 'x', 'drop': 1, 'total_target': .1}]},  # dropping 1 of 5 already removes ~20%
    {'candidates': [{'id': 'x', 'drop': 1, 'total_target': .95}]}, {'candidates': [{'id': 'x', 'drop': 1, 'total_target': 1}]},
    {'candidates': [{'id': 'x', 'drop': 1, 'total_target': float('nan')}]}, {'candidates': [{'id': 'x', 'drop': 1, 'extra': 1}]},
    {'candidates': []}, {'candidates': {'id': 'x', 'drop': 1}}, {'examples': 0, 'candidates': [{'id': 'x', 'drop': 1}]},
    {'examples': 4, 'candidates': [{'id': 'x', 'drop': 1}]}, {'examples': True, 'candidates': [{'id': 'x', 'drop': 1}]},
    {'other': 1, 'candidates': [{'id': 'x', 'drop': 1}]}, {'examples': 3},
])
def test_invalid_layer_drop_configs_fail_before_any_work(base, bad):
    config, tmp = base
    broken = json.loads(json.dumps(config))
    broken['layer_drop'] = bad
    broken.pop('train_candidates')
    with pytest.raises(ValueError):
        campaign.validate_config(broken)
    with pytest.raises(ValueError):
        campaign.run_campaign(broken, tmp / 'never', 'preflight')
    assert not (tmp / 'never').exists() or not any((tmp / 'never').iterdir())


@pytest.mark.xfail(strict=True, reason='Edge case: a layer_drop candidate may be named "original"; collect_run then '
                   'overwrites the original-model baseline variant with that candidate, silently corrupting every '
                   'comparison against "original". The name is not reserved (pre-existing for other kinds too).')
def test_reserved_name_original_rejected(base):
    config, _ = base
    broken = json.loads(json.dumps(config))
    broken['layer_drop'] = {'candidates': [{'id': 'original', 'drop': 1}]}
    broken.pop('train_candidates')
    with pytest.raises(ValueError):
        campaign.validate_config(broken)


def test_shipped_config_matches_plan():
    raw = json.loads((ROOT / 'configs' / 'support-v2-layer-drop.json').read_text())
    longer = json.loads((ROOT / 'configs' / 'support-v2-longer.json').read_text())
    extra = set(raw) - set(longer)
    assert {k: v for k, v in raw.items() if k not in extra} == longer or extra <= {'layer_drop', 'train_candidates', 'uniform_per_target'}
    assert raw['layer_drop'] == {'examples': 150, 'candidates': [{'id': 'layers-7', 'drop': 7},
                                                                 {'id': 'layers-3-uniform-20', 'drop': 3, 'total_target': 0.2}]}
    assert raw['train_candidates'] == ['unpruned-recovery', 'layers-7', 'layers-3-uniform-20']
    assert raw['recovery'] == longer['recovery']
    config = campaign.read_campaign_config(ROOT / 'configs' / 'support-v2-layer-drop.json')
    assert [s['id'] for s in campaign.candidate_specs(config)] == ['unpruned-recovery', 'layers-7', 'layers-3-uniform-20']
