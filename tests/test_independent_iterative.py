"""Independent tests of improvement 7 (iterative pruning), from plans/improvement-7-iterative.md.

Tiny CPU campaigns on a synthetic three-domain archive. Recovery batches are wrapped so the
exact microbatch order seen by every recover_model call can be compared with the one-shot path.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from prune_lab import campaign, recovery, scoring

ROOT = Path(__file__).resolve().parents[1]
GIT_ENV = dict(os.environ, DEVELOPER_DIR='/Library/Developer/CommandLineTools')
PARENT = '3fa7cb3'
COUNTS = {'recovery': 4, 'calibration': 3, 'dev': 3, 'test': 3}


def test_worktree_package_is_under_test():
    assert Path(campaign.__file__).resolve().is_relative_to(ROOT / 'src')


@pytest.fixture
def base(tmp_path, monkeypatch):
    from prune_lab import support_data
    from test_independent_support_v2 import LAYOUT, build_archive
    monkeypatch.chdir(tmp_path)
    torch.set_num_threads(1)
    backend = Tokenizer(models.WordLevel({'[PAD]': 0, '[UNK]': 1, '[EOS]': 2, 'support': 3, 'reply': 4}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]', eos_token='[EOS]')
    torch.manual_seed(11)
    Qwen2ForCausalLM(Qwen2Config(vocab_size=5, hidden_size=16, intermediate_size=64, num_hidden_layers=2,
                                 num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
                                 pad_token_id=0, eos_token_id=2)).save_pretrained(tmp_path / 'model', safe_serialization=True)
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
    config['recovery'].update(rank=2, learning_rate=2e-2, max_steps=4, checkpoints=[2, 4], gradient_accumulation=1)
    return config, tmp_path


def iterative(config, **item):
    result = json.loads(json.dumps(config))
    result['iterative'] = dict({'id': 'iter-20', 'stages': [.1, .2], 'steps': [2, 2]}, **item)
    result['train_candidates'] = ['unpruned-recovery', 'uniform-20', 'iter-20']
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


class Tagged(dict):
    log = None

    def items(self):
        if Tagged.log is not None and getattr(self, 'tag', None) is not None:
            Tagged.log.append(self.tag)
        return super().items()


@pytest.fixture
def spy(monkeypatch):
    """Records recover_model calls (widths, recipe, the model's weights, data order) and selection-time evals."""
    calls, order, evals = [], [], []
    Tagged.log = order
    real_make, real_recover, real_eval = campaign.make_batches, recovery.recover_model, campaign._evaluate
    state = {'inside': False}

    def make(examples, tokenizer, *args, **kwargs):
        batches = real_make(examples, tokenizer, *args, **kwargs)
        examples = list(examples)
        if examples and len(examples) == COUNTS['recovery'] and \
                all(e.key == f.key for e, f in zip(examples, recovery_rows())):
            wrapped = []
            for i, b in enumerate(batches):
                t = Tagged(b)
                t.tag = i
                wrapped.append(t)
            return wrapped
        return batches

    def recover(model, batches, evaluate_dev, config, **kwargs):
        from prune_lab.recovery import LoRALinear
        assert not any(isinstance(m, LoRALinear) for m in model.modules())
        calls.append({'widths': [l.mlp.down_proj.in_features for l in model.model.layers], 'config': dict(config),
                      'gate0': model.model.layers[0].mlp.gate_proj.weight.detach().clone(),
                      'tags': [getattr(b, 'tag', None) for b in batches], 'order_start': len(order)})
        state['inside'] = True
        try:
            result = real_recover(model, batches, evaluate_dev, config, **kwargs)
        finally:
            state['inside'] = False
        calls[-1]['order'] = order[calls[-1]['order_start']:]
        calls[-1]['merged_gate0'] = model.model.layers[0].mlp.gate_proj.weight.detach().clone()
        calls[-1]['result'] = result
        assert not any(isinstance(m, LoRALinear) for m in model.modules())
        return result

    def evaluate(model, tokenizer, examples, config):
        evals.append((state['inside'] and len(calls) > 1, [e.key for e in examples]))  # call 1 is preflight's timing probe
        return real_eval(model, tokenizer, examples, config)
    monkeypatch.setattr(campaign, 'make_batches', make)
    monkeypatch.setattr(recovery, 'recover_model', recover)
    monkeypatch.setattr(campaign, '_evaluate', evaluate)
    yield calls, order, evals
    Tagged.log = None


_RECOVERY = []


def recovery_rows():
    return _RECOVERY


@pytest.fixture(autouse=True)
def _load_recovery_rows(request):
    _RECOVERY.clear()
    if 'base' in request.fixturenames:
        from prune_lab.support_data import read_support_jsonl
        config, _ = request.getfixturevalue('base')
        _RECOVERY.extend(read_support_jsonl(Path(config['support_data_dir']) / 'recovery.jsonl'))
    yield


# --- (1) fairness, (8) defaults ----------------------------------------------------------------

def test_oneshot_candidates_bit_identical_with_and_without_iterative(base):
    config, tmp = base
    plain = fit(config, tmp / 'plain')
    with_iter = fit(iterative(config), tmp / 'iter')
    for name in ('unpruned-recovery', 'uniform-20'):
        assert weights(with_iter[name]) == weights(plain[name]), name
        assert with_iter[name]['recovery']['train_losses'] == plain[name]['recovery']['train_losses']
        assert strip(with_iter[name]['dev']['predictions']) == strip(plain[name]['dev']['predictions'])
        assert with_iter[name]['layer_widths'] == plain[name]['layer_widths']
    alloc_plain = {c['id']: c for c in json.loads((tmp / 'plain' / 'allocation.json').read_text())['candidates']}
    alloc_iter = {c['id']: c for c in json.loads((tmp / 'iter' / 'allocation.json').read_text())['candidates']}
    for name in alloc_plain:
        assert alloc_iter[name]['keep_indices'] == alloc_plain[name]['keep_indices']
    assert list(with_iter) == ['unpruned-recovery', 'uniform-20', 'iter-20']


def test_default_fingerprints_specs_and_recovery_unchanged(tmp_path):
    names = [n for n in ('support.json', 'support-v2.json', 'support-v2-longer.json', 'support-v2-allocator.json',
                         'campaign.json', 'campaign-smoke.json') if (ROOT / 'configs' / n).exists()]
    try:
        tar = subprocess.run(['git', 'archive', PARENT, 'src'], cwd=ROOT, env=GIT_ENV, capture_output=True, check=True, timeout=60).stdout
        subprocess.run(['tar', '-x', '-C', str(tmp_path)], input=tar, check=True)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f'git history unavailable: {exc}')
    script = ('import json,sys; from prune_lab import campaign as c; out={}\n'
              'for n in sys.argv[1:]:\n  cfg=c.read_campaign_config(n); out[n]=[c._fingerprint(cfg), c.candidate_specs(cfg)]\n'
              'print(json.dumps(out))')
    paths = [str(ROOT / 'configs' / n) for n in names]
    old = json.loads(subprocess.run([sys.executable, '-c', script, *paths], env=dict(os.environ, PYTHONPATH=str(tmp_path / 'src')),
                                    capture_output=True, text=True, check=True).stdout)
    for path in paths:
        config = campaign.read_campaign_config(path)
        assert [campaign._fingerprint(config), campaign.candidate_specs(config)] == old[path], path
    # recover_model default (shuffling) path bit-identical to the parent's.
    source = (tmp_path / 'src' / 'prune_lab' / 'recovery.py').read_text().replace('from .', 'from prune_lab.')
    (tmp_path / 'old_recovery.py').write_text(source)
    spec = importlib.util.spec_from_file_location('old_recovery_parent', tmp_path / 'old_recovery.py')
    old_recovery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_recovery)
    from test_independent_recovery_recipes import Rising, batches, state_hash, tiny
    a, b = tiny(), tiny()
    cfg = {'rank': 2, 'alpha': 4, 'learning_rate': 1e-2, 'max_steps': 7, 'gradient_accumulation': 2, 'checkpoints': [3], 'seed': 7}
    old_recovery.recover_model(a, batches(5), Rising(), dict(cfg))
    recovery.recover_model(b, batches(5), Rising(), dict(cfg))
    assert state_hash(a) == state_hash(b)


def test_shipped_config_matches_plan():
    raw = json.loads((ROOT / 'configs' / 'support-v2-iterative.json').read_text())
    longer = json.loads((ROOT / 'configs' / 'support-v2-longer.json').read_text())
    assert {k: v for k, v in raw.items() if k not in ('uniform_per_target', 'iterative', 'train_candidates')} == longer
    config = campaign.read_campaign_config(ROOT / 'configs' / 'support-v2-iterative.json')
    campaign.validate_config(config)
    specs = {s['id']: s for s in campaign.candidate_specs(config)}
    assert list(specs) == ['unpruned-recovery', 'uniform-20', 'iter-20']
    assert specs['iter-20']['steps'] == [225, 225] and sum(specs['iter-20']['steps']) == config['recovery']['max_steps']
    assert [campaign.stage_checkpoints(specs['iter-20'], config['recovery'], k) for k in (0, 1)] == [[75, 150, 225]] * 2
    assert config['recovery']['warmup_steps'] == 20 and config['recovery']['lr_schedule'] == 'cosine'


# --- (2)-(6): stages, data, masks, selection --------------------------------------------------

def test_iterative_stages_data_masks_and_selection(base, spy):
    calls, order, evals = spy
    config, tmp = base
    cfg = iterative(config)
    records = fit(cfg, tmp / 'run')
    fit_calls = calls[1:]  # calls[0] is preflight
    oneshot, stage1, stage2 = fit_calls[1], fit_calls[2], fit_calls[3]
    assert [c['widths'] for c in fit_calls] == [[64, 64], [48, 48], [56, 56], [48, 48]]
    # (4)/(3): equal budget and one pass over the data in exactly the one-shot order.
    assert oneshot['config']['max_steps'] == 4 and stage1['config']['max_steps'] == stage2['config']['max_steps'] == 2
    assert stage1['order'] + stage2['order'] == oneshot['order']
    assert sorted(oneshot['order']) == list(range(4))
    expected = list(range(4))
    random.Random(cfg['recovery']['seed']).shuffle(expected)
    assert oneshot['order'] == expected
    assert stage1['config']['shuffle_batches'] is False and 'shuffle_batches' not in oneshot['config']
    for k, call in ((1, stage1), (2, stage2)):
        digest = int(hashlib.sha256(f"{cfg['recovery']['seed']}:iterative-stage-{k}".encode()).hexdigest()[:8], 16)
        assert call['config']['seed'] == digest
        assert call['config']['checkpoints'] == [1, 2]
        assert call['result']['optimizer_steps'] == 2
    record = records['iter-20']
    info = record['iterative']
    assert info['total_updates'] == 4 == sum(s['updates'] for s in info['stages'])
    assert [s['data_slice'] for s in info['stages']] == [[0, 2], [2, 4]]
    assert info['final_widths'] == info['oneshot_widths'] == records['uniform-20']['layer_widths'] == [48, 48]
    assert record['parameters'] == records['uniform-20']['parameters']
    # (3) Stage 2 starts from the merged stage-1 weights, compacted to the stage-2 keep set.
    stage1_keep = info['stage_keep_indices'][0]
    final_keep = info['stage_keep_indices'][1]
    local = [stage1_keep['0'].index(i) for i in final_keep['0']]
    assert torch.equal(stage2['gate0'], stage1['merged_gate0'][local])
    assert not torch.equal(stage1['merged_gate0'], stage1['gate0'])  # stage 1 actually trained and merged
    allocation = {c['id']: c for c in json.loads((tmp / 'run' / 'allocation.json').read_text())['candidates']}
    assert stage1_keep == allocation['uniform-10']['keep_indices'] if 'uniform-10' in allocation else True
    assert set(map(int, final_keep['0'])) <= set(map(int, stage1_keep['0']))
    assert info['kept_neurons_differing_from_oneshot'] == sum(
        len(set(final_keep[str(i)]) - set(allocation['uniform-20']['keep_indices'][str(i)])) for i in range(2))
    assert info['stages'][0]['scores_sha256'] != info['stages'][1]['scores_sha256']
    # (5) In-training selection only ever saw the selection subset.
    subset = json.loads((tmp / 'run' / 'selection-dev.json').read_text())['keys']
    inside = [keys for flag, keys in evals if flag]
    assert inside and all(sorted(keys) == sorted(subset) for keys in inside)
    # Intermediate weights deleted after the final export; final export intact and recorded.
    assert not list((tmp / 'run' / 'candidates' / 'iter-20').glob('stage-*/checkpoint/*.safetensors'))
    assert campaign.checkpoint_identity(record['checkpoint']) == record['checkpoint_sha256']
    exported = json.loads((Path(record['checkpoint']) / 'config.json').read_text())
    assert exported['prune_lab_keep_indices'] == [final_keep['0'], final_keep['1']]
    assert exported['prune_lab_original_widths'] == [64, 64] and exported['prune_lab_layer_widths'] == [48, 48]


def test_untrained_iterative_export_rows_equal_original_rows_at_recorded_indices(base, monkeypatch):
    """With recovery stubbed out (no training), exported weights are exactly the original rows/columns."""
    config, tmp = base

    def no_training(model, batches, evaluate_dev, cfg, **kwargs):
        dev = evaluate_dev(model)
        return {'initial_dev': dev, 'best_dev': dev, 'best_step': 0, 'history': [], 'optimizer_steps': 0,
                'train_losses': [], 'merged': True, 'seed': cfg['seed']}
    monkeypatch.setattr(recovery, 'recover_model', no_training)
    records = fit(iterative(config), tmp / 'run')
    record = records['iter-20']
    final_keep = record['iterative']['stage_keep_indices'][-1]
    from prune_lab.modeling import load_model
    original, _ = load_model(config['model'], device='cpu', dtype='float32')
    exported, _ = load_model(record['checkpoint'], device='cpu', dtype='float32')
    for i, (a, b) in enumerate(zip(original.model.layers, exported.model.layers)):
        keep = torch.tensor(final_keep[str(i)])
        assert torch.equal(b.mlp.gate_proj.weight, a.mlp.gate_proj.weight[keep])
        assert torch.equal(b.mlp.up_proj.weight, a.mlp.up_proj.weight[keep])
        assert torch.equal(b.mlp.down_proj.weight, a.mlp.down_proj.weight[:, keep])
        assert torch.equal(b.self_attn.q_proj.weight, a.self_attn.q_proj.weight)
    # (6) best_step 0 in every stage is handled and recorded.
    assert [s['best_step'] for s in record['iterative']['stages']] == [0, 0]
    assert record['status'] == 'complete'


# --- (7) resume ----------------------------------------------------------------------------------

def interrupt_on(monkeypatch, predicate):
    real = recovery.recover_model
    seen = []

    def wrapper(model, batches, evaluate_dev, cfg, **kwargs):
        seen.append(cfg)
        if predicate(cfg, len(seen)):
            raise KeyboardInterrupt
        return real(model, batches, evaluate_dev, cfg, **kwargs)
    monkeypatch.setattr(recovery, 'recover_model', wrapper)
    return real, seen


def is_stage(cfg, k, seed):
    return cfg.get('shuffle_batches') is False and cfg['seed'] == int(
        hashlib.sha256(f"{seed}:iterative-stage-{k}".encode()).hexdigest()[:8], 16)


def test_resume_between_stages_bit_identical_and_corrupt_stage_redone(base, monkeypatch):
    config, tmp = base
    cfg = iterative(config)
    seed = cfg['recovery']['seed']
    full = fit(cfg, tmp / 'full')['iter-20']
    out = tmp / 'resumed'
    campaign.run_campaign(cfg, out, 'preflight')
    real, seen = interrupt_on(monkeypatch, lambda c, n: is_stage(c, 2, seed))
    with pytest.raises(KeyboardInterrupt):
        campaign.run_campaign(cfg, out, 'fit')
    stage_dir = out / 'candidates' / 'iter-20' / 'stage-1'
    assert (stage_dir / 'stage.json').exists() and list((stage_dir / 'checkpoint').glob('*.safetensors'))
    monkeypatch.setattr(recovery, 'recover_model', real)
    _, seen = interrupt_on(monkeypatch, lambda c, n: False)
    campaign.run_campaign(cfg, out, 'fit')
    assert not any(is_stage(c, 1, seed) for c in seen) and sum(is_stage(c, 2, seed) for c in seen) == 1
    resumed = {c['id']: c for c in campaign._selection(out)['candidates']}['iter-20']
    assert weights(resumed) == weights(full)
    assert strip(resumed['dev']['predictions']) == strip(full['dev']['predictions'])
    assert resumed['iterative']['stage_keep_indices'] == full['iterative']['stage_keep_indices']

    # Corrupted stage-1 export: stage 1 is redone, result still identical.
    out2 = tmp / 'corrupt'
    campaign.run_campaign(cfg, out2, 'preflight')
    monkeypatch.setattr(recovery, 'recover_model', real)
    real, seen = interrupt_on(monkeypatch, lambda c, n: is_stage(c, 2, seed))
    with pytest.raises(KeyboardInterrupt):
        campaign.run_campaign(cfg, out2, 'fit')
    weights_file = next((out2 / 'candidates' / 'iter-20' / 'stage-1' / 'checkpoint').glob('*.safetensors'))
    data = bytearray(weights_file.read_bytes())
    data[-3] ^= 0xFF
    weights_file.write_bytes(bytes(data))
    monkeypatch.setattr(recovery, 'recover_model', real)
    _, seen = interrupt_on(monkeypatch, lambda c, n: False)
    campaign.run_campaign(cfg, out2, 'fit')
    assert sum(is_stage(c, 1, seed) for c in seen) == 1 and sum(is_stage(c, 2, seed) for c in seen) == 1
    redone = {c['id']: c for c in campaign._selection(out2)['candidates']}['iter-20']
    assert weights(redone) == weights(full)


def test_failure_in_stage_two_is_recorded_without_leaking(base, monkeypatch):
    config, tmp = base
    cfg = iterative(config)
    seed = cfg['recovery']['seed']
    real = recovery.recover_model

    def failing(model, batches, evaluate_dev, c, **kwargs):
        if is_stage(c, 2, seed):
            raise FloatingPointError('Nonfinite recovery loss')
        return real(model, batches, evaluate_dev, c, **kwargs)
    monkeypatch.setattr(recovery, 'recover_model', failing)
    records = fit(cfg, tmp / 'run')
    assert records['iter-20']['status'] == 'failed' and 'Nonfinite' in records['iter-20']['error']
    assert records['uniform-20']['status'] == 'complete'


# --- (9) validation --------------------------------------------------------------------------------

@pytest.mark.parametrize('bad', [
    {'steps': [2, 1]}, {'steps': [3, 3]}, {'steps': [4]}, {'steps': [2, 2, 0]}, {'steps': [2.0, 2]},
    {'stages': [.2, .1]}, {'stages': [.1, .1]}, {'stages': [.2]}, {'stages': [0, .2]}, {'stages': [.1, 1.0]},
    {'stages': [.1, True]}, {'id': ''}, {'id': 'uniform-20'}, {'id': 'unpruned-recovery'}, {'extra': 1},
    {'checkpoints': [[1], [3]]}, {'checkpoints': [[0], [2]]}, {'checkpoints': [[1]]},
])
def test_invalid_iterative_configs_rejected(base, bad):
    config, _ = base
    broken = json.loads(json.dumps(config))
    broken['iterative'] = dict({'id': 'iter-20', 'stages': [.1, .2], 'steps': [2, 2]}, **bad)
    broken.pop('train_candidates')
    with pytest.raises(ValueError):
        campaign.validate_config(broken)


def test_warmup_must_fit_each_stage(base):
    config, _ = base
    broken = json.loads(json.dumps(config))
    broken['recovery'].update(lr_schedule='cosine', warmup_steps=2, min_lr_ratio=.1)
    broken['iterative'] = {'id': 'iter-20', 'stages': [.1, .2], 'steps': [2, 2]}
    broken.pop('train_candidates')
    with pytest.raises(ValueError, match='warmup'):
        campaign.validate_config(broken)


def test_infeasible_stage_target_rejected_up_front(base):
    config, _ = base
    broken = json.loads(json.dumps(config))
    broken['iterative'] = {'id': 'iter-90', 'stages': [.1, .9], 'steps': [2, 2]}
    broken.pop('train_candidates')
    with pytest.raises(ValueError):
        campaign.validate_config(broken)


def test_stages_with_identical_widths_rejected(base):
    from prune_lab.allocation import allocate_neurons
    from prune_lab.modeling import load_model
    config, _ = base
    model, _ = load_model(config['model'], device='cpu', dtype='float32')
    scores = {i: torch.rand(64) for i in range(2)}
    widths = [len(allocate_neurons(model, scores, {0: 0, 1: 0}, t, 4, .25, True)[0]) for t in (.1, .11)]
    assert widths[0] == widths[1]  # precondition: identical uniform widths
    broken = json.loads(json.dumps(config))
    broken['iterative'] = {'id': 'iter-11', 'stages': [.1, .11], 'steps': [2, 2]}
    broken.pop('train_candidates')
    with pytest.raises(ValueError):
        campaign.validate_config(broken)
