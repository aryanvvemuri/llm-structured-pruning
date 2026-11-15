"""Independent tests of improvement 5 (pruned Qwen2.5-1.5B vs native 0.5B), from
plans/improvement-5-larger-model.md. Meta-device shapes from the cached real config files
(no weights), synthetic run directories, and the preflight script on a tiny CPU model with
poisoned dev/test/calibration files.
"""
import glob
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from prune_lab import campaign, cli, multiseed
from prune_lab.allocation import allocate_neurons
from prune_lab.modeling import count_parameters, mlp_layers
from prune_lab.pruning import compact_model

ROOT = Path(__file__).resolve().parents[1]
HUB = Path('/Users/aryan/llm_pruning_optimization/.cache/huggingface/hub')
SCRIPT = ROOT / 'scripts' / 'preflight_model.py'
GIT_ENV = dict(os.environ, DEVELOPER_DIR='/Library/Developer/CommandLineTools')
PARENT = '8c740c9'  # land/4-quantization tip this branch sits on (was improve/1-allocator 3fa7cb3 on improve/5)


def test_worktree_package_is_under_test():
    assert Path(campaign.__file__).resolve().is_relative_to(ROOT / 'src')


def cached_config(name):
    paths = glob.glob(str(HUB / f'models--Qwen--{name}' / 'snapshots' / '*' / 'config.json'))
    if not paths:
        pytest.skip(f'{name} config not cached')
    raw = json.loads(Path(paths[0]).read_text())
    return Qwen2Config(**{k: v for k, v in raw.items() if k not in ('architectures', 'transformers_version', 'torch_dtype')})


def meta_model(name):
    with torch.device('meta'):
        return Qwen2ForCausalLM(cached_config(name))


# --- sizes ---------------------------------------------------------------------------------------

def test_1p5b_sizes_from_the_real_config():
    model = meta_model('Qwen2.5-1.5B-Instruct')
    config = model.config
    assert (config.hidden_size, config.intermediate_size, config.num_hidden_layers, config.tie_word_embeddings) == (1536, 8960, 28, True)
    total = count_parameters(model)
    assert total == 1_543_714_304
    mlp = sum(p.numel() for m in mlp_layers(model) for p in m.parameters())
    embeddings = model.model.embed_tokens.weight.numel()
    attention = sum(p.numel() for l in model.model.layers for p in l.self_attn.parameters())
    assert (round(mlp / 1e6), round(embeddings / 1e6), round(attention / 1e6)) == (1156, 233, 154)  # plan's breakdown
    assert count_parameters(meta_model('Qwen2.5-0.5B-Instruct')) == 494_032_768
    # Through the same uniform allocator the campaign uses, with the config's group size and floor.
    run_config = campaign.read_campaign_config(ROOT / 'configs' / 'support-v2-1p5b.json')
    group, floor_ratio = run_config['group_size'], run_config['min_keep_ratio']
    assert (group, floor_ratio) == (64, .25)
    scores = {i: torch.rand(8960, generator=torch.Generator().manual_seed(i)) for i in range(28)}
    sizes = {}
    for target in run_config['targets']:
        keep = allocate_neurons(model, scores, {i: 0.0 for i in range(28)}, target, group, floor_ratio, uniform=True)
        widths = {len(v) for v in keep.values()}
        assert len(widths) == 1
        width = widths.pop()
        assert width % 64 == 0 and width >= math.ceil(8960 * .25)  # 25% floor respected
        removed = 28 * (8960 - width) * 3 * 1536
        assert removed == total - (total - removed)
        sizes[target] = total - removed
        # Nearest uniform step to the target (one step = 64 neurons in every layer).
        step = 28 * 64 * 3 * 1536
        assert abs(removed - target * total) <= step / 2
        # Lowest Taylor scores removed in each layer.
        for i, kept in keep.items():
            assert set(kept) == set(scores[i].argsort(stable=True)[8960 - width:].tolist())
    assert sizes == {.25: 1_155_610_112, .5: 775_763_456}
    floor_width = math.ceil(8960 * .25)
    maximum = 28 * (8960 - floor_width) * 3 * 1536
    assert maximum == 867_041_280 and total - maximum == 676_673_024
    assert (total - maximum) / 494_032_768 == pytest.approx(1.3697, abs=1e-4)  # "still 37% larger"
    allocate_neurons(model, scores, {i: 0.0 for i in range(28)}, maximum / total - 1e-9, group, floor_ratio, uniform=True)
    with pytest.raises(ValueError, match='Infeasible'):
        allocate_neurons(model, scores, {i: 0.0 for i in range(28)}, maximum / total + 1e-6, group, floor_ratio, uniform=True)


def test_compacted_meta_model_has_the_claimed_size():
    model = meta_model('Qwen2.5-1.5B-Instruct')
    keep = {i: list(range(8960 - 93 * 64)) for i in range(28)}  # uniform-50: 93 steps of 64
    try:
        compact_model(model, keep)
    except Exception as exc:  # pragma: no cover - meta tensors unsupported by a code path
        pytest.skip(f'compaction on meta device unsupported: {exc}')
    assert count_parameters(model) == 775_763_456
    assert model.config.prune_lab_layer_widths == [8960 - 93 * 64] * 28


def test_tokenizers_byte_identical():
    a = glob.glob(str(HUB / 'models--Qwen--Qwen2.5-0.5B-Instruct' / 'snapshots' / '*' / 'tokenizer.json'))
    b = glob.glob(str(HUB / 'models--Qwen--Qwen2.5-1.5B-Instruct' / 'snapshots' / '*' / 'tokenizer.json'))
    if not a or not b:
        pytest.skip('tokenizers not cached')
    assert Path(a[0]).read_bytes() == Path(b[0]).read_bytes()


# --- configs and 0.5B path unchanged ------------------------------------------------------------

def test_1p5b_config_is_longer_plus_four_keys():
    new = json.loads((ROOT / 'configs' / 'support-v2-1p5b.json').read_text())
    longer = json.loads((ROOT / 'configs' / 'support-v2-longer.json').read_text())
    assert set(new) - set(longer) == {'uniform_per_target', 'train_candidates'}
    assert {k for k in longer if new.get(k) != longer[k]} == {'model', 'targets'}
    assert new['model'] == 'Qwen/Qwen2.5-1.5B-Instruct' and new['targets'] == [.25, .5]
    assert new['recovery'] == longer['recovery']
    config = campaign.read_campaign_config(ROOT / 'configs' / 'support-v2-1p5b.json')
    campaign.validate_config(config)
    specs = campaign.candidate_specs(config)
    assert [s['id'] for s in specs] == ['unpruned-recovery', 'uniform-25', 'uniform-50']
    assert [s['target'] for s in specs] == [0.0, .25, .5] and all(s['uniform'] for s in specs[1:])


def test_0p5b_fingerprints_and_specs_unchanged_from_parent(tmp_path):
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
    changed = subprocess.run(['git', 'diff', '--name-only', '7662eb7^', '7662eb7', '--', 'src'], cwd=ROOT, env=GIT_ENV,
                             capture_output=True, text=True, check=True).stdout.split()
    assert set(changed) <= {'src/prune_lab/cli.py', 'src/prune_lab/multiseed.py'}  # masks/training/eval untouched by the improvement-5 commit itself


# --- cross-model comparison ------------------------------------------------------------------------

KEYS = [f'k{i}' for i in range(20)]
SUBSET = KEYS[:5]


def write_seed(root, seed, model='Qwen/Qwen2.5-0.5B-Instruct', variant='unpruned-recovery', level=.4, omit_model=False):
    directory = root / f'seed-{seed}'
    directory.mkdir(parents=True)
    config = campaign.read_campaign_config(ROOT / 'configs' / 'support-v2-longer.json')
    config['recovery']['seed'] = seed
    config['model'] = model
    if omit_model:
        config.pop('model')
    (directory / 'config.json').write_text(json.dumps(config))
    (directory / 'state.json').write_text(json.dumps({'dataset': {'dev': {'count': len(KEYS), 'sha256': 'same'}}, 'test_started': False}))

    def metrics(shift):
        scores = {k: min(1, max(0, level + shift + .02 * ((i * 7 + seed) % 5))) for i, k in enumerate(KEYS)}
        return {'token_f1': sum(scores.values()) / 20, 'rouge_l': sum(scores.values()) / 20, 'answer_token_loss': 1.0,
                'predictions': [{'key': k, 'group_id': f'g{i % 4}', 'score': v, 'rouge_l': v} for i, (k, v) in enumerate(scores.items())]}
    selection = {'baseline': {'dev': metrics(-.2), 'parameters': 100},
                 'candidates': [{'id': variant, 'status': 'complete', 'parameters': 90, 'actual_reduction': 0.0,
                                 'checkpoint': 'x', 'dev': metrics(0)}], 'tiers': []}
    (directory / 'selection.json').write_text(json.dumps(selection))
    (directory / 'selection-lock.json').write_text(json.dumps({'sha256': campaign._fingerprint(selection)}))
    (directory / 'selection-dev.json').write_text(json.dumps({'count': 5, 'keys_sha256': 'h', 'keys': SUBSET}))
    return directory


SEEDS = (20261114, 20261115, 20261116)


@pytest.fixture
def runs(tmp_path):
    small, large = tmp_path / 'support-v2-longer', tmp_path / 'support-v2-1p5b'
    for seed in SEEDS:
        write_seed(small, seed)
        write_seed(large, seed, model='Qwen/Qwen2.5-1.5B-Instruct', variant='uniform-50', level=.5)
    return small, large


def test_cross_model_compare_refused_without_opt_in(runs):
    small, large = runs
    with pytest.raises(ValueError, match='allow-different-models'):
        multiseed.compare_runs(small, large, pairs=[('unpruned-recovery', 'uniform-50')], samples=50)
    with pytest.raises(ValueError, match='allow-different-models'):  # forcing unpaired seeds is not an opt-in
        multiseed.compare_runs(small, large, pairs=[('unpruned-recovery', 'uniform-50')], paired_seeds=False, samples=50)
    with pytest.raises(SystemExit) as exc:  # the plan's primary command as written, without the flag
        cli.main(['multi-seed', 'compare', '--baseline', str(small), '--candidate', str(large),
                  '--pair', 'unpruned-recovery:uniform-50', '--unpaired-seeds', '--samples', '50'])
    assert exc.value.code != 0
    assert not list(large.glob('compare-vs-*'))


def test_cross_model_compare_with_opt_in_is_unpaired_and_records_models(runs):
    small, large = runs
    result = multiseed.compare_runs(small, large, pairs=[('unpruned-recovery', 'uniform-50')], samples=200,
                                    allow_different_models=True)
    assert result['base_models'] == {'baseline': 'Qwen/Qwen2.5-0.5B-Instruct',
                                     'candidate': 'Qwen/Qwen2.5-1.5B-Instruct', 'identical': False}
    assert result['allow_different_models'] is True
    (comparison,) = result['comparisons']
    assert comparison['token_f1']['seed_pairing'] == 'independent'  # same seed values, still unpaired
    assert comparison['token_f1']['estimate_points'] == pytest.approx(10, abs=1e-9)
    assert result['example_set'] == 'dev-holdout' and result['example_count'] == 15
    text = multiseed.render_comparison(result)
    assert 'Qwen/Qwen2.5-0.5B-Instruct' in text and 'Qwen/Qwen2.5-1.5B-Instruct' in text
    out = large / 'cmp.json'
    cli.main(['multi-seed', 'compare', '--baseline', str(small), '--candidate', str(large), '--pair', 'unpruned-recovery:uniform-50',
              '--unpaired-seeds', '--allow-different-models', '--samples', '50', '--output', str(out)])
    saved = json.loads(out.read_text())
    assert saved['base_models']['identical'] is False and saved['comparisons'][0]['token_f1']['seed_pairing'] == 'independent'
    # An explicit --paired-seeds is honoured (and visible), never applied silently.
    forced = multiseed.compare_runs(small, large, pairs=[('unpruned-recovery', 'uniform-50')], paired_seeds=True, samples=50,
                                    allow_different_models=True)
    assert forced['comparisons'][0]['token_f1']['seed_pairing'] == 'paired'


def test_same_model_compare_unchanged_and_still_paired(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    for seed in SEEDS:
        write_seed(a, seed)
        write_seed(b, seed, level=.45)
    result = multiseed.compare_runs(a, b, samples=50)
    assert result['base_models']['identical'] and result['allow_different_models'] is False
    assert result['comparisons'][0]['token_f1']['seed_pairing'] == 'paired'
    assert 'Different base models' not in multiseed.render_comparison(result)


def test_run_mixing_models_is_refused(tmp_path, monkeypatch):
    mixed = tmp_path / 'mixed'
    write_seed(mixed, SEEDS[0])
    write_seed(mixed, SEEDS[1], model='Qwen/Qwen2.5-1.5B-Instruct')
    with pytest.raises(ValueError, match='different base models'):
        multiseed.collect_run(mixed)
    with pytest.raises(ValueError, match='different base models'):
        multiseed.aggregate_run(mixed, samples=20)
    assert not (mixed / 'multi-seed-summary.json').exists()
    # Multi-seed runs cannot be extended with a different model either (base-config check).
    from test_independent_multiseed_run import FakeCampaign
    import prune_lab.campaign as c
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(c, 'run_campaign', FakeCampaign())
    config = c.read_campaign_config()
    config['device'] = 'cpu'
    multiseed.run_seeds(config, tmp_path / 'ms', [1])
    with pytest.raises(ValueError, match='beyond recovery.seed'):
        multiseed.run_seeds(dict(config, model='Qwen/Qwen2.5-1.5B-Instruct'), tmp_path / 'ms', [2])


def test_runs_without_a_recorded_model_still_aggregate_and_compare(tmp_path):
    a, b = tmp_path / 'old-a', tmp_path / 'old-b'
    for seed in SEEDS:
        write_seed(a, seed, omit_model=True)
        write_seed(b, seed, omit_model=True, level=.45)
    assert multiseed.collect_run(a)['base_model'] is None
    assert multiseed.aggregate_run(a, samples=20)['seeds'] == list(SEEDS)
    result = multiseed.compare_runs(a, b, samples=50)
    assert result['base_models']['identical'] and result['comparisons'][0]['token_f1']['seed_pairing'] == 'paired'
    # Every config written by campaign._initialize records a model (read_campaign_config default), so real
    # old runs carry one: e.g. support-v2-longer resolves to the 0.5B id.
    assert campaign.read_campaign_config(ROOT / 'configs' / 'support-v2-longer.json')['model'] == 'Qwen/Qwen2.5-0.5B-Instruct'


# --- preflight script --------------------------------------------------------------------------------

def test_preflight_refuses_output_under_runs(tmp_path):
    for target in (tmp_path / 'runs' / 'p.json', ROOT / 'runs' / 'support-v2-1p5b' / 'p.json'):
        out = subprocess.run([sys.executable, str(SCRIPT), '--config', str(ROOT / 'configs' / 'support-v2-1p5b.json'),
                              '--output', str(target), '--device', 'cpu'], capture_output=True, text=True, timeout=120,
                             env=dict(os.environ, HF_HUB_OFFLINE='1'))
        assert out.returncode == 2 and 'outside runs/' in out.stderr
        assert not target.exists()


@pytest.fixture
def tiny_run(tmp_path, monkeypatch):
    """Tiny model + data directory whose dev/test/calibration files are unreadable."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from prune_lab import support_data
    from test_independent_support_v2 import LAYOUT, build_archive
    monkeypatch.chdir(tmp_path)
    torch.set_num_threads(1)
    backend = Tokenizer(models.WordLevel({'[PAD]': 0, '[UNK]': 1, '[EOS]': 2, 'reply': 3}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]', eos_token='[EOS]')
    torch.manual_seed(1)
    Qwen2ForCausalLM(Qwen2Config(vocab_size=4, hidden_size=16, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2,
                                 num_key_value_heads=1, max_position_embeddings=256, pad_token_id=0, eos_token_id=2)
                     ).save_pretrained(tmp_path / 'model', safe_serialization=True)
    tokenizer.save_pretrained(tmp_path / 'model')
    data = tmp_path / 'data'
    support_data.prepare_support_dataset(build_archive(tmp_path, monkeypatch, LAYOUT), data, tokenizer,
                                         counts={'recovery': 6, 'calibration': 3, 'dev': 6, 'test': 3},
                                         max_length=256, max_answer_tokens=12)
    for name in ('dev', 'test', 'calibration'):
        (data / f'{name}.jsonl').chmod(0)
    config = json.loads((ROOT / 'configs' / 'support-v2-1p5b.json').read_text())
    config.update(model=str(tmp_path / 'model'), support_data_dir=str(data), device='cpu', cpu_threads=1, max_new_tokens=12,
                  max_length=256, eval_batch_size=2)
    path = tmp_path / 'tiny-1p5b.json'
    path.write_text(json.dumps(config))
    yield path, data
    for name in ('dev', 'test', 'calibration'):
        (data / f'{name}.jsonl').chmod(0o600)


def recompute(report, config):
    est, u = report['estimate'], report['unpruned']
    steps = config['recovery']['max_steps']
    replies = 4 * 150 + 150 + 600
    candidate = (steps * u['seconds_per_update'] + replies * u['seconds_per_reply_including_loss_pass'] + 120) / 3600
    baseline = (600 * u['seconds_per_reply_including_loss_pass'] + 600) / 3600
    assert est['replies_per_candidate'] == replies and est['candidates_per_seed'] == 3
    assert est['hours_per_candidate_upper'] == pytest.approx(candidate)
    assert est['hours_five_seeds'] == pytest.approx(baseline + 15 * candidate)
    assert est['hours_three_seeds'] == pytest.approx(baseline + 9 * candidate)
    return baseline + 15 * candidate


def test_preflight_runs_on_recovery_data_only(tiny_run, tmp_path):
    config_path, data = tiny_run
    out = subprocess.run([sys.executable, str(SCRIPT), '--config', str(config_path), '--output', str(tmp_path / 'pre.json'),
                          '--updates', '2', '--replies', '2', '--device', 'cpu', '--prune-uniform', '0.25'],
                         capture_output=True, text=True, timeout=300, env=dict(os.environ, HF_HUB_OFFLINE='1'))
    assert out.returncode == 0, out.stderr[-2000:]
    report = json.loads((tmp_path / 'pre.json').read_text())
    assert report['unpruned']['replies'] == 2 and report['unpruned']['updates'] == 2
    assert report['uniform-0.25']['parameters'] < report['unpruned']['parameters']
    five = recompute(report, json.loads(config_path.read_text()))
    assert report['estimate']['seeds'] == (5 if five <= 10 else 3)
    assert not any(p.name == 'runs' for p in tmp_path.iterdir())


@pytest.mark.parametrize('per_update,seeds', [(3.43, 5), (3.44, 3)])
def test_seed_rule_boundary_depends_only_on_timings(tiny_run, tmp_path, monkeypatch, per_update, seeds):
    """With 0.5 s per reply, the five-seed estimate crosses 10 h between 3.43 and 3.44 s per update."""
    from prune_lab import recovery, support_evaluation
    config_path, _ = tiny_run
    clock = [0.0]
    monkeypatch.setattr(time, 'perf_counter', lambda: clock[0])
    seen = []

    def fake_recover(model, batches, evaluate_dev, recipe, **kw):
        scores = [evaluate_dev(model)]  # whatever the dev callback says must not matter
        seen.append(('dev', scores[0]))
        return {'training_seconds': per_update * recipe['max_steps'], 'peak_mps_driver_bytes': None}

    def fake_evaluate(model, tokenizer, examples, *args, **kwargs):
        seen.append(('replies', [e.key for e in examples]))
        clock[0] += 0.5 * len(examples)
        return {'generation_seconds': 0.4 * len(examples), 'generated_tokens': 7, 'token_f1': 0.99}
    monkeypatch.setattr(recovery, 'recover_model', fake_recover)
    monkeypatch.setattr(support_evaluation, 'evaluate_support', fake_evaluate)
    monkeypatch.setattr(sys, 'argv', ['preflight', '--config', str(config_path), '--output', str(tmp_path / 'p.json'),
                                      '--updates', '4', '--replies', '3', '--device', 'cpu'])
    runpy.run_path(str(SCRIPT), run_name='__main__')
    report = json.loads((tmp_path / 'p.json').read_text())
    five = recompute(report, json.loads(config_path.read_text()))
    assert report['estimate']['seeds'] == seeds and (five <= 10) == (seeds == 5)
    assert seen[0] == ('dev', {'selection_score': 0.0, 'score_name': 'token_f1', 'answer_token_loss': 1.0})
    recovery_keys = {json.loads(l)['key'] for l in (Path(json.loads(config_path.read_text())['support_data_dir']) / 'recovery.jsonl').read_text().splitlines()}
    assert set(seen[1][1]) <= recovery_keys
