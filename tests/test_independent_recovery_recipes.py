"""Independent recover_model-level checks of improvement 2 (2a schedule, 2b distillation, 2c attention LoRA).

Written from plans/improvement-2-recovery.md and the code; does not reuse the writer's helpers.
Tiny random Qwen2 models on CPU.
"""
import copy
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import subprocess

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from prune_lab import campaign, recipes, recovery
from prune_lab.recovery import LoRALinear, distillation_loss, recover_model

ROOT = Path(__file__).resolve().parents[1]
H, I, V = 16, 24, 23


def tiny(seed=0):
    torch.manual_seed(seed)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=V, hidden_size=H, intermediate_size=I, num_hidden_layers=2,
                                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                                         pad_token_id=0, eos_token_id=2))
    return model.float()


def batches(n=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        prompt = torch.randint(3, V, (5 + i % 2,), generator=g)
        answer = torch.randint(3, V, (3 + i % 3,), generator=g)
        ids = torch.cat([prompt, answer]).unsqueeze(0)
        labels = torch.cat([torch.full_like(prompt, -100), answer]).unsqueeze(0)
        out.append({'input_ids': ids, 'attention_mask': torch.ones_like(ids), 'labels': labels})
    return out


class Rising:
    """Dev callback whose score rises each call, so the last checkpoint is selected (trained adapter merged)."""

    def __init__(self, probe=None):
        self.calls, self.probe, self.logits = 0, probe, []

    def __call__(self, model):
        self.calls += 1
        if self.probe is not None:
            with torch.no_grad():
                self.logits.append(model(input_ids=self.probe, use_cache=False).logits.clone())
        return {'selection_score': self.calls / 1000, 'score_name': 'token_f1', 'answer_token_loss': 1.0}


def base_cfg(**extra):
    cfg = {'rank': 2, 'alpha': 4, 'learning_rate': 1e-2, 'max_steps': 6, 'gradient_accumulation': 1,
           'checkpoints': [3, 6], 'seed': 7}
    cfg.update(extra)
    return cfg


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


class RecordingAdamW(torch.optim.AdamW):
    applied = []

    def step(self, *a, **k):
        RecordingAdamW.applied.append([g['lr'] for g in self.param_groups])
        return super().step(*a, **k)


@pytest.fixture
def record_lr(monkeypatch):
    RecordingAdamW.applied = []
    monkeypatch.setattr(torch.optim, 'AdamW', RecordingAdamW)
    return RecordingAdamW.applied


# --- 2a: schedule ---------------------------------------------------------------------------

def plan_factor(u, steps, warmup, ratio):
    if u < warmup:
        return (u + 1) / warmup
    return ratio + (1 - ratio) * (1 + math.cos(math.pi * (u - warmup) / (steps - warmup))) / 2


def test_cosine_rate_actually_applied_each_update_and_recorded(record_lr):
    steps, warmup, peak = 30, 5, 1e-3
    cfg = base_cfg(max_steps=steps, checkpoints=[10, 20, 30], learning_rate=peak, lr_schedule='cosine',
                   warmup_steps=warmup, min_lr_ratio=.1)
    result = recover_model(tiny(), batches(4), Rising(), cfg)  # 30 updates span 7.5 epochs
    expected = [peak * plan_factor(u, steps, warmup, .1) for u in range(steps)]
    assert len(record_lr) == steps and all(len(set(g)) == 1 for g in record_lr)
    applied = [g[0] for g in record_lr]
    assert applied == pytest.approx(expected, rel=1e-12)
    assert result['learning_rates'] == applied
    assert applied[0] == pytest.approx(peak / warmup) and applied[warmup - 1] == pytest.approx(peak)
    assert applied[warmup] == pytest.approx(peak)  # first post-warmup update at peak
    assert all(a > b for a, b in zip(applied[warmup:], applied[warmup + 1:]))
    assert min(applied[warmup:]) > .1 * peak
    for entry in result['history']:
        if entry['step']:
            assert entry['learning_rate'] == applied[entry['step'] - 1]
        else:
            assert entry['learning_rate'] is None
    recipe = result['recipe']
    assert recipe['lr_schedule'] == 'cosine' and recipe['warmup_steps'] == warmup and recipe['min_lr_ratio'] == .1


def test_plan_values_of_the_longer_recipe():
    f = lambda u: recipes.lr_factor(u, 450, 'cosine', 20, .1)
    assert [f(u) for u in (0, 9, 19)] == pytest.approx([1 / 20, 10 / 20, 1.0])
    assert f(20) == pytest.approx(1.0)
    assert f(235) == pytest.approx(.1 + .9 * .5)  # halfway through the decay
    assert f(449) == pytest.approx(.1, abs=1e-4) and f(449) > .1
    assert recipes.lr_factor(123, 450) == 1.0


def load_old_recovery(tmp_path):
    """recovery.py as of d993dc9 (before improvement 2), imported under a private name."""
    env = dict(os.environ, DEVELOPER_DIR='/Library/Developer/CommandLineTools')
    try:
        source = subprocess.run(['git', 'show', 'd993dc9:src/prune_lab/recovery.py'], cwd=ROOT, env=env,
                                capture_output=True, text=True, check=True, timeout=30).stdout
    except Exception as exc:  # pragma: no cover - only when git history is unavailable
        pytest.skip(f'cannot read d993dc9: {exc}')
    source = source.replace('from .modeling', 'from prune_lab.modeling').replace('from .metrics', 'from prune_lab.metrics')
    path = tmp_path / 'old_recovery_d993dc9.py'
    path.write_text(source)
    spec = importlib.util.spec_from_file_location('old_recovery_d993dc9', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('extra', [{}, {'lr_schedule': 'constant'}, {'lora_targets': 'mlp'},
                                   {'distillation': {'alpha': .5, 'temperature': 2.0}}])
def test_default_paths_bit_identical_to_pre_change_recovery(tmp_path, extra):
    old = load_old_recovery(tmp_path)
    reference, model = tiny(), tiny()
    cfg = base_cfg(max_steps=7, checkpoints=[2, 5], gradient_accumulation=2)
    before = old.recover_model(reference, batches(5), Rising(), dict(cfg))
    after = recover_model(model, batches(5), Rising(), dict(cfg, **extra))  # no teacher passed
    assert state_hash(model) == state_hash(reference)
    assert after['train_losses'] == before['train_losses']
    assert after['best_step'] == before['best_step'] and after['optimizer_steps'] == before['optimizer_steps']
    assert after['learning_rates'] == [cfg['learning_rate']] * 7
    assert 'distillation' not in after
    assert set(before) <= set(after)


def test_cosine_with_no_warmup_and_ratio_one_is_constant_training():
    a, b = tiny(), tiny()
    recover_model(a, batches(), Rising(), base_cfg())
    recover_model(b, batches(), Rising(), base_cfg(lr_schedule='cosine', warmup_steps=0, min_lr_ratio=1.0))
    assert state_hash(a) == state_hash(b)


@pytest.mark.parametrize('bad', [
    {'lr_schedule': 'linear'}, {'lr_schedule': None}, {'lr_schedule': 'cosine', 'warmup_steps': 6},
    {'lr_schedule': 'cosine', 'warmup_steps': 7}, {'lr_schedule': 'cosine', 'warmup_steps': -1},
    {'lr_schedule': 'cosine', 'warmup_steps': 2.0}, {'lr_schedule': 'cosine', 'warmup_steps': True},
    {'lr_schedule': 'cosine', 'min_lr_ratio': -.1}, {'lr_schedule': 'cosine', 'min_lr_ratio': 1.5},
    {'lr_schedule': 'cosine', 'min_lr_ratio': float('nan')}, {'lr_schedule': 'cosine', 'min_lr_ratio': True},
    {'warmup_steps': 2}, {'min_lr_ratio': .1}, {'lora_targets': 'attention'}, {'lora_targets': ['mlp']},
    {'distillation': {'alpha': 1.0, 'temperature': 2.0}}, {'distillation': {'alpha': -.1, 'temperature': 2.0}},
    {'distillation': {'alpha': .5, 'temperature': -1.0}}, {'distillation': {'alpha': .5, 'temperature': float('inf')}},
])
def test_invalid_recipes_rejected_at_config_and_training_time(bad):
    config = campaign.read_campaign_config()
    config['recovery'].update(max_steps=6, checkpoints=[3, 6])
    config['recovery'].update(bad)
    # A list-valued lora_targets raises TypeError (unhashable) rather than ValueError; still refused.
    with pytest.raises((ValueError, TypeError)):
        campaign.validate_config(config)
    model = tiny()
    before = state_hash(model)
    with pytest.raises((ValueError, TypeError)):
        recover_model(model, batches(), Rising(), base_cfg(**bad))
    assert state_hash(model) == before and not any(isinstance(m, LoRALinear) for m in model.modules())


def test_distillation_requires_unpruned_control():
    config = campaign.read_campaign_config()
    config['recovery']['distillation'] = {'alpha': .5, 'temperature': 2.0}
    config['include_controls'] = False
    with pytest.raises(ValueError, match='unpruned-recovery'):
        campaign.validate_config(config)


@pytest.mark.xfail(strict=True, reason='Doubtful (pre-existing): a checkpoint beyond max_steps is accepted and '
                   'silently never assessed, yet it is listed in the recorded recipe checkpoints.')
def test_checkpoint_beyond_max_steps_rejected():
    with pytest.raises(ValueError):
        recover_model(tiny(), batches(), Rising(), base_cfg(checkpoints=[3, 6, 9]))


# --- 2b: distillation -------------------------------------------------------------------------

def hand_kl(student, teacher, labels, temperature):
    terms = []
    for row in range(labels.shape[0]):
        for t in range(labels.shape[1] - 1):
            if int(labels[row, t + 1]) == -100:
                continue
            p = torch.softmax(teacher[row, t].double() / temperature, -1)
            logq = torch.log_softmax(student[row, t].double() / temperature, -1)
            terms.append(float((p * (p.log() - logq)).sum()))
    return sum(terms) / len(terms), len(terms)


def test_distillation_loss_hand_computation_masking_and_t_squared():
    g = torch.Generator().manual_seed(3)
    student, teacher = torch.randn(2, 7, 11, generator=g), torch.randn(2, 7, 11, generator=g)
    # Row 0: prompt 3, answer 4. Row 1: prompt 2, answer 2, then 3 padding positions.
    labels = torch.tensor([[-100, -100, -100, 4, 5, 6, 7], [-100, -100, 8, 9, -100, -100, -100]])
    ce = torch.tensor(1.7)
    for alpha, temperature in [(.5, 2.0), (0.0, 1.0), (.25, 3.5)]:
        loss, kl = distillation_loss(student, teacher, labels, ce, alpha, temperature)
        expected, n = hand_kl(student, teacher, labels, temperature)
        assert n == 4 + 2
        assert float(kl) == pytest.approx(expected, rel=1e-5)
        assert float(loss) == pytest.approx(alpha * 1.7 + (1 - alpha) * temperature ** 2 * expected, rel=1e-5)
    # Prompt and padding logits carry no weight.
    noisy_s, noisy_t = student.clone(), teacher.clone()
    for row in range(2):
        for t in range(6):
            if int(labels[row, t + 1]) == -100:
                noisy_s[row, t] += 50 * torch.randn(11, generator=g)
                noisy_t[row, t] -= 50 * torch.randn(11, generator=g)
    noisy_s[:, -1] += 99  # the last position predicts nothing
    a, _ = distillation_loss(student, teacher, labels, ce, .5, 2.0)
    b, _ = distillation_loss(noisy_s, noisy_t, labels, ce, .5, 2.0)
    assert float(a) == pytest.approx(float(b), rel=1e-6)
    # The gradient reaches only answer-token student logits.
    s = student.clone().requires_grad_(True)
    distillation_loss(s, teacher, labels, ce, .5, 2.0)[0].backward()
    assert torch.count_nonzero(s.grad.abs().sum(-1)) == 6
    assert s.grad[0, :2].abs().sum() == 0 and s.grad[1, 3:].abs().sum() == 0


def perturbed_teacher(model, seed=11, scale=.3):
    teacher = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in teacher.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


def test_distillation_training_matches_hand_loss_and_leaves_teacher_untouched():
    model = tiny()
    teacher = perturbed_teacher(model)
    teacher_hash, teacher_flags = state_hash(teacher), [p.requires_grad for p in teacher.parameters()]
    untouched = {k: v.clone() for k, v in model.state_dict().items()
                 if not any(k.endswith(f'mlp.{n}.weight') for n in ('gate_proj', 'up_proj', 'down_proj'))}
    data = batches(4)
    cfg = base_cfg(distillation={'alpha': .5, 'temperature': 2.0})
    # Hand loss of the very first update (student == base, adapters have B = 0).
    order = list(range(4))
    import random as _random
    _random.Random(cfg['seed']).shuffle(order)
    first = data[order[0]]
    with torch.no_grad():
        out = model(**first)
        teacher_logits = teacher(input_ids=first['input_ids'], attention_mask=first['attention_mask']).logits
    kl, _ = hand_kl(out.logits, teacher_logits, first['labels'], 2.0)
    result = recover_model(model, data, Rising(), cfg, teacher=teacher)
    log = result['distillation']
    assert log['alpha'] == .5 and log['temperature'] == 2.0 and len(log['train_kl_losses']) == 6
    assert log['train_ce_losses'][0] == pytest.approx(float(out.loss), rel=1e-5)
    assert log['train_kl_losses'][0] == pytest.approx(kl, rel=1e-4)
    for total, ce, kl in zip(result['train_losses'], log['train_ce_losses'], log['train_kl_losses']):
        assert total == pytest.approx(.5 * ce + .5 * 4 * kl, rel=1e-5)
    assert state_hash(teacher) == teacher_hash and not teacher.training
    assert all(p.grad is None for p in teacher.parameters())
    assert [p.requires_grad for p in teacher.parameters()] == teacher_flags
    assert not any(isinstance(m, LoRALinear) for m in teacher.modules())
    # Only adapter tensors were trained: everything but merged MLP weights is bit-identical.
    after = model.state_dict()
    assert all(torch.equal(after[k], v) for k, v in untouched.items())
    assert all(p.grad is None for p in model.parameters())


def test_identical_teacher_gives_zero_initial_kl():
    model = tiny()
    teacher = copy.deepcopy(model).requires_grad_(False).eval()
    result = recover_model(model, batches(), Rising(), base_cfg(distillation={'alpha': .5, 'temperature': 2.0}),
                           teacher=teacher)
    assert result['distillation']['train_kl_losses'][0] == pytest.approx(0, abs=1e-6)
    assert result['distillation']['train_kl_losses'][-1] > 1e-6


def test_alpha_near_one_approaches_ce_training():
    plain, distilled = tiny(), tiny()
    recover_model(plain, batches(), Rising(), base_cfg())
    teacher = perturbed_teacher(distilled)
    recover_model(distilled, batches(), Rising(), base_cfg(distillation={'alpha': 1 - 1e-7, 'temperature': 2.0}),
                  teacher=teacher)
    for (k, a), b in zip(plain.state_dict().items(), distilled.state_dict().values()):
        assert torch.allclose(a, b, atol=1e-5), k
    far = tiny()
    recover_model(far, batches(), Rising(), base_cfg(distillation={'alpha': 0.0, 'temperature': 2.0}),
                  teacher=perturbed_teacher(far))
    assert state_hash(far) != state_hash(plain)


def test_teacher_must_be_frozen_and_need_settings():
    model = tiny()
    trainable = copy.deepcopy(model)
    with pytest.raises(ValueError, match='frozen'):
        recover_model(model, batches(), Rising(), base_cfg(distillation={'alpha': .5, 'temperature': 2.0}), teacher=trainable)
    with pytest.raises(ValueError, match='requires recovery.distillation'):
        recover_model(model, batches(), Rising(), base_cfg(), teacher=trainable.requires_grad_(False))


def test_pruned_student_of_varied_widths_distils_from_full_teacher():
    from prune_lab.pruning import compact_model
    teacher = tiny().requires_grad_(False).eval()
    student = tiny()
    compact_model(student, {0: list(range(0, 24, 2)), 1: list(range(20))})
    result = recover_model(student, batches(), Rising(), base_cfg(distillation={'alpha': .5, 'temperature': 2.0},
                                                                   lora_targets='mlp+attention'), teacher=teacher)
    assert [layer.mlp.down_proj.in_features for layer in student.model.layers] == [12, 20]
    assert all(math.isfinite(x) for x in result['distillation']['train_kl_losses'])


# --- 2c: attention LoRA ----------------------------------------------------------------------

def adapted_modules(monkeypatch, model):
    names = {id(m): n for n, m in model.named_modules()}
    seen = []
    original = LoRALinear.__init__

    def init(self, base, rank=8, alpha=16):
        seen.append(names[id(base)])
        original(self, base, rank, alpha)
    monkeypatch.setattr(LoRALinear, '__init__', init)
    return seen


def test_attention_targets_exact_merge_exact_biases_untouched(monkeypatch):
    model = tiny()
    seen = adapted_modules(monkeypatch, model)
    biases = {k: v.clone() for k, v in model.state_dict().items() if k.endswith('.bias')}
    assert {k.split('.')[-2] for k in biases} == {'q_proj', 'k_proj', 'v_proj'}
    before = clone_state(model)
    probe = torch.randint(3, V, (1, 9), generator=torch.Generator().manual_seed(4))
    dev = Rising(probe)
    result = recover_model(model, batches(), dev, base_cfg(lora_targets='mlp+attention'))
    expected = [f'model.layers.{i}.{part}' for i in range(2)
                for part in ('mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')] + \
               [f'model.layers.{i}.self_attn.{p}' for i in range(2) for p in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
    assert seen == expected  # MLP adapters first, so MLP init matches the MLP-only recipe
    r = 2
    hand = 2 * r * ((H + I) + (H + I) + (I + H) + (H + H) + (H + 8) + (H + 8) + (H + H))
    assert result['trainable_parameters'] == hand
    assert result['recipe']['lora_targets'] == 'mlp+attention'
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    after = model.state_dict()
    for k, v in biases.items():
        assert torch.equal(after[k], v)
    changed = {k for k in after if not torch.equal(after[k], before[k])}
    assert changed == {f'{n}.weight' for n in expected}
    with torch.no_grad():
        merged = model(input_ids=probe, use_cache=False).logits
    assert torch.allclose(merged, dev.logits[-1], atol=1e-5)  # last assessed = merged state


def test_attention_and_mlp_init_matches_mlp_only_for_mlp_adapters(monkeypatch):
    """Adding attention adapters must not change the MLP adapters' initial A matrices."""
    captured = {}
    original = LoRALinear.__init__

    def init(self, base, rank=8, alpha=16):
        original(self, base, rank, alpha)
        captured.setdefault(captured.get('_mode'), []).append(self.A.detach().clone())
    monkeypatch.setattr(LoRALinear, '__init__', init)
    for mode in ('mlp', 'mlp+attention'):
        captured['_mode'] = mode
        recover_model(tiny(), batches(), Rising(), base_cfg(lora_targets=mode, max_steps=1, checkpoints=[1]))
    mlp_only, both = captured['mlp'], captured['mlp+attention']
    assert len(mlp_only) == 6 and len(both) == 14
    assert all(torch.equal(a, b) for a, b in zip(mlp_only, both[:6]))


def test_export_reload_agrees_with_attention_lora(tmp_path):
    model = tiny()
    recover_model(model, batches(), Rising(), base_cfg(lora_targets='mlp+attention'))
    model.save_pretrained(tmp_path / 'export', safe_serialization=True)
    reloaded = Qwen2ForCausalLM.from_pretrained(tmp_path / 'export', torch_dtype=torch.float32)
    probe = torch.randint(3, V, (1, 8), generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        assert torch.equal(model(input_ids=probe).logits, reloaded(input_ids=probe).logits)


class Boom(Exception):
    pass


@pytest.mark.parametrize('with_teacher', [False, True])
@pytest.mark.parametrize('failure', ['callback', 'nan_adapter'])
def test_failure_mid_training_removes_all_adapters_and_restores_model(failure, with_teacher):
    model = tiny()
    model.train()
    model.lm_head.requires_grad_(False)
    flags = [(n, p.requires_grad) for n, p in model.named_parameters()]
    modes = [(n, m.training) for n, m in model.named_modules()]
    before = state_hash(model)
    teacher = perturbed_teacher(model) if with_teacher else None
    teacher_hash = state_hash(teacher) if teacher is not None else None
    calls = []

    def dev(m):
        calls.append(1)
        if len(calls) == 2:
            if failure == 'callback':
                raise Boom
            for module in m.modules():
                if isinstance(module, LoRALinear) and module.base.out_features == 8:  # an attention k/v adapter
                    with torch.no_grad():
                        module.A.fill_(float('nan'))
        return {'selection_score': .1 * len(calls), 'score_name': 'token_f1', 'answer_token_loss': 1.0}
    cfg = base_cfg(lora_targets='mlp+attention', max_steps=8, checkpoints=[2, 4, 8])
    if with_teacher:
        cfg['distillation'] = {'alpha': .5, 'temperature': 2.0}
    expected = Boom if failure == 'callback' else (FloatingPointError, RuntimeError)
    with pytest.raises(expected):
        recover_model(model, batches(), dev, cfg, **({'teacher': teacher} if teacher is not None else {}))
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    assert state_hash(model) == before
    assert [(n, p.requires_grad) for n, p in model.named_parameters()] == flags
    assert [(n, m.training) for n, m in model.named_modules()] == modes
    assert all(p.grad is None for p in model.parameters())
    for module in model.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks
    if teacher is not None:
        assert state_hash(teacher) == teacher_hash and not teacher.training
        assert all(p.grad is None for p in teacher.parameters())
        assert not any(isinstance(m, LoRALinear) for m in teacher.modules())


def test_nonfinite_loss_raises_with_teacher_and_attention():
    model = tiny()
    teacher = perturbed_teacher(model)
    with torch.no_grad():
        teacher.lm_head.weight.fill_(float('nan'))  # NaN teacher logits -> NaN KL
    before = state_hash(model)
    with pytest.raises(FloatingPointError, match='Nonfinite recovery loss'):
        recover_model(model, batches(), Rising(), base_cfg(lora_targets='mlp+attention',
                                                            distillation={'alpha': .5, 'temperature': 2.0}),
                      teacher=teacher)
    assert state_hash(model) == before and not any(isinstance(m, LoRALinear) for m in model.modules())


def test_attention_lora_rejects_non_linear_projection_before_attaching():
    model = tiny()
    model.model.layers[1].self_attn.o_proj = torch.nn.Identity()
    before = clone_state(model)
    with pytest.raises(ValueError, match='Attention LoRA'):
        recover_model(model, batches(), Rising(), base_cfg(lora_targets='mlp+attention'))
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    after = model.state_dict()
    assert all(torch.equal(after[k], v) for k, v in before.items())
