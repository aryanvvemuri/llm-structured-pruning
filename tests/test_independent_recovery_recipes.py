"""Independent recover_model-level checks of improvement 2a (longer recovery: cosine schedule).

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
from prune_lab.recovery import LoRALinear, recover_model

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


@pytest.mark.parametrize('extra', [{}, {'lr_schedule': 'constant'}])
def test_default_paths_bit_identical_to_pre_change_recovery(tmp_path, extra):
    old = load_old_recovery(tmp_path)
    reference, model = tiny(), tiny()
    cfg = base_cfg(max_steps=7, checkpoints=[2, 5], gradient_accumulation=2)
    before = old.recover_model(reference, batches(5), Rising(), dict(cfg))
    after = recover_model(model, batches(5), Rising(), dict(cfg, **extra))
    assert state_hash(model) == state_hash(reference)
    assert after['train_losses'] == before['train_losses']
    assert after['best_step'] == before['best_step'] and after['optimizer_steps'] == before['optimizer_steps']
    assert after['learning_rates'] == [cfg['learning_rate']] * 7
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
    {'warmup_steps': 2}, {'min_lr_ratio': .1},
])
def test_invalid_recipes_rejected_at_config_and_training_time(bad):
    config = campaign.read_campaign_config()
    config['recovery'].update(max_steps=6, checkpoints=[3, 6])
    config['recovery'].update(bad)
    with pytest.raises((ValueError, TypeError)):
        campaign.validate_config(config)
    model = tiny()
    before = state_hash(model)
    with pytest.raises((ValueError, TypeError)):
        recover_model(model, batches(), Rising(), base_cfg(**bad))
    assert state_hash(model) == before and not any(isinstance(m, LoRALinear) for m in model.modules())


def test_checkpoint_beyond_max_steps_rejected():
    with pytest.raises(ValueError):
        recover_model(tiny(), batches(), Rising(), base_cfg(checkpoints=[3, 6, 9]))
