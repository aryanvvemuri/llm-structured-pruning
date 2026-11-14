"""Opt-in recovery recipes: 2a learning-rate schedule (and default-path identity)."""
import math

import pytest
import torch

from prune_lab import campaign
from prune_lab.recipes import lr_factor, validate_recipe, validate_schedule
from prune_lab.recovery import recover_model
from test_recovery import TinyModel, batch


def test_cosine_schedule_values():
    f = lambda u: lr_factor(u, 450, 'cosine', 20, .1)
    assert f(0) == pytest.approx(1 / 20) and f(9) == pytest.approx(.5) and f(19) == pytest.approx(1)
    assert f(20) == pytest.approx(1) and f(235) == pytest.approx(.55)
    assert f(449) == pytest.approx(.1 + .9 * .5 * (1 + math.cos(math.pi * 429 / 430)))
    assert f(450) == pytest.approx(.1) and all(f(u) >= f(u + 1) for u in range(20, 449))
    assert lr_factor(7, 10) == 1.0


@pytest.mark.parametrize('bad', [{'lr_schedule': 'linear'}, {'lr_schedule': 'cosine', 'warmup_steps': 50, 'max_steps': 50},
                                 {'lr_schedule': 'cosine', 'warmup_steps': -1}, {'lr_schedule': 'cosine', 'min_lr_ratio': 1.5},
                                 {'warmup_steps': 5}, {'lr_schedule': 'cosine', 'warmup_steps': 2.0}])
def test_schedule_validation(bad):
    with pytest.raises(ValueError):
        validate_schedule(dict({'max_steps': 450}, **bad))
    config = campaign.read_campaign_config()
    config['recovery'].update(bad)
    with pytest.raises(ValueError):
        campaign.validate_config(config)


def run(config, seed=3):
    torch.manual_seed(seed)
    model = TinyModel()
    batches = [batch(a) for a in (2, 3, 4, 5, 6, 7)]
    calls = iter(range(100))
    evaluate = lambda m: {'accuracy': .1 * next(calls), 'loss': 1.0}  # Later checkpoints always win.
    result = recover_model(model, batches, evaluate, config)
    return model, result


def test_constant_default_is_unchanged_and_cosine_sets_rates():
    base = {'rank': 2, 'alpha': 4, 'max_steps': 6, 'gradient_accumulation': 2, 'checkpoints': [3, 6], 'learning_rate': .01}
    model_a, a = run(base)
    model_b, b = run(dict(base, lr_schedule='constant'))
    for x, y in zip(model_a.state_dict().values(), model_b.state_dict().values()):
        assert torch.equal(x, y)
    assert a['train_losses'] == b['train_losses'] and a['learning_rates'] == [.01] * 6
    assert a['recipe']['lr_schedule'] == 'constant' and a['training_seconds'] >= 0
    model_c, c = run(dict(base, lr_schedule='cosine', warmup_steps=2, min_lr_ratio=.1))
    assert c['learning_rates'] == pytest.approx([.01 * lr_factor(u, 6, 'cosine', 2, .1) for u in range(6)])
    assert [h['learning_rate'] for h in c['history']] == pytest.approx([None, c['learning_rates'][2], c['learning_rates'][5]])
    assert c['recipe'].items() >= {'lr_schedule': 'cosine', 'peak_learning_rate': .01, 'warmup_steps': 2, 'min_lr_ratio': .1,
                                   'max_steps': 6, 'checkpoints': [3, 6], 'gradient_accumulation': 2, 'rank': 2, 'alpha': 4}.items()
    assert any(not torch.equal(x, y) for x, y in zip(model_a.state_dict().values(), model_c.state_dict().values()))


def test_longer_config_and_defaults():
    validate_recipe({})
    longer = campaign.read_campaign_config('configs/support-v2-longer.json')
    base = campaign.read_campaign_config('configs/support-v2.json')
    campaign.validate_config(longer)
    assert longer['recovery'] == dict(base['recovery'], max_steps=450, checkpoints=[150, 300, 450],
                                      lr_schedule='cosine', warmup_steps=20, min_lr_ratio=.1)
    assert {k: v for k, v in longer.items() if k not in ('recovery', 'store_example_losses')} == \
        {k: v for k, v in base.items() if k != 'recovery'}
