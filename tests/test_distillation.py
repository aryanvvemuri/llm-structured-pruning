"""2b: distillation from the same seed's recovered full model (opt-in)."""
import json
import math

import pytest
import torch

from prune_lab import campaign
from prune_lab.recipes import validate_distillation
from prune_lab.recovery import distillation_loss, recover_model
from test_multiseed import tiny_support_config
from test_recovery import TinyModel, batch


def test_distillation_loss_matches_hand_computation():
    torch.manual_seed(0)
    student, teacher = torch.randn(1, 4, 5), torch.randn(1, 4, 5)
    labels = torch.tensor([[-100, -100, 3, 1]])
    ce = torch.tensor(2.5)
    loss, kl = distillation_loss(student, teacher, labels, ce, .5, 2.0)
    total = 0.0
    for position in (1, 2):  # Logits at t predict labels at t+1.
        p = [math.exp(x / 2) for x in teacher[0, position].tolist()]
        q = [math.exp(x / 2) for x in student[0, position].tolist()]
        p, q = [x / sum(p) for x in p], [x / sum(q) for x in q]
        total += sum(a * math.log(a / b) for a, b in zip(p, q))
    assert float(kl) == pytest.approx(total / 2, rel=1e-5)
    assert float(loss) == pytest.approx(.5 * 2.5 + .5 * 4 * total / 2, rel=1e-5)
    same, zero = distillation_loss(student, student, labels, ce, .5, 2.0)
    assert float(zero) == pytest.approx(0, abs=1e-6) and float(same) == pytest.approx(1.25)


@pytest.mark.parametrize('bad', [{'alpha': 1.0, 'temperature': 2}, {'alpha': .5, 'temperature': 0}, {'alpha': .5},
                                 {'alpha': .5, 'temperature': 2, 'extra': 1}, [.5, 2], {'alpha': True, 'temperature': 2}])
def test_distillation_validation(bad):
    with pytest.raises(ValueError):
        validate_distillation({'distillation': bad})
    assert validate_distillation({}) is None


class TinyTeacher(TinyModel):
    def forward(self, input_ids, labels=None, attention_mask=None, use_cache=None):
        x = self.embedding(input_ids)
        for layer in self.model.layers:
            m = layer.mlp
            x = x + m.down_proj(torch.sigmoid(m.gate_proj(x)) * m.up_proj(x))
        return type('Out', (), {'logits': self.head(x)})()


def test_teacher_frozen_unchanged_and_required_settings():
    torch.manual_seed(2)
    student, teacher = TinyModel(), TinyTeacher()
    teacher.requires_grad_(False)
    before = {k: v.clone() for k, v in teacher.state_dict().items()}
    config = {'rank': 2, 'alpha': 4, 'max_steps': 3, 'gradient_accumulation': 1, 'checkpoints': [3],
              'distillation': {'alpha': .5, 'temperature': 2.0}}
    result = recover_model(student, [batch(2), batch(3)], lambda m: {'accuracy': .5, 'loss': 1.0}, config, teacher=teacher)
    assert all(torch.equal(before[k], v) for k, v in teacher.state_dict().items()) and not teacher.training
    assert len(result['distillation']['train_kl_losses']) == 3 and result['distillation']['alpha'] == .5
    plain = {k: v for k, v in config.items() if k != 'distillation'}
    with pytest.raises(ValueError, match='requires recovery.distillation'):
        recover_model(TinyModel(), [batch()], lambda m: {'accuracy': .5, 'loss': 1.0}, plain, teacher=teacher)
    teacher.requires_grad_(True)
    with pytest.raises(ValueError, match='frozen'):
        recover_model(TinyModel(), [batch()], lambda m: {'accuracy': .5, 'loss': 1.0}, config, teacher=teacher)
    assert 'distillation' not in recover_model(TinyModel(), [batch()], lambda m: {'accuracy': .5, 'loss': 1.0}, config)


def test_campaign_trains_teacher_first_records_identity_and_resumes(tmp_path, monkeypatch):
    config = tiny_support_config(tmp_path, monkeypatch)
    config['recovery']['distillation'] = {'alpha': .5, 'temperature': 2.0}
    order = []
    original = campaign._load_teacher
    monkeypatch.setattr(campaign, '_load_teacher', lambda c, cands: order.append([x['id'] for x in cands]) or original(c, cands))
    output = tmp_path / 'runs' / 'distill'
    campaign.run_campaign(config, output, 'preflight')
    campaign.run_campaign(config, output, 'fit')
    selection = campaign._selection(output)
    ids = [c['id'] for c in selection['candidates']]
    assert ids == ['per-layer-01', 'unpruned-recovery', 'uniform-10']  # Allocation order is kept in the record.
    assert order == [['unpruned-recovery'], ['unpruned-recovery', 'per-layer-01']]
    records = {c['id']: c for c in selection['candidates']}
    assert 'distillation' not in records['unpruned-recovery'] and 'distillation' not in records['unpruned-recovery']['recovery']
    for name in ('per-layer-01', 'uniform-10'):
        d = records[name]['distillation']
        assert d['teacher_checkpoint_sha256'] == records['unpruned-recovery']['checkpoint_sha256']
        assert d['alpha'] == .5 and d['temperature'] == 2.0 and records[name]['recovery']['distillation']['temperature'] == 2.0
    # Resume in the middle of the seed: the teacher is reloaded from its export and hash-checked.
    def interrupt():
        for name in ('selection.json', 'selection-lock.json'):
            (output / name).unlink()
        (output / 'candidates' / 'uniform-10' / 'result.json').unlink()
    interrupt()
    campaign.run_campaign(config, output, 'fit')
    assert order[-1] == ['unpruned-recovery', 'per-layer-01']
    resumed = {c['id']: c for c in campaign._selection(output)['candidates']}
    assert resumed['uniform-10']['distillation']['teacher_checkpoint_sha256'] == records['unpruned-recovery']['checkpoint_sha256']
    interrupt()
    weights = next((output / 'candidates' / 'unpruned-recovery' / 'checkpoint').glob('*.safetensors'))
    weights.write_bytes(weights.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='Saved checkpoint changed'):
        campaign.run_campaign(config, output, 'fit')


def test_distill_config_is_longer_plus_distillation():
    longer = campaign.read_campaign_config('configs/support-v2-longer.json')
    distill = campaign.read_campaign_config('configs/support-v2-distill.json')
    campaign.validate_config(distill)
    longer['recovery']['distillation'] = {'alpha': 0.5, 'temperature': 2.0}
    assert distill == longer


def test_multi_seed_distillation_with_discard(tmp_path, monkeypatch):
    from prune_lab import multiseed
    config = tiny_support_config(tmp_path, monkeypatch)
    config['recovery']['distillation'] = {'alpha': .5, 'temperature': 2.0}
    root = tmp_path / 'runs' / 'ms'
    multiseed.run_seeds(config, root, [5, 6], discard_checkpoints=True)
    assert not list(root.rglob('*.safetensors'))
    for seed in (5, 6):
        records = {c['id']: c for c in campaign._selection(root / f'seed-{seed}')['candidates']}
        assert records['uniform-10']['distillation']['teacher_checkpoint_sha256'] == records['unpruned-recovery']['checkpoint_sha256']
    assert (root / 'seed-5' / 'allocation.json').read_bytes() == (root / 'seed-6' / 'allocation.json').read_bytes()
    assert multiseed.aggregate_run(root, samples=10)['recovery_recipe']['distillation'] == {'alpha': .5, 'temperature': 2.0}
