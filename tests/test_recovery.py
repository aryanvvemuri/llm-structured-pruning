"""Small local tensors only; no model downloads."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from prune_lab.recovery import LoRALinear, recover_model


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type='qwen2', use_cache=True)
        self.embedding = nn.Embedding(8, 4)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        for width in (7, 5):
            layer = nn.Module()
            layer.mlp = nn.Module()
            layer.mlp.gate_proj = nn.Linear(4, width)
            layer.mlp.up_proj = nn.Linear(4, width)
            layer.mlp.down_proj = nn.Linear(width, 4)
            self.model.layers.append(layer)
        self.head = nn.Linear(4, 8)

    def forward(self, input_ids, labels, attention_mask=None):
        x = self.embedding(input_ids)
        for layer in self.model.layers:
            m = layer.mlp
            x = x + m.down_proj(torch.sigmoid(m.gate_proj(x)) * m.up_proj(x))
        logits = self.head(x)
        loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 8), labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss, logits=logits)


def batch(answer=2):
    return {'input_ids': torch.tensor([[1, 2, answer]]), 'labels': torch.tensor([[-100, -100, answer]])}


def test_zero_init_gradient_and_merge_variable_width():
    for width in (3, 7):
        base = nn.Linear(4, width)
        x = torch.randn(2, 4)
        expected = base(x).detach()
        adapter = LoRALinear(base, rank=2, alpha=4)
        torch.testing.assert_close(adapter(x), expected, rtol=0, atol=0)
        adapter(x).sum().backward()
        assert adapter.A.grad is not None and adapter.B.grad is not None
        assert all(p.grad is None and not p.requires_grad for p in base.parameters())
        with torch.no_grad():
            adapter.B.add_(0.01)
        expected = adapter(x).detach()
        merged = adapter.merge()
        assert merged.weight.shape == (width, 4)
        torch.testing.assert_close(merged(x), expected)


def test_recovery_selects_best_and_restores_callback_flags():
    torch.manual_seed(1)
    model = TinyModel().eval()
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    values = iter([(0.3, 3), (0.5, 2), (0.5, 1), (0.4, 0.5)])
    captured = []
    def evaluate(m):
        captured.append(m(**batch()).logits.detach().clone())
        m.eval().requires_grad_(False)
        accuracy, loss = next(values)
        return {'accuracy': accuracy, 'answer_token_loss': loss}
    result = recover_model(model, [batch(), batch(3), batch(4)], evaluate,
                           {'max_steps': 3, 'gradient_accumulation': 2, 'checkpoints': [1, 2, 3], 'rank': 2})
    assert result['optimizer_steps'] == 3 and result['best_step'] == 2
    assert len(result['train_losses']) == 3
    torch.testing.assert_close(model(**batch()).logits, captured[2], atol=1e-6, rtol=1e-5)
    assert not model.training and model.config.use_cache
    assert all(p.requires_grad for p in model.parameters())
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    for name in ('embedding.weight', 'head.weight', 'head.bias'):
        torch.testing.assert_close(model.state_dict()[name], original[name], atol=0, rtol=0)


def test_initial_best_keeps_original_weights():
    model = TinyModel()
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    scores = iter([1.0, 0.0])
    result = recover_model(model, [batch()], lambda m: {'accuracy': next(scores), 'loss': 1},
                           {'max_steps': 1, 'rank': 2})
    assert result['best_step'] == 0
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, original[key], rtol=0, atol=0)


def test_partial_accumulation_matches_single_batch_update():
    model = TinyModel()
    other = TinyModel()
    other.load_state_dict(model.state_dict())
    for m, accumulation in [(model, 1), (other, 4)]:
        scores = iter([0.0, 1.0])
        recover_model(m, [batch()], lambda m: {'accuracy': next(scores), 'loss': 1},
                      {'max_steps': 1, 'rank': 2, 'gradient_accumulation': accumulation})
    for a, b in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_failure_removes_adapters_restores_state():
    model = TinyModel().eval()
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    def fail(m):
        raise RuntimeError('evaluation failed')
    with pytest.raises(RuntimeError, match='evaluation failed'):
        recover_model(model, [batch()], fail, {'max_steps': 1})
    assert model.config.use_cache and not model.training
    assert all(p.requires_grad for p in model.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_failure_after_optimizer_update_keeps_base_and_frozen_flags():
    model = TinyModel().eval().requires_grad_(False)
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    calls = 0

    def fail_after_update(m):
        nonlocal calls
        calls += 1
        if calls == 2:
            assert any(bool(module.B.abs().sum()) for module in m.modules()
                       if isinstance(module, LoRALinear))
            raise RuntimeError('post-update development failure')
        return {'accuracy': 0, 'loss': 1}

    with pytest.raises(RuntimeError, match='post-update'):
        recover_model(model, [batch()], fail_after_update,
                      {'max_steps': 1, 'rank': 2})
    assert calls == 2 and not model.training and model.config.use_cache
    assert not any(isinstance(module, LoRALinear) for module in model.modules())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_nonfinite_training_loss_does_not_modify_base():
    model = TinyModel()
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    forward = model.forward
    def nonfinite(**kwargs):
        output = forward(**kwargs)
        output.loss = output.loss * float('nan')
        return output
    model.forward = nonfinite
    with pytest.raises(FloatingPointError, match='Nonfinite recovery loss'):
        recover_model(model, [batch()], lambda m: {'accuracy': 0, 'loss': 1}, {'max_steps': 1})
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_checkpoint_input_hook_and_cleanup():
    model = TinyModel()
    model.is_gradient_checkpointing = False
    def enable(**kwargs):
        assert kwargs['gradient_checkpointing_kwargs'] == {'use_reentrant': False}
        model.is_gradient_checkpointing = True
    def disable():
        model.is_gradient_checkpointing = False
    model.gradient_checkpointing_enable = enable
    model.gradient_checkpointing_disable = disable
    model.get_input_embeddings = lambda: model.embedding
    scores = iter([0.0, 1.0])
    def evaluate(m):
        assert m.is_gradient_checkpointing
        assert len(m.embedding._forward_hooks) == 1
        return {'accuracy': next(scores), 'loss': 1}
    recover_model(model, [batch()], evaluate, {'max_steps': 1, 'gradient_checkpointing': True})
    assert not model.is_gradient_checkpointing
    assert not model.embedding._forward_hooks


@pytest.mark.parametrize('labels', [
    [[1, -100, 2]],  # Prompt is not masked.
    [[-100, -100, 7]],  # Target differs from the input answer.
    [[-100, 2, -100, 3]],  # Supervision is not one answer suffix.
])
def test_rejects_invalid_answer_only_batches(labels):
    bad = {'input_ids': torch.tensor([[1, 2, 2, 3][:len(labels[0])]]),
           'labels': torch.tensor(labels)}
    with pytest.raises(ValueError):
        recover_model(TinyModel(), [bad], lambda m: {'accuracy': 0, 'loss': 1}, {'max_steps': 1})
