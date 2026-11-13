import copy

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from prune_lab.allocation import allocate_neurons, measure_sensitivity
from prune_lab.modeling import count_parameters
from prune_lab.pruning import compact_model


@pytest.fixture
def model():
    torch.manual_seed(7)
    return Qwen2ForCausalLM(Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=32, attention_dropout=0.0)).eval().requires_grad_(False)


@pytest.fixture
def batch():
    return {"input_ids": torch.tensor([[1, 2, 3, 4, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
            "labels": torch.tensor([[-100, -100, 3, 4, -100]])}


def test_total_parameter_budget_and_sensitivity(model):
    scores = {i: torch.arange(24).float() for i in range(2)}
    before = count_parameters(model)
    keep = allocate_neurons(model, scores, {0: 0.01, 1: 1.0}, 0.15, group_size=4)
    assert len(keep[0]) < len(keep[1])
    assert all(len(v) >= 6 and len(v) % 4 == 0 for v in keep.values())
    assert all(v == list(range(24 - len(v), 24)) for v in keep.values())
    compact_model(model, keep)
    assert abs(before - count_parameters(model) - before * 0.15) <= 4 * 48


def test_layer_score_scale_invariant_and_uniform(model):
    scores = {i: torch.arange(24).float() for i in range(2)}
    sensitivity = {0: 0.1, 1: 0.2}
    keep = allocate_neurons(model, scores, sensitivity, 0.10, group_size=4)
    scores[0] *= 100000
    assert keep == allocate_neurons(model, scores, sensitivity, 0.10, group_size=4)
    uniform = allocate_neurons(model, scores, sensitivity, 0.10, group_size=4, uniform=True)
    assert len(uniform[0]) == len(uniform[1])
    assert allocate_neurons(model, scores, sensitivity, 0.0, group_size=4) == {i: list(range(24)) for i in range(2)}
    with pytest.raises(ValueError, match="Infeasible"):
        allocate_neurons(model, scores, sensitivity, 0.99, group_size=4)


def test_sensitivity_restores_and_weights_answer_tokens(model, batch):
    original = copy.deepcopy(model.state_dict())
    model.train()
    scores = {i: torch.arange(24).float() for i in range(2)}
    values = measure_sensitivity(model, [batch], scores=scores)
    repeated = measure_sensitivity(model, [batch, batch], scores=scores)
    assert values == pytest.approx(repeated)
    assert all(torch.isfinite(torch.tensor(v)) for v in values.values())
    assert model.training and model.config.use_cache
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    assert all(not layer.mlp.down_proj._forward_pre_hooks for layer in model.model.layers)
    batch["labels"].fill_(-100)
    with pytest.raises(ValueError, match="supervised"):
        measure_sensitivity(model, [batch], scores=scores)
    assert model.training and model.config.use_cache
