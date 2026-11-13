import copy

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from prune_lab.modeling import CompactQwen2ForCausalLM, count_parameters
from prune_lab.pruning import compact_model, masked_neurons
from prune_lab.scoring import score_neurons

torch.set_num_threads(2)


@pytest.fixture
def model():
    torch.manual_seed(7)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=24,
                         num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=1, max_position_embeddings=32,
                         attention_dropout=0.0)
    return Qwen2ForCausalLM(config).eval().requires_grad_(False)


@pytest.fixture
def batch():
    return {"input_ids": torch.tensor([[1, 2, 3, 4, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
            "labels": torch.tensor([[-100, -100, 3, 4, -100]])}


def test_compact_matches_mask_and_reload(model, batch, tmp_path):
    keep = {0: [0, 2, 4, 8, 11], 1: [1, 3, 7, 10, 18]}
    before = count_parameters(model)
    with torch.no_grad():
        baseline = model(**batch).logits
        with masked_neurons(model, keep):
            masked = model(**batch).logits
        torch.testing.assert_close(model(**batch).logits, baseline)
        compact_model(model, keep)
        torch.testing.assert_close(model(**batch).logits, masked, atol=1e-6, rtol=1e-5)
        assert count_parameters(model) < before
        model.save_pretrained(tmp_path, safe_serialization=True)
        reloaded = Qwen2ForCausalLM.from_pretrained(tmp_path).eval()
        torch.testing.assert_close(reloaded(**batch).logits, masked, atol=1e-6, rtol=1e-5)
        with pytest.raises(ValueError, match="already been compacted"):
            compact_model(reloaded, keep)


def test_mask_exception_restores(model, batch):
    baseline = model(**batch).logits.detach()
    with pytest.raises(RuntimeError):
        with masked_neurons(model, {0: [1], 1: [2]}):
            raise RuntimeError("test failure")
    torch.testing.assert_close(model(**batch).logits, baseline)
    assert not model.model.layers[0].mlp.down_proj._forward_pre_hooks


@pytest.mark.parametrize("method", ["activation", "contribution", "taylor"])
def test_scores_stream_and_restore(model, batch, method):
    original = copy.deepcopy(model.state_dict())
    model.train()
    scores = score_neurons(model, [batch], method)
    repeated = score_neurons(model, [batch, batch], method)
    assert model.training and model.config.use_cache
    for index in scores:
        assert scores[index].shape == (24,)
        assert scores[index].device.type == "cpu"
        assert torch.isfinite(scores[index]).all() and (scores[index] >= 0).all()
        assert scores[index].sum() > 0
        torch.testing.assert_close(scores[index], repeated[index])
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert not model.get_input_embeddings()._forward_hooks


def test_taylor_under_mask(model, batch):
    keep = {0: [0, 2, 4], 1: [1, 3, 5]}
    with masked_neurons(model, keep):
        scores = score_neurons(model, [batch], "taylor")
    for index, values in scores.items():
        dropped = [i for i in range(24) if i not in keep[index]]
        assert torch.count_nonzero(values[dropped]) == 0


def test_taylor_failure_restores(model, batch):
    batch["labels"].fill_(-100)
    model.train()
    with pytest.raises(ValueError, match="supervised token"):
        score_neurons(model, [batch], "taylor")
    assert model.training and model.config.use_cache
    assert not model.get_input_embeddings()._forward_hooks
    assert all(not mlp.down_proj._forward_pre_hooks for mlp in
               [layer.mlp for layer in model.model.layers])


@pytest.mark.parametrize("keep", [{0: [1]}, {0: [], 1: []},
                                 {0: [1, 1], 1: [1, 2]},
                                 {0: [24], 1: [1]},
                                 {0: [True], 1: [1, 2]}])
def test_invalid_keep_sets(model, keep):
    with pytest.raises(ValueError):
        compact_model(model, keep)
    assert model.config.intermediate_size == 24


def test_heterogeneous_compaction_roundtrip(model, batch, tmp_path):
    keep = {0: [0, 3, 7], 1: [1, 4, 8, 10, 17, 20, 23]}
    with torch.no_grad(), masked_neurons(model, keep):
        expected = model(**batch).logits
    compact_model(model, keep)
    assert model.config.prune_lab_layer_widths == [3, 7]
    assert model.config.prune_lab_original_widths == [24, 24]
    assert model.config.prune_lab_keep_indices == [keep[0], keep[1]]
    model.save_pretrained(tmp_path, safe_serialization=True)
    loaded = CompactQwen2ForCausalLM.from_pretrained(
        tmp_path, use_safetensors=True, low_cpu_mem_usage=True).eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(**batch).logits, expected, atol=1e-6, rtol=1e-5)
    assert count_parameters(loaded) == count_parameters(model)
    for layer, width in zip(loaded.model.layers, [3, 7]):
        assert layer.mlp.gate_proj.out_features == width
        assert layer.mlp.up_proj.out_features == width
        assert layer.mlp.down_proj.in_features == width
    with pytest.raises(ValueError, match="already been compacted"):
        compact_model(loaded, keep)


def test_bad_heterogeneous_metadata(model):
    model.config.prune_lab_layer_widths = [3, 0]
    with pytest.raises(ValueError, match="per-layer"):
        CompactQwen2ForCausalLM(model.config)
