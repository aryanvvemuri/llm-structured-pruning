"""2c: LoRA on attention projections as well as the MLP (opt-in)."""
import pytest
import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from prune_lab import campaign
from prune_lab.recipes import validate_lora_targets
from prune_lab.recovery import LoRALinear, recover_model

NAMES = ('q_proj', 'k_proj', 'v_proj', 'o_proj')


def tiny():
    torch.manual_seed(11)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=12, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32))


def batches():
    return [{'input_ids': torch.tensor([[1, 2, 3, a]]), 'labels': torch.tensor([[-100, -100, -100, a]]),
             'attention_mask': torch.ones(1, 4, dtype=torch.long)} for a in (4, 5, 6, 7)]


CONFIG = {'rank': 2, 'alpha': 4, 'max_steps': 3, 'gradient_accumulation': 1, 'checkpoints': [3], 'learning_rate': .05}
PROBE = torch.tensor([[3, 1, 4, 1, 5]])


def test_attention_adapters_train_merge_exactly_and_keep_biases():
    model = tiny()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    seen = {}

    def evaluate(m):
        step = len(seen)
        attention = m.model.layers[0].self_attn
        seen[step] = {'wrapped': all(isinstance(getattr(attention, n), LoRALinear) for n in NAMES),
                      'logits': m(input_ids=PROBE).logits.detach().clone(),
                      'mlp_A': m.model.layers[0].mlp.gate_proj.A.detach().clone()}
        return {'accuracy': .1 * step, 'loss': 1.0}  # The final checkpoint wins.

    result = recover_model(model, batches(), evaluate, dict(CONFIG, lora_targets='mlp+attention'))
    assert all(v['wrapped'] for v in seen.values()) and result['best_step'] == 3
    hidden, kv = 16, 2 * 4  # k/v project to num_key_value_heads * head_dim = 8
    per_layer = sum(2 * (i + o) for i, o in [(16, 24), (16, 24), (24, 16), (hidden, 16), (hidden, kv), (hidden, kv), (16, 16)])
    assert result['trainable_parameters'] == 2 * per_layer and result['recipe']['lora_targets'] == 'mlp+attention'
    for layer in model.model.layers:
        for name in NAMES:
            assert type(getattr(layer.self_attn, name)) is nn.Linear
    state = model.state_dict()
    for key in before:
        if key.endswith('bias'):
            assert torch.equal(before[key], state[key]), key
    assert all(not torch.equal(before[f'model.layers.0.self_attn.{n}.weight'], state[f'model.layers.0.self_attn.{n}.weight'])
               for n in NAMES)
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids=PROBE).logits, seen[max(seen)]['logits'], rtol=1e-5, atol=1e-5)
    # MLP adapter initialization is identical to the MLP-only recipe (attention adapters are created after).
    mlp_only = {}
    recover_model(tiny(), batches(), lambda m: mlp_only.setdefault(len(mlp_only), m.model.layers[0].mlp.gate_proj.A.detach().clone())
                  is not None and {'accuracy': .1, 'loss': 1.0}, CONFIG)
    assert torch.equal(mlp_only[0], seen[0]['mlp_A'])


def test_failure_removes_attention_adapters_and_restores_weights():
    model = tiny()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    calls = []

    def evaluate(m):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError('boom')
        return {'accuracy': .1, 'loss': 1.0}

    with pytest.raises(RuntimeError, match='boom'):
        recover_model(model, batches(), evaluate, dict(CONFIG, lora_targets='mlp+attention'))
    assert all(type(getattr(layer.self_attn, n)) is nn.Linear for layer in model.model.layers for n in NAMES)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(p.requires_grad for p in model.parameters())


def test_lora_target_validation_and_config():
    assert validate_lora_targets({}) == 'mlp'
    with pytest.raises(ValueError, match='lora_targets'):
        validate_lora_targets({'lora_targets': 'attention'})
    config = campaign.read_campaign_config()
    config['recovery']['lora_targets'] = 'all'
    with pytest.raises(ValueError, match='lora_targets'):
        campaign.validate_config(config)
    longer = campaign.read_campaign_config('configs/support-v2-longer.json')
    attn = campaign.read_campaign_config('configs/support-v2-attn.json')
    campaign.validate_config(attn)
    longer['recovery']['lora_targets'] = 'mlp+attention'
    assert attn == longer and 'distillation' not in attn['recovery']


def test_campaign_with_attention_adapters_exports_and_reloads(tmp_path, monkeypatch):
    from test_multiseed import tiny_support_config
    config = tiny_support_config(tmp_path, monkeypatch)
    config.update(selection_dev_examples=2, skip_full_merged_dev=True)
    config['recovery'].update(lora_targets='mlp+attention', learning_rate=.05)
    output = tmp_path / 'runs' / 'attn'
    campaign.run_campaign(config, output, 'preflight')
    campaign.run_campaign(config, output, 'fit')
    for record in campaign._selection(output)['candidates']:
        assert record['status'] == 'complete' and record['recovery']['recipe']['lora_targets'] == 'mlp+attention'
        assert record['merge_check']['identical_predictions_after_reload'] == record['merge_check']['examples']
