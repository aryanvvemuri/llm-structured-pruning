"""Improvement 8: layer scan, greedy layer dropping and combined layer-drop + uniform MLP candidates."""
import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

from prune_lab import campaign, layerdrop, multiseed, scoring
from prune_lab.modeling import count_parameters
from test_multiseed import tiny_support_config


def tiny(layers=4, seed=3):
    torch.manual_seed(seed)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=11, hidden_size=16, intermediate_size=24, num_hidden_layers=layers,
                                        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                                        pad_token_id=0, eos_token_id=2)).eval()


def batches(n=3):
    g = torch.Generator().manual_seed(1)
    out = []
    for i in range(n):
        ids = torch.randint(3, 11, (1, 8), generator=g)
        labels = ids.clone()
        labels[:, :5] = -100
        out.append({'input_ids': ids, 'attention_mask': torch.ones_like(ids), 'labels': labels})
    return out


def test_drop_layers_is_physical_renumbered_and_stock_loadable(tmp_path):
    model = tiny()
    reference = [model.model.layers[i] for i in (0, 3)]
    x = torch.tensor([[3, 4, 5, 6]])
    kept = layerdrop.drop_layers(model, [1, 2])
    assert kept == [0, 3] and list(model.model.layers) == reference
    assert model.config.num_hidden_layers == 2 and len(model.config.layer_types) == 2
    assert [l.self_attn.layer_idx for l in model.model.layers] == [0, 1]
    assert model.config.prune_lab_dropped_layers == [1, 2] and model.config.prune_lab_kept_layers == [0, 3]
    with torch.no_grad():
        logits = model(input_ids=x).logits
        generated = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=4, do_sample=False)
    model.save_pretrained(tmp_path / 'export', safe_serialization=True)
    stock = AutoModelForCausalLM.from_pretrained(tmp_path / 'export').eval()  # stock class, no project loader
    assert len(stock.model.layers) == 2
    with torch.no_grad():
        torch.testing.assert_close(stock(input_ids=x).logits, logits)
        assert torch.equal(stock.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=4,
                                          do_sample=False), generated)  # KV cache uses the renumbered layer_idx
    with pytest.raises(ValueError):
        layerdrop.drop_layers(model, [0])  # a second drop on an already reduced model


def test_without_layers_restores_and_greedy_matches_brute_force():
    model = tiny(5)
    data = batches()
    before = layerdrop.calibration_loss(model, data)
    with layerdrop.without_layers(model, [1, 3]):
        assert len(model.model.layers) == 3 and model.config.num_hidden_layers == 3
    assert len(model.model.layers) == 5 and [l.self_attn.layer_idx for l in model.model.layers] == list(range(5))
    assert layerdrop.calibration_loss(model, data) == before
    scan = layerdrop.scan_and_select(model, data, 3)

    def loss_without(removed):
        with layerdrop.without_layers(model, removed):
            return layerdrop.calibration_loss(model, data)
    removed = []
    for step in range(3):
        remaining = [i for i in range(5) if i not in removed]
        losses = {i: loss_without(removed + [i]) for i in remaining}
        removed.append(min(remaining, key=lambda i: (losses[i], -i)))
        assert scan['greedy'][step]['loss'] == losses[removed[-1]]
    assert scan['order'] == removed and scan['baseline_loss'] == before
    assert [r['layer'] for r in scan['single_layer']] == list(range(5))
    assert json.loads(json.dumps(scan)) == scan  # lists and string keys only


def test_ties_break_toward_higher_layer(monkeypatch):
    model = tiny(4)
    monkeypatch.setattr(layerdrop, 'calibration_loss', lambda m, b: 1.0)
    assert layerdrop.scan_and_select(model, batches(), 2)['order'] == [3, 2]


def base_config(**layer_drop):
    config = campaign.read_campaign_config('configs/support-v2-longer.json')
    config['layer_drop'] = layer_drop or {'candidates': [{'id': 'layers-7', 'drop': 7}]}
    return config


@pytest.mark.parametrize('layer_drop, message', [
    ({'candidates': [{'id': 'x', 'drop': 0}]}, 'drop must be'),
    ({'candidates': [{'id': 'x', 'drop': 2}, {'id': 'x', 'drop': 3}]}, 'Duplicate'),
    ({'candidates': [{'id': 'uniform-10', 'drop': 2}]}, 'collide'),
    ({'candidates': [{'id': 'x', 'drop': 2, 'total_target': 1.5}]}, 'total_target'),
    ({'candidates': []}, 'nonempty'),
    ({'examples': 151, 'candidates': [{'id': 'x', 'drop': 2}]}, 'exceeds'),
    ({'candidates': [{'id': 'x', 'drop': 2, 'extra': 1}]}, 'needs id and drop'),
])
def test_config_validation_without_shape(layer_drop, message):
    with pytest.raises(ValueError, match=message):
        campaign.validate_config(base_config(**layer_drop))


def test_shape_validation_and_defaults():
    with torch.device('meta'):
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=151936, hidden_size=896, intermediate_size=4864, num_hidden_layers=24,
                                             num_attention_heads=14, num_key_value_heads=2, tie_word_embeddings=True))
    config = base_config(candidates=[{'id': 'a', 'drop': 7}, {'id': 'b', 'drop': 3, 'total_target': .2}])
    assert set(layerdrop.check_shape(config, model)) == {'a', 'b'}
    for bad, message in (({'id': 'c', 'drop': 24}, 'smaller than'), ({'id': 'c', 'drop': 3, 'total_target': .09}, 'does not exceed'),
                         ({'id': 'c', 'drop': 3, 'total_target': .9}, 'not reachable')):
        with pytest.raises(ValueError, match=message):
            layerdrop.check_shape(base_config(candidates=[bad]), model)
    plain = campaign.read_campaign_config('configs/support-v2-longer.json')
    assert all('kind' not in s for s in campaign.candidate_specs(plain))
    real = campaign.read_campaign_config('configs/support-v2-layer-drop.json')
    campaign.validate_config(real)
    assert [s['id'] for s in campaign.candidate_specs(real)] == ['unpruned-recovery', 'layers-7', 'layers-3-uniform-20']
    longer = campaign.read_campaign_config('configs/support-v2-longer.json')
    assert {k: v for k, v in real.items() if k not in ('layer_drop', 'train_candidates')} == longer


@pytest.fixture
def layer_run(tmp_path, monkeypatch):
    config = tiny_support_config(tmp_path, monkeypatch)
    torch.manual_seed(5)
    Qwen2ForCausalLM(Qwen2Config(vocab_size=5, hidden_size=16, intermediate_size=24, num_hidden_layers=4, num_attention_heads=2,
                                 num_key_value_heads=1, max_position_embeddings=256, pad_token_id=0,
                                 eos_token_id=2)).save_pretrained(config['model'], safe_serialization=True)
    config.update(layer_drop={'candidates': [{'id': 'layers-2', 'drop': 2}, {'id': 'layers-1-uniform-40', 'drop': 1, 'total_target': .4}]},
                  train_candidates=['unpruned-recovery', 'layers-2', 'layers-1-uniform-40'])
    campaign.validate_config(config)
    return config, tmp_path / 'runs' / 'ld'


def test_layer_drop_campaign_records_order_resume_and_aggregate(layer_run, monkeypatch):
    config, root = layer_run
    widths_seen = []
    real = scoring.score_neurons
    monkeypatch.setattr(scoring, 'score_neurons', lambda m, b, method='taylor': widths_seen.append(len(m.model.layers)) or real(m, b, method))
    multiseed.run_seeds(config, root, [5])
    seed = root / 'seed-5'
    assert widths_seen == [4, 3]  # intact scores for the others; the combined candidate is re-scored after dropping
    scan = json.loads((seed / 'layer-scan.json').read_text())
    assert scan == json.loads((seed / 'allocation.json').read_text())['layer_scan'] and len(scan['order']) == 2
    records = {c['id']: c for c in campaign._selection(seed)['candidates']}  # JSON round trip keeps the lock valid
    two, combined = records['layers-2'], records['layers-1-uniform-40']
    assert two['dropped_layers'] == sorted(scan['order'][:2]) and len(two['kept_layers']) == 2
    assert combined['dropped_layers'] == sorted(scan['order'][:1]) and len(combined['layer_widths']) == 3
    for record in (two, combined):
        assert record['parameters'] == record['planned_parameters']
    assert two['layer_widths'] == [24, 24] and len(set(combined['layer_widths'])) == 1 and combined['layer_widths'][0] < 24
    original = records['unpruned-recovery']['parameters']
    assert combined['parameters'] <= original * (1 - .4) + 24 * 48  # total target reached within one uniform step
    stock = AutoModelForCausalLM.from_pretrained(two['checkpoint'])
    assert len(stock.model.layers) == 2 and count_parameters(stock) == two['parameters']
    # Resume: only a missing candidate is retrained, reproducing the same weights.
    weights = [p.read_bytes() for p in sorted(Path(combined['checkpoint']).glob('*.safetensors'))]
    for name in ('selection.json', 'selection-lock.json'):
        (seed / name).unlink()
    (seed / 'candidates' / 'layers-1-uniform-40' / 'result.json').unlink()
    multiseed.run_seeds(config, root, [5])
    again = {c['id']: c for c in campaign._selection(seed)['candidates']}['layers-1-uniform-40']
    assert [p.read_bytes() for p in sorted(Path(again['checkpoint']).glob('*.safetensors'))] == weights
    assert 'layers-2' in multiseed.aggregate_run(root, samples=10)['variants']
