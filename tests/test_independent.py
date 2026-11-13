"""Independent behavioral checks, including numerical derivatives and real CLI IO."""
import copy
import json

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from prune_lab.pruning import compact_model, masked_neurons
from prune_lab.scoring import score_neurons
from prune_lab.search import search_masks


@pytest.fixture
def tiny():
    torch.manual_seed(19)
    return Qwen2ForCausalLM(Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=128, attention_dropout=0,
        eos_token_id=2, pad_token_id=0,
    )).eval().requires_grad_(False)


@pytest.fixture
def sample():
    return dict(input_ids=torch.tensor([[4, 5, 6, 7]]),
                attention_mask=torch.ones(1, 4, dtype=torch.long),
                labels=torch.tensor([[-100, -100, 6, 7]]))


def test_taylor_against_tokenwise_finite_difference(tiny, sample):
    # Double precision permits a numerically independent derivative oracle.
    tiny.double()
    scores = score_neurons(tiny, [sample], 'taylor')
    epsilon = 1e-2
    for layer, channel in [(0, 3), (1, 8)]:
        derivative_sum = 0
        for token in range(4):
            losses = []
            for sign in [-1, 1]:
                def perturb(module, args):
                    scale = torch.ones_like(args[0])
                    scale[0, token, channel] += sign * epsilon
                    return (args[0] * scale,)
                handle = tiny.model.layers[layer].mlp.down_proj.register_forward_pre_hook(perturb)
                try:
                    with torch.no_grad():
                        logits = tiny(**sample).logits
                        # HF loss casts logits to float32, so compute the
                        # same shifted CE in float64 for finite differences.
                        losses.append(float(torch.nn.functional.cross_entropy(
                            logits[:, :-1].reshape(-1, 32),
                            sample['labels'][:, 1:].reshape(-1),
                            ignore_index=-100, reduction='sum')))
                finally:
                    handle.remove()
            derivative_sum += abs((losses[1] - losses[0]) / (2 * epsilon))
        assert float(scores[layer][channel]) == pytest.approx(derivative_sum / 4, rel=.015, abs=2e-6)


@pytest.mark.parametrize('method', ['activation', 'contribution', 'taylor'])
def test_right_padding_does_not_change_statistics(tiny, sample, method):
    expected = score_neurons(tiny, [sample], method)
    padded = {key: torch.cat([value, torch.tensor([[0, 0]]) if key != 'labels'
                              else torch.tensor([[17, 18]])], dim=1)
              for key, value in sample.items()}
    # Even nonignored labels at padded positions must be excluded from Taylor loss.
    actual = score_neurons(tiny, [padded], method)
    for layer in expected:
        torch.testing.assert_close(actual[layer], expected[layer], atol=1e-7, rtol=1e-5)


def test_scoring_failure_restores_mixed_state_and_existing_grads(tiny, sample):
    tiny.train()
    tiny.model.layers[1].eval()
    params = list(tiny.parameters())
    params[0].requires_grad_(True)
    params[0].grad = torch.ones_like(params[0])
    flags = [p.requires_grad for p in params]
    modes = [m.training for m in tiny.modules()]
    state = copy.deepcopy(tiny.state_dict())
    def explode(module, args, output):
        raise RuntimeError('injected forward failure')
    handle = tiny.model.layers[1].register_forward_hook(explode)
    try:
        with pytest.raises(RuntimeError, match='injected'):
            score_neurons(tiny, [sample], 'taylor')
    finally:
        handle.remove()
    assert modes == [m.training for m in tiny.modules()]
    assert flags == [p.requires_grad for p in params]
    assert torch.equal(params[0].grad, torch.ones_like(params[0]))
    for name, value in tiny.state_dict().items():
        assert torch.equal(value, state[name])
    assert not tiny.get_input_embeddings()._forward_hooks
    assert all(not layer.mlp.down_proj._forward_pre_hooks for layer in tiny.model.layers)


def test_compact_generation_with_cache_survives_reload(tiny, sample, tmp_path):
    keep = {0: [1, 3, 5, 7, 9, 11], 1: [2, 4, 6, 8, 10, 12]}
    args = dict(input_ids=sample['input_ids'], attention_mask=sample['attention_mask'],
                do_sample=False, min_new_tokens=5, max_new_tokens=5, use_cache=True)
    with masked_neurons(tiny, keep):
        expected = tiny.generate(**args)
    compact_model(tiny, keep).save_pretrained(tmp_path)
    loaded = Qwen2ForCausalLM.from_pretrained(tmp_path).eval()
    assert torch.equal(expected, loaded.generate(**args))


def test_search_guardrail_cannot_accumulate_accuracy_drop(tiny):
    values = iter([1.0, .95, .90])
    def evaluate():
        return {'accuracy': next(values), 'answer_token_loss': 1.0}
    result = search_masks(tiny, [], evaluate, method='random', initial_step=.1,
                          max_rounds=2, max_drop_pp=5)
    assert [row['accepted'] for row in result['history']] == [True, False]
    assert result['selected_dev']['accuracy'] == .95
    assert result['retained_width'] == 22
    assert all(not layer.mlp.down_proj._forward_pre_hooks for layer in tiny.model.layers)


def test_real_cli_search_export_reload_evaluate(tiny, tmp_path, monkeypatch, caplog):
    from prune_lab.cli import main
    monkeypatch.chdir(tmp_path)
    original = tmp_path / 'original'
    vocab = {'[PAD]': 0, '[UNK]': 1, '[EOS]': 2}
    vocab.update({str(i): i+3 for i in range(29)})
    backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]',
                                       unk_token='[UNK]', eos_token='[EOS]')
    tiny.save_pretrained(original)
    tokenizer.save_pretrained(original)
    config = dict(model=str(original), device='cpu', dtype='float32', cpu_threads=2,
                  counts={'calibration': 2, 'dev': 2, 'test': 2}, max_length=128,
                  max_new_tokens=2, search=dict(initial_step=.25, min_step=1,
                    max_rounds=1, max_drop_pp=100, min_keep_ratio=.5))
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    result_path = tmp_path / 'search.json'
    main(['search', '--config', str(path), '--method', 'taylor', '--output', str(result_path)])
    result = json.loads(result_path.read_text())
    assert result['search_seconds'] > 0
    assert result['search']['retained_width'] == 18
    assert result['final_test']['count'] == 2
    exported = tmp_path / 'exported'
    main(['export', '--config', str(path), '--result', str(result_path), '--output', str(exported)])
    manifest = json.loads((exported / 'manifest.json').read_text())
    assert manifest['parameters_after'] < manifest['parameters_before']
    evaluation = tmp_path / 'evaluation.json'
    main(['evaluate', '--config', str(path), '--model', str(exported), '--split', 'test', '--output', str(evaluation)])
    evaluated = json.loads(evaluation.read_text())
    assert evaluated['metrics']['accuracy'] == result['final_test']['accuracy']
    assert evaluated['metrics']['answer_token_loss'] == pytest.approx(result['final_test']['answer_token_loss'], abs=1e-5)
    report = tmp_path / 'report.md'
    main(['report', '--result', str(result_path), '--output', str(report)])
    assert 'does not establish a speedup' in report.read_text()
    assert 'incorrect regex pattern' not in caplog.text
    from prune_lab.modeling import load_model
    _, reloaded_tokenizer = load_model(str(exported), device='cpu')
    for text in ['12 + 19 = 31', 'answer: -2.5', '10\n20']:
        assert reloaded_tokenizer.encode(text) == tokenizer.encode(text)


@pytest.mark.parametrize('field', ['accuracy', 'answer_token_loss'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_guardrail_rejects_nonfinite_metrics(field, value):
    from prune_lab.search import passes_guardrail
    baseline = {'accuracy': .8, 'answer_token_loss': 1.0}
    broken = dict(baseline, **{field: value})
    with pytest.raises(ValueError, match='finite'):
        passes_guardrail(broken, baseline, 0)
    with pytest.raises(ValueError, match='finite'):
        passes_guardrail(baseline, broken, 0)


@pytest.mark.parametrize('bound', ['initial_step', 'min_step', 'max_rounds',
                                   'max_drop_pp', 'min_keep_ratio', 'loss_tolerance'])
def test_search_rejects_nan_bounds_before_evaluation(tiny, bound):
    def unexpected_evaluation():
        pytest.fail('Invalid search must fail before model evaluation')
    with pytest.raises(ValueError, match='finite'):
        search_masks(tiny, [], unexpected_evaluation, **{bound: float('nan')})


def test_unsupported_checkpoint_rejected_before_tokenizer_or_weights(tmp_path, monkeypatch):
    from transformers import LlamaConfig
    import prune_lab.modeling as module
    LlamaConfig(hidden_size=16, intermediate_size=24, num_hidden_layers=1,
                num_attention_heads=2).save_pretrained(tmp_path)
    def unexpected_load(*args, **kwargs):
        pytest.fail('Unsupported architecture must be rejected before loading tokenizer/weights')
    monkeypatch.setattr(module.AutoTokenizer, 'from_pretrained', unexpected_load)
    monkeypatch.setattr(module.AutoModelForCausalLM, 'from_pretrained', unexpected_load)
    with pytest.raises(ValueError, match='Qwen2'):
        module.load_model(str(tmp_path), device='cpu')
