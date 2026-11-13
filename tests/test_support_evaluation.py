from types import SimpleNamespace

import pytest
import torch

from prune_lab.support_evaluation import evaluate_support, normalize_reply, reply_metrics


def test_normalization_and_independent_lexical_values():
    assert normalize_reply(' THE  Refund—An APPROVAL! ') == 'refund approval'
    metrics = reply_metrics('refund refund pending', 'refund pending pending')
    assert metrics['token_f1'] == pytest.approx(2 / 3)
    assert metrics['rouge_l'] == pytest.approx(2 / 3)
    assert metrics['exact_match'] == 0
    reordered = reply_metrics('please send form', 'form send please')
    assert reordered['token_f1'] == 1
    assert reordered['rouge_l'] == pytest.approx(1 / 3)
    assert reply_metrics('The REFUND!', 'refund')['exact_match'] == 1
    assert reply_metrics('', 'refund') == {'token_f1': 0, 'rouge_l': 0, 'exact_match': 0}
    with pytest.raises(ValueError, match='reference'):
        reply_metrics('reply', 'the !')


def test_lexical_limitations_are_explicit():
    # Equivalent intents can have zero overlap; negation can retain high overlap.
    assert reply_metrics('Contact us', 'Get in touch')['token_f1'] == 0
    assert reply_metrics('You are eligible', 'You are not eligible')['token_f1'] == pytest.approx(6 / 7)
    # No numeric extractor accepts a numeric tail while discarding the reply.
    assert reply_metrics('Refund denied. Answer: 42', '42')['token_f1'] < 1


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    chat_template = None

    def encode(self, text, add_special_tokens=True):
        if add_special_tokens:
            return [3, 4]
        return [5] if text == 'refund' else [5, 6]

    def decode(self, ids, skip_special_tokens=True):
        return 'refund' if ids.tolist() == [5, 2] else ''


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.seen_labels = []
        self.generate_calls = []

    def forward(self, input_ids, attention_mask, labels, use_cache):
        assert not use_cache
        assert not torch.is_grad_enabled()
        self.seen_labels.append(labels.clone())
        n = int((labels != -100).sum())
        return SimpleNamespace(loss=torch.tensor(1.0 if n == 2 else 3.0))

    def generate(self, **kwargs):
        assert not torch.is_grad_enabled()
        self.generate_calls.append(kwargs)
        # Warmup and first timed response are nonempty; last response is EOS-only.
        continuation = [5, 2] if len(self.generate_calls) <= 2 else [2]
        return torch.cat((kwargs['input_ids'], torch.tensor([continuation])), dim=1)


def test_evaluator_cpu_labels_warmup_empty_output_and_metadata():
    examples = [SimpleNamespace(prompt='doc prompt', answer='refund', key='one',
                                doc_id='doc', dialogue_id='dialogue', domain='benefits', group_id='family'),
                SimpleNamespace(prompt='other doc prompt', answer='refund pending', key='two')]
    model = ToyModel().train()
    result = evaluate_support(model, ToyTokenizer(), examples, max_length=32, max_new_tokens=8)
    assert not model.training
    assert result['selection_score'] == 0.5
    assert result['score_name'] == 'token_f1'
    assert result['answer_token_loss'] == pytest.approx(11 / 5)
    assert result['supervised_tokens'] == 5
    assert result['generated_tokens'] == 3
    assert result['generation_limit_fraction'] == 0
    assert len(model.generate_calls) == 3
    assert all(call['pad_token_id'] == 0 and call['do_sample'] is False for call in model.generate_calls)
    assert model.seen_labels[0].tolist() == [[-100, -100, 5, 2]]
    assert model.seen_labels[1].tolist() == [[-100, -100, 5, 6, 2]]
    assert result['predictions'][0]['prediction'] == 'refund'
    assert result['predictions'][0]['doc_id'] == 'doc'
    assert result['predictions'][0]['group_id'] == 'family'
    assert result['predictions'][1]['prediction'] == ''
    assert result['predictions'][1]['score'] == 0
    assert 'accuracy' not in result and 'correct' not in result
    assert all('correct' not in row for row in result['predictions'])
    assert result['tokens_per_second'] >= 0
    assert result['memory']['process_rss_bytes'] > 0


def test_evaluator_rejects_empty_reference_before_forward():
    model = ToyModel()
    with pytest.raises(ValueError, match='reference'):
        evaluate_support(model, ToyTokenizer(), [SimpleNamespace(prompt='p', answer='  ', key='x')])
    assert not model.seen_labels and not model.generate_calls


def test_generation_limit_counts_eos_on_last_token_without_claiming_truncation():
    examples = [SimpleNamespace(prompt='p', answer='refund', key='one'),
                SimpleNamespace(prompt='p', answer='refund', key='two')]
    result = evaluate_support(ToyModel(), ToyTokenizer(), examples, max_new_tokens=2)
    # The first measured reply is [refund, EOS]; the second is [EOS].
    assert result['generation_limit_fraction'] == 0.5
    assert 'not confirmed truncation' in result['generation_limit_note']


def test_evaluator_real_tiny_cpu_generation():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(9)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=8, hidden_size=8, intermediate_size=12, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32,
        eos_token_id=2, pad_token_id=0)).eval()
    examples = [SimpleNamespace(prompt='supplied doc', answer='refund', key='one')]
    result = evaluate_support(model, ToyTokenizer(), examples, max_length=16, max_new_tokens=2)
    assert 0 <= result['selection_score'] <= 1
    assert 1 <= result['generated_tokens'] <= 2
    assert result['answer_token_loss'] > 0
