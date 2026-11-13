import contextlib
from decimal import Decimal
from types import SimpleNamespace
import pytest
from prune_lab.data import Example, synthetic_splits, validate_splits, make_batches
from prune_lab.evaluation import numeric_answer
from prune_lab.search import passes_guardrail, search_masks


def test_splits_are_disjoint_and_repeatable():
    first = synthetic_splits(8, 12, 16, 42)
    assert first == synthetic_splits(8, 12, 16, 42)
    validate_splits(first)
    assert len({e.key for rows in first.values() for e in rows}) == 36


def test_duplicate_prompt_rejected_even_with_different_key():
    with pytest.raises(ValueError, match="Duplicate"):
        validate_splits({"calibration": [Example("  A question", "1", "a")], "dev": [Example("a  question", "2", "b")]})


@pytest.mark.parametrize("text,value", [("-3.50", Decimal("-3.5")), ("#### 1,234", Decimal(1234)), ("The answer is -2.", Decimal(-2)), ("Compute 2 + 3", None), ("2 + 3 = 5", None), ("5 apples", None)])
def test_numeric_parser(text, value):
    assert numeric_answer(text) == value


def test_percentage_points_and_loss_guardrail():
    baseline = {"accuracy": .8, "answer_token_loss": 1.0}
    assert passes_guardrail({"accuracy": .78, "answer_token_loss": 1.1}, baseline, 2, .1)
    assert not passes_guardrail({"accuracy": .77, "answer_token_loss": 1.0}, baseline, 2)
    assert not passes_guardrail({"accuracy": .8, "answer_token_loss": 1.2}, baseline, 2, .1)


class Tokenizer:
    chat_template = None
    eos_token_id = 9
    pad_token_id = 0
    def encode(self, text, add_special_tokens=True):
        return [1, 2] if add_special_tokens else [3]


def test_answer_only_labels_and_explicit_overflow():
    ex = [Example("question", "answer", "a")]
    batch = make_batches(ex, Tokenizer(), max_length=4)[0]
    assert batch["labels"].tolist() == [[-100, -100, 3, 9]]
    with pytest.raises(ValueError, match="no truncation"):
        make_batches(ex, Tokenizer(), max_length=3)


def test_search_shrinks_rejection_then_rescores_acceptance(monkeypatch):
    import prune_lab.search as module
    model = SimpleNamespace(config=SimpleNamespace(intermediate_size=10, num_hidden_layers=2))
    state, scored = {"width": 10}, []
    @contextlib.contextmanager
    def mask(model, keep):
        previous = state["width"]
        state["width"] = len(keep[0])
        try:
            yield
        finally:
            state["width"] = previous
    def score(model, batches, method):
        scored.append(state["width"])
        return {0: list(range(10)), 1: list(range(10))}
    def evaluate():
        return {"accuracy": 1.0 if state["width"] >= 7 else 0.0, "answer_token_loss": 1.0}
    monkeypatch.setattr(module, "masked_neurons", mask)
    monkeypatch.setattr(module, "score_neurons", score)
    result = search_masks(model, [], evaluate, initial_step=.4, max_rounds=3)
    assert [row["accepted"] for row in result["history"]] == [False, True, False]
    assert scored == [10, 8]
    assert result["retained_width"] == 8
    assert result["keep_indices"][0] == list(range(2, 10))
    assert state["width"] == 10
    assert result["test_used_for_search"] is False
