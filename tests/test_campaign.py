"""Campaign selection and state safeguards, independent of GPU availability."""
import json
from types import SimpleNamespace

import pytest

from prune_lab import campaign
from prune_lab.reporting import write_json


def baseline():
    return {"parameters": 1000, "dev": {"accuracy": .8, "answer_token_loss": 1.0}}


def candidate(name, parameters, accuracy, **extra):
    return {"id": name, "status": "complete", "control": False,
            "parameters": parameters, "dev": {"accuracy": accuracy, "answer_token_loss": 1.0},
            "checkpoint": f"/checkpoints/{name}", **extra}


def test_selection_uses_percentage_points_and_smallest_qualified_model():
    rows = [candidate("equal", 900, .8), candidate("five", 800, .75), candidate("ten", 700, .70)]
    tiers = campaign.select_tiers(baseline(), rows, [0, 5, 10])
    assert [t["candidate_id"] for t in tiers] == ["equal", "five", "ten"]
    assert tiers[1]["development_floor"] == pytest.approx(.75)


def test_missing_tier_is_not_filled_by_control_invalid_or_unpruned_model():
    rows = [candidate("control", 500, 1.0, control=True),
            candidate("original", 1000, 1.0), candidate("nan", 400, float("nan")),
            {"id": "failed", "status": "failed"}, candidate("worse", 800, .79)]
    tier = campaign.select_tiers(baseline(), rows, [0])[0]
    assert tier["candidate_id"] is None
    assert tier["status"] == "no_qualifying_pruned_model"


def test_one_checkpoint_can_fill_three_tiers_but_is_tested_once():
    row = candidate("good", 700, .8)
    selection = {"candidates": [row], "tiers": campaign.select_tiers(baseline(), [row], [0, 5, 10])}
    assert [t["candidate_id"] for t in selection["tiers"]] == ["good"] * 3
    assert [t["id"] for t in campaign.evaluation_targets(selection)] == ["original", "good"]


def test_equal_size_selection_has_deterministic_quality_tiebreak():
    rows = [candidate("a", 700, .8), candidate("b", 700, .81), candidate("c", 800, .9)]
    assert campaign.select_tiers(baseline(), rows, [0])[0]["candidate_id"] == "b"


def test_paired_comparison_tracks_regressions_even_if_aggregate_equal():
    original = {"predictions": [{"key": "a", "correct": True}, {"key": "b", "correct": False}]}
    compact = {"predictions": [{"key": "a", "correct": False}, {"key": "b", "correct": True}]}
    result = campaign.paired_accuracy_delta(original, compact, samples=100)
    assert result["delta_pp"] == 0
    assert result["regressions"] == result["improvements"] == 1
    assert result["bootstrap_95_percent_interval_pp"][0] < 0
    compact["predictions"].reverse()
    with pytest.raises(ValueError, match="ordered example keys"):
        campaign.paired_accuracy_delta(original, compact)


def test_config_rejects_probe_that_would_fail_after_expensive_scoring():
    config = campaign.read_campaign_config()
    config["probe_fraction"] = 1
    with pytest.raises(ValueError, match="strictly less"):
        campaign.validate_config(config)


def test_checkpoint_identity_detects_changed_weights_and_missing_files(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"original")
    original_hash = campaign.checkpoint_identity(tmp_path)
    weights.write_bytes(b"modified")
    assert campaign.checkpoint_identity(tmp_path) != original_hash
    weights.unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        campaign.checkpoint_identity(tmp_path)


def test_resuming_rejects_configuration_or_dataset_change(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = campaign.read_campaign_config()
    config["counts"] = {"recovery": 8, "calibration": 4, "dev": 4, "test": 4}
    output = tmp_path / "run"
    state, splits = campaign._initialize(config, output)
    assert campaign._initialize(config, output) == (state, splits)
    changed = dict(config, seed=config["seed"] + 1)
    with pytest.raises(ValueError, match="configuration changed"):
        campaign._initialize(changed, output)
    path = output / "data" / "test.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["answer"] = "9999999"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="Persisted datasets changed"):
        campaign._initialize(config, output)


def test_frozen_selection_cannot_be_edited(tmp_path):
    selection = {"tiers": []}
    write_json(tmp_path / "selection.json", selection)
    write_json(tmp_path / "selection-lock.json", {"sha256": campaign._fingerprint(selection)})
    assert campaign._selection(tmp_path) == selection
    write_json(tmp_path / "selection.json", {"tiers": [1]})
    with pytest.raises(ValueError, match="selection changed"):
        campaign._selection(tmp_path)


def test_final_test_failure_is_reported_without_reselection(tmp_path, monkeypatch):
    row = candidate("chosen", 800, .8)
    selection = {"baseline": baseline(), "candidates": [row],
                 "tiers": campaign.select_tiers(baseline(), [row], [0, 5, 10])}
    write_json(tmp_path / "selection.json", selection)
    lock = {"sha256": campaign._fingerprint(selection)}
    write_json(tmp_path / "selection-lock.json", lock)
    calls = []
    monkeypatch.setattr(campaign, "_verify_checkpoint", lambda record: None)
    monkeypatch.setattr(campaign, "_verify_source", lambda *args: None)
    monkeypatch.setattr(campaign, "_provenance", lambda model: {"device": "cpu"})
    monkeypatch.setattr(campaign, "_release", lambda: None)
    monkeypatch.setattr(campaign, "_load", lambda config, checkpoint: (SimpleNamespace(checkpoint=checkpoint), None))

    def evaluate(model, tokenizer, examples, config):
        calls.append(model.checkpoint)
        correct = 8 if model.checkpoint is None else 6
        return {"accuracy": correct / 10, "correct": correct, "count": 10,
                "predictions": [{"key": str(i), "correct": i < correct} for i in range(10)]}

    monkeypatch.setattr(campaign, "_evaluate", evaluate)
    state = {"test_started": False, "dataset": {}}
    result = campaign.finalize({"seed": 42}, tmp_path, state, {"test": list(range(10))})
    assert calls == [None, row["checkpoint"]]
    assert all(tier["test_threshold_met"] is False for tier in result["tiers"])
    assert [tier["candidate_id"] for tier in result["tiers"]] == ["chosen"] * 3
    assert json.loads((tmp_path / "selection-lock.json").read_text()) == lock
    assert json.loads((tmp_path / "state.json").read_text())["test_started"] is True
    campaign.finalize({"seed": 42}, tmp_path, state, {"test": list(range(10))})
    assert len(calls) == 2  # Resume consumes cached test outputs, not fresh experiments.
