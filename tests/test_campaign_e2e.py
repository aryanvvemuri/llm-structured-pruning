"""Independent offline CPU integration: real tokenizer, Qwen, recovery and exports."""
import json
import importlib.util
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from prune_lab import campaign
from prune_lab.modeling import count_parameters, load_model
from prune_lab.recovery import LoRALinear


def test_real_cpu_campaign_export_controls_lock_and_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    torch.set_num_threads(1)
    torch.manual_seed(913)
    source = tmp_path / "source"
    backend = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1, "[EOS]": 2,
                                          "Calculate": 3, "Answer": 4}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                       unk_token="[UNK]", eos_token="[EOS]")
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=128, pad_token_id=0, eos_token_id=2))
    original_count = count_parameters(model)
    model.save_pretrained(source, safe_serialization=True)
    tokenizer.save_pretrained(source)
    del model
    config = campaign.read_campaign_config()
    config.update(model=str(source), device="cpu", cpu_threads=1,
                  counts={"recovery": 3, "calibration": 2, "dev": 2, "test": 2},
                  targets=[.15], group_size=1, sensitivity_examples=2,
                  benchmark_examples=2, benchmark_repeats=2,
                  max_length=128, max_new_tokens=1)
    config["recovery"].update(rank=2, max_steps=2, gradient_accumulation=2,
                              checkpoints=[1, 2])
    output = tmp_path / "campaign"
    calls = []
    evaluate = campaign._evaluate

    def tracked_evaluate(model, tokenizer, examples, config):
        calls.append([ex.key for ex in examples])
        return evaluate(model, tokenizer, examples, config)

    monkeypatch.setattr(campaign, "_evaluate", tracked_evaluate)
    campaign.run_campaign(config, output, "preflight")
    campaign.run_campaign(config, output, "fit")
    state, splits = campaign._initialize(config, output)
    test_keys = {ex.key for ex in splits["test"]}
    assert not test_keys.intersection(key for call in calls for key in call)
    selection = campaign._selection(output)
    assert len(selection["candidates"]) == 3
    assert all(row["status"] == "complete" for row in selection["candidates"])
    assert selection["baseline"]["parameters"] == original_count
    for row in selection["candidates"]:
        loaded, _ = load_model(row["checkpoint"], "cpu")
        assert count_parameters(loaded) == row["parameters"]
        assert row["actual_reduction"] == pytest.approx(1 - row["parameters"] / original_count)
        assert not any(isinstance(module, LoRALinear) for module in loaded.modules())
        assert row["export_reload_verified"]
        assert row["recovery"]["optimizer_steps"] == 2
    assert len(set(selection["candidates"][0]["layer_widths"])) > 1
    before_resume = len(calls)
    campaign.run_campaign(config, output, "fit")
    assert len(calls) == before_resume
    campaign.run_campaign(config, output, "finalize")
    state, splits = campaign._initialize(config, output)
    with pytest.raises(ValueError, match="locked"):
        campaign.fit(config, output, state, splits)
    campaign.run_campaign(config, output, "benchmark")
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary["final_test"]) == {"original", "per-layer-01", "unpruned-recovery", "uniform-10"}
    assert all(row["metrics"]["count"] == 2 for row in summary["final_test"].values())
    assert (output / "report.md").read_text()
    before_resume = len(calls)
    campaign.run_campaign(config, output, "all")
    assert len(calls) == before_resume
    assert campaign._selection(output) == selection
    # Supplementary benchmark forces 16 tokens even for this EOS-capable model.
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_campaign_fixed.py"
    spec = importlib.util.spec_from_file_location("fixed_benchmark", script)
    fixed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixed)
    protected = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    timing = fixed.benchmark(output, examples=2)
    assert set(timing["models"]) == set(summary["final_test"])
    assert timing["run_orders"][0] == timing["run_orders"][2]
    assert timing["run_orders"][1] == list(reversed(timing["run_orders"][0]))
    for metrics in timing["models"].values():
        assert metrics["generated_tokens_per_run"] == [32, 32, 32]
        assert all(x > 0 for x in metrics["mean_example_ms_per_run"])
        assert all(x > 0 for x in metrics["tokens_per_second_per_run"])
    assert all(path.read_bytes() == data for path, data in protected.items())
    assert json.loads((output / "fixed-benchmark" / "summary.json").read_text()) == timing
