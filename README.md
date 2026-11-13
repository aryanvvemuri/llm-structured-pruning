# Local Qwen neuron pruning lab

This project tests structured MLP pruning on numeric arithmetic and document-grounded customer-support replies. It supports uniform pruning, different widths per layer, and short LoRA recovery training. It uses Qwen2.5-0.5B-Instruct in float32, with one loaded model at a time, for a Mac with 24 GB unified memory. The full campaign explicitly requires MPS; CPU is available when explicitly selected.

See [RESULTS.md](RESULTS.md) for validation: **131 current software tests**, verified checkpoint exports, and completed M5 Pro experiments. On 300 held-out Doc2Dial support replies, recovered full Qwen scores **53.72 token F1**, per-layer 10% pruning **52.47**, uniform 10% **51.94**, and per-layer 20% **46.19**. These are word-overlap scores, not customer-support accuracy. No pruned candidate qualified for the zero-drop development tier, and per-layer allocation has not established an advantage over uniform pruning. Earlier arithmetic results and their data-overlap limitations remain in the report.

## Setup

Run from this project directory. Python 3.13 is recommended; use native arm64 Python on Apple Silicon.

```sh
UV_CACHE_DIR=.cache/uv uv venv --python 3.13 .venv
UV_CACHE_DIR=.cache/uv uv pip install --python .venv/bin/python -e '.[test]'
source .venv/bin/activate
export HF_HOME="$PWD/.cache/huggingface"
prune-lab doctor --config configs/smoke.json
pytest -q
```

The first model load downloads weights into `.cache/huggingface/hub`; later runs reuse them. Set `HF_HUB_OFFLINE=1` after downloading to require offline operation. CPU threads default to four. MPS availability depends on the process environment; `doctor` reports the actual choice. Float32 avoids low precision gradient surprises during the first Taylor smoke test. Larger models and contexts can exceed 24 GB; start small.

`requirements-tested.txt` records the exact dependency versions from validation. To reproduce that environment, install it with `uv pip install --python .venv/bin/python -r requirements-tested.txt`, then install this package editable. Structural export temporarily holds replacement feed-forward tensors alongside the originals, so its peak memory can exceed ordinary inference memory.

## Per-layer pruning and recovery campaign

The production configuration uses **1,000 recovery-training, 150 calibration, 200 development, and 300 final-test questions**. These are four disjoint sets of signed addition, subtraction and multiplication, including short word templates. Repeated operations and operands share semantic keys; swapping operands in addition/multiplication does not create a new problem. Previously used arithmetic is excluded using the legacy seed and any existing `runs/adaptive/*.jsonl` files. Optional `exclusion_jsonl` paths add strict exclusions and must exist.

Calibration Taylor scores rank neurons within a layer. Separate layer ablations measure answer-loss sensitivity on 16 calibration examples; this smaller probing subset limits cost. A heuristic uses those sensitivities to allocate aligned neuron groups against **5%, 10%, 15%, and 20% of all original model parameters**. Actual reductions reflect rounding. This is a fixed candidate grid, not an exhaustive search or a guarantee of a global optimum.

Each candidate starts from the original model, physically removes gate/up rows and corresponding down columns, and trains rank-8 adapters on the three MLP projections. Defaults are batch size one, accumulation four, 200 optimizer updates, and development checks at steps 0, 50, 100 and 200. Step zero can win if training makes things worse. Only adapters are trained; the best development state is merged into the base weights before export. A 1,000-example recovery split is available; the default 800 microbatches sample less than one full pass. No second teacher model is loaded.

The campaign also trains an unpruned control and a uniform 10% pruning control with the same recovery budget. Each export is reloaded and evaluated on development data before selection. The smallest qualifying per-layer candidate is selected for each of three tolerances: **0, 5, and 10 percentage points** below the original development accuracy. For example, 80% becomes floors of 80%, 75% and 70%. Controls are excluded from this selection. Several tiers may select the same checkpoint, and a tier can have no qualifying pruned model. Matching observed accuracy is not proof of equivalent accuracy on unseen problems.

### Select by retained baseline quality

New campaigns can instead ask for the **smallest tested model retaining at least 95% of a baseline score**. Add these fields to the campaign configuration:

```json
{
  "selection_reference": "unpruned-recovery",
  "include_controls": true,
  "tiers_pp": null,
  "retention_targets": [0.95]
}
```

The development floor is `baseline_score * 0.95`: baseline accuracy of 80% requires 76% accuracy, rather than 75%. For support, this applies to token F1, **not customer-support accuracy**. Multiple retention targets are supported, such as `[1.0, 0.95, 0.90]`. Set `tiers_pp` to `null` because relative retention and absolute point tolerances are distinct modes. A zero development baseline is rejected because its retention ratio is meaningless.

`targets` still defines the bounded compression candidates to train and compare; the quality requirement determines which resulting size is acceptable. The harness selects the smallest passing tested candidate, so the final compression amount is an outcome. It does not automatically refine the search boundary or establish a global optimum. Minimizing parameters also does not guarantee minimum measured latency. Each candidate receives recovery training before its development score is considered.

Selection is frozen using development data. On final testing, the same fraction is applied to the reference model's **test** score, which can differ from its development score. Reports show the retained score ratio and any threshold failure without choosing a different model. A zero test baseline is reported as unassessable and never counted as a pass. An observed 95% retention result is not a statistical guarantee of population retention.

Use a new output directory for this mode; existing campaign configurations, selections and results remain frozen. Since the existing support test has been inspected, further quality tuning needs a fresh held-out set. The completed support experiment still uses its original absolute-point tiers; this optional mode does not rewrite that experiment or represent a new model-training run.

Run stages separately so preflight measurements can inform the budget before fitting:

```sh
prune-lab campaign --config configs/campaign.json --stage preflight --output runs/campaign-v2
prune-lab campaign --config configs/campaign.json --stage fit --output runs/campaign-v2
prune-lab campaign --config configs/campaign.json --stage finalize --output runs/campaign-v2
prune-lab campaign --config configs/campaign.json --stage benchmark --output runs/campaign-v2
```

`preflight` performs two recovery updates and small development evaluations, recording time and memory snapshots. The full run performs six recoveries with repeated development evaluations; those evaluations can dominate runtime. `--stage all` runs every stage sequentially. `configs/campaign-smoke.json` provides a tiny CPU pipeline check and is not quality evidence.

`fit` writes and locks `selection.json` after all candidates have been attempted. `finalize` evaluates the original, selected candidates, and controls on the held-out test set, then writes `summary.json` and `report.md`. It reports any missed test thresholds without changing selection. `benchmark` adds three timing repetitions on 64 development prompts, with token counts and memory snapshots. Fewer parameters need not mean lower latency on MPS. Timing runs use one process, alternating model order with cache cleanup; memory counters are snapshots, not peaks.

Recovery may change answer length. To separate that effect from computation time, run `HF_HUB_OFFLINE=1 .venv/bin/python scripts/benchmark_campaign_fixed.py --run runs/campaign-v2` after finalization. This supplementary benchmark forces exactly 16 generated tokens on the same 64 development prompts, with three alternating-order repeats. Results go to `fixed-benchmark/summary.json`. These forced continuations measure timing, not answer quality.

Interrupted fitting resumes completed candidates and retries unfinished ones. When fitting reaches the end of the candidate list, it freezes selection, including any recorded failures; another `fit` invocation then skips work. Retrying those failures requires a new run directory before opening the final test. Configuration, dataset, selection, and checkpoint fingerprints prevent accidental mixing of runs. After final testing starts, fitting is locked. Use a new directory for a changed configuration; once test outcomes have been inspected, use a new held-out test set for further tuning. Checkpoints are saved under `candidates/<id>/checkpoint`; six float32 exports require several GB of disk space.

Different retained widths require the project's local loader; vanilla Hugging Face loading is supported for uniform exports only. No remote model code is executed:

```python
from prune_lab.modeling import load_model
model, tokenizer = load_model("runs/campaign-v2/candidates/per-layer-01/checkpoint", device="mps")
```

Custom campaign data can use `dataset: "jsonl"` with four `data_paths` entries and matching `counts`. Each row must carry an arithmetic semantic key such as `"+:2:3"`, an appropriate prompt, and the correct numeric answer. This campaign currently measures arithmetic specialization, not GSM8K or broad mathematical reasoning.

The completed run's holdout shares training templates and operand ranges. A follow-up audit also found algebraically equivalent problems that operation-key deduplication missed: 17 of 300 test questions correspond to examples actually used to train the selected model. The measured scores are unchanged, but claims of fully semantic-disjoint data were too strong. See the follow-up data audit in [RESULTS.md](RESULTS.md). No out-of-distribution or unrelated-task evaluation has been performed.

## Grounded customer-support experiment

The support adapter uses [Doc2Dial v1.0.1](https://doc2dial.github.io/data.html), by Feng et al. (EMNLP 2020). The [IBM dataset card](https://huggingface.co/datasets/IBM/doc2dial) identifies the license as CC-BY-3.0. The archive is pinned by SHA-256, and prepared data retain attribution, provenance, filters, and overlap audits. These are crowdsourced dialogues about public-service documentation across DMV, SSA, VA and student aid; they are not real company helpdesk tickets.

The prepared benchmark has 1,000 recovery / 150 calibration / 200 development / 300 final-test examples, one target reply per conversation. Whole document families stay within one split. Families combine normalized document titles, identical content and near-duplicate documents. The final test comes from official test conversations; recovery/calibration/development use official train/validation conversations. Additional cross-split response and conversation overlap checks are recorded in the manifest. These are custom subsets, not official leaderboard scores. Lexical deduplication does not establish complete semantic independence.

Each prompt contains previous conversation turns and the **annotated relevant document sections**. Choosing evidence from gold references makes this an oracle-evidence response-generation benchmark. It does not evaluate document retrieval, autonomous tool actions, or end-to-end support resolution. Responses over 64 tokens and combined prompt/response sequences over 512 tokens are excluded instead of silently truncated, so longer or more complex cases are underrepresented.

```sh
mkdir -p .cache/datasets/doc2dial
curl --fail --location --output .cache/datasets/doc2dial/doc2dial_v1.0.1.zip https://doc2dial.github.io/file/doc2dial_v1.0.1.zip
.venv/bin/python scripts/prepare_support_data.py
HF_HUB_OFFLINE=1 .venv/bin/python -m prune_lab campaign --config configs/support.json --stage preflight --output runs/support-v1
HF_HUB_OFFLINE=1 .venv/bin/python -m prune_lab campaign --config configs/support.json --stage fit --output runs/support-v1
HF_HUB_OFFLINE=1 .venv/bin/python -m prune_lab campaign --config configs/support.json --stage finalize --output runs/support-v1
HF_HUB_OFFLINE=1 .venv/bin/python -m prune_lab campaign --config configs/support.json --stage benchmark --output runs/support-v1
```

The bounded first support configuration uses 50 optimizer updates, accumulation four, and development checks at updates 0/25/50. Thus each variant visits 200 of the 1,000 available recovery examples; the best checkpoint can be earlier. It compares unpruned recovery, uniform 10% pruning, and per-layer 10% and 20% pruning, each starting from original Qwen2.5-0.5B. The selection reference is the **recovered unpruned model**. Uniform and per-layer candidates can qualify for 0/5/10 absolute token-F1-point drops, with selection frozen before final testing. All predeclared candidates receive final evaluation for comparisons at similar parameter counts.

**Token F1 and ROUGE-L are word-overlap proxies, not customer-support accuracy.** Contradictions can score highly, and valid paraphrases can score poorly. We also record answer-token loss, normalized exact match, output lengths, and the fraction of generations reaching the length cap. Paired intervals resample document families. The report explicitly names the metric and its limits; no policy-correctness claim follows from the overlap score.

`finalize` writes a 30-document-group blind review worksheet and a separate model-identity key. The worksheet asks reviewers to judge grounding, whether the reply addresses the request, and fabricated facts/actions. Ratings start blank and are preserved on resume. Review results must not be used to reselect models on the same held-out set. The fixed-token timing script also works for support campaigns; ordinary response latency includes differences in answer length.

The completed `runs/support-v1` run also includes a separate exploratory LLM review of all 150 anonymized replies in that worksheet. Its ratings were frozen before unblinding, and the human worksheet remains blank. This is not human validation or an accuracy estimate. The 20% model had more incomplete replies in that small audit. Public Doc2Dial data may also have appeared in Qwen's pretraining; the experiment establishes separation of its own splits, not pretraining exclusion. Further tuning needs a new held-out set.

## Original uniform pruning experiment

```sh
prune-lab evaluate --config configs/smoke.json --output runs/baseline.json
prune-lab score --config configs/smoke.json --method taylor --output runs/scores.json
prune-lab search --config configs/smoke.json --method taylor --output runs/search.json
prune-lab report --result runs/search.json --output runs/report.md
prune-lab export --config configs/smoke.json --result runs/search.json --output runs/compact
prune-lab evaluate --config configs/smoke.json --model runs/compact --split test --output runs/compact-evaluation.json
```

`search` evaluates original development performance, ranks each layer's neurons using calibration only, tests reversible masks, and accepts only candidates meeting the original development accuracy floor. `max_drop_pp` is **percentage points**, so 2 means 80% may fall to 78%. Optional `loss_tolerance` is an absolute increase in answer-token cross entropy. Rejection halves the removal step; acceptance recomputes scores under the accepted mask. All layers retain the same width so export can be reloaded with standard Hugging Face classes. Search rounds and minimum retained ratio bound the work. Methods are `activation`, `contribution`, `taylor`, and a search-only `random` baseline. Run each method separately with the same configuration and seed.

Held-out test evaluation happens once after mask selection. Do not choose methods or hyperparameters from repeated test results. The exported checkpoint records original neuron indices and source dataset fingerprints. A mask alone does not reduce dense compute or weight storage; structural export does. Compare reloaded compact and original models using identical device, dtype, prompt set, generation settings, and thread count. Timings include synchronized generation after an unmeasured warmup; token counts, wall time, current RSS, and available MPS allocator counters are reported separately. These are snapshots, not peak system memory. A report describes what was measured, not an inferred speedup.

## Data and limitations

Synthetic arithmetic uses one seeded, sampled-without-replacement operation/operand pool, then partitions it into calibration/development/test. Canonical prompt and example-key duplicates across all splits are rejected. Training labels cover answer tokens only. Overlength examples fail with the required token count; no silent truncation or empty supervision occurs. Numeric parsing accepts an entire numeric completion or an explicit final answer marker, including negative values and decimals. It does not award credit for a number copied from the question. Arbitrary reasoning completions without a clear final answer can be scored wrong.

For your own JSONL data set `dataset` to `jsonl` and `data_paths` to an object with `calibration`, `dev`, and `test` filenames. Each line is `{"prompt": "...", "answer": "42", "key": "optional stable identity"}`. Supply canonical keys for equivalent questions whose text differs. Exact duplicate prompts and keys are rejected; semantic paraphrase overlap cannot be inferred automatically.

For GSM8K install `.[gsm8k]` and set `dataset` to `gsm8k`. Calibration and development are disjoint shuffled subsets of official training data; test is a prefix of the official test partition. Increase `max_length` for worked solutions and `max_new_tokens` for reasoning. The offline synthetic source needs no dataset downloads. JSON configs merge top-level defaults; a supplied `search` object replaces the default search options (omitted search arguments use function defaults).

This is a practical ablation workflow, not a custom LLM agent runtime. It does not claim that arithmetic generalizes to broader capabilities. Smoke sets contain only a few cases, so one answer changes accuracy dramatically; use larger development/test sets and report counts before drawing conclusions.
