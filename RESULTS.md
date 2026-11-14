# Implementation and local validation results

## Evaluation power: v2 data and 5-seed baseline (November 13, 2026)

Added a larger Doc2Dial build (`datasets/doc2dial_v2`: 1,800 recovery / 150 calibration / 600 development / 224 test), a multi-seed runner, batched generation, and a two-interval decision rule. The final test split is smaller than v1's (the official test partition has 656 eligible dialogues; 224 remain after excluding the inspected v1 test and balancing domains) and stays sealed. Decisions use **dev-holdout**: the 450 development examples outside the 150-example checkpoint-selection subset.

Five recovery seeds (20261114–20261118), Qwen2.5-0.5B-Instruct, float32, `mps:0`, 50 optimizer updates, `eval_batch_size` 16 (replies byte-identical to batch one on 96 MPS and 24 CPU examples). Token F1 is a 0–100 word-overlap score, not accuracy.

| Variant | Removed | Mean F1 | Seed sd | Per-seed F1 | Δ vs recovery only | Decision |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| Original Qwen | 0% | 22.05 | — | — | −26.67 | confident |
| Recovery only | 0% | 48.72 | 3.34 | 43.22, 49.01, 48.62, 51.30, 51.47 | — | — |
| Uniform 10% | 10.03% | 47.05 | 3.23 | 42.88, 46.05, 47.65, 51.83, 46.86 | −1.67 | not established |
| Per-layer 10% | 9.99% | 46.64 | 2.11 | 45.09, 45.12, 45.51, 49.95, 47.55 | −2.08 | not established |
| Per-layer 20% | 19.99% | 46.10 | 2.39 | 43.26, 45.62, 46.52, 49.81, 45.29 | −2.63 | not established |

Per-layer 10% minus uniform 10% is −0.41 (bootstrap [−2.38, +1.67], seed t [−2.69, +1.87]): no advantage for the allocator is established. A difference is "confident" only when a hierarchical bootstrap over seeds and document families and a seed-level Student-t interval both exclude zero with the same sign.

**Seed variance dominates.** The same variant moves 2–3.3 F1 points between seeds, mostly as a whole-seed shift shared across variants (between-variant correlation of per-seed means 0.68–0.99). Each seed trains on a different 200 of the 1,800 recovery examples, and the selected checkpoint alternated between update 25 and 50 (10 of 20 cells each). The v1 single-seed 7.5-point loss for 20% pruning is not reproduced on average here. With this variance, a paired-by-seed comparison of two runs detects a true 4-point difference about 80% of the time and a 2-point difference about 33% of the time.

An independent audit recomputed all 15,000 per-example F1/ROUGE-L values (zero mismatches), every mean, difference, interval and decision, parameter counts, mask identity across seeds, 150/150 merge-check agreement in all 20 cells, and confirmed no test-split evaluation exists. Independent tests: **222 passed, 1 expected failure** (documented: one v2 test prompt equals a v1 test prompt). Artifacts: `runs/support-v2-main/multi-seed-summary.json`, `multi-seed-report.md`; checkpoints were deleted after evaluation, per-example records retained.

## Relative baseline-retention mode

Added optional `retention_targets`, such as `[0.95]`, with `tiers_pp: null`. The harness selects the smallest tested candidate whose development score is at least the requested fraction of its chosen baseline. Final testing recomputes the floor from the baseline's test score, reports failures without reselection, and marks a zero test baseline as unassessable. For support this measures token-F1 retention, not accuracy. Candidate generation remains a bounded grid, rather than an adaptive search for maximum compression.

Independent verification passed **131 tests in 4.51 seconds**, including nine additional independent report, compatibility and fit-to-selection checks. Historical run configurations, selection locks and artifacts remained unchanged. A read-only demonstration on saved support development scores gives a 95% floor of 43.0448 F1 against the recovered baseline's 45.3103, which would choose per-layer 10%. This is a retrospective demonstration, not a newly selected or independently validated model. No new training or MPS benchmark was run for this software change. See the configuration example in README.md and `runs/retention-mode-validation.json`.

## Customer-support experiment (November 13, 2026)

Implemented and ran document-grounded reply generation using [Doc2Dial v1.0.1](https://doc2dial.github.io/data.html). These are crowdsourced public-service conversations about DMV, SSA, VA and student aid. The [IBM dataset card](https://huggingface.co/datasets/IBM/doc2dial) lists CC-BY-3.0. This is a useful support-writing benchmark, with supplied relevant documentation; it does not represent real company helpdesk tickets or test retrieval and autonomous ticket resolution.

**Observed result:** approximately 10% compression incurs a small token-F1 loss against the equally trained full model. At 20%, the loss is larger and a separate exploratory reply review found more omissions. No tested pruned model qualified for the zero-loss development tier. This does not establish preserved customer-support accuracy or that per-layer pruning beats uniform pruning.

### Held-out results

All models used Qwen2.5-0.5B-Instruct, float32, batch size one and the actual M5 Pro MPS GPU. The same 300 held-out conversations cover 33 document families. F1 and ROUGE-L below are normalized word-overlap scores on a 0–100 scale, **not percentages of correct or helpful replies**.

| Model | Parameters | Removed | Development F1 | Test F1 | Test ROUGE-L | Test F1 change vs recovered full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original Qwen | 494,032,768 | 0% | 22.31 | 25.26 | 20.91 | −28.46 |
| Recovery only | 494,032,768 | 0% | 45.31 | 53.72 | 51.00 | — |
| Per-layer 10% + recovery | 444,659,584 | 9.994% | 43.32 | 52.47 | 49.72 | −1.24 |
| Uniform 10% + recovery | 444,487,552 | 10.029% | 41.87 | 51.94 | 48.91 | −1.77 |
| Per-layer 20% + recovery | 395,286,400 | 19.988% | 39.14 | 46.19 | 43.69 | −7.52 |

Selection was frozen before opening the test set and compares against **recovered full Qwen**, rather than the weaker original. The zero-point tier has no qualifying pruned model. The five-point tier selected uniform 10%; the ten-point tier selected per-layer 20%. The five-point choice follows the minimum-parameter rule: uniform has 172,032 fewer parameters than per-layer 10%, despite the latter's higher development F1. No test result changed these choices. All five variants were declared for testing beforehand.

Both selected tiers meet their observed test-score thresholds. Document-cluster bootstrap 95% descriptive intervals are −1.77 [−3.73, +0.08] F1 points for uniform 10% versus recovered full, and −7.52 [−10.37, −5.04] for per-layer 20%. The latter interval crosses the allowed ten-point loss, so its observed pass is not a confident population guarantee. Per-layer 10% versus uniform 10% is only +0.53 [−2.35, +3.36]; the experiment does not establish superiority of the allocation heuristic. These intervals do not include variability across new training seeds.

### Data separation and training budget

The prepared splits contain 1,000 recovery-training, 150 calibration, 200 development and 300 final-test examples, one target reply per conversation. Whole document families are held apart across splits, grouping shared normalized titles, identical text and near-duplicate documents. The split family counts are 93/16/20/33 respectively. Final-test conversations come only from the official test partition; the other splits use official train/validation conversations. Each domain contributes 75 final-test examples. These are custom subsets, not official leaderboard scores.

Exact overlap checks passed for document groups, documents, conversations, complete supplied source contexts, conversation histories, replies and prompts. Before any model run, an independent exhaustive check found a near-duplicate generic acknowledgment that an initial sampled check missed. The preparer was fixed, the data rebuilt, and all 785,000 cross-split pairs then had zero response/history matches at the specified 0.8 three-word-shingle Jaccard threshold. This lexical check does not prove semantic independence, and public data may have appeared in Qwen's pretraining. Source archive SHA-256 and attribution are retained in `data-provenance.json`.

Prompts use earlier conversation turns and **gold-annotated relevant document sections**. Target and future turns are excluded. Oracle evidence makes response writing easier than retrieving the right policy. References over 64 tokens and combined prompt/reference sequences over 512 tokens are excluded, so long or complex support interactions are underrepresented. Source annotations can be awkward, incomplete or mistaken; reference overlap cannot resolve those problems.

Every recovered variant starts from original Qwen, with rank-8 MLP LoRA, learning rate 0.0002, accumulation four, and 50 optimizer updates. Each actually visits **200 examples from the 1,000-example recovery pool**. Development checks at updates 0/25/50 select the best state; all four selected update 50. Calibration scoring uses 150 examples, while the cheaper layer-sensitivity probe uses 16. Adapters are merged before export. This is one short run per variant, not a search for optimal training or compression.

Original Qwen reached the 64-token generation cap on 54% of test replies; recovered full did so on 10%, per-layer 10% on 9.33%, uniform 10% on 5.67%, and per-layer 20% on 6.67%. Reaching the cap does not necessarily mean truncation because EOS can occur at the final token. Improved concision and reference style contribute to F1 gains, so the improvement over original cannot be attributed to pruning.

### Separate blinded qualitative check

A separate Astra medium LLM agent rated 30 cases, each from a different document family, with all five model identities hidden. It inspected the supplied documentation and conversation, treated references as fallible, and froze its ratings before the identity key was opened. The ordinary human-review worksheet remains blank.

| Model | Replies judged fully supported | Replies judged to fully address request | Replies flagged for invented facts/policies/actions |
| --- | ---: | ---: | ---: |
| Original | 9/30 | 13/30 | 18/30 |
| Recovery only | 25/30 | 21/30 | 3/30 |
| Per-layer 10% | 25/30 | 22/30 | 3/30 |
| Uniform 10% | 22/30 | 20/30 | 8/30 |
| Per-layer 20% | 24/30 | 13/30 | 6/30 |

These are exploratory judgments from one LLM, not human-validated accuracy, a validated overall pass rate, or a basis for reselection. The categories are separate: a supported answer can still omit essential guidance. Several reference replies were also incomplete. Root checked the review-file hash, all case/alias mappings, preserved human blanks, and three examples against the supplied evidence. In those examples, the 20% model omitted additional paperwork guidance, a retirement-stage condition, or part of a renewal instruction. The unmodified ratings and their evidence notes are saved in `llm-blind-review.json`; the unblinded counts are in `llm-review-summary.json`.

### Software and export validation

Independent testing passed **103 CPU unit/integration tests**, including a real tiny-Qwen support campaign, metadata-preserving resume, generic metric selection, grouped comparisons, and preservation of existing human ratings. The updated report integration checks also passed. A real MPS preflight passed before full fitting.

All four real exports passed SHA-256 identity, float32 storage, exact parameter-count and MLP projection-shape checks. Saved development predictions were identical at the selected recovery state, after adapter merge, and after checkpoint reload. A separate final audit recomputed all 1,500 per-example F1/ROUGE-L/exact-match measurements and aggregates, test correspondence, group IDs, frozen selection and seeded bootstrap intervals. Every final run recorded `mps:0`.

The 20% model physically removes 98,746,368 parameters, saving 394,985,472 bytes of float32 tensor storage. These are smaller dense matrices, not zero-valued masks. Heterogeneous per-layer exports require the project's loader. File-size comparisons must keep dtype fixed; the upstream downloaded model uses BF16 whereas these exports use float32.

### Support timing

Each fixed-output benchmark uses the same 32 development prompts, exactly 16 generated tokens per prompt (512 per run), three alternating-order repetitions, one unmeasured warmup per load and synchronized MPS execution. Tokenization and model loading are excluded; prompt processing and generation are included. These forced continuations measure timing, not quality.

| Model | Fixed 16-token run means (ms) | Median run mean | Normal-response median latency | Normal-response tokens per 32 prompts |
| --- | --- | ---: | ---: | ---: |
| Original | 297.38, 297.39, 296.73 | 297.38 ms | 899.67 ms | 1,665 |
| Recovery only | 317.35, 323.42, 296.44 | 317.35 ms | 412.00 ms | 721 |
| Per-layer 10% | 287.43, 289.79, 290.56 | 289.79 ms | 373.43 ms | 686 |
| Uniform 10% | 287.48, 296.24, 290.61 | 290.61 ms | 330.19 ms | 631 |
| Per-layer 20% | 278.10, 286.24, 279.16 | 279.16 ms | 362.63 ms | 738 |

Per-layer 10%, uniform 10% and per-layer 20% had respectively 2.55%, 2.28% and 6.13% lower fixed-output median latency than original. Relative to recovered full, those reductions were 8.68%, 8.43% and 12.03%. Recovered full varied from 296.44 to 323.42 ms despite sharing the original architecture, which cautions against overinterpreting the larger ratios. These are local timing observations with three repetitions, not general speed guarantees. Ordinary response latency also changes with reply length; its larger improvement cannot be assigned to pruning alone. Memory counters are snapshots, not peak measurements.

### Support artifacts

- Configuration: `configs/support.json`; prepared data: `datasets/doc2dial_v1/`.
- Frozen selection and full results: `runs/support-v1/selection.json`, `summary.json`, `report.md`.
- Data and fit audits: `independent-data-audit.json`, `independent-fit-verification.json`.
- Result audits and comparisons: `independent-final-verification.json`, `independent-comparisons.json`.
- Four checkpoints: `runs/support-v1/candidates/{per-layer-01,per-layer-02,uniform-10,unpruned-recovery}/checkpoint`.
- Human review: `blind-review.jsonl` and its separate identity key. Exploratory LLM review: `llm-blind-review.json`, `llm-review-summary.json`.
- Timing: `benchmark.json` and `fixed-benchmark/summary.json`, with individual measurements retained.

Use `HF_HUB_OFFLINE=1 .venv/bin/python -m prune_lab campaign --config configs/support.json --stage all --output runs/support-v1` to resume completed stages. Run `HF_HUB_OFFLINE=1 .venv/bin/python scripts/benchmark_campaign_fixed.py --run runs/support-v1 --examples 32` for timing only. New tuning after inspecting these results requires a new held-out set. This support experiment retrains from original Qwen; it does not test transfer of the earlier math-pruned checkpoint.

## Phase 2: per-layer pruning and recovery (November 13, 2026)

Completed on the actual Apple M5 Pro with 24 GiB unified memory, using MPS and float32. Independent testing passed **78 tests**, including a real tiny-Qwen campaign and failure cleanup. The additional fixed-length benchmark integration test also passed. All six real Qwen exports passed parameter/shape checks and produced identical development answers before merge, after merge, and after reload.

**Result:** the selected checkpoint removes 19.988% of total parameters (494,032,768 → 395,286,400) and scores 228/300 (76.00%) on held-out arithmetic, versus 107/300 (35.67%) for the original. However, the unpruned model with the same recovery budget scores 275/300 (91.67%). Compression therefore costs 15.67 percentage points relative to the recovered full model.

| Model | Parameters | Removed | Held-out correct | Accuracy | Fixed 16-token latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original Qwen | 494,032,768 | 0.00% | 107/300 | 35.67% | 269.56 ms |
| Recovery only | 494,032,768 | 0.00% | 275/300 | 91.67% | 302.79 ms |
| Uniform pruning + recovery | 444,487,552 | 10.03% | 262/300 | 87.33% | 230.97 ms |
| Per-layer pruning + recovery | 395,286,400 | 19.99% | 228/300 | 76.00% | 238.04 ms |

Latency is the median of three run means on the same 64 development prompts, exactly 16 generated tokens per prompt, one warmup per load, synchronized MPS execution, and alternating model order. The per-layer 20% export had 11.69% lower median latency than original; uniform 10% had 14.32% lower latency in this run. These are local measurements, not general performance guarantees.

| Model | Fixed-length run means (ms) | Normal-response median latency (ms) | Normal-response tokens per 64 prompts |
| --- | --- | ---: | ---: |
| Original Qwen | 269.56, 245.08, 270.25 | 86.23 | 321 |
| Recovery only | 333.18, 302.79, 270.16 | 64.98 | 239 |
| Uniform pruning + recovery | 230.97, 230.15, 240.81 | 61.11 | 239 |
| Per-layer pruning + recovery | 238.04, 234.95, 242.83 | 59.20 | 244 |

The recovered full model had timing spikes; all repeats are retained. Shorter normal answers explain part of the larger task-latency gain, which is why the equal-token benchmark is reported separately. Memory snapshots are not peaks; RSS and MPS allocation counters must not be added together.

### Selection and training

- Data: 1,000 recovery-training, 150 calibration, 200 development, and 300 final-test questions. Arithmetic operations, signs, magnitudes and plain/word templates are balanced. Duplicate operation/operand keys, commutative swaps and prior exposed arithmetic are excluded.
- Each recovery used rank-8 MLP LoRA, batch size one, accumulation four, and 200 optimizer updates (800 training examples sampled from the 1,000-example split). Development checks at steps 0/50/100/200 chose the best accuracy, then answer loss. Adapters were merged before export.
- Per-layer candidate development scores: 5.014% reduction → 84%; 9.994% → 85%; 15.008% → 83%; 19.988% → 77%. Uniform 10.029% → 86.5%; recovery-only → 91%. Original → 38%.
- The 0/5/10 percentage-point tiers compare against the **original model**, not the recovered full model. All three selected the same 20% checkpoint, the smallest qualifying candidate in the fixed grid. All passed their held-out thresholds. The campaign does not establish the maximum feasible compression.
- Selection was frozen before opening the final test; no final-test outcome changed it. Unselected 5/10/15% per-layer candidates were not evaluated on the final test.

### Critical interpretation

The larger gain comes from recovery training; these results do not show pruning improves mathematical capability. The original had 65/300 unparseable answers, and all recovered outputs parsed, so gains include adherence to the numeric-answer format. This is a narrow signed-arithmetic task with simple word templates, not evidence for broad math or customer-support quality.

The selected model improved 132 answers and regressed on 11 versus original: +40.33 percentage points, with a descriptive paired-bootstrap 95% interval of [+34.00, +46.33]. Against recovered unpruned it regressed on 55 and improved 8. There is no claim of equivalence to the recovered full model.

The per-layer allocator has not demonstrated an advantage over uniform pruning: at approximately the same 10% reduction its development score was 85% versus 86.5% for uniform. The held-out per-layer and uniform models have different reductions, so their test difference cannot isolate allocation method. The 20% candidate scored only 3% immediately after pruning and required recovery to reach 77% development accuracy.

All comparisons use float32 parameter storage. The upstream download uses BF16; comparing its file size directly with these float32 exports would conflate pruning and precision.

### Follow-up data audit: limits of the holdout

The recovery and test sets share the same four surface-template families, arithmetic operations, and operand range (-99 to 99). The test measures new sampled operation/operand combinations from the same generator; it does not hold out templates, difficulty ranges, mathematical skills, or unrelated tasks.

The original duplicate checks found zero repeated training/test operation keys or exact prompts, but did **not** cover all algebraic equivalences. A follow-up check normalizing subtraction to addition and multiplication signs found **25/300** test questions equivalent to questions in the 1,000-example recovery pool; **17/300** overlap the 800 examples actually used for the selected 20% model. An actual example is test `(-19) * (-3)` versus training `3 * 19`, treated as separate problems. Earlier descriptions of fully semantic-disjoint data were too strong. The saved measurements are unchanged; this limits their interpretation.

Removing those 17 questions is only a post-hoc diagnostic, not a fresh test. The remaining data still share the same templates and distribution. See `runs/campaign-v2/data-audit.json` for exact examples and counts. Generalization to different math formats, harder math, customer support, or other capabilities remains untested.

### Artifacts and reproduction

- Selected checkpoint for all three tiers: `runs/campaign-v2/candidates/per-layer-04/checkpoint`.
- Other exports: `runs/campaign-v2/candidates/{per-layer-01,per-layer-02,per-layer-03,unpruned-recovery,uniform-10}/checkpoint`.
- Detailed campaign report: `runs/campaign-v2/report.md`; frozen selection and final measurements: `selection.json`, `summary.json`.
- Independent parameter/shape and reload audit: `runs/campaign-v2/verification.json`; comparative diagnostics: `analysis.json`.
- Timing: `runs/campaign-v2/benchmark.json` and `fixed-benchmark/summary.json`, with individual measurements retained.
- Configuration: `configs/campaign.json`. Use `prune_lab.modeling.load_model` for heterogeneous checkpoints.

```sh
HF_HUB_OFFLINE=1 .venv/bin/python -m prune_lab campaign --config configs/campaign.json --stage all --output runs/campaign-v2
HF_HUB_OFFLINE=1 .venv/bin/python scripts/benchmark_campaign_fixed.py --run runs/campaign-v2
```

The first command resumes completed stages without redoing quality selection. The second reruns timing only. New quality experiments after inspecting these test results need a fresh held-out set.

## Historical version 1 results

Validated on November 13, 2026. This is a functioning pruning prototype, not evidence of a generally better math model.

### Implementation

Python CLI supports device checks, numeric math evaluation, three post-gate neuron scoring methods, a random baseline, reversible pruning search, physical Qwen2 MLP compaction, standard Hugging Face checkpoint export/reload, and JSON/Markdown reports. No optimizer or weight training is implemented. Each layer keeps the same width, with different neurons chosen per layer.

### Environment and tests

- Python 3.13.17, PyTorch 2.15.0, Transformers 4.57.6.
- Two GPT-6 Astra medium coding agents implemented separate modules; another Astra medium agent audited and tested the integrated code. Root reviewed source and performed real checkpoint validation.
- 44 automated tests pass, including independent finite-difference checks of Taylor scores, rollback after errors, padding, numerical guardrails, cached generation, and CLI export/reload integration.
- During initial validation, MPS was unavailable to this execution session. The initial experiments below used CPU, float32, four threads, batch size one. After changing session permissions, the MPS follow-up at the end of this report measured GPU performance.
- Dependencies used for this run are recorded in requirements-tested.txt.

### Fixed comparison

Qwen/Qwen2.5-0.5B-Instruct; 8 calibration / 16 development / 32 test arithmetic examples; seed 17. Each method proposed removing 243 of 4,864 MLP channels per layer (~5%). Original development accuracy was 13/16. No drop was permitted; all four proposals were rejected, leaving the original model selected. The baseline on this first test set was 24/32.

| Scoring method | Candidate development accuracy | Decision |
|---|---:|---|
| activation | 11/16 | Rejected |
| contribution | 10/16 | Rejected |
| taylor | 12/16 | Rejected |
| random | 7/16 | Rejected |

### Adaptive Taylor search

The default Taylor method was then allowed five rounds with smaller steps on rejection. Calibration/development data stayed unchanged. Final evaluation used 64 fresh questions disjoint from the first test set; no final-test results were used for pruning selection. The search accepted removal of 60 channels per layer (4,864 → 4,804), preserving 13/16 development accuracy. This is an exploratory, small-sample follow-up.

Search time: 25.42 seconds, including development evaluation and scoring, excluding model loading and final-test evaluation.

| Metric | Original | Reloaded compact checkpoint |
|---|---:|---:|
| Parameters | 494,032,768 | 490,162,048 |
| Parameter tensor bytes (float32) | 1,976,131,072 | 1,960,648,192 |
| Fresh test accuracy | 39/64 (60.94%) | 39/64 (60.94%) |
| Answer-token loss | 0.400669 | 0.405310 |
| Mean generation latency per example | 126.77 ms | 139.19 ms |
| Generated tokens/second | 29.95 | 27.28 |
| Process RSS snapshot | 2,441,641,984 bytes | 2,451,750,912 bytes |

Parameter reduction: 0.783%. The compact model was 9.8% slower in this single CPU timing run. There is no demonstrated latency or process-memory saving. Timings are descriptive, not a statistically established performance difference.

Equal aggregate accuracy does not mean all questions were preserved: 2 examples changed correctness (one improved and one regressed). The set is too small for broad math-quality conclusions.

### Export audit

- All 290 parameter tensors match the expected original rows/columns (or unchanged tensors) exactly by shape and SHA-256 hash.
- Tokenization is preserved. Masked and reloaded compact models produced identical generated strings on all 64 final-test questions.
- Initial float32 logit audit at atol=rtol=1e-4 found one near-zero logit outside tolerance among 7,444,864 values for a checked example (absolute difference 0.0001053121). This observation is retained rather than silently relaxing tolerance.
- A higher-precision CPU check of three calibration examples passed; maximum logit difference was 0.0. Weights/matrix operations were float64, but Qwen RMSNorm internally uses float32. Exact tensor hashes and matching generated outputs independently support correct export.
- The loader explicitly preserves Qwen tokenization to avoid a false Mistral-regex warning in Transformers 4.57.x for local checkpoints. An independent regression test covers reload behavior.

### Files and reproduction

- Checkpoint: runs/adaptive/compact/
- Search and individual candidate results: runs/adaptive/search.json
- Matched evaluation results: runs/adaptive/baseline-test.json and runs/adaptive/compact-test.json
- Export verification: runs/adaptive/verification.json
- Run config and exact task examples: runs/adaptive/config.json and adjacent JSONL files
- Initial method comparison: runs/validation/

```sh
.venv/bin/python -m pytest -q
.venv/bin/prune-lab doctor --config configs/smoke.json
HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify_run.py --float64 --config runs/adaptive/config.json --result runs/adaptive/search.json --checkpoint runs/adaptive/compact --output runs/adaptive/verification.json
```

The exported checkpoint is float32 (~1.96 GB of weights). The downloaded upstream checkpoint uses BF16, so comparing their raw disk sizes would conflate pruning with precision. The table compares parameter storage at the same float32 precision.

### Assessment

The implementation and structural export checks pass. This run demonstrates modest parameter reduction with unchanged aggregate accuracy on a small arithmetic sample, not a faster or better model. Larger calibration/development/test sets and broader math tasks are required before claiming practical compression benefits. The subsequent MPS measurements below also do not establish a reliable pruning speedup.

### MPS follow-up after enabling Full access

MPS became available when the session's execution permissions changed. A GPU allocation succeeded, confirming the earlier failure was an execution-environment restriction. The benchmark ran with CPU fallback disabled. The launcher was corrected to recognize the valid indexed device name `mps:0`.

Three repeats per model used the same 64 held-out questions, float32, batch size one, 16-token generation cap, one warm-up per evaluation, MPS synchronization, and a fresh process per model. Run order alternated. Every run produced 243 generated tokens, making token counts comparable. The test set was reused only to measure execution on a different device; no new pruning selection occurred.

| Metric | Original | Compact |
|---|---:|---:|
| Accuracy in every repeat | 39/64 | 39/64 |
| Median mean response latency | 62.25 ms | 61.63 ms |
| Mean response latency range | 61.81–63.91 ms | 61.54–63.32 ms |
| Median generation throughput | 61.00 tokens/s | 61.61 tokens/s |
| MPS live allocation snapshot | 2,045,192,192 bytes | 2,007,771,136 bytes |
| MPS driver allocation snapshot | 2,808,872,960 bytes | 2,808,872,960 bytes |

The median latency ratio is 1.0101× (about 1% in favor of compact). Timing ranges overlap; this does not demonstrate a reliable speedup. Live MPS allocations decreased modestly, while driver allocations stayed equal; these snapshots are not peak system memory measurements.

Full results: `runs/mps-benchmark-20261113T183056405898Z/summary.json`, plus all six individual result files. Reproduce with `.venv/bin/python scripts/benchmark_mps.py` in a GPU-enabled execution environment.
