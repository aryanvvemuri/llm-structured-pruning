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

## Improvement 2a: longer recovery with a cosine schedule (accepted)

Candidate `runs/support-v2-longer` (`configs/support-v2-longer.json`) against reference `runs/support-v2-main`. The new recipe trains for 450 optimizer updates instead of 50, which is one full pass over the 1,800 recovery examples. The rate warms up linearly over 20 updates, then follows a cosine from the 2e-4 peak down to 10% of it. Development checks run at updates 0/150/300/450. Data, pruning masks, evaluation settings and the five recovery seeds (20261114–20261118) are unchanged. Numbers are token F1 (0–100 word overlap, not correctness) on **dev-holdout**: the 450 development examples outside the 150-example checkpoint-selection subset. The test split remains sealed.

| Variant | Reference mean F1 | 2a mean F1 | Difference | Decision | Seed sd: reference → 2a |
| --- | ---: | ---: | ---: | --- | --- |
| Uniform 10% (predeclared primary) | 47.05 | 54.32 | +7.26 | confident | 3.23 → 0.59 |
| Recovery only | 48.72 | 54.89 | +6.16 | confident | 3.34 → 0.79 |
| Per-layer 10% | 46.64 | 54.05 | +7.40 | confident | 2.11 → 1.05 |
| Per-layer 20% | 46.10 | 52.78 | +6.68 | confident | 2.39 → 0.69 |

The primary comparison, uniform 10% minus uniform 10% paired by seed, is +7.26. Its hierarchical bootstrap interval is [+4.07, +10.51] and its paired seed-level t interval is [+2.66, +11.87]. Both exclude zero, so the decision is "confident", and 2a is accepted. An unpaired Welch interval is about [+3.3, +11.3]. The other three variants are secondary comparisons, reported but not deciding.

Within the 2a run, against recovery only:

| Comparison | Difference | Decision |
| --- | ---: | --- |
| Uniform 10% | −0.57 | not established |
| Per-layer 10% | −0.84 | not established |
| Per-layer 20% | −2.11 | confident |
| Per-layer 10% minus uniform 10% | −0.27 | not established |

The allocator still shows no established advantage over uniform pruning.

The selected checkpoint was update 300 in 7 of 20 variant-seed cells and update 450 in 13, never 0 or 150. Each candidate took about 9.4 minutes instead of 4.7, about 4 hours per 5-seed run.

Limits:
- **2a is a bundle.** It combines a full epoch, 9× more updates, warmup plus cosine decay, and one extra checkpoint, so the gain cannot be attributed to any single part.
- **Pairing no longer helps.** Under the new recipe the seed-to-seed correlation between variants is about 0.14, so pairing by seed no longer removes much noise.
- **The old detectability estimate does not carry over.** The roughly 4-point minimum detectable effect estimated from the baseline does not apply to comparisons against this run and should be re-estimated.
- These are dev-holdout results only, and lexical F1 is not support correctness.

An independent verifier recomputed all 30,000 per-example rows with 0 mismatches.

**`runs/support-v2-longer` is now the reference run for later improvements.**

## Improvements 2b and 2c: distillation and attention LoRA (rejected)

Both are compared with the accepted 2a run, `runs/support-v2-longer`. Numbers are token F1 on dev-holdout, 5 seeds (20261114–20261118), paired by seed. The test split remains sealed. Each component was tested in one predeclared setting and not tuned, to protect the development protocol. The code stays on branch `experiment/2-distill-attn`, not on `main`.

### 2b: distillation from the recovered full model, rejected (no measurable effect)

Run `runs/support-v2-distill`. Pruned candidates trained on equal parts answer-token cross entropy and temperature-2 KL to the same seed's recovered unpruned model.

| Variant | Difference vs 2a | Bootstrap 95% | Seed t 95% | Decision |
| --- | ---: | --- | --- | --- |
| Uniform 10% (predeclared primary) | +0.025 | [−1.30, +1.26] | [−1.61, +1.66] | not established |
| Per-layer 10% | −0.16 | | | not established |
| Per-layer 20% | −0.15 | | | not established |

Answer-token loss was consistently lower, by 0.013–0.026. That was not a predeclared measure, and it did not turn into better replies. Training took about 20% longer. The minimum detectable effect for uniform 10% is about 2.3 F1, and gains above about 1.7 F1 are ruled out.

### 2c: LoRA on attention q/k/v/o as well as the MLP, rejected (no effect, or a gain below about 1 F1)

Run `runs/support-v2-attn`. It adds 1,081,344 trainable parameters.

| Variant | 2a mean F1 | 2c mean F1 | Difference | Bootstrap 95% | Seed t 95% | Decision |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Uniform 10% (predeclared primary) | 54.32 | 54.61 | +0.29 | [−0.48, +1.06] | [−0.26, +0.83] | not established |
| Recovery only | 54.89 | 55.34 | +0.45 | [−0.45, +1.33] | [−0.29, +1.19] | not established |
| Per-layer 10% | 54.05 | 54.77 | +0.73 | [−0.10, +1.53] | [+0.46, +0.99] | not established (only t excludes 0; secondary) |
| Per-layer 20% | 52.78 | 52.10 | −0.68 | [−2.12, +0.69] | [−2.38, +1.02] | not established |

The minimum detectable effect is about 1.1 F1 at 80% power. Training took about 3% longer, and peak MPS driver memory was 7.4–8.6 GB.

Caveats:
- **Per-layer 20%, seed 20261118:** this seed lost 2.88 F1 because update 150 beat update 450 by 0.05 F1 on the selection subset. That is checkpoint-selection noise.
- **One merge check was 149/150:** the merged-vs-reloaded check for recovery only, seed 20261118, matched on 149 of 150 replies. All reported numbers come from the reloaded model.

### Memory fix (kept on `main`)

These longer runs exposed growing MPS allocator caching: peak driver memory reached about 18 GB with a teacher on the 24 GB machine. Recovery now empties the MPS cache after every update and after each assessment, which brings the peak to 8.2 GB with bit-identical training losses.

## Improvements 1 and 7: allocator and iterative pruning at 20% (curve allocator accepted after a predeclared retest; iterative rejected)

Both ran in one run, `runs/support-v2-allocator`: 5 seeds, the 2a recipe, 20% pruning. Numbers are token F1 on dev-holdout, paired by seed, and the test split remains sealed. An independent verifier recomputed all 21,000 per-example scores with 0 mismatches.

| Variant | Parameters | Mean F1 (seed sd) |
| --- | ---: | ---: |
| Original Qwen | 494,032,768 | 22.05 |
| Recovery only | 494,032,768 | 54.89 (0.79) |
| Uniform 20% | 394,942,336 (20.057% removed) | 53.29 (1.33) |
| Per-layer 20%, legacy allocator | 395,286,400 (19.988%) | 52.78 (0.69) |
| Curve allocator, 20% (1a) | 395,286,400 (19.988%) | 54.58 (0.52) |
| Global-standardized, 20% (1b) | 395,286,400 (19.988%) | 53.40 (0.77) |
| Iterative 10%→20% | 394,942,336 (20.057%) | 52.48 (0.31) |

### Improvement 1: curve and global-standardized allocators, first run (not established, suggestive)

The predeclared primary comparison is curve-20 minus uniform-20: **+1.29**. The bootstrap interval is [−0.23, +2.84] and the seed t interval is [+0.12, +2.46], so the decision is "not established". The per-seed differences were +0.39, +0.53, +1.98, +2.56 and +0.99. The gain is consistent across seeds but not shown across document families (families-only interval [+0.07, +2.50]); the minimum detectable effect is about 2.2 F1.

Secondary comparisons, reported but not deciding:

| Comparison | Difference | Decision |
| --- | ---: | --- |
| Curve − per-layer legacy | +1.80 (bootstrap [+0.59, +3.01], t [+1.36, +2.24]) | confident |
| Curve − recovery only | −0.31 | not established |
| Curve − global-standardized | +1.18 | not established (only t excludes 0) |
| Global-standardized − uniform | +0.11 | not established |
| Per-layer legacy − uniform | −0.51 | not established |
| Per-layer legacy − recovery only | −2.11 | confident |
| Uniform − recovery only | −1.60 | not established |

**Removal patterns**
- **Uniform:** 0.316 of every layer's neurons.
- **Legacy:** 0.632 in 9 layers, 0.618 in 2, and 0 in 13.
- **Curve:** 0.5 in 12 layers, 0.25 in 3, about 0.1 in 3, and 0 in 6.
- **Global-standardized:** 0 to 0.421, sparing the last layers.

The sensitivity probes were dominated by noise: 33 of the 72 raw values were negative, mostly within ±0.03. Calibration loss right after pruning, before recovery, was:

| Allocator | Calibration loss (baseline 1.7833) |
| --- | ---: |
| Curve | 1.7865 (in-sample: the curve was fitted on this data) |
| Uniform | 1.8067 |
| Global-standardized | 1.8713 |
| Legacy | 2.0481 |

**First-run conclusion (superseded by the retest below):** the curve allocator's gain over uniform was not established. The legacy per-layer allocator on `main` was the worst option at 20%. Adding seeds to this run would have been optional stopping, so a fresh predeclared run was used instead. The global-standardized allocator showed no gain and was not retested or landed.

### Improvement 1 retest: curve allocator, accepted (predeclared, 10 fresh seeds)

The retest is predeclared in `plans/improvement-1-allocator-retest.md` (commit ec9e477) and configured in `configs/support-v2-allocator-retest.json`. Run `runs/support-v2-allocator-retest` trained `unpruned-recovery`, `uniform-20` and `curve-20` with the same recipe on ten fresh recovery seeds, 20261119–20261128. The decision uses those ten seeds only. Numbers are dev-holdout token F1, paired by seed, and the test split remains sealed.

| Variant | Parameters | Mean F1 ± seed sd |
| --- | ---: | ---: |
| Recovery only | 494,032,768 | 54.61 ± 0.72 |
| Uniform 20% | 394,942,336 | 53.76 ± 0.55 |
| Curve allocator, 20% | 395,286,400 | 55.09 ± 0.92 |

| Comparison | Difference | Bootstrap 95% | Seed t 95% | Decision |
| --- | ---: | --- | --- | --- |
| Curve − uniform (predeclared primary) | **+1.33** | [+0.14, +2.58] | [+0.64, +2.02] | **confident: accepted** |
| Curve − recovery only | +0.48 | | | not established |
| Uniform − recovery only | −0.85 | | | not established |

**Verification.** An independent verifier recomputed all 27,000 per-example rows with 0 mismatches. It found the primary difference to be +1.33, with bootstrap [+0.12, +2.54] and paired t [+0.64, +2.02]: confident under every bootstrap seed and interval method it tried.

**The acceptance is marginal.**
- The bootstrap lower bound is about +0.12.
- Leaving out seed 20261121 or seed 20261125 makes the result "not established", with bootstrap lower bounds of −0.08 and −0.02. The t interval stays above zero in every leave-one-out case.

**What was replicated.** Only the recovery-seed variation. The retest used the same dev set and document families as the first run, and the same fixed curve masks. The gain is therefore shown for this dev distribution and at 20% pruning only.

**Pooled, descriptive only (15 seeds):** +1.32, bootstrap [+0.17, +2.46], t [+0.81, +1.83]. The first run (+1.29) and the retest (+1.33) agree.

**Size.** `curve-20` has 344,064 more parameters than `uniform-20`, 0.09% more.

**Interruptions.** The run was interrupted and resumed twice, and the verifier found both sound:
- The paused seed, 20261126, ran straight through.
- In the relaunched seed, 20261127, the unfinished candidate was retrained from scratch and reproduced its earlier logged history exactly.
- The predeclaration commit, ec9e477, preceded the first retest output by 45 minutes.

**Checkpoint selection does not favour `curve-20`.** The selection subset happens to flatter `uniform-20`, and all decisions use holdout.

**Recommendation.** At 20% removal, use `allocation: {"methods": ["legacy", "curve"]}` and the `curve-20` candidate. Uniform remains the code default, and it is the right choice when a stock-loadable uniform shape matters: curve exports have per-layer widths and need the project loader. The curve allocator was tested only at 20% on this dataset. Its masks are fixed, and only recovery seeds varied. The global-standardized allocator was not landed; it remains on branch `experiment/1-allocator`.

### Improvement 7: iterative pruning, rejected (no benefit)

The schedule had two stages, 10% then 20%, with 225 + 225 updates. Stage 2 re-scored the merged stage-1 model, and the final widths equal uniform-20's.

The primary comparison, iter-20 minus uniform-20, is **−0.81**, with bootstrap [−2.53, +1.05] and t [−2.71, +1.09]. Per-seed differences were −1.76, −2.14, +0.49, +1.19 and −1.82; the minimum detectable effect is about 2.8 F1. Against recovery only, iter-20 is −2.41 (bootstrap [−4.03, −0.82], t [−3.47, −1.34]), confident.

About 9% of kept neurons differ from the one-shot mask (overlap 0.906–0.911). The stage-2 checkpoint choice was erratic: steps 75, 75, 150 and 225, and step 0 for seed 20261118, which discarded its stage-2 training. The code is on branch `experiment/7-iterative`.

### Run notes

- **Reproduction:** recovery only and per-layer legacy reproduce `runs/support-v2-longer` byte for byte in all 5 seeds.
- **Resume:** the run was stopped and resumed once, inside iter-20's stage 2 for seed 20261115. The replay reproduced the interrupted steps exactly.
- **Lock fix:** a lock-fingerprint bug with integer dict keys was fixed. The 5 selection locks were rewritten only after verifying that no selection content had changed.
- **Unexplained slowdown:** global-standardized trained about 40% slower, with about 9.5 GB peak memory; the cause was not found.

## Improvement 4: quantization baselines (infrastructure, verified)

Run `runs/support-v2-quant` converted the `unpruned-recovery` and `uniform-20` exports of `runs/support-v2-allocator` for all 5 seeds. The tools were MLX 0.32.4 and mlx-lm 0.29.1, with affine group quantization at group size 64. Numbers are dev-holdout token F1, mean ± seed sd, and comparisons are paired by seed. The test split remains sealed. An independent verifier recomputed all 24,000 per-example rows with 0 mismatches.

| Variant | PyTorch fp32 | MLX fp16 | MLX 8-bit | MLX 4-bit |
| --- | ---: | ---: | ---: | ---: |
| Recovery only: F1 | 54.89 ± 0.79 | 54.81 ± 0.77 | 54.91 ± 0.65 | 50.52 ± 1.62 |
| Recovery only: weight MB on disk | 1,976 | 988.1 | 525.0 | 278.1 |
| Uniform 20%: F1 | 53.29 ± 1.33 | 53.20 ± 1.32 | 53.30 ± 1.36 | 48.62 ± 1.29 |
| Uniform 20%: weight MB on disk | 1,580 | 789.9 | 419.8 | 222.3 |

Bits per weight are 16.000, 8.501 and 4.502. All 169 Linear and Embedding modules are quantized, including the tied embedding: 136M parameters, 27.5% of the model, which pruning never touches.

| Comparison (b − a) | Difference | Bootstrap 95% | Seed t 95% | Decision |
| --- | ---: | --- | --- | --- |
| fp16 − PyTorch, recovery only | −0.08 | [−0.22, +0.01] | [−0.14, −0.02] | not established |
| fp16 − PyTorch, uniform 20% | −0.09 | [−0.35, +0.11] | [−0.33, +0.15] | not established |
| 8-bit − fp16 (both variants) | +0.10 | | | not established |
| 4-bit − fp16, recovery only | −4.29 | [−6.27, −2.38] | [−5.79, −2.80] | confident |
| 4-bit − fp16, uniform 20% | −4.59 | | | confident |
| Uniform 20% fp16 − recovery only fp16 | −1.61 | [−3.59, +0.28] | | not established |
| Uniform 20% 4-bit − recovery only 4-bit | −1.90 | [−4.22, +0.41] | | not established |
| Uniform 20% 4-bit − recovery only fp16 | −6.19 | [−8.12, −4.30] | [−7.74, −4.65] | confident |
| Recovery only 8-bit − uniform 20% fp16 | +1.71 | | | not established |
| Recovery only 4-bit − uniform 20% 8-bit | −2.78 | [−5.36, −0.20] | [−5.89, +0.33] | not established |

| Variant | Median tokens/s (range), fp16 / 8-bit / 4-bit | Peak MLX memory GB, fp16 / 8-bit / 4-bit |
| --- | --- | --- |
| Recovery only | 188 (167–190) / 199 (160–220) / 277 (237–283) | 1.73–1.89 / 1.27–1.48 / 1.07–1.26 |
| Uniform 20% | 217 (165–218) / 229 (204–254) / 289 (224–304) | 1.58–1.68 / 1.18–1.30 / 1.01–1.13 |

Speed is MLX against MLX only; it is not comparable with the PyTorch-MPS timings elsewhere.

**Reading**
- **The size/F1 frontier** is, in order of size: recovery only 8-bit (525 MB, 54.91), uniform 20% 8-bit (420 MB, 53.30), recovery only 4-bit (278 MB, 50.52) and uniform 20% 4-bit (222 MB, 48.62).
- **8-bit beats 20% pruning on both size and quality.** It is 34% smaller than the pruned fp16 model and not worse.
- **4-bit is costly here,** a confident 4.3–4.6 F1 on this task.
- **Pruning is only worth considering on top of quantization.** There it saves about 20% of the bytes for −1.6 to −1.9 F1, a cost that is not established.

**Caveats**
- **Batching differs.** The PyTorch records used batch 16 with left padding, and MLX used batch 1. Between 576 and 591 of the 600 replies are identical per seed and variant.
- **The logit gate was amended after the run.** The run exited with code 1 because the original gate failed on a single next-token argmax flip, for seed 20261118, uniform 20%, on one calibration prompt. That prompt is a near-tie: the fp32 top-2 margin is 0.0015, and PyTorch's own fp16 flips it too. MLX's maximum logit difference (0.683) matches PyTorch fp16 against fp32 (0.675). The exit code was a false positive of the gate, and all 30 results were written. The gate now ignores next-token flips where the reference margin is at most 0.1, and keeps the pooled position-agreement and maximum-difference thresholds. The gate is a smoke check against gross conversion errors: a subtle wrong-index bug in a weak layer can stay under its absolute limits. The real evidence that MLX reproduces PyTorch is the end-to-end dev F1 agreement above.

## Improvement 6: vocabulary trimming (rejected)

Run `runs/support-v2-vocab` trimmed the vocabulary of the `unpruned-recovery` and `uniform-20` exports of `runs/support-v2-allocator`, 5 seeds, with no training. Numbers are dev-holdout token F1, mean ± seed sd, paired by seed. The test split remains sealed. An independent verifier recomputed all 12,000 per-example rows with 0 mismatches.

The kept set has 7,764 of 151,936 rows. It was built only from the recovery and calibration splits, plus special, added and all 256 byte tokens. 966 token ids used in dev fall outside it.

| Variant | Parameters | F1 |
| --- | ---: | ---: |
| Recovery only | 494,032,768 | 54.89 ± 0.79 |
| Uniform 20% | 394,942,336 | 53.29 ± 1.33 |
| Trimmed recovery only (`vocab-unpruned`) | 364,854,656 | 53.32 ± 0.77 |
| Trimmed uniform 20% (`vocab-uniform-20`) | 265,764,224 | 51.97 ± 1.14 |

The predeclared primary comparison is `vocab-unpruned` minus `uniform-20`: **+0.03**, bootstrap [−1.86, +2.00], t [−2.04, +2.10], not established. **Rejected:** trimming matches 20% MLP pruning at 7.6% fewer parameters, but is not confidently better.

| Secondary comparison | Difference | Bootstrap 95% | Seed t 95% | Decision |
| --- | ---: | --- | --- | --- |
| Trimmed recovery only − recovery only | −1.57 | [−2.42, −0.73] | [−2.11, −1.02] | confident (the plan expected about 0) |
| Trimmed uniform 20% − uniform 20% | −1.32 | [−2.19, −0.51] | [−1.90, −0.75] | confident |
| Trimmed uniform 20% − recovery only | −2.92 | [−4.93, −0.99] | [−4.88, −0.96] | confident |

**Where the cost comes from.** The 450 holdout examples split by two questions: does the prompt need the fallback, and did the original reply use a removed token?

| Group | Examples | Identical replies | ΔF1 | Contribution to the mean |
| --- | ---: | ---: | ---: | ---: |
| A: neither | 112 | 100% | 0.00 | 0.00 |
| B: reply only | 13 | | +0.01 | |
| C: prompt only | 185 | 70% | +0.18 | +0.08 |
| D: both | 140 | 11% | −5.29 | −1.64 |

- **Exactness holds:** group A's replies are 100% identical, so there is no remapping or padding bug.
- **Split prompts alone are harmless:** group C slightly gains.
- **The whole cost is in group D:** replies that copy document words whose tokens were trimmed. 81% of the removed tokens used in those replies appear in the prompt. The dev documents come from families absent from recovery and calibration, and the trimmed model reproduced only 36% of such words.

Fallback statistics: 1.62% of dev prompt tokens need the fallback, affecting 73.8% of dev examples; 2.61% of reference tokens (in 223 examples) cannot be emitted as one token. Answer loss is not comparable between trimmed and untrimmed models: the softmax over 7,764 tokens lowers it, and re-split references raise it.

**Characterised only, not tuned:**
- Also keeping all ids below 30,000 would cover 59% of the removed-token uses and still save 107.4M parameters (against 129.2M now).
- Keeping all ids below 50,000 would cover 78% and save 90.4M.

What would actually work is keeping every token of the deployment's document collection. That is a domain-coverage limitation, not something the training splits can supply.

**Caveats**
- Trimmed exports are usable only through `load_trimmed`/`TrimmedTokenizer`.
- The recorded exactness check covered only 4 calibration prompts; exactness on dev rests on group A.

The code is on branch `experiment/6-vocabulary`.

**Speed (unaudited side measurement).** A scratch benchmark timed the trimmed exports on one Apple GPU (PyTorch float16, batch 16, 64 forced tokens, 32 prompts, 5 seeds × 3 repeats; output in the untracked `runs/support-v2-speed/speed.json`). Against the unpruned model, batched tokens/s rose by 5.7% [3.8, 7.6] for `uniform-20`, 8.1% [5.8, 10.6] for `vocab-unpruned` and 15.7% [11.5, 21.2] for `vocab-uniform-20` (265,764,224 parameters). This run used its own script, not `scripts/speed_benchmark.py`, and was not checked by the verifier, so it is not comparable repeat-for-repeat with the 50-repeat timings under Improvement 8. It does not change the decision: the speed comes with a confident F1 loss.

## Improvement 5: pruned Qwen2.5-1.5B against native 0.5B (no gain; 0.5B recommended)

Run `runs/support-v2-1p5b` (`configs/support-v2-1p5b.json`) used the 2a recipe and v2 data with `Qwen/Qwen2.5-1.5B-Instruct`, 5 seeds. Numbers are token F1 on dev-holdout (450 examples), and the test split remains sealed. The reference is recovered unpruned 0.5B from `runs/support-v2-longer`. The two runs use different base models, so cross-model comparisons resample seeds independently and use Welch t (`--allow-different-models --unpaired-seeds`).

| Model | Parameters | F1 mean ± seed sd |
| --- | ---: | ---: |
| 0.5B recovery only (reference) | 494,032,768 | 54.89 ± 0.79 |
| 1.5B original (no training) | 1,543,714,304 | 27.59 |
| 1.5B recovery only | 1,543,714,304 | 53.92 ± 0.36 |
| 1.5B uniform 25% | 1,155,610,112 | 53.58 ± 0.27 |
| 1.5B uniform 50% | 775,763,456 | 52.89 ± 1.00 |

| Comparison | Difference | Bootstrap 95% | Seed t 95% | Decision |
| --- | ---: | --- | --- | --- |
| 1.5B uniform 50% − 0.5B recovery only (predeclared primary) | −2.00 | [−4.16, +0.10] | [−3.33, −0.67] | not established |
| 1.5B uniform 25% − 0.5B recovery only | −1.31 | [−3.21, +0.68] | [−2.28, −0.34] | not established |
| 1.5B recovery only − 0.5B recovery only | −0.97 | [−3.01, +1.08] | [−1.94, +0.00] | not established |
| 1.5B uniform 25% − 1.5B recovery only | −0.34 | [−1.66, +1.04] | [−0.75, +0.07] | not established |
| 1.5B uniform 50% − 1.5B recovery only | −1.03 | [−2.83, +0.86] | [−1.96, −0.10] | not established |

On the secondary dev-full set, the primary difference is −2.03 and confident.

**Finding.** Under this recipe the 1.5B model gives no gain over the 0.5B on this task, even unpruned. Pruning it therefore cannot beat the native 0.5B. At 50% removed it is still 1.57× the 0.5B's size and scores about 2 F1 lower: the direction is negative, though not established on holdout. The 1.5B tolerates pruning well, since removing half of its parameters costs about 1 F1. That does not help, because its starting point is no better. **Recommendation: use the native 0.5B.** The 1.5B code path (`configs/support-v2-1p5b.json`, `scripts/preflight_model.py`, `--allow-different-models`) is kept as working infrastructure.

**Caveats**
- **The recipe was tuned on the 0.5B** and not re-tuned for the 1.5B: same learning rate, updates and rank. The 1.5B may be under-trained at 450 updates: all 5 unpruned runs, and 11 of the 15 trained 1.5B models, selected the last checkpoint. That could favour the 0.5B.
- **F1 and answer loss disagree.** The unpruned 1.5B has lower dev answer loss than the 0.5B (0.942 against 1.031) but slightly lower F1, so the finding is about word overlap only.
- **No size-matched point was run.** Uniform 50% is the deepest cut; MLP-only pruning cannot go below 676.7M parameters.
- **One task and one dataset.**
- **Cross-model comparisons use unpaired seeds.**

**Cost.** The preflight measured 1.54 s per update and 11.2 GB peak MPS driver memory for the unpruned 1.5B. In the full run the peak was 12.1–13.2 GB for unpruned and uniform-25 and about 11 GB for uniform-50. The whole 5-seed run took about 7.5 hours on this Mac.

## Improvement 8: layer dropping (`layers-3-uniform-20` passes rule (b) narrowly; small practical speed benefit)

Run `runs/support-v2-layer-drop` (`configs/support-v2-layer-drop.json`) used the 2a recipe, 5 seeds (20261114–20261118), and checkpoints kept. Quality is token F1 on dev-holdout, paired by seed against `uniform-20` (53.29 ± 1.33) from `runs/support-v2-allocator`. The test split was not touched. An independent verifier recomputed every number below from raw data, and the re-trained `unpruned-recovery` reproduced the stored one exactly.

The plan (`plans/improvement-8-layer-drop.md`) predeclared two comparisons against `uniform-20` and two pass rules:
- **Rule (a):** a confident F1 gain, and not confidently slower.
- **Rule (b):** confidently faster, with both F1 interval lower bounds above −2.0.

| Candidate | Dropped layers | Parameters | F1 | Δ vs uniform-20 | Bootstrap 95% | Seed t 95% | F1 decision |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| `layers-3-uniform-20` (primary) | 3, 18, 19, then MLP width 3904 in 21 layers | 395,105,536 (20.02%) | 53.92 ± 1.25 | +0.63 | [−1.08, +2.41] | [−1.04, +2.31] | not established; both lower bounds > −2.0 |
| `layers-3` (second) | 3, 18, 19 | 449,295,616 (9.05%) | 54.98 ± 0.36 | +1.69 | [+0.05, +3.39] | [+0.43, +2.95] | formally confident; borderline |
| `layers-7` (descriptive) | 3, 11, 12, 13, 18, 19, 22 | 389,646,080 (21.13%) | 50.25 ± 1.18 | −3.04 | | | confident worse |

### Speed: the clean timing run

The quoted timings come from a clean re-run (`work/speed-layer-drop-clean.json`): PyTorch MPS float16, 50 interleaved repeats over 107 contiguous minutes. It showed no drift and half the noise of the first run. It also reproduces an earlier independent measurement: unpruned against uniform-20 is 0.932 here, against 0.946 before. The verifier judges it the trustworthy run.

Ratios are model / `uniform-20`, as a ratio of medians with repeats resampled jointly, 95% interval.

| Candidate | Batch-16 tokens/s (>1 faster) | Batch-one ms/token (>1 slower) | Prompt reading ms/prompt (<1 faster) | Batch-16 median tokens/s |
| --- | --- | --- | --- | ---: |
| `uniform-20` (reference) | — | — | — | 1085 |
| `layers-3-uniform-20` | 1.026 [1.003, 1.037], faster | 1.008 [1.000, 1.015], effectively equal | 0.916 [0.899, 0.930] | 1114 |
| `layers-3` | 0.987 [0.975, 0.997], slower | 1.054 [1.042, 1.069], slower | 0.960 [0.946, 0.985] | 1071 |
| `layers-7` | 1.143 [1.132, 1.152] | 0.952 [0.947, 0.956] | 0.708 [0.702, 0.722] | 1240 |
| `unpruned-recovery` | 0.932 [0.919, 0.941] | 1.078 [1.071, 1.087] | 1.001 | 1012 |

How robust each verdict is:
- **`layers-3-uniform-20`:** faster in 3 of 4 alternative analyses (per-repeat ratio 1.030 [1.016, 1.038]); it won 37 of 50 repeats.
- **`layers-3`:** slower at batch 16 by the tool; the other analyses have upper bounds of 1.000–1.007, and it won 19 of 50 repeats. It is slower at batch one in every analysis.

**First run, noisy context only.** The predeclared decision was made on a first timing run of 20 repeats. There, `layers-3-uniform-20` was 1.110 [1.019, 1.262] at batch 16, `layers-3` 1.053 [0.901, 1.202] and `layers-7` 1.149 [1.067, 1.315]. Single measurements in that run swung by ±15%, and its apparent 11% gain was noise. The clean run confirms the primary's rule (b) pass with a much smaller effect. The first run's float32 block was **invalid**: all models were resident at once, the machine thrashed, and the reference drifted about 6×. Its numbers are not reported.

**MLX 8-bit, batch one, from the first benchmark** (ratios to `uniform-20`, ms/token):

| Candidate | Ratio |
| --- | --- |
| `layers-7` | 0.870–0.875, faster |
| `layers-3-uniform-20` | 0.973 [0.965, 1.002], 1–2% faster |
| `layers-3` | 1.007, not established |
| `curve-20` | 1.022, not established |
| `unpruned-recovery` | 1.107 [1.091, 1.176], slower |

### Decision

- **`layers-3-uniform-20` passes rule (b) in both runs, but narrowly and with a small practical benefit.** It decodes about 3% faster batched, is unchanged at batch one, and reads prompts 8–9% faster. In MLX 8-bit it is about 1–2% faster. Its F1 is not different from `uniform-20` (+0.63, not established).
- **`layers-3` is not reported as a pass.** Its formal rule (a) pass on the first run does not survive:
  - on the clean run the tool finds it confidently slower at batch 16, which breaks "not confidently slower";
  - it is about 5% slower per token at batch one;
  - its F1 gain was already borderline: the bootstrap lower bound is about 0 and changes sign with the bootstrap seed (the verifier's 20,000 resamples gave [−0.01, +3.44]), and leaving out one seed keeps confidence in only 2 of 5 cases;
  - it is a larger model (449M) with no speed benefit over `uniform-20`. Against size-matched `uniform-10` it is +0.66, and against `unpruned-recovery` (54.89) +0.09, both not established. It also chose update 450 in all 5 seeds, so it may be under-trained.
- **`layers-7` is rejected:** it is 14% faster batched and 5% faster at batch one, but −3.04 F1 against `uniform-20` and −4.64 against `unpruned-recovery`, both confident.

**Mechanism (the verifier's explanation).** Decoding here tracks the bytes of weights read. `layers-3` reads about 14% more layer weights than `uniform-20` (21 × 14.9M against 24 × 10.8M), which fewer per-layer launches only partly offset. `layers-3-uniform-20` reads the same bytes as `uniform-20`, through 3 fewer layers.

**Descriptive comparisons**
- `layers-3-uniform-20` − `curve-20`: −0.66, not established.
- `layers-3` − `curve-20`: +0.40, not established.
- `layers-3-uniform-20` − `unpruned-recovery`: −0.96, not established.
- `curve-20` against `uniform-20` (first run): 1.000 [0.916, 1.148] at batch 16.

**Layer scan** (calibration loss, original model)
- **Critical layers:** layers 0 and 1 (+8.75 and +3.80).
- **Layers that help when removed:** removing layer 3, 4, 8, 11 or 12 alone *lowers* the loss.
- **Greedy order:** 3, 18, 19, 22, 13, 12, 11.
- **Noise:** only steps 3 and 6 were distinguishable (gap > 2 standard errors). Two of the three layers dropped by both 3-layer candidates were therefore chosen without a clear margin: 3 against 8, and 18 against 19.

**Caveats**
- **Rule (b) admits a candidate up to 2 F1 worse.**
- **No multiplicity correction:** two comparisons were declared, and none was applied.
- **Answer loss is worse at equal F1:** `layers-3-uniform-20` 1.205 against `uniform-20` 1.135, and `layers-3` 1.141 against `unpruned-recovery` 1.031.
- **The speed results are platform-specific:** one Apple GPU only. The batched gain is about 3% (PyTorch float16), the batch-one change is about 0, and MLX 8-bit is about 1–2%.
- **One dev set.**

**Second clean timing run, with `curve-20` (November 17, 2026).** A further 50-repeat PyTorch float16 run used `unpruned-recovery` as the reference (`work/speed-readme.json`, untracked). Batched tokens/s against unpruned: `uniform-20` 1.060 [1.044, 1.069], `curve-20` 1.067 [1.054, 1.076], `layers-3-uniform-20` 1.104 [1.086, 1.114], `layers-7` 1.221 [1.208, 1.237]. Batch-one time per token: 0.923, 0.926, 0.916 and 0.875. These agree with the first clean run to within about one point, and show the curve allocator costs no speed against uniform pruning. The README speed table and figure use this run.

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
