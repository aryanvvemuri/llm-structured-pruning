# Local Qwen neuron pruning lab

This project tests structured MLP pruning on numeric arithmetic and document-grounded customer-support replies. It supports uniform pruning, different widths per layer, and short LoRA recovery training. It uses Qwen2.5-0.5B-Instruct in float32, with one loaded model at a time, for a Mac with 24 GB unified memory. The full campaign explicitly requires MPS; CPU is available when explicitly selected.

See [RESULTS.md](RESULTS.md) for validation: **222 current software tests**, verified checkpoint exports, and completed M5 Pro experiments. On 300 held-out Doc2Dial support replies, recovered full Qwen scores **53.72 token F1**, per-layer 10% pruning **52.47**, uniform 10% **51.94**, and per-layer 20% **46.19**. These are word-overlap scores, not customer-support accuracy. No pruned candidate qualified for the zero-drop development tier, and per-layer allocation has not established an advantage over uniform pruning. Earlier arithmetic results and their data-overlap limitations remain in the report.

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

Two opt-in campaign options control which candidates exist: `uniform_per_target: true` adds a uniform control `uniform-<pct>` for every target (besides `uniform-10`), and `train_candidates: [...]` restricts allocation and training to the listed names (it must include the selection reference). Without them the candidate list is unchanged.

**Allocation advice (5 seeds, 20% pruning, dev-holdout):** use uniform allocation. The legacy per-layer allocator was the worst option, 0.51 F1 below uniform and 2.11 below recovery only (confident). Two candidate replacements did not establish a gain over uniform: a curve-based allocator (+1.29, consistent across seeds but not across document families) and a FLAP-style global-standardized one (+0.11). Iterative 10%→20% pruning was 0.81 below one-shot uniform. Both are rejected; see RESULTS.md and branches `experiment/1-allocator` and `experiment/7-iterative`.

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

### Larger v2 support data and multi-seed development comparisons

`datasets/doc2dial_v2` is built from the same pinned archive with the same guarantees (family-disjoint splits, final test only from official test dialogues, one target per dialogue, collision and near-duplicate drops, exhaustive near-match audit, manifest provenance). Default preparation arguments still rebuild `doc2dial_v1` byte-for-byte; the new options only add manifest fields when used.

```sh
HF_HUB_OFFLINE=1 .venv/bin/python scripts/prepare_support_data.py --output datasets/doc2dial_v2 \
  --recovery 1800 --calibration 150 --dev 600 --test 224 \
  --exclude-dialogues-from datasets/doc2dial_v1/test.jsonl --family-cap dev=15 \
  --domain-quotas calibration=balanced --domain-quotas recovery=balanced \
  --family-order dev=random --family-order calibration=random
```

Sizes are 1,800 recovery / 150 calibration / 600 development / 224 final test, from 191/20/65/105 document families. All 300 inspected v1 test dialogues are excluded **by dialogue id**. That leaves 356 eligible official-test dialogues, only 56 of them student aid, so an equal-domain test tops out at 224. Any larger test would also use up most of the SSA training families: test families cannot appear in training. Recovery is therefore below 2,000, with SSA (173) and student aid (186) underrepresented. Development families are sampled at random with at most 15 examples each, which spreads development examples over more families than the size-first v1 order (20 families). The exhaustive audit checks 2,011,200 cross-split pairs in about 2 s with zero matches. Exclusion by dialogue id is not full content disjointness from the inspected v1 test. One v2 test example has the same canonical prompt as a v1 test example. Four distinct v2 test responses are near-duplicates (3-shingle Jaccard ≥0.8) of v1 test responses: seven pairs in all, three of them exact copies. Three v2 development families also have different dialogues in the v1 test. Separately, the v2 test shares 46 document families with v1 training splits, and v2 development reuses some v1 training dialogues. This is harmless for models retrained on v2, but **v1 checkpoints must not be evaluated on v2 development or test data**.

> **Rule:** no method may reuse v1 checkpoints, adapters, distillation targets or anything else trained on `datasets/doc2dial_v1`. 186 v2 development dialogues come from v1 training splits, so such reuse leaks development data. Every method starts from the original Qwen model and trains on v2 recovery data only.

`configs/support-v2.json` mirrors `configs/support.json` on this data. `multi-seed run` runs preflight and fit once per recovery seed, in `runs/<name>/seed-<s>/`. Only `recovery.seed` changes, which sets LoRA initialization and training order (generation is greedy). The first seed computes the original-model baseline, calibration scores and pruning masks; later seeds copy them byte-for-byte. The test split is not evaluated unless `--open-test` runs `finalize` for each seed. `--discard-checkpoints` deletes each seed's exported `*.safetensors` (about 1.7 GB each) after its evaluations are saved and the hash is checked. Saved per-example predictions are kept for aggregation, but `finalize`/`benchmark` can then no longer run for that seed. Runs resume, and rerunning with more seeds extends a run.

```sh
HF_HUB_OFFLINE=1 .venv/bin/python -m prune_lab multi-seed run --config configs/support-v2.json --output runs/support-v2-main --num-seeds 3 --discard-checkpoints
.venv/bin/python -m prune_lab multi-seed aggregate --run runs/support-v2-main
.venv/bin/python -m prune_lab multi-seed compare --baseline runs/support-v2-main --candidate runs/support-v2-new
```

`aggregate` writes `multi-seed-summary.json` and `multi-seed-report.md`: per-variant per-seed, mean and sd development F1, ROUGE-L and answer loss, parameter counts, and paired differences for every variant pair. `compare` contrasts the same variants across two run directories (`--pair A:B` chooses others), writing `compare-vs-<baseline>.json/.md` in the candidate run. Intervals are 95% percentile hierarchical bootstraps that resample recovery seeds and whole development document families, with examples paired by id. Seeds are paired within one run. Pairing by seed is also the documented protocol for run-versus-run decisions: `compare` pairs a variant pair whenever both runs have identical seed sets, and resamples independently otherwise. `--paired-seeds` forces pairing; `--unpaired-seeds` forces independence. The report states which was used. The same seed value fixes LoRA initialization and training-data order, so pairing reflects shared randomness only; it does not mean the runs are otherwise matched. Intervals cover seed-to-seed training variation and the sampling of document families. They do not cover different data splits, mask/calibration variation, hyperparameter or hardware changes, multiple comparisons, or optimism from repeated development decisions. With few seeds the percentile bootstrap alone is too narrow. In seed-dominated simulations it rejected a true null 27% of the time with 3 seeds and 17% with 5. Each comparison therefore also reports a seed-level Student-t interval over per-seed mean differences: paired t with n−1 df within a run or with `--paired-seeds`, Welch otherwise. It also reports one `decision`, which is "confident" only when both intervals exclude zero with the same sign, and "not established" otherwise. Reports lead with that decision and warn when fewer than 5 seeds were run. Under the same simulations the combined decision's false-positive rate was 4–5% when seed variation dominates, and 6–7% (3/5 seeds) when family and seed variation are mixed. Use `decision` as the acceptance rule. Accept or reject each improvement on **one predeclared pair** (for example new `per-layer-01` versus baseline `per-layer-01`), never by scanning the all-pairs table.

Independent verification estimated these operating characteristics for 5 seeds on the 450-example dev-holdout (64 document families), using the variances observed in the 5-seed baseline `runs/support-v2-main` (seed-to-seed sd of a variant's mean F1 is 2–3.3 points with the 50-update recovery; sd of paired per-seed differences 1.8–2.4):
- False positives are about 5%.
- Paired-by-seed power at a true 2-point F1 difference is about 33%, at 3 points about 60%, at 4 points about 80%.
- The realistic minimum detectable effect is therefore about 4 points paired, and about 6 points unpaired. Pairing by seed is essential; reaching 2 points would need roughly 10–12 seeds or lower training variance.

Gains of 1–2 points will usually come out "not established", which is not evidence of no effect. An earlier estimate of 2.5–3 points assumed a seed sd of 0.65–1.67 and was too optimistic.

When a selection subset exists, `aggregate` and `compare` compute everything twice. **dev-holdout** (development examples outside the selection subset, 450 on v2) gives the headline numbers and the `decision`. **dev-full** is reported as secondary. `compare` excludes the union of both runs' subsets. Optimism from the subset can differ between methods with different checkpoint schedules or learning curves. Different recovery schedules (`checkpoints`/`max_steps`) are fine for dev-holdout, and both schedules are recorded. `compare` refuses a dev-full decision between runs with different schedules unless `--allow-different-schedules` is given, and that override is recorded. Runs without a subset (selection on all of dev) need identical schedules for any dev decision. Answer loss gets only seed means, because per-example losses are not stored.

Two opt-in campaign options, both enabled in `configs/support-v2.json` and absent from v1 configurations (whose behaviour and fingerprints are unchanged), reduce evaluation cost:

- `selection_dev_examples: 150` uses a fixed, domain-stratified subset of 150 development examples for in-training checkpoint selection at updates 0/25/50. The subset depends only on the development data and the top-level `seed`, never the recovery seed, so it is identical for every candidate, seed and run. Its ids are saved in `selection-dev.json`, and its size and hash in each candidate record.
- `skip_full_merged_dev: true` drops the full after-merge evaluation. The merged model is evaluated only on the selection subset, and its predictions are compared with the full after-reload evaluation (`merge_check`).

`eval_batch_size: 16` (also in `configs/support-v2.json`; default 1 keeps the original batch-one path) batches reply generation in every support evaluation of a campaign: baseline, selection checks, merge check, after-reload dev and final test. The latency `benchmark` stage stays batch-one. Prompts are sorted by length and key and left-padded with an attention mask. Decoding is the same greedy setup, and each reply is cut after its own EOS. Answer-token loss stays batch-one, so it is numerically unchanged, and per-reply `seconds` becomes batch time divided by batch size. On MPS with original Qwen and 96 v2 development prompts, batch sizes 4, 8 and 16 gave replies byte-identical to batch one, at 0.33, 0.25 and 0.15 s per reply instead of 0.85 s. Batching can still change a reply through floating-point differences. Qwen's generation config also applies `repetition_penalty` 1.1 under greedy decoding, and that penalty also scores left-pad tokens in batched rows, a theoretical source of batch-size dependence. A baseline and every run compared with it must therefore use the same setting. Seeds of one multi-seed run must share the whole configuration, and `aggregate`/`compare` refuse mixed `eval_batch_size`, `max_length`, `max_new_tokens`, `dtype` or `device`.

The original-model baseline and every candidate's reported dev result are still full 600-example evaluations of the reloaded export; aggregation uses only these and rejects incomplete records. **The selection subset is part of that reported development set**, so reported development scores are slightly optimistic about choosing checkpoint updates; the multi-seed report states this. Per seed, the cost is about 5,400 development generations (600 baseline + 4 candidates × (3 × 150 selection + 150 merge check + 600 reload)) instead of 12,600. Later seeds reuse the baseline, so they need about 4,800. Preflight's 16 short checks and one warmup per evaluation come on top.

### Improvement 2 recovery recipes (opt-in)

See `plans/improvement-2-recovery.md`. Every option lives under `recovery` and is absent by default, so existing configurations, fingerprints and training are unchanged.

**Accepted: 2a.** On dev-holdout with 5 seeds, the predeclared uniform-10% comparison against `runs/support-v2-main` improved token F1 from 47.05 to 54.32 (+7.26; bootstrap [+4.07, +10.51], paired t [+2.66, +11.87]; confident). Seed sd fell from 2–3.3 to 0.6–1.05 points. The recipe is a bundle of changes, so the gain is not attributed to any single part, and training cost is about doubled. See RESULTS.md. `runs/support-v2-longer` with `configs/support-v2-longer.json` is now the reference for later improvements.

**Rejected: 2b and 2c.** Distillation from the recovered full model (uniform-10% +0.03 F1, gains above about 1.7 ruled out) and attention LoRA (+0.29, gains above about 1 F1 unlikely) were not established on dev-holdout over 5 seeds against `runs/support-v2-longer`. Their code is on branch `experiment/2-distill-attn`; see RESULTS.md. Recovery on MPS empties the allocator cache after every update and assessment, which keeps peak driver memory near 8 GB instead of 16–19 GB, with identical numerics.

- **2a, longer schedule** (`configs/support-v2-longer.json`): one full pass over the 1,800 recovery examples, with `max_steps: 450`, `checkpoints: [150, 300, 450]` and `lr_schedule: "cosine"`. The rate is linear warmup `(u+1)/warmup_steps` over 20 updates, then a cosine from the 2e-4 peak, reaching `min_lr_ratio` (10%) after the last update. The default `"constant"` never touches the optimizer's rate.
- **Recorded recipe:** each candidate record stores `recovery.recipe`, the rate of every update (`learning_rates`), the rate in force at each assessed checkpoint, and `training_seconds`/`assessment_seconds`. `multi-seed compare` records both runs' recipes.
- **Per-example losses** (`store_example_losses: true`, on in the improvement-2 configs): dev records gain per-example `answer_token_loss`/`answer_tokens`, so loss can be bootstrapped later. Nothing else in the record changes.

### Improvement 4: 4-bit quantization baselines (MLX, optional)

Verified on 5 seeds; see RESULTS.md. 8-bit MLX keeps dev-holdout F1 (54.91 against PyTorch 54.89) at 525 MB, so it beats 20% pruning on both size and quality. 4-bit costs a confident 4.3–4.6 F1. Install the optional extra with `uv pip install --python .venv/bin/python -e '.[quantize]'`, which pins `mlx-lm==0.29.1`. Then run `prune-lab quantize --source <run with kept exports> --output <dir> --variants ...`.

See `plans/improvement-4-quantization.md`. `prune-lab quantize` needs the optional `mlx-lm` package. Install it with `uv pip install --python .venv/bin/python 'mlx-lm==0.29.1'`; newer mlx-lm requires transformers 5. `prune_lab` still imports without it.

- **Conversion:** float32 exports become MLX `fp16`, `8bit` or `4bit` (mlx-lm affine group quantization, group size 64). Per-layer MLP widths come from `prune_lab_layer_widths`.
- **Logit check:** the converted model's logits are compared with the PyTorch export on calibration prompts, with a margin-aware gate. The next-token argmax must agree only where the reference top-2 margin exceeds `--margin-tolerance` (0.1), because near-ties flip under any fp16 arithmetic, PyTorch's included. Pooled position argmax agreement must be at least `--min-position-agreement` (0.97), and the maximum absolute logit difference at most `--max-logit-difference` (2.0). Margins and thresholds are recorded. A failed gate is recorded, the remaining work still runs, and the command then exits non-zero.
- **Evaluation:** dev is evaluated with the PyTorch path's prompt template, greedy decoding, `max_new_tokens`, EOS ids and repetition penalty, and its scoring code. Weight bytes, MLX peak memory and tokens/s (32 dev prompts, 3 repeats, after a warm-up) are recorded.
- **Output:** each `<output>/seed-<s>/` is campaign-shaped. Candidates are `<variant>-<precision>` plus the source records as `<variant>-pytorch`, so `prune-lab multi-seed aggregate --run <output>` reports everything on dev-holdout.
- **Limits:** MLX speed is only comparable with MLX speed. The test split is never read. The `<variant>-pytorch` records come from batched PyTorch evaluation (`eval_batch_size` 16). There, transformers' repetition penalty also penalizes the left-pad token (Qwen2.5's `<|endoftext|>`, an EOS id), while the MLX loop reproduces batch-one greedy decoding. The fp16 MLX − PyTorch row can therefore include this batching effect as well as MLX numerics.

```sh
HF_HUB_OFFLINE=1 .venv/bin/python -m prune_lab quantize --source runs/support-v2-allocator --output runs/support-v2-quant \
  --variants unpruned-recovery uniform-20 --precisions fp16 8bit 4bit
.venv/bin/python -m prune_lab multi-seed aggregate --run runs/support-v2-quant
```

**Vocabulary trimming (improvement 6) was rejected.** Trimming the embedding to the 7,764 tokens of the training splits matched 20% MLP pruning (+0.03 F1, not established). It cost 1.57 F1 against the untrimmed model, because replies copy document words whose tokens were removed. See RESULTS.md and branch `experiment/6-vocabulary`.

### Improvement 5: pruned Qwen2.5-1.5B against native 0.5B

**Outcome: no gain; use the native 0.5B.** On dev-holdout over 5 seeds, recovered 1.5B scores 53.92 against the 0.5B's 54.89, even unpruned. Uniform 50% pruning (775.8M parameters, 1.57× the 0.5B) scores 52.89, a difference of −2.00 that is not established. The 1.5B tolerates pruning well (−1.03 F1 at 50%), but it starts no better. The recipe was not re-tuned for the 1.5B; see RESULTS.md. The configuration, preflight script and cross-model compare remain as infrastructure.

See `plans/improvement-5-larger-model.md`. `configs/support-v2-1p5b.json` is the 2a recipe and v2 data with `Qwen/Qwen2.5-1.5B-Instruct`. It trains `unpruned-recovery`, `uniform-25` and `uniform-50`; the tokenizer is byte-identical to 0.5B's. The recipe values are the 0.5B ones, fixed in advance and not tuned for 1.5B.

Exact sizes: original 1,543,714,304 parameters, `uniform-25` 1,155,610,112 (25.14% removed), and `uniform-50` 775,763,456 (49.75%). MLP-only pruning with a 25% floor removes at most 867,041,280 (56.17%), leaving 676,673,024. That is still 37% larger than the 494,032,768-parameter 0.5B model, so a size-matched comparison is impossible here.

`multi-seed compare` refuses runs of different base models unless `--allow-different-models` is given; seeds are then unpaired by default. `scripts/preflight_model.py` measures seconds per update and per reply, and peak MPS driver memory, on recovery data only. It also applies the plan's seed-count rule.

### Improvement 8: layer dropping (opt-in, under evaluation)

See `plans/improvement-8-layer-drop.md`. Configure it with a `layer_drop` block, for example `{"examples": 150, "candidates": [{"id": "layers-7", "drop": 7}, {"id": "layers-3-uniform-20", "drop": 3, "total_target": 0.2}]}`.

- **Scan and selection:** the allocation stage scans single-layer removals on the original model, measuring calibration loss only. It then greedily removes, k times, the layer whose removal gives the lowest loss in the already-reduced network, breaking ties toward the higher index. The scan and order go into `allocation.json` and `layer-scan.json`.
- **Physical removal:** dropped layers are removed from the model, and `num_hidden_layers`, `layer_types` and `layer_idx` are renumbered. A `layers-N` export loads with the stock Hugging Face class.
- **Combined candidates:** `total_target` re-scores Taylor importance on the reduced network, then prunes MLPs uniformly so the *total* reduction against the original model reaches the target, using the uniform candidates' nearest-step rounding. Recovery is one pass on the final shape.

On Qwen2.5-0.5B, `layers-7` has 389,646,080 parameters (21.13% removed) and `layers-3-uniform-20` has 395,105,536 (20.02%; MLP width 3904 in 21 layers). `configs/support-v2-layer-drop.json` trains `unpruned-recovery`, `layers-7` and `layers-3-uniform-20` with the 2a recipe.

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
