# Improvement 8 plan: layer dropping, alone and combined with MLP pruning

Shared plan for the writer, tester and verifier agents. Branch:
`improve/8-layer-drop` (from `main` at 02d34e3). Written 2026-11-15 and
revised the same day after review; both versions predate any trained candidate.

## Why

The user's goal is speed. In the one timing we have (MLX, 16-bit, batch one,
noisy), uniform 20% MLP pruning generated about 12% faster than the unpruned
model. The working hypothesis is that MLP pruning leaves attention and
per-layer overhead untouched, while removing a whole transformer layer removes
attention, MLP and norms together, so compute should fall roughly in
proportion to depth. This is a hypothesis; the timings below test it.

Questions:

1. **Each layer individually:** which layers matter?
2. **Layer dropping alone:** how does depth reduction compare with width
   reduction (uniform MLP pruning) at equal speed and at equal size?
3. **Combined:** drop a few layers, then prune MLPs uniformly in what remains.

Qwen2.5-0.5B: 24 layers, 14,912,384 parameters each, 494,032,768 total.
References (same recipe, `runs/support-v2-allocator` / `-longer`):
`uniform-20` 394,942,336 parameters, 53.29 F1; `uniform-10` 444,487,552, 54.32.

## Components (all opt-in; with no `layer_drop` key every existing output is unchanged, byte for byte)

**8a — single-layer scan ("each layer individually").**
- On the original pretrained model, remove one layer at a time (24 variants,
  no training) and record the mean answer-token loss on all 150 calibration
  examples, and its increase over the intact model.
- Stored once in the shared allocation record and written to each seed
  directory as `layer-scan.json`. Calibration data only: never dev or test.

**8b — greedy layer selection.**
- To drop k layers: repeat k times — among the remaining layers, remove the
  one whose removal gives the lowest calibration loss, re-measuring after each
  removal (later choices are made in the already-reduced network).
- Ties broken toward the higher layer index. No layer is protected by rule.
- **Noise check.** Without training, the loss after several removals may be
  so high that the choice among remaining layers is arbitrary. For every
  greedy step record the chosen layer, the runner-up, both losses, and the
  paired standard error of their per-example loss difference over the
  calibration examples. A step whose gap is under 2 standard errors is marked
  `"distinguishable": false`. This does not change the selection; the
  verifier reports which steps were not distinguishable.
- A dropped layer is physically removed: `num_hidden_layers` is reduced and
  the export contains only the kept layers.

**8c — candidates.** Config:

```json
"layer_drop": {
  "examples": 150,
  "candidates": [
    {"id": "layers-3", "drop": 3},
    {"id": "layers-3-uniform-20", "drop": 3, "total_target": 0.2},
    {"id": "layers-7", "drop": 7}
  ]
}
```

- `layers-3`: drop 3 layers, no MLP pruning. 449,295,616 parameters (9.05%
  removed). 3 of 24 layers is 12.5% of depth, about the speedup `uniform-20`
  showed, so this is the depth-against-width test at roughly equal speed. It
  is size-matched to `uniform-10`, not `uniform-20`.
- `layers-3-uniform-20`: drop 3 layers, then prune MLPs uniformly in the
  remaining 21 so the **total** reduction reaches 20% (395,105,536; same
  rounding rule as `uniform-20`). Size-matched to `uniform-20`.
  - Order: drop layers first, then compute Taylor scores on the reduced
    network and prune. Scores from the intact network must not be reused.
- `layers-7`: drop 7 layers, no MLP pruning. 389,646,080 (21.13% removed).
  The far end of the depth curve; descriptive only.
- Recovery: one pass with the unchanged 2a recipe (450 updates, checkpoints
  150/300/450) on the final shape. No intermediate training.
- A `layers-N` export must load with the stock Hugging Face class. Combined
  exports load through the project loader.
- Invalid configs fail before any GPU work.

**8d — speed benchmark in the repository** (`scripts/speed_benchmark.py`,
based on the orchestrator's scratch script, which the writer is given).
- Inputs: a list of `name=export_dir` pairs. PyTorch on MPS, float16 and
  float32. Fixed set of 32 dev prompts (first 32 by key), forced greedy
  decoding for exactly 64 new tokens so every model does the same work.
- Measures per repeat: prompt-reading time per prompt (batch one),
  generation ms per token (batch one), generation tokens per second at batch
  16 (left-padded).
- Replication is by **repeats, not seeds**: layer choices and masks are the
  same for every seed, so every seed's export has the same shape. One seed's
  exports, at least 20 repeats, models interleaved within each repeat in
  rotated order, one warm-up pass per model.
- Output: per-model medians, and for each model against a named reference the
  ratio with a 95% bootstrap interval over repeats.
- Optional MLX mode: 8-bit conversion via the existing quantization code and
  batch-one generation timing with the same prompts and token count.
  Reduced-depth and reduced-depth-plus-pruned exports must convert and pass
  the existing logit gate (new test).

**Config file:** `configs/support-v2-layer-drop.json` — the 2a recipe, the
`layer_drop` block above, and `train_candidates: ["unpruned-recovery",
"layers-3", "layers-3-uniform-20", "layers-7"]`.

## Experiment (run by the orchestrator, not the agents)

- 5 seeds, 20261114–20261118, the same seeds as `runs/support-v2-allocator`,
  which holds `uniform-20` and `unpruned-recovery` for the same recipe.
- Output `runs/support-v2-layer-drop`. Checkpoints are kept for all seeds
  (disk is not a constraint: 1.3 TB free); timing uses seed 20261114 only.
- The re-trained `unpruned-recovery` should reproduce the stored one per seed;
  the verifier checks this as a sanity test of comparability.

## Decision (predeclared)

**Quality.** Token F1 on dev-holdout (450 examples), seeds paired across the
two runs. "Confident" = hierarchical bootstrap and seed t intervals both
exclude zero, same sign.

**Speed.** Primary speed metric: generation tokens per second at batch 16,
PyTorch MPS, float16, as a ratio to `uniform-20`. "Confidently faster" = the
95% interval on the ratio lies wholly above 1; "confidently slower" = wholly
below 1.

**Declared comparisons, each against `uniform-20`:**
1. Primary: `layers-3-uniform-20` (equal size).
2. Second: `layers-3` (roughly equal speed, larger model).

**Pass rule.** The user's original rule (confident F1 gain at matched size)
would almost certainly reject every candidate here: at equal size a
depth-reduced model is expected to be faster, and the realistic best case is
"F1 about equal, clearly faster". So for this improvement a candidate passes
if either holds:

- **(a) Better quality, not slower:** confident F1 gain, and not confidently
  slower.
- **(b) Faster, quality not worse by more than 2 F1:** confidently faster, and
  the lower bounds of both F1 intervals are above −2.0. The margin is 2
  because the reference run could only detect about 2.2 F1 with 5 seeds; a
  tighter margin could not be shown.

A candidate that passes is merged to `main` as an opt-in method with the
result documented. If none passes, the code goes to
`experiment/8-layer-drop`. Two comparisons are declared, with no correction
for that; the write-up says so.

**Secondary, descriptive only:** `layers-7` against everything; `layers-3`
against `uniform-10` (equal size); each candidate against `unpruned-recovery`;
batch-one and prompt-reading times; float32 timings; MLX 8-bit timings (the
deployment point the quantization results recommend); the scan table and
which greedy steps were distinguishable.

**Known limits.** Timings are on an Apple GPU; a data-center GPU with a
serving stack was not available and ratios may differ there. Rule (b) can
pass a candidate that is truly up to 2 F1 worse.

## Roles

- **Writer:** implement 8a–8d and the config on `improve/8-layer-drop` in
  `/Users/aryan/llm_pruning_8`, with unit tests. CPU only.
- **Tester:** independent tests in `/Users/aryan/llm_pruning_test` (branch
  `test/8-layer-drop`), written from this plan, not from the writer's tests:
  defaults unchanged; scan and greedy order correct on a tiny model; noise
  fields correct; dropped layers really absent; parameter counts;
  score-after-drop ordering; stock loading of `layers-N`; config validation;
  resume after interruption; benchmark ratio and interval arithmetic; MLX
  conversion of reduced-depth exports.
- **Verifier:** after the run, recompute scores, differences, parameter
  counts and speed ratios from raw outputs; confirm the scan used calibration
  data only, the recipe matches `uniform-20`'s run, the test split was not
  touched; report greedy steps that were not distinguishable.

Rules for all agents: CPU only; never write to `runs/`, `datasets/` or the
main checkout; local commits only; commit messages end with
`Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`; git needs
`DEVELOPER_DIR=/Library/Developer/CommandLineTools`.
