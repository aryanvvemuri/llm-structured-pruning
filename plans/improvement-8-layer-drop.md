# Improvement 8 plan: layer dropping, alone and combined with MLP pruning

Shared plan for the writer, tester and verifier agents. Branch:
`improve/8-layer-drop` (from `main` at 02d34e3). Written 2026-11-15, before any
candidate was trained.

## Why

MLP neuron pruning removes 20% of parameters but only about 12% of generation
time, because attention and per-layer overhead are untouched. Removing whole
transformer layers removes attention, MLP and norms together, so compute
should fall roughly in proportion to the layers removed. The user's goal is
speed. Two questions:

1. **Layer dropping alone:** which layers matter, and how does a model with
   layers removed compare with uniform MLP pruning at the same size?
2. **Combined:** drop a few layers, then prune MLPs uniformly in what remains.

Qwen2.5-0.5B: 24 layers, about 14.9M parameters each, 494,032,768 total.
Reference at 20%: `uniform-20`, 394,942,336 parameters, 53.29 F1.

## Components (all opt-in; with no `layer_drop` key every existing output is unchanged, byte for byte)

**8a — single-layer scan ("each layer individually").**
- On the original pretrained model, remove one layer at a time (24 variants,
  no training) and record the mean answer-token loss on all 150 calibration
  examples, and its increase over the intact model.
- Written per seed directory as `layer-scan.json`; deterministic, so identical
  across seeds. Calibration data only: never dev or test.

**8b — greedy layer selection.**
- To drop k layers: repeat k times — among the remaining layers, remove the
  one whose removal gives the lowest calibration loss, re-measuring after each
  removal (later choices are made in the already-reduced network).
- Ties broken toward the higher layer index. No layer is protected by rule;
  the record shows what was chosen. The order and the loss after each step go
  in `layer-scan.json`.
- A dropped layer is physically removed: `num_hidden_layers` is reduced and
  the export contains only the kept layers.

**8c — candidates.** Config:

```json
"layer_drop": {
  "examples": 150,
  "candidates": [
    {"id": "layers-7", "drop": 7},
    {"id": "layers-3-uniform-20", "drop": 3, "total_target": 0.2}
  ]
}
```

- `layers-7`: drop 7 layers, no MLP pruning. Expected about 389.6M parameters
  (21.1% removed), the closest layer count at or below `uniform-20`'s size.
- `layers-3-uniform-20`: drop 3 layers (about 9%), then prune MLPs uniformly
  in the remaining 21 layers so the **total** reduction against the original
  model reaches 20% (same group size and rounding rules as `uniform-20`).
  - Order: drop layers first, then compute Taylor scores on the reduced
    network and prune. Scores from the intact network must not be reused.
  - `total_target` must exceed what the dropped layers already remove;
    otherwise fail before any work.
- Recovery: one pass with the unchanged 2a recipe (450 updates, checkpoints
  150/300/450) on the final shape. No intermediate training.
- Works with `train_candidates`. Candidate records include dropped layer
  indices, kept layer indices, layer widths and parameter counts.
- A `layers-N` export (no MLP pruning) must load with the stock Hugging Face
  class. Combined exports load through the project loader like other pruned
  exports.
- Invalid configs (drop < 1, drop >= number of layers, duplicate ids, id
  clashing with an existing candidate) fail before any GPU work.

**Config file:** `configs/support-v2-layer-drop.json` — the 2a recipe, the
`layer_drop` block above, and
`train_candidates: ["unpruned-recovery", "layers-7", "layers-3-uniform-20"]`.

## Experiment (run by the orchestrator, not the agents)

- 5 seeds, 20261114–20261118, the same seeds as `runs/support-v2-allocator`,
  which holds `uniform-20` and `unpruned-recovery` for the same recipe.
- Output `runs/support-v2-layer-drop`, checkpoints kept for speed measurement.
- The re-trained `unpruned-recovery` should reproduce the stored one per seed;
  the verifier checks this as a sanity test of comparability.

## Decision (predeclared)

Token F1 on dev-holdout (450 examples), seeds paired across the two runs.
"Confident" = hierarchical bootstrap and seed t intervals both exclude zero,
same sign.

- **Primary:** `layers-3-uniform-20` − `uniform-20`.
- **Second declared comparison:** `layers-7` − `uniform-20`.
- Pass rule (the user's): merge to `main` only on a confident F1 gain.
  Otherwise the code goes to `experiment/8-layer-drop`.

**Speed (predeclared, because speed is the user's stated goal).** For each
candidate and `uniform-20`, all 5 seeds, PyTorch on MPS: generation time per
token at batch one, batched throughput at 16, and prompt-reading time, using
the existing forced-length benchmark script. A candidate that is faster than
`uniform-20` on every seed but whose F1 difference is "not established" does
**not** pass automatically. It is reported to the user as a speed/quality
trade for them to decide.

Secondary, descriptive only: each candidate minus `unpruned-recovery`;
`layers-7` − `layers-3-uniform-20`; which layers were dropped; scan table.

## Roles

- **Writer:** implement 8a–8c and the config on `improve/8-layer-drop` in
  `/Users/aryan/llm_pruning_8`, with unit tests. CPU only.
- **Tester:** independent tests in `/Users/aryan/llm_pruning_test` (branch
  `test/8-layer-drop`), written from this plan, not from the writer's tests:
  defaults unchanged; scan and greedy order correct on a tiny model; dropped
  layers really absent; parameter counts; score-after-drop ordering; stock
  loading of `layers-N`; config validation; resume after interruption.
- **Verifier:** after the run, recompute scores, differences and parameter
  counts from raw outputs; confirm the scan used calibration data only, the
  recipe matches `uniform-20`'s run, and the test split was not touched.

Rules for all agents: CPU only; never write to `runs/`, `datasets/` or the
main checkout; local commits only; commit messages end with
`Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`; git needs
`DEVELOPER_DIR=/Library/Developer/CommandLineTools`.
