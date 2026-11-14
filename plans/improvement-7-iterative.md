# Improvement 7 plan: iterative pruning (prune → recover → prune)

Shared plan for the writer, tester and verifier agents. Branch:
`improve/7-iterative` (from `main`, or the latest accepted branch the main
session names).

## Why

Every candidate so far is pruned once and then recovered. Pruning in stages
lets the importance scores be re-measured after the model has adapted, which
is the standard argument for iterative pruning. With the 2a recipe, 20%
one-shot pruning costs about 2 F1 against the unpruned model, so there is
room to see a gain at 20%.

## What to build

An opt-in candidate kind (config key `iterative`, e.g.
`{"id": "iter-20", "stages": [0.10, 0.20], "steps": [225, 225]}`):

1. Stage 1: score neurons (Taylor, calibration split) on the original model,
   prune uniformly per layer to the first target (same per-layer counts as the
   uniform candidate at that target), compact, recover with LoRA for the
   stage's updates, select the best checkpoint on the selection subset as
   usual, merge.
2. Stage 2: **re-score** on the merged stage-1 model (same calibration data),
   prune the remaining neurons uniformly per layer to the final target (same
   final per-layer widths as one-shot `uniform-20`), compact, recover again
   with fresh adapters, select, merge, export.
3. Each stage uses the configured schedule shape over its own updates
   (warmup 20, cosine to 10%, peak 2e-4), a stage-specific seed derived from
   the recovery seed, and continues through the recovery data rather than
   restarting it, so the two stages together see each recovery example once.
4. **Equal budget:** total optimizer updates across stages equals the one-shot
   candidate's (225 + 225 = 450). Record per-stage masks, scores hash, chosen
   checkpoints, and how many neurons kept in stage 2 differ from the one-shot
   mask.

Layer widths at the end are identical to one-shot `uniform-20`, so parameter
counts match exactly and the only differences are which neurons are removed
and how the training budget is split.

## Experiment and decision rule

Config `configs/support-v2-iterative.json`: the `support-v2-longer.json`
recipe and data, trained candidates `unpruned-recovery`, `uniform-20`
(one-shot, 450 updates) and `iter-20`. Same five seeds, test sealed. Run
`runs/support-v2-iterative` (about 2.5 hours).

**Primary, predeclared comparison:** `iter-20` minus `uniform-20`, within the
run, token F1 on dev-holdout, paired by seed.

- **Accept** (merge to `main`): "confident" and positive.
- **Reject** (code stays on `experiment/7-iterative`): anything else, with a
  statement of whether it looks like no effect, an effect too small to
  resolve, or harm.

Secondary: each vs `unpruned-recovery`; seed sd; checkpoint chosen per stage;
overlap between the iterative and one-shot masks; dev selection-subset F1
right after each pruning step (before recovery); wall time.

Sanity: `unpruned-recovery` and `uniform-20` should reproduce the same
candidates in `runs/support-v2-allocator` (report whether exactly or within
numeric noise).

## Engineering constraints

- Opt-in; existing configs, fingerprints, masks, runs unchanged.
- Recovery invariants hold per stage (failure removes adapters and leaves
  base weights untouched; merged/reloaded/pre-merge predictions agree;
  non-finite values raise). Resume must work between and within stages.
- Checkpoint selection in both stages uses only the 150-example selection
  subset; dev-holdout is never used for any choice.
- While a GPU run is executing from the main checkout, work only in a
  separate git worktree; CPU only; never touch `runs/` or `datasets/`.
- Plain `git` is broken: prefix with
  `DEVELOPER_DIR=/Library/Developer/CommandLineTools`. `HF_HUB_OFFLINE=1`.
  Local commits only.

## Roles

- **Writer:** implementation, config, unit tests, launch command, time
  estimate.
- **Tester:** independent tests: final widths equal one-shot; total updates
  equal; stage-2 scores computed on the merged stage-1 model (not the
  original); data order across stages covers each example once; per-stage
  schedule values; resume between stages; failure cleanup; default path
  unchanged; config validation.
- **Verifier:** recompute numbers from raw records, confirm equal sizes and
  equal update budgets from the records, the recipe per stage, sealed test
  split, and the decision.
