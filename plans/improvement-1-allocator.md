# Improvement 1 plan: fix the per-layer allocator

Shared plan for the writer, tester and verifier agents. Branch: `improve/1-allocator`
(from `main`, which has improvement 2a). Reference run: `runs/support-v2-longer`.

## Why

With the 2a recipe (5 seeds, dev-holdout token F1) the current allocator shows no
benefit: per-layer 10% minus uniform 10% is −0.27, not established. Its known
weaknesses (`src/prune_lab/allocation.py`):

- sensitivity is measured on 16 of the 150 calibration examples;
- one probe per layer at 10% removal, then extrapolated with a fixed
  `(1 + percentile)^2` shape;
- negative sensitivities (noise) are clamped to 1% of the largest one, which
  makes those layers the cheapest to prune, i.e. noise is treated as strong
  evidence of safety;
- a layer may lose up to 75% of its neurons (`min_keep_ratio = 0.25`).

Seed sd is now 0.6–1.05, so a difference of roughly 1.5–2 F1 should be
detectable. At 10% removal uniform trails recovery-only by only 0.57, which
leaves no room to show a gain, so the decision is made at **20%**, where
per-layer pruning costs a confident −2.11.

## Components (all opt-in; defaults reproduce today's masks exactly)

**1a — allocator v2** (`allocation.method: "curve"`; default `"legacy"`).
- Sensitivity on all 150 calibration examples (`sensitivity_examples: 150`).
- Probe each layer at removal fractions 0.10, 0.25 and 0.50 (lowest Taylor
  scores first), giving per layer a loss-increase curve through
  (0, 0), (0.10, d1), (0.25, d2), (0.50, d3).
- Noise handling: clamp negative increases to 0, then make each curve
  non-decreasing (running maximum). A zero increase means "no evidence of
  harm up to here", not "free": ties are broken toward the layer with the
  smallest fraction already removed, so zero-signal layers are pruned evenly
  instead of being emptied first.
- Per-layer cap: at most 50% of a layer's neurons removed
  (`min_keep_ratio: 0.5`).
- Allocation: greedy over 64-neuron groups, always taking the group with the
  smallest marginal loss increase per parameter removed, read from the
  piecewise-linear curve; stop at the nearest reachable budget (same budget
  rule as today). Within a layer, neurons are still removed in Taylor order.
- Record the raw probe results, the processed curves and the per-layer
  removal fractions in `allocation.json`.

**1b — comparison allocator, FLAP-style** (`allocation.method: "global-standardized"`).
- Standardize Taylor scores within each layer (subtract the layer mean, divide
  by the layer sd), rank all neurons globally, remove the lowest until the
  budget is met, in 64-neuron groups per layer, with the same 50% cap.
  No sensitivity probes. This is the published-style baseline the new
  allocator should match or beat.

**Uniform at 20%.** The campaign must be able to emit a uniform candidate for
every target (today only `uniform-10` exists), so `uniform-20` is available.
Uniform pruning moves all 24 layers by one 64-neuron group at a time, so
`uniform-20` lands at 20.057% removed versus 19.988% for the other 20%
candidates; this is accepted and parameter counts are reported.

No value above is tuned. If a pilot is ever needed it may use calibration loss
only, never dev or test.

## Experiment and decision rule

Config `configs/support-v2-allocator.json`: the `support-v2-longer.json` recipe
and data unchanged, same five seeds, `--discard-checkpoints`, test sealed.
Trained candidates: `unpruned-recovery`, `uniform-20`, `per-layer-02` (legacy
allocator, 20%), `curve-20`, `global-std-20`. Run: `runs/support-v2-allocator`.
(About 9.4 minutes per candidate, so roughly 4 hours.)

Sanity checks on the run: `unpruned-recovery` and legacy `per-layer-02` should
reproduce `runs/support-v2-longer` (same masks; scores equal or within
run-to-run numeric noise — report which).

**Primary, predeclared comparison:** `curve-20` minus `uniform-20`, within the
run, token F1 on dev-holdout, paired by seed.

- **Accept** (merge to `main`, `curve` becomes the documented allocator):
  decision "confident" and positive.
- **Reject** (code stays on `experiment/1-allocator`): anything else. The
  report must say whether it looks like no effect, a small effect below what
  the protocol resolves, or harm, and whether the right conclusion is "use
  uniform allocation".

Secondary, reported but not deciding: `curve-20` − `per-layer-02`;
`global-std-20` − `uniform-20`; `curve-20` − `global-std-20`; each pruned
variant − `unpruned-recovery`; seed sd; per-layer removal fractions of each
allocator; calibration loss right after pruning (before recovery) for each
allocator; matched size (all 20% candidates within one 64-neuron group of the
target, parameter counts reported).

**Combined run:** to save GPU time this experiment runs together with improvement 7 as one run, `runs/support-v2-allocator` (`configs/support-v2-allocator-iterative.json`, checkpoints kept), with the same candidates, settings and predeclared comparisons; `unpruned-recovery` and `uniform-20` are shared, and a test shows each candidate trains identically to the separate configs.

## Engineering constraints

- Defaults unchanged: existing configs keep identical masks, fingerprints and
  results; `runs/support-v2-main` and `runs/support-v2-longer` stay valid.
- A GPU run may be executing from the main checkout
  (`/Users/aryan/llm_pruning_optimization`). Do not switch branches or edit
  `src/` there. Work in the worktree `/Users/aryan/llm_pruning_2a` (create
  branch `improve/1-allocator` from `main`), run tests on CPU with
  `PYTHONPATH=src /Users/aryan/llm_pruning_optimization/.venv/bin/python -m pytest`,
  and confirm the worktree's `prune_lab` is the one imported. Data may be
  symlinked in read-only for tests; never commit the symlink.
- Local commits only, suite green at each; no pushing; the main session merges.
- Plain `git` is broken: prefix with
  `DEVELOPER_DIR=/Library/Developer/CommandLineTools`. `HF_HUB_OFFLINE=1`.
  Never touch `runs/`, `datasets/`. Nothing trained on `datasets/doc2dial_v1`
  may be reused.
- GPU experiments run one at a time and are launched by the main session.
  Sensitivity probing at 150 examples × 3 fractions × 24 layers is ~18× the
  current probe work: report the measured or estimated time per seed, and
  note that masks do not depend on the recovery seed (if the campaign can
  safely compute the allocation once and reuse it across seeds, say so, but
  do not add caching unless it is simple and verified).

## Roles

- **Writer:** implements 1a, 1b, the per-target uniform candidate and the
  config, with unit tests; reports per-layer removal fractions for each
  allocator at 20% if they can be computed on CPU in reasonable time,
  otherwise leaves that to the GPU run.
- **Tester:** independent CPU tests from this plan and the code: legacy path
  bit-identical; curve construction (negative clamp, monotone envelope);
  tie-breaking spreads zero-signal layers evenly; 50% cap never crossed;
  budget reached within one group; greedy order against a hand-worked small
  example; global-standardized ranking against a hand computation; uniform
  candidate per target; config validation; allocation record contents.
- **Verifier:** after the GPU run, recomputes the primary and secondary
  numbers from raw per-example records, checks the masks were produced by the
  configured method (recompute from the recorded scores and curves), matched
  sizes, sanity reproduction of the reference run, sealed test split, and
  states the decision independently.
