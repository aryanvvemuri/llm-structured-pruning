# Improvement 1 retest: curve allocator against uniform at 20% (predeclared)

Written and committed on 2026-11-15, before any retest seed was run.

## Why

The first run (`runs/support-v2-allocator`, seeds 20261114–20261118) gave
curve-20 − uniform-20 = +1.29 dev-holdout token F1: seed t interval
[+0.12, +2.46], hierarchical bootstrap [−0.23, +2.84]. Not established under
the pass rule, and the run could only detect about 2.2 F1. Adding seeds to
that run after seeing the result would be optional stopping, so this is a
separate run on fresh seeds with everything fixed in advance.

## Design

- Code: branch `experiment/7-iterative` (allocator plus the stable selection
  lock). Config: `configs/support-v2-allocator-retest.json`, identical to the
  first run's recipe (450 updates, checkpoints 150/300/450, rank 8, peak 2e-4,
  warmup 20, cosine to 10%). Trained candidates: `unpruned-recovery` (required
  as the selection reference), `uniform-20`, `curve-20`.
- Seeds: ten fresh recovery seeds, 20261119–20261128. All ten are run and all
  ten are used; no seed is dropped or added after results are seen.
- Output: `runs/support-v2-allocator-retest`.
- Pruning masks are deterministic and the same as in the first run; only the
  recovery seed varies.
- Data: same dev-holdout set (450 examples). The test split stays sealed.

## Decision

- One primary comparison: curve-20 − uniform-20, token F1 on dev-holdout,
  seeds paired, **using the ten fresh seeds only**.
- Same rule as before: "confident" only if the hierarchical bootstrap interval
  and the paired seed t interval both exclude zero with the same sign.
- Confident and positive: the curve allocator is accepted and merged to `main`
  as an opt-in method, with uniform remaining the default until documented.
- Anything else: the rejection stands. No further retests of this comparison
  on this dev set.

## Secondary (descriptive only, no decision)

- The 15-seed pooled difference (5 old + 10 fresh).
- curve-20 − unpruned-recovery and uniform-20 − unpruned-recovery.

## Known limit

Ten seeds narrow the seed part of the interval (expected t half-width about
0.7 if the paired sd stays near 0.94). The bootstrap also resamples the 64
dev document families, and that part does not shrink with more seeds. A true
effect near +1.3 may therefore still come out "not established".
