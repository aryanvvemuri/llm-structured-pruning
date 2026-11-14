# Improvement 5 plan: pruned Qwen2.5-1.5B against native Qwen2.5-0.5B

Shared plan for the writer, tester and verifier agents. Branch:
`improve/5-larger-model` (from `main`).

## Question

Is a pruned larger model better than a natively small one? At 0.5B there was
nothing smaller to beat. Here the reference is the recovered, unpruned
Qwen2.5-0.5B (494.0M parameters, dev-holdout F1 54.89 ± 0.79 over 5 seeds in
`runs/support-v2-longer`).

## What is and is not possible (state this in every report)

Qwen2.5-1.5B has about 1,544M parameters: MLP ≈1,156M, embeddings ≈233M
(tied), attention ≈154M. This project only removes MLP neurons, with a
per-layer floor of 25% kept, so at most ≈867M can go and the smallest
reachable model is ≈677M — **still 37% larger than the 0.5B model.** A truly
size-matched comparison is impossible with MLP-only pruning. What can be
measured:

1. the ceiling: recovered unpruned 1.5B vs recovered unpruned 0.5B;
2. a size–quality curve for uniformly pruned 1.5B at 25% and 50% of all
   parameters removed (≈1,158M and ≈772M);
3. whether the 772M pruned 1.5B still beats the 494M native 0.5B. If it does
   not, pruning the larger model loses even with a 56% size advantage, and
   the answer to the question is "no" for this method. If it does, the answer
   is "not shown either way at matched size; better at 1.56× the size".

## What to build

- Mostly configuration: `configs/support-v2-1p5b.json` = the
  `support-v2-longer.json` recipe and data with
  `model: "Qwen/Qwen2.5-1.5B-Instruct"`, targets `[0.25, 0.50]`,
  uniform candidate per target, trained candidates `unpruned-recovery`,
  `uniform-25`, `uniform-50` (use the per-target uniform and
  `train_candidates` options from the allocator branch if it is merged;
  otherwise add the minimal equivalent). No per-layer allocator candidates
  unless improvement 1 is accepted, in which case add its 50% candidate.
- Remove anything in the code that assumes the 0.5B shape (hidden size 896,
  24 layers, parameter totals, memory estimates, default model names in
  checks). The 0.5B path must stay bit-identical.
- Weights are already downloaded to `.cache/huggingface` (float32, ≈6.2 GB in
  memory). Memory budget: 24 GB unified; training state for 1.5B must fit with
  margin — the preflight must report peak MPS driver memory and seconds per
  update and per reply.
- Same recipe values as 0.5B (rank 8, lr 2e-4, 450 updates, cosine): fixed in
  advance, not tuned for 1.5B. Say so in the report; a tuned recipe might
  favour either side.

## Experiment and decision rule

Run `runs/support-v2-1p5b`, same seeds, test sealed, `--discard-checkpoints`
unless disk use is modest (three 6.2 GB exports per seed ≈ 93 GB for 5 seeds —
discard).

Seeds: five (20261114–20261118) if the preflight estimate for the whole run is
at most 10 hours; otherwise the first three (20261114–20261116). The choice is
made from the preflight estimate only, before any dev-holdout score is seen,
and recorded.

**Primary, predeclared comparison:** `uniform-50` of 1.5B (≈772M) minus
`unpruned-recovery` of 0.5B (494M, `runs/support-v2-longer`), token F1 on
dev-holdout, **unpaired** (different base models; seeds share numbers only):

```
.venv/bin/python -m prune_lab multi-seed compare --baseline runs/support-v2-longer \
  --candidate runs/support-v2-1p5b --pair unpruned-recovery:uniform-50 --unpaired-seeds
```

Reading: confident and positive → pruned 1.5B is better at 1.56× the size
(matched size not shown). Not established or negative → pruned 1.5B does not
beat native 0.5B even while larger; MLP-only pruning of the larger model is
not competitive.

Secondary: unpruned 1.5B − unpruned 0.5B (ceiling); `uniform-25` − 0.5B;
pruned − unpruned within the 1.5B run; original (untrained) 1.5B score;
parameter counts; seed sd; time and memory per candidate; F1 per million
parameters table across both models.

**Merge rule:** this is a question, not a tweak. The code and config merge to
`main` if they run correctly, the 0.5B path is unchanged, and the verifier
confirms the numbers; the finding is reported as it comes out.

## Engineering constraints

- Additive/opt-in; existing configs, fingerprints and runs unchanged.
- While a GPU run is executing from the main checkout, work only in a
  separate git worktree; never switch branches or edit `src/` in the main
  checkout; never touch `runs/` or `datasets/`. CPU-only until the main
  session says the GPU is free (loading 1.5B on CPU for parameter counts and
  shape checks is fine: about 6 GB RAM, do it once and briefly).
- The run's evaluation protocol check in `multi-seed compare` must allow a
  different base model only when asked explicitly (a flag or the existing
  `--allow-different-schedules`-style opt-in), never silently.
- Plain `git` is broken: prefix with
  `DEVELOPER_DIR=/Library/Developer/CommandLineTools`. `HF_HUB_OFFLINE=1`,
  `HF_HOME=<main checkout>/.cache/huggingface`. Local commits only.

## Roles

- **Writer:** config, shape-generalisation fixes, unit tests, exact parameter
  counts for each candidate, launch command, and a statement of what was
  0.5B-specific.
- **Tester:** independent tests: candidate sizes for the 1.5B shape (meta
  device, no weights needed); 0.5B fingerprints/masks unchanged; cross-model
  compare refuses without the explicit opt-in and works with it, unpaired;
  uniform-50 respects the 25% floor; config validation.
- **Verifier:** after the run, recomputes primary and secondary numbers from
  raw records, checks parameter counts against the exports' recorded widths,
  recipe as configured, seed-count rule applied as predeclared, test sealed,
  and states the reading independently.
