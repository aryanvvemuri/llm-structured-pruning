# Improvement 2 plan: longer recovery, distillation, attention LoRA

Shared plan for the writer, tester and verifier agents. Branch: `improve/2-recovery`.

## Why

The 5-seed baseline (`runs/support-v2-main`, see RESULTS.md) trains every variant
for 50 optimizer updates = 200 of the 1,800 recovery examples at a constant
learning rate. The result is dominated by seed luck: a variant's mean dev F1
moves 2–3.3 points between seeds, and the best checkpoint flips between update
25 and 50. Pruned variants trail recovery-only by 1.7–2.6 F1, not established.

Goal: a recovery recipe that (a) raises the quality of pruned models at the same
size and (b) reduces seed-to-seed variance so later comparisons are sharper.

## Components (each opt-in, each decided separately)

**2a — train longer.** One full pass over the recovery split.
- `recovery.max_steps: 450` (1,800 examples at accumulation 4),
  `checkpoints: [150, 300, 450]` (step 0 is always assessed).
- New opt-in learning-rate schedule: linear warmup for 20 updates, then cosine
  decay to 10% of the peak (`recovery.lr_schedule: "cosine"`,
  `warmup_steps: 20`, `min_lr_ratio: 0.1`). Peak stays 2e-4. Default
  (`"constant"`) must reproduce today's training exactly.
- Config: `configs/support-v2-longer.json`. Run: `runs/support-v2-longer`.

**2b — distillation from the recovered full model.**
- Teacher: the same seed's `unpruned-recovery` export from the same run
  (trained with the run's recipe, no distillation). Fit order must put
  `unpruned-recovery` first when distillation is on; the teacher is loaded
  frozen, eval mode, float32, alongside the student.
- Student loss on answer tokens only:
  `alpha * CE(student, labels) + (1 - alpha) * T^2 * KL(teacher_T || student_T)`
  with `alpha = 0.5`, `T = 2.0` (`recovery.distillation: {alpha, temperature}`).
  Applies to pruned candidates only; `unpruned-recovery` trains as in the
  reference recipe.
- The teacher never sees dev or test data beyond what the recipe already uses
  for checkpoint selection (the 150-example selection subset).
- Config: `configs/support-v2-distill.json`. Run: `runs/support-v2-distill`.

**2c — LoRA on attention as well as MLP.**
- Rank-8, alpha-16 adapters on `q_proj, k_proj, v_proj, o_proj` in addition to
  the three MLP projections (`recovery.lora_targets: "mlp+attention"`; default
  `"mlp"`). Merged before export like the MLP adapters; biases untouched.
- Applies to every trained candidate, including `unpruned-recovery`.
- Config: `configs/support-v2-attn.json`. Run: `runs/support-v2-attn`.

No hyperparameter in this plan is tuned. Values above are fixed in advance. If
any pilot/tuning is ever needed it may use only the 150-example selection
subset, never dev-holdout or test.

## Experiment sequence and decision rule

All runs: `configs/support-v2.json` settings otherwise unchanged, same five
seeds (20261114–20261118), `--discard-checkpoints`, test split sealed.

| Step | Candidate run | Reference run | Recipe |
| --- | --- | --- | --- |
| 2a | support-v2-longer | support-v2-main | baseline + 2a |
| 2b | support-v2-distill | best accepted so far | reference recipe + 2b |
| 2c | support-v2-attn | best accepted so far | reference recipe + 2c |

"Best accepted so far" is the latest accepted run, else `support-v2-main`.
If 2a is rejected, 2b and 2c use the 50-update schedule.

**Primary, predeclared comparison for every step:** `uniform-10` in the
candidate run minus `uniform-10` in the reference run, token F1 on dev-holdout,
paired by seed:

```
.venv/bin/python -m prune_lab multi-seed compare --baseline runs/<reference> \
  --candidate runs/<candidate> --pair uniform-10:uniform-10
```

- **Accept** (merge to `main`): decision is "confident" and positive.
- **Reject** (code stays on `experiment/2x-<name>`, not `main`): anything else.
  The report must say whether it looks like no effect, a small positive effect
  the protocol cannot resolve (minimum detectable ≈4 F1), or harm.

Secondary, reported but not deciding: the same comparison for
`unpruned-recovery`, `per-layer-01`, `per-layer-02`; seed sd of each variant's
mean; pruned-minus-unpruned gaps within the run; which checkpoint was selected;
training wall time per candidate.

## Engineering constraints

- Every new option is opt-in. With defaults, existing configs
  (`support.json`, `support-v2.json`) keep identical behaviour, fingerprints
  and results; `runs/support-v2-main` must stay valid as a reference.
- One local commit per component (2a, 2b, 2c), in that order, each leaving the
  suite green, with component code as separable as practical so a rejected
  component can be left off `main`. No pushing; main session handles merges.
- Recovery invariants stay: failures remove every adapter and leave base
  weights untouched; best-dev adapter is merged; merged, reloaded and
  pre-merge predictions agree; non-finite values raise.
- Memory budget 24 GB unified: student + teacher (float32, ~2 GB each) plus
  training state must fit with margin; record MPS memory snapshots.
- Plain `git` is broken on this machine: prefix with
  `DEVELOPER_DIR=/Library/Developer/CommandLineTools`. Use `.venv/bin/python`,
  `HF_HUB_OFFLINE=1`. Never touch `runs/support-v2-main`, `runs/support-v1`,
  `datasets/`. Nothing trained on `datasets/doc2dial_v1` may be reused.
- GPU experiments run one at a time and are launched by the main session.

## Roles

- **Writer:** implements 2a, 2b, 2c with unit tests and the three configs;
  reports a measured time and memory estimate per seed for each recipe from a
  short MPS preflight.
- **Tester:** independent CPU tests, written from this plan and the code:
  schedule values at each step, default path unchanged, distillation loss
  against a hand computation, teacher frozen and unaffected, fit order,
  attention adapters merged exactly, failure cleanup with the new adapter set,
  resume in the middle of a distillation run, config validation.
- **Verifier:** after each GPU run, recomputes from raw per-example records the
  primary comparison and secondary numbers, checks the recipe actually ran as
  configured (steps, schedule, teacher identity, adapter targets), that the
  test split stayed sealed, and states the accept/reject decision
  independently.

## Estimated cost

Baseline candidates take about 4.5 minutes each with 50 updates. With 450
updates, expect roughly 12–15 minutes per candidate, so about 4–5 hours per
5-seed run, more with a teacher forward pass. Three runs: about 13–16 GPU hours
in total, to be refined by the writer's preflight.
