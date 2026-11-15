# Improvement 6 plan: prune the embeddings (vocabulary trimming)

Shared plan for the writer, tester and verifier agents. Branch:
`improve/6-vocabulary` (from `main`, or from the latest accepted branch the
main session names).

## Why

Qwen2.5-0.5B has 151,936 vocabulary rows × 896 = 136.1M parameters in the
(tied) embedding/output matrix: 27.6% of the model, untouched by MLP pruning.
A support-reply model for English documents uses a small part of that
vocabulary. Removing unused rows is exact for every text whose tokens are
kept, so it may remove more parameters than 20% MLP pruning at no quality
cost.

## What to build

`prune-lab trim-vocab` (new module `src/prune_lab/vocab.py`), applied to a
finished export (after recovery and merge; no training):

1. **Kept set**, chosen from the *recovery and calibration* splits only (never
   dev or test): every token id that occurs in their rendered prompts and
   reference replies, plus all special/control tokens, the chat-template
   tokens, EOS/pad ids, and every single-byte token, so any text remains
   encodable.
2. **Model**: slice the embedding rows to the kept set (output layer stays
   tied), update `vocab_size` and special-token ids in the config, save a new
   export with a recorded old→new id map and the source export hash.
3. **Tokenizer fallback**: wrap encoding so the original tokenizer runs first
   and any id outside the kept set is split recursively into the BPE pieces it
   was merged from until all pieces are kept (worst case single bytes), then
   mapped to new ids. Decoding maps new→old ids and uses the original
   tokenizer. Text round-trips exactly. The model simply cannot emit a removed
   token.
4. **Evaluation** through the existing `evaluate_support` path with the
   wrapped tokenizer, writing per-example records in the existing format into
   a run directory that `multi-seed aggregate/compare` accept.
5. **Reporting** per export: kept vocabulary size, parameters removed, share
   of dev prompt tokens that needed the fallback, share of dev examples
   affected, share of dev reference-reply tokens the trimmed model cannot
   emit as a single token.

Exactness check (required): for prompts whose tokens are all kept, the trimmed
model's logits over kept tokens equal the original model's logits restricted
to those tokens (bit-identical on CPU), and generation is identical unless the
original would have emitted a removed token.

## Experiment and decision rule

Source exports: `runs/support-v2-allocator` (checkpoints kept), five seeds.
Trim `unpruned-recovery` → `vocab-unpruned`; trim `uniform-20` →
`vocab-uniform-20` (stacking). Output `runs/support-v2-vocab`. Evaluation is
about three minutes per model; no training.

**Primary, predeclared comparison** (matched or smaller size, per the
project's acceptance rule): `vocab-unpruned` minus `uniform-20` (the 20%
MLP-pruned model from the source run), token F1 on dev-holdout, paired by
seed — valid only if `vocab-unpruned` has no more parameters than
`uniform-20` (report both counts; if the trimmed model is larger, the
comparison is reported but cannot accept).

- **Accept** (merge to `main`): "confident" and positive, i.e. vocabulary
  trimming gives a better model than MLP pruning at no larger size.
- **Reject**: anything else; report whether trimming is at least harmless.

Secondary: `vocab-unpruned` − `unpruned-recovery` (cost of trimming itself;
expect ≈0); `vocab-uniform-20` − `uniform-20`; `vocab-uniform-20` −
`unpruned-recovery` (both reductions stacked, ≈40%+ smaller); parameter
counts; fallback statistics; a size-vs-F1 table.

Caveat to state in every report: the trimmed model is specialised. It cannot
emit tokens outside the kept set (other languages, rare symbols), and inputs
that use them are re-split into pieces the model saw less often in
pre-training. The dev split measures this only for the benchmark's domain.

## Engineering constraints

- Additive and opt-in; existing configs, fingerprints, runs unchanged.
- While a GPU run is executing from the main checkout, work only in a
  separate git worktree; CPU only; never touch `runs/` or `datasets/`.
  Real-model trimming/evaluation is launched by the main session.
- Plain `git` is broken: prefix with
  `DEVELOPER_DIR=/Library/Developer/CommandLineTools`. `HF_HUB_OFFLINE=1`.
  Local commits only.

## Roles

- **Writer:** module, CLI, unit tests on tiny models/tokenizers, the kept-set
  size and parameter count for the real tokenizer and data (CPU, no model
  weights needed), launch commands.
- **Tester:** independent tests: text round trip for arbitrary Unicode
  through the fallback; logits exactness; tied output layer still tied; ids
  for EOS/pad/chat template remapped correctly; kept set built without dev or
  test; generation identical when no removed token is involved; records
  aggregate; source export hash checked.
- **Verifier:** recompute all numbers from raw records, parameter counts from
  the trimmed exports, confirm the kept set came only from recovery and
  calibration, fallback statistics, sealed test split, and the decision.
