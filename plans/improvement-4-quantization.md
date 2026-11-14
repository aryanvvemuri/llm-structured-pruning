# Improvement 4 plan: 4-bit quantization baselines

Shared plan for the writer, tester and verifier agents. Branch:
`improve/4-quantization` (from `main`; rebase onto the allocator work if it is
accepted). This is an **infrastructure/baseline** item: it passes if it runs
correctly and is verified. There is no "gain" to accept; the numbers go in
RESULTS.md whatever they are.

## Why

Pruning 10–20% of parameters saves 10–20% of size. 4-bit quantization saves
roughly 70% with, usually, a small quality cost. A reader needs to see pruning
next to quantization, and the two combined, on the same benchmark, to judge
whether pruning is worth doing here at all.

## What to build

A `prune-lab quantize` path (new module `src/prune_lab/quantize.py`, optional
dependency `mlx-lm`, installed into `.venv`; importing `prune_lab` must still
work without it) that, for an exported candidate directory:

1. Converts the float32 safetensors export to MLX in three precisions:
   **fp16** (framework control), **8-bit**, **4-bit** (mlx-lm default group
   quantization; record bits, group size and which tensors were quantized,
   plus effective bits per weight).
2. Supports pruned exports. Uniformly pruned models have one
   `intermediate_size`; per-layer pruned models have a different MLP width per
   layer, which stock mlx-lm Qwen2 does not support — build the MLX model with
   per-layer widths (subclass or patch after construction). Verify by logits:
   fp16 MLX next-token logits must agree with the PyTorch export on a handful
   of prompts (argmax equal, max abs difference small and reported).
3. Evaluates each converted model on the dev split with the same prompts,
   greedy decoding, `max_new_tokens` and scoring as `support_evaluation`
   (reuse the scoring code; only generation differs), writing per-example
   records in the same format so `multi-seed aggregate/compare` and the
   verifier's scripts work unchanged. Dev-holdout is the headline set; test
   stays sealed.
4. Measures on-disk size (bytes of weight files), peak memory, and
   generation speed (tokens/s over a fixed 32-example dev slice, 3 repeats,
   after a warm-up), for each precision. Speed numbers are MLX-to-MLX only;
   never compare MLX speed with PyTorch-MPS speed as if it were a
   quantization effect.

## Experiment

Source exports: the 5-seed run that keeps its checkpoints (the allocator run
`runs/support-v2-allocator`, launched without `--discard-checkpoints`; if the
allocator work is rejected the same run still supplies the exports).
Variants quantized, for each of the five seeds:

- `unpruned-recovery` (no pruning)
- `uniform-20` (20% pruning)
- the accepted allocator's 20% candidate, only if improvement 1 is accepted

at fp16, 8-bit and 4-bit. Output: `runs/support-v2-quant/seed-*/<variant>-<precision>/`.

Reported (dev-holdout token F1, 5 seeds, same decision machinery):

| Question | Comparison |
| --- | --- |
| Does MLX reproduce PyTorch? | fp16 MLX − PyTorch export, per variant (expect ≈0) |
| Cost of 4-bit | unpruned 4-bit − unpruned fp16 |
| Cost of 20% pruning | uniform-20 fp16 − unpruned fp16 |
| Pruning + quantization | uniform-20 4-bit − unpruned 4-bit; − unpruned fp16 |
| Size/speed | bytes, tokens/s, peak memory per variant × precision |

Plus a size-vs-F1 table: each variant × precision with bytes and mean F1, so
the reader can see which point dominates.

**Pass rule:** runs end to end on all seeds; fp16 MLX matches PyTorch
(logit check passes; dev F1 difference not established and small); verifier
recomputes the table from raw records with no mismatches.

## Engineering constraints

- New code is additive and optional; existing configs, fingerprints and runs
  unchanged; suite green without `mlx-lm` installed (tests that need it skip).
- MLX uses the GPU and unified memory. While a PyTorch GPU run is in flight,
  only tiny-model tests may run; real-model conversions and evaluations are
  launched by the main session, one at a time.
- While a GPU run is executing from the main checkout, work only in a
  separate git worktree; never switch branches or edit `src/` in the main
  checkout; never touch `runs/` or `datasets/` (read-only use of exports is
  fine once the main session says so).
- `pip install mlx-lm` into `.venv` is approved by the user. Record the
  installed versions of `mlx` and `mlx-lm` in the output.
- Plain `git` is broken: prefix with
  `DEVELOPER_DIR=/Library/Developer/CommandLineTools`. Local commits only.

## Roles

- **Writer:** module, CLI, tiny-model unit tests (conversion round trip,
  per-layer widths, record format), launch commands, and a time/memory
  estimate per seed.
- **Tester:** independent tests: per-layer-width MLX model equals PyTorch on
  a tiny random model; quantized weight files are actually smaller and
  actually low-bit; generation settings equal the PyTorch path (prompt
  template, stop tokens, max tokens, greedy); records aggregate with the
  existing tools; behaviour without `mlx-lm`; sealed test split.
- **Verifier:** after the run, recomputes every table entry from raw records,
  checks file sizes on disk, the fp16-vs-PyTorch agreement, that the right
  export (hash) was quantized for each seed/variant, and that test stayed
  sealed.
