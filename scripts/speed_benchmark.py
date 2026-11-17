#!/usr/bin/env python3
"""Forced-length speed benchmark of exports (PyTorch, default MPS), interleaved repeats.

Every model generates exactly --steps greedy tokens (no EOS stop) for the same --prompts dev prompts
(first by key), so all do the same work. Per repeat: prompt-reading time per prompt (batch one),
decode ms per token (batch one) and decode tokens/s at --batch (left-padded). All models stay
loaded; each repeat visits them in rotated order after one warm-up pass per model. Output: per-model
medians and, against --reference, ratios with 95% bootstrap intervals over repeats (JSON + markdown).
--max-loaded N bounds resident models (least recently used is unloaded; load and warm-up are not timed);
each repeat row records wall-clock started_at/finished_at.
Plain, MLP-pruned and reduced-depth exports load through the project loader. --mlx-8bit also times
batch-one generation of 8-bit MLX conversions made with the project's quantization code.

  HF_HUB_OFFLINE=1 .venv/bin/python scripts/speed_benchmark.py --reference uniform-20 \\
      --dev runs/support-v2-allocator/seed-20261114/data/dev.jsonl --output speed.json \\
      uniform-20=<export> layers-3=<export> ...
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def parse_models(items):
    models = {}
    for item in items:
        name, sep, path = item.partition('=')
        if not sep or not name or not path or name in models:
            raise SystemExit(f'Expected distinct name=export_dir pairs, got {item!r}')
        if not Path(path).is_dir():
            raise SystemExit(f'Export directory not found: {path}')
        models[name] = path
    return models


def make_timers(torch, device):
    def sync():
        if device == 'mps':
            torch.mps.synchronize()
        elif device.startswith('cuda'):
            torch.cuda.synchronize()

    def single(model, ids, steps):
        x = torch.tensor([ids], device=device)
        sync(); t0 = time.perf_counter()
        out = model(input_ids=x, use_cache=True)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        sync(); t1 = time.perf_counter()
        past = out.past_key_values
        for _ in range(steps):
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        sync(); t2 = time.perf_counter()
        return t1 - t0, t2 - t1

    def batched(model, rows, pad, steps):
        width = max(len(r) for r in rows)
        x = torch.tensor([[pad] * (width - len(r)) + r for r in rows], device=device)
        mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows], device=device)
        sync(); t0 = time.perf_counter()
        out = model(input_ids=x, attention_mask=mask, use_cache=True)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        sync(); t1 = time.perf_counter()
        past = out.past_key_values
        ones = torch.ones((len(rows), 1), dtype=mask.dtype, device=device)
        for _ in range(steps):
            mask = torch.cat([mask, ones], dim=1)
            out = model(input_ids=nxt, attention_mask=mask, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        sync(); t2 = time.perf_counter()
        return t1 - t0, t2 - t1
    return single, batched


def measure_once(timers, model, prompts, pad, steps, batch):
    single, batched = timers
    pre = dec = 0.0
    for ids in prompts:
        a, b = single(model, ids, steps); pre += a; dec += b
    bpre = bdec = 0.0
    for start in range(0, len(prompts), batch):
        a, b = batched(model, prompts[start:start + batch], pad, steps); bpre += a; bdec += b
    return {'prefill_ms_per_prompt': 1000 * pre / len(prompts), 'decode_ms_per_token': 1000 * dec / (len(prompts) * steps),
            'batch_prefill_ms_per_prompt': 1000 * bpre / len(prompts),
            'batch_decode_tokens_per_second': len(prompts) * steps / bdec}


def rotated(names, repeat):
    k = repeat % len(names)
    return names[k:] + names[:k]


def pytorch_runs(models, examples, args, dtype):
    """Interleaved repeats. With --max-loaded N > 0 at most N models are resident: a model that is not
    loaded is loaded (least recently used one unloaded first) and warmed up before it is timed, so
    load and warm-up time are never measured. 0 keeps every model loaded (the original behaviour)."""
    from collections import OrderedDict
    import torch
    from prune_lab.data import prompt_ids
    from prune_lab.modeling import count_parameters, load_model
    loaded, info = OrderedDict(), {}
    timers = make_timers(torch, args.device)

    def release():
        import gc
        gc.collect()
        if args.device == 'mps':
            torch.mps.empty_cache()

    def get(name):
        if name in loaded:
            loaded.move_to_end(name)
            return loaded[name]
        if args.max_loaded and len(loaded) >= args.max_loaded:
            loaded.popitem(last=False)
            release()
        model, tokenizer = load_model(models[name], device=args.device, dtype=dtype)
        prompts = [list(prompt_ids(tokenizer, ex.prompt)) for ex in examples]
        entry = (model, prompts, tokenizer.pad_token_id)
        with torch.inference_mode():  # one warm-up pass per load
            timers[0](model, prompts[0], args.steps)
            timers[1](model, prompts[:args.batch], tokenizer.pad_token_id, args.steps)
        if name not in info:
            info[name] = {'export': models[name], 'parameters': count_parameters(model), 'layers': len(model.model.layers),
                          'mlp_widths': sorted({layer.mlp.down_proj.in_features for layer in model.model.layers}),
                          'prompt_tokens_mean': sum(map(len, prompts)) / len(prompts), 'loads': 0}
        info[name]['loads'] += 1
        loaded[name] = entry
        return entry

    runs = {name: [] for name in models}
    names = list(models)
    with torch.inference_mode():
        for name in names:  # initial load and warm-up (only the last --max-loaded stay resident)
            get(name)
        for repeat in range(args.repeats):
            for name in rotated(names, repeat):
                model, prompts, pad = get(name)
                started = time.time()
                row = measure_once(timers, model, prompts, pad, args.steps, args.batch)
                runs[name].append(dict(row, repeat=repeat, started_at=started, finished_at=time.time()))
            print(f'{dtype} repeat {repeat + 1}/{args.repeats}', flush=True)
    loaded.clear()
    release()
    return runs, info


def mlx_runs(models, examples, args):
    """Batch-one forced decoding of 8-bit MLX conversions (same prompts and token count)."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from transformers import AutoTokenizer
    from prune_lab import quantize
    from prune_lab.data import prompt_ids
    loaded, info = {}, {}
    for name, path in models.items():
        target = Path(args.mlx_dir) / name
        record = quantize.convert(path, target, '8bit') if not (target / 'conversion.json').exists() \
            else json.loads((target / 'conversion.json').read_text())
        model, _ = quantize.load_mlx(target)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False, fix_mistral_regex=False)
        loaded[name] = (model, [list(prompt_ids(tokenizer, ex.prompt)) for ex in examples])
        info[name] = {'mlx_dir': str(target), 'weight_bytes': record['weight_bytes'],
                      'effective_bits_per_weight': record['effective_bits_per_weight']}

    def timed(model, ids):
        cache = make_prompt_cache(model)
        start = time.perf_counter()
        logits = model(mx.array([ids]), cache=cache)[:, -1, :]
        token = mx.argmax(logits, axis=-1)
        mx.eval(token)
        prefill = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(args.steps):
            logits = model(token[:, None], cache=cache)[:, -1, :]
            token = mx.argmax(logits, axis=-1)
            mx.eval(token)
        return prefill, time.perf_counter() - start
    names = list(models)
    for name in names:
        timed(loaded[name][0], loaded[name][1][0])
    runs = {name: [] for name in names}
    for repeat in range(args.repeats):
        for name in rotated(names, repeat):
            model, prompts = loaded[name]
            started = time.time()
            pre = dec = 0.0
            for ids in prompts:
                a, b = timed(model, ids); pre += a; dec += b
            runs[name].append({'prefill_ms_per_prompt': 1000 * pre / len(prompts),
                               'decode_ms_per_token': 1000 * dec / (len(prompts) * args.steps),
                               'repeat': repeat, 'started_at': started, 'finished_at': time.time()})
        print(f'mlx-8bit repeat {repeat + 1}/{args.repeats}', flush=True)
    return runs, info


def markdown(report):
    from prune_lab.speed import PRIMARY
    lines = [f"# Speed benchmark (reference `{report['reference']}`)", '',
             f"{report['prompts']} dev prompts, {report['steps']} forced tokens, batch {report['batch']}, "
             f"{report['repeats']} interleaved repeats, device {report['device']}. Ratios are model / reference with "
             "95% bootstrap intervals over repeats.", '']
    for mode, block in report['results'].items():
        lines += [f'## {mode}', '', '| Model | Layers | Parameters | Batch tok/s | Ratio [95%] | Batch-one ms/token | Ratio [95%] | Prompt ms |',
                  '| --- | ---: | ---: | ---: | --- | ---: | --- | ---: |']
        for name, entry in block['summary'].items():
            med, vs = entry['medians'], entry.get('vs_reference', {})
            meta = block['models'][name]
            cell = lambda m: (f"{vs[m]['ratio']:.3f} [{vs[m]['interval_95'][0]:.3f}, {vs[m]['interval_95'][1]:.3f}] "
                              f"{vs[m]['confidently']}") if m in vs else '—'
            fmt = lambda m, d=1: f'{med[m]:.{d}f}' if m in med else '—'
            size = meta.get('parameters', meta.get('weight_bytes'))
            size = f'{size:,}' + ('' if 'parameters' in meta else ' bytes') if size is not None else '—'
            lines.append(f"| {name} | {meta.get('layers', '—')} | {size} | "
                         f"{fmt(PRIMARY)} | {cell(PRIMARY)} | {fmt('decode_ms_per_token', 2)} | {cell('decode_ms_per_token')} | "
                         f"{fmt('prefill_ms_per_prompt', 2)} |")
        lines.append('')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('models', nargs='+', metavar='name=export_dir')
    parser.add_argument('--reference', required=True)
    parser.add_argument('--dev', required=True, help='Prepared dev split JSONL (prompts are the first --prompts by key)')
    parser.add_argument('--output', required=True, help='JSON report path; a .md summary is written next to it')
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--dtypes', default='float16,float32')
    parser.add_argument('--prompts', type=int, default=32)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--mlx-8bit', action='store_true')
    parser.add_argument('--mlx-dir', help='Where 8-bit MLX conversions are written (required with --mlx-8bit)')
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--max-loaded', type=int, default=0,
                        help='Keep at most N PyTorch models loaded (0 = all; e.g. 1-2 for float32 on 24 GB). Interleaving is kept.')
    args = parser.parse_args(argv)
    models = parse_models(args.models)
    if args.reference not in models:
        parser.error('--reference must be one of the model names')
    if args.repeats < 2 or args.prompts < 1 or args.steps < 1 or args.batch < 1 or args.max_loaded < 0:
        parser.error('--repeats >= 2 and positive --prompts/--steps/--batch required')
    if args.mlx_8bit and not args.mlx_dir:
        parser.error('--mlx-8bit needs --mlx-dir')
    if any(part == 'runs' for part in Path(args.output).resolve().parts):
        parser.error('Write the benchmark report outside runs/')
    from prune_lab.speed import summarize
    from prune_lab.support_data import read_support_jsonl
    examples = sorted(read_support_jsonl(args.dev), key=lambda ex: ex.key)[:args.prompts]
    report = {'reference': args.reference, 'prompts': len(examples), 'steps': args.steps, 'batch': args.batch,
              'max_loaded': args.max_loaded,
              'repeats': args.repeats, 'device': args.device, 'prompt_keys': [ex.key for ex in examples], 'results': {}}
    for dtype in [d for d in args.dtypes.split(',') if d]:
        runs, info = pytorch_runs(models, examples, args, dtype)
        report['results'][f'pytorch-{dtype}'] = {'models': info, 'runs': runs,
                                                 'summary': summarize(runs, args.reference, args.bootstrap_samples)}
    if args.mlx_8bit:
        runs, info = mlx_runs(models, examples, args)
        report['results']['mlx-8bit'] = {'models': info, 'runs': runs,
                                         'summary': summarize(runs, args.reference, args.bootstrap_samples)}
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=1) + '\n')
    output.with_suffix('.md').write_text(markdown(report))
    print(output.with_suffix('.md').read_text())


if __name__ == '__main__':
    main()
