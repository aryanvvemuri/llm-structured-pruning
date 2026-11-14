"""Vocabulary trimming of finished exports (improvement 6).

The kept set comes from the recovery and calibration splits only (rendered prompts and reference
replies), plus every special/added token, EOS/pad/BOS ids and all 256 byte-level base tokens, so
any text stays encodable. The embedding (and tied output) rows are sliced to the kept set. A
wrapper tokenizer runs the original tokenizer and splits any removed token back into the BPE
pieces it was merged from (recursively, down to single bytes), then maps to new ids; decoding
maps new ids back and uses the original tokenizer, so text round-trips exactly. The trimmed model
cannot emit a removed token.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil

from .reporting import write_json

MAP_FILE = "vocab_map.json"


def bytes_to_unicode():
    """GPT-2 / Qwen byte-level BPE alphabet: byte value -> printable character."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class BPEStructure:
    """Vocabulary, merge parents and byte tokens read from a fast tokenizer's tokenizer.json."""

    def __init__(self, tokenizer):
        state = json.loads(tokenizer.backend_tokenizer.to_str())
        model = state["model"]
        if model.get("type") != "BPE":
            raise ValueError("Vocabulary trimming supports byte-level BPE tokenizers only")
        self.vocab = dict(model["vocab"])
        self.tokens = {i: t for t, i in self.vocab.items()}
        self.parents = {}
        for merge in model["merges"]:
            left, right = merge.split(" ", 1) if isinstance(merge, str) else merge
            merged = left + right
            # The first (highest-priority) merge producing a token is how BPE builds it.
            self.parents.setdefault(merged, (left, right))
        alphabet = bytes_to_unicode().values()
        missing = [c for c in alphabet if c not in self.vocab]
        if missing:
            raise ValueError(f"Tokenizer lacks {len(missing)} byte-level base tokens; fallback would be incomplete")
        self.byte_ids = sorted(self.vocab[c] for c in alphabet)
        self.added_ids = sorted(int(i) for i in tokenizer.added_tokens_decoder)

    def pieces(self, token_id, kept):
        """Split ``token_id`` into kept ids whose strings concatenate to the original token."""
        if token_id in kept:
            return [token_id]
        token = self.tokens.get(token_id)
        if token is None or token not in self.parents:
            raise ValueError(f"Token id {token_id} is not kept and cannot be split into BPE pieces")
        left, right = self.parents[token]
        return self.pieces(self.vocab[left], kept) + self.pieces(self.vocab[right], kept)


def special_ids(tokenizer, structure):
    ids = set(structure.added_ids) | set(structure.byte_ids) | set(tokenizer.all_special_ids)
    for value in (tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id):
        if value is not None:
            ids.add(int(value))
    return ids


def example_ids(tokenizer, example):
    """Original ids of a rendered prompt and of a reference reply (with EOS), as training sees them."""
    from .data import prompt_ids
    answer = tokenizer.encode(example.answer, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        answer = answer + [tokenizer.eos_token_id]
    return list(prompt_ids(tokenizer, example.prompt)), answer


def build_kept_set(tokenizer, splits, extra_ids=()):
    """Kept original ids from ``splits`` = {name: examples}; only recovery/calibration are allowed."""
    if not splits or set(splits) - {"recovery", "calibration"}:
        raise ValueError("The kept set may only be built from the recovery and calibration splits")
    structure = BPEStructure(tokenizer)
    kept = special_ids(tokenizer, structure) | {int(i) for i in extra_ids}
    counts = {}
    for name, rows in splits.items():
        before = len(kept)
        for row in rows:
            prompt, answer = example_ids(tokenizer, row)
            kept.update(prompt)
            kept.update(answer)
        counts[name] = {"examples": len(rows), "new_ids": len(kept) - before}
    return sorted(kept), {"from_splits": sorted(splits), "split_counts": counts,
                          "special_and_byte_ids": len(special_ids(tokenizer, structure))}


class TrimmedTokenizer:
    """Delegating wrapper: original tokenization, removed tokens split into kept BPE pieces, ids remapped."""

    def __init__(self, tokenizer, kept_ids):
        self._tokenizer = tokenizer
        self.kept_ids = list(kept_ids)
        if self.kept_ids != sorted(set(self.kept_ids)):
            raise ValueError("kept_ids must be sorted and unique")
        self._new = {old: new for new, old in enumerate(self.kept_ids)}
        self._kept = set(self.kept_ids)
        self._structure = BPEStructure(tokenizer)
        self.fallback_tokens = 0  # original tokens that needed splitting, cumulative

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

    def __len__(self):
        return len(self.kept_ids)

    @lru_cache(maxsize=None)
    def _pieces(self, token_id):
        return tuple(self._structure.pieces(token_id, self._kept))

    def map_ids(self, ids):
        result = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id not in self._kept:
                self.fallback_tokens += 1
            result.extend(self._new[p] for p in self._pieces(token_id))
        return result

    def unmap_ids(self, ids):
        return [self.kept_ids[int(i)] for i in ids]

    def _mapped_id(self, value):
        return None if value is None else self._new[int(value)]

    @property
    def eos_token_id(self):
        return self._mapped_id(self._tokenizer.eos_token_id)

    @property
    def pad_token_id(self):
        return self._mapped_id(self._tokenizer.pad_token_id)

    @property
    def bos_token_id(self):
        return self._mapped_id(self._tokenizer.bos_token_id)

    @property
    def all_special_ids(self):
        return [self._new[i] for i in self._tokenizer.all_special_ids]

    def encode(self, text, *args, **kwargs):
        return self.map_ids(self._tokenizer.encode(text, *args, **kwargs))

    def apply_chat_template(self, *args, **kwargs):
        output = self._tokenizer.apply_chat_template(*args, **kwargs)
        if kwargs.get("tokenize", True) is False:
            return output
        if isinstance(output, dict) or hasattr(output, "keys"):
            raise ValueError("Only list-of-ids chat template output is supported")
        return self.map_ids(output)

    def decode(self, ids, *args, **kwargs):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self._tokenizer.decode(self.unmap_ids(ids), *args, **kwargs)


def fallback_statistics(tokenizer, kept_ids, examples):
    """Dev-side statistics without any model: prompt fallback shares and unemittable reference tokens."""
    kept = set(kept_ids)
    prompt_tokens = prompt_removed = affected = reference_tokens = reference_removed = 0
    for example in examples:
        prompt, answer = example_ids(tokenizer, example)
        removed = sum(i not in kept for i in prompt)
        prompt_tokens += len(prompt)
        prompt_removed += removed
        affected += removed > 0
        reference_tokens += len(answer)
        reference_removed += sum(i not in kept for i in answer)
    return {"examples": len(examples), "prompt_tokens": prompt_tokens,
            "prompt_tokens_needing_fallback": prompt_removed,
            "prompt_fallback_share": prompt_removed / prompt_tokens if prompt_tokens else 0.0,
            "examples_with_prompt_fallback": affected,
            "examples_affected_share": affected / len(examples) if examples else 0.0,
            "reference_tokens": reference_tokens, "reference_tokens_not_emittable": reference_removed,
            "reference_unemittable_share": reference_removed / reference_tokens if reference_tokens else 0.0,
            "note": "Reference tokens include the EOS; a reference token outside the kept set cannot be emitted "
                    "as that single token (the trimmed model may still spell it with kept pieces)."}


def trimmed_parameters(config, kept_size, parameters):
    """Parameter count after trimming an export with ``parameters`` parameters."""
    hidden, vocab = config["hidden_size"], config["vocab_size"]
    matrices = 1 if config.get("tie_word_embeddings", False) else 2
    return parameters - matrices * (vocab - kept_size) * hidden


def trim_export(source, destination, kept_ids, provenance=None):
    """Slice embedding/output rows of an export to ``kept_ids`` and save a new export with its id map."""
    import torch
    from .artifacts import save_checkpoint
    from .campaign import checkpoint_identity
    from .modeling import count_parameters, load_model
    source, destination = Path(source), Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Trimmed export exists and is not empty: {destination}")
    source_hash = checkpoint_identity(source)
    model, tokenizer = load_model(str(source), device="cpu", dtype="float32")
    kept_ids = sorted(set(int(i) for i in kept_ids))
    vocab = model.config.vocab_size
    if not kept_ids or kept_ids[-1] >= vocab or kept_ids[0] < 0:
        raise ValueError("kept_ids outside the model vocabulary")
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    before = count_parameters(model)
    index = torch.tensor(kept_ids)
    with torch.no_grad():
        embedding = torch.nn.Embedding(len(kept_ids), model.config.hidden_size, dtype=model.model.embed_tokens.weight.dtype)
        embedding.weight.copy_(model.model.embed_tokens.weight.index_select(0, index))
        model.model.embed_tokens = embedding
        if tied:
            model.lm_head = torch.nn.Linear(model.config.hidden_size, len(kept_ids), bias=False)
            model.lm_head.weight = model.model.embed_tokens.weight
        else:
            head = torch.nn.Linear(model.config.hidden_size, len(kept_ids), bias=False)
            head.weight.copy_(model.lm_head.weight.index_select(0, index))
            model.lm_head = head
    new = {old: i for i, old in enumerate(kept_ids)}
    remap = lambda value: (None if value is None else [new[int(v)] for v in value] if isinstance(value, list) else new[int(value)])
    model.config.vocab_size = len(kept_ids)
    for key in ("eos_token_id", "pad_token_id", "bos_token_id"):
        if getattr(model.config, key, None) is not None:
            setattr(model.config, key, remap(getattr(model.config, key)))
        if getattr(model.generation_config, key, None) is not None:
            setattr(model.generation_config, key, remap(getattr(model.generation_config, key)))
    model.requires_grad_(False)
    after = count_parameters(model)
    if after != trimmed_parameters({"hidden_size": model.config.hidden_size, "vocab_size": vocab,
                                    "tie_word_embeddings": tied}, len(kept_ids), before):
        raise ValueError("Unexpected parameter count after trimming")
    manifest = {"vocab_trim": {"source_checkpoint": str(source), "source_checkpoint_sha256": source_hash,
                               "original_vocab_size": vocab, "kept_vocab_size": len(kept_ids), "tied": tied,
                               "parameters_before": before, "parameters_after": after, **(provenance or {})}}
    save_checkpoint(model, tokenizer, destination, manifest)
    write_json(destination / MAP_FILE, {"kept_old_ids": kept_ids, "original_vocab_size": vocab,
                                        "sha256": hashlib.sha256(json.dumps(kept_ids).encode()).hexdigest(),
                                        "source_checkpoint_sha256": source_hash, **(provenance or {})})
    del model
    return {"parameters_before": before, "parameters_after": after, "kept_vocab_size": len(kept_ids),
            "original_vocab_size": vocab, "source_checkpoint_sha256": source_hash}


def load_trimmed(directory, device="cpu", dtype="float32"):
    """Load a trimmed export with its wrapped tokenizer."""
    from .modeling import load_model
    model, tokenizer = load_model(str(directory), device=device, dtype=dtype)
    mapping = json.loads((Path(directory) / MAP_FILE).read_text())
    return model, TrimmedTokenizer(tokenizer, mapping["kept_old_ids"])


def logit_check(source, trimmed_dir, examples, device="cpu"):
    """Trimmed logits must equal the original logits restricted to kept ids, on prompts whose tokens are kept."""
    import torch
    from .data import prompt_ids
    from .modeling import load_model
    original, tokenizer = load_model(str(source), device=device, dtype="float32")
    model, wrapped = load_trimmed(trimmed_dir, device)
    kept = torch.tensor(wrapped.kept_ids)
    results = []
    with torch.inference_mode():
        for example in examples:
            ids = list(prompt_ids(tokenizer, example.prompt))
            if any(i not in wrapped._kept for i in ids):
                results.append({"key": example.key, "skipped": "prompt uses a removed token"})
                continue
            expected = original(input_ids=torch.tensor([ids], device=device), use_cache=False).logits[0][:, kept.to(device)]
            actual = model(input_ids=torch.tensor([wrapped.map_ids(ids)], device=device), use_cache=False).logits[0]
            results.append({"key": example.key, "bit_identical": bool(torch.equal(actual, expected)),
                            "max_abs_difference": float((actual - expected).abs().max())})
    checked = [r for r in results if "skipped" not in r]
    del original, model
    return {"prompts": results, "checked": len(checked),
            "passed": bool(checked) and all(r["bit_identical"] for r in checked),
            "max_abs_difference": max((r["max_abs_difference"] for r in checked), default=None)}


def output_name(variant):
    return "vocab-unpruned" if variant == "unpruned-recovery" else f"vocab-{variant}"


def trim_run(source, output, variants, seeds=None, check_prompts=4):
    """Trim and evaluate ``variants`` for every seed of a multi-seed run with kept exports.

    The kept set is built per seed from that seed's recovery and calibration data. Writes
    ``<output>/seed-<s>/<vocab-name>/{export,result.json}`` and a campaign-shaped seed directory
    whose candidates are the trimmed models plus the source records (same ids), so
    ``multi-seed aggregate`` and ``compare`` work unchanged. Dev only; the test split is never read.
    """
    from .campaign import _evaluate, _fingerprint, _selection, checkpoint_identity, read_campaign_config
    from .support_data import read_support_jsonl
    from transformers import AutoTokenizer
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("Provide distinct variants")
    directories = sorted((p for p in Path(source).glob("seed-*") if p.is_dir() and p.name[5:].isdigit()),
                         key=lambda p: int(p.name[5:]))
    if seeds:
        wanted = {f"seed-{s}" for s in seeds}
        if wanted - {p.name for p in directories}:
            raise ValueError(f"Seeds not found in {source}: {sorted(wanted - {p.name for p in directories})}")
        directories = [p for p in directories if p.name in wanted]
    if not directories:
        raise ValueError(f"No seed-* campaign directories under {source}")
    missing = []
    for seed_dir in directories:
        complete = {c["id"] for c in _selection(seed_dir)["candidates"] if c.get("status") == "complete"}
        missing += [f"{seed_dir.name}/{v}" for v in variants if v not in complete]
    if missing:
        raise ValueError(f"Variants without a complete export record: {missing}")
    output = Path(output)
    summary = []
    for seed_dir in directories:
        selection = _selection(seed_dir)
        config = read_campaign_config(seed_dir / "config.json")
        splits = {name: read_support_jsonl(seed_dir / "data" / f"{name}.jsonl") for name in ("recovery", "calibration", "dev")}
        records = {c["id"]: c for c in selection["candidates"] if c.get("status") == "complete"}
        target = output / seed_dir.name
        target.mkdir(parents=True, exist_ok=True)
        candidates = []
        for variant in variants:
            record = records[variant]
            export = Path(record["checkpoint"])
            identity = checkpoint_identity(export)
            if identity != record["checkpoint_sha256"]:
                raise ValueError(f"{seed_dir.name}/{variant}: export hash differs from its record")
            candidates.append(dict(record))
            name = output_name(variant)
            directory = target / name
            result_path = directory / "result.json"
            if result_path.exists() and json.loads(result_path.read_text()).get("source_checkpoint_sha256") != identity:
                shutil.rmtree(directory)
            if result_path.exists():
                candidates.append(json.loads(result_path.read_text()))
                continue
            print(f"Trim vocabulary {seed_dir.name}/{variant} -> {name}", flush=True)
            if (directory / "export").exists():
                shutil.rmtree(directory / "export")
            tokenizer = AutoTokenizer.from_pretrained(str(export), trust_remote_code=False, fix_mistral_regex=False)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            kept, provenance = build_kept_set(tokenizer, {"recovery": splits["recovery"], "calibration": splits["calibration"]})
            trimmed = trim_export(export, directory / "export", kept, {"kept_set": provenance})
            check = logit_check(export, directory / "export", splits["calibration"][:check_prompts]) if check_prompts else None
            statistics = fallback_statistics(tokenizer, kept, splits["dev"])
            model, wrapped = load_trimmed(directory / "export", config["device"], config["dtype"])
            # Fallback re-splits can lengthen a few prompts/replies past max_length, which only guards the
            # loss pass (nothing is truncated); raise it just enough. Generation keeps max_new_tokens.
            from .data import prompt_ids
            needed = max(len(prompt_ids(wrapped, ex.prompt)) + len(wrapped.encode(ex.answer, add_special_tokens=False)) + 1
                         for ex in splits["dev"])
            eval_config = dict(config, max_length=max(config["max_length"], needed))
            wrapped.fallback_tokens = 0
            try:
                metrics = _evaluate(model, wrapped, splits["dev"], eval_config)
            finally:
                del model
            result = {"id": name, "status": "complete", "variant": variant, "source_checkpoint": str(export),
                      "source_checkpoint_sha256": identity, "checkpoint": str(directory / "export"),
                      "parameters": trimmed["parameters_after"],
                      "actual_reduction": 1 - trimmed["parameters_after"] / selection["baseline"]["parameters"],
                      "source_parameters": record["parameters"], "trim": trimmed, "kept_set": provenance,
                      "logit_check": check, "dev_fallback": statistics,
                      "runtime_fallback_tokens": wrapped.fallback_tokens, "evaluation_max_length": eval_config["max_length"],
                      "dev_examples_over_config_max_length": sum(
                          len(prompt_ids(wrapped, ex.prompt)) + len(wrapped.encode(ex.answer, add_special_tokens=False)) + 1
                          > config["max_length"] for ex in splits["dev"]),
                      "loss_note": "answer_token_loss is normalized over the kept vocabulary and uses fallback "
                                   "tokenization, so it is not comparable with untrimmed models",
                      "dev": metrics}
            write_json(result_path, result)
            candidates.append(result)
        for name in ("config.json", "state.json", "selection-dev.json"):
            if (seed_dir / name).exists():
                shutil.copyfile(seed_dir / name, target / name)
        assembled = {"selected_on": "development_only", "test_used_for_selection": False,
                     "baseline": selection["baseline"], "candidates": candidates, "reference_id": "original",
                     "selection_score_name": "token_f1", "tiers": [],
                     "source_run": str(Path(source).resolve() / seed_dir.name),
                     "note": "Assembled by prune-lab trim-vocab: trimmed exports plus their source records."}
        write_json(target / "selection.json", assembled)
        write_json(target / "selection-lock.json", {"sha256": _fingerprint(assembled)})
        summary.append({"seed_dir": str(target), "candidates": [c["id"] for c in candidates]})
    write_json(output / "trim-vocab-run.json", {"source": str(Path(source).resolve()), "variants": variants, "seeds": summary})
    return summary
