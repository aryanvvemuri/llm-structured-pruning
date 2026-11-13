"""Generation accuracy, answer-token loss and synchronized process-local timing."""
import re
import time
from decimal import Decimal, InvalidOperation
from .data import make_batches, prompt_ids

NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"


def numeric_answer(text):
    """Accept a numeric-only completion or explicit final-answer marker, never a question echo."""
    text = text.strip()
    if "####" in text:
        text = text.rsplit("####", 1)[1].strip()
    else:
        marked = re.search(r"(?:final answer|the answer is|answer)\s*[:=]?\s*(" + NUMBER + r")\s*[.!]?\s*$", text, re.I)
        if marked:
            text = marked.group(1)
    match = re.fullmatch(r"\s*\$?(" + NUMBER + r")\s*[.!]?\s*", text)
    if not match:
        return None
    try:
        return Decimal(match.group(1).replace(",", ""))
    except InvalidOperation:
        return None


def synchronize(device):
    import torch
    if str(device).startswith("mps"):
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def memory_snapshot(device):
    import psutil
    import torch
    result = {"process_rss_bytes": psutil.Process().memory_info().rss}
    if str(device).startswith("mps"):
        result["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
        result["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    if str(device).startswith("cuda"):
        result["cuda_allocated_bytes"] = torch.cuda.memory_allocated(device)
    return result


def evaluate_model(model, tokenizer, examples, max_length=256, max_new_tokens=32):
    import torch
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    device = next(model.parameters()).device
    model.eval()
    batches = make_batches(examples, tokenizer, max_length=max_length)
    loss_sum, supervised = 0.0, 0
    with torch.inference_mode():
        for batch in batches:
            batch = {k: v.to(device) for k, v in batch.items()}
            n = int((batch["labels"][:, 1:] != -100).sum())
            output = model(**batch)
            loss_sum += float(output.loss) * n
            supervised += n
        def generate(ex):
            ids = torch.tensor([prompt_ids(tokenizer, ex.prompt)], device=device)
            result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, use_cache=True)
            return result[0, ids.shape[1]:]
        # Same generation configuration, one unmeasured warmup on development/evaluation input.
        generate(examples[0])
        synchronize(device)
        rows, elapsed, tokens, correct = [], 0.0, 0, 0
        for ex in examples:
            synchronize(device)
            start = time.perf_counter()
            generated = generate(ex)
            synchronize(device)
            seconds = time.perf_counter() - start
            text = tokenizer.decode(generated, skip_special_tokens=True)
            prediction, expected = numeric_answer(text), numeric_answer(ex.answer)
            if expected is None:
                raise ValueError(f"Unparseable reference numeric answer: {ex.key}")
            success = prediction is not None and prediction == expected
            correct += int(success)
            elapsed += seconds
            tokens += len(generated)
            rows.append({"key": ex.key, "prediction": text, "expected": str(expected), "correct": success, "seconds": seconds, "generated_tokens": len(generated)})
    return {"count": len(examples), "correct": correct, "accuracy": correct / len(examples), "answer_token_loss": loss_sum / supervised, "supervised_tokens": supervised, "generation_seconds": elapsed, "generated_tokens": tokens, "tokens_per_second": tokens / elapsed if elapsed else None, "mean_example_seconds": elapsed / len(examples), "warmup_examples": 1, "small_sample_warning": len(examples) < 100, "memory": memory_snapshot(device), "generation_config": {"do_sample": False, "max_new_tokens": max_new_tokens, "batch_size": 1}, "predictions": rows}
