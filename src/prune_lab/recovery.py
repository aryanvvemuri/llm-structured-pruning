"""Native MLP LoRA recovery. Successful recovery merges the best dev adapter.

Base weights remain unchanged until final merge; failures remove every adapter.
Only the best checkpoint's adapter tensors are kept, on CPU.
"""
import math
import random
import time

import torch
from torch import nn

from .modeling import mlp_layers
from .metrics import selection_score, score_name
from .recipes import lr_factor, validate_schedule


# Release cached MPS blocks after every optimizer update (N=1) and each assessment. Variable-length
# batches otherwise make the MPS allocator cache ever more block sizes (driver memory reached
# 16-19 GB over 450 updates on a 24 GB machine). Frees cache only: no numerical effect; measured
# with a teacher over 160 updates: peak driver 18.0 GB -> 8.2 GB (N=1), 11.0 GB (N=10), same s/update.
MPS_EMPTY_CACHE_EVERY = 1


def _mps_housekeeping(stats, device, completed=None):
    """Track peak MPS driver memory in ``stats`` and periodically empty the allocator cache."""
    if device.type != 'mps':
        return
    stats['peak_mps_driver_bytes'] = max(stats.get('peak_mps_driver_bytes', 0), torch.mps.driver_allocated_memory())
    if completed is None or (MPS_EMPTY_CACHE_EVERY and completed % MPS_EMPTY_CACHE_EVERY == 0):
        torch.mps.empty_cache()


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA needs a Linear, positive rank and finite positive alpha")
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.base.requires_grad_(False)

    def forward(self, inputs):
        return self.base(inputs) + (inputs @ self.A.t() @ self.B.t()) * self.scale

    @torch.no_grad()
    def merge(self):
        self.base.weight.add_(self.B @ self.A, alpha=self.scale)
        return self.base


def attention_modules(model):
    """Self-attention blocks of a dense Qwen2 model, with Linear q/k/v/o projections."""
    modules = [layer.self_attn for layer in model.model.layers]
    for module in modules:
        if not all(isinstance(getattr(module, name, None), nn.Linear) for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj')):
            raise ValueError('Attention LoRA needs Linear q_proj/k_proj/v_proj/o_proj in every layer')
    return modules


def _metric(metrics):
    accuracy = selection_score(metrics)
    loss = float(metrics.get('answer_token_loss', metrics.get('loss', float('nan'))))
    if not math.isfinite(accuracy) or not math.isfinite(loss) or loss < 0 or not 0 <= accuracy <= 1:
        raise ValueError('Development score and loss must be finite, score in [0, 1], loss >= 0')
    return accuracy, -loss


def distillation_loss(student_logits, teacher_logits, labels, ce, alpha, temperature):
    """alpha * CE + (1 - alpha) * T^2 * mean over answer tokens of KL(teacher_T || student_T)."""
    mask = labels[:, 1:] != -100
    student = torch.log_softmax(student_logits[:, :-1][mask].float() / temperature, dim=-1)
    teacher = torch.log_softmax(teacher_logits[:, :-1][mask].float() / temperature, dim=-1)
    kl = (teacher.exp() * (teacher - student)).sum(-1).mean()
    return alpha * ce + (1 - alpha) * temperature ** 2 * kl, kl


def recover_model(model, train_batches, evaluate_dev, config, teacher=None):
    """Train adapters on answer-only batches and merge the best development state.

    ``evaluate_dev(model)`` returns selection_score (or accuracy) and answer_token_loss (or loss).
    ``max_steps`` counts optimizer updates. Data are shuffled per epoch and a
    final partial accumulation group is normalized by its actual target tokens.
    Step zero participates in selection, so recovery can retain the original.
    Evaluation callbacks may change mode/gradient flags; these are restored.
    """
    rank, alpha = config.get('rank', 8), config.get('alpha', 16)
    steps = config.get('max_steps', 200)
    accumulation = config.get('gradient_accumulation', 4)
    lr, clip = config.get('learning_rate', 2e-4), config.get('max_grad_norm', 1.0)
    checkpoints = set(config.get('checkpoints', [50, 100, 200])) | {steps}
    if any(type(v) is not int or v < 1 for v in (rank, steps, accumulation)):
        raise ValueError('rank, max_steps, gradient_accumulation must be positive integers')
    if any(type(v) is not int or v < 1 for v in checkpoints):
        raise ValueError('checkpoints must be positive integer optimizer steps')
    if any(not math.isfinite(v) or v <= 0 for v in (alpha, lr, clip)):
        raise ValueError('alpha, learning_rate and max_grad_norm must be positive and finite')
    from .recipes import validate_checkpoints
    validate_checkpoints(config)
    schedule, warmup, min_ratio = validate_schedule(config)
    from .recipes import validate_distillation  # 2b; kept local so components stay separable
    distill = validate_distillation(config)
    if teacher is not None and distill is None:
        raise ValueError('A teacher requires recovery.distillation settings')
    if teacher is not None and any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('The distillation teacher must be frozen')
    batches = list(train_batches)
    if not batches:
        raise ValueError('Recovery requires supervised batches')
    token_counts = []
    for batch in batches:
        labels = batch['labels']
        if labels.ndim != 2 or labels.shape[0] != 1:
            raise ValueError('Recovery requires batch size one')
        n = int((labels[:, 1:] != -100).sum())
        if not n or int(labels[0, 0]) != -100:
            raise ValueError('Recovery batches require answer tokens and masked prompt labels')
        ids = batch['input_ids']
        if ids.shape != labels.shape or not torch.equal(ids[labels != -100], labels[labels != -100]):
            raise ValueError('Recovery labels must match input answer tokens')
        supervised_positions = (labels[0] != -100).nonzero().flatten()
        if int(supervised_positions[-1] - supervised_positions[0] + 1) != n:
            raise ValueError('Recovery requires a contiguous answer span after the masked prompt')
        mask = batch.get('attention_mask')
        if mask is not None and (mask.shape != labels.shape or bool(((mask == 0) & (labels != -100)).any())):
            raise ValueError('Padding must not contribute recovery labels')
        token_counts.append(n)
    mlps = mlp_layers(model)
    from .recipes import validate_lora_targets  # 2c; kept local so components stay separable
    targets = validate_lora_targets(config)
    if targets == 'mlp+attention':
        attention_modules(model)  # Validate before any adapter is attached.
    original_flags = [(p, p.requires_grad) for p in model.parameters()]
    original_modes = [(m, m.training) for m in model.modules()]
    old_cache = getattr(model.config, 'use_cache', None)
    old_checkpointing = bool(getattr(model, 'is_gradient_checkpointing', False))
    wrappers, input_hook = [], None
    seed = config.get('seed', 20261113)
    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    best_snapshot, best_metrics, best_step = None, None, 0
    history = []
    completed = 0
    optimizer = None
    try:
        model.requires_grad_(False)
        model.config.use_cache = False
        for mlp in mlps:
            for name in ('gate_proj', 'up_proj', 'down_proj'):
                adapter = LoRALinear(getattr(mlp, name), rank, alpha)
                wrappers.append((mlp, name, adapter))
                setattr(mlp, name, adapter)
        if targets == 'mlp+attention':
            # Created after the MLP adapters, so MLP adapter initialization matches the MLP-only recipe.
            for attention in attention_modules(model):
                for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                    adapter = LoRALinear(getattr(attention, name), rank, alpha)
                    wrappers.append((attention, name, adapter))
                    setattr(attention, name, adapter)
        parameters = [p for _, _, module in wrappers for p in (module.A, module.B)]
        if config.get('gradient_checkpointing', False) and not old_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if bool(getattr(model, 'is_gradient_checkpointing', False)):
            input_hook = model.get_input_embeddings().register_forward_hook(
                lambda module, args, output: output.requires_grad_(True))
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
        device = next(model.parameters()).device

        timing = {'training': 0.0, 'assessment': 0.0}
        distill_log = {'ce': [], 'kl': []}
        if teacher is not None:
            teacher.eval()
        learning_rates = []

        def assess(step):
            nonlocal best_snapshot, best_metrics, best_step
            assess_start = time.perf_counter()
            modes = [(module, module.training) for module in model.modules()]
            flags = [(p, p.requires_grad) for p in model.parameters()]
            try:
                with torch.no_grad():
                    metrics = evaluate_dev(model)
                score = _metric(metrics)
            finally:
                for module, mode in modes:
                    module.training = mode
                for p, flag in flags:
                    p.requires_grad_(flag)
                model.config.use_cache = False
            history.append({'step': step, 'dev': metrics,
                            'learning_rate': learning_rates[-1] if learning_rates else None})
            print(f"Recovery step {step}/{steps}: dev {score_name(metrics)} {100 * score[0]:.2f}/100, answer loss {-score[1]:.4f}", flush=True)
            if best_metrics is None or score > _metric(best_metrics):
                best_metrics, best_step = metrics, step
                best_snapshot = [p.detach().cpu().clone() for p in parameters]
            timing['assessment'] += time.perf_counter() - assess_start
            _mps_housekeeping(timing, next(model.parameters()).device)
            return metrics

        initial = assess(0)
        train_losses = []
        while completed < steps:
            order = list(range(len(batches)))
            rng.shuffle(order)
            for start in range(0, len(order), accumulation):
                update_start = time.perf_counter()
                group = order[start:start + accumulation]
                total_tokens = sum(token_counts[i] for i in group)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                group_loss = group_ce = group_kl = 0.0
                for i in group:
                    batch = {k: v.to(device) for k, v in batches[i].items()}
                    with torch.enable_grad():
                        output = model(**batch)
                        loss = output.loss
                        if teacher is not None:
                            with torch.no_grad():
                                teacher_logits = teacher(input_ids=batch['input_ids'], attention_mask=batch.get('attention_mask'),
                                                         use_cache=False).logits
                            ce = loss
                            loss, kl = distillation_loss(output.logits, teacher_logits, batch['labels'], ce, *distill)
                            group_ce += float(ce.detach()) * token_counts[i] / total_tokens
                            group_kl += float(kl.detach()) * token_counts[i] / total_tokens
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError('Nonfinite recovery loss')
                        weighted_loss = loss * (token_counts[i] / total_tokens)
                        weighted_loss.backward()
                    group_loss += float(loss.detach()) * token_counts[i] / total_tokens
                torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
                if schedule != 'constant':  # The constant default never touches the optimizer's rate.
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr * lr_factor(completed, steps, schedule, warmup, min_ratio)
                learning_rates.append(optimizer.param_groups[0]['lr'])
                optimizer.step()
                # Reduce on device before one host check; per-tensor bool calls
                # otherwise force hundreds of MPS synchronizations per update.
                if not bool(torch.stack([torch.isfinite(p).all() for p in parameters]).all()):
                    raise FloatingPointError('Nonfinite adapter parameter')
                completed += 1
                train_losses.append(group_loss)
                if teacher is not None:
                    distill_log['ce'].append(group_ce)
                    distill_log['kl'].append(group_kl)
                # The finiteness check above already synchronized the device.
                timing['training'] += time.perf_counter() - update_start
                _mps_housekeeping(timing, device, completed)
                if completed in checkpoints:
                    assess(completed)
                if completed == steps:
                    break
        with torch.no_grad():
            for p, best in zip(parameters, best_snapshot):
                p.copy_(best)
        # Validate every delta before touching any base weights.
        for _, _, adapter in wrappers:
            with torch.no_grad():
                delta = (adapter.B @ adapter.A) * adapter.scale
                if not bool(torch.isfinite(adapter.base.weight + delta).all()):
                    raise FloatingPointError('Nonfinite merged weight')
        for _, _, adapter in wrappers:
            adapter.merge()
        return {'peak_mps_driver_bytes': timing.get('peak_mps_driver_bytes'), 'mps_empty_cache_every': MPS_EMPTY_CACHE_EVERY,
                'initial_dev': initial, 'best_dev': best_metrics, 'best_step': best_step,
                'history': history, 'optimizer_steps': completed,
                'trainable_parameters': sum(p.numel() for p in parameters),
                'train_losses': train_losses, 'merged': True, 'seed': seed,
                'learning_rates': learning_rates, 'training_seconds': timing['training'],
                'assessment_seconds': timing['assessment'],
                'recipe': {'lora_targets': targets, 'lr_schedule': schedule, 'peak_learning_rate': lr, 'warmup_steps': warmup,
                           'min_lr_ratio': min_ratio, 'max_steps': steps, 'checkpoints': sorted(checkpoints),
                           'gradient_accumulation': accumulation, 'rank': rank, 'alpha': alpha},
                **({'distillation': {'alpha': distill[0], 'temperature': distill[1],
                                     'train_ce_losses': distill_log['ce'], 'train_kl_losses': distill_log['kl']}}
                   if teacher is not None else {})}
    finally:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if input_hook is not None:
            input_hook.remove()
        if not old_checkpointing and bool(getattr(model, 'is_gradient_checkpointing', False)):
            model.gradient_checkpointing_disable()
        for mlp, name, adapter in wrappers:
            setattr(mlp, name, adapter.base)
        for p, flag in original_flags:
            p.requires_grad_(flag)
        for module, mode in original_modes:
            module.training = mode
        if old_cache is None:
            if hasattr(model.config, 'use_cache'):
                delattr(model.config, 'use_cache')
        else:
            model.config.use_cache = old_cache
