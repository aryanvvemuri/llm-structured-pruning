"""Pure (torch-free) validation and schedules for opt-in recovery recipes.

Every option is absent by default; absent options reproduce the original recipe.
"""
import math


LR_SCHEDULES = ('constant', 'cosine')


def validate_schedule(config):
    """Validate the opt-in learning-rate schedule keys (absent keys keep the constant default)."""
    schedule = config.get('lr_schedule', 'constant')
    if schedule not in LR_SCHEDULES:
        raise ValueError(f'lr_schedule must be one of {LR_SCHEDULES}')
    steps = config.get('max_steps', 200)
    warmup = config.get('warmup_steps', 0)
    ratio = config.get('min_lr_ratio', 0.0)
    if type(warmup) is not int or warmup < 0 or (type(steps) is int and warmup >= steps):
        raise ValueError('warmup_steps must be a non-negative integer smaller than max_steps')
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError('min_lr_ratio must be in [0, 1]')
    if schedule == 'constant' and ('warmup_steps' in config or 'min_lr_ratio' in config):
        raise ValueError('warmup_steps/min_lr_ratio apply only to lr_schedule "cosine"')
    return schedule, warmup, float(ratio)


def lr_factor(update, steps, schedule='constant', warmup=0, min_ratio=0.0):
    """Multiplier of the peak rate for 0-based optimizer update ``update``.

    cosine: linear warmup (update+1)/warmup for the first ``warmup`` updates, then
    min_ratio + (1 - min_ratio) * (1 + cos(pi * (update - warmup) / (steps - warmup))) / 2,
    so the first post-warmup update uses the peak and the rate reaches min_ratio at ``steps``.
    """
    if schedule == 'constant':
        return 1.0
    if update < warmup:
        return (update + 1) / warmup
    progress = (update - warmup) / (steps - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def validate_distillation(recovery):
    """Return (alpha, temperature) or None. Loss: alpha*CE + (1-alpha)*T^2*KL(teacher_T || student_T)."""
    settings = recovery.get('distillation')
    if settings is None:
        return None
    if not isinstance(settings, dict) or set(settings) != {'alpha', 'temperature'}:
        raise ValueError('distillation must be {"alpha": a, "temperature": T}')
    alpha, temperature = settings['alpha'], settings['temperature']
    for value in (alpha, temperature):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('distillation alpha and temperature must be finite numbers')
    if not 0 <= alpha < 1 or temperature <= 0:
        raise ValueError('distillation needs 0 <= alpha < 1 and temperature > 0')
    return float(alpha), float(temperature)


def validate_recipe(recovery):
    """Config-time validation of all opt-in recovery recipe keys."""
    validate_schedule(recovery)
    validate_distillation(recovery)
