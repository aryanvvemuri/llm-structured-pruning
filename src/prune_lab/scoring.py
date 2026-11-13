"""Stream neuron statistics; retain at most one example's activation graph."""
from __future__ import annotations

import torch

from .modeling import mlp_layers


def score_neurons(model, batches, method: str = "taylor"):
    """Score post-gate channels over valid tokens (batch size one).

    Activation: mean |a|. Contribution: mean |a| times the L2 norm of
    the corresponding down-projection column. Taylor: mean |a*dL/da|,
    where L is the sum of supervised answer-token cross entropies.
    Summed loss removes the varying answer-length scale of HF's mean loss.
    """
    if method not in {"activation", "contribution", "taylor"}:
        raise ValueError("method must be activation, contribution, or taylor")
    mlps = mlp_layers(model)
    sums = {i: torch.zeros(m.down_proj.in_features, dtype=torch.float64)
            for i, m in enumerate(mlps)}
    norms = {i: m.down_proj.weight.detach().float().norm(dim=0).cpu().double()
             for i, m in enumerate(mlps)} if method == "contribution" else {}
    training = [(module, module.training) for module in model.modules()]
    grad_flags = [(p, p.requires_grad) for p in model.parameters()]
    use_cache = model.config.use_cache
    handles, activations = [], {}
    total_tokens = 0
    device = next(model.parameters()).device
    try:
        model.eval()
        model.requires_grad_(False)
        model.config.use_cache = False
        if method == "taylor":
            def enable_grad(module, args, output):
                return output.detach().requires_grad_(True)
            handles.append(model.get_input_embeddings().register_forward_hook(enable_grad))
        for index, mlp in enumerate(mlps):
            def capture(module, args, index=index):
                activations[index] = args[0]
            handles.append(mlp.down_proj.register_forward_pre_hook(capture))
        for batch in batches:
            activations.clear()
            inputs = {key: batch[key].to(device) for key in
                      ("input_ids", "attention_mask", "labels") if key in batch}
            ids = inputs["input_ids"]
            if ids.ndim != 2 or ids.shape[0] != 1:
                raise ValueError("Scoring requires batch size one")
            mask = inputs.get("attention_mask", torch.ones_like(ids)).bool()
            valid_count = int(mask.sum().item())
            if not valid_count:
                raise ValueError("Scoring example contains no valid tokens")
            supervised = 0
            if method == "taylor":
                if "labels" not in inputs:
                    raise ValueError("Taylor scoring requires answer-only labels")
                inputs["labels"] = inputs["labels"].clone().masked_fill(~mask, -100)
                supervised = int((inputs["labels"][:, 1:] != -100).sum().item())
                if not supervised:
                    raise ValueError("Taylor scoring requires at least one shifted supervised token")
            else:
                inputs.pop("labels", None)
            with torch.set_grad_enabled(method == "taylor"):
                output = model(**inputs, use_cache=False)
                if method == "taylor":
                    if not torch.isfinite(output.loss):
                        raise ValueError("Nonfinite supervised loss")
                    gradients = torch.autograd.grad(output.loss * supervised,
                                                    [activations[i] for i in range(len(mlps))])
                for index in range(len(mlps)):
                    values = activations[index].detach().float()
                    if method == "taylor":
                        values = values * gradients[index].detach().float()
                    statistic = values.abs()[mask].sum(dim=0).cpu().double()
                    if method == "contribution":
                        statistic *= norms[index]
                    if not torch.isfinite(statistic).all():
                        raise ValueError("Nonfinite neuron scores")
                    sums[index] += statistic
            total_tokens += valid_count
            activations.clear()
            del output, values, statistic
            if method == "taylor":
                del gradients
        if not total_tokens:
            raise ValueError("No scoring examples supplied")
        return {index: (value / total_tokens).float() for index, value in sums.items()}
    finally:
        activations.clear()
        for handle in handles:
            handle.remove()
        model.config.use_cache = use_cache
        for parameter, requires_grad in grad_flags:
            parameter.requires_grad_(requires_grad)
        for module, was_training in training:
            module.training = was_training
