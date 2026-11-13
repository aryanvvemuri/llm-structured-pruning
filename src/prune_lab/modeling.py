"""Load supported dense Qwen models without remote code execution."""
from __future__ import annotations

import torch
from copy import deepcopy
from numbers import Integral
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def layer_widths(config):
    widths = getattr(config, "prune_lab_layer_widths", None)
    if widths is None:
        return None
    if (not isinstance(widths, list) or len(widths) != config.num_hidden_layers
            or any(not isinstance(w, Integral) or isinstance(w, bool) or w <= 0 for w in widths)):
        raise ValueError("Invalid per-layer MLP widths in checkpoint config")
    return widths


class CompactQwen2ForCausalLM(Qwen2ForCausalLM):
    """Local architecture constructor; never requires checkpoint remote code.

    HF's low-memory from_pretrained constructs these modules on meta before
    loading safetensors. No dense checkpoint or second state dict is loaded.
    """

    def __init__(self, config):
        if config.model_type != "qwen2":
            raise ValueError("Only model_type=qwen2 is supported")
        widths = layer_widths(config)
        local_config = deepcopy(config)
        if widths is not None:
            local_config.intermediate_size = min(widths)
        super().__init__(local_config)
        local_config.intermediate_size = config.intermediate_size
        if widths is not None:
            for layer, width in zip(self.model.layers, widths):
                if layer.mlp.intermediate_size != width:
                    mlp_config = deepcopy(local_config)
                    mlp_config.intermediate_size = width
                    layer.mlp = Qwen2MLP(mlp_config)
                    layer.mlp.apply(self._init_weights)


def resolve_device(device: str = "auto") -> str:
    if device == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if device not in {"cpu", "mps"}:
        raise ValueError("device must be auto, cpu, or mps")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; select --device cpu explicitly")
    return device


def mlp_layers(model):
    if getattr(model.config, "model_type", None) != "qwen2":
        raise ValueError("Only dense Qwen2/Qwen2.5 (model_type=qwen2) models are supported")
    layers = list(model.model.layers)
    if not layers:
        raise ValueError("Model has no decoder layers")
    for layer in layers:
        mlp = layer.mlp
        if not all(isinstance(getattr(mlp, name, None), torch.nn.Linear)
                   for name in ("gate_proj", "up_proj", "down_proj")):
            raise ValueError("Expected dense gated MLP projections")
    return [layer.mlp for layer in layers]


def load_model(model_id: str = DEFAULT_MODEL, device: str = "auto",
               dtype: str = "float32", cache_dir: str | None = None):
    resolved = resolve_device(device)
    types = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}
    if dtype not in types:
        raise ValueError(f"Unsupported dtype: {dtype}")
    config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir,
                                        trust_remote_code=False)
    if config.model_type != "qwen2":
        raise ValueError("Only dense Qwen2/Qwen2.5 (model_type=qwen2) models are supported")
    widths = layer_widths(config)
    # Transformers 4.57.x can misidentify locally saved Qwen tokenizers as
    # Mistral. Preserve the original Qwen regex rather than applying that fix.
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir,
                                              trust_remote_code=False,
                                              fix_mistral_regex=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    constructor = CompactQwen2ForCausalLM if widths is not None else AutoModelForCausalLM
    model = constructor.from_pretrained(
        model_id, cache_dir=cache_dir, torch_dtype=types[dtype],
        config=config, use_safetensors=True, low_cpu_mem_usage=True,
        trust_remote_code=False, attn_implementation="eager")
    mlp_layers(model)
    model.to(resolved).eval()
    model.requires_grad_(False)
    return model, tokenizer


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
