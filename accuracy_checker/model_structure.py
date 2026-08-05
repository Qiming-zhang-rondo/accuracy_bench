"""Model structure discovery shared by boundary, L1 and L2 flows.

The checker only needs a small set of structural capabilities.  Discovering
those capabilities from the loaded module tree is more robust than maintaining
one class hierarchy per marketing model name, especially for multimodal
wrappers whose text model may live under more than one intermediate module.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn


_CONTAINER_ATTRS = (
    "language_model",
    "model",
    "transformer",
    "decoder",
    "encoder",
)
_LAYER_ATTRS = ("layers", "h", "blocks")


def _safe_getattr(obj, name: str):
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def iter_model_containers(model: nn.Module) -> Iterator[nn.Module]:
    """Yield likely text-model containers in deterministic breadth-first order."""
    queue = [model]
    seen = set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for attr in _CONTAINER_ATTRS:
            child = _safe_getattr(current, attr)
            if isinstance(child, nn.Module) and id(child) not in seen:
                queue.append(child)


def _find_layers(container: nn.Module) -> Optional[nn.ModuleList]:
    for attr in _LAYER_ATTRS:
        layers = _safe_getattr(container, attr)
        if isinstance(layers, (nn.ModuleList, list, tuple)):
            return layers
    return None


def get_text_model(model: nn.Module) -> nn.Module:
    """Return the innermost text container that owns decoder layers."""
    for container in iter_model_containers(model):
        if _find_layers(container) is not None:
            return container
    raise ValueError(f"Cannot find decoder layers in {type(model)}")


def _find_component(containers, names, preferred=None):
    ordered = []
    if preferred is not None:
        ordered.append(preferred)
    ordered.extend(c for c in containers if c is not preferred)
    for container in ordered:
        for name in names:
            value = _safe_getattr(container, name)
            if isinstance(value, nn.Module):
                return value
    return None


@dataclass(frozen=True)
class ModelComponents:
    """The model components used by acc_bench core flows."""

    text_model: nn.Module
    layers: nn.ModuleList
    embed: Optional[nn.Module]
    final_norm: Optional[nn.Module]
    lm_head: Optional[nn.Module]
    rotary_emb: Optional[nn.Module]
    visual: Optional[nn.Module]


def get_model_components(model: nn.Module) -> ModelComponents:
    """Resolve text-model components without relying on a model-name registry.

    Covered layouts include, among others:

    - CausalLM: ``model.model.layers``
    - Qwen3.6 VLM: ``model.model.language_model.layers``
    - Kimi K3: ``model.language_model.model.layers``
    - legacy GLM: ``model.transformer.layers`` / ``.h``
    """
    containers = list(iter_model_containers(model))
    text_model = get_text_model(model)
    layers = _find_layers(text_model)
    if layers is None:
        raise ValueError(f"Cannot find decoder layers in {type(text_model)}")

    return ModelComponents(
        text_model=text_model,
        layers=layers,
        embed=_find_component(
            containers, ("embed_tokens", "embedding", "word_embeddings"), text_model
        ),
        final_norm=_find_component(
            containers, ("norm", "final_layernorm", "ln_f"), text_model
        ),
        lm_head=_find_component(
            containers, ("lm_head", "output_proj", "output_layer")
        ),
        rotary_emb=_find_component(containers, ("rotary_emb",), text_model),
        visual=_find_component(
            containers, ("visual", "vision_tower", "vision_model")
        ),
    )


def get_text_config(config):
    """Return the nested text config used by multimodal wrappers, if present."""
    return _safe_getattr(config, "text_config") or config


def get_moe_module(layer: nn.Module) -> Tuple[Optional[str], Optional[nn.Module]]:
    """Return ``(attribute_name, module)`` for the layer's routed MoE block."""
    for name in ("mlp", "block_sparse_moe", "moe"):
        module = _safe_getattr(layer, name)
        if module is None:
            continue
        if _safe_getattr(module, "experts") is not None and (
            _safe_getattr(module, "gate") is not None
            or _safe_getattr(module, "router") is not None
        ):
            return name, module
    return None, None


def uses_attention_residuals(layer: nn.Module) -> bool:
    """Whether a decoder layer carries Kimi-style block residual state."""
    return bool(_safe_getattr(layer, "use_attn_residuals"))


def get_layer_state_kwarg(layer: nn.Module) -> Optional[str]:
    """Return the cross-layer state keyword understood by this decoder layer."""
    if uses_attention_residuals(layer):
        return "block_residual"
    attn = _safe_getattr(layer, "self_attn")
    if attn is not None and _safe_getattr(attn, "indexer") is not None:
        return "prev_topk_indices"
    try:
        if "prev_topk_indices" in inspect.signature(layer.forward).parameters:
            return "prev_topk_indices"
    except (TypeError, ValueError):
        pass
    return None


def is_kimi_k3_layer(layer: nn.Module) -> bool:
    """Detect Kimi K3 from stable structural markers, not class-name strings."""
    attn = _safe_getattr(layer, "self_attn")
    moe = _safe_getattr(layer, "block_sparse_moe")
    return bool(
        uses_attention_residuals(layer)
        or (
            moe is not None
            and _safe_getattr(moe, "routed_expert_down_proj") is not None
            and _safe_getattr(moe, "shared_experts") is not None
        )
        or (
            attn is not None
            and _safe_getattr(attn, "q_conv1d") is not None
            and _safe_getattr(attn, "f_a_proj") is not None
        )
    )


def has_indexed_routed_experts(layer: nn.Module) -> bool:
    """Whether routed experts are represented as one module per expert."""
    _, moe = get_moe_module(layer)
    return bool(moe is not None and isinstance(_safe_getattr(moe, "experts"), nn.ModuleList))


def build_replay_attention_mask(
    layer: nn.Module,
    hidden_states: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Build the additive causal mask required by eager Kimi/Qwen attention.

    Kimi KDA consumes a 2-D padding mask and therefore ignores this 4-D mask;
    Kimi MLA and Qwen3.6 full-attention layers consume it directly.  Keeping
    this helper structural avoids changing legacy GLM replay behavior.
    """
    attn = _safe_getattr(layer, "self_attn")
    if attn is None:
        return None
    _, moe = get_moe_module(layer)
    is_qwen_full_attention = bool(
        moe is not None
        and _safe_getattr(attn, "q_proj") is not None
        and _safe_getattr(attn, "q_a_proj") is None
    )
    if not (is_kimi_k3_layer(layer) or is_qwen_full_attention):
        return None

    batch_size, seq_len = hidden_states.shape[:2]
    min_value = torch.finfo(hidden_states.dtype).min
    mask = torch.full(
        (seq_len, seq_len), min_value,
        dtype=hidden_states.dtype, device=hidden_states.device,
    )
    mask = torch.triu(mask, diagonal=1)
    return mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, -1, -1)
