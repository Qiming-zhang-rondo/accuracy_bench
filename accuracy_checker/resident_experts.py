"""Resident packed-expert execution for pure Transformers boundary runs.

The checkpoint keeps routed experts in W4/W8 form.  Keeping those compact
tensors on the NPU avoids expanding weights on CPU and copying BF16 expert
matrices for every decode step.  Only the active expert chunk is dequantized.
"""

from __future__ import annotations

import gc
import logging
import os
import re
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import nn

from .model_loader import (
    _MX_QUANT_TYPES,
    dequantize_weight_dynamic,
    dequantize_weight_mx,
    dequantize_weight_static,
    unpack_int4_to_int8,
)
from .utils import parse_base_name

logger = logging.getLogger(__name__)

_EXPERT_KEY = re.compile(r"^(?P<prefix>.*\.experts)\.(?P<eid>\d+)\.(?P<tail>.+)$")


@dataclass
class ResidentExpertStore:
    """Per-layer expert tensors stacked on their owning device."""

    prefix: str
    device: str
    num_experts: int
    tensors: dict[str, torch.Tensor]
    bytes_on_device: int
    cache_limit: int = 0
    bf16_cache: OrderedDict | None = None
    cache_hits: int = 0
    cache_misses: int = 0

    def get(self, key: str):
        match = _EXPERT_KEY.match(key)
        if match is None or match.group("prefix") != self.prefix:
            return None
        tensor = self.tensors.get(match.group("tail"))
        if tensor is None:
            return None
        expert_id = int(match.group("eid"))
        return tensor[expert_id] if expert_id < tensor.shape[0] else None

    def cached(self, expert_id: int):
        if not self.bf16_cache or expert_id not in self.bf16_cache:
            self.cache_misses += 1
            return None
        self.cache_hits += 1
        value = self.bf16_cache.pop(expert_id)
        self.bf16_cache[expert_id] = value
        return value

    def remember(self, expert_id: int, value):
        if self.cache_limit <= 0:
            return
        if self.bf16_cache is None:
            self.bf16_cache = OrderedDict()
        old = self.bf16_cache.pop(expert_id, None)
        if old is not None:
            del old
        self.bf16_cache[expert_id] = value
        while len(self.bf16_cache) > self.cache_limit:
            _, evicted = self.bf16_cache.popitem(last=False)
            del evicted


class _ExpertView:
    """Mapping view used by the common dequantizer for one expert."""

    def __init__(self, source, prefix: str, expert_id: int):
        self.source = source
        self.prefix = prefix
        self.expert_id = expert_id

    def get(self, key: str, default=None):
        value = self.source.get(key)
        return default if value is None else value


def _module_device(module: nn.Module) -> str:
    for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        if tensor.device.type != "meta":
            return str(tensor.device)
    raise RuntimeError(f"cannot determine device for {type(module).__name__}")


def _group_expert_keys(keys: Iterable[str], valid_prefixes: set[str]):
    grouped: dict[str, list[str]] = {prefix: [] for prefix in valid_prefixes}
    for key in keys:
        match = _EXPERT_KEY.match(key)
        if match is not None and match.group("prefix") in grouped:
            grouped[match.group("prefix")].append(key)
    return grouped


def _build_resident_store(prefix: str, keys: list[str], quant_weights: Mapping,
                          device: str, num_experts: int) -> ResidentExpertStore:
    by_tail: dict[str, dict[int, torch.Tensor]] = {}
    for key in keys:
        match = _EXPERT_KEY.match(key)
        if match is None:
            continue
        by_tail.setdefault(match.group("tail"), {})[int(match.group("eid"))] = quant_weights[key]

    tensors: dict[str, torch.Tensor] = {}
    total_bytes = 0
    try:
        for tail, expert_tensors in by_tail.items():
            if len(expert_tensors) != num_experts:
                raise ValueError(
                    f"incomplete expert metadata for {prefix}.{tail}: "
                    f"{len(expert_tensors)}/{num_experts}")
            ordered = [expert_tensors[i] for i in range(num_experts)]
            first_shape = ordered[0].shape
            if any(t.shape != first_shape or t.dtype != ordered[0].dtype for t in ordered):
                raise ValueError(f"inconsistent expert tensor shapes for {prefix}.{tail}")
            stacked = torch.stack(ordered, dim=0)
            resident = stacked.to(device, non_blocking=True)
            tensors[tail] = resident
            total_bytes += resident.numel() * resident.element_size()
            del stacked
        required = {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}
        if not required.issubset(tensors):
            missing = sorted(required.difference(tensors))
            raise KeyError(f"missing resident expert tensors: {missing}")
        return ResidentExpertStore(prefix, device, num_experts, tensors, total_bytes)
    except Exception:
        tensors.clear()
        gc.collect()
        if device.startswith("npu") and hasattr(torch, "npu"):
            torch.npu.empty_cache()
        raise


def _dequant_expert_weight(quant_name: str, source, quant_desc: Mapping[str, str],
                           dtype: torch.dtype, target_device: str):
    weight_key = f"{quant_name}.weight"
    base_name = parse_base_name(weight_key)
    quant_type = quant_desc.get(weight_key, quant_desc.get(base_name, "FLOAT"))
    weight = source.get(weight_key)
    if weight is None:
        return None
    if quant_type == "FLOAT":
        if not weight.dtype.is_floating_point:
            raise ValueError(f"integer expert weight classified as FLOAT: {weight_key}")
        return weight.to(device=target_device, dtype=dtype)

    scale = source.get(f"{quant_name}.weight_scale")
    offset = source.get(f"{quant_name}.weight_offset")
    if quant_type in ("W8A8_DYNAMIC", "W8A8_MIX") and scale is not None:
        return dequantize_weight_dynamic(
            weight.to(target_device), scale.to(target_device),
            offset.to(target_device) if offset is not None else None, dtype,
        )
    if quant_type in ("W4A8_DYNAMIC", "W4A8", "W4A16") and scale is not None:
        unpacked = unpack_int4_to_int8(weight.to(target_device))
        return dequantize_weight_dynamic(
            unpacked, scale.to(target_device),
            offset.to(target_device) if offset is not None else None, dtype,
        )
    if quant_type in _MX_QUANT_TYPES and scale is not None:
        return dequantize_weight_mx(
            weight.to(target_device), scale.to(target_device), quant_type, dtype=dtype,
        )

    deq_scale = source.get(f"{quant_name}.deq_scale")
    input_scale = source.get(f"{quant_name}.input_scale")
    if deq_scale is not None:
        result, _ = dequantize_weight_static(
            weight.to(target_device), deq_scale.to(target_device),
            input_scale.to(target_device) if input_scale is not None else None,
            dtype=dtype,
        )
        return result
    raise ValueError(f"unsupported or incomplete expert quantization: {quant_type} ({weight_key})")


def _dequant_expert_triplet(prefix: str, expert_id: int, source,
                            quant_desc: Mapping[str, str], dtype: torch.dtype,
                            target_device: str, cache_result: bool = False):
    if cache_result and isinstance(source, ResidentExpertStore):
        cached = source.cached(expert_id)
        if cached is not None:
            return cached
    view = _ExpertView(source, prefix, expert_id)
    gate = _dequant_expert_weight(
        f"{prefix}.{expert_id}.gate_proj", view, quant_desc, dtype, target_device)
    up = _dequant_expert_weight(
        f"{prefix}.{expert_id}.up_proj", view, quant_desc, dtype, target_device)
    down = _dequant_expert_weight(
        f"{prefix}.{expert_id}.down_proj", view, quant_desc, dtype, target_device)
    if gate is None or up is None or down is None:
        raise KeyError(f"expert {prefix}.{expert_id} is incomplete")
    result = (torch.cat((gate, up), dim=0), down)
    if cache_result and isinstance(source, ResidentExpertStore):
        source.remember(expert_id, result)
    return result


def _flatten_batched_qparam(value, batch_size: int, output_rows: int):
    """Flatten [expert, output, ...] metadata alongside a flattened weight."""
    if value is None:
        return None
    if value.dim() == 1:
        return value.repeat_interleave(output_rows)
    if value.shape[1] != output_rows:
        raise ValueError(
            f"quant metadata output rows {value.shape[1]} != {output_rows}")
    return value.reshape(batch_size * output_rows, *value.shape[2:])


def _dequant_resident_projection_batch(prefix: str, expert_ids: list[int],
                                        projection: str,
                                        source: ResidentExpertStore,
                                        quant_desc: Mapping[str, str],
                                        dtype: torch.dtype):
    """Vectorized common W4/W8 dynamic dequant for one resident chunk."""
    quant_types = set()
    for expert_id in expert_ids:
        weight_key = f"{prefix}.{expert_id}.{projection}.weight"
        quant_types.add(quant_desc.get(
            weight_key, quant_desc.get(parse_base_name(weight_key), "FLOAT")))
    if len(quant_types) != 1:
        return None
    quant_type = quant_types.pop()
    if quant_type not in {"W4A8_DYNAMIC", "W4A8", "W4A16",
                          "W8A8_DYNAMIC", "W8A8_MIX"}:
        return None

    weight = source.tensors.get(f"{projection}.weight")
    scale = source.tensors.get(f"{projection}.weight_scale")
    if weight is None or scale is None:
        return None
    ids = torch.tensor(expert_ids, dtype=torch.long, device=weight.device)
    selected = weight.index_select(0, ids)
    selected_scale = scale.index_select(0, ids)
    offset = source.tensors.get(f"{projection}.weight_offset")
    selected_offset = offset.index_select(0, ids) if offset is not None else None

    batch_size, packed_rows, input_cols = selected.shape
    if quant_type.startswith("W4"):
        flat_weight = unpack_int4_to_int8(
            selected.reshape(batch_size * packed_rows, input_cols))
        output_rows = packed_rows * 2
    else:
        output_rows = packed_rows
        flat_weight = selected.reshape(batch_size * output_rows, input_cols)
    flat_scale = _flatten_batched_qparam(selected_scale, batch_size, output_rows)
    flat_offset = _flatten_batched_qparam(selected_offset, batch_size, output_rows)
    result = dequantize_weight_dynamic(
        flat_weight, flat_scale, flat_offset, dtype)
    return result.reshape(batch_size, output_rows, input_cols)


def _dequant_resident_chunk(prefix: str, expert_ids: list[int],
                            source: ResidentExpertStore,
                            quant_desc: Mapping[str, str], dtype: torch.dtype,
                            target_device: str, cache_result: bool):
    """Return expert triplets in request order, batching uncached W4/W8 work."""
    results = [None] * len(expert_ids)
    missing_positions = []
    missing_ids = []
    if cache_result:
        for position, expert_id in enumerate(expert_ids):
            cached = source.cached(expert_id)
            if cached is not None:
                results[position] = cached
            else:
                missing_positions.append(position)
                missing_ids.append(expert_id)
    else:
        missing_positions = list(range(len(expert_ids)))
        missing_ids = list(expert_ids)

    if missing_ids:
        gate = _dequant_resident_projection_batch(
            prefix, missing_ids, "gate_proj", source, quant_desc, dtype)
        up = _dequant_resident_projection_batch(
            prefix, missing_ids, "up_proj", source, quant_desc, dtype)
        down = _dequant_resident_projection_batch(
            prefix, missing_ids, "down_proj", source, quant_desc, dtype)
        if gate is not None and up is not None and down is not None:
            for batch_idx, (position, expert_id) in enumerate(
                    zip(missing_positions, missing_ids)):
                value = (torch.cat((gate[batch_idx], up[batch_idx]), dim=0),
                         down[batch_idx])
                results[position] = value
                if cache_result:
                    source.remember(expert_id, value)
        else:
            for position, expert_id in zip(missing_positions, missing_ids):
                results[position] = _dequant_expert_triplet(
                    prefix, expert_id, source, quant_desc, dtype, target_device,
                    cache_result=cache_result)
    return results


def _run_active_chunk(hidden_states, expert_mask, top_k_weights, active_experts,
                      source, prefix, quant_desc, dtype, target_device,
                      final_hidden_states, activation, cache_result):
    if isinstance(source, ResidentExpertStore):
        triplets = _dequant_resident_chunk(
            prefix, active_experts, source, quant_desc, dtype, target_device,
            cache_result)
    else:
        triplets = [
            _dequant_expert_triplet(
                prefix, expert_id, source, quant_desc, dtype, target_device,
                cache_result=cache_result)
            for expert_id in active_experts
        ]
    gate_up = [value[0] for value in triplets]
    down = [value[1] for value in triplets]
    gate_up_stack = torch.stack(gate_up)
    down_stack = torch.stack(down)
    del gate_up, down

    for offset, expert_id in enumerate(active_experts):
        top_k_pos, token_idx = torch.where(expert_mask[expert_id])
        current_state = hidden_states[token_idx]
        gate, up = nn.functional.linear(current_state, gate_up_stack[offset]).chunk(2, dim=-1)
        current_hidden = activation(gate) * up
        current_hidden = nn.functional.linear(current_hidden, down_stack[offset])
        current_hidden *= top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(
            0, token_idx, current_hidden.to(final_hidden_states.dtype))
    del gate_up_stack, down_stack


def install_resident_streaming_moe(
    model: nn.Module,
    quant_weights: dict[str, torch.Tensor],
    quant_desc: Mapping[str, str],
    dtype: torch.dtype,
    *,
    chunk_size: int = 8,
    verbose: bool = True,
) -> None:
    """Install resident packed-expert forward, with CPU compact fallback."""
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaNaiveMoe

    modules = [(module, name) for name, module in model.named_modules()
               if isinstance(module, GlmMoeDsaNaiveMoe)]
    if not modules:
        logger.warning("  未找到 GlmMoeDsaNaiveMoe, 无法安装 resident expert")
        return
    chunk_size = max(1, int(chunk_size))
    key_groups = _group_expert_keys(quant_weights.keys(), {name for _, name in modules})
    resident_enabled = os.getenv("ACC_BOUNDARY_RESIDENT_EXPERTS", "1") != "0"
    cache_limit = max(0, int(os.getenv("ACC_BOUNDARY_EXPERT_CACHE_PER_LAYER", "16")))
    resident_count = 0
    resident_bytes = 0

    for module, prefix in modules:
        device = _module_device(module)
        source = None
        keys = key_groups.get(prefix, [])
        if resident_enabled:
            try:
                source = _build_resident_store(
                    prefix, keys, quant_weights, device, module.num_experts)
                source.cache_limit = cache_limit
                resident_count += 1
                resident_bytes += source.bytes_on_device
                if verbose and resident_count % 5 == 0:
                    logger.info(
                        "    resident expert 进度: %d/%d 层, %.1f GiB",
                        resident_count, len(modules), resident_bytes / 2**30)
            except Exception as exc:  # noqa: BLE001 - OOM must fall back per layer
                logger.warning("  %s resident expert 回退到 CPU compact streaming: %s", prefix, exc)
        if source is None:
            source = {key: quant_weights[key] for key in keys}
        module._resident_streaming = {
            "source": source, "prefix": prefix, "quant_desc": quant_desc,
            "dtype": dtype, "chunk_size": chunk_size,
            "last_token_rows": None, "decode_phase": False,
        }

    # Resident stores own their NPU tensors; fallback modules hold their compact
    # CPU tensors.  Dense/shared weights were already materialized on NPU.
    quant_weights.clear()
    gc.collect()
    original_forward = GlmMoeDsaNaiveMoe.forward
    forward_calls = [0]
    module_count = len(modules)

    def resident_forward(self, hidden_states, top_k_index, top_k_weights):
        cfg = getattr(self, "_resident_streaming", None)
        if cfg is None:
            return original_forward(self, hidden_states, top_k_index, top_k_weights)
        expert_mask = nn.functional.one_hot(
            top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        active = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()
        final = torch.zeros_like(hidden_states)
        token_rows = hidden_states.shape[0]
        previous_rows = cfg["last_token_rows"]
        if previous_rows is not None and token_rows < previous_rows:
            cfg["decode_phase"] = True
        elif previous_rows is not None and token_rows > previous_rows:
            cfg["decode_phase"] = False
        cfg["last_token_rows"] = token_rows
        cache_decode = cfg["decode_phase"]
        if cache_decode and isinstance(cfg["source"], ResidentExpertStore):
            cached_ids = set((cfg["source"].bf16_cache or {}).keys())
            active.sort(key=lambda expert_id: expert_id not in cached_ids)
        for start in range(0, len(active), cfg["chunk_size"]):
            _run_active_chunk(
                hidden_states, expert_mask, top_k_weights,
                active[start:start + cfg["chunk_size"]], cfg["source"],
                cfg["prefix"], cfg["quant_desc"], cfg["dtype"],
                str(hidden_states.device), final, self.act_fn, cache_decode,
            )
        forward_calls[0] += 1
        calls = forward_calls[0]
        if (calls <= module_count and calls % 10 == 0) or calls % 100 == 0:
            phase = "decode" if cache_decode else "prefill"
            source = cfg["source"]
            cache_stats = ""
            if isinstance(source, ResidentExpertStore):
                total = source.cache_hits + source.cache_misses
                cache_stats = f", cache_hit={source.cache_hits}/{total}" if total else ""
            logger.info("  [resident expert] %s %d/%d, layer=%s, active=%d, chunk=%d%s",
                        phase, min(calls, module_count), module_count, cfg["prefix"],
                        len(active), cfg["chunk_size"], cache_stats)
        return final

    GlmMoeDsaNaiveMoe.forward = resident_forward
    model._resident_expert_sources = [
        module._resident_streaming["source"] for module, _ in modules]
    if verbose:
        logger.info(
            "  resident packed expert: %d/%d 层常驻 NPU, %.1f GiB, chunk=%d, BF16 cache=%d/layer",
            resident_count, module_count, resident_bytes / 2**30, chunk_size, cache_limit)
