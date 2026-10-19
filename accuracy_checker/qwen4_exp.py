"""Text-only Qwen3.8-Flash-Next (HF qwen4_exp) integration.

Keep the upstream QSA, GatedDeltaNet, PLE and hyper-connection math. Only
checkpoint I/O and the inputs required by isolated decoder replay live here.
PLE tables are mmap-backed, including HF's row-sharded checkpoint layout;
they must never be materialized by ``to_empty`` or moved wholesale to an NPU.
"""
import logging
import re

import torch
from torch import nn

logger = logging.getLogger(__name__)

PLE_HASH_BUFFER_NAMES = ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")


def _ple_text_config(config):
    """Use installed HF defaults (including seed), never guess missing fields."""
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    if isinstance(config, dict):
        return Qwen4ExpTextConfig(**config.get("text_config", config))
    return getattr(config, "text_config", config)


def rebuild_ple_hash_buffers(config, layer_idx):
    """Reconstruct only deterministic integer metadata, not the embedding table.

    The official meta constructor determines the head layout; its Python
    attributes survive meta construction. Multipliers are generated on CPU by
    the same official helper. No learned tensor or checkpoint file is changed.
    """
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextNGramEmbedding, _build_layer_multipliers,
    )
    config = _ple_text_config(config)
    if layer_idx + 1 not in config.ple_layer_ids:
        raise ValueError(f"Qwen4 layer {layer_idx} has no PLE in config")
    ple_index = config.ple_layer_ids.index(layer_idx + 1)
    with torch.device("meta"):
        module = Qwen4ExpTextNGramEmbedding(config, config.ple_embed_dim, layer_idx, ple_index)
    with torch.device("cpu"):
        return {
            "layer_multipliers": _build_layer_multipliers(
                module.unigram_vocab_size, module.ngram_size, module.ple_layer_index, module.seed),
            "ngram_heads_vocab_sizes": torch.tensor(module.head_vocab_sizes, dtype=torch.int64),
            "ngram_heads_offsets": torch.tensor(module.head_offsets, dtype=torch.int64),
        }


def resolve_ple_hash_buffers(config, layer_idx, raw_buffers):
    """Validate exact metadata or recover a *verified* floating-point export.

    Do not recover by casting floats back to int64. Recovery is allowed only
    when all checkpoint values equal the official integers rounded to that
    exact export dtype. Conflicting integers, arbitrary floats and missing
    metadata fail closed rather than silently changing the model's contract.
    """
    expected = rebuild_ple_hash_buffers(config, layer_idx)
    recovered = []
    float_dtypes = (torch.bfloat16, torch.float16, torch.float32, torch.float64)
    for name, canonical in expected.items():
        raw = raw_buffers.get(name)
        prefix = f"Qwen4 PLE hash layer {layer_idx} {name}"
        if raw is None or raw.is_meta or raw.shape != canonical.shape:
            raise ValueError(f"{prefix}: missing or wrong shape; cannot validate metadata")
        raw = raw.detach().cpu()
        if raw.dtype == torch.int64:
            if not torch.equal(raw, canonical):
                raise ValueError(f"{prefix}: int64 checkpoint conflicts with config-derived values; refusing to overwrite")
        elif raw.dtype in float_dtypes:
            rounded = canonical.to(raw.dtype)
            # Both operands have the SAME dtype: no cross-dtype torch.equal
            # promotion that could mistake a rounded integer for an exact one.
            if not bool(torch.isfinite(raw).all() and torch.isfinite(rounded).all()) or not torch.equal(raw, rounded):
                raise ValueError(f"{prefix}: {raw.dtype} checkpoint is not a validated rounded config export; cannot recover safely")
            recovered.append(name)
        else:
            raise ValueError(f"{prefix}: unsupported metadata dtype {raw.dtype}; expected int64 or a validated floating export")
    return expected, recovered


def validate_ple_pair_configs(ref_config, quant_config):
    """Paired L1/L2 runs must use the same ngram semantics; no ref-side copying."""
    if not is_qwen4(ref_config) and not is_qwen4(quant_config):
        return
    if not is_qwen4(ref_config) or not is_qwen4(quant_config):
        raise ValueError("Qwen4 PLE comparison requires matching ref/quant model types")
    ref, quant = _ple_text_config(ref_config), _ple_text_config(quant_config)
    keys = ("vocab_size", "ngram_size", "heads_per_ngram", "ngram_vocab_size_base",
            "seed", "ple_layer_ids", "ple_embed_dim", "make_ngram_vocab_size_divisible_by")
    differences = [key for key in keys if getattr(ref, key) != getattr(quant, key)]
    def eos(config):
        value = config.eos_token_id
        return value[0] if isinstance(value, list) else value
    if eos(ref) != eos(quant):
        differences.append("eos_token_id")
    if differences:
        raise ValueError("Qwen4 PLE ref/quant config mismatch: " + ", ".join(differences))


def is_qwen4(config):
    kind = config.get("model_type", "") if isinstance(config, dict) else getattr(config, "model_type", "")
    return kind in {"qwen4_exp", "qwen4_exp_text"}


def is_qwen4_layer(layer):
    return type(layer).__name__ == "Qwen4ExpTextDecoderLayer"


def check_parallel_mode(model_path, mode, device_groups=None):
    from .model_loader import _read_raw_model_config
    if is_qwen4(_read_raw_model_config(model_path)):
        from .qwen4_tp import install_query_parallel, configure_options
        install_query_parallel(mode, device_groups)
        logger.info("  Qwen4 QSA indexer runtime: mode=%s, groups=%s, options=%s; "
                    "main attention/GDN/decode remain on owner", mode, device_groups, configure_options())


def expand_embedding(model, hidden):
    config = model.config
    config = getattr(config, "text_config", config)
    return hidden.repeat(1, 1, config.hc_count) if is_qwen4(config) else hidden


def layer_kwargs(layer, hidden, position_ids, position_embeddings, input_ids):
    """Unpadded, cache-free L1/L2 replay. Never synthesize PLE token IDs."""
    from .model_structure import build_replay_attention_mask
    if getattr(layer, "ple", None) is not None:
        if input_ids is None or tuple(input_ids.shape) != tuple(hidden.shape[:2]):
            raise ValueError("Qwen4 PLE replay requires the original input_ids; rerun L1 to refresh the cache")
    return dict(
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        attention_mask=build_replay_attention_mask(layer, hidden),
        conv_mask=None,
        ple_input_ids=input_ids.to(hidden.device) if input_ids is not None else None,
    )


class MMapNGramEmbedding(nn.Module):
    """Read only requested rows, never concatenate the ~95 GiB PLE table.

    The CPU weight marker is intentionally not a parameter/buffer: module
    movement and shard unload must not relocate the backing table. The HF
    PLE caller moves IDs to weight.device and the lookup result back itself.
    """
    def __init__(self, reader, name, shape, dtype):
        super().__init__()
        self.reader = reader
        self.shape = tuple(shape)
        self.weight = torch.empty(0, dtype=dtype, device="cpu")
        key, filename = reader._resolve_key(name + ".weight")
        if key is not None:
            keys = [(0, key, filename)]
        else:
            keys = []
            # Resolve runtime text-only prefixes against the original MM keys.
            for candidate in reader.weight_map:
                match = re.fullmatch(re.escape(name) + r"\.shard_(\d+)\.weight", candidate.replace("model.language_model.", "model."))
                if match:
                    keys.append((int(match[1]), candidate, reader.weight_map[candidate]))
            keys.sort()
            if [i for i, _, _ in keys] != list(range(len(keys))):
                raise ValueError(f"Non-contiguous PLE shard IDs: {name}")
        self.parts = []
        offset = 0
        for _, key, filename in keys:
            tensor = reader._get_sf(filename).get_slice(key)
            rows, width = tensor.get_shape()
            if width != self.shape[1] or tensor.get_dtype() not in {"BF16", "F16", "F32"}:
                raise ValueError(f"Unsupported PLE layout/dtype for {key}; expected floating-point embedding rows")
            self.parts.append((offset, offset + rows, filename, key))
            offset += rows
        if offset != self.shape[0]:
            raise ValueError(f"Missing/incomplete PLE table {name}: {offset} rows, expected {self.shape[0]}")

    def forward(self, ids):
        flat = ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        if flat.numel() and (flat.min() < 0 or flat.max() >= self.shape[0]):
            raise IndexError("PLE row ID outside checkpoint table")
        unique, inverse = flat.unique(sorted=True, return_inverse=True)
        values = torch.empty((unique.numel(), self.shape[1]), dtype=self.weight.dtype)
        for start, end, filename, key in self.parts:
            selected = (unique >= start) & (unique < end)
            if selected.any():
                rows = (unique[selected] - start).tolist()
                values[selected] = self.reader._get_sf(filename).get_slice(key)[rows].to(self.weight.dtype)
        return values[inverse].reshape(*ids.shape, self.shape[1])


def install_lazy_ple(model, model_path):
    from .model_loader import ShardWeightReader, build_weight_index
    reader = ShardWeightReader(model_path, build_weight_index(model_path))
    count = 0
    for name, module in list(model.named_modules()):
        if name.endswith(".ple_embedding.ngram_embedding") and isinstance(module, nn.Embedding):
            parent_name, child = name.rsplit(".", 1)
            setattr(model.get_submodule(parent_name), child, MMapNGramEmbedding(
                reader, name, module.weight.shape, module.weight.dtype))
            count += 1
    model._acc_qwen4_ple_reader = reader
    logger.info("  Qwen4 text runtime: %d mmap PLE table(s); official QSA/GatedDeltaNet + hyper-connections", count)


def load_boundary_weights(model, model_path, dtype, expert_chunk_size=None):
    """Reuse indexed dequantization and L1's tested expert projection runner.

    No giant load_file()/PLE concatenate, no vLLM dependency. Ordinary layer
    weights stay resident on their PP owner; selected routed experts are read
    on demand. This is a correctness baseline, not a high-throughput engine.
    """
    import json
    import os
    from .model_loader import (ShardWeightReader, build_weight_index,
        load_layer_weights_indexed, is_quantized_model,
        is_compressed_tensors_model, native_quant_description)
    from .layer1_block_compare import ShardedBlockComparator
    from .utils import get_model_components, normalize_quant_desc_keys
    parts = get_model_components(model)
    reader = ShardWeightReader(model_path, build_weight_index(model_path))
    desc_path = os.path.join(model_path, "quant_model_description.json")
    if os.path.isfile(desc_path):
        with open(desc_path) as handle:
            desc = json.load(handle)
    else:
        desc = native_quant_description(model_path)
    desc = normalize_quant_desc_keys(desc or {}, model)
    is_quant = is_quantized_model(model_path)
    is_ct = is_compressed_tensors_model(model_path)
    runner = ShardedBlockComparator(model_path, model_path, None,
        ref_device="cpu", quant_device="cpu", dtype=dtype,
        use_fake_quant=False, verbose=False, collect_full_logits=False)
    for selection, modules in [([], [parts.embed]), ([-1], [parts.lm_head, parts.text_model.hyper_connection_mixer])] + [
        ([i], [layer]) for i, layer in enumerate(parts.layers)
    ]:
        devices = [next(m.parameters()).device for m in modules]
        load_layer_weights_indexed(model, model_path, selection, str(devices[0]), dtype,
            reader.weight_map, reader, is_quant=is_quant, is_ct=is_ct, quant_desc=desc,
            skip_routed_experts=True, strict_embed_only=True, strict_final_only=True,
            verbose=False)
        for module, device in zip(modules, devices):
            module.to(device)
    for i, layer in enumerate(parts.layers):
        device = str(next(layer.parameters()).device)
        layer.mlp.forward = runner._make_chunked_mlp_forward(
            layer.mlp.forward, layer, [device], expert_chunk_size or 8,
            i, reader, desc, is_quant, is_ct)
    model._acc_qwen4_expert_runner = runner
    model._acc_qwen4_expert_reader = reader
    logger.info("  Qwen4 Boundary: indexed ordinary weights + on-demand routed experts; PLE mmap; PP ownership unchanged")
