"""Read-only Qwen4 PLE probes: no decoder/expert/model allocation.

The scalar-row audit checks the optimized mmap gather against a separate
slice-based reader. It cannot prove that the exporter labelled its shards
correctly. Cross-checkpoint row comparisons provide separate evidence for that.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from .model_loader import ShardWeightReader, build_weight_index
from .qwen4_exp import MMapNGramEmbedding


BUFFER_NAMES = ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")
CONTRACT_FIELDS = ("vocab_size", "ngram_size", "heads_per_ngram", "eos_token_id",
                   "ple_layer_ids", "ple_embed_dim", "ngram_vocab_size_base", "seed",
                   "make_ngram_vocab_size_divisible_by")


def tensor_comparison(ref, quant):
    """Exact equality plus numerical diagnostics; never turn NaN into a pass."""
    ref, quant = ref.detach().cpu(), quant.detach().cpu()
    same_shape = ref.shape == quant.shape
    finite = bool(torch.isfinite(ref).all() and torch.isfinite(quant).all())
    result = dict(shape_match=same_shape, finite=finite,
                  ref_shape=list(ref.shape), quant_shape=list(quant.shape),
                  equal=False, max_abs_diff=None, rel_l2=None)
    if not same_shape or not finite:
        return result
    # Equality is evaluated before float conversion (hash integers exceed 2**53).
    result["equal"] = bool(torch.equal(ref, quant))
    result["different_elements"] = int((ref != quant).sum())
    result["first_differences"] = (ref != quant).nonzero()[:8].tolist()
    if ref.numel():
        a, b = ref.double(), quant.double()
        diff = torch.linalg.vector_norm(a - b).item()
        norm = torch.linalg.vector_norm(a).item()
        result["max_abs_diff"] = (a - b).abs().max().item()
        result["rel_l2"] = diff / norm if norm else (0.0 if diff == 0 else None)
        result["numeric_metrics_note"] = "float64 diagnostics; exact equality uses original dtype"
    return result


def validate_input_ids(value):
    if isinstance(value, torch.Tensor):
        if value.dtype not in (torch.int64, torch.int32, torch.int16):
            raise ValueError("input_ids must be integer tokens, not floats/bools")
        ids = value.detach().cpu().long()
    else:
        rows = value if isinstance(value, list) else []
        flat = [t for row in rows for t in row] if rows and isinstance(rows[0], list) else rows
        if not flat or any(type(t) is not int for t in flat):
            raise ValueError("input_ids must be a nonempty integer list or rectangular batch")
        ids = torch.tensor(value, dtype=torch.long)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2 or not ids.numel() or bool((ids < 0).any()):
        raise ValueError("input_ids must be nonempty [B,S], with nonnegative token IDs")
    return ids


def token_hash(ids):
    return hashlib.sha256(ids.cpu().contiguous().numpy().astype("<i8").tobytes()).hexdigest()


def compute_ngram_ids(input_ids, *, multipliers, head_vocab_sizes, head_offsets,
                      ngram_size, heads_per_ngram, eos_token_id):
    """HF cache-free hashing, retaining the entire prefix and EOS boundaries.

    Only integer operations are used. This independent probe is also compared
    with the installed Transformers forward below, not assumed to be its oracle.
    """
    heads = (ngram_size - 1) * heads_per_ngram
    if ngram_size < 2 or heads_per_ngram < 1:
        raise ValueError("Invalid ngram_size/heads_per_ngram")
    for name, tensor, shape in (("multipliers", multipliers, (ngram_size,)),
                                ("head_vocab_sizes", head_vocab_sizes, (heads,)),
                                ("head_offsets", head_offsets, (heads,))):
        if tensor.dtype != torch.int64 or tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must be int64 with shape {shape}; will not repair hash metadata")
    if bool((head_vocab_sizes <= 0).any()) or bool((head_offsets < 0).any()):
        raise ValueError("Invalid ngram vocabulary sizes/offsets")
    device = input_ids.device
    multipliers, head_vocab_sizes, head_offsets = [t.to(device) for t in
                                                  (multipliers, head_vocab_sizes, head_offsets)]
    ids = input_ids.long()
    history = torch.cat([ids.new_full((ids.shape[0], ngram_size - 1), eos_token_id), ids], dim=1)
    positions = torch.arange(history.shape[1], device=device, dtype=torch.long)
    eos_positions = torch.where(history == eos_token_id, positions, -1)
    previous = torch.cummax(eos_positions, dim=1).values
    previous = torch.cat([eos_positions.new_full((ids.shape[0], 1), -1), previous[:, :-1]], dim=1)
    distance = positions.unsqueeze(0) - (previous + 1)
    shifted = [history]
    for shift in range(1, ngram_size):
        source = positions - shift
        gathered = history.gather(1, source.clamp_min(0).unsqueeze(0).expand(ids.shape[0], -1))
        valid = (distance >= shift) & (source.unsqueeze(0) >= 0)
        shifted.append(torch.where(valid, gathered, ids.new_full((), eos_token_id)))
    blocks = []
    for n in range(2, ngram_size + 1):
        mixed = shifted[0] * multipliers[0]
        for i in range(1, n):
            mixed = torch.bitwise_xor(mixed, shifted[i] * multipliers[i])
        start, end = (n - 2) * heads_per_ngram, (n - 1) * heads_per_ngram
        blocks.append(torch.remainder(mixed.unsqueeze(-1), head_vocab_sizes[start:end]) + head_offsets[start:end])
    return torch.cat(blocks, dim=-1)[:, -ids.shape[1]:]


@dataclass
class PLECheckpoint:
    path: str
    layer_idx: int
    config: dict
    reader: ShardWeightReader
    buffers: dict
    table: MMapNGramEmbedding | None
    issues: list
    table_metadata: list

    def close(self):
        self.reader.close()

    def hash_kwargs(self):
        eos = self.config["eos_token_id"]
        if isinstance(eos, list):
            eos = eos[0]
        return dict(multipliers=self.buffers[BUFFER_NAMES[0]],
                    head_vocab_sizes=self.buffers[BUFFER_NAMES[1]],
                    head_offsets=self.buffers[BUFFER_NAMES[2]],
                    ngram_size=int(self.config["ngram_size"]),
                    heads_per_ngram=int(self.config["heads_per_ngram"]), eos_token_id=int(eos))


def inspect_checkpoint(path, layer_idx, *, dtype=torch.bfloat16):
    path = str(Path(path).resolve())
    raw = json.loads((Path(path) / "config.json").read_text())
    config = raw.get("text_config", raw)
    if config.get("model_type", raw.get("model_type")) not in {"qwen4_exp", "qwen4_exp_text"}:
        raise ValueError(f"Not a Qwen4 checkpoint: {path}")
    if layer_idx + 1 not in config.get("ple_layer_ids", []):
        raise ValueError(f"Layer {layer_idx} has no PLE: ple_layer_ids={config.get('ple_layer_ids')} (one-based)")
    reader = ShardWeightReader(path, build_weight_index(path))
    checkpoint = PLECheckpoint(path, layer_idx, config, reader, {}, None, [], [])
    def issue(code, message):
        checkpoint.issues.append(dict(severity="error", code=code, message=message))
    prefix = f"model.layers.{layer_idx}.ple.ple_embedding"
    n = int(config["ngram_size"])
    heads = (n - 1) * int(config["heads_per_ngram"])
    try:
        for name in BUFFER_NAMES:
            value = reader.get_tensor(prefix + "." + name)
            if value is None:
                issue("missing_hash_buffer", name)
                continue
            checkpoint.buffers[name] = value
            if value.dtype != torch.int64:
                issue("hash_buffer_dtype", f"{name}: {value.dtype}, expected int64; no silent cast/repair")
            if tuple(value.shape) != (n if name == BUFFER_NAMES[0] else heads,):
                issue("hash_buffer_shape", f"{name}: {tuple(value.shape)}")
        if not checkpoint.issues:
            sizes, offsets = [checkpoint.buffers[name] for name in BUFFER_NAMES[1:]]
            expected_offsets = torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)[:-1]])
            if bool((sizes <= 0).any()) or not torch.equal(offsets, expected_offsets):
                issue("hash_head_layout", "head sizes must be positive and offsets cumulative")
            else:
                divisor = int(config.get("make_ngram_vocab_size_divisible_by", 1))
                embed_dim = int(config["ple_embed_dim"])
                if divisor < 1 or embed_dim % heads:
                    raise ValueError("Invalid PLE table width/padding config")
                rows = math.ceil(int(sizes.sum()) / divisor) * divisor
                checkpoint.table = MMapNGramEmbedding(reader, prefix + ".ngram_embedding",
                                                     (rows, embed_dim // heads), dtype)
                checkpoint.table_metadata = [dict(start=a, end=b, filename=f, key=k,
                                                  dtype=reader._get_sf(f).get_slice(k).get_dtype())
                                             for a, b, f, k in checkpoint.table.parts]
        return checkpoint
    except Exception:
        checkpoint.close()
        raise


def audit_lookup(checkpoint, row_ids):
    """Independent scalar contiguous slices, not the production unique/list gather.

    The manifest's contiguous-row interpretation is shared and explicitly NOT
    validated against an unknown exporter by this test.
    """
    rows = row_ids.detach().cpu().long().reshape(-1)
    table = checkpoint.table
    if table is None:
        raise ValueError("Cannot audit table with invalid hash metadata")
    actual = table(rows)
    expected = []
    ends = [part[1] for part in table.parts]
    handles = {}
    try:
        for row in rows.tolist():
            part = bisect.bisect_right(ends, row)
            if part >= len(table.parts) or row < 0:
                raise IndexError("Audit row outside PLE table")
            start, _, filename, key = table.parts[part]
            if filename not in handles:
                handles[filename] = safe_open(str(Path(checkpoint.path) / filename), framework="pt", device="cpu")
            expected.append(handles[filename].get_slice(key)[row - start:row - start + 1].to(table.weight.dtype))
    finally:
        handles.clear()
    expected = torch.cat(expected) if expected else actual.clone()
    return dict(checked_rows=rows.numel(), comparison=tensor_comparison(expected, actual),
                scope="sampled scalar slices vs production gather; does not validate export shard ordering")


def capture_runtime_ngram_ids(checkpoint, input_ids, device="cpu"):
    """Run the INSTALLED HF hash forward with a row-ID capture instead of a table.

    Construct only the NGram module on meta. No model, table payload, or expert
    allocation. A replaced embedding captures ids and returns tiny dummy values.
    """
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding
    class Capture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.empty(0, device="cpu")
            self.ids = None

        def forward(self, ids):
            self.ids = ids.detach().cpu().clone()
            return torch.zeros((*ids.shape, 1), dtype=torch.float32)

    config = Qwen4ExpTextConfig(**checkpoint.config)
    ple_index = config.ple_layer_ids.index(checkpoint.layer_idx + 1)
    with torch.device("meta"):
        module = Qwen4ExpTextNGramEmbedding(config, config.ple_embed_dim, checkpoint.layer_idx, ple_index)
    for name in BUFFER_NAMES:
        module.register_buffer(name, checkpoint.buffers[name].to(device))
    capture = Capture()
    module.ngram_embedding = capture
    with torch.no_grad():
        module(input_ids.to(device), past_key_values=None)
    if capture.ids is None:
        raise RuntimeError("Installed HF forward did not call ngram_embedding")
    return capture.ids


def _sample_positions(length, limit, ids, eos_ids):
    # Hashing happens BEFORE sampling, so no ngram context is lost.
    count = min(length, limit)
    positions = torch.linspace(0, length - 1, count).long().unique().tolist()
    candidates = []
    for eos in eos_ids:
        for pos in (ids == eos).nonzero()[:16, 1].tolist():
            candidates.extend(range(max(0, pos - 2), min(length, pos + 3)))
    # Extra EOS probes are bounded separately, and visible in output.
    return sorted(set(positions + candidates))


def diagnose_pair(ref, quant, input_ids, *, max_positions=256, device="cpu"):
    if max_positions < 1:
        raise ValueError("max_positions must be positive")
    ids = validate_input_ids(input_ids)
    report = dict(schema_version=1, layer=ref.layer_idx, input_shape=list(ids.shape),
                  input_ids_sha256=token_hash(ids), device=device, checks={}, checkpoints={},
                  findings=[], limitations=[
                      "No decoder/PLE projection forward: a passing lookup does not clear PLE gating, convolution or quantization.",
                      "Only selected lookup rows are read, not the full table; no whole-checkpoint correctness claim.",
                      "The scalar audit assumes contiguous row shards, as the production loader does."])
    def finding(code, message):
        report["findings"].append(dict(code=code, message=message))
    for side, checkpoint in (("ref", ref), ("quant", quant)):
        report["checkpoints"][side] = dict(path=checkpoint.path, config={k: checkpoint.config.get(k) for k in CONTRACT_FIELDS},
                issues=checkpoint.issues, table_parts=checkpoint.table_metadata,
                buffers={k: dict(dtype=str(v.dtype), shape=list(v.shape), values=v.tolist())
                         for k, v in checkpoint.buffers.items()})
    contract_diff = {k: dict(ref=ref.config.get(k), quant=quant.config.get(k)) for k in CONTRACT_FIELDS
                     if ref.config.get(k) != quant.config.get(k)}
    report["checks"]["config_differences"] = contract_diff
    if contract_diff:
        finding("CONFIG_DIFFERENCE", "核对两边 PLE 配置；缺省字段与显式默认值也会列出，不自动认定不等价。")
    for name in BUFFER_NAMES:
        if name in ref.buffers and name in quant.buffers:
            comparison = tensor_comparison(ref.buffers[name], quant.buffers[name])
            report["checks"]["buffer_" + name] = comparison
            if not comparison["equal"]:
                finding("HASH_METADATA_DIFFERENCE", f"{name} 不一致：先核对导出/配置，不是浮点量化容差。")
    if ref.issues or quant.issues:
        report["status"] = "INVALID_METADATA"
        finding("INVALID_METADATA", "哈希 buffer 格式异常，已停止查表；不会转 dtype 掩盖问题。")
        return report
    for checkpoint in (ref, quant):
        if int(ids.max()) >= int(checkpoint.config["vocab_size"]):
            raise ValueError(f"Input token exceeds {checkpoint.path} vocab_size")
    hashes = {side: compute_ngram_ids(ids, **checkpoint.hash_kwargs())
              for side, checkpoint in (("ref", ref), ("quant", quant))}
    comparison = tensor_comparison(hashes["ref"], hashes["quant"])
    report["checks"]["ngram_ids"] = comparison
    if not comparison["equal"]:
        finding("HASH_ROW_DIFFERENCE", "同一 input_ids 的 ref/quant 查表行号不同；先核对 EOS/哈希参数。")
    for side, checkpoint in (("ref", ref), ("quant", quant)):
        for target in dict.fromkeys(("cpu", device)):
            key = f"{side}_runtime_hash_{target}"
            try:
                actual = capture_runtime_ngram_ids(checkpoint, ids, target)
                report["checks"][key] = tensor_comparison(hashes[side], actual)
                if not report["checks"][key]["equal"]:
                    finding("RUNTIME_HASH_DIFFERENCE", f"{side} {target} 的 HF 哈希与独立 CPU 结果不一致；检查运行时/整数算子。")
            except Exception as exc:
                report["checks"][key] = dict(status="unavailable", error=f"{type(exc).__name__}: {exc}")
                finding("RUNTIME_CHECK_UNAVAILABLE", f"{key} 未完成；不能视作校验通过。")
    positions = _sample_positions(ids.shape[1], max_positions, ids,
                                  [ref.hash_kwargs()["eos_token_id"], quant.hash_kwargs()["eos_token_id"]])
    report["sampled_positions"] = positions
    report["checked_hash_positions"] = ids.shape[0] * ids.shape[1]
    lookup_values = {}
    for side, checkpoint in (("ref", ref), ("quant", quant)):
        selected = hashes[side][:, positions]
        lookup_values[side] = checkpoint.table(selected)
        # Keep independent scalar reads bounded: 64 actual rows + all shard ends.
        unique = selected.unique(sorted=True)
        sampled = unique[torch.linspace(0, unique.numel() - 1, min(64, unique.numel())).long()]
        boundaries = [row for start, end, _, _ in checkpoint.table.parts for row in (start, end - 1)]
        audit_rows = torch.cat([sampled, torch.tensor(boundaries, dtype=torch.long)]).unique()
        audit = audit_lookup(checkpoint, audit_rows)
        report["checks"][side + "_lookup_audit"] = audit
        if not audit["comparison"]["equal"]:
            finding("LOOKUP_IMPLEMENTATION_DIFFERENCE", f"{side} 分片 gather 与独立逐行读取不一致。")
    report["checks"]["sampled_lookup_outputs"] = tensor_comparison(lookup_values["ref"], lookup_values["quant"])
    # Compare the same row numbers too, even if hashes differ. A different table
    # is evidence to inspect its export/quantization, not proof of a reader bug.
    common = torch.cat([hashes[s][:, positions].flatten() for s in hashes]).unique()
    common = common[common < min(ref.table.shape[0], quant.table.shape[0])]
    common = common[torch.linspace(0, common.numel() - 1, min(512, common.numel())).long()] if common.numel() else common
    same_rows = tensor_comparison(ref.table(common), quant.table(common))
    same_rows["checked_rows"] = common.numel()
    same_rows["row_ids"] = common.tolist()
    report["checks"]["same_row_table_values"] = same_rows
    if not same_rows["equal"]:
        finding("TABLE_VALUE_DIFFERENCE", "相同行号读出的表值不同；可能是导出/表变换/量化，不直接判为读取 bug。")
    if not report["checks"]["sampled_lookup_outputs"]["equal"]:
        finding("LOOKUP_OUTPUT_DIFFERENCE", "本次抽查位置的查表输出不一致；结合行号和同行表值检查定位。")
    if not report["findings"]:
        report["status"] = "SAMPLED_LOOKUP_ALIGNED"
        report["next_step"] = "本次哈希和抽查查表结果一致。下一步给两边 PLE 同一 hidden state，定位投影、门控、卷积及后续层内误差；不是整层验收通过。"
    else:
        report["status"] = "NEEDS_REVIEW"
        report["next_step"] = "先处理 findings 中最早的配置/哈希/查表异常；未完成的检查不得视为 PASS。"
    return report
