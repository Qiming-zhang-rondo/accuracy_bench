"""Memory-bounded DeepSeek-V4 long-prefill execution.

The official eager implementation materializes index scores shaped
``[batch, sequence, index_heads, compressed_sequence]`` and then a dense
compressor mask.  At 64k tokens those intermediates dominate memory even
though CSA ultimately keeps only ``index_topk`` entries.  This module keeps
the same projections, compression, causal rules, sinks and global top-k, but
tiles the score computation and performs attention directly over the local
window plus the selected compressed entries.
"""

from __future__ import annotations

import logging
import os
import time

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _normalise_device_groups(device_groups):
    groups = []
    for group in device_groups or ():
        if isinstance(group, str):
            group = group.split(",")
        cleaned = [str(device).strip() for device in (group or ()) if str(device).strip()]
        if cleaned:
            groups.append(cleaned)
    return groups


def _same_device(left, right):
    return str(left) == str(right)


def _select_tp_devices(source_device, device_groups):
    """Return the configured TP group containing the current layer owner."""
    source = str(source_device)
    for group in _normalise_device_groups(device_groups):
        if any(_same_device(source, device) for device in group):
            return group
    return []


def _query_block_assignments(seq_len, query_block, num_devices):
    """Round-robin query blocks without ever concatenating a mega-shard."""
    if seq_len <= 0 or query_block <= 0 or num_devices <= 0:
        return [[] for _ in range(max(0, num_devices))]
    blocks = [
        (start, min(seq_len, start + query_block))
        for start in range(0, seq_len, query_block)
    ]
    assignments = [[] for _ in range(num_devices)]
    for block_index, block in enumerate(blocks):
        assignments[block_index % num_devices].append(block)
    return assignments


def _synchronise(device):
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize(device)
    elif torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _compute_indexer_query_block_topk(
    q_tile, k_full, weights_tile, causal_tile, key_block, top_k, scale,
):
    """Exact DeepSeek-V4 indexer top-k for one bounded query block."""
    batch, query_len = q_tile.shape[:2]
    device = q_tile.device
    q_float = q_tile.float()
    weights_float = weights_tile.float()
    compressed_len = k_full.shape[1]
    running_values = torch.full(
        (batch, query_len, top_k), float("-inf"),
        dtype=torch.float32, device=device,
    )
    running_indices = torch.full(
        (batch, query_len, top_k), -1,
        dtype=torch.int64, device=device,
    )
    for k_start in range(0, compressed_len, key_block):
        k_end = min(compressed_len, k_start + key_block)
        k_tile = k_full[:, k_start:k_end].float()
        scores = torch.matmul(
            q_float, k_tile.transpose(-1, -2).unsqueeze(1)
        )
        scores = torch.relu(scores) * scale
        index_scores = torch.matmul(
            weights_float.unsqueeze(-2), scores
        ).squeeze(-2)
        key_ids = torch.arange(k_start, k_end, device=device)
        index_scores = index_scores.masked_fill(
            key_ids.view(1, 1, -1) >= causal_tile.unsqueeze(-1),
            float("-inf"),
        )
        candidate_values = torch.cat((running_values, index_scores), dim=-1)
        candidate_indices = torch.cat((
            running_indices,
            key_ids.view(1, 1, -1).expand(batch, query_len, -1),
        ), dim=-1)
        running_values, selected = torch.topk(
            candidate_values, top_k, dim=-1, sorted=True
        )
        running_indices = candidate_indices.gather(-1, selected)
    invalid = (
        (running_indices < 0)
        | (running_indices >= causal_tile.unsqueeze(-1))
    )
    return torch.where(
        invalid, torch.full_like(running_indices, -1), running_indices
    ).to(torch.int32)


def _tp_query_parallel_indexer_topk(
    q, compressed, weights, position_ids, compress_rate, query_block,
    key_block, top_k, scale, devices, owner_device,
):
    """Exact query-parallel DeepSeek-V4 indexer.

    Compressed K is copied once per active helper and reused for every assigned
    query block.  Each helper owns complete K, so no cross-device top-k merge is
    needed; only int32 indices are gathered back to the layer owner.
    """
    assignments = _query_block_assignments(q.shape[1], query_block, len(devices))
    active = [
        (device, blocks) for device, blocks in zip(devices, assignments) if blocks
    ]
    if not active:
        return torch.empty(
            q.shape[0], 0, top_k, dtype=torch.int32, device=owner_device
        )
    started = time.perf_counter()
    k_by_device = []
    for device, _ in active:
        k_by_device.append((
            device,
            compressed if _same_device(device, owner_device)
            else compressed.to(device, non_blocking=True),
        ))
    k_replicate = time.perf_counter() - started

    dispatch_started = time.perf_counter()
    block_states = []
    for (device, blocks), (_, k_dev) in zip(active, k_by_device):
        device_states = []
        for q_start, q_end in blocks:
            q_tile = q[:, q_start:q_end]
            w_tile = weights[:, q_start:q_end]
            # DeepSeek compressed entries are indexed by compression window,
            # not raw token position (the latter is used by GLM indexer).
            causal_tile = (
                (position_ids[:, q_start:q_end] + 1) // compress_rate
            )
            if not _same_device(device, owner_device):
                q_tile = q_tile.to(device, non_blocking=True)
                w_tile = w_tile.to(device, non_blocking=True)
                causal_tile = causal_tile.to(device, non_blocking=True)
            indices = _compute_indexer_query_block_topk(
                q_tile, k_dev, w_tile, causal_tile,
                key_block, top_k, scale,
            )
            device_states.append((q_start, q_end, indices))
        block_states.append((device, device_states))
    dispatch_compute = time.perf_counter() - dispatch_started

    sync_started = time.perf_counter()
    for device, _ in block_states:
        _synchronise(device)
    sync_wait = time.perf_counter() - sync_started

    gather_started = time.perf_counter()
    ordered = []
    remote = False
    for device, states in block_states:
        local = torch.cat([item[2] for item in states], dim=1)
        if not _same_device(device, owner_device):
            local = local.to(owner_device, non_blocking=True)
            remote = True
        offset = 0
        for q_start, q_end, _ in states:
            q_len = q_end - q_start
            ordered.append((q_start, local[:, offset:offset + q_len]))
            offset += q_len
    ordered.sort(key=lambda item: item[0])
    if remote:
        _synchronise(owner_device)
    result = torch.cat([item[1] for item in ordered], dim=1)
    logger.info(
        "[DeepSeek-V4 TP timing] strategy=query_parallel indexer q=%d k=%d "
        "devices=%s query_blocks=%s K_replicate_launch=%.4fs "
        "Q_dispatch_compute_launch=%.4fs sync_wait=%.4fs indices_gather=%.4fs "
        "total=%.4fs [host launch timings]",
        q.shape[1], compressed.shape[1], devices,
        [blocks for _, blocks in active], k_replicate, dispatch_compute,
        sync_wait, time.perf_counter() - gather_started,
        time.perf_counter() - started,
    )
    return result


def _compute_attention_query_block(
    q_tile, kv_full, compressed, selected_tile, position_tile,
    scaling, window, compress_rate, sink,
):
    """Compute one exact DSv4 attention query block with bounded scores."""
    batch, heads, query_len, _ = q_tile.shape
    key_len = kv_full.shape[2]
    q_float = q_tile.float()
    key_positions = torch.arange(key_len, device=q_tile.device)
    local_start = max(0, int(position_tile.min().item()) - window + 1)
    local_end = min(key_len, int(position_tile.max().item()) + 1)
    local_kv = kv_full[:, :, local_start:local_end].float()
    local_scores = torch.matmul(q_float, local_kv.transpose(-1, -2)) * scaling
    local_ids = key_positions[local_start:local_end]
    invalid_local = (
        (local_ids.view(1, 1, 1, -1) > position_tile[:, None, :, None])
        | (local_ids.view(1, 1, 1, -1)
           <= position_tile[:, None, :, None] - window)
    )
    local_scores.masked_fill_(invalid_local, float("-inf"))

    compressed_values = None
    compressed_scores = None
    if compressed is not None and compressed.shape[1] > 0:
        if selected_tile is not None:
            ids = selected_tile
            compressed_values = _gather_compressed(compressed, ids).float()
            compressed_scores = torch.einsum(
                "bhqd,bqkd->bhqk", q_float, compressed_values
            ) * scaling
            compressed_len = compressed.shape[1]
            compressed_limit = (position_tile + 1) // compress_rate
            valid = (
                (ids >= 0) & (ids < compressed_len)
                & (ids < compressed_limit.unsqueeze(-1))
            )
            compressed_scores.masked_fill_(~valid[:, None], float("-inf"))
        else:
            compressed_values = compressed.float()
            compressed_scores = torch.matmul(
                q_float, compressed_values.transpose(-1, -2).unsqueeze(1)
            ) * scaling
            entry_ids = torch.arange(compressed.shape[1], device=q_tile.device)
            compressed_limit = (position_tile + 1) // compress_rate
            valid = entry_ids.view(1, 1, -1) < compressed_limit.unsqueeze(-1)
            compressed_scores.masked_fill_(~valid[:, None], float("-inf"))

    score_parts = [local_scores]
    if compressed_scores is not None:
        score_parts.append(compressed_scores)
    sink = sink.float().view(1, -1, 1, 1)
    score_parts.append(sink.expand(batch, -1, query_len, -1))
    logits = torch.cat(score_parts, dim=-1)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    probabilities = F.softmax(logits, dim=-1)
    local_n = local_scores.shape[-1]
    local_prob = probabilities[..., :local_n]
    output = torch.matmul(local_prob.to(local_kv.dtype), local_kv)
    if compressed_scores is not None:
        comp_prob = probabilities[..., local_n:local_n + compressed_scores.shape[-1]]
        if selected_tile is not None:
            output = output + torch.einsum(
                "bhqk,bqkd->bhqd", comp_prob, compressed_values
            )
        else:
            output = output + torch.matmul(
                comp_prob, compressed_values.unsqueeze(1)
            )
    return output.to(q_tile.dtype)


def _tp_query_parallel_attention(
    q, kv, compressed, selected, position_ids, compress_rate, scaling,
    window, query_block, devices, owner_device, sink,
):
    """Run DSv4 attention by query blocks with one full-K replica per helper."""
    assignments = _query_block_assignments(q.shape[2], query_block, len(devices))
    active = [
        (device, blocks) for device, blocks in zip(devices, assignments) if blocks
    ]
    if not active:
        return torch.empty_like(q)
    started = time.perf_counter()
    replicas = []
    for device, _ in active:
        if _same_device(device, owner_device):
            kv_dev, compressed_dev = kv, compressed
        else:
            kv_dev = kv.to(device, non_blocking=True)
            compressed_dev = (
                compressed.to(device, non_blocking=True)
                if compressed is not None else None
            )
        replicas.append((device, kv_dev, compressed_dev))
    k_replicate = time.perf_counter() - started

    dispatch_started = time.perf_counter()
    block_states = []
    for (device, blocks), (_, kv_dev, compressed_dev) in zip(active, replicas):
        device_states = []
        sink_dev = sink if _same_device(device, owner_device) else sink.to(
            device, non_blocking=True
        )
        for q_start, q_end in blocks:
            q_tile = q[:, :, q_start:q_end]
            selected_tile = selected[:, q_start:q_end] if selected is not None else None
            position_tile = position_ids[:, q_start:q_end]
            if not _same_device(device, owner_device):
                q_tile = q_tile.to(device, non_blocking=True)
                if selected_tile is not None:
                    selected_tile = selected_tile.to(device, non_blocking=True)
                position_tile = position_tile.to(device, non_blocking=True)
            output = _compute_attention_query_block(
                q_tile, kv_dev, compressed_dev, selected_tile, position_tile,
                scaling, window, compress_rate, sink_dev,
            )
            device_states.append((q_start, q_end, output))
        block_states.append((device, device_states))
    dispatch_compute = time.perf_counter() - dispatch_started

    sync_started = time.perf_counter()
    for device, _ in block_states:
        _synchronise(device)
    sync_wait = time.perf_counter() - sync_started

    gather_started = time.perf_counter()
    ordered = []
    remote = False
    for device, states in block_states:
        local = torch.cat([item[2] for item in states], dim=2)
        if not _same_device(device, owner_device):
            local = local.to(owner_device, non_blocking=True)
            remote = True
        offset = 0
        for q_start, q_end, _ in states:
            q_len = q_end - q_start
            ordered.append((q_start, local[:, :, offset:offset + q_len]))
            offset += q_len
    ordered.sort(key=lambda item: item[0])
    if remote:
        _synchronise(owner_device)
    result = torch.cat([item[1] for item in ordered], dim=2)
    logger.info(
        "[DeepSeek-V4 TP timing] strategy=query_parallel attention q=%d "
        "devices=%s query_blocks=%s K_replicate_launch=%.4fs "
        "Q_dispatch_compute_launch=%.4fs sync_wait=%.4fs "
        "output_gather=%.4fs total=%.4fs [host launch timings]",
        q.shape[2], devices, [blocks for _, blocks in active], k_replicate,
        dispatch_compute, sync_wait, time.perf_counter() - gather_started,
        time.perf_counter() - started,
    )
    return result


def _cache_layer(past_key_values, layer_idx):
    if past_key_values is None:
        return None
    return past_key_values.layers[layer_idx]


def _compress_entries(compressor, hidden_states, position_ids, past_key_values,
                      layer_idx, apply_rotary_pos_emb, *, overlap: bool):
    """Run the official HCA/CSA window compressor without a dense bias."""
    batch = hidden_states.shape[0]
    kv = compressor.kv_proj(hidden_states)
    gate = compressor.gate_proj(hidden_states)
    cache_layer = _cache_layer(past_key_values, layer_idx)
    if cache_layer is None:
        usable = (kv.shape[1] // compressor.compress_rate) * compressor.compress_rate
        chunk_kv, chunk_gate, first_position = kv[:, :usable], gate[:, :usable], 0
    else:
        chunk_kv, chunk_gate, first_position = cache_layer.store_compression_weights(
            "compressor", kv, gate
        )

    ratio = compressor.compress_rate
    if chunk_kv.shape[1] > 0:
        windows = chunk_kv.shape[1] // ratio
        chunk_kv = chunk_kv.view(batch, windows, ratio, -1)
        chunk_gate = chunk_gate.view(batch, windows, ratio, -1) + compressor.position_bias
        if overlap:
            head_dim = compressor.head_dim
            mixed_kv = chunk_kv.new_zeros((batch, windows, 2 * ratio, head_dim))
            mixed_gate = chunk_gate.new_full(
                (batch, windows, 2 * ratio, head_dim), float("-inf")
            )
            mixed_kv[:, :, ratio:] = chunk_kv[..., head_dim:]
            mixed_gate[:, :, ratio:] = chunk_gate[..., head_dim:]
            if windows > 1:
                mixed_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, :head_dim]
                mixed_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, :head_dim]
            if cache_layer is not None:
                prior_kv, prior_gate = cache_layer.update_overlap_state(
                    "compressor", chunk_kv, chunk_gate, head_dim
                )
                if prior_kv is not None:
                    mixed_kv[:, 0, :ratio] = prior_kv.to(mixed_kv.dtype)
                    mixed_gate[:, 0, :ratio] = prior_gate.to(mixed_gate.dtype)
            compressed = compressor.kv_norm(
                (mixed_kv * mixed_gate.softmax(dim=2, dtype=torch.float32)
                 .to(mixed_kv.dtype)).sum(dim=2)
            )
        else:
            compressed = compressor.kv_norm(
                (chunk_kv * chunk_gate.softmax(dim=2, dtype=torch.float32)
                 .to(chunk_kv.dtype)).sum(dim=2)
            )
        positions = torch.arange(windows, device=compressed.device)
        positions = (positions * ratio + first_position).unsqueeze(0).expand(batch, -1)
        cos, sin = compressor.rotary_emb(
            compressed, position_ids=positions, layer_type=compressor.rope_layer_type
        )
        compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
    else:
        compressed = chunk_kv.new_zeros((batch, 0, compressor.head_dim))

    if cache_layer is not None:
        compressed = cache_layer.update_compressor_states("compressor", compressed)
    return compressed


def _install_blockwise_indexer(module, query_block: int, key_block: int,
                               threshold: int, *, parallel_mode: str = "pp",
                               device_groups=None) -> bool:
    cls = getattr(module, "DeepseekV4Indexer", None)
    if cls is None:
        return False
    config = (query_block, key_block, threshold, str(parallel_mode).lower(),
              tuple(tuple(group) for group in _normalise_device_groups(device_groups)))
    if (getattr(cls, "_acc_blockwise_installed", False)
            and getattr(cls, "_acc_blockwise_config", None) == config):
        return True
    original = getattr(cls, "_acc_blockwise_original_forward", cls.forward)
    rope = module.apply_rotary_pos_emb

    @torch.no_grad()
    def blockwise_forward(self, hidden_states, q_residual, position_ids,
                          past_key_values, layer_idx):
        batch, seq_len, _ = hidden_states.shape
        if seq_len <= threshold:
            return original(
                self, hidden_states, q_residual, position_ids,
                past_key_values, layer_idx
            )

        cache_layer = _cache_layer(past_key_values, layer_idx)
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_position = cache_layer.store_compression_weights(
                "indexer", kv, gate
            )

        ratio = self.compress_rate
        if chunk_kv.shape[1] > 0:
            windows = chunk_kv.shape[1] // ratio
            chunk_kv = chunk_kv.view(batch, windows, ratio, -1)
            chunk_gate = chunk_gate.view(batch, windows, ratio, -1) + self.position_bias
            mixed_kv = chunk_kv.new_zeros(
                (batch, windows, 2 * ratio, self.head_dim)
            )
            mixed_gate = chunk_gate.new_full(
                (batch, windows, 2 * ratio, self.head_dim), float("-inf")
            )
            mixed_kv[:, :, ratio:] = chunk_kv[..., self.head_dim:]
            mixed_gate[:, :, ratio:] = chunk_gate[..., self.head_dim:]
            if windows > 1:
                mixed_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, :self.head_dim]
                mixed_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, :self.head_dim]
            if cache_layer is not None:
                prior_kv, prior_gate = cache_layer.update_overlap_state(
                    "indexer", chunk_kv, chunk_gate, self.head_dim
                )
                if prior_kv is not None:
                    mixed_kv[:, 0, :ratio] = prior_kv.to(mixed_kv.dtype)
                    mixed_gate[:, 0, :ratio] = prior_gate.to(mixed_gate.dtype)
            compressed = self.kv_norm(
                (mixed_kv * mixed_gate.softmax(dim=2, dtype=torch.float32)
                 .to(mixed_kv.dtype)).sum(dim=2)
            )
            positions = torch.arange(windows, device=compressed.device)
            positions = (positions * ratio + first_position).unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(
                compressed, position_ids=positions, layer_type=self.rope_layer_type
            )
            compressed = rope(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))
        if cache_layer is not None:
            compressed = cache_layer.update_compressor_states("indexer", compressed)

        compressed_len = compressed.shape[1]
        top_k = min(self.index_topk, compressed_len)
        if top_k == 0:
            return torch.empty(
                batch, seq_len, 0, dtype=torch.long, device=hidden_states.device
            )
        cos_q, sin_q = self.rotary_emb(
            hidden_states, position_ids=position_ids, layer_type=self.rope_layer_type
        )
        q = self.q_b_proj(q_residual).view(
            batch, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        q = rope(q, cos_q, sin_q).transpose(1, 2)  # [B,S,H,D]
        # Keep the projection dtype while dispatching query blocks; each
        # helper casts locally to FP32 for the exact score computation.
        weights = self.scorer.weights_proj(hidden_states)
        weights = weights * self.scorer.weights_scaling
        tp_devices = (
            _select_tp_devices(q.device, device_groups)
            if str(parallel_mode).lower() == "tp" else []
        )
        if len(tp_devices) >= 2:
            return _tp_query_parallel_indexer_topk(
                q, compressed, weights, position_ids, self.compress_rate,
                query_block, key_block, top_k, self.scorer.softmax_scale,
                tp_devices, q.device,
            )
        outputs = []
        for q_start in range(0, seq_len, query_block):
            q_end = min(seq_len, q_start + query_block)
            q_tile = q[:, q_start:q_end].float()
            w_tile = weights[:, q_start:q_end].float()
            running_values = torch.full(
                (batch, q_end - q_start, top_k), float("-inf"),
                device=q.device, dtype=torch.float32
            )
            running_indices = torch.full(
                (batch, q_end - q_start, top_k), -1,
                device=q.device, dtype=torch.long
            )
            causal_threshold = (
                position_ids[:, q_start:q_end] + 1
            ) // self.compress_rate
            for k_start in range(0, compressed_len, key_block):
                k_end = min(compressed_len, k_start + key_block)
                scores = torch.matmul(
                    q_tile, compressed[:, k_start:k_end].float()
                    .transpose(-1, -2).unsqueeze(1)
                )
                scores = F.relu(scores) * self.scorer.softmax_scale
                scores = (scores * w_tile.unsqueeze(-1)).sum(dim=2)
                key_ids = torch.arange(k_start, k_end, device=q.device)
                scores.masked_fill_(
                    key_ids.view(1, 1, -1) >= causal_threshold.unsqueeze(-1),
                    float("-inf")
                )
                candidate_values = torch.cat((running_values, scores), dim=-1)
                candidate_ids = torch.cat((
                    running_indices,
                    key_ids.view(1, 1, -1).expand(batch, q_end - q_start, -1),
                ), dim=-1)
                running_values, selected = candidate_values.topk(
                    top_k, dim=-1, sorted=True
                )
                running_indices = candidate_ids.gather(-1, selected)
            invalid = (
                (running_indices < 0)
                | (running_indices >= causal_threshold.unsqueeze(-1))
            )
            outputs.append(torch.where(
                invalid, torch.full_like(running_indices, -1), running_indices
            ))
        return torch.cat(outputs, dim=1)

    cls.forward = blockwise_forward
    cls._acc_blockwise_installed = True
    cls._acc_blockwise_original_forward = original
    cls._acc_blockwise_config = config
    return True


def _gather_compressed(compressed: torch.Tensor, indices: torch.Tensor):
    """Gather ``[B,Q,K,D]`` without materializing an expanded source."""
    batch, _, dim = compressed.shape
    safe = indices.clamp_min(0)
    offsets = torch.arange(batch, device=compressed.device).view(batch, 1, 1)
    flat_ids = (safe + offsets * compressed.shape[1]).reshape(-1)
    gathered = compressed.reshape(-1, dim).index_select(0, flat_ids)
    return gathered.view(*safe.shape, dim)


def _install_blockwise_attention(module, query_block: int, threshold: int, *,
                                 parallel_mode: str = "pp",
                                 device_groups=None) -> bool:
    cls = getattr(module, "DeepseekV4Attention", None)
    if cls is None:
        return False
    config = (query_block, threshold, str(parallel_mode).lower(),
              tuple(tuple(group) for group in _normalise_device_groups(device_groups)))
    if (getattr(cls, "_acc_blockwise_installed", False)
            and getattr(cls, "_acc_blockwise_config", None) == config):
        return True
    original = getattr(cls, "_acc_blockwise_original_forward", cls.forward)
    rope = module.apply_rotary_pos_emb

    @torch.no_grad()
    def blockwise_forward(self, hidden_states, position_embeddings, position_ids,
                          attention_mask=None, past_key_values=None, **kwargs):
        batch, seq_len, _ = hidden_states.shape
        if seq_len <= threshold:
            return original(
                self, hidden_states, position_embeddings, position_ids,
                attention_mask, past_key_values=past_key_values, **kwargs
            )
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings[self.rope_layer_type]
        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        q = rope(self.q_b_norm(q), cos, sin)
        kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
        kv = rope(kv, cos, sin)
        if past_key_values is not None:
            kv = past_key_values.update(kv, kv, self.layer_idx)[0]

        compressed = None
        selected = None
        if self.compressor is not None:
            is_csa = hasattr(self.compressor, "indexer")
            compressed = _compress_entries(
                self.compressor, hidden_states, position_ids, past_key_values,
                self.layer_idx, rope, overlap=is_csa
            )
            if is_csa:
                selected = self.compressor.indexer(
                    hidden_states, q_residual, position_ids,
                    past_key_values, self.layer_idx
                )

        key_len = kv.shape[2]
        tp_devices = (
            _select_tp_devices(q.device, device_groups)
            if str(parallel_mode).lower() == "tp" else []
        )
        if len(tp_devices) >= 2:
            attn_output = _tp_query_parallel_attention(
                q, kv, compressed, selected, position_ids,
                getattr(self.compressor, "compress_rate", 1), self.scaling,
                int(self.sliding_window or key_len), query_block,
                tp_devices, q.device, self.sinks,
            )
            attn_output = rope(attn_output, cos, -sin).transpose(1, 2)
            grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
            grouped = self.o_a_proj(grouped).flatten(2)
            return self.o_b_proj(grouped), None
        key_positions = torch.arange(key_len, device=hidden_states.device)
        outputs = []
        sink = self.sinks.float().view(1, -1, 1, 1)
        window = int(self.sliding_window or key_len)
        for q_start in range(0, seq_len, query_block):
            q_end = min(seq_len, q_start + query_block)
            q_tile = q[:, :, q_start:q_end].float()
            query_positions = position_ids[:, q_start:q_end]
            local_start = max(0, int(query_positions.min().item()) - window + 1)
            local_end = min(key_len, int(query_positions.max().item()) + 1)
            local_kv = kv[:, :, local_start:local_end].float()
            local_scores = torch.matmul(q_tile, local_kv.transpose(-1, -2)) * self.scaling
            local_ids = key_positions[local_start:local_end]
            invalid_local = (
                (local_ids.view(1, 1, 1, -1) > query_positions[:, None, :, None])
                | (local_ids.view(1, 1, 1, -1)
                   <= query_positions[:, None, :, None] - window)
            )
            local_scores.masked_fill_(invalid_local, float("-inf"))

            compressed_values = None
            compressed_scores = None
            compressed_valid = None
            if compressed is not None and compressed.shape[1] > 0:
                if selected is not None:
                    ids = selected[:, q_start:q_end]
                    compressed_valid = ids >= 0
                    compressed_values = _gather_compressed(compressed, ids).float()
                    compressed_scores = torch.einsum(
                        "bhqd,bqkd->bhqk", q_tile, compressed_values
                    ) * self.scaling
                    compressed_scores.masked_fill_(
                        ~compressed_valid[:, None], float("-inf")
                    )
                else:
                    compressed_values = compressed.float()
                    compressed_scores = torch.matmul(
                        q_tile, compressed_values.transpose(-1, -2).unsqueeze(1)
                    ) * self.scaling
                    threshold_ids = (
                        query_positions + 1
                    ) // self.compressor.compress_rate
                    entry_ids = torch.arange(
                        compressed.shape[1], device=compressed.device
                    )
                    compressed_valid = (
                        entry_ids.view(1, 1, -1) < threshold_ids.unsqueeze(-1)
                    )
                    compressed_scores.masked_fill_(
                        ~compressed_valid[:, None], float("-inf")
                    )

            score_parts = [local_scores]
            if compressed_scores is not None:
                score_parts.append(compressed_scores)
            score_parts.append(sink.expand(batch, -1, q_end - q_start, -1))
            logits = torch.cat(score_parts, dim=-1)
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probabilities = F.softmax(logits, dim=-1)
            local_n = local_scores.shape[-1]
            local_prob = probabilities[..., :local_n]
            output = torch.matmul(local_prob.to(local_kv.dtype), local_kv)
            if compressed_scores is not None:
                comp_prob = probabilities[..., local_n:local_n + compressed_scores.shape[-1]]
                if selected is not None:
                    output = output + torch.einsum(
                        "bhqk,bqkd->bhqd", comp_prob, compressed_values
                    )
                else:
                    output = output + torch.matmul(
                        comp_prob, compressed_values.unsqueeze(1)
                    )
            outputs.append(output.to(hidden_states.dtype))

        attn_output = torch.cat(outputs, dim=2)  # [B,H,S,D]
        attn_output = rope(attn_output, cos, -sin).transpose(1, 2)
        grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        return self.o_b_proj(grouped), None

    cls.forward = blockwise_forward
    cls._acc_blockwise_installed = True
    cls._acc_blockwise_original_forward = original
    cls._acc_blockwise_config = config
    return True


def _install_mask_bypass(module, threshold: int) -> None:
    original = getattr(module, "create_sliding_window_causal_mask", None)
    if original is None or getattr(original, "_acc_v4_mask_wrapper", False):
        return

    def bounded_mask(*args, **kwargs):
        inputs = kwargs.get("inputs_embeds")
        if inputs is None and len(args) >= 2:
            inputs = args[1]
        if isinstance(inputs, torch.Tensor) and inputs.shape[1] > threshold:
            return None
        return original(*args, **kwargs)

    bounded_mask._acc_v4_mask_wrapper = True
    module.create_sliding_window_causal_mask = bounded_mask


def install_deepseek_v4_blockwise_runtime(
    *, query_block: int | None = None, key_block: int | None = None,
    threshold: int | None = None, parallel_mode: str = "pp",
    device_groups=None,
) -> bool:
    """Install long-prefill patches when official DeepSeek-V4 is available."""
    try:
        from transformers.models.deepseek_v4 import modeling_deepseek_v4 as module
    except ImportError:
        return False
    threshold = max(1, threshold or int(os.getenv("ACC_DEEPSEEK_V4_BLOCKWISE_THRESHOLD", "8192")))
    query_block = max(1, query_block or int(os.getenv("ACC_DEEPSEEK_V4_QUERY_BLOCK", "64")))
    key_block = max(1, key_block or int(os.getenv("ACC_DEEPSEEK_V4_KEY_BLOCK", "1024")))
    mode = str(parallel_mode or "pp").lower()
    if mode not in {"pp", "tp"}:
        mode = "pp"
    groups = _normalise_device_groups(device_groups)
    indexer_ok = _install_blockwise_indexer(
        module, query_block, key_block, threshold,
        parallel_mode=mode, device_groups=groups,
    )
    attention_ok = _install_blockwise_attention(
        module, query_block, threshold,
        parallel_mode=mode, device_groups=groups,
    )
    if indexer_ok and attention_ok:
        _install_mask_bypass(module, threshold)
        if not getattr(module, "_acc_v4_runtime_logged", False):
            logger.info(
                "  DeepSeek-V4 blockwise long-prefill 已安装: threshold=%d, "
                "query_block=%d, key_block=%d, mode=%s, strategy=%s, groups=%s",
                threshold, query_block, mode,
                "query_parallel" if mode == "tp" else "owner_blockwise", groups,
            )
            module._acc_v4_runtime_logged = True
        return True
    return False
