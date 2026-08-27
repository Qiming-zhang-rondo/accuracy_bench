"""Memory-bounded GLM-MoE-DSA indexer and attention for long prefill.

Transformers' eager indexer materializes ``[batch, query, heads, keys]``
scores.  For a 65k-token prompt that intermediate is hundreds of GiB.  This
patch preserves the same projections, ReLU, causal mask and global top-k, but
processes query/key tiles and keeps only the running top-k values.

The eager attention path has a second, independent quadratic allocation after
the indexer: ``[batch, heads, query, key]`` attention scores.  For long
prefills we also consume the exact DSA-selected indices directly, process
small query/selected-key tiles and use online softmax.  No dense QxK sparse
mask or score tensor is materialized.
"""

from __future__ import annotations

import logging
import os
import time

import torch

logger = logging.getLogger(__name__)


def _gather_selected_attention_states(states, indices):
    """Gather ``[B,H,Q,T,D]`` from ``[B,H,K,D]`` without expanding K.

    ``indices`` is shared by all attention heads.  Flattening batch and head
    lets ``index_select`` gather only the selected vectors instead of first
    expanding the complete K/V tensor to ``[B,H,Q,K,D]``.
    """
    batch_size, num_heads, key_len, head_dim = states.shape
    safe = indices.clamp(0, max(0, key_len - 1)).long()
    query_len, selected_len = safe.shape[1:]
    offsets = torch.arange(
        batch_size * num_heads, device=states.device, dtype=torch.long
    ).view(batch_size, num_heads, 1, 1) * key_len
    flat_ids = (
        safe[:, None, :, :].expand(-1, num_heads, -1, -1) + offsets
    ).reshape(-1)
    # Callers make states contiguous once per attention forward, so this view
    # does not copy the full K/V tensor for every query block.
    flat_states = states.view(batch_size * num_heads * key_len, head_dim)
    gathered = flat_states.index_select(0, flat_ids)
    return gathered.view(
        batch_size, num_heads, query_len, selected_len, head_dim
    )


def _selected_attention_mask(
    attention_mask, indices, q_start, q_end, key_len,
):
    """Gather only selected entries from a compact or additive mask."""
    if attention_mask is None:
        return None
    safe = indices.clamp(0, max(0, key_len - 1)).long()
    batch_size, query_len, selected_len = safe.shape
    if attention_mask.dim() == 2:
        gathered = attention_mask.gather(
            1, safe.reshape(batch_size, -1)
        )
        return gathered.view(batch_size, 1, query_len, selected_len)
    if attention_mask.dim() == 3:
        mask = attention_mask[:, q_start:q_end, :]
        return mask.gather(-1, safe).unsqueeze(1)
    if attention_mask.dim() == 4:
        mask = attention_mask[:, :, q_start:q_end, :]
        gather_ids = safe[:, None, :, :].expand(
            -1, mask.shape[1], -1, -1
        )
        return mask.gather(-1, gather_ids)
    raise ValueError(
        "GLM DSA attention expects a [B,K], [B,Q,K], or [B,H,Q,K] mask"
    )


def _compute_sparse_attention_query_block(
    query_tile, key_states, value_states, selected_indices, position_tile,
    attention_mask, q_start, scaling, selected_block,
):
    """Compute exact sparse attention for one bounded query block.

    Softmax is accumulated online across selected-key tiles, so the result is
    mathematically equivalent to softmax over all DSA-selected keys while the
    temporary gathered K/V tensors stay bounded by
    ``[B,H,Q_block,selected_block,D]``.
    """
    batch_size, num_heads, query_len, _ = query_tile.shape
    key_len = key_states.shape[2]
    value_dim = value_states.shape[-1]
    query_float = query_tile.float()
    running_max = torch.full(
        (batch_size, num_heads, query_len, 1), float("-inf"),
        dtype=torch.float32, device=query_tile.device,
    )
    running_sum = torch.zeros_like(running_max)
    running_output = torch.zeros(
        (batch_size, num_heads, query_len, value_dim),
        dtype=torch.float32, device=query_tile.device,
    )

    total_selected = selected_indices.shape[-1]
    for selected_start in range(0, total_selected, selected_block):
        selected_end = min(total_selected, selected_start + selected_block)
        indices = selected_indices[:, :, selected_start:selected_end]
        valid = (indices >= 0) & (indices < key_len)
        # DSA top-k is causal, but early query positions are padded by topk
        # with future -inf candidates.  Reapply causality after gather.
        valid = valid & (indices <= position_tile[:, :, None])

        selected_keys = _gather_selected_attention_states(key_states, indices)
        scores = torch.matmul(
            query_float.unsqueeze(-2),
            selected_keys.float().transpose(-1, -2),
        ).squeeze(-2) * scaling
        del selected_keys

        mask = _selected_attention_mask(
            attention_mask, indices, q_start, q_start + query_len, key_len
        )
        if mask is not None:
            if attention_mask.dim() == 2:
                # Transformers' compact padding mask uses 1/True for valid
                # keys and 0/False for padding.
                valid = valid[:, None] & (mask > 0)
            elif mask.dtype == torch.bool or not mask.dtype.is_floating_point:
                valid = valid[:, None] & mask
            else:
                mask_float = mask.float()
                # HF additive masks use either -inf or dtype minimum for
                # disallowed entries.  Preserve finite additive biases while
                # treating those sentinels as invalid.
                mask_floor = torch.finfo(mask.dtype).min / 2
                mask_valid = torch.isfinite(mask_float) & (mask_float > mask_floor)
                valid = valid[:, None] & mask_valid
                scores = scores + mask_float
        else:
            valid = valid[:, None]
        scores = scores.masked_fill(~valid, float("-inf"))

        tile_max = scores.max(dim=-1, keepdim=True).values
        new_max = torch.maximum(running_max, tile_max)
        old_scale = torch.where(
            torch.isfinite(running_max),
            torch.exp(running_max - new_max),
            torch.zeros_like(running_max),
        )
        shifted = torch.where(
            valid,
            scores - new_max,
            torch.full_like(scores, float("-inf")),
        )
        exp_scores = torch.where(
            torch.isfinite(new_max),
            torch.exp(shifted),
            torch.zeros_like(scores),
        )
        new_sum = running_sum * old_scale + exp_scores.sum(
            dim=-1, keepdim=True
        )

        # Release gathered K before gathering V.  K and V have the same
        # selected shape, so the allocator can reuse the bounded workspace.
        selected_values = _gather_selected_attention_states(
            value_states, indices
        )
        tile_output = torch.matmul(
            exp_scores.unsqueeze(-2), selected_values.float()
        ).squeeze(-2)
        del selected_values, exp_scores, scores
        running_output = running_output * old_scale + tile_output
        running_max = new_max
        running_sum = new_sum

    tiny = torch.finfo(running_sum.dtype).tiny
    output = running_output / running_sum.clamp_min(tiny)
    return output.to(query_tile.dtype)


def _tp_sparse_query_parallel_attention(
    query_states, key_states, value_states, topk_indices, position_ids,
    attention_mask, scaling, query_block, selected_block, devices, base_device,
):
    """Run sparse attention query blocks across one side's TP device group.

    Full K/V is copied exactly once to each active helper and reused for every
    assigned query block.  Q/top-k/positions go only to the responsible helper;
    only the completed attention output is gathered back to the layer owner.
    """
    batch_size, num_heads, seq_len, _ = query_states.shape
    value_dim = value_states.shape[-1]
    assignments = _query_block_assignments(
        seq_len, query_block, len(devices)
    )
    active = [
        (device, blocks) for device, blocks in zip(devices, assignments)
        if blocks
    ]
    if not active:
        return torch.empty(
            batch_size, num_heads, 0, value_dim,
            dtype=query_states.dtype, device=base_device,
        )

    started = time.perf_counter()
    replicated = []
    for device, blocks in active:
        if _same_device(device, base_device):
            key_dev = key_states
            value_dev = value_states
            compact_mask = attention_mask if (
                attention_mask is not None and attention_mask.dim() == 2
            ) else None
        else:
            key_dev = key_states.to(device, non_blocking=True)
            value_dev = value_states.to(device, non_blocking=True)
            compact_mask = (
                attention_mask.to(device, non_blocking=True)
                if attention_mask is not None and attention_mask.dim() == 2
                else None
            )
        replicated.append((device, blocks, key_dev, value_dev, compact_mask))
    replicate_launch = time.perf_counter() - started

    launch_started = time.perf_counter()
    states_by_device = []
    for device, blocks, key_dev, value_dev, compact_mask in replicated:
        local_len = sum(q_end - q_start for q_start, q_end in blocks)
        local_output = torch.empty(
            batch_size, num_heads, local_len, value_dim,
            dtype=query_states.dtype, device=device,
        )
        offset = 0
        for q_start, q_end in blocks:
            q_tile = query_states[:, :, q_start:q_end, :]
            selected_tile = topk_indices[:, q_start:q_end, :]
            position_tile = position_ids[:, q_start:q_end]
            local_mask = compact_mask
            local_q_start = q_start
            if attention_mask is not None and attention_mask.dim() > 2:
                # Correctness fallback for an already-materialized additive
                # mask: transfer only this query block, never full QxK.
                if attention_mask.dim() == 3:
                    local_mask = attention_mask[:, q_start:q_end, :]
                else:
                    local_mask = attention_mask[:, :, q_start:q_end, :]
                local_q_start = 0
            if not _same_device(device, base_device):
                q_tile = q_tile.to(device, non_blocking=True)
                selected_tile = selected_tile.to(device, non_blocking=True)
                position_tile = position_tile.to(device, non_blocking=True)
                if local_mask is not None and local_mask is not compact_mask:
                    local_mask = local_mask.to(device, non_blocking=True)
            block_output = _compute_sparse_attention_query_block(
                q_tile,
                key_dev,
                value_dev,
                selected_tile,
                position_tile,
                local_mask,
                local_q_start,
                scaling,
                selected_block,
            )
            q_len = q_end - q_start
            local_output[:, :, offset:offset + q_len, :] = block_output
            offset += q_len
        states_by_device.append((device, blocks, local_output))
    compute_launch = time.perf_counter() - launch_started

    sync_started = time.perf_counter()
    for device, _, _ in states_by_device:
        _synchronise(device)
    sync_wait = time.perf_counter() - sync_started

    gather_started = time.perf_counter()
    gathered_blocks = []
    has_remote = False
    for device, blocks, local_output in states_by_device:
        if not _same_device(device, base_device):
            local_output = local_output.to(base_device, non_blocking=True)
            has_remote = True
        offset = 0
        for q_start, q_end in blocks:
            q_len = q_end - q_start
            gathered_blocks.append((
                q_start,
                local_output[:, :, offset:offset + q_len, :],
            ))
            offset += q_len
    if has_remote:
        _synchronise(base_device)
    gathered_blocks.sort(key=lambda item: item[0])
    output = torch.cat([block for _, block in gathered_blocks], dim=2)
    gather_time = time.perf_counter() - gather_started
    logger.info(
        "[GLM DSA attention TP timing] strategy=query_parallel devices=%s "
        "query_blocks_per_device=%s KV_replicate_launch=%.4fs "
        "query_dispatch+compute_launch=%.4fs sync_wait=%.4fs "
        "output_gather=%.4fs total=%.4fs "
        "[host launch timings; device work included in sync_wait]",
        [device for device, _ in active],
        [len(blocks) for _, blocks in active],
        replicate_launch,
        compute_launch,
        sync_wait,
        gather_time,
        time.perf_counter() - started,
    )
    return output


def _normalise_device_groups(device_groups):
    """Return non-empty device groups as canonical string lists."""
    groups = []
    for group in device_groups or ():
        if isinstance(group, str):
            group = group.split(",")
        values = []
        for value in group or ():
            value = str(value).strip()
            if value and value not in values:
                values.append(value)
        if values:
            groups.append(values)
    return groups


def _same_device(left, right):
    """Compare device spellings without requiring a live accelerator."""
    left = str(left)
    right = str(right)
    if left == right:
        return True
    # torch reports ``npu:0`` while callers occasionally pass ``0``.
    def key(value):
        if ":" in value:
            kind, index = value.rsplit(":", 1)
            return kind, index
        return None, value
    left_kind, left_index = key(left)
    right_kind, right_index = key(right)
    if left_index != right_index:
        return False
    return left_kind == right_kind or left_kind is None or right_kind is None


def _select_tp_devices(source_device, device_groups):
    """Pick the TP group owning ``source_device``; otherwise use no TP."""
    groups = _normalise_device_groups(device_groups)
    for group in groups:
        if any(_same_device(source_device, value) for value in group):
            return group
    return []


def _synchronise(device):
    """Synchronise one accelerator after launching work on several devices."""
    device_obj = torch.device(device)
    if device_obj.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize(device_obj)
    elif device_obj.type == "cuda" and hasattr(torch, "cuda"):
        torch.cuda.synchronize(device_obj)


def _tp_indexer_topk(
    q_tile, k, w_tile, q_start, key_block, topk, position_ids,
    attention_mask, scale, devices, base_device,
):
    """Compute one query tile with key-axis tensor parallelism.

    Each device owns a disjoint set of key blocks and keeps a local top-k.
    The union of local top-k sets is sufficient for the exact global top-k;
    only the small ``num_devices * topk`` candidate tensor is copied back.
    """
    batch_size, query_len = q_tile.shape[:2]
    total_keys = k.shape[1]
    assignments = [[] for _ in devices]
    for block_index, k_start in enumerate(range(0, total_keys, key_block)):
        assignments[block_index % len(devices)].append(k_start)

    states = []
    for device, starts in zip(devices, assignments):
        if not starts:
            continue
        q_dev = q_tile.to(device, non_blocking=True)
        w_dev = w_tile.to(device, non_blocking=True)
        pos_dev = position_ids[:, q_start:q_start + query_len].to(
            device, non_blocking=True)
        local_topk = min(topk, sum(
            min(key_block, total_keys - start) for start in starts))
        local_values = torch.full(
            (batch_size, query_len, local_topk), float("-inf"),
            dtype=torch.float32, device=device)
        local_indices = torch.full(
            (batch_size, query_len, local_topk), -1,
            dtype=torch.int64, device=device)
        for k_start in starts:
            k_end = min(total_keys, k_start + key_block)
            k_dev = k[:, k_start:k_end].to(device, non_blocking=True)
            scores = torch.matmul(
                q_dev, k_dev.transpose(-1, -2).float().unsqueeze(1)
            ) * scale
            scores = torch.relu(scores)
            index_scores = torch.matmul(
                w_dev.unsqueeze(-2), scores).squeeze(-2)
            if attention_mask is not None:
                mask_tile = attention_mask[
                    :, q_start:q_start + query_len, k_start:k_end
                ].to(device, non_blocking=True)
                index_scores = index_scores + mask_tile
            else:
                key_positions = torch.arange(k_start, k_end, device=device)
                causal = key_positions[None, None, :] > pos_dev[:, :, None]
                index_scores = index_scores.masked_fill(causal, float("-inf"))
            candidates = torch.cat((local_values, index_scores), dim=-1)
            candidate_indices = torch.cat((
                local_indices,
                torch.arange(k_start, k_end, device=device)
                .view(1, 1, -1).expand(batch_size, query_len, -1),
            ), dim=-1)
            local_values, selected = torch.topk(
                candidates, local_topk, dim=-1, sorted=True)
            local_indices = candidate_indices.gather(-1, selected)
            del k_dev, scores, index_scores, candidates, candidate_indices
        states.append((device, local_values, local_indices))
        del q_dev, w_dev, pos_dev

    # Operations on distinct NPU streams are queued above before waiting.
    for device, _, _ in states:
        _synchronise(device)

    running_values = torch.full(
        (batch_size, query_len, topk), float("-inf"),
        dtype=torch.float32, device=base_device)
    running_indices = torch.full(
        (batch_size, query_len, topk), -1,
        dtype=torch.int64, device=base_device)
    for device, local_values, local_indices in states:
        values = local_values.to(base_device, non_blocking=True)
        indices = local_indices.to(base_device, non_blocking=True)
        candidates = torch.cat((running_values, values), dim=-1)
        candidate_indices = torch.cat((running_indices, indices), dim=-1)
        running_values, selected = torch.topk(
            candidates, topk, dim=-1, sorted=True)
        running_indices = candidate_indices.gather(-1, selected)
        del values, indices, candidates, candidate_indices
    query_positions = position_ids[:, q_start:q_start + query_len]
    valid = (
        (running_indices >= 0)
        & (running_indices <= query_positions[:, :, None])
    )
    return torch.where(valid, running_indices, -1)


def _query_block_assignments(seq_len, query_block, num_devices):
    """Assign individual query blocks round-robin to helper devices.

    The assignment deliberately keeps block boundaries intact.  A device may
    own many blocks, but it must never receive a concatenated mega-shard: the
    score tensor's query dimension is bounded by ``query_block`` for every
    invocation below.
    """
    if seq_len <= 0 or query_block <= 0 or num_devices <= 0:
        return [[] for _ in range(max(0, num_devices))]
    blocks = [(start, min(seq_len, start + query_block))
              for start in range(0, seq_len, query_block)]
    assignments = [[] for _ in range(num_devices)]
    for block_index, block in enumerate(blocks):
        assignments[block_index % len(assignments)].append(block)
    return assignments


def _attention_mask_tile(attention_mask, q_start, q_end, k_start, k_end):
    """Take only one bounded mask tile; never materialize a Q×full-K slice."""
    if attention_mask is None:
        return None
    if attention_mask.dim() == 3:
        return attention_mask[:, q_start:q_end, k_start:k_end]
    if attention_mask.dim() == 4 and attention_mask.shape[1] == 1:
        return attention_mask[:, 0, q_start:q_end, k_start:k_end]
    raise ValueError(
        "GLM DSA indexer expects a [B,Q,K] or singleton-head "
        "[B,1,Q,K] attention mask"
    )


def _compute_query_block_topk(
    q_tile, k_full, w_tile, q_start, key_block, topk, position_tile,
    attention_mask, scale,
):
    """Compute one query block's exact running top-k on one device."""
    batch_size, query_len = q_tile.shape[:2]
    device = q_tile.device
    q_float = q_tile.float()
    w_float = w_tile.float()
    pos_tile = position_tile
    local_values = torch.full(
        (batch_size, query_len, topk), float("-inf"),
        dtype=torch.float32, device=device,
    )
    local_indices = torch.full(
        (batch_size, query_len, topk), -1,
        dtype=torch.int64, device=device,
    )
    total_keys = k_full.shape[1]
    for k_start in range(0, total_keys, key_block):
        k_end = min(total_keys, k_start + key_block)
        k_tile = k_full[:, k_start:k_end].float()
        scores = torch.matmul(
            q_float, k_tile.transpose(-1, -2).unsqueeze(1)
        ) * scale
        scores = torch.relu(scores)
        index_scores = torch.matmul(
            w_float.unsqueeze(-2), scores
        ).squeeze(-2)
        mask_tile = _attention_mask_tile(
            attention_mask, q_start, q_start + query_len, k_start, k_end
        )
        if mask_tile is not None:
            # The mask transfer, when needed, is bounded by q_block×k_block.
            if not _same_device(mask_tile.device, device):
                mask_tile = mask_tile.to(device, non_blocking=True)
            index_scores = index_scores + mask_tile
        else:
            key_positions = torch.arange(k_start, k_end, device=device)
            causal = key_positions[None, None, :] > pos_tile[:, :, None]
            index_scores = index_scores.masked_fill(causal, float("-inf"))
        candidates = torch.cat((local_values, index_scores), dim=-1)
        candidate_indices = torch.cat((
            local_indices,
            torch.arange(k_start, k_end, device=device)
            .view(1, 1, -1).expand(batch_size, query_len, -1),
        ), dim=-1)
        local_values, selected = torch.topk(
            candidates, topk, dim=-1, sorted=True
        )
        local_indices = candidate_indices.gather(-1, selected)
    valid = (
        (local_indices >= 0)
        & (local_indices <= position_tile[:, :, None])
    )
    return torch.where(valid, local_indices, -1)


def _tp_query_parallel_topk(
    q, k, weights, query_block, key_block, topk, position_ids,
    attention_mask, scale, devices, base_device,
):
    """Exact query-parallel indexer with one K replication per forward.

    Query tokens are independent for the indexer top-k.  Each helper therefore
    receives individual query blocks and the full K tensor exactly once;
    local running top-k is already the global result for those queries.  Q, K,
    weights and masks are transferred in their source dtype and promoted to
    FP32 only on the destination immediately before matmul.
    """
    seq_len = q.shape[1]
    batch_size = q.shape[0]
    assignments = _query_block_assignments(
        seq_len, query_block, len(devices)
    )
    active = [
        (device, blocks) for device, blocks in zip(devices, assignments)
        if blocks
    ]
    if not active:
        return torch.empty(
            batch_size, 0, topk, dtype=torch.int32, device=base_device
        )

    # K is the only large tensor replicated to every helper.  Keep the
    # original dtype during the transfer and reuse each copy for all Q blocks.
    t0 = time.perf_counter()
    k_by_device = []
    for device, _ in active:
        if _same_device(device, base_device):
            k_by_device.append((device, k))
        else:
            k_by_device.append((device, k.to(device, non_blocking=True)))
    k_replicate_host = time.perf_counter() - t0

    states_by_device = []
    t_dispatch = time.perf_counter()
    for (device, blocks), (_, k_dev) in zip(active, k_by_device):
        block_states = []
        for q_start, q_end in blocks:
            # Only this query block is dispatched.  Multiple blocks on one
            # device remain separate launches and never form a mega-shard.
            q_tile = q[:, q_start:q_end]
            w_tile = weights[:, q_start:q_end]
            pos_tile = position_ids[:, q_start:q_end]
            if not _same_device(device, base_device):
                q_tile = q_tile.to(device, non_blocking=True)
                w_tile = w_tile.to(device, non_blocking=True)
                pos_tile = pos_tile.to(device, non_blocking=True)
            local_indices = _compute_query_block_topk(
                q_tile, k_dev, w_tile, q_start, key_block, topk,
                pos_tile, attention_mask, scale,
            )
            block_states.append((q_start, q_end, local_indices))
        states_by_device.append((device, block_states))
    dispatch_compute_launch_host = time.perf_counter() - t_dispatch

    # All helper streams have now been launched.  One barrier per device is
    # sufficient; unlike K-parallel there is no per-query-block synchronize or
    # global candidate merge.
    t_sync = time.perf_counter()
    for device, _ in states_by_device:
        _synchronise(device)
    sync_host = time.perf_counter() - t_sync

    t_gather = time.perf_counter()
    # Concatenate each device's indices once, then perform one device→owner
    # copy per helper.  Reconstruct original query order from block offsets.
    gathered_blocks = []
    has_remote = False
    for device, block_states in states_by_device:
        local_indices = torch.cat(
            [indices for _, _, indices in block_states], dim=1
        ).to(torch.int32)
        if not _same_device(device, base_device):
            local_indices = local_indices.to(base_device, non_blocking=True)
            has_remote = True
        offset = 0
        for q_start, q_end, _ in block_states:
            q_len = q_end - q_start
            gathered_blocks.append((q_start, local_indices[:, offset:offset + q_len]))
            offset += q_len
    gathered_blocks.sort(key=lambda item: item[0])
    result = torch.cat([indices for _, indices in gathered_blocks], dim=1)
    # Copies above are queued after helper synchronization.  Wait once on the
    # owner before returning to the model, never once per query block.
    if has_remote:
        _synchronise(base_device)
    gather_host = time.perf_counter() - t_gather
    sync_waits = len(states_by_device) + (1 if has_remote else 0)
    logger.info(
        "[GLM DSA TP timing] strategy=query_parallel q=%d k=%d devices=%d "
        "query_blocks=%s "
        "K_replicate_launch=%.4fs Q_dispatch+compute_launch=%.4fs "
        "sync_wait=%.4fs (waits=%d) indices_gather=%.4fs total=%.4fs "
        "[host launch timings; device work included in sync_wait]",
        seq_len, k.shape[1], len(states_by_device),
        [blocks for _, blocks in active], k_replicate_host,
        dispatch_compute_launch_host, sync_host, sync_waits, gather_host,
        time.perf_counter() - t0,
    )
    return result


def _install_glm_dsa_blockwise_attention(
    attention_cls, rope, *, threshold, query_block, selected_block,
    parallel_mode, device_groups,
):
    """Install the memory-bounded eager/SDPA long-prefill attention path."""
    original = getattr(
        attention_cls, "_acc_blockwise_attention_original", None
    )
    if original is None:
        original = getattr(attention_cls, "forward", None)
    if original is None:
        return False

    @torch.no_grad()
    def blockwise_attention_forward(
        self, hidden_states, position_embeddings, attention_mask,
        past_key_values=None, position_ids=None, prev_topk_indices=None,
        **kwargs,
    ):
        batch_size, seq_len = hidden_states.shape[:-1]
        implementation = str(
            getattr(self.config, "_attn_implementation", "eager")
        ).lower()
        # Flash-MLA already consumes sparse indices without a dense QxK score
        # matrix.  Training keeps the original dropout/autograd behavior.
        if (
            seq_len <= threshold
            or implementation not in {"eager", "sdpa"}
            or self.training
            or position_ids is None
        ):
            return original(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                prev_topk_indices=prev_topk_indices,
                **kwargs,
            )

        started = time.perf_counter()
        query_shape = (batch_size, seq_len, -1, self.qk_head_dim)
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(
            q_states,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        k_rot = k_rot.view(
            batch_size, 1, seq_len, self.qk_rope_head_dim
        )
        cos, sin = position_embeddings
        q_rot, k_rot = rope(q_rot, k_rot, cos, sin)
        query_states = torch.cat((q_pass, q_rot), dim=-1)
        expand_kv = getattr(self, "expand_kv", None)
        if expand_kv is not None:
            # Newer generated Transformers layouts inherit this helper from
            # DeepseekV3Attention.
            k_pass = self.kv_a_layernorm(k_pass)
            key_states, value_states = expand_kv(k_pass, k_rot)
        else:
            # Transformers 5.12.0 keeps kv_b_proj + split inline in GLM's
            # forward and has no expand_kv method.
            key_shape = (
                batch_size,
                seq_len,
                -1,
                self.qk_nope_head_dim + self.v_head_dim,
            )
            k_pass = self.kv_b_proj(
                self.kv_a_layernorm(k_pass)
            ).view(key_shape).transpose(1, 2)
            k_pass, value_states = torch.split(
                k_pass,
                [self.qk_nope_head_dim, self.v_head_dim],
                dim=-1,
            )
            k_rot = k_rot.expand(*k_pass.shape[:-1], -1)
            key_states = torch.cat((k_pass, k_rot), dim=-1)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        if self.indexer is not None:
            indexer_mask = (
                attention_mask[:, 0, :, :]
                if attention_mask is not None and attention_mask.dim() == 4
                else attention_mask
            )
            topk_indices = self.indexer(
                hidden_states,
                q_resid,
                position_embeddings,
                indexer_mask,
                position_ids,
                past_key_values=past_key_values,
            )
        else:
            if prev_topk_indices is None:
                raise ValueError(
                    "Shared DSA layers require top-k indices from a previous "
                    "full indexer layer."
                )
            topk_indices = prev_topk_indices

        del q_states, q_pass, q_rot
        del compressed_kv, k_rot, k_pass, cos, sin, q_resid

        # Make the source layout stable once.  The selected-state helper then
        # uses a view for every query block rather than copying full K/V.
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        num_attention_heads = query_states.shape[1]
        tp_devices = []
        if parallel_mode == "tp":
            tp_devices = _select_tp_devices(
                query_states.device, device_groups
            )
            if len(tp_devices) < 2:
                tp_devices = []
        if tp_devices:
            attention_strategy = "query_parallel"
            sparse_output = _tp_sparse_query_parallel_attention(
                query_states,
                key_states,
                value_states,
                topk_indices,
                position_ids,
                attention_mask,
                self.scaling,
                query_block,
                selected_block,
                tp_devices,
                query_states.device,
            )
        else:
            attention_strategy = "owner_blockwise"
            sparse_output = torch.empty(
                batch_size,
                num_attention_heads,
                seq_len,
                value_states.shape[-1],
                dtype=query_states.dtype,
                device=query_states.device,
            )
            for q_start in range(0, seq_len, query_block):
                q_end = min(seq_len, q_start + query_block)
                sparse_output[:, :, q_start:q_end, :] = (
                    _compute_sparse_attention_query_block(
                        query_states[:, :, q_start:q_end, :],
                        key_states,
                        value_states,
                        topk_indices[:, q_start:q_end, :],
                        position_ids[:, q_start:q_end],
                        attention_mask,
                        q_start,
                        self.scaling,
                        selected_block,
                    )
                )

        attn_output = sparse_output.transpose(1, 2).contiguous()
        del sparse_output, query_states, key_states, value_states
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        logger.info(
            "[GLM DSA attention timing] strategy=%s "
            "layer=%s q=%d selected=%d query_block=%d selected_block=%d "
            "peak_score_shape=[%d,%d,%d,%d] total=%.4fs",
            attention_strategy,
            self.layer_idx,
            seq_len,
            topk_indices.shape[-1],
            query_block,
            selected_block,
            batch_size,
            num_attention_heads,
            min(query_block, seq_len),
            min(selected_block, topk_indices.shape[-1]),
            time.perf_counter() - started,
        )
        return attn_output, None, topk_indices

    attention_cls._acc_blockwise_attention_original = original
    attention_cls.forward = blockwise_attention_forward
    attention_cls._acc_blockwise_attention_installed = True
    return True


def install_glm_dsa_blockwise_indexer(
    *, query_block: int | None = None, key_block: int | None = None,
    threshold: int | None = None, parallel_mode: str = "pp",
    device_groups=None,
) -> bool:
    """Patch the HF GLM indexer; return whether the class was found."""
    try:
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
            GlmMoeDsaAttention,
            GlmMoeDsaIndexer,
        )
    except ImportError:
        return False

    original = getattr(GlmMoeDsaIndexer, "_acc_blockwise_original", None)
    if original is None:
        original = getattr(GlmMoeDsaIndexer, "forward", None)
    if original is None:
        return original is not None

    parallel_mode = str(parallel_mode or "pp").lower()
    if parallel_mode not in {"pp", "tp"}:
        raise ValueError("GLM prefill parallel_mode must be 'pp' or 'tp'")
    device_groups = _normalise_device_groups(device_groups)
    tp_strategy = os.getenv("ACC_GLM_DSA_TP_STRATEGY", "query").strip().lower()
    if tp_strategy in {"query_parallel", "query-parallel"}:
        tp_strategy = "query"
    if tp_strategy in {"key_parallel", "key-parallel"}:
        tp_strategy = "key"
    if tp_strategy not in {"query", "key"}:
        raise ValueError(
            "ACC_GLM_DSA_TP_STRATEGY must be 'query' (default) or 'key'"
        )
    tp_strategy_label = (
        "query_parallel" if tp_strategy == "query" else "key_parallel"
    )

    query_block = query_block or int(os.getenv("ACC_GLM_DSA_QUERY_BLOCK", "1024"))
    key_block = key_block or int(os.getenv("ACC_GLM_DSA_KEY_BLOCK", "4096"))
    threshold = threshold or int(os.getenv("ACC_GLM_DSA_BLOCKWISE_THRESHOLD", "16384"))
    attention_query_block = int(os.getenv(
        "ACC_GLM_DSA_ATTN_QUERY_BLOCK", "64"
    ))
    attention_selected_block = int(os.getenv(
        "ACC_GLM_DSA_ATTN_SELECTED_BLOCK", "512"
    ))
    query_block = max(1, query_block)
    key_block = max(1, key_block)
    threshold = max(1, threshold)
    attention_query_block = max(1, attention_query_block)
    attention_selected_block = max(1, attention_selected_block)
    globals_ = getattr(original, "__globals__", {})
    rope = globals_.get("apply_rotary_pos_emb_interleave")
    if rope is None:
        # Transformers releases generated from the modular GLM source may
        # resolve the helper through the DeepSeek-V3 module rather than keep
        # it in modeling_glm_moe_dsa.__globals__.  Import the canonical
        # implementation explicitly instead of silently falling back to the
        # eager indexer (which materializes the huge [Q,H,K] score tensor).
        try:
            from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
                apply_rotary_pos_emb_interleave,
            )
            rope = apply_rotary_pos_emb_interleave
        except (ImportError, AttributeError):
            logger.warning(
                "GLM DSA interleaved RoPE helper not found; blockwise disabled "
                "(eager indexer may OOM on long prompts)"
            )
            return False

    @torch.no_grad()
    def blockwise_forward(self, hidden_states, q_resid, position_embeddings,
                          attention_mask, position_ids, past_key_values=None):
        batch_size, seq_len, _ = hidden_states.shape
        if seq_len <= threshold:
            return original(self, hidden_states, q_resid, position_embeddings,
                            attention_mask, position_ids, past_key_values)

        cos, sin = position_embeddings
        q = self.wq_b(q_resid).view(
            batch_size, seq_len, self.n_heads, self.head_dim)
        q_rot, q_pass = torch.split(
            q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
        k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)
        k_rot, k_pass = torch.split(
            k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
        q_rot, k_rot = rope(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)
        if past_key_values is not None:
            k = past_key_values.update_indexer(k, self.layer_idx)

        # Keep transport dtype intact.  Both TP implementations promote on
        # the destination immediately before the score matmul.
        weights = self.weights_proj(
            hidden_states.to(self.weights_proj.weight.dtype))
        weights = weights * (self.n_heads ** -0.5)
        total_keys = k.shape[1]
        topk = min(self.index_topk, total_keys)
        outputs = []
        scale = self.softmax_scale
        tp_devices = []
        if parallel_mode == "tp":
            tp_devices = _select_tp_devices(q.device, device_groups)
            # A group with one device is equivalent to the old PP path.  Do
            # not pay the extra copies/merge cost in that case.
            if len(tp_devices) < 2:
                tp_devices = []
        if tp_devices:
            logger.debug(
                "GLM DSA TP indexer layer=%s strategy=%s source=%s devices=%s",
                self.layer_idx,
                "query_parallel" if tp_strategy == "query" else "key_parallel",
                q.device, tp_devices)
        if tp_devices and tp_strategy == "query":
            return _tp_query_parallel_topk(
                q, k, weights, query_block, key_block, topk, position_ids,
                attention_mask, scale, tp_devices, q.device,
            )
        for q_start in range(0, seq_len, query_block):
            q_end = min(seq_len, q_start + query_block)
            q_tile = q[:, q_start:q_end].float()
            w_tile = weights[:, q_start:q_end].float()
            if tp_devices:
                outputs.append(_tp_indexer_topk(
                    q_tile, k, w_tile, q_start, key_block, topk,
                    position_ids, attention_mask, scale, tp_devices, q.device,
                ))
                del q_tile, w_tile
                continue
            running_values = torch.full(
                (batch_size, q_end - q_start, topk), float("-inf"),
                dtype=torch.float32, device=q.device)
            running_indices = torch.full(
                (batch_size, q_end - q_start, topk), -1,
                dtype=torch.int64, device=q.device)
            for k_start in range(0, total_keys, key_block):
                k_end = min(total_keys, k_start + key_block)
                scores = torch.matmul(
                    q_tile, k[:, k_start:k_end].transpose(-1, -2).float()
                    .unsqueeze(1)) * scale
                scores = torch.relu(scores)
                index_scores = torch.matmul(
                    w_tile.unsqueeze(-2), scores).squeeze(-2)
                if attention_mask is not None:
                    index_scores = index_scores + attention_mask[
                        :, q_start:q_end, k_start:k_end]
                else:
                    key_positions = torch.arange(
                        k_start, k_end, device=q.device)
                    causal = key_positions[None, None, :] > position_ids[
                        :, q_start:q_end, None]
                    index_scores = index_scores.masked_fill(causal, float("-inf"))
                candidates = torch.cat((running_values, index_scores), dim=-1)
                candidate_indices = torch.cat(
                    (running_indices,
                     torch.arange(k_start, k_end, device=q.device)
                     .view(1, 1, -1).expand(batch_size, q_end - q_start, -1)),
                    dim=-1)
                running_values, selected = torch.topk(
                    candidates, topk, dim=-1, sorted=True)
                running_indices = candidate_indices.gather(-1, selected)
                del scores, index_scores, candidates, candidate_indices
            query_positions = position_ids[:, q_start:q_end]
            valid = (
                (running_indices >= 0)
                & (running_indices <= query_positions[:, :, None])
            )
            outputs.append(torch.where(
                valid, running_indices, -1
            ).to(torch.int32))
            del running_values, running_indices, q_tile, w_tile
        return torch.cat(outputs, dim=1)

    GlmMoeDsaIndexer._acc_blockwise_original = original
    GlmMoeDsaIndexer.forward = blockwise_forward
    GlmMoeDsaIndexer._acc_blockwise_installed = True
    attention_ok = _install_glm_dsa_blockwise_attention(
        GlmMoeDsaAttention,
        rope,
        threshold=threshold,
        query_block=attention_query_block,
        selected_block=attention_selected_block,
        parallel_mode=parallel_mode,
        device_groups=device_groups,
    )
    if not attention_ok:
        logger.warning(
            "GLM DSA blockwise indexer installed, but attention patch failed; "
            "long eager attention may still allocate a dense QxK tensor"
        )
        return False
    logger.info(
        "  GLM DSA blockwise runtime 已安装: mode=%s, threshold=%d, "
        "index_query_block=%d, index_key_block=%d, strategy=%s, "
        "attn_strategy=%s, attn_query_block=%d, attn_selected_block=%d, "
        "tp_groups=%s",
        parallel_mode, threshold, query_block, key_block, tp_strategy_label,
        "query_parallel" if parallel_mode == "tp" else "owner_blockwise",
        attention_query_block, attention_selected_block,
        device_groups if parallel_mode == "tp" else "disabled")
    return True
