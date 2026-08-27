"""Memory-bounded GLM-MoE-DSA indexer for very long prefill inputs.

Transformers' eager indexer materializes ``[batch, query, heads, keys]``
scores.  For a 65k-token prompt that intermediate is hundreds of GiB.  This
patch preserves the same projections, ReLU, causal mask and global top-k, but
processes query/key tiles and keeps only the running top-k values.
"""

from __future__ import annotations

import logging
import os
import time

import torch

logger = logging.getLogger(__name__)


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
    return running_indices


def _query_shard_ranges(seq_len, query_block, num_devices):
    """Partition complete query blocks into contiguous, balanced shards."""
    blocks = [(start, min(seq_len, start + query_block))
              for start in range(0, seq_len, query_block)]
    if not blocks:
        return []
    num_shards = min(len(blocks), max(1, num_devices))
    base, remainder = divmod(len(blocks), num_shards)
    ranges = []
    cursor = 0
    for shard in range(num_shards):
        count = base + (1 if shard < remainder else 0)
        start = blocks[cursor][0]
        end = blocks[cursor + count - 1][1]
        ranges.append((start, end))
        cursor += count
    return ranges


def _tp_query_parallel_topk(
    q, k, weights, query_block, key_block, topk, position_ids,
    attention_mask, scale, devices, base_device,
):
    """Exact query-parallel indexer with one K replication per forward.

    Query tokens are independent for the indexer top-k.  Each helper therefore
    receives one contiguous query shard and the full K tensor exactly once;
    local running top-k is already the global result for those queries.  Q, K,
    weights and masks are transferred in their source dtype and promoted to
    FP32 only on the destination immediately before matmul.
    """
    seq_len = q.shape[1]
    batch_size = q.shape[0]
    ranges = _query_shard_ranges(seq_len, query_block, len(devices))
    if not ranges:
        return torch.empty(
            batch_size, 0, topk, dtype=torch.int32, device=base_device
        )

    # K is the only large tensor replicated to every helper.  Keep the
    # original dtype during the transfer and reuse each copy for all Q blocks.
    t0 = time.perf_counter()
    k_by_device = []
    for device in devices[:len(ranges)]:
        if _same_device(device, base_device):
            k_by_device.append(k)
        else:
            k_by_device.append(k.to(device, non_blocking=True))
    k_replicate_host = time.perf_counter() - t0

    states = []
    t_dispatch = time.perf_counter()
    for (q_start, q_end), device, k_dev in zip(ranges, devices, k_by_device):
        q_shard = q[:, q_start:q_end]
        w_shard = weights[:, q_start:q_end]
        pos_shard = position_ids[:, q_start:q_end]
        mask_shard = None
        if attention_mask is not None:
            # The indexer mask is normally [B,Q,K].  Accept a 4-D mask too so
            # this helper remains usable with newer Transformers releases.
            if attention_mask.dim() == 3:
                mask_shard = attention_mask[:, q_start:q_end, :]
            else:
                # Indexer scores have no head axis; current HF masks use a
                # singleton axis here.  Drop it before adding to [B,Q,K].
                mask_shard = attention_mask[:, 0, q_start:q_end, :]
        if not _same_device(device, base_device):
            q_shard = q_shard.to(device, non_blocking=True)
            w_shard = w_shard.to(device, non_blocking=True)
            pos_shard = pos_shard.to(device, non_blocking=True)
            if mask_shard is not None:
                mask_shard = mask_shard.to(device, non_blocking=True)

        # Preserve the same key-block running-topk order as PP.  The local
        # result is exact because this helper owns every key for its queries.
        q_float = q_shard.float()
        w_float = w_shard.float()
        local_q_len = q_end - q_start
        local_values = torch.full(
            (batch_size, local_q_len, topk), float("-inf"),
            dtype=torch.float32, device=device,
        )
        local_indices = torch.full(
            (batch_size, local_q_len, topk), -1,
            dtype=torch.int64, device=device,
        )
        for k_start in range(0, k.shape[1], key_block):
            k_end = min(k.shape[1], k_start + key_block)
            k_tile = k_dev[:, k_start:k_end].float()
            scores = torch.matmul(
                q_float, k_tile.transpose(-1, -2).unsqueeze(1)
            ) * scale
            scores = torch.relu(scores)
            index_scores = torch.matmul(
                w_float.unsqueeze(-2), scores
            ).squeeze(-2)
            if mask_shard is not None:
                index_scores = index_scores + mask_shard[..., k_start:k_end]
            else:
                key_positions = torch.arange(k_start, k_end, device=device)
                causal = key_positions[None, None, :] > pos_shard[:, :, None]
                index_scores = index_scores.masked_fill(causal, float("-inf"))
            candidates = torch.cat((local_values, index_scores), dim=-1)
            candidate_indices = torch.cat((
                local_indices,
                torch.arange(k_start, k_end, device=device)
                .view(1, 1, -1).expand(batch_size, local_q_len, -1),
            ), dim=-1)
            local_values, selected = torch.topk(
                candidates, topk, dim=-1, sorted=True
            )
            local_indices = candidate_indices.gather(-1, selected)

        states.append((q_start, q_end, device, local_indices))
    dispatch_compute_launch_host = time.perf_counter() - t_dispatch

    # All helper streams have now been launched.  One barrier per device is
    # sufficient; unlike K-parallel there is no per-query-block synchronize or
    # global candidate merge.
    t_sync = time.perf_counter()
    for _, _, device, _ in states:
        _synchronise(device)
    sync_host = time.perf_counter() - t_sync

    t_gather = time.perf_counter()
    ordered = []
    for q_start, q_end, device, indices in states:
        if not _same_device(device, base_device):
            indices = indices.to(base_device, non_blocking=True)
        ordered.append(indices.to(torch.int32))
    result = torch.cat(ordered, dim=1)
    # Copies above are queued after the synchronization, so synchronize the
    # owner only when a remote gather was used before returning to the model.
    if any(not _same_device(device, base_device) for _, _, device, _ in states):
        _synchronise(base_device)
    gather_host = time.perf_counter() - t_gather
    logger.info(
        "[GLM DSA TP timing] strategy=query_parallel q=%d k=%d devices=%d "
        "query_ranges=%s "
        "K_replicate_launch=%.4fs Q_dispatch+compute_launch=%.4fs "
        "sync=%.4fs indices_gather=%.4fs total=%.4fs",
        seq_len, k.shape[1], len(states), ranges, k_replicate_host,
        dispatch_compute_launch_host, sync_host, gather_host,
        time.perf_counter() - t0,
    )
    return result


def install_glm_dsa_blockwise_indexer(
    *, query_block: int | None = None, key_block: int | None = None,
    threshold: int | None = None, parallel_mode: str = "pp",
    device_groups=None,
) -> bool:
    """Patch the HF GLM indexer; return whether the class was found."""
    try:
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaIndexer
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
    query_block = max(1, query_block)
    key_block = max(1, key_block)
    threshold = max(1, threshold)
    globals_ = getattr(original, "__globals__", {})
    rope = globals_.get("apply_rotary_pos_emb_interleave")
    if rope is None:
        logger.warning("GLM DSA interleaved RoPE helper not found; blockwise disabled")
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
            outputs.append(running_indices.to(torch.int32))
            del running_values, running_indices, q_tile, w_tile
        return torch.cat(outputs, dim=1)

    GlmMoeDsaIndexer._acc_blockwise_original = original
    GlmMoeDsaIndexer.forward = blockwise_forward
    GlmMoeDsaIndexer._acc_blockwise_installed = True
    logger.info(
        "  GLM DSA blockwise indexer 已安装: mode=%s, threshold=%d, "
        "query_block=%d, key_block=%d, strategy=%s, "
        "tp_groups=%s",
        parallel_mode, threshold, query_block, key_block, tp_strategy_label,
        device_groups if parallel_mode == "tp" else "disabled")
    return True
