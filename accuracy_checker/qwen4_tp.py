"""Exact QSA index selection on query-parallel devices (not whole-model TP).

Only causal + key-padding masks use the fast path. Their visible keys form
prefixes of one ordered list, so complete compressed blocks can be pooled,
normalized and rotated once, instead of once per query. Each query still uses
the upstream full score vector/topk (including its tie behavior), not a
different blockwise candidate merge. Main attention, GDN and decode are intact.
"""
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import logging
import math
import time

import torch

from .glm_dsa_blockwise import (
    _normalise_device_groups, _query_block_assignments, _select_tp_devices,
    _synchronise,
)

logger = logging.getLogger(__name__)
_options = {"query_block": 64, "min_tokens": 1024}
_runtime = {"mode": "pp", "groups": []}


def configure_options(query_block=None, min_tokens=None):
    for name, value in (("query_block", query_block), ("min_tokens", min_tokens)):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Qwen4 {name} must be a positive integer")
            _options[name] = value
    return dict(_options)


def _causal_key_lists(mask, batch, queries, query_block):
    """Validate compactability without copying a [B,Q,K] mask to helpers.

    A single scalar sync checks all owner-side tiles. Non-prefix/custom masks
    fall back before projections/cache updates, so replay state is untouched.
    """
    if mask is None or mask.ndim != 4 or tuple(mask.shape[:3]) != (batch, 1, queries):
        return None
    keys = mask.shape[-1]
    if keys < queries:
        return None
    valid = mask[:, 0, -1] if mask.dtype == torch.bool else mask[:, 0, -1] == 0
    key_positions = torch.arange(keys, device=mask.device)
    matches = torch.ones((), dtype=torch.bool, device=mask.device)
    for start in range(0, queries, query_block):
        end = min(start + query_block, queries)
        positions = torch.arange(keys - queries + start, keys - queries + end, device=mask.device)
        expected = valid[:, None, :] & (key_positions[None, :] <= positions[:, None])
        tile = mask[:, 0, start:end]
        actual = tile if tile.dtype == torch.bool else tile == 0
        matches = matches & (actual == expected).all()
    if not bool(matches.item()):
        return None
    # This [B,K] metadata is the only mask-related device-to-host copy.
    return [row.nonzero().flatten().tolist() for row in valid.detach().cpu()]


def _compressed_keys(indexer, raw_keys, positions, key_lists, apply_rope):
    keys_by_batch = []
    ratio = indexer.compress_ratio
    for b, visible in enumerate(key_lists):
        groups = len(visible) // ratio
        ids = torch.tensor(visible[:groups * ratio], device=raw_keys.device, dtype=torch.long)
        if not groups:
            keys_by_batch.append(raw_keys.new_empty((0, indexer.index_head_dim)))
            continue
        blocks = ids.view(groups, ratio)
        pooled = raw_keys[b].index_select(0, ids).view(groups, ratio, indexer.index_head_dim)
        pooled = indexer.k_layernorm(pooled.float().mean(dim=1).to(raw_keys.dtype))
        starts = blocks[:, 0]
        rotated = apply_rope(pooled.unsqueeze(1),
            cos=positions[0][b].index_select(0, starts),
            sin=positions[1][b].index_select(0, starts)).squeeze(1)
        keys_by_batch.append(rotated)
    return keys_by_batch


def _copy_to(tensor, device):
    """Small seam for testing remote transport without an NPU in CI."""
    return tensor.to(device=device, non_blocking=True)


def _worker_context(device):
    dev = torch.device(device)
    api = getattr(torch, dev.type, None)
    if dev.type in {"cuda", "npu"}:
        api.set_device(dev)
        # One stream per active worker, independent of the launching thread.
        return api.stream(api.Stream(device=dev))
    return nullcontext()


def _query_worker(device, assignments, q, compressed, key_lists, key_offset, ratio, budget):
    with torch.no_grad(), _worker_context(device):
        # Transport original dtype. FP32 casting occurs on the target device.
        local_keys = [_copy_to(k, device).float() for k in compressed]
        local_ids = [_copy_to(torch.tensor(ids, dtype=torch.int32), device) for ids in key_lists]
        chunks = []
        for start, end in assignments:
            tile = _copy_to(q[:, start:end], device).float()
            indices = torch.full((q.shape[0], end - start, budget + ratio - 1),
                                 -1, dtype=torch.int32, device=device)
            for b, visible in enumerate(key_lists):
                for relative, query in enumerate(range(start, end)):
                    count = bisect_right(visible, key_offset + query)
                    complete = count // ratio
                    if complete:
                        # Same matmul shape, reduction order and topk width as
                        # upstream: no cross-device candidate merge or tie rule.
                        scores = (tile[b, relative] @ local_keys[b][:complete].T).T
                        scores = torch.relu(scores).sum(-1) / math.sqrt(q.shape[-1])
                        chosen = scores.topk(min(budget // ratio, complete), dim=0).indices
                        selected = local_ids[b][:complete * ratio].view(complete, ratio).index_select(0, chosen).flatten()
                    else:
                        selected = local_ids[b][:0]
                    selected = torch.cat((selected, local_ids[b][complete * ratio:count]))
                    indices[b, relative, :selected.numel()] = selected
            chunks.append((start, end, indices))
        # No per-query/per-block barrier. Device duration is not host launch time.
        _synchronise(device)
        return chunks


def _query_parallel_indices(q, compressed, key_lists, total_keys, ratio, budget, devices, query_block):
    assignments = _query_block_assignments(q.shape[1], query_block, len(devices))
    active = [(device, blocks) for device, blocks in zip(devices, assignments) if blocks]
    # Finish owner projection/normalization before independent streams consume it.
    _synchronise(q.device)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(active), thread_name_prefix="qwen4-qsa") as pool:
        futures = [pool.submit(_query_worker, device, blocks, q, compressed, key_lists,
                               total_keys - q.shape[1], ratio, budget) for device, blocks in active]
        pieces = [piece for future in futures for piece in future.result()]
    workers_finished = time.perf_counter()
    result = torch.full((q.shape[0], q.shape[1], budget + ratio - 1), -1,
                        dtype=torch.int32, device=q.device)
    for start, end, indices in pieces:
        result[:, start:end] = _copy_to(indices, q.device)
    _synchronise(q.device)
    logger.info("  [Qwen4 QSA TP] strategy=query_parallel, devices=%s, assignment=round_robin, "
                "blocks=%s, query_block=%d; worker_wall=%.3fs (copies+compute+wait), indices_gather=%.3fs",
                [str(d) for d, _ in active], [len(blocks) for _, blocks in active], query_block,
                workers_finished - started, time.perf_counter() - workers_finished)
    return result


def _indices_to_mask(indices, attention_mask):
    keys = attention_mask.shape[-1]
    selected = torch.zeros((*indices.shape[:-1], keys + 1), device=indices.device, dtype=torch.bool)
    selected.scatter_(-1, torch.where(indices >= 0, indices, keys), True)
    selected = selected[..., :keys].unsqueeze(1)
    if attention_mask.is_floating_point():
        selected = torch.where(selected, attention_mask.new_zeros(()), torch.finfo(attention_mask.dtype).min)
    return selected


def install_query_parallel(mode="pp", device_groups=None, query_block=None, min_tokens=None):
    """Install an idempotent Qwen-only wrapper. PP/single-card/decode stay HF."""
    if mode not in {"pp", "tp"}:
        raise ValueError("Qwen4 parallel mode must be pp or tp")
    configure_options(query_block, min_tokens)
    from transformers.models.qwen4_exp import modeling_qwen4_exp as hf
    cls = hf.Qwen4ExpTextQSAIndexer
    _runtime.update(mode=mode, groups=_normalise_device_groups(device_groups))
    if hasattr(cls, "_acc_query_parallel_original"):
        return
    original = cls.forward
    cls._acc_query_parallel_original = original

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
        options = dict(_options)
        devices = _select_tp_devices(hidden_states.device, _runtime["groups"])
        batch, queries = hidden_states.shape[:2]
        if (_runtime["mode"] != "tp" or queries <= 1 or queries < options["min_tokens"]
                or len(devices) < 2 or self.training):
            return original(self, hidden_states, position_embeddings, attention_mask, past_key_values)
        started = time.perf_counter()
        key_lists = _causal_key_lists(attention_mask, batch, queries, options["query_block"])
        if key_lists is None:
            logger.warning("  [Qwen4 QSA TP] custom/non-prefix mask: using official owner path")
            return original(self, hidden_states, position_embeddings, attention_mask, past_key_values)
        keys = attention_mask.shape[-1]
        if any(x.shape[0] != batch or x.shape[1] != keys for x in position_embeddings):
            return original(self, hidden_states, position_embeddings, attention_mask, past_key_values)
        qk = self.index_qk_proj(hidden_states)
        q, raw_keys = torch.split(qk, [self.index_n_heads * self.index_head_dim,
                                      self.index_kv_heads * self.index_head_dim], dim=-1)
        q = self.q_layernorm(q.reshape(batch, queries, -1, self.index_head_dim))
        raw_keys = raw_keys.reshape(batch, queries, self.index_head_dim)
        q = hf.apply_rotary_pos_emb(q, cos=position_embeddings[0][:, -queries:],
                                   sin=position_embeddings[1][:, -queries:], unsqueeze_dim=2)
        if past_key_values is not None:
            raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)
        if raw_keys.shape[1] != keys:
            raise ValueError("Qwen4 QSA cached keys and attention mask disagree")
        compressed = _compressed_keys(self, raw_keys, position_embeddings, key_lists, hf.apply_rotary_pos_emb)
        indices = _query_parallel_indices(q, compressed, key_lists, keys, self.compress_ratio,
                                         self.token_budget, devices, options["query_block"])
        result = _indices_to_mask(indices, attention_mask)
        _synchronise(hidden_states.device)
        logger.info("  [Qwen4 QSA TP] indexer total=%.3fs; pooled K reused once/helper; "
                    "main attention/GDN/decode unchanged", time.perf_counter() - started)
        return result

    cls.forward = forward
