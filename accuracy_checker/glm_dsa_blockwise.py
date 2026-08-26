"""Memory-bounded GLM-MoE-DSA indexer for very long prefill inputs.

Transformers' eager indexer materializes ``[batch, query, heads, keys]``
scores.  For a 65k-token prompt that intermediate is hundreds of GiB.  This
patch preserves the same projections, ReLU, causal mask and global top-k, but
processes query/key tiles and keeps only the running top-k values.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


def install_glm_dsa_blockwise_indexer(
    *, query_block: int | None = None, key_block: int | None = None,
    threshold: int | None = None,
) -> bool:
    """Patch the HF GLM indexer; return whether the class was found."""
    try:
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaIndexer
    except ImportError:
        return False

    original = getattr(GlmMoeDsaIndexer, "forward", None)
    if original is None or getattr(GlmMoeDsaIndexer, "_acc_blockwise_installed", False):
        return original is not None

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

        weights = self.weights_proj(
            hidden_states.to(self.weights_proj.weight.dtype)).float()
        weights = weights * (self.n_heads ** -0.5)
        total_keys = k.shape[1]
        topk = min(self.index_topk, total_keys)
        outputs = []
        scale = self.softmax_scale
        for q_start in range(0, seq_len, query_block):
            q_end = min(seq_len, q_start + query_block)
            q_tile = q[:, q_start:q_end].float()
            w_tile = weights[:, q_start:q_end]
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

    GlmMoeDsaIndexer.forward = blockwise_forward
    GlmMoeDsaIndexer._acc_blockwise_installed = True
    logger.info(
        "  GLM DSA blockwise indexer 已安装: threshold=%d, query_block=%d, key_block=%d",
        threshold, query_block, key_block)
    return True
