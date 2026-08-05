"""
Qwen3 MoE 模型 Adapter

实现 Qwen3 MoE (Qwen3-30B-A3B) 系列的模型结构映射。

模型结构:
    model (Qwen3MoeForCausalLM)
    └── model (Qwen3MoeModel)
        ├── layers (ModuleList)
        │   └── [i] (Qwen3MoeDecoderLayer)
        │       ├── input_layernorm
        │       ├── self_attn (Qwen3MoeAttention)
        │       │   ├── q_proj
        │       │   ├── k_proj
        │       │   ├── v_proj
        │       │   ├── o_proj
        │       │   ├── q_norm
        │       │   └── k_norm
        │       ├── post_attention_layernorm
        │       └── mlp (Qwen3MoeSparseMoeBlock)
        │           ├── gate
        │           └── experts (Qwen3MoeExperts)
        │               └── gate_up_proj: [num_experts, 2*intermediate_size, hidden_size]
        ├── embed_tokens
        └── norm
"""

import torch.nn as nn
from typing import Optional
from .base import BaseModelAdapter


class Qwen3MoEAdapter(BaseModelAdapter):
    """
    Qwen3 MoE 模型适配器

    适用于 Qwen3-30B-A3B-Instruct-2507-w8a8c8 等模型
    """

    # ========================================================================
    # 1. 层级访问
    # ========================================================================

    def get_layers(self) -> nn.ModuleList:
        # Qwen3 MoE: layers 在 model.model 中
        return self.model.model.layers

    def get_num_layers(self) -> int:
        return len(self.model.model.layers)

    def get_layer(self, idx: int) -> nn.Module:
        return self.model.model.layers[idx]

    # ========================================================================
    # 2. 模块访问
    # ========================================================================

    def get_input_norm(self, layer: nn.Module) -> nn.Module:
        return layer.input_layernorm

    def get_post_attn_norm(self, layer: nn.Module) -> nn.Module:
        return layer.post_attention_layernorm

    def get_attention(self, layer: nn.Module) -> nn.Module:
        return layer.self_attn

    def get_mlp(self, layer: nn.Module) -> nn.Module:
        return layer.mlp

    # ========================================================================
    # 3. Projection 访问
    # ========================================================================

    def get_q_proj(self, attn: nn.Module) -> nn.Module:
        return attn.q_proj

    def get_k_proj(self, attn: nn.Module) -> nn.Module:
        return attn.k_proj

    def get_v_proj(self, attn: nn.Module) -> nn.Module:
        return attn.v_proj

    def get_o_proj(self, attn: nn.Module) -> nn.Module:
        return attn.o_proj

    # ========================================================================
    # 4. 特殊层访问
    # ========================================================================

    def get_embed_layer(self) -> nn.Module:
        return self.model.model.embed_tokens

    def get_final_norm(self) -> nn.Module:
        return self.model.model.norm

    def get_lm_head(self) -> nn.Module:
        return self.model.lm_head

    # ========================================================================
    # 5. Hook 点定义 (key 名称)
    # ========================================================================

    def get_block_output_name(self, layer_idx: int) -> str:
        return f"layer.{layer_idx}.block_output"

    def get_pre_rope_q_name(self, layer_idx: int) -> str:
        return f"layer.{layer_idx}.pre_rope.q_proj"

    def get_pre_rope_k_name(self, layer_idx: int) -> str:
        return f"layer.{layer_idx}.pre_rope.k_proj"

    def get_embedding_name(self) -> str:
        return "embedding"

    def get_final_norm_name(self) -> str:
        return "final_norm"

    def get_logits_name(self) -> str:
        return "logits"

    # ========================================================================
    # 6. 参数名映射 (关键！)
    # ========================================================================

    def get_quant_key_prefix(self) -> str:
        """
        返回 quant_desc 中权重 key 的前缀
        Qwen3 MoE: quant_desc 用 model.layers，模型用 model.model.layers
        直接返回空字符串，不做映射（两边都是 model.layers 开头）
        """
        return ""


__all__ = ["Qwen3MoEAdapter"]
