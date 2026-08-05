"""
Qwen3 模型 Adapter

实现 Qwen3/ Qwen2 系列的模型结构映射。
"""

import torch.nn as nn
from typing import Optional
from .base import BaseModelAdapter


class Qwen3Adapter(BaseModelAdapter):
    """
    Qwen3 模型适配器

    模型结构:
        model
        ├── model (Transformer)
        │   ├── layers (ModuleList)
        │   │   └── [i] (DecoderLayer)
        │   │       ├── input_layernorm
        │   │       ├── self_attn
        │   │       │   ├── q_proj
        │   │       │   ├── k_proj
        │   │       │   ├── v_proj
        │   │       │   └── o_proj
        │   │       ├── post_attention_layernorm
        │   │       └── mlp
        │   │           ├── gate_proj
        │   │           ├── up_proj
        │   │           └── down_proj
        │   ├── embed_tokens
        │   └── norm
        └── lm_head
    """

    # ========================================================================
    # 1. 层级访问
    # ========================================================================

    def get_layers(self) -> nn.ModuleList:
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
    # 6. MLP 子模块访问 (Qwen3 特有)
    # ========================================================================

    def get_mlp_gate_proj(self, mlp: nn.Module) -> nn.Module:
        return mlp.gate_proj

    def get_mlp_up_proj(self, mlp: nn.Module) -> nn.Module:
        return mlp.up_proj

    def get_mlp_down_proj(self, mlp: nn.Module) -> nn.Module:
        return mlp.down_proj
