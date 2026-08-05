"""
Qwen3-VL 模型 Adapter

实现 Qwen3-VL 系列的模型结构映射。
"""

import torch.nn as nn
from typing import Optional
from .base import BaseModelAdapter


class Qwen3VLAdapter(BaseModelAdapter):
    """
    Qwen3-VL 模型适配器

    模型结构 (与 Qwen3 相同，只是模型类型不同):
        model
        ├── model (Qwen3VLModel)
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
        # VL 模型: layers 在 language_model 中
        return self.model.model.language_model.layers

    def get_num_layers(self) -> int:
        return len(self.model.model.language_model.layers)

    def get_layer(self, idx: int) -> nn.Module:
        return self.model.model.language_model.layers[idx]

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
    # 3. 参数访问
    # ========================================================================

    def get_q_proj(self, attn: nn.Module) -> nn.Module:
        return attn.q_proj

    def get_k_proj(self, attn: nn.Module) -> nn.Module:
        return attn.k_proj

    def get_v_proj(self, attn: nn.Module) -> nn.Module:
        return attn.v_proj

    def get_o_proj(self, attn: nn.Module) -> nn.Module:
        return attn.o_proj

    def get_gate_proj(self, mlp: nn.Module) -> nn.Module:
        return mlp.gate_proj

    def get_up_proj(self, mlp: nn.Module) -> nn.Module:
        return mlp.up_proj

    def get_down_proj(self, mlp: nn.Module) -> nn.Module:
        return mlp.down_proj

    # ========================================================================
    # 4. 其它模块
    # ========================================================================

    def get_embed_tokens(self) -> nn.Module:
        # VL 模型: embed_tokens 在 language_model 中
        return self.model.model.language_model.embed_tokens

    def get_output_norm(self) -> nn.Module:
        # VL 模型: norm 在 language_model 中
        return self.model.model.language_model.norm

    def get_final_norm(self) -> nn.Module:
        return self.model.model.language_model.norm

    def get_embed_layer(self) -> nn.Module:
        return self.model.model.language_model.embed_tokens

    def get_lm_head(self) -> nn.Module:
        return self.model.lm_head

    # ========================================================================
    # 5. 能力声明
    # ========================================================================

    def supports_pre_rope_qk(self) -> bool:
        return True

    def supports_mlp_detail(self) -> bool:
        return True


__all__ = ["Qwen3VLAdapter"]
