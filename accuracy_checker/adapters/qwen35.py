"""
Qwen3.5 MoE 模型 Adapter

实现 Qwen3.5 MoE 系列的模型结构映射。

支持两种模型类:
  - Qwen3_5MoeForConditionalGeneration (有 language_model 中间层)
  - Qwen3_5MoeForCausalLM (无 language_model 中间层)

自动探测模型结构，优先 language_model 路径。
"""

import torch.nn as nn
from typing import Optional
from .base import BaseModelAdapter


class Qwen35MoEAdapter(BaseModelAdapter):
    """
    Qwen3.5 MoE 模型适配器

    自动探测:
      - ForConditionalGeneration: model.model.language_model.layers
      - ForCausalLM: model.model.layers
    """

    def _get_language_model(self):
        """获取 language_model 子模块 (ForConditionalGeneration 格式)"""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'language_model'):
            return self.model.model.language_model
        return None

    def _get_inner_model(self):
        """获取内部 model 子模块 (统一入口)"""
        lm = self._get_language_model()
        if lm is not None:
            return lm
        return self.model.model

    # ========================================================================
    # 1. 层级访问
    # ========================================================================

    def get_layers(self) -> nn.ModuleList:
        return self._get_inner_model().layers

    def get_num_layers(self) -> int:
        return len(self._get_inner_model().layers)

    def get_layer(self, idx: int) -> nn.Module:
        return self._get_inner_model().layers[idx]

    # ========================================================================
    # 2. 模块访问
    # ========================================================================

    def get_input_norm(self, layer: nn.Module) -> nn.Module:
        return layer.input_layernorm

    def get_post_attn_norm(self, layer: nn.Module) -> nn.Module:
        return layer.post_attention_layernorm

    def get_attention(self, layer: nn.Module) -> nn.Module:
        if hasattr(layer, 'self_attn'):
            return layer.self_attn
        if hasattr(layer, 'linear_attn'):
            return layer.linear_attn
        raise AttributeError(f"Layer {type(layer).__name__} has neither self_attn nor linear_attn")

    def get_mlp(self, layer: nn.Module) -> nn.Module:
        return layer.mlp

    # ========================================================================
    # 3. Projection 访问
    # ========================================================================

    def get_q_proj(self, attn: nn.Module) -> Optional[nn.Module]:
        return getattr(attn, 'q_proj', None)

    def get_k_proj(self, attn: nn.Module) -> Optional[nn.Module]:
        return getattr(attn, 'k_proj', None)

    def get_v_proj(self, attn: nn.Module) -> Optional[nn.Module]:
        return getattr(attn, 'v_proj', None)

    def get_o_proj(self, attn: nn.Module) -> Optional[nn.Module]:
        return getattr(attn, 'o_proj', None)

    def is_linear_attention_layer(self, layer: nn.Module) -> bool:
        return hasattr(layer, 'linear_attn')

    # ========================================================================
    # 4. 特殊层访问
    # ========================================================================

    def get_embed_layer(self) -> nn.Module:
        return self._get_inner_model().embed_tokens

    def get_final_norm(self) -> nn.Module:
        return self._get_inner_model().norm

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

        ForConditionalGeneration: quant_desc 用 model.language_model.layers
                                   模型 module name 是 model.model.language_model.layers
        ForCausalLM: 两者都是 model.layers
        """
        if self._get_language_model() is not None:
            return "model."
        return ""


__all__ = ["Qwen35MoEAdapter"]
