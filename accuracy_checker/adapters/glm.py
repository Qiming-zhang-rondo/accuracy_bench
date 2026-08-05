"""
GLM 模型 Adapter

实现 GLM 系列的模型结构映射。
- GLMAdapter: 老 GLM / ChatGLM2/3 (标准 q/k/v/o proj)
- GLM5MoEAdapter: GLM-5.1 (MLA attention + MoE + DSA indexer)
"""

import torch.nn as nn
from typing import Optional
from .base import BaseModelAdapter


class GLM5MoEAdapter(BaseModelAdapter):
    """
    GLM-5.1 MoE DSA 模型适配器

    模型结构:
        model (GlmMoeDsaForCausalLM)
        └── model (GlmMoeDsaModel)
            ├── layers (ModuleList)
            │   └── [i] (DecoderLayer)
            │       ├── input_layernorm
            │       ├── self_attn (MLA + DSA Indexer)
            │       │   ├── q_a_proj       (压缩 q 投影)
            │       │   ├── q_a_layernorm
            │       │   ├── q_b_proj       (解压 q heads)
            │       │   ├── kv_a_proj_with_mqa  (压缩 kv 投影)
            │       │   ├── kv_a_layernorm
            │       │   ├── kv_b_proj      (解压 kv heads)
            │       │   ├── o_proj
            │       │   └── indexer        (DSA 稀疏注意力索引)
            │       │       ├── wq_b
            │       │       ├── wk
            │       │       ├── k_norm
            │       │       └── weights_proj
            │       ├── post_attention_layernorm
            │       └── mlp (MoE)
            │           ├── gate_proj / shared_expert
            │           └── experts (256 routed experts)
            │               └── [e].{gate_proj, up_proj, down_proj}
            ├── embed_tokens
            └── norm
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
    # 3. Projection 访问 (MLA 映射)
    # ========================================================================

    def get_q_proj(self, attn: nn.Module) -> nn.Module:
        # MLA: q_a_proj 是 q 的入口投影，hook 住它最有诊断价值
        return attn.q_a_proj

    def get_k_proj(self, attn: nn.Module) -> nn.Module:
        # MLA: kv_a_proj_with_mqa 是 kv 的入口投影
        return attn.kv_a_proj_with_mqa

    def get_v_proj(self, attn: nn.Module) -> nn.Module:
        # MLA 没有独立的 v_proj，kv_a_proj_with_mqa 同时处理 k/v
        return attn.kv_a_proj_with_mqa

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
        return f"layer.{layer_idx}.pre_rope.q_a_proj"

    def get_pre_rope_k_name(self, layer_idx: int) -> str:
        return f"layer.{layer_idx}.pre_rope.kv_a_proj"

    def get_embedding_name(self) -> str:
        return "embedding"

    def get_final_norm_name(self) -> str:
        return "final_norm"

    def get_logits_name(self) -> str:
        return "logits"

    # ========================================================================
    # 6. 能力声明
    # ========================================================================

    def supports_pre_rope_qk(self) -> bool:
        # MLA: q_a_proj 输出是压缩表示，不是标准 pre-RoPE q
        return False

    def supports_attn_detail(self) -> bool:
        # MLA 结构不是标准 q/k/v 分离，细粒度诊断受限
        return False

    # ========================================================================
    # 7. 参数名映射
    # ========================================================================

    def get_quant_key_prefix(self) -> str:
        return ""


class GLMAdapter(BaseModelAdapter):
    """
    GLM 模型适配器

    模型结构 (可能有变体):
        model
        ├── transformer
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

    def _get_transformer(self):
        """获取 transformer 模块"""
        if hasattr(self.model, "transformer"):
            return self.model.transformer
        if hasattr(self.model, "model"):
            return self.model.model
        return self.model

    # ========================================================================
    # 1. 层级访问
    # ========================================================================

    def get_layers(self) -> nn.ModuleList:
        transformer = self._get_transformer()
        if hasattr(transformer, "layers"):
            return transformer.layers
        elif hasattr(transformer, "h"):
            return transformer.h
        raise ValueError(f"Cannot find layers in {type(transformer)}")

    def get_num_layers(self) -> int:
        return len(self.get_layers())

    def get_layer(self, idx: int) -> nn.Module:
        return self.get_layers()[idx]

    # ========================================================================
    # 2. 模块访问
    # ========================================================================

    def get_input_norm(self, layer: nn.Module) -> nn.Module:
        # GLM 可能用 input_layernorm 或 ln_1
        if hasattr(layer, "input_layernorm"):
            return layer.input_layernorm
        elif hasattr(layer, "ln_1"):
            return layer.ln_1
        raise ValueError(f"Cannot find input_norm in {type(layer)}")

    def get_post_attn_norm(self, layer: nn.Module) -> nn.Module:
        # GLM 可能用 post_attention_layernorm 或 ln_2
        if hasattr(layer, "post_attention_layernorm"):
            return layer.post_attention_layernorm
        elif hasattr(layer, "ln_2"):
            return layer.ln_2
        raise ValueError(f"Cannot find post_attn_norm in {type(layer)}")

    def get_attention(self, layer: nn.Module) -> nn.Module:
        # GLM 可能用 self_attn 或 attention
        if hasattr(layer, "self_attn"):
            return layer.self_attn
        elif hasattr(layer, "attention"):
            return layer.attention
        raise ValueError(f"Cannot find attention in {type(layer)}")

    def get_mlp(self, layer: nn.Module) -> nn.Module:
        return layer.mlp

    # ========================================================================
    # 3. Projection 访问
    # ========================================================================

    def get_q_proj(self, attn: nn.Module) -> nn.Module:
        if hasattr(attn, "q_proj"):
            return attn.q_proj
        elif hasattr(attn, "qKV_proj"):  # 某些 GLM 变体
            return attn.qKV_proj
        raise ValueError(f"Cannot find q_proj in {type(attn)}")

    def get_k_proj(self, attn: nn.Module) -> nn.Module:
        if hasattr(attn, "k_proj"):
            return attn.k_proj
        raise ValueError(f"Cannot find k_proj in {type(attn)}")

    def get_v_proj(self, attn: nn.Module) -> nn.Module:
        if hasattr(attn, "v_proj"):
            return attn.v_proj
        raise ValueError(f"Cannot find v_proj in {type(attn)}")

    def get_o_proj(self, attn: nn.Module) -> nn.Module:
        if hasattr(attn, "o_proj"):
            return attn.o_proj
        raise ValueError(f"Cannot find o_proj in {type(attn)}")

    # ========================================================================
    # 4. 特殊层访问
    # ========================================================================

    def get_embed_layer(self) -> nn.Module:
        transformer = self._get_transformer()
        if hasattr(transformer, "embed_tokens"):
            return transformer.embed_tokens
        elif hasattr(transformer, "embedding"):
            return transformer.embedding
        raise ValueError("Cannot find embed_tokens")

    def get_final_norm(self) -> nn.Module:
        transformer = self._get_transformer()
        if hasattr(transformer, "norm"):
            return transformer.norm
        raise ValueError("Cannot find final norm")

    def get_lm_head(self) -> nn.Module:
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        elif hasattr(self.model, "output_proj"):
            return self.model.output_proj
        raise ValueError("Cannot find lm_head")

    # ========================================================================
    # 5. Hook 点定义
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
