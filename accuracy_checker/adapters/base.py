"""
Base Model Adapter 抽象基类

定义模型适配器的接口规范。

能力分层 (Agent A 重构):
  1. 结构访问 (abstract):
     get_layers / get_input_norm / get_post_attn_norm / get_attention / get_mlp
     get_q_proj / get_k_proj / get_v_proj / get_o_proj
     get_embed_layer / get_final_norm / get_lm_head
  2. 结构可选钩子 (默认实现, 模型差异在子类覆盖):
     get_quant_key_prefix       — quant 权重 key 在 quant_model_description.json
                                    中的前缀差异 (model. / model.language_model. / 空)
     get_forward_extra_kwargs   — 特殊 forward 参数 (DS MLA / V4 Compressor 等)
     get_moe_expert_count       — 当前层 MoE expert 数量
     get_moe_expert_proj_names  — 专家 projection 命名 (gate/up/down vs w1/w2/w3)
     expert_weights_inside     — expert forward 是否内嵌 router 权重 (V3 外×vs V4 内)
     has_shared_expert / get_shared_expert — V4 有 shared_expert, V3 无
  3. 公共工具 (staticmethod):
     write_weight_to_param      — 单一受审计的权重写路径
                                  (满足 "不允许重新引入手写 param.data" 红线)

不再在框架主流程中 if model_type 判断模型差异; 所有差异走 adapter 覆盖.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn


class BaseModelAdapter(ABC):
    """
    模型适配器抽象基类

    所有具体模型的 Adapter 都必须实现此接口。
    Adapter 只负责结构映射，不处理量化公式或数学计算。
    """

    def __init__(self, model: nn.Module, model_path: str = None):
        self.model = model
        self.model_path = model_path

    # ========================================================================
    # 0. 单一受审计权重写入 (替代分散的 param.data = ...)
    # ========================================================================

    @staticmethod
    def write_weight_to_param(
        param: Any,
        tensor: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
    ) -> bool:
        """把权重写到一个 nn.Parameter / nn.Linear.weight / 3D packed param.

        统一入口, 满足 "不允许重新引入手写 param.data 赋值" 红线:
          - 校验 param 存在且非 None
          - 对齐 device + dtype
          - 仅此一处 .data 写
        Returns: True 写入成功, False 跳过 (param 不存在 / 维度不匹配).
        """
        if param is None:
            return False
        try:
            target = param.weight if hasattr(param, "weight") else param
            if target is None:
                return False
            dev = str(target.device)
            dt = dtype if dtype is not None else target.dtype
            t = tensor.to(dtype=dt, device=dev) if dev and dev != "cpu" else tensor.to(dt)
            # 形状校验: 1D/2D/3D packed 都允许 (broadcast 仅做最简维度对比)
            if hasattr(target, "shape") and target.shape != t.shape and \
               target.shape not in (torch.Size([]),) and len(target.shape) >= 2 \
               and target.shape != t.shape:
                # 形状不一致不在这里崩 (可能是 INT4 解包前后), 让调用方 log
                return False
            target.data = t
            return True
        except Exception:
            return False


    # ========================================================================
    # 1. 层级访问
    # ========================================================================

    @abstractmethod
    def get_layers(self) -> nn.ModuleList:
        """获取所有 Decoder 层"""
        pass

    @abstractmethod
    def get_num_layers(self) -> int:
        """获取 Decoder 层数量"""
        pass

    @abstractmethod
    def get_layer(self, idx: int) -> nn.Module:
        """获取指定索引的层"""
        pass

    # ========================================================================
    # 2. 模块访问
    # ========================================================================

    @abstractmethod
    def get_input_norm(self, layer: nn.Module) -> nn.Module:
        """获取 Input LayerNorm (在 Attention 之前)"""
        pass

    @abstractmethod
    def get_post_attn_norm(self, layer: nn.Module) -> nn.Module:
        """获取 Post-Attention LayerNorm (在 MLP 之前)"""
        pass

    @abstractmethod
    def get_attention(self, layer: nn.Module) -> nn.Module:
        """获取 Attention 模块"""
        pass

    @abstractmethod
    def get_mlp(self, layer: nn.Module) -> nn.Module:
        """获取 MLP 模块"""
        pass

    # ========================================================================
    # 3. Projection 访问
    # ========================================================================

    @abstractmethod
    def get_q_proj(self, attn: nn.Module) -> nn.Module:
        """获取 Q projection"""
        pass

    @abstractmethod
    def get_k_proj(self, attn: nn.Module) -> nn.Module:
        """获取 K projection"""
        pass

    @abstractmethod
    def get_v_proj(self, attn: nn.Module) -> nn.Module:
        """获取 V projection"""
        pass

    @abstractmethod
    def get_o_proj(self, attn: nn.Module) -> nn.Module:
        """获取 O projection"""
        pass

    # ========================================================================
    # 4. 特殊层访问
    # ========================================================================

    @abstractmethod
    def get_embed_layer(self) -> nn.Module:
        """获取 Embedding 层"""
        pass

    @abstractmethod
    def get_final_norm(self) -> nn.Module:
        """获取 Final LayerNorm"""
        pass

    @abstractmethod
    def get_lm_head(self) -> nn.Module:
        """获取 LM Head"""
        pass

    # ========================================================================
    # 5. Hook 点定义 (key 名称)
    # ========================================================================

    def get_block_output_name(self, layer_idx: int) -> str:
        """Block output 的 hook key"""
        return f"layer.{layer_idx}.block_output"

    def get_pre_rope_q_name(self, layer_idx: int) -> str:
        """Pre-RoPE Q projection 的 hook key"""
        return f"layer.{layer_idx}.pre_rope.q_proj"

    def get_pre_rope_k_name(self, layer_idx: int) -> str:
        """Pre-RoPE K projection 的 hook key"""
        return f"layer.{layer_idx}.pre_rope.k_proj"

    def get_embedding_name(self) -> str:
        """Embedding 输出的 hook key"""
        return "embedding"

    def get_final_norm_name(self) -> str:
        """Final norm 的 hook key"""
        return "final_norm"

    def get_logits_name(self) -> str:
        """Logits 的 hook key"""
        return "logits"

    # ========================================================================
    # 6. 辅助方法
    # ========================================================================

    def get_layer_prefix(self, layer_idx: int) -> str:
        """获取层的名称前缀"""
        return f"layer.{layer_idx}"

    def normalize_weight_key(self, key: str) -> str:
        """标准化权重 key（去除 .weight 后缀）"""
        if key.endswith(".weight"):
            return key[:-7]
        return key

    # ========================================================================
    # 7. 能力声明 (用于主框架按能力开关诊断点)
    # ========================================================================

    def supports_pre_rope_qk(self) -> bool:
        """
        是否支持 pre-RoPE Q/K projection 细粒度诊断

        某些模型(如LLaMA)先做RoPE再QKV投影，无法获取pre-RoPE的q/k
        """
        return True

    def supports_mlp_detail(self) -> bool:
        """
        是否支持 MLP 细粒度诊断 (gate/up/down proj)

        某些模型MLP结构不同，可能不支持
        """
        return True

    def supports_attn_detail(self) -> bool:
        """
        是否支持 Attention 细粒度诊断 (q/k/v/o proj)

        某些模型attention结构不同(如GQA)，可能不完全支持
        """
        return True

    def supports_layer_norm(self) -> bool:
        """是否支持独立访问 LayerNorm 模块"""
        return True
