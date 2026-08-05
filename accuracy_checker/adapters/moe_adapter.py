"""
MoE (Mixture of Experts) 权重适配器

用于处理 MoE 模型的权重打包/解包逻辑。
不同的 MoE 实现可能有不同的权重组织方式:

1. Dense 格式: 每个 expert 是独立的 module
2. Fused/Packed 格式: 多个 expert 的权重打包在一起

这个模块提供统一的接口来适配不同的 MoE 权重格式。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import logging



logger = logging.getLogger(__name__)
class BaseMoEAdapter(ABC):
    """
    MoE 权重适配器基类

    定义处理 MoE 权重的接口:
    - 如何从 quant_weights 中查找 expert 权重
    - 如何将分离的权重打包成模型期望的格式
    - 如何将打包的权重解压回分离格式 (用于对比)
    """

    def __init__(self, model: nn.Module = None):
        self.model = model

    @abstractmethod
    def get_expert_weight_key(
        self,
        layer_idx: int,
        expert_idx: int,
        proj_name: str,
    ) -> str:
        """
        获取专家权重的 key

        Args:
            layer_idx: 层索引
            expert_idx: expert 索引 (0, 1, 2, ...)
            proj_name: 投影名称 (如 'gate_proj', 'up_proj', 'down_proj')

        Returns:
            quant_weights 中的 key
        """
        pass

    @abstractmethod
    def pack_gate_up(
        self,
        gate_weights: List[torch.Tensor],
        up_weights: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        将分离的 gate+up 权重打包成模型期望的格式

        Args:
            gate_weights: [expert_0_gate, expert_1_gate, ...] 每个 shape [intermediate, hidden]
            up_weights: [expert_0_up, expert_1_up, ...] 每个 shape [intermediate, hidden]

        Returns:
            打包后的 tensor，模型期望的 shape
        """
        pass

    @abstractmethod
    def pack_down(
        self,
        down_weights: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        将分离的 down 权重堆叠成模型期望的格式

        Args:
            down_weights: [expert_0_down, expert_1_down, ...] 每个 shape [hidden, intermediate]

        Returns:
            堆叠后的 tensor，模型期望的 shape
        """
        pass

    @abstractmethod
    def get_target_param_names(self, layer_idx: int) -> Tuple[str, str]:
        """
        获取模型中目标参数的名称

        Args:
            layer_idx: 层索引

        Returns:
            (gate_up_param_name, down_param_name) 如 ('gate_up_proj', 'down_proj')
        """
        pass

    def _dequantize_weight(
        self,
        quant_weights: Dict[str, torch.Tensor],
        quant_desc: Dict[str, str],
        weight_key: str,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """反量化单个权重"""
        # 避免循环导入，在函数内部导入
        import accuracy_checker.model_loader as ml
        dequantize_weight_dynamic = ml.dequantize_weight_dynamic
        dequantize_weight_static = ml.dequantize_weight_static

        weight_int8 = quant_weights.get(weight_key)
        if weight_int8 is None:
            return None

        # 查找量化类型
        base_name = weight_key.rsplit('.', 1)[0]  # 去掉 .weight
        quant_type = None
        if quant_desc:
            quant_type = quant_desc.get(weight_key) or quant_desc.get(base_name)

        # 如果不是量化类型，直接返回
        if quant_type is None or quant_type == "FLOAT":
            return weight_int8.to(dtype)

        # 反量化
        if quant_type in ("W8A8_DYNAMIC", "W8A8_MIX"):
            weight_scale = quant_weights.get(f"{base_name}.weight_scale")
            weight_offset = quant_weights.get(f"{base_name}.weight_offset")
            if weight_scale is not None:
                return dequantize_weight_dynamic(
                    weight_int8, weight_scale, weight_offset, dtype
                )

        # W8A8 静态量化
        if quant_type == "W8A8":
            deq_scale = quant_weights.get(f"{base_name}.deq_scale")
            input_scale = quant_weights.get(f"{base_name}.input_scale")
            if deq_scale is not None:
                w_fp, _ = dequantize_weight_static(
                    weight_int8, deq_scale, input_scale, dtype=dtype
                )
                return w_fp

        # 默认返回原始值
        return weight_int8.to(dtype)

    def load_expert_weights_to_layer(
        self,
        layer: nn.Module,
        layer_idx: int,
        quant_weights: Dict[str, torch.Tensor],
        quant_desc: Dict[str, str] = None,
        dtype: torch.dtype = torch.float16,
    ) -> int:
        """
        加载一个层的所有 expert 权重

        Args:
            layer: 模型层 (如 model.model.layers[i])
            layer_idx: 层索引
            quant_weights: 量化权重字典
            quant_desc: 量化描述字典 (可选，用于反量化)
            dtype: 目标数据类型

        Returns:
            加载的权重数量
        """
        gate_up_name, down_name = self.get_target_param_names(layer_idx)

        # 获取 mlp.experts 模块
        mlp = getattr(layer, 'mlp', None)
        if mlp is None:
            return 0

        experts = getattr(mlp, 'experts', None)
        if experts is None:
            return 0

        # 获取 num_experts
        num_experts = getattr(experts, 'num_experts', len(gate_up_name))

        # 收集所有 expert 的权重
        gate_weights = []
        up_weights = []
        down_weights = []

        for exp_idx in range(num_experts):
            # gate_proj
            gate_key = self.get_expert_weight_key(layer_idx, exp_idx, 'gate_proj')
            gate_w = self._dequantize_weight(quant_weights, quant_desc, gate_key, dtype)
            if gate_w is not None:
                gate_weights.append(gate_w)

            # up_proj
            up_key = self.get_expert_weight_key(layer_idx, exp_idx, 'up_proj')
            up_w = self._dequantize_weight(quant_weights, quant_desc, up_key, dtype)
            if up_w is not None:
                up_weights.append(up_w)

            # down_proj
            down_key = self.get_expert_weight_key(layer_idx, exp_idx, 'down_proj')
            down_w = self._dequantize_weight(quant_weights, quant_desc, down_key, dtype)
            if down_w is not None:
                down_weights.append(down_w)

        loaded = 0

        # 打包 gate_up
        if len(gate_weights) > 0 and len(up_weights) > 0:
            try:
                packed_gate_up = self.pack_gate_up(gate_weights, up_weights)
                gate_up_param = getattr(experts, gate_up_name, None)
                if gate_up_param is not None:
                    gate_up_param.data = packed_gate_up
                    loaded += 1
            except Exception as e:
                logger.warning(f"Failed to pack gate_up for layer {layer_idx}: {e}")

        # 堆叠 down
        if len(down_weights) > 0:
            try:
                stacked_down = self.pack_down(down_weights)
                down_param = getattr(experts, down_name, None)
                if down_param is not None:
                    down_param.data = stacked_down
                    loaded += 1
            except Exception as e:
                logger.warning(f"Failed to pack down for layer {layer_idx}: {e}")

        return loaded


class QwenPackedMoEAdapter(BaseMoEAdapter):
    """
    Qwen3.5 MoE 的权重适配器

    模型期望的格式:
    - gate_up_proj: [num_experts, 2*intermediate_size, hidden_size]
    - down_proj: [num_experts, hidden_size, intermediate_size]

    quant_weights 中的格式:
    - experts.{i}.gate_proj.weight: [intermediate_size, hidden_size]
    - experts.{i}.up_proj.weight: [intermediate_size, hidden_size]
    - experts.{i}.down_proj.weight: [hidden_size, intermediate_size]
    """

    # quant_weights 中的 key 模板
    KEY_TEMPLATE = "model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"

    # 模型中的参数名
    GATE_UP_PARAM = "gate_up_proj"
    DOWN_PARAM = "down_proj"

    def __init__(
        self,
        model: nn.Module = None,
        model_key_prefix: str = "model.",
        quant_key_prefix: str = "model.language_model.",
    ):
        super().__init__(model)
        self.model_key_prefix = model_key_prefix
        self.quant_key_prefix = quant_key_prefix

    def get_expert_weight_key(
        self,
        layer_idx: int,
        expert_idx: int,
        proj_name: str,
    ) -> str:
        key = self.KEY_TEMPLATE.format(
            layer=layer_idx,
            expert=expert_idx,
            proj=proj_name,
        )
        # 添加 quant key 前缀
        if self.quant_key_prefix and not key.startswith(self.quant_key_prefix):
            key = key.replace(self.model_key_prefix, self.quant_key_prefix, 1)
        return key

    def pack_gate_up(
        self,
        gate_weights: List[torch.Tensor],
        up_weights: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        打包 gate+up: [num_experts, 2*intermediate, hidden]
        """
        num_experts = len(gate_weights)
        # gate 和 up 都是 [intermediate, hidden]
        # 拼接: [2*intermediate, hidden]
        # 然后 stack: [num_experts, 2*intermediate, hidden]
        fused = []
        for gate, up in zip(gate_weights, up_weights):
            # gate/up 都是 [intermediate, hidden]
            fused.append(torch.cat([gate, up], dim=0))  # [2*intermediate, hidden]

        return torch.stack(fused, dim=0)  # [num_experts, 2*intermediate, hidden]

    def pack_down(
        self,
        down_weights: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        堆叠 down: [num_experts, hidden, intermediate]
        """
        return torch.stack(down_weights, dim=0)

    def get_target_param_names(self, layer_idx: int) -> Tuple[str, str]:
        return (self.GATE_UP_PARAM, self.DOWN_PARAM)


class DenseMoEAdapter(BaseMoEAdapter):
    """
    Dense MoE 适配器 (每个 expert 是独立 module)

    模型期望每个 expert 是独立的子模块:
    - mlp.experts.0.gate_proj
    - mlp.experts.0.up_proj
    - mlp.experts.0.down_proj
    - mlp.experts.1.gate_proj
    - ...
    """

    def __init__(
        self,
        model: nn.Module = None,
        model_key_prefix: str = "model.",
        quant_key_prefix: str = "model.language_model.",
    ):
        super().__init__(model)
        self.model_key_prefix = model_key_prefix
        self.quant_key_prefix = quant_key_prefix

    def get_expert_weight_key(
        self,
        layer_idx: int,
        expert_idx: int,
        proj_name: str,
    ) -> str:
        key = f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}.weight"
        return key

    def pack_gate_up(
        self,
        gate_weights: List[torch.Tensor],
        up_weights: List[torch.Tensor],
    ) -> torch.Tensor:
        # Dense 格式不需要打包，直接返回第一个（不应该被调用）
        raise NotImplementedError("Dense MoE does not use packed format")

    def pack_down(
        self,
        down_weights: List[torch.Tensor],
    ) -> torch.Tensor:
        # Dense 格式不需要堆叠，直接返回第一个（不应该被调用）
        raise NotImplementedError("Dense MoE does not use packed format")

    def get_target_param_names(self, layer_idx: int) -> Tuple[str, str]:
        # Dense 格式不使用打包参数
        return (None, None)

    def load_expert_weights_to_layer(
        self,
        layer: nn.Module,
        layer_idx: int,
        quant_weights: Dict[str, torch.Tensor],
        dtype: torch.dtype = torch.float16,
    ) -> int:
        """Dense 格式: 逐个加载到 experts.X.gate_proj 等"""
        mlp = getattr(layer, 'mlp', None)
        if mlp is None:
            return 0

        experts = getattr(mlp, 'experts', None)
        if experts is None:
            return 0

        loaded = 0
        num_experts = getattr(experts, 'num_experts', 64)

        for exp_idx in range(num_experts):
            # 获取 experts[exp_idx] 子模块
            exp_module = experts[exp_idx] if hasattr(experts, '__getitem__') else None

            for proj in ['gate_proj', 'up_proj', 'down_proj']:
                weight_key = self.get_expert_weight_key(layer_idx, exp_idx, proj)
                weight = quant_weights.get(weight_key)

                if weight is not None and exp_module is not None:
                    param = getattr(exp_module, proj, None)
                    if param is not None and hasattr(param, 'weight'):
                        param.weight.data = weight.to(dtype)
                        loaded += 1

        return loaded


# 注册表
MOE_ADAPTER_REGISTRY = {
    "qwen3_5_moe": QwenPackedMoEAdapter,
    "qwen3_moe": QwenPackedMoEAdapter,  # 可能结构类似
    "dense": DenseMoEAdapter,
}


def get_moe_adapter(model_type: str, model: nn.Module = None, **kwargs) -> BaseMoEAdapter:
    """
    根据模型类型获取合适的 MoE 适配器

    Args:
        model_type: 模型类型标识 (如 'qwen3_5_moe')
        model: 模型实例
        **kwargs: 传递给适配器的额外参数

    Returns:
        MoE 适配器实例
    """
    adapter_class = MOE_ADAPTER_REGISTRY.get(model_type, QwenPackedMoEAdapter)
    return adapter_class(model=model, **kwargs)
