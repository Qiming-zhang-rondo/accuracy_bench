"""
WeightPatcher: forward-level functional weight patch.

对 quant module 临时替换 forward，使其用 ref module 的 weight 重新计算，
而不是直接返回 ref output.

这是真正的 weight-level counterfactual:
output 仍然由当前输入 x 重新算出来，但用的是 ref 的 weight.

对比 output patch (capture_ref_outputs / patch_op):
  output patch:  跳过整个计算，直接返回 ref output
  weight patch:  保持计算链路，只换 weight

支持的 strategy:
  - linear_ref_weight: F.linear(x, ref_weight, ref_bias)

用法:
    from accuracy_checker.weight_patcher import patch_by_path

    with patch_by_path(quant_layer, ref_layer, "mlp.gate_proj", "linear_ref_weight"):
        patched_out = quant_layer(common_input)

恢复来源: accuracy_checker_v2 分支 (158 行, 自包含, 仅依赖 torch/types/contextlib).
集成位置: 供 subgraph_locate.py / L2 反事实诊断调用 (Agent C/D 整合).
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn.functional as F


def get_module_by_path(root: torch.nn.Module, path: str) -> torch.nn.Module:
    """
    根据字符串路径拿 module.
    例如 path = "mlp.gate_proj" → root.mlp.gate_proj
    """
    return root.get_submodule(path)


def _get_ref_weight_bias(ref_module: torch.nn.Module) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """获取 ref module 的 weight 和 bias (detach, CPU)."""
    w = ref_module.weight.detach().cpu() if hasattr(ref_module, "weight") else None
    b = None
    if hasattr(ref_module, "bias") and ref_module.bias is not None:
        b = ref_module.bias.detach().cpu()
    return w, b


def _make_linear_ref_weight_forward(
    quant_module: torch.nn.Module,
    ref_weight: torch.Tensor,
    ref_bias: Optional[torch.Tensor],
) -> types.MethodType:
    """
    创建一个替换 forward，用 ref weight 做 F.linear，绑定到 quant_module 实例.

    使用 cache 避免每次 forward 都 to(device)，只在 device/dtype 变化时重建.
    """
    cache: Dict[Tuple[str, torch.dtype], Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}

    def new_forward(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        key = (str(x.device), x.dtype)
        if key not in cache:
            w = ref_weight.to(device=x.device, dtype=x.dtype)
            b = ref_bias.to(device=x.device, dtype=x.dtype) if ref_bias is not None else None
            cache[key] = (w, b)
        else:
            w, b = cache[key]
        return F.linear(x, w, b)

    return types.MethodType(new_forward, quant_module)


@contextmanager
def patch_by_path(
    quant_root: torch.nn.Module,
    ref_root: torch.nn.Module,
    module_path: str,
    strategy: str = "linear_ref_weight",
) -> Iterator[None]:
    """
    对 quant_root 下指定 module_path 的 module 做 forward-level patch.

    Args:
        quant_root: quant 模型的根 module (通常是某一层 decoder layer)
        ref_root:   ref 模型的根 module (和 quant_root 同结构)
        module_path: module 路径, 如 "mlp.gate_proj"
        strategy:   patch 策略, 当前支持 "linear_ref_weight"

    用法:
        with patch_by_path(quant_layer, ref_layer, "mlp.gate_proj", "linear_ref_weight"):
            patched_out = quant_layer(common_input)
    """
    quant_module = get_module_by_path(quant_root, module_path)
    ref_module = get_module_by_path(ref_root, module_path)

    if strategy == "linear_ref_weight":
        ref_weight, ref_bias = _get_ref_weight_bias(ref_module)
        if ref_weight is None:
            raise ValueError(f"{module_path}: ref module has no weight")

        new_forward = _make_linear_ref_weight_forward(quant_module, ref_weight, ref_bias)
        old_forward = quant_module.forward
        quant_module.forward = new_forward
        try:
            yield
        finally:
            quant_module.forward = old_forward
    else:
        raise ValueError(f"Unsupported patch strategy: {strategy}")


@contextmanager
def patch_multi(
    quant_root: torch.nn.Module,
    ref_root: torch.nn.Module,
    module_paths: list,
    strategy: str = "linear_ref_weight",
) -> Iterator[None]:
    """
    同时对 quant_root 下多个 module_path 做 forward-level patch.

    等价于依次对每个 module_path 调 patch_by_path, 但保证所有 patch
    同时生效, 退出后全部恢复.

    Args:
        quant_root: quant 模型的根 module
        ref_root:   ref 模型的根 module
        module_paths: module 路径列表, 如 ["mlp.gate_proj", "mlp.up_proj"]
        strategy:   patch 策略

    用法:
        with patch_multi(quant_layer, ref_layer,
                         ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]):
            patched_out = quant_layer(common_input)
    """
    if strategy == "linear_ref_weight":
        old_forwards: dict = {}
        for path in module_paths:
            q_mod = get_module_by_path(quant_root, path)
            r_mod = get_module_by_path(ref_root, path)
            ref_w, ref_b = _get_ref_weight_bias(r_mod)
            if ref_w is None:
                raise ValueError(f"{path}: ref module has no weight")
            new_fwd = _make_linear_ref_weight_forward(q_mod, ref_w, ref_b)
            old_forwards[path] = q_mod.forward
            q_mod.forward = new_fwd
        try:
            yield
        finally:
            for path in module_paths:
                q_mod = get_module_by_path(quant_root, path)
                q_mod.forward = old_forwards[path]
    else:
        raise ValueError(f"Unsupported patch strategy: {strategy}")
