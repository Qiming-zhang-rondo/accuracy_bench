"""
OperatorPatcher: 反事实替换的核心执行器。

包含：
  - ReplacementHook: 替换 nn.Linear output 的 forward hook
  - _resolve_path: 通过点分隔路径查找子模块
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Optional, Tuple

import torch
import torch.nn as nn


class ReplacementHook:
    """
    Forward hook 实现 output 替换。

    在 quant forward 中，将指定 module 的 output 替换为预先捕获的 ref output。
    """

    def __init__(
        self,
        ref_output: torch.Tensor,
    ):
        self.ref_output = ref_output
        self._active = False

    def __call__(
        self,
        module: nn.Module,
        module_input: Tuple[torch.Tensor, ...],
        module_output,
    ) -> Optional[torch.Tensor]:
        if self._active:
            if isinstance(module_output, tuple):
                target_device = module_output[0].device
                ref = self.ref_output.to(target_device, non_blocking=True)
                return (ref,) + module_output[1:]
            ref = self.ref_output.to(module_output.device, non_blocking=True)
            return ref
        return None

    @contextmanager
    def active(self) -> Generator[None, None, None]:
        old = self._active
        self._active = True
        try:
            yield
        finally:
            self._active = old


def _resolve_path(module: nn.Module, path: str) -> Optional[nn.Module]:
    """通过点分隔路径查找子模块。例: _resolve_path(layer_module, "mlp.gate_proj")"""
    parts = path.split(".")
    current = module
    for part in parts:
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current
