"""
INT4 per-token 对称线性激活伪量化模块。

对齐 msmodelslim `W4A4DynamicPerChannelFakeQuantLinear` /
`W4A4DynamicPerGroupFakeQuantLinear` / `W4A4LAOSFakeQuantLinear` 的激活侧
(三者激活侧一致: INT4 per-token symmetric)。

决策 (docs/proposals/a4_activation.md §6):
  D4: round = half-even (torch.round() 默认, 与 msmodelslim int_quantize .round_() 一致)
  D5: clamp [-8, 7] (标准对称 INT4, scale = amax / 7)
  per-token 粒度 (沿 hidden 维算 amax), float32 线性 scale (非 E8M0)
"""

from __future__ import annotations

import torch
from torch import Tensor

INT4_MAX = 7  # symmetric INT4: scale = amax/7, clamp range [-8, 7]
_EPS = torch.finfo(torch.float32).eps


def int4_fake_quant_per_token_sym(x: Tensor) -> Tensor:
    """Per-token 对称 INT4 激活伪量化。

    在 float32 中模拟 INT4 per-token symmetric 的量化+反量化 (fake quant)。
    可在 NPU 或 CPU 上执行。

    Args:
        x: 输入 tensor, 任意形状, 最后一维是 hidden_size
    Returns:
        伪量化后的 tensor, 与输入同 shape 和 dtype
    """
    original_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    original_shape = x_fp32.shape
    hidden_size = original_shape[-1]

    # reshape to [batch*seq, hidden] (per-token = per-row)
    x_2d = x_fp32.reshape(-1, hidden_size)

    # per-token amax = max(|amin|, |amax|) along hidden dim
    x_max = x_2d.amax(dim=1, keepdim=True)
    x_min = x_2d.amin(dim=1, keepdim=True)
    amax = torch.max(-x_min, x_max).clamp(min=_EPS)

    # float32 线性 scale (非 E8M0), symmetric: scale = amax / 7
    scale = amax / INT4_MAX

    # 量化: round half-even (D4, torch.round 默认) + clamp [-8, 7] (D5)
    x_q = torch.round(x_2d / scale)
    x_q = torch.clamp(x_q, min=-8.0, max=7.0)

    # 反量化
    x_fake_quant = x_q * scale

    return x_fake_quant.reshape(original_shape).to(original_dtype)
