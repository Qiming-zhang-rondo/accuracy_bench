"""
MXFP8 per-block 激活伪量化模块。

移植自 vLLM-Ascend w8a8_mxfp8.py，在 NPU/CPU 上用 float32 模拟
FP8_E4M3FN 的量化+反量化过程（fake quantization），不需要 float8_e4m3fn dtype。

量化公式 (对齐 Ascend NPU 推理):
  1. 按 block_size=32 分块
  2. 每块算 amax → shared_scale = 2^ceil(log2(amax/448))
  3. 量化: x_q = clamp(round(x / shared_scale * 8) / 8, -448, 448) * shared_scale
  4. 反量化: 直接用 x_q (fake quant 不改变 dtype)

支持 INT 量化的 A8 (W8A8_DYNAMIC 等) 由 fake_quant.py 的 FakeQuantizedLinear 处理。
本模块专门处理 MXFP 系列 (W8A8_MXFP8, W4A8_MXFP) 的激活伪量化。
"""

from __future__ import annotations

import torch
from torch import Tensor

FP8_E4M3FN_MAX = 448.0


def mxfp8_fake_quant_per_block(
    x: Tensor,
    block_size: int = 32,
) -> Tensor:
    """Per-block MXFP8 (E4M3FN) 激活伪量化。

    在 float32 中模拟 FP8_E4M3FN 的量化+反量化，
    不使用 float8_e4m3fn dtype (NPU 不支持 .float() 转换)。
    可在 NPU 或 CPU 上执行。

    Args:
        x: 输入 tensor, 任意形状, 最后一维是 hidden_size
        block_size: MXFP block 大小, 默认 32
    Returns:
        伪量化后的 tensor, 与输入同 shape 和 dtype
    """
    original_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    original_shape = x_fp32.shape
    hidden_size = original_shape[-1]

    if hidden_size % block_size != 0:
        return x

    # reshape to [..., hidden_size // block_size, block_size]
    new_shape = original_shape[:-1] + (hidden_size // block_size, block_size)
    x_blocked = x_fp32.reshape(new_shape)

    # per-block amax → shared scale (E8M0 格式, 2 的幂次)
    x_amax = x_blocked.abs().amax(dim=-1, keepdim=True).clamp(min=torch.finfo(torch.float32).eps)
    scale_log2 = torch.ceil(torch.log2(x_amax / FP8_E4M3FN_MAX))
    shared_scale = torch.pow(2.0, scale_log2)

    # 量化: 缩放到 [-448, 448] 范围, 模拟 E4M3FN 精度 (3 bit mantissa)
    x_scaled = x_blocked / shared_scale
    sign = torch.sign(x_scaled)
    x_abs = torch.abs(x_scaled)

    # E4M3FN: 4 bit exp, 3 bit mantissa → step = 2^(exp-3)
    exp = torch.floor(torch.log2(x_abs + torch.finfo(torch.float32).eps))
    exp = torch.clamp(exp, min=-6.0, max=14.0)

    step = torch.pow(2.0, exp - 3.0)
    x_quant_abs = (x_abs / step).round() * step

    # clip to max representable: 2^exp * (1 + 7/8)
    max_val = torch.pow(2.0, exp) * (1.0 + 7.0 / 8.0)
    x_quant_abs = torch.min(x_quant_abs, max_val)

    # 反量化: 恢复 scale
    x_quantized = x_quant_abs * sign
    x_fake_quantized = x_quantized * shared_scale

    return x_fake_quantized.reshape(original_shape).to(original_dtype)
