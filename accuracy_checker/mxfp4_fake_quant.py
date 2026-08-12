"""
MXFP4 per-block 激活伪量化模块 (E2M1FN microscaling)。

对齐 msmodelslim `W4A4MXDynamicPerBlockFakeQuantLinear` 激活侧 + vLLM `qdq_mxfp4_torch`。
在 float32 中模拟 E2M1 (1 sign + 2 exp + 1 mantissa) 的量化+反量化，不需要 fp4 dtype。

E2M1 可表示值 (与 model_loader.py E2M1_VALUES 一致):
  [0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6]

决策 (docs/proposals/a4_activation.md §6):
  D2: scale = 2^(floor(log2(amax)) - 2)  [OCP even, 对齐 msmodelslim/vLLM 真值]
  D3: round = half-up (floor(|x|/step + 0.5) * step, 对齐 msmodelslim floor(|x|+0.5))
  block_size = 32, max_norm = 6.0
"""

from __future__ import annotations

import torch
from torch import Tensor

E2M1_MAX = 6.0
_E2M1_EMAX = 2  # E2M1 指数位 2 bits → emax=2, max representable = 1.5 * 2^2 = 6.0
_EPS = torch.finfo(torch.float32).eps


def mxfp4_fake_quant_per_block(
    x: Tensor,
    block_size: int = 32,
) -> Tensor:
    """Per-block MXFP4 (E2M1FN) 激活伪量化。

    在 float32 中模拟 E2M1 的量化+反量化 (fake quant)，不使用 fp4 dtype。
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

    # per-block amax → E8M0 shared scale (D2: floor OCP even)
    # shared_exp = floor(log2(amax)) - emax  (= floor(log2(amax)) - 2)
    # amax/scale ∈ (2^emax, 2^(emax+1)] = (4, 8], 超过 6 的值饱和
    x_amax = x_blocked.abs().amax(dim=-1, keepdim=True).clamp(min=_EPS)
    shared_exp = torch.floor(torch.log2(x_amax)) - _E2M1_EMAX
    # E8M0 实际指数范围 [-127, 127] (biased 0-254, bias=127)
    shared_exp = torch.clamp(shared_exp, min=-127.0, max=127.0)
    shared_scale = torch.pow(2.0, shared_exp)

    # 缩放到 E2M1 表示范围
    x_scaled = x_blocked / shared_scale
    sign = torch.sign(x_scaled)
    x_abs = torch.abs(x_scaled)

    # E2M1 量化: 1 bit mantissa
    # private_exp = floor(log2(|x_s|)), clamped to [0, emax] (emin=0, subnormal 由 clamp 处理)
    private_exp = torch.floor(torch.log2(x_abs + _EPS))
    private_exp = torch.clamp(private_exp, min=0.0, max=float(_E2M1_EMAX))

    # 1 mantissa bit → step = 2^(private_exp - 1)
    step = torch.pow(2.0, private_exp - 1.0)

    # round half-up (D3): floor(|x|/step + 0.5) * step
    x_q_abs = torch.floor(x_abs / step + 0.5) * step

    # clamp to global max_norm (不 clamp 到 per-exp max, 允许跨指数进位)
    x_q_abs = torch.clamp(x_q_abs, min=0.0, max=E2M1_MAX)

    # 反量化: 恢复 scale
    x_quantized = x_q_abs * sign
    x_fake_quantized = x_quantized * shared_scale

    return x_fake_quantized.reshape(original_shape).to(original_dtype)
