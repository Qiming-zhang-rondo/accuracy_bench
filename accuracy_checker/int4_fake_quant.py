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


def _int4_fake_quant_per_token_sym_torch(x: Tensor) -> Tensor:
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


def _unpack_npu_int4(packed: Tensor, original_shape: torch.Size) -> Tensor:
    """Unpack ``npu_dynamic_quant(..., quint4x2)`` in operator byte order.

    torch_npu stores eight signed INT4 values in every int32 result element.
    Within each byte the low nibble precedes the high nibble.  This is the
    reference conversion used by Ascend op-plugin's own dynamic-quant tests.
    """
    packed_u8 = packed.contiguous().view(torch.uint8).reshape(-1, 1)
    low = ((packed_u8 & 0x0F) << 4).view(torch.int8) >> 4
    high = (packed_u8 & 0xF0).view(torch.int8) >> 4
    unpacked = torch.cat((low, high), dim=-1).reshape(-1)
    expected_numel = 1
    for dim in original_shape:
        expected_numel *= int(dim)
    if unpacked.numel() != expected_numel:
        raise RuntimeError(
            "npu_dynamic_quant INT4 packing contract mismatch: "
            f"packed={tuple(packed.shape)}/{packed.dtype}, "
            f"unpacked_numel={unpacked.numel()}, expected={expected_numel}"
        )
    return unpacked.reshape(original_shape)


def _int4_fake_quant_per_token_sym_npu(x: Tensor) -> Tensor:
    """Run the actual Ascend INT4 dynamic-quant operator and dequantize it.

    The previous pure-Torch path has the same ideal formula, but an NPU kernel
    may choose adjacent INT4 values at rounding boundaries.  L1 is intended to
    reproduce deployment numerics, so NPU input must use the real operator.
    """
    if x.device.type != "npu":
        raise ValueError(
            "the npu activation-quant backend requires an NPU tensor, got "
            f"{x.device}"
        )
    try:
        import torch_npu
    except ImportError as exc:  # pragma: no cover - only reachable on NPU envs
        raise RuntimeError(
            "W4A4 activation quantization on NPU requires torch_npu"
        ) from exc

    try:
        packed, per_token_scale = torch_npu.npu_dynamic_quant(
            x, dst_type=torch.quint4x2
        )
    except Exception as exc:
        raise RuntimeError(
            "torch_npu.npu_dynamic_quant(INT4) failed. The correctness path "
            "does not silently fall back to an approximate Torch QDQ on NPU."
        ) from exc

    quantized = _unpack_npu_int4(packed, x.shape)
    scale = per_token_scale.to(torch.float32)
    if tuple(scale.shape) == tuple(x.shape[:-1]):
        scale = scale.unsqueeze(-1)
    elif scale.numel() == x.numel() // x.shape[-1]:
        scale = scale.reshape(*x.shape[:-1], 1)
    else:
        raise RuntimeError(
            "npu_dynamic_quant returned an unexpected per-token scale shape: "
            f"input={tuple(x.shape)}, scale={tuple(scale.shape)}"
        )

    result = quantized.to(torch.float32) * scale
    return result.to(dtype=x.dtype)


def int4_fake_quant_per_token_sym(x: Tensor, backend: str = "auto") -> Tensor:
    """Per-token symmetric INT4 QDQ using the deployment operator when possible.

    ``auto`` uses ``torch_npu.npu_dynamic_quant`` for NPU tensors and the
    formula-equivalent Torch reference for CPU/CUDA tensors. ``torch`` remains
    available for operator-independent unit tests and explicit diagnostics.
    """
    backend = str(backend).strip().lower()
    if backend not in {"auto", "npu", "torch"}:
        raise ValueError(
            f"unsupported INT4 activation backend {backend!r}; "
            "expected auto, npu, or torch"
        )
    resolved = "npu" if backend == "auto" and x.device.type == "npu" else backend
    if resolved == "auto":
        resolved = "torch"
    if resolved == "npu":
        return _int4_fake_quant_per_token_sym_npu(x)
    return _int4_fake_quant_per_token_sym_torch(x)
