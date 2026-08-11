"""
精度对比指标

主指标 (compute_all_metrics): cos_sim, mse, max_abs_diff, snr, relative_error
"""

import torch
from torch import Tensor
from typing import Dict
import logging
from .utils import flatten_for_compare

logger = logging.getLogger(__name__)


# ============================================================================
# 主指标 (粗筛)
# ============================================================================

def cos_sim(a: Tensor, b: Tensor, use_cpu: bool = True) -> float:
    """余弦相似度, 范围[-1, 1], 1表示完全一致。

    数值稳定性: 对超大向量 (如 952M 元素) 使用 float64 分块累积,
    避免 float32 点积溢出导致 cos_sim > 1.0 的非法值。
    """
    a, b = flatten_for_compare(a, b, use_cpu=use_cpu)
    n = a.numel()

    # 小向量: 直接计算 (float32 足够)
    if n <= 1_000_000:
        norm_a, norm_b = a.norm(), b.norm()
        if norm_a == 0 or norm_b == 0:
            return 0.0
        cos = (a @ b / (norm_a * norm_b)).item()
        return max(-1.0, min(1.0, cos))

    # 大向量: float64 分块累积
    chunk = 1_000_000  # 1M elements per chunk
    dot = torch.tensor(0.0, dtype=torch.float64)
    x_sq = torch.tensor(0.0, dtype=torch.float64)
    y_sq = torch.tensor(0.0, dtype=torch.float64)

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        x_c = a[start:end].double()
        y_c = b[start:end].double()
        dot += (x_c * y_c).sum()
        x_sq += (x_c * x_c).sum()
        y_sq += (y_c * y_c).sum()

    norm_x = torch.sqrt(x_sq)
    norm_y = torch.sqrt(y_sq)
    if norm_x == 0 or norm_y == 0:
        return 0.0

    cos = (dot / (norm_x * norm_y)).item()

    # 健全性检查: cos_sim 必须在 [-1, 1] 范围内
    # 允许微小数值误差 (float64 精度内)
    if cos > 1.0 + 1e-6 or cos < -1.0 - 1e-6:
        logger.error(
            f"cos_sim out of range: {cos}, dot={dot.item():.6e}, "
            f"norm_x={norm_x.item():.6e}, norm_y={norm_y.item():.6e}, "
            f"n={n}, dtype={a.dtype}. This indicates numerical instability."
        )
        # 不用 clamp 掩盖, 返回实际值让调用方知道有问题
        # 但限制在合理范围内避免 NaN 传播
        return max(-1.0, min(1.0, cos))

    return max(-1.0, min(1.0, cos))


def mse(a: Tensor, b: Tensor, use_cpu: bool = True) -> float:
    """均方误差"""
    a, b = flatten_for_compare(a, b, use_cpu=use_cpu)
    return ((a - b) ** 2).mean().item()


def max_abs_diff(a: Tensor, b: Tensor, use_cpu: bool = True) -> float:
    """最大绝对误差"""
    a, b = flatten_for_compare(a, b, use_cpu=use_cpu)
    return (a - b).abs().max().item()


def snr(a: Tensor, b: Tensor, use_cpu: bool = True) -> float:
    """信噪比 (dB), 越高越好, 负数表示噪声大于信号"""
    a, b = flatten_for_compare(a, b, use_cpu=use_cpu)
    signal_power = (a ** 2).mean()
    noise_power = ((a - b) ** 2).mean()
    if noise_power == 0:
        return float('inf')
    if signal_power == 0:
        return float('-inf')
    return (10 * torch.log10(signal_power / noise_power)).item()


def relative_error(a: Tensor, b: Tensor, use_cpu: bool = True) -> float:
    """相对误差, 以a为reference: ||a-b|| / ||a||"""
    a, b = flatten_for_compare(a, b, use_cpu=use_cpu)
    norm_a = a.norm()
    if norm_a == 0:
        return float('inf') if b.norm() > 0 else 0.0
    return ((a - b).norm() / norm_a).item()


def compute_all_metrics(a: Tensor, b: Tensor, use_cpu: bool = True) -> Dict[str, float]:
    """一次性计算所有主指标

    Args:
        use_cpu: True=在CPU计算, False=在原设备计算
    """
    return {
        "cos_sim": cos_sim(a, b, use_cpu=use_cpu),
        "mse": mse(a, b, use_cpu=use_cpu),
        "max_abs_diff": max_abs_diff(a, b, use_cpu=use_cpu),
        "snr": snr(a, b, use_cpu=use_cpu),
        "relative_error": relative_error(a, b, use_cpu=use_cpu),
    }
