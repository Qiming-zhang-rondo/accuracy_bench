"""
V2 核心指标: rel_l2 + recovery_ratio + cos_sim
"""

from typing import List, Optional

import torch


def _select_positions(t: torch.Tensor, positions: Optional[List[int]] = None) -> torch.Tensor:
    if positions is None:
        return t
    # t: (batch, seq, ...), positions index into seq dim
    if t.dim() >= 2:
        return t[:, positions]
    return t


def rel_l2(
    a: torch.Tensor,
    b: torch.Tensor,
    token_positions: Optional[List[int]] = None,
) -> torch.Tensor:
    """相对 L2 误差: ||a - b||_2 / ||b||_2"""
    a = _select_positions(a, token_positions)
    b = _select_positions(b, token_positions)
    diff = a - b
    return torch.norm(diff) / (torch.norm(b) + 1e-12)


def cos_sim(
    a: torch.Tensor,
    b: torch.Tensor,
    token_positions: Optional[List[int]] = None,
) -> torch.Tensor:
    """Cosine similarity: 逐 token 算 cos，再取平均。"""
    a = _select_positions(a, token_positions)
    b = _select_positions(b, token_positions)
    a_flat = a.flatten(0, -2)
    b_flat = b.flatten(0, -2)
    dot = (a_flat * b_flat).sum(dim=-1)
    norm_a = torch.norm(a_flat, dim=-1)
    norm_b = torch.norm(b_flat, dim=-1)
    return (dot / (norm_a * norm_b + 1e-12)).mean()


def recovery_ratio(baseline_error: float, patched_error: float) -> float:
    """恢复比例: (baseline - patched) / baseline"""
    if abs(baseline_error) < 1e-12:
        return 0.0
    return (baseline_error - patched_error) / baseline_error
