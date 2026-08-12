"""Reusable end-to-end inference comparison helpers.

This module turns Ref/Quant generated text, token sequences, and optional
logits collections into the shared :class:`InferenceCompareData` report
schema.  It intentionally has no model-loading responsibilities.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Optional, Sequence

import torch

from .report_schema import InferenceCompareData


_LONG_CHAR_RUN = re.compile(r"(.)\1{5,}", re.DOTALL)


def detect_repeat(text: str, tokens: Optional[Sequence[str]] = None) -> bool:
    """Return whether generated output has a strong repetition signal."""
    if not text:
        return False
    if _LONG_CHAR_RUN.search(text):
        return True

    values = list(tokens or [])
    if len(values) >= 4:
        dominant = max(Counter(values).values(), default=0) / len(values)
        if dominant >= 0.5:
            return True

    # Consecutive repeated character n-grams catch loops such as
    # ``abc,abc,abc,`` without flagging ordinary recurring punctuation.
    for width in range(2, min(16, len(text) // 3) + 1):
        for start in range(0, len(text) - width * 3 + 1):
            gram = text[start:start + width]
            if gram.strip() and text[start:start + width * 3] == gram * 3:
                return True
    return False


def detect_garbled(text: str) -> bool:
    """Return whether non-printable control characters dominate the text."""
    if not text:
        return False
    bad = sum(1 for ch in text if not ch.isprintable() and ch not in "\n\r\t")
    return bad / len(text) > 0.30


def _logits_metrics(ref_lc, quant_lc, topk: int):
    """Compute aggregate logits metrics; return ``(nan_inf, cos, kl, overlap)``."""
    if ref_lc is None or quant_lc is None:
        return False, None, None, None

    ref = getattr(ref_lc, "logits", None)
    quant = getattr(quant_lc, "logits", None)
    if not isinstance(ref, torch.Tensor) or not isinstance(quant, torch.Tensor):
        return False, None, None, None
    if ref.dim() != 2 or quant.dim() != 2:
        return False, None, None, None

    positions = min(ref.shape[0], quant.shape[0])
    vocab = min(ref.shape[1], quant.shape[1])
    if positions <= 0 or vocab <= 0:
        return False, None, None, None
    ref = ref[:positions, :vocab].detach().to("cpu", dtype=torch.float32)
    quant = quant[:positions, :vocab].detach().to("cpu", dtype=torch.float32)
    if not torch.isfinite(ref).all() or not torch.isfinite(quant).all():
        return True, None, None, None

    ref_flat = ref.reshape(-1)
    quant_flat = quant.reshape(-1)
    denom = torch.linalg.vector_norm(ref_flat) * torch.linalg.vector_norm(quant_flat)
    cos = 0.0 if denom.item() <= 1e-12 else float(
        torch.dot(ref_flat, quant_flat).item() / denom.item()
    )

    ref_logp = torch.log_softmax(ref, dim=-1)
    quant_logp = torch.log_softmax(quant, dim=-1)
    ref_prob = torch.softmax(ref, dim=-1)
    kl = float(torch.sum(ref_prob * (ref_logp - quant_logp), dim=-1).mean().item())

    k = max(1, min(int(topk), vocab))
    ref_topk = torch.topk(ref, k=k, dim=-1).indices
    quant_topk = torch.topk(quant, k=k, dim=-1).indices
    overlaps = []
    for row in range(positions):
        left = set(ref_topk[row].tolist())
        right = set(quant_topk[row].tolist())
        overlaps.append(len(left & right) / k)
    overlap = float(sum(overlaps) / len(overlaps))
    return False, cos, kl, overlap


def compare_inference(
    ref_text: str,
    quant_text: str,
    ref_tokens: Sequence[str],
    quant_tokens: Sequence[str],
    *,
    prompt: str = "",
    ref_lc=None,
    quant_lc=None,
    topk: int = 5,
) -> InferenceCompareData:
    """Build a report-ready Ref/Quant generation comparison."""
    ref_tokens = list(ref_tokens or [])
    quant_tokens = list(quant_tokens or [])
    min_len = min(len(ref_tokens), len(quant_tokens))
    first_divergence = next(
        (idx for idx in range(min_len) if ref_tokens[idx] != quant_tokens[idx]),
        None,
    )
    if first_divergence is None and len(ref_tokens) != len(quant_tokens):
        first_divergence = min_len

    max_len = max(len(ref_tokens), len(quant_tokens))
    matches = sum(
        1 for idx in range(min_len) if ref_tokens[idx] == quant_tokens[idx]
    )
    exact = ref_text == quant_text
    match_rate = matches / max_len if max_len else (1.0 if exact else 0.0)
    nan_inf, logits_cos, logits_kl, topk_overlap = _logits_metrics(
        ref_lc, quant_lc, topk
    )

    return InferenceCompareData(
        prompt=prompt,
        ref_output=ref_text,
        quant_output=quant_text,
        ref_tokens=ref_tokens,
        quant_tokens=quant_tokens,
        token_match_rate=match_rate,
        exact_match=exact,
        first_divergence_pos=first_divergence,
        max_new_tokens=max_len,
        ref_garbled=detect_garbled(ref_text),
        quant_garbled=detect_garbled(quant_text),
        ref_repeat=detect_repeat(ref_text, ref_tokens),
        quant_repeat=detect_repeat(quant_text, quant_tokens),
        logits_cos_sim=logits_cos,
        logits_kl=logits_kl,
        topk_overlap=topk_overlap,
        logits_nan_inf=nan_inf,
    )


__all__ = ["compare_inference", "detect_repeat", "detect_garbled"]
