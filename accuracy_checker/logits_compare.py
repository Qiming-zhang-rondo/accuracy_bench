"""
Logits 采集与对比

提供两类能力::

    collect_logits(model, tokenizer, prompt, device, max_new_tokens)
        -> LogitsCollection   # 生成 max_new_tokens 个位置, 采集每步全词表 logits

    compare_logits(ref, quant, tokenizer, top_k, scatter_sample, hist_bins)
        -> LogitsComparison    # top-k prob / scatter / token-wise metrics / 直方图

产出 4 类可视化数据:
  A. ref/quant top-k token probability (按 position 并排柱)
  B. ref vs quant 全词表 logits 散点 (采样, y=x 参考线)
  C. token-wise cos / KL / topk-overlap / top1-match (折线)
  D. ref/quant logits 分布直方图overlay

LogitsComparison.to_logits_data() 把结果转成 ``report_schema.LogitsData``, 供 HTML 消费。

数值安全:
  * 所有 logits 在 CPU fp32 上算 (collect 时搬回 CPU, 避免 NPU 内存堆积)。
  * KL 用 softmax 后的概率分布算, 屏蔽数值下溢。
  * scatter 采样上限 default 2000 点, 防止 50k 词表把 HTML 拖垮。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import re

import torch

from .report_schema import LogitsData, TokenProb

# 匹配 Unicode 不可见字符 (零宽空格 \u200b, 零宽连字 \u200c/d, BOM \ufeff, 等控制字符)
_INVISIBLE_RE = re.compile(r'[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\x00-\x1f\x7f]')


def _clean_token_str(raw: str, tid: int = 0) -> str:
    """清除不可见字符, 空格替换为 ·, 返回可见标签或 #token_id 兜底."""
    cleaned = _INVISIBLE_RE.sub('', raw).replace(" ", "·").strip()
    return cleaned or f"#{tid}"


# ---------------------------------------------------------------------------
# 采集
# ---------------------------------------------------------------------------

@dataclass
class LogitsCollection:
    """单模型 forward 采集结果。logits: [num_positions, vocab_size] (CPU fp32)"""
    token_positions: List[int]
    logits: torch.Tensor            # [num_positions, vocab_size]
    input_ids: Optional[torch.Tensor] = None  # 完整 prompt+生成 ids
    # generation: each row is one autoregressive decode step.
    # prompt_prefill: rows are teacher-forced predictions inside the prompt;
    #                 the final row predicts the first decode token.
    position_mode: str = "generation"

    def to(self, *args, **kwargs):  # noqa: D401 - 便捷搬运
        return LogitsCollection(
            token_positions=list(self.token_positions),
            logits=self.logits.to(*args, **kwargs),
            input_ids=None if self.input_ids is None else self.input_ids.to(*args, **kwargs),
            position_mode=self.position_mode,
        )

    @property
    def num_positions(self) -> int:
        return int(self.logits.shape[0]) if self.logits.dim() >= 2 else 0


def _ensure_device(model, device: str) -> torch.device:
    """优先用模型当前 param device, fallback 给定 device。"""
    try:
        p = next(model.parameters())
        return p.device
    except (StopIteration, Exception):
        return torch.device(device)


def collect_logits(model, tokenizer, prompt: str, device: str = "cpu",
                   max_new_tokens: int = 32,
                   input_ids: Optional[torch.Tensor] = None) -> LogitsCollection:
    """对最后一个 token 做 forward, 连续生成 max_new_tokens 步, 收集每步全词表 logits。

    Args:
        model: HF causal LM (已 eval, 已在 target device)
        tokenizer: 对应 tokenizer (decode 用)
        prompt: 提示词
        device: 'cpu' / 'npu:0' / ...
        max_new_tokens: 采集多少个生成位置
        input_ids: 显式给 token ids (跳过 tokenize)

    Returns:
        LogitsCollection: logits [max_new_tokens, vocab_size] on CPU fp32。
    """
    model.eval()
    dev = _ensure_device(model, device)

    if input_ids is None:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(dev)
    generated = input_ids
    all_logits: List[torch.Tensor] = []
    positions: List[int] = []

    with torch.no_grad():
        past_kv = None
        for step in range(max_new_tokens):
            # Use KV cache: first step processes full prompt, subsequent steps
            # only process the new token (10x+ speedup for streaming expert MoE)
            if past_kv is not None:
                inp = generated[:, -1:]
                out = model(inp, past_key_values=past_kv, use_cache=True)
            else:
                out = model(generated, use_cache=True)
            past_kv = out.past_key_values
            lm_logits = out.logits if hasattr(out, "logits") else out[0]
            # (1, seq, vocab) -> 末尾 token 的 logits
            last_logits = lm_logits[0, -1, :].detach().to("cpu", dtype=torch.float32)
            all_logits.append(last_logits)
            positions.append(step)
            # greedy argmax 续写, 保持确定性
            next_tok = int(torch.argmax(lm_logits[0, -1, :]).item())
            next_tok_t = torch.tensor([[next_tok]], device=dev, dtype=generated.dtype)
            generated = torch.cat([generated, next_tok_t], dim=-1)

    logits_tensor = torch.stack(all_logits, dim=0)  # [N, vocab]
    return LogitsCollection(
        token_positions=positions,
        logits=logits_tensor,
        input_ids=generated.detach().to("cpu"),
    )


def collect_last_logits(model, tokenizer, prompt: str, device: str = "cpu",
                        input_ids: Optional[torch.Tensor] = None) -> LogitsCollection:
    """单步 forward, 仅取最后一个 token 的全词表 logits (无续写)。"""
    return collect_logits(model, tokenizer, prompt, device, max_new_tokens=1,
                          input_ids=input_ids)


# ---------------------------------------------------------------------------
# 对比
# ---------------------------------------------------------------------------

@dataclass
class LogitsComparison:
    """ref vs quant logits 对比结果 (含 4 类可视化数据)"""
    token_positions: List[int] = field(default_factory=list)
    position_mode: str = "unknown"
    ref_topk: List[List[TokenProb]] = field(default_factory=list)
    quant_topk: List[List[TokenProb]] = field(default_factory=list)
    ref_argmax_logits: List[float] = field(default_factory=list)       # 每 position ref argmax logit
    quant_argmax_logits: List[float] = field(default_factory=list)
    token_wise_cos: List[Optional[float]] = field(default_factory=list)
    token_wise_kl: List[Optional[float]] = field(default_factory=list)        # KL(quant || ref)
    token_wise_topk_overlap: List[Optional[float]] = field(default_factory=list)
    token_wise_top1_match: List[bool] = field(default_factory=list)
    scatter_ref: List[float] = field(default_factory=list)             # 采样成对样本
    scatter_quant: List[float] = field(default_factory=list)
    hist_bins: List[float] = field(default_factory=list)
    hist_ref_counts: List[int] = field(default_factory=list)
    hist_quant_counts: List[int] = field(default_factory=list)

    def to_logits_data(self) -> LogitsData:
        """转成 report_schema.LogitsData (供 HTML 报告直接消费)。"""
        return LogitsData(
            token_positions=list(self.token_positions),
            position_mode=self.position_mode,
            ref_topk=[list(pos) for pos in self.ref_topk],
            quant_topk=[list(pos) for pos in self.quant_topk],
            ref_logits=list(self.ref_argmax_logits),
            quant_logits=list(self.quant_argmax_logits),
            token_wise_cos=list(self.token_wise_cos),
            token_wise_kl=list(self.token_wise_kl),
            token_wise_topk_overlap=list(self.token_wise_topk_overlap),
            token_wise_top1_match=list(self.token_wise_top1_match),
            scatter_ref=list(self.scatter_ref),
            scatter_quant=list(self.scatter_quant),
            hist_bins=list(self.hist_bins),
            hist_ref_counts=list(self.hist_ref_counts),
            hist_quant_counts=list(self.hist_quant_counts),
        )


def _topk_prob(logit_row: torch.Tensor, k: int) -> Tuple[List[int], torch.Tensor]:
    p = torch.softmax(logit_row, dim=-1)
    topk_p, topk_i = torch.topk(p, k=k, dim=-1)
    return topk_i.tolist(), topk_p


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na = torch.norm(a)
    nb = torch.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))


def _kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(p || q) 对 logits, p/q 都是 [vocab] logits。"""
    p = torch.log_softmax(p_logits, dim=-1)
    q = torch.log_softmax(q_logits, dim=-1)
    # KL = sum(p_prob * (log_p - log_q))
    p_prob = torch.softmax(p_logits, dim=-1)
    return float(torch.sum(p_prob * (p - q)).item())


def _histogram(ref_flat: torch.Tensor, quant_flat: torch.Tensor,
               bins: int) -> Tuple[List[float], List[int], List[int]]:
    lo = float(min(ref_flat.min().item(), quant_flat.min().item()))
    hi = float(max(ref_flat.max().item(), quant_flat.max().item()))
    if hi - lo < 1e-9:
        hi = lo + 1.0
    edges = torch.linspace(lo, hi, steps=bins + 1)
    r_counts = torch.histc(ref_flat, bins=bins, min=lo, max=hi)
    q_counts = torch.histc(quant_flat, bins=bins, min=lo, max=hi)
    return edges.tolist(), r_counts.tolist(), q_counts.tolist()


def compare_logits(ref: LogitsCollection, quant: LogitsCollection,
                   tokenizer,
                   top_k: int = 10,
                   scatter_sample: int = 2000,
                   hist_bins: int = 40) -> LogitsComparison:
    """对比 ref 与 quant 的 logits, 产出 4 类可视化数据。

    Args:
        ref/quant: :class:`LogitsCollection` (位置数应一致; 不一致按较小者取)
        tokenizer: 把 token_id 解码成 token_str
        top_k: 每 position 取前 k 个 token 并排
        scatter_sample: 全词表 logits 散点采样上限 (随机不放回)
        hist_bins: 直方图分箱数
    """
    ref_logits = ref.logits.to(torch.float32)
    quant_logits = quant.logits.to(torch.float32)
    n = min(ref_logits.shape[0], quant_logits.shape[0])
    vocab = min(ref_logits.shape[1], quant_logits.shape[1])
    # Keep the requested generation Top-K, but never ask torch.topk for more
    # entries than the common vocabulary actually contains.
    k = max(1, min(int(top_k), vocab))
    ref_logits = ref_logits[:n, :vocab]
    quant_logits = quant_logits[:n, :vocab]

    ref_positions = list(ref.token_positions[:n])
    quant_positions = list(quant.token_positions[:n])
    positions: List[int] = (
        ref_positions
        if len(ref_positions) == n and ref_positions == quant_positions
        else list(range(n))
    )
    position_mode = (
        ref.position_mode
        if ref.position_mode == quant.position_mode
        else "unknown"
    )
    ref_topk: List[List[TokenProb]] = []
    quant_topk: List[List[TokenProb]] = []
    ref_argmax: List[float] = []
    quant_argmax: List[float] = []
    t_cos: List[Optional[float]] = []
    t_kl: List[Optional[float]] = []
    t_overlap: List[Optional[float]] = []
    t_top1: List[bool] = []

    for i in range(n):
        r_row = ref_logits[i]
        q_row = quant_logits[i]
        r_ids, r_p = _topk_prob(r_row, k)
        q_ids, q_p = _topk_prob(q_row, k)
        r_id_to_p = {tid: float(p) for tid, p in zip(r_ids, r_p)}
        q_id_to_p = {tid: float(p) for tid, p in zip(q_ids, q_p)}
        all_ids = list(dict.fromkeys(r_ids + q_ids))  # 保留顺序去重

        r_list, q_list = [], []
        for tid in all_ids:
            ts = ""
            try:
                ts = _clean_token_str(tokenizer.decode([tid]), tid)
            except Exception:
                ts = f"#{tid}"
            r_list.append(TokenProb(token_id=tid, token_str=ts,
                                    ref_prob=r_id_to_p.get(tid),
                                    quant_prob=None))
            q_list.append(TokenProb(token_id=tid, token_str=ts,
                                    ref_prob=None,
                                    quant_prob=q_id_to_p.get(tid)))
        ref_topk.append(r_list)
        quant_topk.append(q_list)

        ref_argmax.append(float(r_row.max().item()))
        quant_argmax.append(float(q_row.max().item()))
        t_cos.append(_cos(r_row, q_row))
        t_kl.append(_kl_divergence(r_row, q_row))

        overlap = len(set(r_ids) & set(q_ids))
        t_overlap.append(overlap / k if k else None)
        t_top1.append(bool(r_ids[0] == q_ids[0]) if r_ids and q_ids else False)

    # 散点: 全词表 logits 成对采样 (避免 50k 点拖垮 HTML)
    ref_flat = ref_logits.flatten()
    quant_flat = quant_logits.flatten()
    total = ref_flat.numel()
    if total > scatter_sample:
        idx = torch.randperm(total)[:scatter_sample]
        sr = ref_flat[idx]
        sq = quant_flat[idx]
    else:
        sr = ref_flat
        sq = quant_flat
    bins, r_counts, q_counts = _histogram(ref_flat, quant_flat, hist_bins)

    return LogitsComparison(
        token_positions=positions,
        position_mode=position_mode,
        ref_topk=ref_topk,
        quant_topk=quant_topk,
        ref_argmax_logits=ref_argmax,
        quant_argmax_logits=quant_argmax,
        token_wise_cos=t_cos,
        token_wise_kl=t_kl,
        token_wise_topk_overlap=t_overlap,
        token_wise_top1_match=t_top1,
        scatter_ref=sr.tolist(),
        scatter_quant=sq.tolist(),
        hist_bins=bins,
        hist_ref_counts=r_counts,
        hist_quant_counts=q_counts,
    )


# ---------------------------------------------------------------------------
# 便捷: 从两个模型一次跑完
# ---------------------------------------------------------------------------

def run_logits_compare(ref_model, ref_tokenizer, quant_model, quant_tokenizer,
                       prompt: str, device: str = "cpu",
                       max_new_tokens: int = 32, top_k: int = 10) -> LogitsData:
    """一键采集 ref/quant 并对比, 直接返回 schema LogitsData。"""
    ref_c = collect_logits(ref_model, ref_tokenizer, prompt, device, max_new_tokens)
    quant_c = collect_logits(quant_model, quant_tokenizer, prompt, device, max_new_tokens)
    return compare_logits(ref_c, quant_c, ref_tokenizer, top_k=top_k).to_logits_data()


__all__ = [
    "LogitsCollection", "LogitsComparison",
    "collect_logits", "collect_last_logits", "compare_logits", "run_logits_compare",
]
