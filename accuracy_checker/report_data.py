"""
报告数据组装

把各模块的原始结果统一收敛到 ``report_schema.ReportData`` ::

    assemble_report(
        l1_report=None,            # BlockCompareReport
        l2_results=None,           # List[diagnose_layer dict]
        boundary_result=None,      # List[boundary inference dict]  (旧名 boundary_results)
        logits_comparison=None,    # LogitsComparison 或 LogitsData
        inference_compare_data=None, # InferenceCompareData 或 dict
        model_name="",
        ref_model_path="", quant_model_path="",
        quant_format="", device_mode="", prompt="",
    ) -> ReportData

字段映射 (per-field 真实结构 -> schema):
  * L1: BlockCompareResult.metrics (cos_sim/snr/relative_error) -> L1LayerData
  * L2: diagnose_layer dict 的 per-field 字典 (subgraphs/subgraph_selfroterr/
        subgraph_rotberr/subgraph_quant_types/chain_deltas/indexer_flip_rate/
        experts_routing_flip) -> 合并到 L2LayerData.subgraphs: List[SubgraphData]
  * boundary: 重复检测启发式 -> OverviewData.boundary_result (CLEAN/GARBLED/TRUNCATED)

None/NaN 安全, 不存在的模块留空不影响出报告 (run_status 标 PARTIAL)。
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from . import report_schema as S
from .report_schema import (
    InferenceCompareData, L1LayerData, L2LayerData, LogitsData, OverviewData,
    ReportData, SubgraphData, SUCCESS, PARTIAL, INCONCLUSIVE, INVALID_RUN,
)

logger = logging.getLogger(__name__)

# 复用 html_report 的重复检测 (boundary 定界核心启发式)
from .html_report import _detect_repetition  # noqa: E402  (避免循环: html_report 不 import 本模块)

_BOUNDARY_CLEAN = "CLEAN"
_BOUNDARY_GARBLED = "GARBLED"
_BOUNDARY_TRUNCATED = "TRUNCATED"


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _layer_idx_from_name(name: str, default: int = -1) -> int:
    """'layer.4' / 'layers.4.input' / 'block.4' -> 4。失败 -> default。"""
    if not name:
        return default
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else default


def _classify_boundary(results: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """对 boundary 推理结果做重复检测, 返回 (verdict, prompt)。"""
    if not results:
        return None, None
    prompt = ""
    for r in results:
        for m in (r.get("messages") or []):
            if m.get("role") == "user":
                prompt = m.get("content", "") or ""
                break
    # 思维链截断也算异常信号
    any_trunc = any(r.get("thinking_truncated") for r in results)
    garbled = 0
    for r in results:
        gen = r.get("generated") or ""
        sus, _ = _detect_repetition(gen)
        if sus:
            garbled += 1
    total = len(results)
    if any_trunc and total:
        return _BOUNDARY_TRUNCATED, prompt
    if garbled and garbled >= max(1, total // 2):
        return _BOUNDARY_GARBLED, prompt
    if garbled:
        return _BOUNDARY_GARBLED, prompt  # 少量乱码也判定为 GARBLED (人工复核)
    return _BOUNDARY_CLEAN, prompt


# ---------------------------------------------------------------------------
# L1 映射
# ---------------------------------------------------------------------------

def _build_delta_map(l1_report) -> Dict[str, Any]:
    """从 l1_report._detection_cache 构建 layer_name -> LayerDeltaInfo 映射。"""
    detection = getattr(l1_report, "_detection_cache", None)
    if detection is None:
        return {}
    return {info.layer_name: info for info in getattr(detection, "layer_deltas", [])}


def _find_min_cos_layer(results) -> Optional[str]:
    """返回 cos_sim 最低的层名 (全 None 时返回 None)。"""
    min_cos = None
    min_cos_name: Optional[str] = None
    for r in results:
        cs = _safe_float((getattr(r, "metrics", {}) or {}).get("cos_sim"))
        if cs is not None and (min_cos is None or cs < min_cos):
            min_cos = cs
            min_cos_name = getattr(r, "layer_name", "")
    return min_cos_name


def _build_l1_layer_item(r, delta_map, first_bad_name, min_cos_name) -> L1LayerData:
    """构建单个 L1LayerData。"""
    name = getattr(r, "layer_name", "") or ""
    m = getattr(r, "metrics", {}) or {}
    cos = _safe_float(m.get("cos_sim"))
    di = delta_map.get(name)
    return L1LayerData(
        layer_idx=_layer_idx_from_name(name),
        layer_name=name,
        cos_sim=cos,
        rel_l2=_safe_float(m.get("relative_error")),
        snr=_safe_float(m.get("snr")),
        delta_cos=_safe_float(di.delta_cos) if di else None,
        drop_percent=_safe_float(di.drop_percent) if di else None,
        is_first_divergence=(name == first_bad_name) if first_bad_name else False,
        is_max_error=(name == min_cos_name) if min_cos_name else False,
    )


def _build_l1_layers(l1_report) -> Tuple[List[L1LayerData], Optional[int], Optional[int]]:
    """BlockCompareReport -> (List[L1LayerData], first_divergence_idx, first_threshold_idx)。

    first_divergence: delta 检测的显著突降层 (根因定位)
    first_threshold: cos_sim < 0.99 首次跨过 (辅助告警)
    """
    if l1_report is None:
        return [], None, None
    results = getattr(l1_report, "results", None) or []
    if not results:
        return [], None, None

    first_bad_name = getattr(l1_report, "first_bad_block", None)
    first_bad_idx = _layer_idx_from_name(first_bad_name) if first_bad_name else None

    first_threshold_name = getattr(l1_report, "first_threshold_crossing", None)
    first_threshold_idx = _layer_idx_from_name(first_threshold_name) if first_threshold_name else None

    delta_map = _build_delta_map(l1_report)
    min_cos_name = _find_min_cos_layer(results)

    layers: List[L1LayerData] = [
        _build_l1_layer_item(r, delta_map, first_bad_name, min_cos_name)
        for r in results
    ]
    return layers, first_bad_idx, first_threshold_idx


# ---------------------------------------------------------------------------
# L2 映射
# ---------------------------------------------------------------------------

def _build_subgraphs(raw: Dict[str, Any]) -> List[SubgraphData]:
    """合并 diagnose_layer 的 per-field 字典 -> List[SubgraphData]。

    支持两种输入:
      1. 原始 diagnose_layer dict: subgraphs={name:val}, subgraph_selfroterr={name:val}, ...
      2. 已序列化的 dict (from_dict round-trip): subgraphs=[{name,patch_recovery,...}, ...]
    """
    sg_raw = raw.get("subgraphs")
    impact_bnd = raw.get("impact_boundary")
    root_sus = raw.get("root_suspect")

    # Case 2: already a list of SubgraphData dicts (round-trip from to_dict)
    if isinstance(sg_raw, list):
        out: List[SubgraphData] = []
        for item in sg_raw:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "")
            out.append(SubgraphData(
                name=name,
                quant_type=str(item.get("quant_type", "")),
                patch_recovery=_safe_float(item.get("patch_recovery")),
                self_rot_err=_safe_float(item.get("self_rot_err")),
                rot_b_err=_safe_float(item.get("rot_b_err")),
                flip_rate=_safe_float(item.get("flip_rate")),
                is_repair_point=bool(item.get("is_repair_point", False)),
                is_source_candidate=bool(item.get("is_source_candidate", False)),
            ))
        return out

    # Case 1: original per-field dicts
    recoveries = sg_raw if isinstance(sg_raw, dict) else {}
    selfroterr = raw.get("subgraph_selfroterr") or {}
    rotberr = raw.get("subgraph_rotberr") or {}
    qtypes = raw.get("subgraph_quant_types") or {}

    out = []
    seen = list(dict.fromkeys(
        list(recoveries.keys()) + list(selfroterr.keys()) + list(rotberr.keys())
    ))
    for name in seen:
        out.append(SubgraphData(
            name=name,
            quant_type=str(qtypes.get(name, "")),
            patch_recovery=_safe_float(recoveries.get(name)),
            self_rot_err=_safe_float(selfroterr.get(name)),
            rot_b_err=_safe_float(rotberr.get(name)),
            flip_rate=None,
            is_repair_point=(name == impact_bnd) if impact_bnd else False,
            is_source_candidate=(name == root_sus) if root_sus else False,
        ))
    return out


def _build_l2_layer(raw: Dict[str, Any]) -> L2LayerData:
    idx = raw.get("layer_idx")
    try:
        idx = int(idx) if idx is not None else -1
    except (TypeError, ValueError):
        idx = -1
    flip_rates: Optional[Dict[str, Any]] = None
    idx_flip = raw.get("indexer_flip_rate")
    exp_flip = raw.get("experts_routing_flip")
    if idx_flip is not None or exp_flip is not None:
        flip_rates = {"indexer": _safe_float(idx_flip), "experts": exp_flip}
    return L2LayerData(
        layer_idx=idx,
        base_l2=_safe_float(raw.get("baseline_l2")),
        input_recovery=_safe_float(raw.get("input_recovery")),
        subgraphs=_build_subgraphs(raw),
        impact_boundary=raw.get("impact_boundary"),
        root_suspect=raw.get("root_suspect"),
        chain_delta=raw.get("chain_deltas") or raw.get("chain_delta"),
        flip_rates=flip_rates,
    )


def _build_l2_layers(l2_results) -> List[L2LayerData]:
    if not l2_results:
        return []
    out: List[L2LayerData] = []
    for raw in l2_results:
        if raw is None:
            continue
        try:
            out.append(_build_l2_layer(raw))
        except Exception as e:  # 单层失败不致命
            logger.warning("L2 layer 映射失败: %s", e)
    return out


# ---------------------------------------------------------------------------
# overview 汇总
# ---------------------------------------------------------------------------

def _confidence(l1_layers, l2_layers, fd_idx) -> Optional[float]:
    """简单可信度: 0.9 (L2 命中 source+repair), 0.6 (仅命中一项), 0.2 (都没命中), None (无数据)。"""
    if fd_idx is None:
        return None
    target = next((l for l in l2_layers if l.layer_idx == fd_idx), None)
    if target is None:
        return 0.2
    has_src = bool(target.root_suspect)
    has_rep = bool(target.impact_boundary)
    if has_src and has_rep:
        return 0.9
    if has_src or has_rep:
        return 0.6
    return 0.2


def _problem_path(fd_idx, l2_target: Optional[L2LayerData]) -> Optional[str]:
    if fd_idx is None:
        return None
    if l2_target is None:
        return f"layer {fd_idx} 发散 (无 L2 诊断)"
    parts = [f"layer {fd_idx}"]
    if l2_target.root_suspect:
        parts.append(f"-> {l2_target.root_suspect}")
    elif l2_target.impact_boundary:
        parts.append(f"-> boundary {l2_target.impact_boundary}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# run_status 判定
# ---------------------------------------------------------------------------

def _decide_status(l1_layers, l2_layers, boundary_verdict,
                   first_div_idx, has_logits, has_inference_compare) -> str:
    has_l1 = bool(l1_layers)
    has_l2 = bool(l2_layers)
    has_bnd = boundary_verdict is not None
    # INVALID_RUN: L1 全是 None cos_sim (forward 坏)
    if has_l1 and all(l.cos_sim is None for l in l1_layers):
        return INVALID_RUN
    # INCONCLUSIVE: L1 全对齐却 boundary 乱码 或 boundary clean 但 L1 明显发散
    l1_aligned = has_l1 and all(
        (l.cos_sim is not None and l.cos_sim > 0.99) for l in l1_layers
    )
    if l1_aligned and boundary_verdict == _BOUNDARY_GARBLED:
        return INCONCLUSIVE
    if has_l1 and not l1_aligned and boundary_verdict == _BOUNDARY_CLEAN:
        # L1 发散但生成通顺 - 推理框架层误差, 仍可信但标 INCONCLUSIVE 让人工复核
        return INCONCLUSIVE
    if has_l1 and has_l2 and has_bnd:
        return SUCCESS
    return PARTIAL


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def _coerce_logits(obj) -> Optional[LogitsData]:
    if obj is None:
        return None
    if isinstance(obj, LogitsData):
        return obj
    # LogitsComparison 有 to_logits_data
    if hasattr(obj, "to_logits_data"):
        return obj.to_logits_data()
    if isinstance(obj, dict):
        return S._build_logits(obj)
    return None


def _coerce_inference_compare(obj) -> Optional[InferenceCompareData]:
    if obj is None:
        return None
    if isinstance(obj, InferenceCompareData):
        return obj
    if isinstance(obj, dict):
        return InferenceCompareData(
            prompt=obj.get("prompt", ""),
            ref_output=obj.get("ref_output", ""),
            quant_output=obj.get("quant_output", ""),
            ref_tokens=list(obj.get("ref_tokens") or []),
            quant_tokens=list(obj.get("quant_tokens") or []),
            token_match_rate=_safe_float(obj.get("token_match_rate")),
            exact_match=bool(obj.get("exact_match", False)),
            first_divergence_pos=obj.get("first_divergence_pos"),
            max_new_tokens=int(obj.get("max_new_tokens", 0)),
        )
    return None


def assemble_report(
    l1_report=None,
    l2_results=None,
    boundary_result=None,
    logits_comparison=None,
    inference_compare_data=None,
    model_name: str = "",
    ref_model_path: str = "",
    quant_model_path: str = "",
    quant_format: str = "",
    device_mode: str = "",
    prompt: str = "",
) -> ReportData:
    """从各模块收集结果, 组装成统一 ReportData JSON。

    参数都 Optional, 缺失模块留空, ``run_status`` 自动标 PARTIAL/INCONCLUSIVE。
    ``boundary_result`` 兼容旧名 boundary_results (list[dict])。
    """
    # --- L1 ---
    l1_layers, first_div_idx, first_threshold_idx = _build_l1_layers(l1_report)

    # --- L2 ---
    l2_layers = _build_l2_layers(l2_results)

    # --- boundary ---
    boundary_verdict, bnd_prompt = _classify_boundary(boundary_result or [])
    use_prompt = prompt or bnd_prompt or ""

    # --- overview ---
    l2_at_fd = next((l for l in l2_layers if first_div_idx is not None and l.layer_idx == first_div_idx), None)
    confidence = _confidence(l1_layers, l2_layers, first_div_idx)
    best_repair = l2_at_fd.impact_boundary if l2_at_fd else None
    source_cand = l2_at_fd.root_suspect if l2_at_fd else None
    problem_path = _problem_path(first_div_idx, l2_at_fd)

    # --- inference compare ---
    inference_compare = _coerce_inference_compare(inference_compare_data)

    overview = OverviewData(
        model_name=model_name,
        ref_model_path=ref_model_path,
        quant_model_path=quant_model_path,
        quant_format=quant_format,
        device_mode=device_mode,
        prompt=use_prompt,
        boundary_result=boundary_verdict,
        first_divergence_layer=first_div_idx,
        first_threshold_crossing_layer=first_threshold_idx,
        problem_path=problem_path,
        best_repair_point=best_repair,
        source_candidate=source_cand,
        confidence=confidence,
    )

    # --- logits ---
    # 优先用调用方显式传入的 logits_comparison; 若没有, 但 L1 报告带了 logits_data
    # (ShardedBlockComparator._collect_full_logits 在同一 forward 采集), 自动复用 —
    # 这样 L1 跑完一次 forward, 报告就直接带 logits 4 面板数据, 无需独立模型加载。
    logits = _coerce_logits(logits_comparison)
    if logits is None and l1_report is not None:
        logits = _coerce_logits(getattr(l1_report, "logits_data", None))

    # --- status ---
    status = _decide_status(
        l1_layers, l2_layers, boundary_verdict, first_div_idx,
        logits is not None, inference_compare is not None,
    )

    return ReportData(
        overview=overview,
        l1_layers=l1_layers,
        l2_results=l2_layers,
        logits=logits,
        inference_compare=inference_compare,
        run_status=status,
    )


__all__ = ["assemble_report"]
