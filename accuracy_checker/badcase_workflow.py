"""
Bad case workflow 基础设施

acc_bench 精度诊断工具的"反事实评测底座": 用户手工对一个已知模块注入量化 bad case,
acc_bench 的 L2 子图诊断去定位, 再与本模块对比"定位到的根因"是否命中 ground truth。

提供:
  - BadCaseManifest: bad case 配置清单 (注入信息 + 预期根因)
  - load_manifest / save_manifest: JSON 读写
  - GroundTruthComparison: acc_bench 定位结果 vs ground truth 的对比
  - compare_with_ground_truth(l2_result, manifest): 主对比逻辑

与 accuracy_checker.subgraph_locate.diagnose_layer 的返回结构对接:
  l2_result dict 含: layer_idx, root_suspect, impact_boundary, subgraphs,
  subgraph_quant_types, subgraph_selfroterr, subgraph_rotberr, chain_deltas ...
  diagnose_layers 返回 List[dict]; 本模块同时兼容单层 dict 与多层 list。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Union

import logging

from .utils import parse_base_name

logger = logging.getLogger(__name__)


# ============================================================================
# Bad case 清单
# ============================================================================

@dataclass
class BadCaseManifest:
    """一次 bad case 评测的配置清单。

    量化工程师手工构造一个已知缺陷 (如把某层 q_proj 的 scale 置错/注入噪声),
    记录于此; acc_bench 跑 L2 诊断后与 injected_* 对比, 评估定位能力。

    Attributes:
        model: 模型名 (如 "glm-5.1-9b")。
        baseline_config: baseline (正确) 量化配置路径或描述。
        badcase_config: bad case (注入缺陷) 量化配置路径或描述。
        injected_layer: 注入缺陷的层 (如 "5" / "layer.5")。
        injected_module: 注入缺陷的模块 (如 "model.layers.5.self_attn.q_proj")。
        injected_change: 注入的变更描述 (如 "scale 置零"、"weight 加噪声")。
        calibration_data: 校准/评测数据来源 (路径或数据集名)。
        seed: 随机种子。
        expected_root_cause: 预期根因 (自由文本, 便于人工核对)。
    """
    model: str = ""
    baseline_config: str = ""
    badcase_config: str = ""
    injected_layer: str = ""
    injected_module: str = ""
    injected_change: str = ""
    calibration_data: str = ""
    seed: int = 0
    expected_root_cause: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_manifest(path: str) -> BadCaseManifest:
    """从 JSON 文件读取 BadCaseManifest。

    宽容: JSON 中缺字段时用 dataclass 默认值; 多余字段忽略。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"manifest JSON must be an object, got {type(data)}: {path}")
    known = {k for k in BadCaseManifest().__dict__}  # dataclass field names
    filtered = {k: v for k, v in data.items() if k in known}
    return BadCaseManifest(**filtered)


def save_manifest(manifest: BadCaseManifest, path: str) -> None:
    """写 BadCaseManifest 到 JSON 文件 (pretty)。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, ensure_ascii=False, indent=2)


# ============================================================================
# Ground truth 对比
# ============================================================================

@dataclass
class GroundTruthComparison:
    """acc_bench L2 定位结果与 bad case ground truth 的对比。

    Attributes:
        problem_path: acc_bench 定位到的问题路径 (impact_boundary + chain + root_suspect)。
        best_repair_point: 最佳修复点 (= root_suspect, 系统认定最值得修的算子)。
        source_candidate: 源候选 (量化误差本身最大的算子, 取 selfroterr argmax)。
        ground_truth: 实际注入的模块 (manifest.injected_module)。
        whether_hit_ground_truth: 定位是否命中 ground truth。
        hit_detail: 命中/未命中详情 (含层级匹配信息)。
    """
    problem_path: str = ""
    best_repair_point: str = ""
    source_candidate: str = ""
    ground_truth: str = ""
    whether_hit_ground_truth: bool = False
    hit_detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# 模块名归一化 (ref/quant/L2 subgraph 名称统一到去前缀的 module 路径)
# ============================================================================

_PREFIX_PATTERNS = [
    re.compile(r"^model\.model\.language_model\."),
    re.compile(r"^model\.model\."),
    re.compile(r"^model\.language_model\."),
    re.compile(r"^model\.layers\.\d+\."),
    re.compile(r"^layers\.\d+\."),
    re.compile(r"^model\."),
]


def normalize_module_name(name: Optional[str]) -> str:
    """把各种 module 命名归一化为去前缀的相对路径, 便于跨来源比对。

    "model.layers.5.self_attn.q_proj.weight_scale" -> "self_attn.q_proj"
    "self_attn.q_proj"                            -> "self_attn.q_proj"
    None                                          -> ""
    """
    if not name:
        return ""
    s = parse_base_name(str(name))
    # 2. 去结构性前缀
    changed = True
    while changed:
        changed = False
        for pat in _PREFIX_PATTERNS:
            new = pat.sub("", s)
            if new != s:
                s = new
                changed = True
                break
    return s


def _module_match(injected: str, candidate: str) -> bool:
    """匹配 injected_module 与某个定位候选 (子串/路径后缀)。"""
    ni = normalize_module_name(injected)
    nc = normalize_module_name(candidate)
    if not ni or not nc:
        return False
    if ni == nc:
        return True
    # 路径后缀包含 (如注入写全 "model.layers.5.self_attn.q_proj", 定位给 "self_attn.q_proj")
    if nc.endswith("." + ni) or ni.endswith("." + nc):
        return True
    # 末尾 token 一致 (如都落在 q_proj, 但父前缀不同) -- 仅在 ni/nc 非空时
    if ni.split(".")[-1] == nc.split(".")[-1] and "." not in ni and "." not in nc:
        return True
    return False


def _try_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def _extract_layer_idx(s) -> Optional[int]:
    """从 'layer.5' / '5' / 'model.layers.5...' / 5(int) 取出层号。"""
    if s is None:
        return None
    if isinstance(s, int):
        return s
    i = _try_int(s)
    if i is not None:
        return i
    n = normalize_module_name(s)  # may strip prefix
    if n:
        m = re.search(r"\d+", n)
        if m:
            return int(m.group())
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else None


# ============================================================================
# 主对比逻辑
# ============================================================================

def _normalize_l2_result(l2_result, manifest):
    """规整 l2_result: list → 选出与 injected_layer 匹配的层; 单 dict 直接用。"""
    if not isinstance(l2_result, list):
        return l2_result if isinstance(l2_result, dict) else {}
    target_idx = _extract_layer_idx(manifest.injected_layer)
    chosen = None
    for r in l2_result:
        if not isinstance(r, dict):
            continue
        li = r.get("layer_idx")
        if target_idx is not None and li == target_idx:
            chosen = r
            break
        if chosen is None:
            chosen = r
    if chosen is None or (target_idx is not None and chosen.get("layer_idx") != target_idx):
        logger.warning(
            f"no L2 result matches injected_layer={manifest.injected_layer}; "
            f"available layer_idx={[r.get('layer_idx') for r in l2_result if isinstance(r, dict)]}"
        )
    return chosen or {}


def _compute_source_candidate(root_suspect, selfroterr, best_repair_point):
    """source_candidate := 量化误差最大的算子 (selfroterr argmax), 兜底 root_suspect。"""
    source_candidate = best_repair_point
    root_sus_has_serr = (
        root_suspect is not None
        and root_suspect in selfroterr
        and selfroterr[root_suspect] is not None
    )
    if selfroterr and root_sus_has_serr:
        try:
            finite_se = {k: v for k, v in selfroterr.items() if v is not None}
            if finite_se:
                source_candidate = max(finite_se, key=lambda k: finite_se[k])
        except Exception as e:  # noqa: BLE001
            logger.debug(f"source_candidate argmax failed: {e}")
    return source_candidate


def _build_problem_path(layer_idx, impact_boundary, root_suspect, chain_deltas):
    """problem_path: impact_boundary + chain + root_suspect 串成可读路径。"""
    parts = []
    if layer_idx is not None:
        parts.append(f"layer={layer_idx}")
    if impact_boundary:
        parts.append(f"impact_boundary={impact_boundary}")
    if root_suspect:
        parts.append(f"root_suspect={root_suspect}")
    if chain_deltas:
        chain_strs = []
        for cname, ops in chain_deltas.items():
            if isinstance(ops, dict) and ops:
                chain_strs.append(f"{cname}:[{','.join(str(o) for o in ops.keys())}]")
        if chain_strs:
            parts.append("chains={" + "; ".join(chain_strs) + "}")
    return " | ".join(parts)


def _build_hit_candidates(best_repair_point, source_candidate, impact_boundary,
                           subgraphs, chain_deltas):
    """收集所有命中候选 (best_repair + source + boundary + subgraph keys + chain ops)。"""
    candidates = [best_repair_point, source_candidate, impact_boundary]
    sg = subgraphs if isinstance(subgraphs, dict) else {}
    for k in sg.keys():
        if k:
            candidates.append(k)
    for ops in chain_deltas.values():
        if isinstance(ops, dict):
            for op in ops.keys():
                if op:
                    candidates.append(op)
    return candidates


def _build_hit_detail(module_hit, ground_truth, candidates, layer_hit,
                       layer_idx, inj_layer):
    """构建命中详情字符串。"""
    detail_bits = []
    if module_hit:
        matched = [c for c in candidates if c and _module_match(ground_truth, c)]
        detail_bits.append(f"module HIT: ground_truth={ground_truth!r} matched candidates={matched}")
    else:
        norm_gt = normalize_module_name(ground_truth)
        norm_cands = [normalize_module_name(c) for c in candidates if c]
        detail_bits.append(
            f"module MISS: ground_truth={ground_truth!r}(norm={norm_gt!r}) "
            f"not among candidates(norm)={norm_cands}"
        )
    if layer_hit is True:
        detail_bits.append(f"layer HIT: located layer={layer_idx} == injected_layer={inj_layer}")
    elif layer_hit is False:
        detail_bits.append(f"layer MISS: located layer={layer_idx} != injected_layer={inj_layer}")
    elif inj_layer is None:
        detail_bits.append("layer check skipped (injected_layer not numeric)")
    return "; ".join(detail_bits)


def compare_with_ground_truth(
    l2_result: Union[Dict[str, Any], List[Dict[str, Any]]],
    manifest: BadCaseManifest,
) -> GroundTruthComparison:
    """将 acc_bench L2 定位结果与 bad case ground truth 对比。"""
    l2 = _normalize_l2_result(l2_result, manifest)
    layer_idx = l2.get("layer_idx")
    root_suspect = l2.get("root_suspect")
    impact_boundary = l2.get("impact_boundary")
    subgraphs = l2.get("subgraphs") or {}
    selfroterr = l2.get("subgraph_selfroterr") or {}
    chain_deltas = l2.get("chain_deltas") or {}

    best_repair_point = root_suspect or ""
    source_candidate = _compute_source_candidate(root_suspect, selfroterr, best_repair_point)
    problem_path = _build_problem_path(layer_idx, impact_boundary, root_suspect, chain_deltas)

    ground_truth = manifest.injected_module or ""
    candidates = _build_hit_candidates(
        best_repair_point, source_candidate, impact_boundary, subgraphs, chain_deltas)
    module_hit = any(_module_match(ground_truth, c) for c in candidates if c)

    inj_layer = _extract_layer_idx(manifest.injected_layer)
    layer_hit = (int(layer_idx) == inj_layer) if (inj_layer is not None and layer_idx is not None) else None

    hit_detail = _build_hit_detail(
        module_hit, ground_truth, candidates, layer_hit, layer_idx, inj_layer)

    return GroundTruthComparison(
        problem_path=problem_path,
        best_repair_point=best_repair_point,
        source_candidate=source_candidate,
        ground_truth=ground_truth,
        whether_hit_ground_truth=bool(module_hit),
        hit_detail=hit_detail,
    )


__all__ = [
    "BadCaseManifest",
    "load_manifest",
    "save_manifest",
    "GroundTruthComparison",
    "compare_with_ground_truth",
    "normalize_module_name",
]
