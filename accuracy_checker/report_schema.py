"""
统一报告数据结构 (schema)

把 L1 逐层对比 / L2 subgraph 诊断 / boundary 定界 / logits 对比 / badcase
统一收敛到一份 ``ReportData`` JSON，供 HTML 报告消费。

数据流::

    raw diagnose_layer dict  ─┐
    BlockCompareReport (L1) ─┤
    boundary results        ─┼─► assemble_report() ─► ReportData ─► generate_product_html_report()
    LogitsComparison        ─┤                              │
    BadCaseData             ─┘                              ▼
                                                  自包含 HTML 报告

设计原则:
  * 所有字段 Optional —— 单模块可用即出报告 (run_status 区分完整度)。
  * NaN/Inf → None —— JSON 标准不允许 NaN，统一在 to_dict 归一。
  * 字段名与 HTML 渲染一一对应，避免再做二次映射。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 运行状态机
# ---------------------------------------------------------------------------

SUCCESS = "SUCCESS"                  # 完整跑完：boundary + L1 + L2 全有
PARTIAL = "PARTIAL"                  # 跑完部分模块 (L2 缺失 / boundary 缺失 等)
INVALID_RUN = "INVALID_RUN"          # 输入无效 (模型加载失败/forward 报错/全 NaN)
INCONCLUSIVE = "INCONCLUSIVE"        # 跑完但结论不可信 (L1 全对齐却 boundary 乱码)
FAILED = "FAILED"                    # 流程异常中断

VALID_STATUSES = {SUCCESS, PARTIAL, INVALID_RUN, INCONCLUSIVE, FAILED}


def _clean_num(v: Any) -> Any:
    """NaN/Inf → None；其余原样返回。递归穿透 list/dict。"""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    if isinstance(v, list):
        return [_clean_num(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean_num(x) for k, x in v.items()}
    return v


# ---------------------------------------------------------------------------
# 叶子结构
# ---------------------------------------------------------------------------

@dataclass
class OverviewData:
    """一屏给结论：模型/量化格式/设备/定界结论/首个发散层/最佳修复点/源候选"""
    model_name: str = ""
    ref_model_path: str = ""
    quant_model_path: str = ""
    quant_format: str = ""                 # W8A8 / W4A8_MXFP4 / ...
    device_mode: str = ""                  # NPU / CPU / NPUx4 ...
    prompt: str = ""                        # 触发本报告的 prompt
    boundary_result: Optional[str] = None   # CLEAN / GARBLED / TRUNCATED / None=未做定界
    first_divergence_layer: Optional[int] = None            # delta 检测: 首个显著突降层
    first_threshold_crossing_layer: Optional[int] = None     # cos_sim < 0.99 首次跨过 (辅助)
    problem_path: Optional[str] = None      # 从首个发散层到根因的链路描述
    best_repair_point: Optional[str] = None # = L2 impact_boundary (关键边界)
    source_candidate: Optional[str] = None  # = L2 root_suspect (主要嫌疑)
    confidence: Optional[float] = None      # 0-1, 定位结论的可信度
    ground_truth_hit: Optional[bool] = None # badcase 注入模块 == 工具定位模块?
    # Appended after historical fields to keep positional construction stable.
    input_mode: str = ""                    # prompt / messages (chat template)
    # L1 comparison semantics.  ``unknown`` keeps old report JSON honest:
    # historical artifacts did not record whether activation QDQ was enabled.
    comparison_scope: str = "unknown"       # weight_only / weight_plus_activation_qdq / unknown
    quant_method: str = ""                  # dequantize / fake_quant / ...
    activation_quant_enabled: Optional[bool] = None
    activation_quant_type: str = ""          # AUTO / W4A4_DYNAMIC / ...
    activation_quant_backend: str = ""       # auto / npu / torch
    activation_quant_group_size: Optional[int] = None


@dataclass
class L1LayerData:
    """单层 L1 对比指标 (cos_sim 粗筛、断崖)"""
    layer_idx: int
    layer_name: str = ""
    cos_sim: Optional[float] = None
    rel_l2: Optional[float] = None
    snr: Optional[float] = None
    delta_cos: Optional[float] = None         # cos_sim[i] - cos_sim[i-1], 负=下降
    drop_percent: Optional[float] = None      # delta_cos * 100
    is_first_divergence: bool = False          # 该层 == L1 first_bad_block (delta 检测)
    is_max_error: bool = False                # 该层 cos_sim 最低 (相对误差最大)


@dataclass
class SubgraphData:
    """单子图：patch_recovery / self_rot_err / rot_b_err / flip_rate 合并视图"""
    name: str
    quant_type: str = ""                  # FLOAT / INT8 / MXFP4 / UNPATCHABLE / UNKNOWN
    patch_recovery: Optional[float] = None  # results[name]: (base-patched)/base, 高=好；负=耦合
    self_rot_err: Optional[float] = None    # subgraph_selfroterr: 自身量化误差
    rot_b_err: Optional[float] = None       # subgraph_rotberr: 含上游的边界误差
    flip_rate: Optional[float] = None       # 离散选择 (indexer/gate) top-k 翻转率
    is_repair_point: bool = False         # name == impact_boundary (绿框)
    is_source_candidate: bool = False     # name == root_suspect (红框)


@dataclass
class L2LayerData:
    """单层 L2 反事实诊断 (subgraph 级定位)"""
    layer_idx: int
    base_l2: Optional[float] = None            # 整层量化相对误差 rel_l2(quant,ref)
    input_recovery: Optional[float] = None      # 上游输入 patch 恢复率
    subgraphs: List[SubgraphData] = field(default_factory=list)
    impact_boundary: Optional[str] = None       # = 关键边界 best_repair_point
    root_suspect: Optional[str] = None          # = 主要嫌疑 source_candidate
    chain_delta: Optional[Dict[str, Any]] = None  # 串联子链增量 {chain: {op: delta}}
    flip_rates: Optional[Dict[str, Any]] = None   # {indexer: float, experts: {...}}


@dataclass
class TokenProb:
    """某 position 下一个 token 的概率对照 (ref/quant 并排)"""
    token_id: int
    token_str: str
    ref_prob: Optional[float] = None
    quant_prob: Optional[float] = None


@dataclass
class LogitsData:
    """生成序列逐位置 logits 对比 (驱动 4 类可视化: topk 并排柱 / scatter / token-wise 折线 / 直方图)"""
    token_positions: List[int] = field(default_factory=list)
    # generation: autoregressive decode steps; prompt_prefill: prompt token
    # positions whose final row predicts the first generated token.
    position_mode: str = "unknown"
    ref_topk: List[List[TokenProb]] = field(default_factory=list)   # 每 position top-k (并排柱状图原料)
    quant_topk: List[List[TokenProb]] = field(default_factory=list)
    ref_logits: List[float] = field(default_factory=list)            # 每 position ref argmax logit (折线)
    quant_logits: List[float] = field(default_factory=list)          # 每 position quant argmax logit (折线)
    token_wise_cos: List[Optional[float]] = field(default_factory=list)
    token_wise_kl: List[Optional[float]] = field(default_factory=list)
    token_wise_topk_overlap: List[Optional[float]] = field(default_factory=list)
    token_wise_top1_match: List[bool] = field(default_factory=list)
    # visualization B: ref vs quant 全词表 logits 散点 (采样后, 成对样本)
    scatter_ref: List[float] = field(default_factory=list)
    scatter_quant: List[float] = field(default_factory=list)
    # visualization D: 全词表 logits 分布直方图 (预分箱)
    hist_bins: List[float] = field(default_factory=list)              # 箱边界
    hist_ref_counts: List[int] = field(default_factory=list)
    hist_quant_counts: List[int] = field(default_factory=list)


@dataclass
class InferenceCompareData:
    """推理结果对比：ref vs quant 模型在同一 prompt 下的生成结果逐 token 对比"""
    prompt: str = ""
    ref_output: str = ""                        # ref 模型生成文本
    quant_output: str = ""                      # quant 模型生成文本
    ref_tokens: List[str] = field(default_factory=list)    # ref 生成 token 序列
    quant_tokens: List[str] = field(default_factory=list)  # quant 生成 token 序列
    token_match_rate: Optional[float] = None     # token 匹配率 0-1
    exact_match: bool = False                   # 生成文本完全一致
    first_divergence_pos: Optional[int] = None   # 首个 token 分歧位置
    max_new_tokens: int = 0                     # 生成 token 数
    # 复用 scripts/badcase_eval.py 的异常检测器输出的额外字段 (退化/乱码诊断)
    ref_garbled: bool = False                  # ref 输出是否乱码 (detect_garbled)
    quant_garbled: bool = False                # quant 输出是否乱码
    ref_repeat: bool = False                   # ref 输出是否复读 (detect_repeat)
    quant_repeat: bool = False                 # quant 输出是否复读
    logits_cos_sim: Optional[float] = None      # ref vs quant logits 余弦相似度
    logits_kl: Optional[float] = None           # KL(ref || quant) on logits
    topk_overlap: Optional[float] = None        # top-k token 集合 IoU
    logits_nan_inf: bool = False               # logits 中是否存在 NaN/Inf


# ---------------------------------------------------------------------------
# 顶层容器
# ---------------------------------------------------------------------------

@dataclass
class ReportData:
    """acc_bench 完整报告数据 (所有模块的统一载体)"""
    overview: OverviewData = field(default_factory=OverviewData)
    l1_layers: List[L1LayerData] = field(default_factory=list)
    l2_results: List[L2LayerData] = field(default_factory=list)
    logits: Optional[LogitsData] = None
    inference_compare: Optional[InferenceCompareData] = None
    run_status: str = PARTIAL               # 默认部分跑通，assemble 时细化

    # ---- 序列化 ----
    def to_dict(self) -> Dict[str, Any]:
        """转 dict (NaN/Inf→None)，可直接 json.dump。"""
        return _clean_num(asdict(self))

    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReportData":
        """从 dict 重建 (JSON 反序列化的入口)。"""
        ov = _build_overview(d.get("overview") or {})
        l1 = [_build_l1_layer(x) for x in (d.get("l1_layers") or [])]
        l2 = [_build_l2_layer(x) for x in (d.get("l2_results") or [])]
        logits = _build_logits(d.get("logits"))
        inference_compare = _build_inference_compare(d.get("inference_compare") or d.get("badcase"))
        status = d.get("run_status") or PARTIAL
        return cls(
            overview=ov, l1_layers=l1, l2_results=l2,
            logits=logits, inference_compare=inference_compare, run_status=status,
        )


# ---------------------------------------------------------------------------
# from_dict 辅助
# ---------------------------------------------------------------------------

def _build_overview(d: Dict[str, Any]) -> OverviewData:
    activation_quant_enabled = d.get("activation_quant_enabled")
    if activation_quant_enabled is not None:
        activation_quant_enabled = bool(activation_quant_enabled)
    activation_quant_group_size = d.get("activation_quant_group_size")
    try:
        activation_quant_group_size = (
            int(activation_quant_group_size)
            if activation_quant_group_size is not None else None
        )
    except (TypeError, ValueError):
        activation_quant_group_size = None
    if activation_quant_group_size is not None and activation_quant_group_size <= 0:
        activation_quant_group_size = None
    return OverviewData(
        model_name=d.get("model_name", ""),
        ref_model_path=d.get("ref_model_path", ""),
        quant_model_path=d.get("quant_model_path", ""),
        quant_format=d.get("quant_format", ""),
        device_mode=d.get("device_mode", ""),
        prompt=d.get("prompt", ""),
        input_mode=d.get("input_mode", ""),
        comparison_scope=d.get("comparison_scope", "unknown") or "unknown",
        quant_method=d.get("quant_method", ""),
        activation_quant_enabled=activation_quant_enabled,
        activation_quant_type=d.get("activation_quant_type", ""),
        activation_quant_backend=d.get("activation_quant_backend", ""),
        activation_quant_group_size=activation_quant_group_size,
        boundary_result=d.get("boundary_result"),
        first_divergence_layer=d.get("first_divergence_layer"),
        first_threshold_crossing_layer=d.get("first_threshold_crossing_layer"),
        problem_path=d.get("problem_path"),
        best_repair_point=d.get("best_repair_point"),
        source_candidate=d.get("source_candidate"),
        confidence=d.get("confidence"),
        ground_truth_hit=d.get("ground_truth_hit"),
    )


def _build_l1_layer(d: Dict[str, Any]) -> L1LayerData:
    return L1LayerData(
        layer_idx=int(d.get("layer_idx", -1)),
        layer_name=d.get("layer_name", ""),
        cos_sim=d.get("cos_sim"),
        rel_l2=d.get("rel_l2"),
        snr=d.get("snr"),
        delta_cos=d.get("delta_cos"),
        drop_percent=d.get("drop_percent"),
        is_first_divergence=bool(d.get("is_first_divergence", False)),
        is_max_error=bool(d.get("is_max_error", False)),
    )


def _build_subgraph(d: Dict[str, Any]) -> SubgraphData:
    return SubgraphData(
        name=d.get("name", ""),
        quant_type=d.get("quant_type", ""),
        patch_recovery=d.get("patch_recovery"),
        self_rot_err=d.get("self_rot_err"),
        rot_b_err=d.get("rot_b_err"),
        flip_rate=d.get("flip_rate"),
        is_repair_point=bool(d.get("is_repair_point", False)),
        is_source_candidate=bool(d.get("is_source_candidate", False)),
    )


def _build_l2_layer(d: Dict[str, Any]) -> L2LayerData:
    return L2LayerData(
        layer_idx=int(d.get("layer_idx", -1)),
        base_l2=d.get("base_l2"),
        input_recovery=d.get("input_recovery"),
        subgraphs=[_build_subgraph(x) for x in (d.get("subgraphs") or [])],
        impact_boundary=d.get("impact_boundary"),
        root_suspect=d.get("root_suspect"),
        chain_delta=d.get("chain_delta"),
        flip_rates=d.get("flip_rates"),
    )


def _build_token_prob(d: Dict[str, Any]) -> TokenProb:
    return TokenProb(
        token_id=int(d.get("token_id", -1)),
        token_str=d.get("token_str", ""),
        ref_prob=d.get("ref_prob"),
        quant_prob=d.get("quant_prob"),
    )


def _build_logits(d: Optional[Dict[str, Any]]) -> Optional[LogitsData]:
    if not d:
        return None
    return LogitsData(
        token_positions=list(d.get("token_positions") or []),
        position_mode=str(d.get("position_mode") or "unknown"),
        ref_topk=[[_build_token_prob(x) for x in pos] for pos in d.get("ref_topk") or []],
        quant_topk=[[_build_token_prob(x) for x in pos] for pos in d.get("quant_topk") or []],
        ref_logits=list(d.get("ref_logits") or []),
        quant_logits=list(d.get("quant_logits") or []),
        token_wise_cos=list(d.get("token_wise_cos") or []),
        token_wise_kl=list(d.get("token_wise_kl") or []),
        token_wise_topk_overlap=list(d.get("token_wise_topk_overlap") or []),
        token_wise_top1_match=list(d.get("token_wise_top1_match") or []),
        scatter_ref=list(d.get("scatter_ref") or []),
        scatter_quant=list(d.get("scatter_quant") or []),
        hist_bins=list(d.get("hist_bins") or []),
        hist_ref_counts=list(d.get("hist_ref_counts") or []),
        hist_quant_counts=list(d.get("hist_quant_counts") or []),
    )


def _build_inference_compare(d: Optional[Dict[str, Any]]) -> Optional[InferenceCompareData]:
    if not d:
        return None
    return InferenceCompareData(
        prompt=d.get("prompt", ""),
        ref_output=d.get("ref_output", ""),
        quant_output=d.get("quant_output", ""),
        ref_tokens=list(d.get("ref_tokens") or []),
        quant_tokens=list(d.get("quant_tokens") or []),
        token_match_rate=d.get("token_match_rate"),
        exact_match=bool(d.get("exact_match", False)),
        first_divergence_pos=d.get("first_divergence_pos"),
        max_new_tokens=int(d.get("max_new_tokens", 0)),
    )


__all__ = [
    "SUCCESS", "PARTIAL", "INVALID_RUN", "INCONCLUSIVE", "FAILED", "VALID_STATUSES",
    "OverviewData", "L1LayerData", "SubgraphData", "L2LayerData",
    "TokenProb", "LogitsData", "InferenceCompareData", "ReportData",
]
