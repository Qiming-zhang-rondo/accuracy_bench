"""
量化模型精度对齐工具

漏斗式诊断:
  L0:  模型完整性预检 (权重 key/scale/NaN/旋转矩阵) — 诊断前把关
  L0.5 Boundary: 定界 — 区分误差来自权重/量化 vs 推理框架
  L1:  逐 block 对比 — embedding, 每层 block output, final logits
  L2:  Sub-graph 反事实诊断 — 定位具体坏算子
  BadCase: 对照 ground truth 评估定位能力
  报告: 统一 ReportData → 自包含 HTML (v2)
"""

# ---- 基础设施 (无内部依赖) ----
from .metrics import cos_sim, compute_all_metrics
from .utils import (
    to_cpu_fp32,
    get_decoder_layers,
    get_embed_module,
    get_norm_module,
    get_lm_head_module,
    get_num_layers,
    auto_device,
    parse_dtype,
    clear_device_cache,
)
from .model_structure import ModelComponents, get_model_components, get_text_config
from .dspark import (
    DSparkContract, DSparkSample, DSparkComparator,
    is_dspark_checkpoint, load_dspark_contract, validate_dspark_pair,
    load_dspark_sample,
)

# ---- 报告 schema (standalone) ----
from .report_schema import (
    ReportData, OverviewData, L1LayerData, L2LayerData, SubgraphData,
    LogitsData, TokenProb, InferenceCompareData,
)

# ---- 有效性检查 (standalone) ----
from .validity_checks import (
    RunStatus, CheckItem, ValidityChecker,
    CHECK_PASS, CHECK_WARN, CHECK_FAIL, CHECK_SKIP,
    worst_status, aggregate_overall,
)

# ---- Logits 对比 (依赖 report_schema) ----
from .logits_compare import (
    LogitsCollection, LogitsComparison,
    collect_logits, collect_last_logits, compare_logits, compare_captured_topk,
)
from .captured_logits import CapturedToken, CapturedLogits, load_captured_logits
from .inference_compare import compare_inference, detect_repeat, detect_garbled

# ---- HTML 报告 (standalone, v1 + v2) ----
from .html_report import (
    generate_html_report, generate_product_html_report, generate_index_html,
)

# ---- 报告组装 (依赖 report_schema + html_report) ----
from .report_data import assemble_report

# ---- Bad case 工作流 (依赖 utils) ----
from .badcase_workflow import (
    BadCaseManifest,
    load_manifest, save_manifest,
    GroundTruthComparison, compare_with_ground_truth, normalize_module_name,
)

# ---- L0 完整性校验 (依赖 validity_checks + utils) ----
from .l0_sanity import run_l0_sanity, L0SanityResult

# ---- 历史核心路径 ----
from .layer1_block_compare import ShardedBlockComparator
from .report import AlignmentReport
from .model_loader import load_model_for_comparison
from .inference_check import (
    hf_inference_check,
    qwen35_inference_check,
    distribute_model,
    dequantize_model_on_devices,
    parse_devices,
    run_boundary, classify_boundary, BoundaryResult,
    boundary_result_to_dict,
    INTERMITTENT_LOGITS_ALIGNED, INTERMITTENT_LOGITS_MISMATCH,
    INTERMITTENT_RANKING_SENSITIVE,
)

__all__ = [
    # metrics / utils
    "cos_sim", "compute_all_metrics",
    "to_cpu_fp32",
    "get_decoder_layers", "get_embed_module", "get_norm_module", "get_lm_head_module",
    "get_num_layers",
    "ModelComponents", "get_model_components", "get_text_config",
    "DSparkContract", "DSparkSample", "DSparkComparator",
    "is_dspark_checkpoint", "load_dspark_contract", "validate_dspark_pair",
    "load_dspark_sample",
    "auto_device", "parse_dtype", "clear_device_cache",
    # 报告 schema
    "ReportData", "OverviewData", "L1LayerData", "L2LayerData", "SubgraphData",
    "LogitsData", "TokenProb", "InferenceCompareData",
    # 有效性检查
    "RunStatus", "CheckItem", "ValidityChecker",
    "CHECK_PASS", "CHECK_WARN", "CHECK_FAIL", "CHECK_SKIP",
    "worst_status", "aggregate_overall",
    # logits 对比
    "LogitsCollection", "LogitsComparison",
    "collect_logits", "collect_last_logits", "compare_logits", "compare_captured_topk",
    "CapturedToken", "CapturedLogits", "load_captured_logits",
    "compare_inference", "detect_repeat", "detect_garbled",
    # HTML 报告
    "generate_html_report", "generate_product_html_report", "generate_index_html", "assemble_report",
    # bad case
    "BadCaseManifest", "load_manifest", "save_manifest",
    "GroundTruthComparison", "compare_with_ground_truth", "normalize_module_name",
    # L0 校验
    "run_l0_sanity", "L0SanityResult",
    # 历史核心
    "ShardedBlockComparator",
    "AlignmentReport",
    "load_model_for_comparison",
    # 定界恢复 (inference_check)
    "hf_inference_check",
    "qwen35_inference_check",
    "distribute_model",
    "dequantize_model_on_devices",
    "parse_devices",
    "run_boundary", "classify_boundary", "BoundaryResult",
    "boundary_result_to_dict",
    "INTERMITTENT_LOGITS_ALIGNED", "INTERMITTENT_LOGITS_MISMATCH",
    "INTERMITTENT_RANKING_SENSITIVE",
]
