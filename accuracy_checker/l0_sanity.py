"""
L0 模型完整性校验 (Pre-quantization Sanity)

在量化精度诊断 (L1/L2) 之前, 对 ref / quant 模型的"文件 + 权重"做完整性把关,
确保诊断结果可信 (避免因权重缺失/NaN/meta 残留 导致后续 L1/L2 误判)。

设计:
  - 轻量级: 读取 safetensors 索引 + 少量小 tensor (scale/offset/norm), 不实例化整模型到 NPU。
  - 路径驱动: run_l0_sanity(ref_model_path, quant_model_path, **kwargs)。
  - 若 kwargs 中传入已加载的 nn.Module (ref_model/quant_model), 则额外做 meta 残留 /
    device/dtype 校验 (复用 ValidityChecker), 否则这两项 SKIP 并注明。

检查项:
  1. 权重 key 完整性 (ref vs quant key 集合对比)
  2. scale/offset 合法性 (无 NaN/Inf, shape 匹配 base weight)
  3. 量化配置一致性 (quant_model_description.json / config.quantization_config vs 实际权重后缀)
  4. NaN/Inf 检查 (抽样权重 tensor)
  5. head/norm 关键参数存在性 (lm_head / embed_tokens / final norm)
  6. device/dtype 检查 (采样 tensor 的 dtype; device 对磁盘权重无意义 -> SKIP / 若有模型则校验)
  7. 旋转矩阵 shape 校验 (若提供 rotation_matrix)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

import logging

from .validity_checks import (
    CheckItem, RunStatus, ValidityChecker,
    CHECK_PASS, CHECK_WARN, CHECK_FAIL, CHECK_SKIP,
)
from .utils import parse_base_name

logger = logging.getLogger(__name__)


# ============================================================================
# 量化权重后缀 (与 utils.parse_base_name 保持一致, 这里用于判定 scale/offset)
# ============================================================================

_SCALE_SUFFIXES = (
    ".weight_scale", ".weight_offset",
    ".deq_scale", ".input_scale", ".input_offset", ".quant_bias",
    ".kv_cache_scale", ".kv_cache_offset", ".scale_bias",
)
# 量化主体权重后缀
_WEIGHT_SUFFIX = ".weight"
# 关键参数名片段 (head / norm / embed)
_CRITICAL_KEY_HINTS = ("lm_head", "embed_tokens", "norm.weight")
# 抽样上限
_DEFAULT_MAX_NAN_SAMPLES = 8
_DEFAULT_MAX_SCALE_SAMPLES = 24
_NaN_SCAN_ELEMENT_CAP = 200_000_000  # 单 tensor 最多扫 ~200M 元素 (切片)


# ============================================================================
# 结果结构
# ============================================================================

@dataclass
class L0SanityResult:
    """L0 校验整体结果。"""
    checks: List[CheckItem] = field(default_factory=list)
    overall_status: str = RunStatus.INCONCLUSIVE.value  # RunStatus.value 字符串
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checks": [c.to_dict() for c in self.checks],
            "overall_status": self.overall_status,
            "summary": self.summary,
        }

    def __str__(self) -> str:
        return self.summary


# ============================================================================
# 磁盘权重读取工具
# ============================================================================

def _is_msmodelslim(model_path: str) -> bool:
    return os.path.exists(os.path.join(model_path, "quant_model_description.json"))


def _load_index(model_path: str) -> Optional[dict]:
    """读取 safetensors 索引 (msmodelslim 或 标准 HF)。

    返回 index dict (含 weight_map), 或 None (单文件/无索引)。
    """
    candidates = [
        "quant_model_weights.safetensors.index.json",  # msmodelslim
        "model.safetensors.index.json",                # 标准 HF (含 symlink 后)
    ]
    for name in candidates:
        p = os.path.join(model_path, name)
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, PermissionError, OSError) as e:
                logger.warning(f"Failed to read index {p} ({e.__class__.__name__}): {e}")
                return {"__broken__": p, "__error__": f"{e.__class__.__name__}: {e}"}
    return None


def _load_weight_map(model_path: str, index: Optional[dict]) -> Dict[str, str]:
    """从索引或单文件得到 key -> shard 文件名 映射 (单文件时 value='')。"""
    if index is None:
        return {}
    if isinstance(index, dict) and "__broken__" in index:
        return {}
    weight_map = index.get("weight_map", {}) if isinstance(index, dict) else {}
    return dict(weight_map)


def _single_shard_file(model_path: str) -> Optional[str]:
    """无索引时的单文件兜底 (model.safetensors)。"""
    for name in ("model.safetensors", "quant_model_weights.safetensors"):
        p = os.path.join(model_path, name)
        if os.path.exists(p):
            return name
    return None


def _list_keys_from_disk(model_path: str, index: Optional[dict]) -> List[str]:
    """得到全部权重 key (索引 weight_map, 否则用 safe_open.keys())。"""
    wm = _load_weight_map(model_path, index)
    if wm:
        return list(wm.keys())
    single = _single_shard_file(model_path)
    if single is not None:
        try:
            from safetensors import safe_open
            with safe_open(os.path.join(model_path, single), framework="pt", device="cpu") as f:
                return list(f.keys())
        except Exception as e:  # noqa: BLE001
            logger.warning(f"safe_open keys failed for {single}: {e}")
    return []


def _read_tensor(model_path: str, shard_file: str, key: str) -> Optional[torch.Tensor]:
    """从指定 shard 读单个 tensor (CPU)。失败返回 None。"""
    if not shard_file:
        # 单文件: 尝试已知名字
        single = _single_shard_file(model_path)
        if single is None:
            return None
        shard_file = single
    full = os.path.join(model_path, shard_file)
    if not os.path.exists(full):
        return None
    try:
        from safetensors import safe_open
        with safe_open(full, framework="pt", device="cpu") as f:
            if key not in f.keys():
                return None
            return f.get_tensor(key)
    except Exception as e:  # noqa: BLE001 - OOM/损坏 不应中断整体检查
        logger.debug(f"read tensor {key} from {shard_file} failed: {e}")
        return None


def _load_config(model_path: str) -> dict:
    p = os.path.join(model_path, "config.json")
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _load_quant_desc(model_path: str) -> dict:
    p = os.path.join(model_path, "quant_model_description.json")
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            return {}
    return {}


# ============================================================================
# 各项子检查 (均返回 CheckItem / List[CheckItem])
# ============================================================================

def _check_key_completeness(ref_path, quant_path, ref_keys, quant_keys) -> CheckItem:
    """ref 与 quant 权重 key 集合对比 (缺失/多余)。"""
    name = "weight_key_completeness"
    ref_set, quant_set = set(ref_keys), set(quant_keys)
    # 量化后主体权重 key 应基本一致 (quant 额外有 scale/offset; 可能缺 lm_head 因 tied)
    missing = ref_set - quant_set
    extra = quant_set - ref_set
    # 过滤掉预期内的差异: quant 多出的全是 scale/offset 后缀 (正常); ref 有 lm_head 但
    # quant 因 weight tying 缺省 (embed_tokens 共享) 不算致命
    expected_extra = {k for k in extra if k.endswith(_SCALE_SUFFIXES)}
    unexpected_extra = extra - expected_extra
    # 缺失: 若是 lm_head 且 quant 有 embed_tokens -> tied, 视为可选缺失 (WARN 而非 FAIL)
    tied_candidates = {k for k in missing if "lm_head" in k and any("embed_tokens" in q for q in quant_set)}

    real_missing = missing - tied_candidates
    if real_missing or unexpected_extra:
        return CheckItem(
            name, CHECK_FAIL,
            f"key sets diverge: missing={len(real_missing)} unexpected_extra={len(unexpected_extra)}",
            evidence=f"missing_sample={sorted(real_missing)[:10]} unexpected_extra_sample={sorted(unexpected_extra)[:10]}",
        )
    if missing or extra:  # 均为可接受差异 (tied/scale)
        return CheckItem(
            name, CHECK_WARN,
            f"acceptable divergence: tied_missing={len(tied_candidates)} scale_offset_extra={len(expected_extra)}",
            evidence=f"tied_missing_sample={sorted(tied_candidates)[:5]} scale_extra_sample={sorted(expected_extra)[:5]}",
        )
    return CheckItem(name, CHECK_PASS, f"key sets match ({len(ref_set)} keys)",
                     evidence=f"n_keys={len(ref_set)}")


def _check_scale_offset(quant_path, quant_keys, weight_map, max_samples) -> CheckItem:
    """quant 权重的 scale/offset: 无 NaN/Inf + shape 与 base weight 广播兼容。"""
    name = "scale_offset_legality"
    scale_keys = [k for k in quant_keys if k.endswith(_SCALE_SUFFIXES)]
    if not scale_keys:
        return CheckItem(name, CHECK_PASS,
                         "no scale/offset keys present (非 msmodelslim/compressed-tensors 量化模型)",
                         evidence="n_scale_keys=0")
    sample = scale_keys[:max_samples]
    n_bad = 0
    n_checked = 0
    n_shape_mismatch = 0
    bad_evidence = []
    for sk in sample:
        shard = weight_map.get(sk, "")
        t = _read_tensor(quant_path, shard, sk)
        if t is None:
            continue
        n_checked += 1
        if not torch.isfinite(t).all().item():
            n_bad += 1
            bad_evidence.append(f"{sk}:has_nan/inf")
            continue
        # shape 一致性: 与对应 base weight (.weight) 比较
        base_key = parse_base_name(sk) + ".weight"
        bw = weight_map.get(base_key)
        base_t = _read_tensor(quant_path, weight_map.get(base_key, ""), base_key) if bw else None
        if base_t is not None:
            # per-channel scale: 0D 标量或 1D (per-channel, dim 对齐 weight 任一维) 或 2D == weight shape
            if t.numel() == 1:
                pass  # scalar always ok
            elif t.dim() == 1:
                if t.shape[0] not in base_t.shape and t.shape[0] != 1:
                    n_shape_mismatch += 1
                    bad_evidence.append(f"{sk}:shape{tuple(t.shape)} vs {base_key}{tuple(base_t.shape)}")
            elif t.dim() == 2 and t.shape != base_t.shape:
                n_shape_mismatch += 1
                bad_evidence.append(f"{sk}:shape{tuple(t.shape)} vs {base_key}{tuple(base_t.shape)}")
            elif t.dim() > 2:
                n_shape_mismatch += 1
                bad_evidence.append(f"{sk}:shape{tuple(t.shape)} unexpected rank")
    if n_checked == 0:
        return CheckItem(name, CHECK_SKIP, "no scale/offset tensors readable",
                         evidence=f"sampled={len(sample)} readable=0")
    if n_bad:
        return CheckItem(name, CHECK_FAIL, f"{n_bad}/{n_checked} scale/offset have NaN/Inf",
                         evidence="; ".join(bad_evidence[:10]))
    if n_shape_mismatch:
        return CheckItem(name, CHECK_WARN, f"{n_shape_mismatch}/{n_checked} scale shape suspects mismatch",
                         evidence="; ".join(bad_evidence[:10]))
    return CheckItem(name, CHECK_PASS,
                     f"scale/offset finite & shape-ok ({n_checked}/{len(sample)} sampled, total {len(scale_keys)})",
                     evidence=f"n_scale_keys={len(scale_keys)} sampled={n_checked}")


def _check_quant_config_consistency(quant_path, quant_keys) -> CheckItem:
    """量化配置 (quant_model_description.json / config.quantization_config) 与实际权重后缀一致。"""
    name = "quantization_config_consistency"
    has_scales = any(k.endswith(_SCALE_SUFFIXES) for k in quant_keys)
    desc = _load_quant_desc(quant_path)
    config = _load_config(quant_path)
    # 描述里的量化类型分布
    declared_types = {}
    if desc:
        for k, v in desc.items():
            if isinstance(v, str):
                declared_types[v] = declared_types.get(v, 0) + 1
    qconfig = config.get("quantization_config", {}) if isinstance(config, dict) else {}
    quant_method = qconfig.get("quant_method") if isinstance(qconfig, dict) else None

    if not desc and not qconfig:
        if not has_scales:
            return CheckItem(name, CHECK_PASS, "no quantization config & no scales (FP/DENSE model)",
                             evidence="quant_desc=config=none")
        return CheckItem(name, CHECK_WARN,
                         "scales present but no quant config -> config may be stale/lost",
                         evidence="has_scales=true quant_desc=none config_quantization=none")
    # 一致性: 若声明量化 (有非 FLOAT 类型 或 quant_method set) 则应存在 scale 后缀
    has_non_float = any(t != "FLOAT" and t != "float" for t in declared_types)
    if (has_non_float or quant_method) and not has_scales:
        return CheckItem(name, CHECK_FAIL,
                         "config declares quantization but no scale/offset weights found",
                         evidence=f"declared_types={declared_types} quant_method={quant_method}")
    return CheckItem(name, CHECK_PASS,
                     f"quant config consistent ({len(declared_types)} declared types, method={quant_method})",
                     evidence=f"declared_types={dict(declared_types)} quant_method={quant_method} has_scales={has_scales}")


def _check_nan_inf_sampling(model_path, keys, weight_map, max_samples, label) -> CheckItem:
    """抽样权重 tensor 做 NaN/Inf 检查 (优先采样 scale/norm 等小 tensor)。"""
    name = f"nan_inf_sampling:{label}"
    if not keys:
        return CheckItem(name, CHECK_SKIP, "no keys to sample",
                         evidence="n_keys=0")
    # 优先采样小张量: weight_scale/norm/小投影; 避免直接读超大 embed
    small_pref = []
    big_rest = []
    for k in keys:
        if k.endswith(_SCALE_SUFFIXES) or "norm" in k:
            small_pref.append(k)
        elif "embed_tokens" in k or "lm_head" in k:
            big_rest.append(k)  # 可能很大, 放后面
        elif k.endswith(_WEIGHT_SUFFIX):
            small_pref.append(k)
    # 抽样
    sample = (small_pref[:max_samples] + big_rest[:2])[:max_samples + 2]
    n_checked = 0
    n_bad = 0
    bad = []
    for k in sample:
        t = _read_tensor(model_path, weight_map.get(k, ""), k)
        if t is None:
            continue
        n_checked += 1
        # 大 tensor 切片扫, 避免全量统计 OOM (虽然已读入, .all() 也需遍历)
        view = t.float().flatten()
        if view.numel() > _NaN_SCAN_ELEMENT_CAP:
            view = view[:_NaN_SCAN_ELEMENT_CAP]
        if not torch.isfinite(view).all().item():
            n_nan = int(torch.isnan(view).sum().item())
            n_inf = int(torch.isinf(view).sum().item())
            n_bad += 1
            bad.append(f"{k}:nan={n_nan} inf={n_inf}")
    if n_checked == 0:
        return CheckItem(name, CHECK_SKIP, "no tensors readable",
                         evidence=f"sampled={len(sample)} readable=0")
    if n_bad:
        return CheckItem(name, CHECK_FAIL, f"{n_bad}/{n_checked} sampled tensors have NaN/Inf",
                         evidence="; ".join(bad[:10]))
    return CheckItem(name, CHECK_PASS, f"no NaN/Inf in {n_checked} sampled tensors",
                     evidence=f"sampled={n_checked}")


def _check_critical_headnorm(ref_path, quant_path, ref_keys, quant_keys) -> CheckItem:
    """lm_head / embed_tokens / final norm 关键参数是否存在于权重。"""
    name = "head_norm_keys"
    missing = []
    present = []
    # 在 ref 侧检查这些 key 存在 (quant 可能因 tying 缺 lm_head)
    for hint in _CRITICAL_KEY_HINTS:
        hits = [k for k in ref_keys if hint in k]
        if hits:
            present.append(hits[0])
            continue
        # quant 侧也查 (某些模型 norm 在 quant)
        hits_q = [k for k in quant_keys if hint in k]
        if hits_q:
            present.append(hits_q[0] + "(quant)")
            continue
        missing.append(hint)
    if "lm_head" in missing:
        # weight tying: embed_tokens 存在则 lm_head 缺失可接受
        if any("embed_tokens" in k for k in ref_keys):
            missing = [m for m in missing if m != "lm_head"]
            present.append("lm_head(tied to embed_tokens)")
    if missing:
        return CheckItem(name, CHECK_FAIL, f"critical params missing: {missing}",
                         evidence=f"missing={missing}")
    return CheckItem(name, CHECK_PASS, f"critical head/norm params present ({len(present)})",
                     evidence=f"present={present}")


def _check_dtype(model_path, keys, weight_map, label, expected_dtype) -> CheckItem:
    """采样一个 tensor 的 dtype 校验 (磁盘权重无 device 概念)。"""
    name = f"weight_dtype:{label}"
    if not keys:
        return CheckItem(name, CHECK_SKIP, "no keys", evidence="n_keys=0")
    # 任取一个 .weight tensor 读 dtype
    cand = [k for k in keys if k.endswith(_WEIGHT_SUFFIX)] or keys
    for k in cand[:3]:
        t = _read_tensor(model_path, weight_map.get(k, ""), k)
        if t is not None:
            dt = str(t.dtype).replace("torch.", "")
            if expected_dtype is not None:
                from .validity_checks import _normalize_dtype
                exp = _normalize_dtype(expected_dtype)
                if _normalize_dtype(dt) != exp:
                    return CheckItem(name, CHECK_WARN,
                                     f"dtype {dt} != expected {exp}",
                                     evidence=f"sample_key={k} dtype={dt} expected={exp}")
            return CheckItem(name, CHECK_PASS, f"sampled dtype ok ({dt})",
                             evidence=f"sample_key={k} dtype={dt}")
    return CheckItem(name, CHECK_SKIP, "no tensor readable for dtype check",
                     evidence=f"candidates={cand[:3]}")


def _check_rotation_matrices(rot_path, config) -> CheckItem:
    """旋转矩阵 shape 校验 (若提供 rotation_matrix)。"""
    name = "rotation_matrix_shapes"
    if not rot_path:
        return CheckItem(name, CHECK_SKIP, "no rotation_matrix provided", evidence="")
    # 复用 subgraph_locate.load_all_rotation_matrices (lazy import, 避免重型依赖前置)
    try:
        from .subgraph_locate import load_all_rotation_matrices
        rot_mats = load_all_rotation_matrices(rot_path)
    except Exception as e:  # noqa: BLE001
        return CheckItem(name, CHECK_FAIL, f"failed to load rotation matrices: {e}",
                         evidence=str(e))
    if not rot_mats:
        return CheckItem(name, CHECK_WARN,
                         f"rotation file present but no matrices found: {rot_path}",
                         evidence=str(rot_path))
    hidden = config.get("hidden_size") if isinstance(config, dict) else None
    inter = config.get("intermediate_size") if isinstance(config, dict) else None
    valid_dims = {d for d in (hidden, inter) if d}
    findings = []
    n_bad = 0
    for k, R in rot_mats.items():
        shape = tuple(R.shape) if isinstance(R, torch.Tensor) else None
        if not isinstance(R, torch.Tensor) or R.dim() != 2:
            n_bad += 1
            findings.append(f"{k}:invalid_shape={shape}")
            continue
        finite = bool(torch.isfinite(R.float()).all().item())
        square = (R.shape[0] == R.shape[1])
        # 与 config 维度匹配: 至少一方等于 hidden 或 inter
        matched = (not valid_dims) or (R.shape[0] in valid_dims) or (R.shape[1] in valid_dims)
        if not finite or not square or not matched:
            n_bad += 1
            findings.append(f"{k}:shape={shape} finite={finite} square={square} matched={matched}")
        else:
            findings.append(f"{k}:shape={shape} OK")
    expected_n = 4  # QuaRot/GLM5: rot/rot_b_proj/rot_uv/rot_kv_b_proj
    if n_bad:
        return CheckItem(name, CHECK_FAIL, f"{n_bad}/{len(rot_mats)} rotation matrices invalid",
                         evidence="; ".join(findings))
    status = CHECK_PASS if len(rot_mats) >= expected_n else CHECK_WARN
    return CheckItem(name, status,
                     f"rotation matrices ok ({len(rot_mats)} found, expected ~{expected_n})",
                     evidence="; ".join(findings))


# ============================================================================
# 主入口
# ============================================================================

def run_l0_sanity(
    ref_model_path: str,
    quant_model_path: str,
    *,
    rotation_matrix: Optional[str] = None,
    expected_dtype: Optional[Any] = None,
    ref_model: Optional[nn.Module] = None,
    quant_model: Optional[nn.Module] = None,
    expected_device: Optional[str] = None,
    max_nan_samples: int = _DEFAULT_MAX_NAN_SAMPLES,
    max_scale_samples: int = _DEFAULT_MAX_SCALE_SAMPLES,
) -> L0SanityResult:
    """L0 模型完整性校验。

    Args:
        ref_model_path: 参考模型路径 (FP16/BF16)。
        quant_model_path: 量化模型路径。
        rotation_matrix: 可选旋转矩阵文件 (路径), 用于验证 shape。
        expected_dtype: 期望的主体权重 dtype (str/torch.dtype); 不传则只采样不比对。
        ref_model / quant_model: 可选, 已加载的 nn.Module; 提供则额外做 meta 残留 /
            device/dtype 校验 (复用 ValidityChecker), 否则这两项 SKIP。
        expected_device: 与 quant_model 配套的期望设备 (如 'npu:0')。
        max_nan_samples / max_scale_samples: NaN/scale 抽样上限。

    Returns:
        L0SanityResult (含 checks 列表 + overall_status + summary)。
    """
    checks: List[CheckItem] = []
    vc = ValidityChecker()

    # ---- 路径存在性 ----
    for label, path in (("ref", ref_model_path), ("quant", quant_model_path)):
        if not path or not os.path.isdir(path):
            checks.append(CheckItem(f"model_path:{label}", CHECK_FAIL,
                                    f"{label} model path not found: {path}",
                                    evidence=str(path)))
        else:
            checks.append(CheckItem(f"model_path:{label}", CHECK_PASS,
                                    f"{label} path exists", evidence=str(path)))
    if any(c.status == CHECK_FAIL for c in checks):
        return _finalize(checks, prefix="L0 aborted: model path invalid")

    # ---- 读取索引/配置 ----
    ref_index = _load_index(ref_model_path)
    quant_index = _load_index(quant_model_path)
    ref_keys = _list_keys_from_disk(ref_model_path, ref_index)
    quant_keys = _list_keys_from_disk(quant_model_path, quant_index)
    ref_wm = _load_weight_map(ref_model_path, ref_index)
    quant_wm = _load_weight_map(quant_model_path, quant_index)
    quant_config = _load_config(quant_model_path)

    # ---- 1. key 完整性 ----
    checks.append(_check_key_completeness(ref_model_path, quant_model_path, ref_keys, quant_keys))

    # ---- 2. scale/offset 合法性 ----
    checks.append(_check_scale_offset(quant_model_path, quant_keys, quant_wm, max_scale_samples))

    # ---- 3. 量化配置一致性 ----
    checks.append(_check_quant_config_consistency(quant_model_path, quant_keys))

    # ---- 4. NaN/Inf 抽样 (ref + quant) ----
    checks.append(_check_nan_inf_sampling(ref_model_path, ref_keys, ref_wm, max_nan_samples, "ref"))
    checks.append(_check_nan_inf_sampling(quant_model_path, quant_keys, quant_wm, max_nan_samples, "quant"))

    # ---- 5. head/norm 关键参数 ----
    checks.append(_check_critical_headnorm(ref_model_path, quant_model_path, ref_keys, quant_keys))

    # ---- 6. dtype (device 对磁盘权重无意义, 单列; 有模型则用 ValidityChecker) ----
    checks.append(_check_dtype(ref_model_path, ref_keys, ref_wm, "ref", expected_dtype))
    checks.append(_check_dtype(quant_model_path, quant_keys, quant_wm, "quant", expected_dtype))

    # ---- 6b. meta 残留 / device/dtype (需要已加载模型) ----
    if ref_model is not None:
        ci = vc.check_meta_residual(ref_model)
        checks.append(CheckItem("meta_residual:ref", ci.status, ci.detail, ci.evidence))
    else:
        checks.append(CheckItem("meta_residual:ref", CHECK_SKIP,
                                "no ref_model loaded; run with loaded model to verify meta residual",
                                evidence="skipped"))
    if quant_model is not None:
        ci = vc.check_meta_residual(quant_model)
        checks.append(CheckItem("meta_residual:quant", ci.status, ci.detail, ci.evidence))
        ci = vc.check_weight_device_dtype(quant_model, expected_device, expected_dtype)
        checks.append(CheckItem("weight_device_dtype:quant", ci.status, ci.detail, ci.evidence))
    else:
        checks.append(CheckItem("meta_residual:quant", CHECK_SKIP,
                                "no quant_model loaded; run with loaded model to verify",
                                evidence="skipped"))
        checks.append(CheckItem("weight_device_dtype:quant", CHECK_SKIP,
                                "requires loaded quant_model",
                                evidence="skipped"))

    # ---- 7. 旋转矩阵 ----
    checks.append(_check_rotation_matrices(rotation_matrix, quant_config))

    return _finalize(checks)


def _finalize(checks: List[CheckItem], prefix: str = "L0 sanity check") -> L0SanityResult:
    """L0 级别聚合: SKIP 视为"可选/未加载模型"的软项, 不拖累整体结论。

    规约:
      - 任一 FAIL -> INVALID_RUN
      - 任一 WARN (无 FAIL) -> PARTIAL
      - 仅有 PASS(+SKIP) -> SUCCESS
      - 全 SKIP / 无 PASS -> INCONCLUSIVE
    """
    n_pass = sum(1 for c in checks if c.status == CHECK_PASS)
    n_warn = sum(1 for c in checks if c.status == CHECK_WARN)
    n_fail = sum(1 for c in checks if c.status == CHECK_FAIL)
    n_skip = sum(1 for c in checks if c.status == CHECK_SKIP)
    if n_fail:
        overall = RunStatus.INVALID_RUN
    elif n_warn:
        overall = RunStatus.PARTIAL
    elif n_pass:
        overall = RunStatus.SUCCESS
    else:
        overall = RunStatus.INCONCLUSIVE
    summary = (
        f"{prefix}: overall={overall.value} | "
        f"PASS={n_pass} WARN={n_warn} FAIL={n_fail} SKIP={n_skip} (total {len(checks)})"
    )
    return L0SanityResult(
        checks=checks,
        overall_status=overall.value,
        summary=summary,
    )


__all__ = [
    "L0SanityResult",
    "run_l0_sanity",
    "CheckItem",
    "RunStatus",
]
