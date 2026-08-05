"""
精度诊断过程的有效性检查 (Validity Checks)

提供:
  - RunStatus: 统一状态枚举 (SUCCESS / PARTIAL / INVALID_RUN / INCONCLUSIVE / FAILED)
  - CheckItem: 单项检查结果的标准化载体
  - ValidityChecker: 对诊断流程各环节做有效性校验

设计原则:
  - 纯函数式检查，不依赖 NPU/CUDA。NaN/Inf 等检查用 CPU 上 detach 的 tensor。
  - 每个检查返回 CheckItem，便于汇总成结构化报告 (Agent C 读取生成 HTML)。
  - CheckItem 与 RunStatus 同时被 l0_sanity.py 复用 (from .validity_checks import ...)，
    避免循环导入：本模块不 import l0_sanity。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)


# ============================================================================
# 统一状态枚举
# ============================================================================

class RunStatus(Enum):
    """一次 (L0/L1/L2/badcase) 诊断的整体运行状态。

    SUCCESS       全部检查通过，结果可信。
    PARTIAL       部分检查通过 / 部分跳过，结果可用但需注意。
    INVALID_RUN   存在致命问题 (NaN/Inf、meta 残留、权重缺失等)，结果不可信。
    INCONCLUSIVE  检查通过但证据不足以得出结论 (例如 hook 未命中且无数据)。
    FAILED        检查执行本身失败 (异常、依赖缺失)，无有效结果。
    """

    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    INVALID_RUN = "INVALID_RUN"
    INCONCLUSIVE = "INCONCLUSIVE"
    FAILED = "FAILED"

    def __str__(self) -> str:
        return self.value


# ============================================================================
# 单项检查结果
# ============================================================================

# CheckItem 的状态取值 (与 RunStatus 不同粒度: 描述单项而非整体)
CHECK_PASS = "PASS"
CHECK_WARN = "WARN"
CHECK_FAIL = "FAIL"
CHECK_SKIP = "SKIP"
_CHECK_ORDER = {CHECK_PASS: 0, CHECK_WARN: 1, CHECK_SKIP: 2, CHECK_FAIL: 3}


@dataclass
class CheckItem:
    """单项检查结果。

    Attributes:
        name: 检查项名称 (如 "nan_inf:lm_head.weight")。
        status: PASS / WARN / FAIL / SKIP。
        detail: 人类可读说明。
        evidence: 具体数据 (如 "3 of 1024 elements are NaN")，便于落库/HTML 展示。
    """
    name: str
    status: str
    detail: str
    evidence: str = ""

    def __post_init__(self):
        if self.status not in _CHECK_ORDER:
            raise ValueError(
                f"CheckItem.status must be one of {list(_CHECK_ORDER)}, got {self.status!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _dtype_from_str(d: Any) -> Any:
    """str(torch.float32)='torch.float32' -> torch.dtype。已是 torch.dtype 直接返回。"""
    return _normalize_dtype(d)


def worst_status(statuses: List[str]) -> str:
    """取多个单项状态中最严重的一个 (FAIL > SKIP > WARN > PASS)。空列表返回 PASS。"""
    if not statuses:
        return CHECK_PASS
    return max(statuses, key=lambda s: _CHECK_ORDER.get(s, 0))


def aggregate_overall(checks: List[CheckItem]) -> RunStatus:
    """根据单项检查列表推导整体 RunStatus。

    - 任一 FAIL -> INVALID_RUN
    - 任一 SKIP 且无 FAIL -> INCONCLUSIVE (证据不足)
    - 任一 WARN 且无 FAIL/SKIP -> PARTIAL
    - 全部 PASS -> SUCCESS
    - 无检查 -> INCONCLUSIVE
    """
    if not checks:
        return RunStatus.INCONCLUSIVE
    worst = worst_status([c.status for c in checks])
    if worst == CHECK_FAIL:
        return RunStatus.INVALID_RUN
    if worst == CHECK_SKIP:
        return RunStatus.INCONCLUSIVE
    if worst == CHECK_WARN:
        return RunStatus.PARTIAL
    return RunStatus.SUCCESS


# ============================================================================
# 有效性检查器
# ============================================================================

def _tensor_view(t: Any) -> torch.Tensor:
    """把可能是 list/数值 的输入规整成 tensor (CPU)。已是 tensor 则 detach+cpu。"""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu()
    return torch.as_tensor(t)


# dtype 字符串 -> torch.dtype (兼容 'float32' / 'torch.float32' / torch.float32)
_DTYPE_ALIAS = {
    "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float32": torch.float32, "fp32": torch.float32, "float": torch.float32,
    "float64": torch.float64, "fp64": torch.float64, "double": torch.float64,
    "int8": torch.int8, "int16": torch.int16, "int32": torch.int32,
    "uint8": torch.uint8, "bool": torch.bool,
}


def _normalize_dtype(d: Any) -> Any:
    """接受 torch.dtype / 'float32' / 'torch.float32', 统一返回 torch.dtype。原样已是 torch.dtype 则直接返回。"""
    if isinstance(d, torch.dtype):
        return d
    key = str(d).replace("torch.", "").strip().lower()
    return _DTYPE_ALIAS.get(key, d)


class ValidityChecker:
    """精度诊断过程的有效性检查器。

    所有方法都是无状态的纯检查，返回 CheckItem。可在 L0/L1/L2/badcase 流程中按需调用。
    """

    # ---- 输入一致性 ----

    def check_tokenizer_consistency(self, ref_tokenizer, quant_tokenizer) -> CheckItem:
        """确认 ref / quant tokenizer 一致 (vocab 大小 + 特殊 token)。

        量化不应改变 tokenizer；若不一致，则后续对比的 input_ids/logits 不可比。
        """
        name = "tokenizer_consistency"
        try:
            ref_vocab = getattr(ref_tokenizer, "vocab_size", None) or len(
                getattr(ref_tokenizer, "get_vocab", lambda: {})()
            )
            quant_vocab = getattr(quant_tokenizer, "vocab_size", None) or len(
                getattr(quant_tokenizer, "get_vocab", lambda: {})()
            )
            ref_special = {
                getattr(ref_tokenizer, tok, None)
                for tok in ("bos_token", "eos_token", "pad_token")
            }
            quant_special = {
                getattr(quant_tokenizer, tok, None)
                for tok in ("bos_token", "eos_token", "pad_token")
            }
            if ref_vocab != quant_vocab:
                return CheckItem(name, CHECK_FAIL,
                                 f"vocab size mismatch: ref={ref_vocab} quant={quant_vocab}",
                                 evidence=f"ref_vocab={ref_vocab} quant_vocab={quant_vocab}")
            diff = ref_special.symmetric_difference(quant_special)
            # None 元素 (某 token 不存在) 不计为差异；只比较都存在的特殊 token
            real_diff = {d for d in diff if d is not None}
            if real_diff:
                return CheckItem(name, CHECK_WARN,
                                 f"special tokens differ: {real_diff}",
                                 evidence=str(sorted(map(str, real_diff))))
            return CheckItem(name, CHECK_PASS,
                             "tokenizer consistent (vocab + special tokens match)",
                             evidence=f"vocab_size={ref_vocab}")
        except Exception as e:  # noqa: BLE001 - 检查不应本身崩溃阻塞流程
            return CheckItem(name, CHECK_SKIP, f"tokenizer check skipped: {e}",
                             evidence=str(e))

    def check_input_ids_consistency(self, ref_ids, quant_ids) -> CheckItem:
        """确认两次推理输入的 input_ids 完全一致。"""
        name = "input_ids_consistency"
        try:
            a = _tensor_view(ref_ids).to(torch.long)
            b = _tensor_view(quant_ids).to(torch.long)
            if a.shape != b.shape:
                return CheckItem(name, CHECK_FAIL,
                                 f"shape mismatch: ref={tuple(a.shape)} quant={tuple(b.shape)}",
                                 evidence=f"ref_shape={tuple(a.shape)} quant_shape={tuple(b.shape)}")
            n_diff = int((a != b).sum().item())
            total = a.numel()
            if n_diff:
                return CheckItem(name, CHECK_FAIL,
                                 f"{n_diff}/{total} token ids differ",
                                 evidence=f"n_diff={n_diff}/{total}")
            return CheckItem(name, CHECK_PASS, "input ids identical",
                             evidence=f"n_tokens={total}")
        except Exception as e:  # noqa: BLE001
            return CheckItem(name, CHECK_SKIP, f"input_ids check skipped: {e}",
                             evidence=str(e))

    def check_generation_config(self, ref_config, quant_config) -> CheckItem:
        """对比 generation_config 关键采样参数 (seed/temperature/top_p/top_k 等)。

        ref_config / quant_config: dict 或任意可通过 .get() 访问的对象。
        """
        name = "generation_config"
        keys = ["seed", "temperature", "top_p", "top_k", "do_sample", "max_new_tokens"]
        try:
            def _get(c, k):
                if c is None:
                    return None
                if isinstance(c, dict):
                    return c.get(k)
                return getattr(c, k, None)
            diffs = []
            for k in keys:
                rv, qv = _get(ref_config, k), _get(quant_config, k)
                if rv != qv:
                    diffs.append(f"{k}: ref={rv} quant={qv}")
            if diffs:
                return CheckItem(name, CHECK_WARN,
                                 f"generation config differs: {'; '.join(diffs)}",
                                 evidence="; ".join(diffs))
            return CheckItem(name, CHECK_PASS, "generation config consistent",
                             evidence=f"checked keys={keys}")
        except Exception as e:  # noqa: BLE001
            return CheckItem(name, CHECK_SKIP, f"generation config check skipped: {e}",
                             evidence=str(e))

    # ---- 数值合法性 ----

    def check_nan_inf(self, tensor: Any, name_suffix: str = "") -> CheckItem:
        """检查单个 tensor 不含 NaN/Inf。"""
        name = "nan_inf" + (f":{name_suffix}" if name_suffix else "")
        try:
            t = _tensor_view(tensor).float()
            total = t.numel()
            if total == 0:
                return CheckItem(name, CHECK_SKIP, "empty tensor",
                                 evidence="numel=0")
            n_nan = int(torch.isnan(t).sum().item())
            n_inf = int(torch.isinf(t).sum().item())
            n_bad = n_nan + n_inf
            if n_bad:
                return CheckItem(name, CHECK_FAIL,
                                 f"{n_bad}/{total} elements are NaN/Inf "
                                 f"(nan={n_nan}, inf={n_inf})",
                                 evidence=f"nan={n_nan} inf={n_inf} total={total}")
            return CheckItem(name, CHECK_PASS, "no NaN/Inf",
                             evidence=f"total={total}")
        except Exception as e:  # noqa: BLE001
            return CheckItem(name, CHECK_SKIP, f"nan/inf check skipped: {e}",
                             evidence=str(e))

    # ---- 流程/装置有效性 ----

    def _collect_hook_hits(self, hook_manager: Any) -> Dict[str, int]:
        """从 dict 或对象收集 hook 命中计数。"""
        hits: Dict[str, int] = {}
        if isinstance(hook_manager, dict):
            for hk, hv in hook_manager.items():
                hits[hk] = int(hv) if isinstance(hv, (int, float)) else (
                    len(hv) if hasattr(hv, "__len__") else 0
                )
            return hits
        # 对象: 收集各种可能属性
        for attr in ("hit_count", "hits", "captured", "records"):
            val = getattr(hook_manager, attr, None)
            if val is None:
                continue
            if isinstance(val, dict):
                for hk, hv in val.items():
                    hits[hk] = len(hv) if hasattr(hv, "__len__") else int(hv or 0)
            elif isinstance(val, (int, float)):
                hits[attr] = int(val)
            elif hasattr(val, "__len__"):
                hits[attr] = len(val)
        return hits

    def check_hook_hit(self, hook_manager: Any) -> CheckItem:
        """确认 hook 真正命中过 (有捕获数据)，而非静默未触发。

        hook_manager: 暴露 hit_count / captured / records 等属性之一的对象，
                      或一个 dict (key=hook 名, value=命中次数/数据量)。
        """
        name = "hook_hit"
        try:
            hits = self._collect_hook_hits(hook_manager)
            if not hits:
                return CheckItem(name, CHECK_SKIP, "no hook stats available",
                                 evidence="hook_manager exposes no hit/records attr")
            total = sum(hits.values())
            zero_hooks = [k for k, v in hits.items() if v == 0]
            if total == 0:
                return CheckItem(name, CHECK_FAIL,
                                 f"all hooks unhit: {list(hits)}",
                                 evidence=str(hits))
            if zero_hooks:
                return CheckItem(name, CHECK_WARN,
                                 f"some hooks unhit: {zero_hooks} "
                                 f"(total hits={total})",
                                 evidence=str(hits))
            return CheckItem(name, CHECK_PASS, f"all hooks hit (total={total})",
                             evidence=str(hits))
        except Exception as e:  # noqa: BLE001
            return CheckItem(name, CHECK_SKIP, f"hook hit check skipped: {e}",
                             evidence=str(e))

    def check_patch_recovery(self, before: Any, after: Any) -> CheckItem:
        """确认 patch (monkey-patch/forward 替换) 执行后被还原。

        通过比较 before/after 是否指向同一对象或等价判断。
        before/after: 可比较的对象 (nn.Module forward / callable / tensor)。
        """
        name = "patch_recovery"
        try:
            # callable / forward 等对象身份比较
            same = before is after
            if not same and hasattr(before, "__code__") and hasattr(after, "__code__"):
                same = before.__code__ is after.__code__
            if not same and isinstance(before, (int, float, str)):
                same = before == after
            if same:
                return CheckItem(name, CHECK_PASS, "patch restored (before == after)",
                                 evidence=f"type(before)={type(before).__name__}")
            return CheckItem(name, CHECK_FAIL,
                             "patch NOT restored: before != after after teardown",
                             evidence=f"before={type(before).__name__} after={type(after).__name__}")
        except Exception as e:  # noqa: BLE001
            return CheckItem(name, CHECK_SKIP, f"patch recovery check skipped: {e}",
                             evidence=str(e))

    def check_run_stability(self, result1: Any, result2: Any) -> CheckItem:
        """确认两次相同输入的运行结果稳定 (结果一致或差异极小)。

        支持 tensor / 数值 / dict(含 'logits' 或可 flatten 值)。
        """
        name = "run_stability"
        try:
            def _flatten(r: Any) -> torch.Tensor:
                if isinstance(r, torch.Tensor):
                    return r.detach().cpu().float().flatten()
                if isinstance(r, dict) and "logits" in r:
                    return _flatten(r["logits"])
                if isinstance(r, (int, float)):
                    return torch.tensor([float(r)])
                return torch.as_tensor(r, dtype=torch.float32).flatten()
            a = _flatten(result1)
            b = _flatten(result2)
            if a.shape != b.shape:
                return CheckItem(name, CHECK_FAIL,
                                 f"result shape differs across runs: {tuple(a.shape)} vs {tuple(b.shape)}",
                                 evidence=f"shape1={tuple(a.shape)} shape2={tuple(b.shape)}")
            max_abs = float((a - b).abs().max().item()) if a.numel() else 0.0
            # 用相对阈值: 1e-5 (fp32 级别可复现) 放宽到 1e-3 允许算子非确定性
            tol = 1e-3
            if not math.isfinite(max_abs):
                return CheckItem(name, CHECK_FAIL,
                                 "non-finite difference across runs",
                                 evidence=f"max_abs={max_abs}")
            if max_abs <= tol:
                return CheckItem(name, CHECK_PASS,
                                 f"runs stable (max_abs_diff={max_abs:.3e} <= {tol:.0e})",
                                 evidence=f"max_abs_diff={max_abs}")
            return CheckItem(name, CHECK_WARN,
                             f"runs differ: max_abs_diff={max_abs:.3e} > {tol:.0e}",
                             evidence=f"max_abs_diff={max_abs}")
        except Exception as e:  # noqa: BLE001
            return CheckItem(name, CHECK_SKIP, f"stability check skipped: {e}",
                             evidence=str(e))

    def check_meta_residual(self, model: nn.Module) -> CheckItem:
        """确认加载后没有参数仍停留在 meta device (未真正加载权重)。

        meta 残留意味着权重是 shape-only 占位，forward 会段错误/随机值。
        """
        name = "meta_residual"
        try:
            residual = []
            n_params = 0
            for pn, p in model.named_parameters(recurse=True):
                n_params += 1
                if p.is_meta:
                    residual.append(pn)
            if not n_params:
                return CheckItem(name, CHECK_SKIP, "model has no parameters to check",
                                 evidence="n_params=0")
            if residual:
                return CheckItem(name, CHECK_FAIL,
                                 f"{len(residual)} params still on meta device",
                                 evidence=str(residual[:20]))
            return CheckItem(name, CHECK_PASS, f"no meta residual ({n_params} params)",
                             evidence=f"n_params={n_params}")
        except Exception as e:  # noqa: BLE001
            return CheckItem(name, CHECK_SKIP, f"meta residual check skipped: {e}",
                             evidence=str(e))

    def check_weight_device_dtype(self, model: nn.Module,
                                  expected_device: Optional[str] = None,
                                  expected_dtype: Optional[Any] = None) -> CheckItem:
        """确认权重在期望设备上、dtype 正确。

        若不传 expected_device/dtype，则仅校验所有参数设备/dtype 一致 (无错位)。
        """
        name = "weight_device_dtype"
        try:
            devices = set()
            dtypes = set()
            n_params = 0
            for _, p in model.named_parameters(recurse=True):
                if p.is_meta:
                    continue
                n_params += 1
                devices.add(str(p.device))
                dtypes.add(str(p.dtype))
            if not n_params:
                return CheckItem(name, CHECK_SKIP, "no materialized params",
                                 evidence="n_params=0")
            issues = []
            if expected_device is not None:
                # 归一化: 'cpu'/'CPU'/'torch.device cpu' -> 'cpu', 'npu:0' 等
                exp_dev = str(expected_device).replace("torch.device", "").strip(" ()").lower()
                bad_devs = [d for d in devices if d != exp_dev]
                if bad_devs:
                    issues.append(f"device mismatch: expected {exp_dev}, got {sorted(bad_devs)}")
            else:
                if len(devices) > 1:
                    issues.append(f"params span multiple devices: {sorted(devices)}")
            if expected_dtype is not None:
                # 归一化为 torch.dtype 比较, 兼容 'float32'/'torch.float32'/torch.float32
                exp_dt = _normalize_dtype(expected_dtype)
                bad_dts = sorted({d for d in dtypes if _dtype_from_str(d) != exp_dt})
                if bad_dts:
                    issues.append(f"dtype mismatch: expected {exp_dt}, got {bad_dts}")
            # 量化模型可能合法地有多种 dtype (int8 weight + fp16 scale); 无显式期望时仅报错位
            if issues:
                return CheckItem(name, CHECK_WARN,
                                 "; ".join(issues),
                                 evidence=f"devices={sorted(devices)} dtypes={sorted(dtypes)}")
            return CheckItem(name, CHECK_PASS,
                             f"device/dtype ok (devices={sorted(devices)}, dtypes={sorted(dtypes)})",
                             evidence=f"devices={sorted(devices)} dtypes={sorted(dtypes)}")
        except Exception as e:  # noqa: BLE001
            return CheckItem(name, CHECK_SKIP, f"device/dtype check skipped: {e}",
                             evidence=str(e))


__all__ = [
    "RunStatus",
    "CheckItem",
    "ValidityChecker",
    "CHECK_PASS", "CHECK_WARN", "CHECK_FAIL", "CHECK_SKIP",
    "worst_status", "aggregate_overall",
]
