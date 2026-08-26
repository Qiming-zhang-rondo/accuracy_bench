"""
Boundary 定界 (L0.5) — 框架 vs 权重/量化

从 inference_check.py 拆分而来, 保持向后兼容 (inference_check re-export 全部符号)。

角色定位：先于 L1/L2。用 Transformers 原生 forward/generate 跑反量化后的
quant 权重（+可选 ref BF16 权重）→ 判断 Bad Case 在原生框架中是否复现：
  - 原生 Transformers 复现  → 量化权重/模型产物问题
  - 原生 Transformers 不复现, 部署框架复现 → 推理框架问题

输出 BoundaryResult，明确区分五类：
  WEIGHT_OR_QUANTIZATION / INFERENCE_FRAMEWORK / BOTH
      / INCONCLUSIVE / INVALID_RUN

入口：
  - 编程式：from accuracy_checker.boundary_check import run_boundary
            result = run_boundary(quant_model_path=..., ...)
  - CLI：   python -m accuracy_checker.inference_check --mode boundary \
            --model_path <quant> --ref_model_path <bf16> --prompt "..." \
            --framework_bad_output "..."
  - 上层 CLI：run_accuracy_check.py --mode boundary（由 Agent D 接线，
            调本模块 run_boundary API）
"""

from __future__ import annotations

import json
import logging
import os
import tempfile as _tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ----- boundary 分类常量 -----
WEIGHT_OR_QUANTIZATION = "WEIGHT_OR_QUANTIZATION"
INFERENCE_FRAMEWORK = "INFERENCE_FRAMEWORK"
BOTH = "BOTH"
INCONCLUSIVE = "INCONCLUSIVE"
INVALID_RUN = "INVALID_RUN"


@dataclass
class BoundaryResult:
    """定界结构化结果。

    字段对应 docs/proposals/boundary_recovery.md §1 与 Agent A 验收要求：
      framework_badcase_reproduced / transformers_badcase_reproduced /
      boundary_result / evidence / limitations。
    """
    framework_name: Optional[str]            # 部署框架名 (vllm/mindie/...)
    framework_badcase_reproduced: Optional[bool]  # 部署框架是否复现 (None=未知)
    transformers_badcase_reproduced: Optional[bool]  # 原生 Transformers(quant) 是否复现
    ref_badcase_reproduced: Optional[bool]   # 原生 Transformers(ref BF16) 是否复现 (None=没跑)
    boundary_result: str                    # 五分类之一
    evidence: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)


def normalize_request_payload(data: Any) -> Dict[str, Any]:
    """校验并规范化 OpenAI/vLLM Chat 请求；messages 数组可直接简写。"""
    if isinstance(data, list):
        data = {"messages": data}
    if not isinstance(data, dict):
        raise ValueError("完整请求必须是 JSON object，或直接传 messages 数组")
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("完整请求的 messages 必须是非空数组")
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ValueError(f"messages[{index}] 必须是含 role 的 object")
        if "content" not in message and not message.get("tool_calls"):
            raise ValueError(f"messages[{index}] 缺少 content/tool_calls")
    for key in ("max_tokens", "max_new_tokens"):
        if key in data and (not isinstance(data[key], int) or data[key] <= 0):
            raise ValueError(f"{key} 必须是正整数")
    if "tools" in data and not isinstance(data["tools"], list):
        raise ValueError("tools 必须是数组")
    return dict(data)


def parse_request_json(raw: str) -> Dict[str, Any]:
    """解析命令行直接粘贴的完整请求 JSON。"""
    try:
        return normalize_request_payload(json.loads(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"request_json 不是合法 JSON: {exc}") from exc


# ----- badcase 启发式判定 -----

def repeat_4gram_ratio(text: str, n: int = 4) -> float:
    """字符级 4-gram 重复占比 (0~1)。值越高越像乱码/重复。

    对中文友好（按字符滑窗）；空文本返回 0。
    """
    if len(text) <= n:
        return 0.0
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    if not grams:
        return 0.0
    from collections import Counter
    counts = Counter(grams)
    # 重复 = 出现次数 >= 2 的 gram 总数占比
    repeated = sum(c for c in counts.values() if c >= 2)
    return repeated / len(grams)


def nonprintable_ratio(text: str) -> float:
    """非可打印字符占比 (乱码倾向)。CJK/常见标点/拉丁字母数字算可打印。"""
    if not text:
        return 0.0
    bad = 0
    for ch in text:
        # 控制字符、替换符、罕见私用区等视为乱码倾向
        code = ord(ch)
        if code < 32 or code in (0x7f,) or 0xe000 <= code <= 0xf8ff:
            bad += 1
    return bad / len(text)


def _detect_badcase_explicit_assertions(text: str, p: Dict[str, Any],
                                       metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """显式断言类 pattern (expected_substr/forbidden_substr/expected_json) 判定。
    返回 None 表示无定论、继续走启发式。"""
    if "expected_substr" in p:
        if p["expected_substr"] not in text:
            return {"reproduced": True,
                    "reason": f"未出现期待子串 {p['expected_substr']!r}",
                    "metrics": metrics}
        return {"reproduced": False,
                "reason": f"含期待子串 {p['expected_substr']!r}",
                "metrics": metrics}
    if "forbidden_substr" in p:
        # 出现 → bad; 不出现 → 继续走启发式 (absence 非定论)
        if p["forbidden_substr"] in text:
            return {"reproduced": True,
                    "reason": f"出现禁用子串 {p['forbidden_substr']!r}",
                    "metrics": metrics}
    if p.get("expected_json"):
        try:
            import json as _json
            _json.loads(text)
        except Exception as e:
            return {"reproduced": True, "reason": f"非合法 JSON: {e}", "metrics": metrics}
        return {"reproduced": False, "reason": "合法 JSON", "metrics": metrics}
    return None


def _detect_badcase_heuristics(text: str, p: Dict[str, Any], output_tokens: int,
                              metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """启发式判定 (长度门禁 + 4-gram/乱码比)。
    返回 None 表示通过启发式未发现坏 case 特征。"""
    # 长度门禁 (仅对启发式默认生效)
    min_tokens = p.get("min_output_tokens", 10)
    if output_tokens and output_tokens < min_tokens:
        return {"reproduced": None,
                "reason": f"输出过短 ({output_tokens}<{min_tokens} tokens)",
                "metrics": metrics}

    rep = repeat_4gram_ratio(text)
    npr = nonprintable_ratio(text)
    metrics["repeat_4gram_ratio"] = round(rep, 4)
    metrics["nonprintable_ratio"] = round(npr, 4)

    if "repeat_4gram_max" in p and rep > p["repeat_4gram_max"]:
        return {"reproduced": True, "reason": f"4-gram 重复比 {rep:.2f} 超阈值 "
                f"{p['repeat_4gram_max']}", "metrics": metrics}
    if "nonprintable_max" in p and npr > p["nonprintable_max"]:
        return {"reproduced": True, "reason": f"非可打印字符比 {npr:.2f} 超阈值 "
                f"{p['nonprintable_max']}", "metrics": metrics}
    # 默认启发式 (未显式给阈值)
    if "repeat_4gram_max" not in p and rep > 0.5:
        return {"reproduced": True,
                "reason": f"4-gram 重复比 {rep:.2f} (>0.5 默认)",
                "metrics": metrics}
    if "nonprintable_max" not in p and npr > 0.3:
        return {"reproduced": True,
                "reason": f"非可打印字符比 {npr:.2f} (>0.3 默认)",
                "metrics": metrics}
    return None


def detect_badcase(
    text: str,
    thinking_truncated: bool = False,
    output_tokens: int = 0,
    pattern: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """对单条生成结果做 badcase 判定。

    Returns:
        {reproduced: bool|None, reason: str, metrics: dict}
        reproduced=None 表示证据不足以判定 (INCONCLUSIVE)。

    pattern 支持的字段 (全部可选、缺省走启发式默认):
      min_output_tokens:    output_tokens < 该值 → INCONCLUSIVE (无法判定)
      repeat_4gram_max:     4-gram 重复比超过该值 → bad (默认 0.5)
      nonprintable_max:     乱码字符比超过该值 → bad (默认 0.3)
      expected_substr:      期待文本含此子串；不含 → bad (显式断言, 不受 min_output_tokens 限制)
      forbidden_substr:     出现此子串 → bad (显式断言)
      expected_json:        True 时尝试 json.loads；失败 → bad (显式断言)
    """
    p = pattern or {}
    metrics: Dict[str, Any] = {
        "output_tokens": output_tokens,
        "thinking_truncated": thinking_truncated,
    }

    if not text or not text.strip():
        return {"reproduced": None, "reason": "空生成, 无法判定", "metrics": metrics}

    # 显式断言类 pattern 最先 (调用方明确给出期待/禁用 → 不受长度门禁限制,
    # 且成功/失败都给出定论, 不再 fall through 到启发式)
    result = _detect_badcase_explicit_assertions(text, p, metrics)
    if result is not None:
        return result

    # 启发式 (长度门禁 + 4-gram/乱码比)
    result = _detect_badcase_heuristics(text, p, output_tokens, metrics)
    if result is not None:
        return result

    if thinking_truncated:
        return {"reproduced": None, "reason": "thinking 被截断且未发现明确复读, 无法判定",
                "metrics": metrics}

    return {"reproduced": False, "reason": "通过启发式检测, 未发现坏 case 特征",
            "metrics": metrics}


# ----- Transformers 原生路径 (复用 hf_inference_check 加载链) -----

def _run_transformers_on_path(
    model_path: str,
    devices: str,
    dtype: str,
    messages: List[Dict],
    thinking: str,
    max_new_tokens: int,
    framework_gen_config: Optional[Dict],
    verbose: bool,
    label: str,
    num_runs: int = 1,
    concurrency: int = 1,
    stop_on_first_badcase: bool = False,
    bad_pattern: Optional[Dict[str, Any]] = None,
    expert_chunk_size: Optional[int] = None,
    prefill_parallel: str = "pp",
) -> List[Dict[str, Any]]:
    """在给定 model_path 上跑原生 Transformers generate, 复用 hf_inference_check 加载链。

    通过临时 prompt_file 把单轮对话喂给 hf_inference_check, 避免重复实现
    skeleton/distribute/dequantize 逻辑。返回全部批次的逐次解码结果。

    若 do_sample 启用, hf_inference_check 当前固定 greedy, 这里记录 limitation。
    """
    # 延迟 import 避免循环
    from .inference_check import hf_inference_check

    # 临时写入单对话 prompt 文件 (load_prompt_file 接受 dict 带 messages+tools)
    payload = {"messages": messages}
    if framework_gen_config and framework_gen_config.get("tools"):
        payload["tools"] = framework_gen_config["tools"]
    with _tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as tf:
        json.dump(payload, tf, ensure_ascii=False)
        prompt_file = tf.name

    try:
        # framework_gen_config 对齐: 当前 hf_inference_check 只暴露 max_new_tokens/
        # thinking(do_sample=False), 其余 (temperature/top_p/top_k) 无法对齐 → 记录
        def _is_bad(run: Dict[str, Any]) -> bool:
            text = run.get("raw_text") or run.get("generated", "")
            detected = detect_badcase(
                text, run.get("thinking_truncated", False),
                run.get("output_tokens", 0), bad_pattern,
            )
            return detected["reproduced"] is True

        results = hf_inference_check(
            model_path=model_path,
            devices=devices,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            prompt_file=prompt_file,
            skip_ppl=True,
            thinking=thinking,
            verbose=verbose,
            num_runs=num_runs,
            concurrency=concurrency,
            stop_predicate=_is_bad if stop_on_first_badcase else None,
            expert_chunk_size=expert_chunk_size,
            prefill_parallel=prefill_parallel,
        )
    finally:
        try:
            os.remove(prompt_file)
        except OSError:
            pass

    if not results:
        raise RuntimeError(f"[{label}] hf_inference_check 未返回结果")

    for run in results:
        run["__label__"] = label
    return results


def _summarize_transformers_runs(
    outputs: List[Dict[str, Any]], requested_runs: int, concurrency: int,
    bad_pattern: Optional[Dict[str, Any]],
) -> Tuple[Optional[bool], Dict[str, Any]]:
    """逐次检测并汇总偶现 bad case；任一次命中即视为复现。"""
    runs = []
    bad_count = inconclusive_count = 0
    first_bad_run = None
    for index, output in enumerate(outputs, 1):
        text = output.get("raw_text") or output.get("generated", "")
        detected = detect_badcase(
            text, output.get("thinking_truncated", False),
            output.get("output_tokens", 0), bad_pattern,
        )
        if detected["reproduced"] is True:
            bad_count += 1
            first_bad_run = first_bad_run or output.get("run_index", index)
        elif detected["reproduced"] is None:
            inconclusive_count += 1
        runs.append({
            "run_index": output.get("run_index", index),
            "output_full": text,
            "answer": output.get("generated", ""),
            "thinking": output.get("thinking", ""),
            "thinking_truncated": output.get("thinking_truncated", False),
            "input_tokens": output.get("input_tokens"),
            "output_tokens": output.get("output_tokens"),
            "batch_time": output.get("batch_time", output.get("time")),
            "detect": detected,
        })

    completed = len(runs)
    clean_count = completed - bad_count - inconclusive_count
    reproduced = True if bad_count else (None if inconclusive_count else False)
    representative = next(
        (r for r in runs if r["detect"]["reproduced"] is True),
        runs[0] if runs else {},
    )
    return reproduced, {
        "requested_runs": requested_runs,
        "completed_runs": completed,
        "stopped_early": completed < requested_runs,
        "concurrency": concurrency,
        "badcase_runs": bad_count,
        "clean_runs": clean_count,
        "inconclusive_runs": inconclusive_count,
        "badcase_rate": (bad_count / completed if completed else None),
        "first_badcase_run": first_bad_run,
        "output_preview": representative.get("output_full", "")[:200],
        "output_full": representative.get("output_full", ""),
        "runs": runs,
    }


def _check_tokenizer_consistency(
    quant_model_path: str, ref_model_path: Optional[str]
) -> Tuple[bool, str]:
    """检查 ref/quant tokenizer 是否一致 (vocab_size + 样例 round-trip)。

    仅对本地存在的路径做检查, 避免对不存在路径触发 HF Hub 网络解析 (会 hang)。
    """
    try:
        from transformers import AutoTokenizer
        if not os.path.isdir(quant_model_path):
            return True, (f"quant 路径非目录/不存在 ({quant_model_path}), "
                         "跳过 tokenizer 一致性检查")
        qt = AutoTokenizer.from_pretrained(
            quant_model_path, trust_remote_code=True, local_files_only=True)
        if ref_model_path is None:
            return True, "无 ref, 跳过 tokenizer 一致性检查"
        if not os.path.isdir(ref_model_path):
            return True, (f"ref 路径非目录/不存在 ({ref_model_path}), "
                         "跳过 tokenizer 一致性检查")
        rt = AutoTokenizer.from_pretrained(
            ref_model_path, trust_remote_code=True, local_files_only=True)
        if qt.vocab_size != rt.vocab_size:
            return False, (f"vocab_size 不一致: quant={qt.vocab_size} "
                           f"ref={rt.vocab_size}")
        probe = "你好世界 accuracy_check boundary 一致性测试 1234"
        q_ids = qt(probe)["input_ids"]
        r_ids = rt(probe)["input_ids"]
        if q_ids != r_ids:
            return False, "样例 tokenize 结果不一致, tokenizer 版本差异"
        return True, f"vocab_size={qt.vocab_size}, 样例 tokenize 一致"
    except Exception as e:
        return False, f"tokenizer 检查异常 (非致命): {e}"


def classify_boundary(
    framework_reproduced: Optional[bool],
    transformers_quant_reproduced: Optional[bool],
    ref_reproduced: Optional[bool],
) -> str:
    """根据三方证据给出五分类结果。

    判定矩阵 (T=复现, F=不复现, ?=证据不足):
      fw=T, tf_q=T, ref=F   → WEIGHT_OR_QUANTIZATION (quant 回归, ref 正常)
      fw=T, tf_q=F          → INFERENCE_FRAMEWORK (原生干净, 部署坏)
      fw=T, tf_q=T, ref=T   → BOTH (Base/产物本身也复现, 非单纯 quant 回归)
      其它                 → INCONCLUSIVE
      任何运行未完成        → INVALID_RUN (调用方在 evidence.error 标注)
    """
    fw, tq, ref = framework_reproduced, transformers_quant_reproduced, ref_reproduced

    if fw is True and tq is True:
        # tf_q 复现即说明 quant 侧出了问题; 是否同时是 base-model 本征看 ref
        if ref is True:
            return BOTH
        if ref is False:
            return WEIGHT_OR_QUANTIZATION
        # ref 未跑 — quant 复现已足以下结论 (ref 可选)
        return WEIGHT_OR_QUANTIZATION

    if fw is True and tq is False:
        return INFERENCE_FRAMEWORK

    return INCONCLUSIVE


def run_boundary(
    quant_model_path: str,
    devices: str = "npu:0",
    *,
    ref_model_path: Optional[str] = None,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict]] = None,
    prompt_file: Optional[str] = None,
    request_payload: Optional[Dict[str, Any]] = None,
    max_new_tokens: Optional[int] = None,
    dtype: str = "bfloat16",
    thinking: str = "chat",
    # 框架侧 (部署框架已知坏 case)
    framework_name: Optional[str] = None,
    framework_bad_output: Optional[str] = None,
    framework_bad_reproduced: Optional[bool] = None,
    framework_gen_config: Optional[Dict] = None,
    # badcase 特征
    bad_pattern: Optional[Dict] = None,
    # 运行开关
    run_ref: bool = True,
    run_quant: bool = True,
    verbose: bool = True,
    num_runs: int = 1,
    concurrency: int = 1,
    stop_on_first_badcase: bool = False,
    expert_chunk_size: Optional[int] = None,
    prefill_parallel: str = "pp",
) -> BoundaryResult:
    """定界主入口 — 区分 WEIGHT / INFERENCE_FRAMEWORK / BOTH / INCONCLUSIVE / INVALID_RUN。

    必给的 Bad Case 来源 (三选一, 优先级 prompt_file > messages > prompt):
      prompt       : str, 单轮 plain text (自动包为 user message)
      messages     : List[Dict], chat-template 对话
      prompt_file  : vLLM 请求格式或对话列表 JSON

    框架侧 Bad Case 来源 (二选一, 推荐 framework_bad_output):
      framework_bad_output      : 部署框架实际生成的坏文本 (对其跑 detect_badcase)
      framework_bad_reproduced   : 直接给 bool (True=已确认框架复现)

    ref (BF16) 可选: 给了 run_ref=True 会跑一次原生参考, 用于区分"quant 回归"
    vs"base-model 本征"(→ BOTH); 不给则只比 framework vs transformers(quant)。

    生效条件:
      - 量化模型 → hf_inference_check 内 NPU 加速反量化路径 (CPU fallback 未回迁)
      - 非量化模型 → 直接 safetensors 加载
    """
    request_payload = dict(request_payload or {})
    if prompt_file and not request_payload:
        try:
            with open(prompt_file, encoding="utf-8") as f:
                loaded = json.load(f)
            request_payload = normalize_request_payload(loaded)
        except Exception:
            # 统一交给 _boundary_resolve_conversation 生成 INVALID_RUN。
            pass
    if request_payload:
        request_payload = normalize_request_payload(request_payload)
    if messages is None and isinstance(request_payload.get("messages"), list):
        messages = request_payload["messages"]
    if max_new_tokens is None:
        max_new_tokens = int(
            request_payload.get("max_new_tokens")
            or request_payload.get("max_tokens")
            or 1024
        )
    request_gen_config = {
        key: request_payload[key]
        for key in (
            "temperature", "top_p", "top_k", "repetition_penalty",
            "frequency_penalty", "presence_penalty", "seed", "tools",
        )
        if key in request_payload
    }
    if framework_gen_config is None:
        framework_gen_config = request_gen_config or None
    elif request_gen_config:
        framework_gen_config = {**request_gen_config, **framework_gen_config}

    # ---- resolve single conversation ----
    messages, invalid_result = _boundary_resolve_conversation(
        messages, prompt, prompt_file, framework_name,
        framework_gen_config, framework_bad_reproduced)
    if invalid_result is not None:
        return invalid_result

    limitations: List[str] = []
    evidence: Dict[str, Any] = {
        "quant_model_path": quant_model_path,
        "ref_model_path": ref_model_path,
        "devices": devices,
        "dtype": dtype,
        "max_new_tokens": max_new_tokens,
        "thinking": thinking,
        "num_runs": num_runs,
        "concurrency": concurrency,
        "stop_on_first_badcase": stop_on_first_badcase,
        "expert_chunk_size": expert_chunk_size or 8,
        "prefill_parallel": prefill_parallel,
        "resident_experts": os.getenv("ACC_BOUNDARY_RESIDENT_EXPERTS", "1") != "0",
        "expert_cache_per_layer": int(
            os.getenv("ACC_BOUNDARY_EXPERT_CACHE_PER_LAYER", "16")),
        "generation_config": {
            "do_sample": False,  # hf_inference_check 当前固定 greedy
            "use_cache": True,
        },
        "framework_gen_config": framework_gen_config,
        "request_model": request_payload.get("model"),
        "messages_preview": (messages[-1].get("content", "")[:120]
                             if messages else ""),
    }

    # ---- tokenizer 一致性 ----
    _boundary_check_tokenizer(quant_model_path, ref_model_path, evidence, limitations)

    # ---- generation config 对齐 + seed ----
    _boundary_check_gen_config(framework_gen_config, evidence, limitations)

    # ---- 框架侧 reproduced ----
    fw_reproduced = _boundary_detect_framework(
        framework_bad_reproduced, framework_bad_output, bad_pattern,
        evidence, limitations)

    # ---- transformers(quant) 路径 ----
    tq_reproduced: Optional[bool] = None
    evidence["transformers_run"] = {}
    invalid_result, tq_reproduced = _boundary_run_quant(
        run_quant, quant_model_path, devices, dtype, messages, thinking,
        max_new_tokens, framework_gen_config, verbose, bad_pattern,
        evidence, limitations, framework_name, framework_bad_reproduced, fw_reproduced,
        num_runs, concurrency, stop_on_first_badcase, expert_chunk_size,
        prefill_parallel)
    if invalid_result is not None:
        return invalid_result

    # ---- transformers(ref) 路径 (可选) ----
    ref_reproduced = _boundary_run_ref(
        run_ref, ref_model_path, devices, dtype, messages, thinking,
        max_new_tokens, framework_gen_config, verbose, bad_pattern, evidence, limitations,
        num_runs, concurrency, stop_on_first_badcase, expert_chunk_size,
        prefill_parallel)

    # ---- 分类 ----
    result_kind = classify_boundary(fw_reproduced, tq_reproduced, ref_reproduced)

    return BoundaryResult(
        framework_name=framework_name,
        framework_badcase_reproduced=fw_reproduced,
        transformers_badcase_reproduced=tq_reproduced,
        ref_badcase_reproduced=ref_reproduced,
        boundary_result=result_kind,
        evidence=evidence,
        limitations=limitations,
    )


def _boundary_resolve_conversation(messages, prompt, prompt_file, framework_name,
                                  framework_gen_config, framework_bad_reproduced):
    """解析单条对话; 返回 (messages, invalid_result)。invalid_result 为 None 表示成功"""
    if messages is not None:
        return messages, None
    if prompt is None and prompt_file is None:
        return None, _invalid("缺少 prompt/messages/prompt_file", framework_name,
                              framework_gen_config, framework_bad_reproduced)
    if prompt_file:
        try:
            from .inference_check import load_prompt_file
            convs, _ = load_prompt_file(prompt_file)
            return convs[0], None
        except Exception as e:
            return None, _invalid(f"prompt_file 加载失败: {e}", framework_name,
                                  framework_gen_config, framework_bad_reproduced)
    return [{"role": "user", "content": prompt}], None


def _boundary_check_tokenizer(quant_model_path: str, ref_model_path: Optional[str],
                             evidence: Dict[str, Any], limitations: List[str]):
    """检查 ref/quant tokenizer 一致性"""
    tok_ok, tok_msg = _check_tokenizer_consistency(quant_model_path, ref_model_path)
    evidence["tokenizer_check"] = {"ok": tok_ok, "detail": tok_msg}
    if not tok_ok:
        limitations.append(f"tokenizer 不一致: {tok_msg}")


def _boundary_check_gen_config(framework_gen_config: Optional[Dict],
                                evidence: Dict[str, Any],
                                limitations: List[str]):
    """generation config 对齐 + seed 一致性检查"""
    if framework_gen_config:
        # hf_inference_check 只对齐了 max_new_tokens + thinking(do_sample=False)
        unsupported = []
        for k in ("temperature", "top_p", "top_k", "repetition_penalty",
                  "frequency_penalty", "presence_penalty"):
            if k in framework_gen_config and framework_gen_config[k] not in (None, 1.0, 0):
                unsupported.append(f"{k}={framework_gen_config[k]}")
        if unsupported:
            limitations.append(
                "原生 Transformers 无法对齐框架 gen_config: "
                + ", ".join(unsupported)
                + " (hf_inference_check 走 greedy)"
            )
        evidence["framework_gen_diff"] = unsupported

    # ---- seed ----
    # do_sample=False 时 greedy 本身确定性; 如框架用 sampling, 记录 limitation
    fw_sampling = bool(framework_gen_config and (
        framework_gen_config.get("temperature", 0) not in (None, 0, 1.0)
        or framework_gen_config.get("top_p") not in (None, 1.0)
        or framework_gen_config.get("top_k", -1) != -1
    ))
    if fw_sampling:
        limitations.append(
            "框架侧疑似用 sampling; 原生 Transformers 走 greedy, 复现性可能不严格"
        )
    evidence["seed_consistency"] = ("greedy 确定性" if not fw_sampling
                                    else "框架 sampling, seed 不可严格对齐")


def _boundary_detect_framework(framework_bad_reproduced: Optional[bool],
                              framework_bad_output: Optional[str],
                              bad_pattern: Optional[Dict],
                              evidence: Dict[str, Any],
                              limitations: List[str]) -> Optional[bool]:
    """框架侧 reproduced 判定"""
    if framework_bad_reproduced is not None:
        evidence["framework_reproduced_reason"] = "调用方显式指定"
        return framework_bad_reproduced
    if framework_bad_output is not None:
        d_fw = detect_badcase(framework_bad_output,
                              thinking_truncated=False,
                              output_tokens=0,
                              pattern=bad_pattern)
        fw_reproduced = d_fw["reproduced"]
        evidence["framework_bad_output_preview"] = framework_bad_output[:200]
        evidence["framework_detect"] = d_fw
        evidence["framework_reproduced_reason"] = d_fw["reason"]
        return fw_reproduced
    fw_reproduced = None
    evidence["framework_reproduced_reason"] = "未提供框架侧坏输出, 框架复现性未知"
    limitations.append("未提供 framework_bad_output, 无法独立验证框架侧是否复现")
    return fw_reproduced


def _boundary_run_quant(run_quant: bool, quant_model_path: str, devices: str,
                       dtype: str, messages: List[Dict], thinking: str,
                       max_new_tokens: int, framework_gen_config: Optional[Dict],
                       verbose: bool, bad_pattern: Optional[Dict],
                       evidence: Dict[str, Any], limitations: List[str],
                       framework_name: Optional[str],
                       framework_bad_reproduced: Optional[bool],
                       fw_reproduced: Optional[bool], num_runs: int,
                       concurrency: int,
                       stop_on_first_badcase: bool,
                       expert_chunk_size: Optional[int],
                       prefill_parallel: str = "pp") -> Tuple[Optional[BoundaryResult], Optional[bool]]:
    """transformers(quant) 路径运行; 成功时返回 (None, tq_reproduced),
    失败时返回 (_invalid 结果, None)"""
    if not run_quant:
        return None, None
    try:
        qt_outputs = _run_transformers_on_path(
            quant_model_path, devices, dtype, messages, thinking,
            max_new_tokens, framework_gen_config, verbose, label="quant",
            num_runs=num_runs, concurrency=concurrency,
            stop_on_first_badcase=stop_on_first_badcase, bad_pattern=bad_pattern,
            expert_chunk_size=expert_chunk_size, prefill_parallel=prefill_parallel)
        tq_reproduced, summary = _summarize_transformers_runs(
            qt_outputs, num_runs, concurrency, bad_pattern)
        evidence["transformers_run"]["quant"] = summary
        return None, tq_reproduced
    except Exception as e:
        return _invalid(f"transformers(quant) 运行失败: {e}", framework_name,
                        framework_gen_config, framework_bad_reproduced,
                        fw_reproduced=fw_reproduced, tq_reproduced=None,
                        evidence=evidence, limitations=limitations), None


def _boundary_run_ref(run_ref: bool, ref_model_path: Optional[str],
                     devices: str, dtype: str, messages: List[Dict],
                     thinking: str, max_new_tokens: int,
                     framework_gen_config: Optional[Dict], verbose: bool,
                     bad_pattern: Optional[Dict], evidence: Dict[str, Any],
                     limitations: List[str], num_runs: int, concurrency: int,
                     stop_on_first_badcase: bool,
                     expert_chunk_size: Optional[int],
                     prefill_parallel: str = "pp") -> Optional[bool]:
    """transformers(ref) 路径运行 (可选); 失败时记录到 limitations 不中止"""
    if not (run_ref and ref_model_path):
        return None
    try:
        ref_outputs = _run_transformers_on_path(
            ref_model_path, devices, dtype, messages, thinking,
            max_new_tokens, framework_gen_config, verbose, label="ref",
            num_runs=num_runs, concurrency=concurrency,
            stop_on_first_badcase=stop_on_first_badcase, bad_pattern=bad_pattern,
            expert_chunk_size=expert_chunk_size, prefill_parallel=prefill_parallel)
        ref_reproduced, summary = _summarize_transformers_runs(
            ref_outputs, num_runs, concurrency, bad_pattern)
        evidence["transformers_run"]["ref"] = summary
        return ref_reproduced
    except Exception as e:
        evidence["transformers_run"]["ref_error"] = str(e)
        limitations.append(f"ref 运行失败, BOTH 判定退化为只看 quant: {e}")
        return None


def _invalid(
    reason: str,
    framework_name: Optional[str],
    framework_gen_config: Optional[Dict],
    framework_bad_reproduced: Optional[bool],
    fw_reproduced: Optional[bool] = None,
    tq_reproduced: Optional[bool] = None,
    evidence: Optional[Dict] = None,
    limitations: Optional[List[str]] = None,
) -> BoundaryResult:
    e = evidence or {}
    e["error"] = reason
    return BoundaryResult(
        framework_name=framework_name,
        framework_badcase_reproduced=(fw_reproduced
                                     if fw_reproduced is not None
                                     else framework_bad_reproduced),
        transformers_badcase_reproduced=tq_reproduced,
        ref_badcase_reproduced=None,
        boundary_result=INVALID_RUN,
        evidence=e,
        limitations=(list(limitations) if limitations else [reason]),
    )


def boundary_result_to_dict(r: BoundaryResult) -> Dict[str, Any]:
    """供 Agent C 的 HTML/JSON schema 与 Agent D CLI 打印使用。"""
    return {
        "framework_name": r.framework_name,
        "framework_badcase_reproduced": r.framework_badcase_reproduced,
        "transformers_badcase_reproduced": r.transformers_badcase_reproduced,
        "ref_badcase_reproduced": r.ref_badcase_reproduced,
        "boundary_result": r.boundary_result,
        "evidence": r.evidence,
        "limitations": r.limitations,
    }


# ----- CLI helpers (供 inference_check.main 调用) -----

def run_boundary_cli(args):
    """--mode boundary CLI: 调 run_boundary 并打印结构化结果。"""
    bad_pattern: Optional[Dict[str, Any]] = None
    if args.repeat_4gram_max is not None or args.nonprintable_max is not None:
        bad_pattern = {}
        if args.repeat_4gram_max is not None:
            bad_pattern["repeat_4gram_max"] = args.repeat_4gram_max
        if args.nonprintable_max is not None:
            bad_pattern["nonprintable_max"] = args.nonprintable_max

    fw_repro = None
    if args.framework_bad_reproduced is not None:
        fw_repro = (args.framework_bad_reproduced == "true")

    request_payload = parse_request_json(args.request_json) if args.request_json else None
    result = run_boundary(
        quant_model_path=args.model_path,
        devices=args.devices,
        ref_model_path=args.ref_model_path,
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        request_payload=request_payload,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        thinking=args.thinking,
        framework_name=args.framework_name,
        framework_bad_output=args.framework_bad_output,
        framework_bad_reproduced=fw_repro,
        bad_pattern=bad_pattern,
        run_ref=(not args.no_ref),
        verbose=True,
        num_runs=args.num_runs,
        concurrency=args.concurrency,
        stop_on_first_badcase=args.stop_on_first_badcase,
        expert_chunk_size=getattr(args, "expert_chunk_size", None),
        prefill_parallel=getattr(args, "prefill_parallel", "pp"),
    )

    out = boundary_result_to_dict(result)
    if args.json_out:
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    else:
        logger.info("=" * 70)
        logger.info("  定界结果 (boundary)")
        logger.info("=" * 70)
        logger.info(f"  framework_name              : {out['framework_name']}")
        logger.info(f"  framework_badcase_reproduced : {out['framework_badcase_reproduced']}")
        logger.info(f"  transformers_badcase_reproduced : {out['transformers_badcase_reproduced']}")
        logger.info(f"  ref_badcase_reproduced      : {out['ref_badcase_reproduced']}")
        logger.info(f"  >>> boundary_result          : {out['boundary_result']}  <<<")
        if out.get("limitations"):
            logger.info("  limitations:")
            for lim in out["limitations"]:
                logger.info(f"    - {lim}")
        logger.info("=" * 70)
        print_boundary_verdict_explain(out["boundary_result"])


def print_boundary_verdict_explain(kind: str) -> None:
    explains = {
        "WEIGHT_OR_QUANTIZATION":
            "原生 Transformers 在 quant 模型上复现 Bad Case -> "
            "问题在量化权重/模型产物, 进 L1/L2 逐层定位",
        "INFERENCE_FRAMEWORK":
            "原生 Transformers 不复现, 部署框架复现 -> "
            "问题在推理框架 (算子融合/量化反算/KV cache/调度)",
        "BOTH":
            "quant 与 ref(BF16) 都在原生 Transformers 上复现 -> "
            "Base-model/产物本征特征, 非单纯 quant 回归, 需深挖",
        "INCONCLUSIVE":
            "证据不足 (thinking 截断/输出过短/框架侧未提供坏输出) -> 无法定界, 重跑或补证据",
        "INVALID_RUN":
            "运行未完成 (加载/反量化/generate 失败) -> 修运行环境后重来",
    }
    logger.info(f"  {kind}: {explains.get(kind, '未知')}")
    logger.info("  (自动判定仅供参考, 最终靠人眼对照原文)")
    logger.info("=" * 70)


__all__ = [
    "BoundaryResult",
    "WEIGHT_OR_QUANTIZATION", "INFERENCE_FRAMEWORK", "BOTH",
    "INCONCLUSIVE", "INVALID_RUN",
    "repeat_4gram_ratio", "nonprintable_ratio",
    "detect_badcase", "normalize_request_payload", "parse_request_json",
    "classify_boundary", "run_boundary", "boundary_result_to_dict",
    "run_boundary_cli", "print_boundary_verdict_explain",
]
