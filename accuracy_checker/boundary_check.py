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
import math
import os
import tempfile as _tempfile
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ----- boundary 分类常量 -----
WEIGHT_OR_QUANTIZATION = "WEIGHT_OR_QUANTIZATION"
INFERENCE_FRAMEWORK = "INFERENCE_FRAMEWORK"
BOTH = "BOTH"
INCONCLUSIVE = "INCONCLUSIVE"
INVALID_RUN = "INVALID_RUN"
INTERMITTENT_LOGITS_ALIGNED = "INTERMITTENT_LOGITS_ALIGNED"
INTERMITTENT_LOGITS_MISMATCH = "INTERMITTENT_LOGITS_MISMATCH"
INTERMITTENT_RANKING_SENSITIVE = "INTERMITTENT_RANKING_SENSITIVE"


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


def _request_thinking_mode(request_payload: Dict[str, Any], default: str) -> str:
    """Map OpenAI-style chat_template_kwargs.thinking to our CLI mode.

    DeepSeek requests commonly carry ``{"thinking": false}`` inside
    ``chat_template_kwargs``.  Without this bridge Boundary would silently
    use the CLI default (``chat``) and compare a different rendered prompt.
    Unknown/malformed values deliberately fall back to the explicit CLI mode.
    """
    kwargs = request_payload.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        return default
    value = kwargs.get("thinking", kwargs.get("enable_thinking"))
    if isinstance(value, bool):
        return "chat" if value else "none"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"chat", "on", "true", "1", "yes"}:
            return "chat"
        if normalized in {"none", "off", "false", "0", "no"}:
            return "none"
    return default


def _request_uses_sampling(config: Optional[Dict[str, Any]]) -> bool:
    if not config:
        return False
    if config.get("do_sample") is not None:
        return bool(config["do_sample"])
    temperature = config.get("temperature")
    if temperature is not None:
        return float(temperature) > 0
    return (
        config.get("top_p") not in (None, 1, 1.0)
        or config.get("top_k") not in (None, -1, 0)
    )


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
    glm_attn_query_block: Optional[int] = None,
    glm_attn_selected_block: Optional[int] = None,
    deepseek_v4_query_block: Optional[int] = None,
    deepseek_v4_key_block: Optional[int] = None,
    chat_template_mode: str = "auto",
    print_full_output: bool = False,
) -> List[Dict[str, Any]]:
    """在给定 model_path 上跑原生 Transformers generate, 复用 hf_inference_check 加载链。

    通过临时 prompt_file 把单轮对话喂给 hf_inference_check, 避免重复实现
    skeleton/distribute/dequantize 逻辑。返回全部批次的逐次解码结果。

    Sampling 参数由 framework_gen_config 对齐到 hf_inference_check。
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
            glm_attn_query_block=glm_attn_query_block,
            glm_attn_selected_block=glm_attn_selected_block,
            deepseek_v4_query_block=deepseek_v4_query_block,
            deepseek_v4_key_block=deepseek_v4_key_block,
            chat_template_mode=chat_template_mode,
            generation_config=framework_gen_config,
            print_full_output=print_full_output,
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
    glm_attn_query_block: Optional[int] = None,
    glm_attn_selected_block: Optional[int] = None,
    deepseek_v4_query_block: Optional[int] = None,
    deepseek_v4_key_block: Optional[int] = None,
    chat_template_mode: str = "auto",
    print_full_output: bool = False,
    boundary_issue_mode: str = "reproducible",
    captured_logits_json: Optional[str] = None,
    captured_request_json: Optional[str] = None,
    boundary_logits_cos_threshold: float = 0.99,
    boundary_logits_kl_threshold: float = 0.05,
    boundary_logits_margin_threshold: float = 0.05,
) -> BoundaryResult:
    """定界主入口 — 区分 WEIGHT / INFERENCE_FRAMEWORK / BOTH / INCONCLUSIVE / INVALID_RUN。

    必给的 Bad Case 来源 (三选一, 优先级 prompt_file > messages > prompt):
      prompt       : str, 单轮 plain text (由 chat_template_mode 决定是否包装)
      messages     : List[Dict], chat-template 对话
      prompt_file  : vLLM 请求格式或对话列表 JSON

    框架侧 Bad Case 来源 (二选一, 推荐 framework_bad_output):
      framework_bad_output      : 部署框架实际生成的坏文本 (对其跑 detect_badcase)
      framework_bad_reproduced   : 直接给 bool (True=已确认框架复现)

    ref (BF16) 可选: 给了 run_ref=True 会跑一次原生参考, 用于区分"quant 回归"
    vs"base-model 本征"(→ BOTH); 不给则只比 framework vs transformers(quant)。

    ``boundary_issue_mode="intermittent"`` 不启动部署框架，也不重新生成文本；
    它读取 ``captured_logits_json`` 中的 positions 和现场 logits；input_ids
    可来自响应的 prompt_token_ids，或 ``captured_request_json`` 中已经是
    token-ID 数组的 prompt。Transformers 在完全相同 token 序列上 replay。

    生效条件:
      - 量化模型 → hf_inference_check 内 NPU 加速反量化路径 (CPU fallback 未回迁)
      - 非量化模型 → 直接 safetensors 加载
    """
    from .input_resolver import NEVER_MESSAGES_ERROR, normalize_chat_template_mode
    try:
        chat_template_mode = normalize_chat_template_mode(chat_template_mode)
    except ValueError as exc:
        return _invalid(str(exc), framework_name, framework_gen_config,
                         framework_bad_reproduced)

    boundary_issue_mode = str(boundary_issue_mode or "reproducible").strip().lower()
    if boundary_issue_mode not in {"reproducible", "intermittent"}:
        return _invalid(
            "boundary_issue_mode must be reproducible or intermittent",
            framework_name, framework_gen_config, framework_bad_reproduced,
        )

    captured = None
    if boundary_issue_mode == "intermittent":
        if not captured_logits_json:
            return _invalid(
                "intermittent mode requires --captured_logits_json",
                framework_name, framework_gen_config, framework_bad_reproduced,
            )
        try:
            from .captured_logits import load_captured_logits
            captured = load_captured_logits(
                captured_logits_json, request_path=captured_request_json,
            )
        except Exception as exc:
            return _invalid(
                f"captured logits JSON 加载失败: {exc}", framework_name,
                framework_gen_config, framework_bad_reproduced,
            )

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
    # A structured request is the source of truth for template-specific
    # thinking settings.  This keeps DeepSeek-style ``thinking: false`` from
    # being rendered again with the CLI's default ``--thinking chat``.
    thinking = _request_thinking_mode(request_payload, thinking)
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
            "frequency_penalty", "presence_penalty", "seed", "do_sample", "tools",
        )
        if key in request_payload
    }
    if framework_gen_config is None:
        framework_gen_config = request_gen_config or None
    elif request_gen_config:
        framework_gen_config = {**request_gen_config, **framework_gen_config}

    # ---- resolve single conversation (reproducible generation only) ----
    if captured is None:
        messages, invalid_result = _boundary_resolve_conversation(
            messages, prompt, prompt_file, framework_name,
            framework_gen_config, framework_bad_reproduced)
        if invalid_result is not None:
            return invalid_result
        if (chat_template_mode == "never"
                and not (len(messages) == 1 and messages[0].get("_raw_prompt") is True)):
            return _invalid(NEVER_MESSAGES_ERROR, framework_name,
                            framework_gen_config, framework_bad_reproduced)
    else:
        # Captured input_ids are authoritative.  Do not resolve prompt,
        # messages or chat template, even when metadata contains a prompt.
        messages = []

    limitations: List[str] = []
    evidence: Dict[str, Any] = {
        "quant_model_path": quant_model_path,
        "ref_model_path": ref_model_path,
        "devices": devices,
        "dtype": dtype,
        "max_new_tokens": max_new_tokens,
        "thinking": thinking,
        "request_chat_template_kwargs": request_payload.get("chat_template_kwargs"),
        "num_runs": num_runs,
        "concurrency": concurrency,
        "stop_on_first_badcase": stop_on_first_badcase,
        "expert_chunk_size": expert_chunk_size or 8,
        "prefill_parallel": prefill_parallel,
        "deepseek_v4_query_block": deepseek_v4_query_block or int(
            os.getenv("ACC_DEEPSEEK_V4_QUERY_BLOCK", "64")
        ),
        "deepseek_v4_key_block": deepseek_v4_key_block or int(
            os.getenv("ACC_DEEPSEEK_V4_KEY_BLOCK", "1024")
        ),
        "chat_template_mode": chat_template_mode,
        "boundary_issue_mode": boundary_issue_mode,
        "captured_logits_json": captured_logits_json,
        "captured_request_json": captured_request_json,
        "resident_experts": os.getenv("ACC_BOUNDARY_RESIDENT_EXPERTS", "1") != "0",
        "expert_cache_per_layer": int(
            os.getenv("ACC_BOUNDARY_EXPERT_CACHE_PER_LAYER", "16")),
        "generation_config": {
            "do_sample": _request_uses_sampling(framework_gen_config),
            **(framework_gen_config or {}),
            "use_cache": True,
        },
        "framework_gen_config": framework_gen_config,
        "request_model": request_payload.get("model"),
        "messages_preview": (messages[-1].get("content", "")[:120]
                             if messages else ""),
    }

    if captured is not None:
        evidence["captured_logits"] = {
            "source": captured.source_path,
            "input_token_count": int(captured.input_ids.shape[1]),
            "input_ids_shape": list(captured.input_ids.shape),
            "has_attention_mask": captured.attention_mask is not None,
            "positions": list(captured.token_positions),
            "has_full_logits": captured.has_full_logits,
            "metadata": captured.metadata,
        }
        evidence["replay_input_ids_source"] = (
            captured.metadata.get("input_ids_source")
            or "captured JSON (no tokenize/chat_template)"
        )

        fw_reproduced = _boundary_detect_framework(
            framework_bad_reproduced, framework_bad_output, bad_pattern,
            evidence, limitations)
        try:
            replay = _run_transformers_replay_on_path(
                quant_model_path, devices, dtype, captured, verbose,
                prefill_parallel=prefill_parallel,
                glm_attn_query_block=glm_attn_query_block,
                glm_attn_selected_block=glm_attn_selected_block,
                deepseek_v4_query_block=deepseek_v4_query_block,
                deepseek_v4_key_block=deepseek_v4_key_block,
            )
            comparison, logits_data, summary = _compare_captured_replay(
                captured, replay, quant_model_path,
                framework_gen_config=framework_gen_config,
                cos_threshold=boundary_logits_cos_threshold,
                kl_threshold=boundary_logits_kl_threshold,
                margin_threshold=boundary_logits_margin_threshold,
            )
            evidence["captured_logits_replay"] = summary
            evidence["captured_logits_replay"]["logits_data"] = asdict(logits_data)
            result_kind = summary["verdict"]
            return BoundaryResult(
                framework_name=framework_name,
                framework_badcase_reproduced=fw_reproduced,
                transformers_badcase_reproduced=None,
                ref_badcase_reproduced=None,
                boundary_result=result_kind,
                evidence=evidence,
                limitations=limitations,
            )
        except Exception as exc:
            return _invalid(
                f"captured logits Transformers replay 失败: {exc}", framework_name,
                framework_gen_config, framework_bad_reproduced,
                fw_reproduced=fw_reproduced, evidence=evidence,
                limitations=limitations,
            )

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
        prefill_parallel, glm_attn_query_block, glm_attn_selected_block,
        deepseek_v4_query_block, deepseek_v4_key_block,
        chat_template_mode, print_full_output)
    if invalid_result is not None:
        return invalid_result

    # ---- transformers(ref) 路径 (可选) ----
    ref_reproduced = _boundary_run_ref(
        run_ref, ref_model_path, devices, dtype, messages, thinking,
        max_new_tokens, framework_gen_config, verbose, bad_pattern, evidence, limitations,
        num_runs, concurrency, stop_on_first_badcase, expert_chunk_size,
        prefill_parallel, glm_attn_query_block, glm_attn_selected_block,
        deepseek_v4_query_block, deepseek_v4_key_block,
        chat_template_mode, print_full_output)

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
    # Keep plain --prompt as text in auto/never mode.  The marker lets the
    # shared renderer distinguish it from structured messages; ``always``
    # will still wrap the text as a user message at render time.
    return [{"role": "user", "content": prompt, "_raw_prompt": True}], None


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
    unsupported = []
    if framework_gen_config:
        for key in ("frequency_penalty", "presence_penalty"):
            value = framework_gen_config.get(key)
            if value not in (None, 0, 0.0):
                unsupported.append(f"{key}={value}")
        if unsupported:
            limitations.append(
                "原生 Transformers 无法严格对齐框架 gen_config: "
                + ", ".join(unsupported)
            )
    evidence["framework_gen_diff"] = unsupported

    # Sampling parameters are mirrored, but different runtimes do not promise
    # identical RNG consumption or kernels even with the same seed.
    fw_sampling = _request_uses_sampling(framework_gen_config)
    if fw_sampling:
        limitations.append(
            "已对齐 sampling 参数；vLLM/Transformers 的 RNG 消费顺序与采样 kernel "
            "可能不同，不保证逐 token 位级一致"
        )
    seed = (framework_gen_config or {}).get("seed")
    evidence["seed_consistency"] = (
        "greedy 确定性" if not fw_sampling else
        (f"sampling seed={seed}" if seed is not None else "sampling 未指定 seed")
    )


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


def _run_transformers_replay_on_path(
    model_path: str, devices: str, dtype: str, captured, verbose: bool,
    *, prefill_parallel: str = "pp", glm_attn_query_block: Optional[int] = None,
    glm_attn_selected_block: Optional[int] = None,
    deepseek_v4_query_block: Optional[int] = None,
    deepseek_v4_key_block: Optional[int] = None,
):
    """Load the quant checkpoint and replay captured ids without generation."""
    from .inference_check import hf_inference_check
    return hf_inference_check(
        model_path=model_path,
        devices=devices,
        dtype=dtype,
        max_new_tokens=1,
        skip_ppl=True,
        thinking="none",
        verbose=verbose,
        num_runs=1,
        concurrency=1,
        prefill_parallel=prefill_parallel,
        glm_attn_query_block=glm_attn_query_block,
        glm_attn_selected_block=glm_attn_selected_block,
        deepseek_v4_query_block=deepseek_v4_query_block,
        deepseek_v4_key_block=deepseek_v4_key_block,
        replay_input_ids=captured.input_ids,
        replay_positions=captured.token_positions,
        replay_attention_mask=captured.attention_mask,
    )


def _captured_display_tokenizer(model_path: str):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
    except Exception:
        class _Ids:
            @staticmethod
            def decode(ids, **kwargs):
                return f"#{ids[0]}"
        return _Ids()


def _captured_top_k(captured, framework_gen_config: Optional[Dict]) -> int:
    cfg = framework_gen_config or {}
    value = cfg.get("top_k")
    if value is None:
        value = (captured.metadata.get("generation_config") or {}).get("top_k")
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = 10
    return max(1, value)


def _compare_captured_replay(
    captured, replay, model_path: str, *, framework_gen_config: Optional[Dict],
    cos_threshold: float, kl_threshold: float, margin_threshold: float,
):
    from .logits_compare import LogitsCollection, compare_logits, compare_captured_topk
    tokenizer = _captured_display_tokenizer(model_path)
    top_k = _captured_top_k(captured, framework_gen_config)
    if not captured.has_full_logits and captured.topk:
        observed = [len(row) for row in captured.topk if row]
        if observed:
            # Compare like-for-like when the capture contains fewer
            # candidates than the request's generation top_k metadata.
            top_k = min(top_k, min(observed))
    if captured.has_full_logits:
        cap_collection = LogitsCollection(
            token_positions=list(captured.token_positions), logits=captured.logits,
            input_ids=captured.input_ids, position_mode="captured_vllm",
        )
        comparison = compare_logits(
            cap_collection, replay, tokenizer, top_k=top_k,
        )
    else:
        comparison = compare_captured_topk(
            captured.topk or [], replay, tokenizer, top_k=top_k,
        )
    logits_data = comparison.to_logits_data()
    logits_data.input_ids = captured.input_ids.tolist()

    cos_values = [v for v in comparison.token_wise_cos
                  if v is not None and math.isfinite(float(v))]
    kl_values = [v for v in comparison.token_wise_kl
                 if v is not None and math.isfinite(float(v))]
    overlaps = comparison.token_wise_topk_overlap
    top1 = comparison.token_wise_top1_match
    mismatches = []
    top1_flips = []
    low_margin_flips = []
    nonfinite_positions = []
    if captured.has_full_logits:
        import torch
        for index, position in enumerate(comparison.token_positions):
            cap_ok = bool(torch.isfinite(captured.logits[index]).all().item())
            replay_ok = bool(torch.isfinite(replay.logits[index]).all().item())
            if not (cap_ok and replay_ok):
                nonfinite_positions.append(position)
    for index, position in enumerate(comparison.token_positions):
        cos_bad = comparison.token_wise_cos[index] is not None and comparison.token_wise_cos[index] < cos_threshold
        kl_bad = comparison.token_wise_kl[index] is not None and comparison.token_wise_kl[index] > kl_threshold
        topk_bad = overlaps[index] is not None and overlaps[index] < 1.0
        if cos_bad or kl_bad or topk_bad:
            mismatches.append(position)
        if not top1[index]:
            top1_flips.append(position)
            rm = comparison.ref_top1_margin[index] if index < len(comparison.ref_top1_margin) else None
            qm = comparison.quant_top1_margin[index] if index < len(comparison.quant_top1_margin) else None
            if rm is not None and qm is not None and max(abs(rm), abs(qm)) <= margin_threshold:
                low_margin_flips.append(position)
    for position in nonfinite_positions:
        if position not in mismatches:
            mismatches.append(position)

    if not captured.has_full_logits:
        missing = "full-vocabulary logits absent: cosine/KL/max-abs/histogram unavailable"
    else:
        missing = None
    significant_flips = [p for p in top1_flips if p not in low_margin_flips]
    if mismatches and significant_flips:
        verdict = INTERMITTENT_LOGITS_MISMATCH
    elif top1_flips and not significant_flips:
        verdict = INTERMITTENT_RANKING_SENSITIVE
    elif mismatches:
        verdict = INTERMITTENT_LOGITS_MISMATCH
    else:
        verdict = INTERMITTENT_LOGITS_ALIGNED
    max_abs_diff = None
    if captured.has_full_logits:
        max_abs_diff = float((captured.logits - replay.logits[:len(captured.token_positions)]).abs().max().item())
    summary = {
        "input_token_count": int(captured.input_ids.shape[1]),
        "compared_positions": list(comparison.token_positions),
        "compared_position_count": len(comparison.token_positions),
        "top_k": top_k,
        "has_full_logits": captured.has_full_logits,
        "top1_match_count": sum(bool(x) for x in top1),
        "top1_total": len(top1),
        "top1_flip_positions": top1_flips,
        "top1_flip_count": len(top1_flips),
        "low_margin_flip_positions": low_margin_flips,
        "low_margin_flip_count": len(low_margin_flips),
        "first_mismatch_position": mismatches[0] if mismatches else None,
        "first_top1_flip_position": top1_flips[0] if top1_flips else None,
        "nonfinite_positions": nonfinite_positions,
        "nonfinite_count": len(nonfinite_positions),
        "mean_cosine": (sum(cos_values) / len(cos_values)) if cos_values else None,
        "max_kl": max(kl_values) if kl_values else None,
        "max_abs_diff": max_abs_diff,
        "min_topk_overlap": min(overlaps) if overlaps else None,
        "cos_threshold": cos_threshold,
        "kl_threshold": kl_threshold,
        "margin_threshold": margin_threshold,
        "verdict": verdict,
        "missing_metrics": missing,
    }
    return comparison, logits_data, summary


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
                       prefill_parallel: str = "pp",
                       glm_attn_query_block: Optional[int] = None,
                       glm_attn_selected_block: Optional[int] = None,
                       deepseek_v4_query_block: Optional[int] = None,
                       deepseek_v4_key_block: Optional[int] = None,
                       chat_template_mode: str = "auto",
                       print_full_output: bool = False) -> Tuple[Optional[BoundaryResult], Optional[bool]]:
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
            expert_chunk_size=expert_chunk_size, prefill_parallel=prefill_parallel,
            glm_attn_query_block=glm_attn_query_block,
            glm_attn_selected_block=glm_attn_selected_block,
            deepseek_v4_query_block=deepseek_v4_query_block,
            deepseek_v4_key_block=deepseek_v4_key_block,
            chat_template_mode=chat_template_mode,
            print_full_output=print_full_output)
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
                     prefill_parallel: str = "pp",
                     glm_attn_query_block: Optional[int] = None,
                     glm_attn_selected_block: Optional[int] = None,
                     deepseek_v4_query_block: Optional[int] = None,
                     deepseek_v4_key_block: Optional[int] = None,
                     chat_template_mode: str = "auto",
                     print_full_output: bool = False) -> Optional[bool]:
    """transformers(ref) 路径运行 (可选); 失败时记录到 limitations 不中止"""
    if not (run_ref and ref_model_path):
        return None
    try:
        ref_outputs = _run_transformers_on_path(
            ref_model_path, devices, dtype, messages, thinking,
            max_new_tokens, framework_gen_config, verbose, label="ref",
            num_runs=num_runs, concurrency=concurrency,
            stop_on_first_badcase=stop_on_first_badcase, bad_pattern=bad_pattern,
            expert_chunk_size=expert_chunk_size, prefill_parallel=prefill_parallel,
            glm_attn_query_block=glm_attn_query_block,
            glm_attn_selected_block=glm_attn_selected_block,
            deepseek_v4_query_block=deepseek_v4_query_block,
            deepseek_v4_key_block=deepseek_v4_key_block,
            chat_template_mode=chat_template_mode,
            print_full_output=print_full_output)
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
    merged_limitations = list(limitations) if limitations else []
    if reason not in merged_limitations:
        merged_limitations.append(reason)
    return BoundaryResult(
        framework_name=framework_name,
        framework_badcase_reproduced=(fw_reproduced
                                     if fw_reproduced is not None
                                     else framework_bad_reproduced),
        transformers_badcase_reproduced=tq_reproduced,
        ref_badcase_reproduced=None,
        boundary_result=INVALID_RUN,
        evidence=e,
        limitations=merged_limitations,
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
        glm_attn_query_block=getattr(args, "glm_attn_query_block", None),
        glm_attn_selected_block=getattr(args, "glm_attn_selected_block", None),
        deepseek_v4_query_block=getattr(args, "deepseek_v4_query_block", None),
        deepseek_v4_key_block=getattr(args, "deepseek_v4_key_block", None),
        chat_template_mode=getattr(args, "chat_template_mode", "auto"),
        boundary_issue_mode=getattr(args, "boundary_issue_mode", "reproducible"),
        captured_logits_json=getattr(args, "captured_logits_json", None),
        captured_request_json=getattr(args, "captured_request_json", None),
        boundary_logits_cos_threshold=getattr(args, "boundary_logits_cos_threshold", 0.99),
        boundary_logits_kl_threshold=getattr(args, "boundary_logits_kl_threshold", 0.05),
        boundary_logits_margin_threshold=getattr(args, "boundary_logits_margin_threshold", 0.05),
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
        replay_summary = out.get("evidence", {}).get("captured_logits_replay", {})
        if replay_summary:
            logger.info(
                "  captured replay: positions=%d, top1=%s/%s, cosine=%s, KL=%s, overlap=%s"
                % (
                    len(replay_summary.get("compared_positions", [])),
                    replay_summary.get("top1_match_count", 0),
                    replay_summary.get("top1_total", 0),
                    replay_summary.get("mean_cosine", "n/a"),
                    replay_summary.get("max_kl", "n/a"),
                    replay_summary.get("min_topk_overlap", "n/a"),
                )
            )
        if out.get("limitations"):
            logger.info("  limitations:")
            for lim in out["limitations"]:
                logger.info(f"    - {lim}")
        logger.info("=" * 70)
        print_boundary_verdict_explain(out["boundary_result"])


def print_boundary_verdict_explain(kind: str) -> None:
    explains = {
        INTERMITTENT_LOGITS_ALIGNED:
            "vLLM captured logits 与 Transformers replay 基本一致；当前 sampler 前 logits boundary 未发现明显偏差，"
            "继续关注 sampling/logits processor/request state 或低 margin 分叉",
        INTERMITTENT_LOGITS_MISMATCH:
            "同一 input_ids、同一 position 的 vLLM/Transformers sampler 前 logits 明显不一致；"
            "问题偏向框架执行路径、量化算子、KV cache 或通信并行",
        INTERMITTENT_RANKING_SENSITIVE:
            "发现 Top-1 flip，但 Ref/Quant margin 均较小；属于 ranking-sensitive/low-margin flip，"
            "不能直接判为严重框架异常",
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
    "INTERMITTENT_LOGITS_ALIGNED", "INTERMITTENT_LOGITS_MISMATCH",
    "INTERMITTENT_RANKING_SENSITIVE",
    "repeat_4gram_ratio", "nonprintable_ratio",
    "detect_badcase", "normalize_request_payload", "parse_request_json",
    "classify_boundary", "run_boundary", "boundary_result_to_dict",
    "run_boundary_cli", "print_boundary_verdict_explain",
]
