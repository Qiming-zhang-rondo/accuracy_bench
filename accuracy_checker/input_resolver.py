"""Unified prompt/message rendering for model inputs.

The resolver deliberately keeps already-rendered text untouched in ``auto``
mode.  Structured messages are the only input kind that is implicitly
template-rendered by the historical behavior.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


CHAT_TEMPLATE_MODES = ("auto", "always", "never")
NEVER_MESSAGES_ERROR = (
    "structured messages require chat template; use --prompt/--prompt_file "
    "with an already-rendered prompt when mode=never"
)


def normalize_chat_template_mode(mode: Optional[str]) -> str:
    value = str(mode or "auto").strip().lower()
    if value not in CHAT_TEMPLATE_MODES:
        raise ValueError(
            "chat_template_mode must be one of: auto, always, never"
        )
    return value


def _extract_input_ids(value):
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict):
        value = value["input_ids"]
    if value.dim() == 1:
        value = value.unsqueeze(0)
    return value


def _template_kwargs(messages: List[Dict[str, Any]], request_tools=None,
                     thinking: str = "chat") -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "conversation": messages,
        "add_generation_prompt": True,
        "enable_thinking": thinking == "chat",
    }
    if request_tools:
        kwargs["tools"] = request_tools
    return kwargs


def _is_deepseek_v4_tokenizer(tokenizer) -> bool:
    """识别 V4 tokenizer；V4 官方 checkpoint 默认没有 Jinja 模板。"""
    candidates = [
        getattr(tokenizer, "name_or_path", ""),
        getattr(tokenizer, "_name_or_path", ""),
        getattr(getattr(tokenizer, "init_kwargs", {}), "get", lambda *_: "")(
            "model_type", ""
        ),
    ]
    text = " ".join(str(value).lower() for value in candidates)
    return "deepseek_v4" in text or "deepseek-v4" in text or "deepseekv4" in text


def _render_deepseek_v4_messages(messages: List[Dict[str, Any]],
                                  request_tools=None,
                                  thinking: str = "chat") -> str:
    """Render the standard V4 encoder format without mutating the tokenizer.

    DeepSeek-V4 releases intentionally omit a Jinja template and use the
    encoder in ``encoding_dsv4.py``.  This covers the public chat-completion
    path (system/user/tool/assistant and the generation prompt), including
    the important ``</think>`` marker when thinking is disabled.
    """
    bos = "<｜begin▁of▁sentence｜>"
    eos = "<｜end▁of▁sentence｜>"
    user_tag = "<｜User｜>"
    assistant_tag = "<｜Assistant｜>"
    think_open = "<think>"
    think_close = "</think>"
    parts = [bos]
    in_user = False

    # The public API normally supplies tools separately.  Keep this compact
    # fallback deliberately conservative; model-specific tool formatting is
    # still delegated to a real chat_template whenever one is available.
    if request_tools and (not messages or messages[0].get("role") != "system"):
        parts.append("\n\n## Tools\n\n")
        parts.append(json.dumps(request_tools, ensure_ascii=False))

    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content") or ""
        if role == "system":
            parts.append(str(content))
            in_user = False
        elif role in ("user", "tool"):
            if in_user:
                parts.append("\n\n")
            else:
                parts.append(user_tag)
                in_user = True
            if role == "tool":
                parts.append("<tool_result>" + str(content) + "</tool_result>")
            else:
                parts.append(str(content))
        elif role == "assistant":
            in_user = False
            parts.append(assistant_tag)
            if thinking == "chat":
                parts.append(think_open + str(message.get("reasoning_content") or "")
                             + think_close)
            else:
                parts.append(think_close)
            parts.append(str(content))
            parts.append(eos)
        else:
            # Preserve unknown/developer turns as user content instead of
            # silently dropping information in the fallback path.
            if in_user:
                parts.append("\n\n")
            else:
                parts.append(user_tag)
                in_user = True
            parts.append(str(content))

    if messages and messages[-1].get("role") in ("user", "tool"):
        parts.append(assistant_tag)
        parts.append(think_open if thinking == "chat" else think_close)
    return "".join(parts)


def _tokenize_rendered(tokenizer, rendered: str):
    """Tokenize a fallback-rendered string without auto-added special tokens."""
    try:
        return _extract_input_ids(tokenizer(
            rendered, return_tensors="pt", add_special_tokens=False
        ))
    except TypeError:
        return _extract_input_ids(tokenizer(rendered, return_tensors="pt"))


def _apply_chat_template(tokenizer, messages: List[Dict[str, Any]],
                         request_tools=None, thinking: str = "chat"):
    """Render text and ids, tolerating older tokenizers without enable_thinking."""
    kwargs = _template_kwargs(messages, request_tools, thinking)
    try:
        rendered = tokenizer.apply_chat_template(
            tokenize=False, **kwargs
        )
        tokenized = tokenizer.apply_chat_template(
            tokenize=True, return_tensors="pt", **kwargs
        )
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        kwargs.pop("enable_thinking", None)
        rendered = tokenizer.apply_chat_template(
            tokenize=False, **kwargs
        )
        tokenized = tokenizer.apply_chat_template(
            tokenize=True, return_tensors="pt", **kwargs
        )
    except (ValueError, RuntimeError) as exc:
        missing_template = "chat_template" in str(exc).lower() and (
            "not set" in str(exc).lower() or "no template" in str(exc).lower()
        )
        if not (missing_template and _is_deepseek_v4_tokenizer(tokenizer)):
            raise
        rendered = _render_deepseek_v4_messages(
            messages, request_tools=request_tools, thinking=thinking
        )
        return rendered, _tokenize_rendered(tokenizer, rendered)
    try:
        input_ids = _extract_input_ids(tokenized)
    except (AttributeError, KeyError, TypeError):
        # A few lightweight/test tokenizers implement only the text-returning
        # form of apply_chat_template; preserve compatibility by tokenizing
        # the rendered string in that case.
        input_ids = _extract_input_ids(tokenizer(rendered, return_tensors="pt"))
    return rendered, input_ids


def resolve_model_input(
    tokenizer,
    *,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    source_kind: Optional[str] = None,
    request_tools=None,
    thinking: str = "chat",
    chat_template_mode: str = "auto",
):
    """Resolve one text/messages source into rendered text and token ids.

    ``messages`` is intentionally distinct from a text containing ChatML
    markers.  In ``never`` mode silently concatenating structured messages
    would change their semantics, so it fails fast with an actionable error.
    """
    mode = normalize_chat_template_mode(chat_template_mode)
    structured = messages is not None or source_kind == "messages"
    if structured:
        if messages is None:
            raise ValueError("messages source is empty")
        if mode == "never":
            raise ValueError(NEVER_MESSAGES_ERROR)
        rendered_text, input_ids = _apply_chat_template(
            tokenizer, list(messages), request_tools=request_tools,
            thinking=thinking,
        )
        return {
            "source_kind": "messages",
            "rendered_text": rendered_text,
            "input_ids": input_ids,
        }

    text = "" if prompt is None else str(prompt)
    if mode == "always":
        rendered_text, input_ids = _apply_chat_template(
            tokenizer, [{"role": "user", "content": text}],
            thinking=thinking,
        )
    else:
        rendered_text = text
        encoded = tokenizer(text, return_tensors="pt")
        input_ids = _extract_input_ids(encoded)
    return {
        "source_kind": "text",
        "rendered_text": rendered_text,
        "input_ids": input_ids,
    }
