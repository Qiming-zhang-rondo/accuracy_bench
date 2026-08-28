"""Unified prompt/message rendering for model inputs.

The resolver deliberately keeps already-rendered text untouched in ``auto``
mode.  Structured messages are the only input kind that is implicitly
template-rendered by the historical behavior.
"""

from __future__ import annotations

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
