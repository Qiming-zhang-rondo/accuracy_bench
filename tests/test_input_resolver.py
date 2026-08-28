"""Tests for unified prompt/chat-template input resolution."""

import json

import pytest
import torch

from accuracy_checker.input_resolver import resolve_model_input


class _Tokenizer:
    def __init__(self):
        self.template_calls = []
        self.tokenize_calls = []

    def apply_chat_template(self, conversation, tokenize=False,
                            return_tensors=None, **kwargs):
        self.template_calls.append((conversation, tokenize, kwargs))
        text = "<template>" + conversation[0]["content"]
        if tokenize:
            return torch.tensor([[11, 12, 13]])
        return text

    def __call__(self, text, return_tensors=None):
        self.tokenize_calls.append(text)
        return {"input_ids": torch.tensor([[1, 2]])}


def test_auto_prompt_stays_raw():
    tok = _Tokenizer()
    resolved = resolve_model_input(tok, prompt="hello", chat_template_mode="auto")
    assert resolved["source_kind"] == "text"
    assert resolved["rendered_text"] == "hello"
    assert resolved["input_ids"].tolist() == [[1, 2]]
    assert not tok.template_calls


def test_always_prompt_is_wrapped_and_rendered():
    tok = _Tokenizer()
    resolved = resolve_model_input(tok, prompt="hello", chat_template_mode="always")
    assert resolved["rendered_text"] == "<template>hello"
    assert tok.template_calls[0][0] == [{"role": "user", "content": "hello"}]
    assert resolved["input_ids"].tolist() == [[11, 12, 13]]


def test_never_prompt_stays_raw():
    tok = _Tokenizer()
    resolved = resolve_model_input(tok, prompt="hello", chat_template_mode="never")
    assert resolved["rendered_text"] == "hello"
    assert not tok.template_calls


@pytest.mark.parametrize("mode", ["auto", "always"])
def test_messages_use_chat_template(mode):
    tok = _Tokenizer()
    messages = [{"role": "user", "content": "hello"}]
    resolved = resolve_model_input(tok, messages=messages, chat_template_mode=mode)
    assert resolved["source_kind"] == "messages"
    assert tok.template_calls


def test_never_messages_fails_fast():
    tok = _Tokenizer()
    with pytest.raises(ValueError, match="structured messages require chat template"):
        resolve_model_input(
            tok, messages=[{"role": "user", "content": "hello"}],
            chat_template_mode="never",
        )


def test_text_file_modes(tmp_path):
    path = tmp_path / "prompt.txt"
    path.write_text("already rendered", encoding="utf-8")
    text = path.read_text(encoding="utf-8")
    tok = _Tokenizer()
    auto = resolve_model_input(tok, prompt=text, source_kind="text", chat_template_mode="auto")
    assert auto["rendered_text"] == "already rendered"
    tok = _Tokenizer()
    always = resolve_model_input(tok, prompt=text, source_kind="text", chat_template_mode="always")
    assert always["rendered_text"] == "<template>already rendered"


def test_cache_identity_includes_non_default_template_mode():
    from run_accuracy_check import _cache_input_identity

    auto = type("Args", (), {"prompt": "hello", "messages": None,
                              "prompt_file": None, "chat_template_mode": "auto"})()
    always = type("Args", (), {"prompt": "hello", "messages": None,
                                "prompt_file": None, "chat_template_mode": "always"})()
    assert _cache_input_identity(auto) == "prompt:hello"
    assert _cache_input_identity(auto) != _cache_input_identity(always)


def test_request_json_messages_are_structured():
    tok = _Tokenizer()
    request = {"model": "x", "messages": [{"role": "user", "content": "hello"}]}
    resolved = resolve_model_input(
        tok, messages=request["messages"], source_kind="messages",
        chat_template_mode="auto",
    )
    assert resolved["source_kind"] == "messages"
    assert json.dumps(request["messages"], ensure_ascii=False)
