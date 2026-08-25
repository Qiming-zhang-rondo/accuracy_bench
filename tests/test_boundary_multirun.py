"""Boundary 多轮 batched Transformers 推理回归测试。"""

import json
from types import SimpleNamespace

import torch


def test_direct_request_json_validation():
    from accuracy_checker.boundary_check import parse_request_json

    request = parse_request_json(json.dumps({
        "model": "glm-52", "max_tokens": 102400,
        "messages": [{"role": "user", "content": "problem"}],
    }))
    assert request["max_tokens"] == 102400
    assert request["messages"][0]["role"] == "user"


def test_summarize_transformers_runs_keeps_every_output():
    from accuracy_checker.boundary_check import _summarize_transformers_runs

    outputs = [
        {"run_index": 1, "raw_text": "normal answer", "output_tokens": 12},
        {"run_index": 2, "raw_text": "abcd" * 40, "output_tokens": 40},
        {"run_index": 3, "raw_text": "another clean answer", "output_tokens": 12},
    ]
    reproduced, summary = _summarize_transformers_runs(outputs, 3, 2, None)

    assert reproduced is True
    assert summary["requested_runs"] == 3
    assert summary["completed_runs"] == 3
    assert summary["badcase_runs"] == 1
    assert summary["first_badcase_run"] == 2
    assert [run["output_full"] for run in summary["runs"]] == [
        output["raw_text"] for output in outputs
    ]


def test_truncated_thinking_still_detects_clear_repetition():
    from accuracy_checker.boundary_check import detect_badcase

    result = detect_badcase(
        "loop token " * 100,
        thinking_truncated=True,
        output_tokens=100,
    )
    assert result["reproduced"] is True


def test_prompt_file_max_tokens_and_quant_only(monkeypatch, tmp_path):
    from accuracy_checker import boundary_check as bc

    request = tmp_path / "request.json"
    request.write_text(json.dumps({
        "model": "glm-52",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "hello"}],
    }), encoding="utf-8")

    monkeypatch.setattr(bc, "_boundary_check_tokenizer", lambda *args: None)
    monkeypatch.setattr(bc, "_run_transformers_on_path", lambda *args, **kwargs: [
        {"run_index": 1, "raw_text": "clean response text", "output_tokens": 12}
    ])
    result = bc.run_boundary(
        "/quant", prompt_file=str(request), run_ref=False,
        framework_bad_reproduced=True,
    )

    assert result.boundary_result == bc.INFERENCE_FRAMEWORK
    assert result.ref_badcase_reproduced is None
    assert result.evidence["max_new_tokens"] == 4096
    assert result.evidence["request_model"] == "glm-52"


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, **kwargs):
        return "prompt"

    def __call__(self, text, return_tensors="pt"):
        del text, return_tensors
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def decode(self, tokens, skip_special_tokens=False):
        del skip_special_tokens
        values = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
        return "".join(str(value) for value in values)


class _FakeModel:
    def __init__(self):
        self.batch_sizes = []

    def generate(self, input_ids, **kwargs):
        del kwargs
        self.batch_sizes.append(input_ids.shape[0])
        suffix = torch.tensor([[7, 8]]).repeat(input_ids.shape[0], 1)
        return SimpleNamespace(sequences=torch.cat([input_ids, suffix], dim=1))


def test_hf_generation_batches_total_runs_not_runs_times_concurrency():
    from accuracy_checker.inference_check import _hf_run_generation

    model = _FakeModel()
    results = _hf_run_generation(
        model, _FakeTokenizer(), None, "none", 16, "cpu", False,
        num_runs=10, concurrency=3,
        conversations_override=[[{"role": "user", "content": "hello"}]],
    )

    assert model.batch_sizes == [3, 3, 3, 1]
    assert len(results) == 10
    assert [result["run_index"] for result in results] == list(range(1, 11))
