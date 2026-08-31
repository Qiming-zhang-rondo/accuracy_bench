"""Captured vLLM logits JSON compatibility tests (CPU only)."""

import json

import pytest

torch = pytest.importorskip("torch")

from accuracy_checker.captured_logits import load_captured_logits


def test_load_full_vocab_capture_selects_positions(tmp_path):
    path = tmp_path / "full.json"
    payload = {
        "model": "demo",
        "input_ids": [[10, 11, 12]],
        "positions": [0, 2],
        "logits": [[1, 2], [3, 4], [5, 6]],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    capture = load_captured_logits(str(path))
    assert capture.token_positions == [0, 2]
    assert capture.has_full_logits
    assert capture.logits.tolist() == [[1.0, 2.0], [5.0, 6.0]]
    assert capture.input_ids.tolist() == [[10, 11, 12]]


def test_load_topk_capture_and_logprob_display_value(tmp_path):
    path = tmp_path / "topk.json"
    payload = {
        "input_ids": [[10, 11]],
        "token_positions": [1],
        "top_k": [[
            {"token_id": 42, "logprob": -0.5, "token_str": "x"},
            {"token_id": 43, "probability": 0.1},
        ]],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    capture = load_captured_logits(str(path))
    assert not capture.has_full_logits
    assert capture.topk[0][0].token_id == 42
    assert capture.topk[0][0].probability == pytest.approx(0.60653066)

