"""UT — collect_logits_streaming.compare_inference 复用 scripts/badcase_eval.py.

回归点:
  1. 入口签名兼容老调用 (ref_text, quant_text, ref_tokens, quant_tokens),
     多余字段 ref_lc/quant_lc/topk 用 None 时不破坏基本比较.
  2. 用 badcase_eval.detect_repeat 正确识别退化输出 ("1,1,1,11111111") → quant_repeat=True.
  3. 用 badcase_eval.detect_garbled 正确识别非打印字符占多数的输出.
  4. 给出 ref_lc/quant_lc 时, logits 指标 (cos_sim / kl / topk_overlap) 被填入.
"""
from __future__ import annotations

import os
import importlib.util
import sys

import torch

_REPO = os.path.join(os.path.dirname(__file__), "..")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_SCRIPTS = os.path.join(_REPO, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


def _load_cls():
    """Load collect_logits_streaming.py as a module (avoid main())."""
    spec = importlib.util.spec_from_file_location(
        "collect_logits_streaming",
        os.path.join(_REPO, "collect_logits_streaming.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_basic_token_matching_unchanged():
    m = _load_cls()
    infc = m.compare_inference(
        "你好", "你好吗",
        ["你好"], ["你好", "吗"])
    assert infc.exact_match is False
    assert infc.first_divergence_pos == 1  # 第 1 个 token 位置就分歧
    assert abs(infc.token_match_rate - 0.5) < 1e-6  # 1 / 2


def test_detect_repeat_on_degenerate_output():
    """退化输出 '1,1,1,11111111' 应被 detect_repeat 判为复读."""
    m = _load_cls()
    bad_output = "1,1,1,11111111"
    bad_tokens = ["1", ",", "1", ",", "1", ",", "1", "11111111"]
    infc = m.compare_inference("正常输出", bad_output,
                                ["正常输出"], bad_tokens)
    assert infc.quant_repeat, "退化串应判为复读 (quant_repeat=True)"
    assert infc.ref_repeat is False


def test_detect_garbled_on_control_chars():
    """控制字符占多数应被 detect_garbled 判为乱码."""
    m = _load_cls()
    garbled = "\x01\x02\x03\x04\x05\x06\x07" * 5  # 35 个控制字符
    infc = m.compare_inference("正常", garbled,
                                ["正常"], list(garbled))
    assert infc.quant_garbled, "全控制字符应判为乱码"


def test_logits_metrics_populated_when_passed():
    """传入 ref_lc/quant_lc 时, cos_sim / kl / topk_overlap 应非空."""
    m = _load_cls()
    from accuracy_checker.logits_compare import LogitsCollection

    # 构造一对小 logits (位置数=3, 词表=8), ref/quant 高度相似 (cos≈1, kl≈0)
    ref_logits = torch.tensor([[3.0, 1.0, 0.5, 0.2, 0.1, 0.1, 0.05, 0.05],
                                [2.5, 1.2, 0.7, 0.3, 0.2, 0.1, 0.05, 0.05],
                                [3.1, 1.1, 0.6, 0.25, 0.15, 0.1, 0.05, 0.05]],
                               dtype=torch.float32)
    quant_logits = ref_logits.clone() + 0.01  # tiny perturbation
    positions = [0, 1, 2]
    ref_lc = LogitsCollection(token_positions=positions, logits=ref_logits)
    quant_lc = LogitsCollection(token_positions=positions, logits=quant_logits)
    infc = m.compare_inference("你好", "你好",
                                ["你好"], ["你好"],
                                ref_lc=ref_lc, quant_lc=quant_lc, topk=5)
    assert infc.logits_nan_inf is False
    assert infc.logits_cos_sim is not None and infc.logits_cos_sim > 0.99
    assert infc.logits_kl is not None and infc.logits_kl < 1e-3
    assert infc.topk_overlap is not None and infc.topk_overlap >= 0.8


def test_nan_inf_detected():
    """logits 中出现 NaN 应被 detect_nan_inf 捕获, 不进 cos/kl 计算."""
    m = _load_cls()
    from accuracy_checker.logits_compare import LogitsCollection
    bad = torch.tensor([[float("nan"), 1.0, 0.5], [0.3, 0.3, 0.4]], dtype=torch.float32)
    good = torch.tensor([[1.0, 0.5, 0.3], [0.4, 0.4, 0.2]], dtype=torch.float32)
    ref_lc = LogitsCollection(token_positions=[0, 1], logits=bad)
    quant_lc = LogitsCollection(token_positions=[0, 1], logits=good)
    infc = m.compare_inference("ok", "ok", ["ok"], ["ok"],
                                ref_lc=ref_lc, quant_lc=quant_lc)
    assert infc.logits_nan_inf is True
    # cos/kl/overlap 不算 (nan_inf=True 时短路)
    assert infc.logits_cos_sim is None
    assert infc.logits_kl is None
