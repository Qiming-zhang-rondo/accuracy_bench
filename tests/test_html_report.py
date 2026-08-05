"""
工作项5 (HTML 输出 v1) UT — 验证 generate_html_report 生成结构正确。

纯 Python 模块，不依赖 torch/NPU，用 mock 数据验证:
  1. 文件生成 + 自包含 (无外网依赖)
  2. 定界 banner 重复检测
  3. L2 表格 root_suspect 红框 + impact_boundary 绿框
  4. 颜色 class 映射 (good/warn/bad/na)
  5. Recovery 公式块存在
  6. 各 None / 空数据边界
"""

import os
import tempfile
from pathlib import Path

import pytest

from accuracy_checker.html_report import (
    generate_html_report,
    _detect_repetition,
    _err_class,
    _rec_class,
    _flip_class,
    _fmt_pct,
    _fmt_err,
)


# ---------------------------------------------------------------------------
# 1. 纯函数: 颜色 class 阈值
# ---------------------------------------------------------------------------

def test_rec_class():
    assert _rec_class(0.90) == "good"      # >=0.80
    assert _rec_class(0.50) == "warn"      # 0.30-0.80
    assert _rec_class(0.10) == "bad"       # <0.30
    assert _rec_class(-0.1) == "bad"       # 负值=耦合
    assert _rec_class(None) == "na"


def test_err_class():
    assert _err_class(0.01, "selfroterr") == "good"   # <0.02
    assert _err_class(0.05, "selfroterr") == "warn"   # 0.02-0.10
    assert _err_class(0.20, "selfroterr") == "bad"    # >=0.10
    assert _err_class(None, "selfroterr") == "na"
    # baseline_l2 阈值不同
    assert _err_class(0.03, "baseline_l2") == "good"  # <0.05
    assert _err_class(0.10, "baseline_l2") == "warn"  # 0.05-0.20
    assert _err_class(0.30, "baseline_l2") == "bad"   # >=0.20


def test_flip_class():
    assert _flip_class(0.005) == "good"   # <0.01
    assert _flip_class(0.05) == "warn"    # 0.01-0.10
    assert _flip_class(0.20) == "bad"     # >=0.10
    assert _flip_class(None) == "na"


def test_fmt_pct():
    assert _fmt_pct(0.8) == "80.0%"
    assert _fmt_pct(None) == "—"


def test_fmt_err():
    assert _fmt_err(0.1234) == "0.1234"
    assert _fmt_err(None) == "—"


# ---------------------------------------------------------------------------
# 2. 重复检测 (定界判定核心)
# ---------------------------------------------------------------------------

def test_detect_repetition_clean():
    is_sus, _ = _detect_repetition("你好，我是 GLM 模型，很高兴认识你。")
    assert not is_sus


def test_detect_repetition_garbled():
    is_sus, reason = _detect_repetition("你好你好你好你好你好你好你好你好")  # 2-gram 重复
    assert is_sus
    assert "重复" in reason


def test_detect_repetition_short_text():
    is_sus, _ = _detect_repetition("你好")
    assert not is_sus  # 太短不检测


def test_detect_repetition_empty():
    is_sus, _ = _detect_repetition("")
    assert not is_sus


# ---------------------------------------------------------------------------
# 3. 完整 HTML 生成: 结构 + 关键元素
# ---------------------------------------------------------------------------

def _mock_boundary_clean():
    return [{
        "messages": [{"role": "user", "content": "你好"}],
        "generated": "你好，我是 GLM 模型，很高兴认识你。",
        "thinking": "",
        "input_tokens": 2,
        "output_tokens": 15,
        "time": 1.2,
        "thinking_truncated": False,
    }]


def _mock_boundary_garbled():
    return [{
        "messages": [{"role": "user", "content": "你好"}],
        "generated": "你好你好你好你好你好你好你好你好你好你好",
        "thinking": "",
        "input_tokens": 2,
        "output_tokens": 20,
        "time": 1.0,
        "thinking_truncated": False,
    }]


def _mock_l2_results():
    return [
        {
            "layer_idx": 4,
            "baseline_l2": 0.143,       # bad (>=0.20? no, 0.143 is warn for baseline_l2)
            "model_type": "glm_mla",
            "subgraphs": {
                "self_attn": 0.15,       # warn recovery
                "mlp": 0.85,             # good recovery
                "mlp.gate": None,
            },
            "subgraph_quant_types": {
                "self_attn": "INT8",
                "mlp": "INT8",
                "mlp.gate": "FLOAT",
            },
            "input_recovery": 0.72,      # warn
            "impact_boundary": "mlp",    # green border
            "root_suspect": "self_attn",  # red border
        },
        {
            "layer_idx": 10,
            "baseline_l2": 0.25,         # bad
            "model_type": "glm_mla",
            "subgraphs": {
                "self_attn": 0.90,
                "mlp": 0.05,             # bad recovery
            },
            "subgraph_quant_types": {
                "self_attn": "INT8",
                "mlp": "INT8",
            },
            "input_recovery": 0.10,      # bad
            "impact_boundary": "self_attn",
            "root_suspect": "mlp",
        },
    ]


def test_generate_html_report_full(tmp_path):
    out = tmp_path / "report.html"
    path = generate_html_report(
        boundary_results=_mock_boundary_clean(),
        l2_results=_mock_l2_results(),
        model_name="glm-5.1-mla",
        output_path=str(out),
    )
    assert path == str(out)
    content = out.read_text(encoding="utf-8")

    # 自包含: 无外网依赖
    assert "https://" not in content
    assert "http://" not in content
    assert "<style>" in content  # inline CSS

    # 标题 + 元信息
    assert "acc_bench 精度对齐报告" in content
    assert "glm-5.1-mla" in content

    # 定界 section
    assert "定界" in content
    assert "排除框架影响" in content
    # clean → banner-ok
    assert "banner-ok" in content
    assert "量化本身没问题" in content

    # L2 section
    assert "subgraph 级别定位" in content
    assert "Recovery 公式" in content
    assert "(baseline_l2 − patched_l2) / baseline_l2" in content

    # root_suspect 红框 (Layer 4: self_attn)
    assert "root-suspect" in content
    assert "self_attn" in content
    # impact_boundary 绿框 (Layer 4: mlp)
    assert "impact-boundary" in content

    # 颜色 class 出现
    for cls in ("good", "warn", "bad", "na"):
        assert cls in content

    # FLOAT 标记
    assert "FLOAT†" in content

    # 图例
    assert "图例" in content
    assert "红框=主要嫌疑" in content
    assert "绿框=关键边界" in content


def test_generate_html_report_garbled_boundary(tmp_path):
    """乱码生成 → banner-bad"""
    out = tmp_path / "bad.html"
    path = generate_html_report(
        boundary_results=_mock_boundary_garbled(),
        l2_results=None,
        model_name="test",
        output_path=str(out),
    )
    content = out.read_text(encoding="utf-8")
    assert "banner-bad" in content
    assert "量化本身有问题" in content
    # 没有 L2 section
    assert "subgraph 级别定位" not in content


def test_generate_html_report_only_l2(tmp_path):
    """只做 L2，无定界"""
    out = tmp_path / "l2only.html"
    generate_html_report(
        boundary_results=None,
        l2_results=_mock_l2_results(),
        model_name="test",
        output_path=str(out),
    )
    content = out.read_text(encoding="utf-8")
    assert "subgraph 级别定位" in content
    assert "定界" not in content


def test_generate_html_report_empty(tmp_path):
    """无任何数据 → 警告 banner"""
    out = tmp_path / "empty.html"
    generate_html_report(
        boundary_results=None,
        l2_results=None,
        model_name="test",
        output_path=str(out),
    )
    content = out.read_text(encoding="utf-8")
    assert "未提供任何诊断数据" in content
    assert "banner-warn" in content


def test_generate_html_report_auto_path(tmp_path):
    """output_path=None → 自动生成路径到 reports/"""
    # 临时切换工作目录到 tmp_path, 让 reports/ 生成在临时目录
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # generate_html_report 用 __file__ 推算 repo_root, 不会写到 tmp_path
        # 但我们只验证它返回一个路径且文件存在
        path = generate_html_report(
            boundary_results=_mock_boundary_clean(),
            l2_results=None,
            model_name="auto-test",
            output_path=None,
        )
        assert os.path.exists(path)
        assert "auto-test" in path
        assert path.endswith(".html")
    finally:
        os.chdir(cwd)


def test_generate_html_report_escapes_xss(tmp_path):
    """生成文本含 <script> → 必须被转义"""
    bad_results = [{
        "messages": [{"role": "user", "content": "<script>alert(1)</script>"}],
        "generated": "<img src=x onerror=alert(1)>",
        "thinking": "",
        "input_tokens": 1,
        "output_tokens": 1,
        "time": 0.1,
        "thinking_truncated": False,
    }]
    out = tmp_path / "xss.html"
    generate_html_report(
        boundary_results=bad_results,
        l2_results=None,
        model_name="xss-test",
        output_path=str(out),
    )
    content = out.read_text(encoding="utf-8")
    # 原始标签不应出现 (应被转义)
    assert "<script>alert(1)</script>" not in content
    assert "<img src=x onerror" not in content
    # 转义后的应出现
    assert "&lt;script&gt;" in content
    assert "&lt;img" in content
