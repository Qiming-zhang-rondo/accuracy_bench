"""
工作项2 (定界恢复) UT — 验证 inference_check + adapters 加载链。

不依赖 NPU / 大模型，用纯函数 + AST + import 检查覆盖核心逻辑。
运行: cd <your_workspace>/tools/accuracy_bench && python3 -m pytest tests/test_boundary_recovery.py -v
"""

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
INFERENCE_CHECK_PATH = ROOT / "accuracy_checker" / "inference_check.py"


# ---------------------------------------------------------------------------
# 1. parse_devices — 纯函数，无依赖
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inp,expected", [
    ("0", ["npu:0"]),
    ("npu:0", ["npu:0"]),
    ("0,1,2", ["npu:0", "npu:1", "npu:2"]),
    ("npu:0,cuda:1", ["npu:0", "cuda:1"]),
    (" 0 , 1 ", ["npu:0", "npu:1"]),
    ("0,,1", ["npu:0", "npu:1"]),  # 空段跳过
    ("", []),
])
def test_parse_devices(inp, expected):
    from accuracy_checker.inference_check import parse_devices
    assert parse_devices(inp) == expected


# ---------------------------------------------------------------------------
# 2. preprocess_messages — tool_calls.arguments JSON string → dict
# ---------------------------------------------------------------------------

def test_preprocess_messages_decodes_tool_call_arguments():
    from accuracy_checker.inference_check import preprocess_messages
    msgs = [{
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"city": "北京", "temp": 25}),
            },
        }],
    }]
    out = preprocess_messages(msgs)
    args = out[0]["tool_calls"][0]["function"]["arguments"]
    assert args == {"city": "北京", "temp": 25}
    # 不修改原对象
    assert isinstance(msgs[0]["tool_calls"][0]["function"]["arguments"], str)


def test_preprocess_messages_passes_through_without_tool_calls():
    from accuracy_checker.inference_check import preprocess_messages
    msgs = [{"role": "user", "content": "你好"}]
    assert preprocess_messages(msgs) == msgs


# ---------------------------------------------------------------------------
# 3. import 链 — 验证 __init__ 导出齐全
# ---------------------------------------------------------------------------

def test_package_exports_importable():
    import accuracy_checker
    for name in [
        "hf_inference_check",
        "qwen35_inference_check",
        "distribute_model",
        "dequantize_model_on_devices",
        "parse_devices",
    ]:
        assert hasattr(accuracy_checker, name), f"accuracy_checker 缺少导出: {name}"


def test_inference_check_module_imports():
    from accuracy_checker import inference_check as ic
    assert callable(ic.hf_inference_check)
    assert callable(ic.qwen35_inference_check)
    assert callable(ic.distribute_model)
    assert callable(ic.dequantize_model_on_devices)


def test_adapters_package_imports():
    from accuracy_checker.adapters import get_model_adapter
    assert callable(get_model_adapter)


def test_glm5_inference_check_script_exists():
    """薄壳脚本应存在且可读"""
    script = ROOT / "scripts" / "glm5_inference_check.py"
    assert script.exists(), f"缺少 {script}"
    # 语法检查
    ast.parse(script.read_text())


# ---------------------------------------------------------------------------
# 4. CPU fallback 已禁用 — AST 检查 use_cpu_dequant 路径 raise NotImplementedError
# ---------------------------------------------------------------------------

def test_cpu_fallback_raises_not_implemented():
    """line 765 附近 use_cpu_dequant 分支必须 raise NotImplementedError
    (dequantize_model 未在 clean 分支恢复)"""
    src = INFERENCE_CHECK_PATH.read_text()
    tree = ast.parse(src)

    # 找到 hf_inference_check 函数
    hf_func = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "hf_inference_check":
            hf_func = node
            break
    assert hf_func is not None, "未找到 hf_inference_check 函数"

    # 函数体内应存在 raise NotImplementedError 且消息含 'dequantize_model'
    found = False
    for node in ast.walk(hf_func):
        if isinstance(node, ast.Raise) and node.exc is not None:
            exc = node.exc
            # raise NotImplementedError("...") 形式
            if (isinstance(exc, ast.Call)
                    and isinstance(exc.func, ast.Name)
                    and exc.func.id == "NotImplementedError"):
                for arg in exc.args:
                    if (isinstance(arg, ast.Constant)
                            and "dequantize_model" in str(arg.value)):
                        found = True
    assert found, "use_cpu_dequant 路径未 raise NotImplementedError(dequantize_model)"


def test_no_dangling_dequantize_model_import():
    """clean 分支不应再 import dequantize_model / dequantize_weight_compressed_tensors
    (model_loader.py 已无这两个函数)"""
    src = INFERENCE_CHECK_PATH.read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == ".model_loader":
            names = {a.name for a in node.names}
            assert "dequantize_model" not in names, \
                "仍 import 了已删除的 dequantize_model"
            assert "dequantize_weight_compressed_tensors" not in names, \
                "仍 import 了已删除的 dequantize_weight_compressed_tensors"


def test_logging_not_swallowed_by_docstring():
    """回归: 之前 import logging / logger=... 被吞进 docstring 导致 logger 未定义。
    验证 logging 是真正的 module-level import。"""
    src = INFERENCE_CHECK_PATH.read_text()
    tree = ast.parse(src)
    module_imports = [
        n.names[0].name for n in tree.body
        if isinstance(n, ast.Import)
    ]
    assert "logging" in module_imports, "logging 不在 module-level import (可能被 docstring 吞了)"

    # logger 应是 module-level 赋值
    module_assigns = [
        n for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "logger" for t in n.targets)
    ]
    assert module_assigns, "未找到 module-level `logger = ...` 赋值"


# ---------------------------------------------------------------------------
# 5. 入口签名 — 关键参数齐全
# ---------------------------------------------------------------------------

def test_hf_inference_check_signature():
    import inspect
    from accuracy_checker.inference_check import hf_inference_check
    sig = inspect.signature(hf_inference_check)
    params = set(sig.parameters)
    expected = {
        "model_path", "devices", "dtype", "max_new_tokens",
        "prompt_file", "skip_ppl", "thinking", "verbose",
        "use_cpu_dequant", "noquit",
    }
    assert expected.issubset(params), f"缺参数: {expected - params}"
