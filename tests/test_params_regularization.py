"""
工作项3 (入参规整) UT — 验证 CLI 参数规整落地。

纯 Python AST/字符串检查, 不依赖 torch/NPU:
  1. --quant_method 默认 dequantize (不是 fake_quant)
  2. --available_devices 已删除 (死参数)
  3. --model_type 新增 (auto/dense/moe/glm_mla/glm_moe_dsa)
  4. --boundary 新增 (定界占位)
  5. --activation_quant_type 新增 (5 choices)
  6. subgraph_locate.py 无 import argparse (死 import 删除)
  7. subgraph_locate.py docstring 无 python -m 示例 (改为库 API)
"""

import ast
import inspect
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_SCRIPT = REPO_ROOT / "run_accuracy_check.py"
SUBGRAPH_FILE = REPO_ROOT / "accuracy_checker" / "subgraph_locate.py"


def _parse_run_script():
    """解析 run_accuracy_check.py 的 AST, 返回 module node."""
    return ast.parse(RUN_SCRIPT.read_text(encoding="utf-8"))


def _get_add_argument_calls(node):
    """提取所有 parser.add_argument(...) 调用, 返回 [(name, kwargs_dict), ...]."""
    calls = []
    for node_iter in ast.walk(node):
        if not isinstance(node_iter, ast.Call):
            continue
        if not isinstance(node_iter.func, ast.Attribute):
            continue
        if node_iter.func.attr != "add_argument":
            continue
        # 第一个位置参数 = argument name
        if not node_iter.args:
            continue
        name_node = node_iter.args[0]
        if isinstance(name_node, ast.Constant):
            name = name_node.value
        else:
            continue
        # kwargs
        kwargs = {}
        for kw in node_iter.keywords:
            val = kw.value
            if isinstance(val, ast.Constant):
                kwargs[kw.arg] = val.value
            elif isinstance(val, ast.List):
                # choices=["a", "b"] → list literal
                items = []
                for elt in val.elts:
                    if isinstance(elt, ast.Constant):
                        items.append(elt.value)
                kwargs[kw.arg] = items
        calls.append((name, kwargs))
    return calls


# ---------------------------------------------------------------------------
# 1. --quant_method 默认 dequantize
# ---------------------------------------------------------------------------

def test_quant_method_default_dequantize():
    """--quant_method 默认改为 dequantize (与 README 示例一致)"""
    calls = _get_add_argument_calls(_parse_run_script())
    found = False
    for name, kwargs in calls:
        if name == "--quant_method":
            found = True
            assert kwargs.get("default") == "dequantize", \
                f"--quant_method 默认应为 dequantize, got {kwargs.get('default')}"
            assert "dequantize" in kwargs.get("choices", []), \
                f"choices 应含 dequantize"
    assert found, "--quant_method 参数未找到"


# ---------------------------------------------------------------------------
# 2. --available_devices 已删除
# ---------------------------------------------------------------------------

def test_available_devices_deleted():
    """--available_devices 是死参数, 应已从 CLI 删除"""
    calls = _get_add_argument_calls(_parse_run_script())
    names = [name for name, _ in calls]
    assert "--available_devices" not in names, \
        "--available_devices 应已删除 (死参数, 全代码无引用)"


# ---------------------------------------------------------------------------
# 3. --model_type 新增
# ---------------------------------------------------------------------------

def test_model_type_added():
    """--model_type 新增, choices 含 auto/dense/moe/glm_mla/glm_moe_dsa"""
    calls = _get_add_argument_calls(_parse_run_script())
    found = False
    for name, kwargs in calls:
        if name == "--model_type":
            found = True
            assert kwargs.get("default") == "auto", \
                f"--model_type 默认应为 auto, got {kwargs.get('default')}"
            choices = kwargs.get("choices", [])
            for expected in ("auto", "dense", "moe", "glm_mla", "glm_moe_dsa",
                             "qwen3", "qwen3_moe", "qwen3_5_moe", "qwen3_vl",
                             "qwen3_6", "qwen3_6_moe", "kimi_k3"):
                assert expected in choices, \
                    f"--model_type choices 应含 {expected}, got {choices}"
    assert found, "--model_type 参数未找到"


# ---------------------------------------------------------------------------
# 4. --boundary 新增
# ---------------------------------------------------------------------------

def test_boundary_added():
    """--boundary 新增 (定界占位, action=store_true)"""
    calls = _get_add_argument_calls(_parse_run_script())
    found = False
    for name, kwargs in calls:
        if name == "--boundary":
            found = True
            assert kwargs.get("action") == "store_true", \
                f"--boundary 应 action=store_true, got {kwargs.get('action')}"
    assert found, "--boundary 参数未找到"


# ---------------------------------------------------------------------------
# 5. --activation_quant_type 新增
# ---------------------------------------------------------------------------

def test_activation_quant_type_added():
    """--activation_quant_type 新增, 5 choices"""
    calls = _get_add_argument_calls(_parse_run_script())
    found = False
    for name, kwargs in calls:
        if name == "--activation_quant_type":
            found = True
            choices = kwargs.get("choices", [])
            for expected in ("W8A8_MXFP8", "W4A8_MXFP", "W4A4_MXFP4", "W4A4_DYNAMIC", "W4A4_LAOS"):
                assert expected in choices, \
                    f"--activation_quant_type choices 应含 {expected}, got {choices}"
    assert found, "--activation_quant_type 参数未找到"


# ---------------------------------------------------------------------------
# 6. subgraph_locate.py 无 import argparse
# ---------------------------------------------------------------------------

def test_subgraph_no_argparse_import():
    """subgraph_locate.py 的 import argparse 是死代码, 应已删除"""
    tree = ast.parse(SUBGRAPH_FILE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "argparse", \
                    "subgraph_locate.py 不应 import argparse (死代码已删)"


# ---------------------------------------------------------------------------
# 7. subgraph_locate.py docstring 无 python -m 示例
# ---------------------------------------------------------------------------

def test_subgraph_docstring_no_python_m():
    """subgraph_locate.py docstring 不应有 python -m 示例 (改为库 API)"""
    tree = ast.parse(SUBGRAPH_FILE.read_text(encoding="utf-8"))
    if not tree.body:
        return
    first = tree.body[0]
    if not isinstance(first, ast.Expr) or not isinstance(first.value, (ast.Constant, ast.Str)):
        return
    docstring = first.value.value if isinstance(first.value, ast.Constant) else first.value.s
    assert "python3 -m accuracy_checker.subgraph_locate" not in docstring, \
        "docstring 不应有 python -m 示例 (应改为库 API 调用)"


# ---------------------------------------------------------------------------
# 8. --boundary 运行时调用定界逻辑 (占位 '恢复中' 消息已移除)
# ---------------------------------------------------------------------------

def test_boundary_runtime_message():
    """--boundary 运行时应调用 run_boundary, 不再打印 '恢复中' 占位消息"""
    result = subprocess.run(
        [sys.executable, str(RUN_SCRIPT), "--boundary",
         "--quant_model", "/dummy", "--device", "cpu"],
        capture_output=True, text=True, timeout=60,
    )
    combined = result.stdout + result.stderr
    # 定界逻辑应被执行 (打印 Boundary/定界 关键字)
    assert "Boundary" in combined or "定界" in combined, \
        f"--boundary 应打印 Boundary/定界, got stdout={result.stdout[:200]}"
    # 占位 '恢复中' 消息必须已移除 (能力已恢复)
    assert "恢复中" not in combined, \
        f"--boundary 不应再打印 '恢复中' 占位, got stdout={result.stdout[:200]}"
