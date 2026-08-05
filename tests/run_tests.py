#!/usr/bin/env python3
"""
acc_bench 轻量测试运行器 (无 pytest 依赖)

容器内未安装 pytest 时使用此运行器。它复用 ``tests/conftest.py`` 安装的
pytest stub (``import pytest`` / ``pytest.mark.parametrize`` / ``pytest.raises``),
并为测试函数注入 ``tmp_path`` / ``capsys`` / ``monkeypatch`` 等常用 fixture。

发现规则 (与 pytest 兼容):
  - 文件: tests/test_*.py
  - 函数: 模块级 def test_*(...)
  - 方法: class Test* 下的 def test_*(self, ...)
  - parametrize: stub 只跑第一组参数, 与 conftest.py 行为一致

用法:
    python3 tests/run_tests.py                 # 跑全部 + 打印汇总
    python3 tests/run_tests.py tests/test_a4_fake_quant.py   # 跑指定文件
    python3 tests/run_tests.py -v               # 详细 (每个用例名)

退出码: 0=全绿, 1=有失败/错误。
"""
import argparse
import importlib
import inspect
import io
import os
import shutil
import sys
import tempfile
import traceback
import types

# 把仓库根加入 sys.path, 让 ``import accuracy_checker`` 可用
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, "tests") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "tests"))

# 显式 import conftest 以触发 pytest stub 安装 (conftest.py 检测到真 pytest 缺席
# 时会把带 mark.parametrize/raises/fixture 的 stub 注入 sys.modules['pytest'])。
# 必须先 import conftest, 不要预先 sys.modules['pytest']=空模块 —— 否则 conftest
# 会跳过 stub 安装, 导致用 parametrize 的测试拿到空 mark。
try:
    importlib.import_module("conftest")  # tests/conftest.py
except Exception:  # noqa: BLE001
    pass

import pytest  # noqa: F401,E402  (此时已是 stub 或真 pytest)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

class _Capsys:
    """pytest capsys 的最小替身: 捕获 stdout/stderr。"""
    def __init__(self):
        self._out = io.StringIO()
        self._err = io.StringIO()
        self._old_out = None
        self._old_err = None

    def _snap(self):
        return self._out.getvalue(), self._err.getvalue()

    @property
    def captured(self):
        return self._snap()

    def readouterr(self):
        return self._snap()

    def __enter__(self):
        self._old_out = sys.stdout
        self._old_err = sys.stderr
        sys.stdout = self._out
        sys.stderr = self._err
        return self

    def __exit__(self, *exc):
        sys.stdout = self._old_out
        sys.stderr = self._old_err


class _MonkeyPatch:
    """pytest monkeypatch 的极小子集: setattr + undo。"""
    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value):
        if isinstance(target, str):
            mod_path, _, attr = target.rpartition(".")
            mod = importlib.import_module(mod_path) if mod_path else __main__
            target = mod
        old = getattr(target, name)
        self._undo.append((target, name, old))
        setattr(target, name, value)

    def setenv(self, name, value):
        old = os.environ.get(name)
        self._undo.append(("__env__", name, old))
        os.environ[name] = value

    def undo(self):
        for target, name, old in reversed(self._undo):
            if target == "__env__":
                if old is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old
            else:
                setattr(target, name, old)


_FIXTURES = {
    "tmp_path": lambda: _TmpPath(),
    "capsys": lambda: _Capsys(),
    "monkeypatch": lambda: _MonkeyPatch(),
    "tmp_path_factory": lambda: _TmpPath(),
}


class _TmpPath:
    """pathlib.Path 的临时目录替身 + 自动清理。"""
    def __init__(self):
        self._d = tempfile.mkdtemp(prefix="accbench_test_")
        self._p = __import__("pathlib").Path(self._d)

    # 用作传给测试的"Path", 支持 Path 接口的子集
    def __fspath__(self):
        return self._d

    def __truediv__(self, other):
        return self._p / other

    def __getattr__(self, item):
        # 透传到真实 Path 对象 (read_text / write_text / exists / mkdir / ...)
        return getattr(self._p, item)

    def __str__(self):
        return self._d

    def __repr__(self):
        return f"TmpPath({self._d!r})"

    def cleanup(self):
        shutil.rmtree(self._d, ignore_errors=True)


# ---------------------------------------------------------------------------
# 发现用例
# ---------------------------------------------------------------------------

def _is_test_func(obj, name):
    return name.startswith("test_") and callable(obj)


def _is_test_class(name, obj):
    return name.startswith("Test") and isinstance(obj, type)


def collect(module):
    """从模块收集 (id, callable, needs_self)。返回 [(fqname, func, is_method)]."""
    cases = []
    for name, obj in inspect.getmembers(module):
        if _is_test_func(obj, name):
            cases.append((f"{module.__name__}::{name}", obj, False))
        elif _is_test_class(name, obj):
            # 实例化 class (无参 ctor)
            try:
                inst = obj()
            except Exception:
                inst = obj  # 回退: 用类直接调, boundmethod 缺失
            for mname, mobj in inspect.getmembers(obj):
                if _is_test_func(mobj, mname) and inspect.isfunction(mobj):
                    bound = getattr(inst, mname)
                    cases.append((f"{module.__name__}::{name}::{mname}", bound, True))
    return cases


# ---------------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------------

def _resolve_args(func, is_method):
    sig = inspect.signature(func)
    kwargs = {}
    holders = []  # 需清理的 fixture (tmp_path 等)
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        if pname in _FIXTURES:
            fx = _FIXTURES[pname]()
            holders.append(fx)
            kwargs[pname] = fx
        else:
            # 未知参数: 给 None, 让测试自己处理或失败
            kwargs[pname] = None
    return kwargs, holders


def run_one(fqname, func, is_method, verbose):
    # context-manager 式 fixtures: capsys 需要 enter/exit
    sig = inspect.signature(func)
    needs_capsys = "capsys" in sig.parameters

    holders = []
    try:
        kwargs, holders = _resolve_args(func, is_method)

        # capsys 需要 enter 上下文 (捕获 stdout/stderr)
        cap = kwargs.get("capsys")
        if needs_capsys and isinstance(cap, _Capsys):
            with cap:
                func(**kwargs)
        else:
            func(**kwargs)
        return True, ""
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        return False, f"{type(e).__name__}: {e}\n{tb}"
    finally:
        for h in holders:
            if hasattr(h, "undo"):
                try:
                    h.undo()
                except Exception:
                    pass
            if hasattr(h, "cleanup"):
                try:
                    h.cleanup()
                except Exception:
                    pass


def main():
    ap = argparse.ArgumentParser(description="acc_bench 轻量测试运行器")
    ap.add_argument("paths", nargs="*", help="指定 test_*.py 文件; 空=跑全部")
    ap.add_argument("-v", "--verbose", action="store_true", help="打印每个用例名")
    ap.add_argument("-x", "--stopfirst", action="store_true", help="首个失败即停止")
    args = ap.parse_args()

    tests_dir = os.path.join(_ROOT, "tests")
    if args.paths:
        files = [os.path.abspath(p) for p in args.paths]
    else:
        files = sorted(
            os.path.join(tests_dir, f) for f in os.listdir(tests_dir)
            if f.startswith("test_") and f.endswith(".py")
        )

    total = passed = 0
    failed = []
    for fpath in files:
        base = os.path.splitext(os.path.basename(fpath))[0]  # e.g. test_html_report
        try:
            mod = importlib.import_module(base)
        except Exception:  # noqa: BLE001
            print(f"  ERROR importing {base}: {traceback.format_exc().splitlines()[-1]}")
            failed.append((f"{base} (import)", traceback.format_exc()))
            continue
        for fqname, func, is_method in collect(mod):
            total += 1
            if args.verbose:
                print(f"  RUN     {fqname}")
            ok, err = run_one(fqname, func, is_method, args.verbose)
            if ok:
                passed += 1
                if args.verbose:
                    print(f"  PASS    {fqname}")
            else:
                failed.append((fqname, err))
                print(f"  FAIL    {fqname}")
                if err and args.verbose:
                    # 只打印最后一行, 避免 verbose 模式刷屏
                    last = err.strip().splitlines()[-1]
                    print(f"          {last}")
                if args.stopfirst:
                    print("\n  (stopped at first failure)")
                    break
        if failed and args.stopfirst:
            break

    print("\n" + "=" * 60)
    print(f"  结果: {passed}/{total} 通过, {len(failed)} 失败")
    if failed and not args.verbose:
        print("  失败用例 (用 -v 看详情):")
        for fqname, err in failed:
            last = err.strip().splitlines()[-1] if err else "(unknown)"
            print(f"    - {fqname}: {last}")
    elif failed and args.verbose:
        print("  失败详情:")
        for fqname, err in failed:
            print(f"\n  ---- {fqname} ----\n{err}")
    print("=" * 60)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
