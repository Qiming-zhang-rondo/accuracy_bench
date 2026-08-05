"""
Minimal pytest stub — allows running tests without pytest installed.

Usage: python3 -m pytest tests/ -q
  or:  python3 -c "import tests.conftest; ..." (manual runner)

Supports: import pytest, pytest.mark.parametrize, pytest.raises, pytest.skip
"""
import sys
import types
import functools

# Only stub if real pytest is not available
if 'pytest' not in sys.modules:
    try:
        import pytest  # noqa: F401
    except ImportError:
        _pytest = types.ModuleType('pytest')

        class _Mark:
            def __init__(self, name, args, kwargs):
                self.name = name
                self.args = args
                self.kwargs = kwargs

            def __call__(self, func):
                @functools.wraps(func)
                def wrapper(*a, **kw):
                    return func(*a, **kw)
                return wrapper

        class _MarkDecorator:
            def __init__(self, mark):
                self.mark = mark

            def __call__(self, func):
                @functools.wraps(func)
                def wrapper(*a, **kw):
                    return func(*a, **kw)
                return wrapper

        class _MarkGenerator:
            def __getattr__(self, name):
                def marker(*args, **kwargs):
                    return _MarkDecorator(_Mark(name, args, kwargs))
                return marker

        _pytest.mark = _MarkGenerator()

        def _parametrize(argnames, argvalues, **kwargs):
            """Stub: inject argvalues as attributes, run once with first set."""
            def decorator(func):
                @functools.wraps(func)
                def wrapper(*a, **kw):
                    # Parse argnames
                    names = [n.strip() for n in argnames.split(',')] if isinstance(argnames, str) else argnames
                    if argvalues:
                        first = argvalues[0]
                        if not isinstance(first, (list, tuple)):
                            first = (first,)
                        if len(names) == 1:
                            kw[names[0]] = first[0]
                        else:
                            for n, v in zip(names, first):
                                kw[n] = v
                    return func(*a, **kw)
                return wrapper
            return decorator

        _pytest.mark.parametrize = _parametrize

        def raises(exception, match=None):
            class _RaisesContext:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    if exc_type is None:
                        raise AssertionError(f"DID NOT RAISE {exception}")
                    if not issubclass(exc_type, exception):
                        return False
                    if match:
                        import re
                        if not re.search(match, str(exc_val)):
                            raise AssertionError(f"Pattern '{match}' not found in '{exc_val}'")
                    return True
            return _RaisesContext()

        _pytest.raises = raises

        def skip(reason=""):
            raise unittest.SkipTest(reason)

        _pytest.skip = skip

        class _FixtureFunctionMarker:
            def __call__(self, func):
                return func

        _pytest.fixture = _FixtureFunctionMarker()

        sys.modules['pytest'] = _pytest
