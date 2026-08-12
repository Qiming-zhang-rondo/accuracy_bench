"""Backward-compatible inference/logits comparison entry points.

The reusable implementation now lives in :mod:`accuracy_checker`; this file
is retained because older workflows import ``collect_logits_streaming.py``
directly.
"""

from accuracy_checker.inference_compare import (
    compare_inference,
    detect_garbled,
    detect_repeat,
)
from accuracy_checker.logits_compare import (
    LogitsCollection,
    LogitsComparison,
    collect_last_logits,
    collect_logits,
    compare_logits,
)


__all__ = [
    "LogitsCollection",
    "LogitsComparison",
    "collect_logits",
    "collect_last_logits",
    "compare_logits",
    "compare_inference",
    "detect_repeat",
    "detect_garbled",
]
