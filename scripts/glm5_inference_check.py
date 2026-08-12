#!/usr/bin/env python3
"""Compatibility CLI for the historical GLM inference-check command.

The implementation is model-structure driven and now shared by GLM, Qwen,
and other supported Hugging Face checkpoints.  Keep this thin entry point so
existing internal scripts continue to work after pulling the open-source tree.
"""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from accuracy_checker.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
