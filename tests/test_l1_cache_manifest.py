"""Regression tests for latest-run L1/L2 cache selection."""

from accuracy_checker.cache import (
    load_latest_l1_cache_manifest,
    save_latest_l1_cache_manifest,
    set_cache_dir,
)


def test_latest_manifest_replaces_layer_set_without_deleting_history(tmp_path):
    set_cache_dir(str(tmp_path))
    try:
        args = ("/models/ref", "/models/quant", "prompt:hello", "dequantize")
        save_latest_l1_cache_manifest(*args, layers=[9, 10, 11])
        assert load_latest_l1_cache_manifest(*args) == [9, 10, 11]

        save_latest_l1_cache_manifest(*args, layers=[14, 15])
        assert load_latest_l1_cache_manifest(*args) == [14, 15]
    finally:
        set_cache_dir(None)


def test_manifest_identity_separates_model_pairs(tmp_path):
    set_cache_dir(str(tmp_path))
    try:
        save_latest_l1_cache_manifest(
            "/models/ref-53", "/models/quant-53", "prompt:x", "dequantize", [14]
        )
        assert load_latest_l1_cache_manifest(
            "/models/ref-52", "/models/quant-52", "prompt:x", "dequantize"
        ) is None
    finally:
        set_cache_dir(None)
