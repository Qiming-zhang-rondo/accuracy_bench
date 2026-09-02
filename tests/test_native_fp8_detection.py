"""Native FP8 checkpoint detection regression tests."""

import json

from accuracy_checker.model_loader import (
    _config_declares_native_fp8,
    is_quantized_model,
    native_quant_description,
)


def test_glm_native_fp8_config_is_detected(tmp_path):
    config = {
        "model_type": "glm_moe_dsa",
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
        },
    }
    assert _config_declares_native_fp8(config)
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert is_quantized_model(str(tmp_path))
    assert native_quant_description(str(tmp_path)) == {
        "__acc_native_fp8__": "FP8_E4M3"
    }


def test_bf16_config_is_not_marked_native_fp8():
    config = {"model_type": "glm_moe_dsa", "torch_dtype": "bfloat16"}
    assert not _config_declares_native_fp8(config)
