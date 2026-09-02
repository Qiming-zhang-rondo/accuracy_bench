"""Native FP8 checkpoint detection regression tests."""

import json

from accuracy_checker.model_loader import (
    _config_declares_native_fp8,
    is_native_fp8_model,
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


def test_safetensors_header_detects_fp8_when_config_omits_quantization(tmp_path):
    config = {"model_type": "glm_moe_dsa", "dtype": "bfloat16"}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    header = json.dumps({
        "model.layers.0.self_attn.q_a_proj.weight": {
            "dtype": "F8_E4M3",
            "shape": [128, 128],
            "data_offsets": [0, 16384],
        }
    }).encode("utf-8")
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header
    )

    assert is_native_fp8_model(str(tmp_path))
    assert is_quantized_model(str(tmp_path))
    assert native_quant_description(str(tmp_path)) == {
        "__acc_native_fp8__": "FP8_E4M3"
    }
