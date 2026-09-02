"""Native FP8 checkpoint detection regression tests."""

import json

import torch

from accuracy_checker.model_loader import (
    _config_declares_native_fp8,
    dequantize_native_fp8_weight,
    is_native_fp8_model,
    is_quantized_model,
    native_quant_description,
)


def test_glm_native_fp8_config_is_detected(tmp_path):
    config = {
        "model_type": "glm_moe_dsa",
        "dtype": "bfloat16",
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "activation_scheme": "dynamic",
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


def test_native_fp8_dequant_allows_partial_tail_block():
    # E4M3FN byte 0x38 is exactly 1.0.  576 rows require five 128-row
    # scale blocks, with the final block covering only rows [512, 576).
    weight = torch.full((576, 256), 0x38, dtype=torch.uint8)
    scale = torch.arange(1, 11, dtype=torch.float32).reshape(5, 2)

    actual = dequantize_native_fp8_weight(
        weight,
        scale,
        dtype=torch.float32,
        block_size=(128, 128),
    )

    expected = scale.repeat_interleave(128, dim=0)
    expected = expected.repeat_interleave(128, dim=1)[:576, :256]
    torch.testing.assert_close(actual, expected)


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
