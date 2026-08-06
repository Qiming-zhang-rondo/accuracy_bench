"""CPU-only regression tests for msModelSlim W8A8 dequantization."""

import torch

from accuracy_checker.layer1_block_compare import ShardedBlockComparator
from accuracy_checker.model_loader import dequantize_weight_static


def _encode_ascend_deq_scale(scale: torch.Tensor) -> torch.Tensor:
    """Mirror msModelSlim: float32 bits -> int32 -> int64."""
    return scale.to(torch.float32).contiguous().view(torch.int32).to(torch.int64)


def test_static_w8a8_decodes_ascend_int64_bit_pattern():
    weight = torch.tensor([[10, -20], [30, 40]], dtype=torch.int8)
    input_scale = torch.tensor([0.5, 0.25], dtype=torch.float32)
    weight_scale = torch.tensor([0.02, 0.04], dtype=torch.float32)
    encoded = _encode_ascend_deq_scale(input_scale * weight_scale)

    actual, had_input_scale = dequantize_weight_static(
        weight, encoded, input_scale, dtype=torch.float32
    )

    expected = weight.float() * weight_scale.unsqueeze(1)
    assert had_input_scale
    torch.testing.assert_close(actual, expected)


def test_static_w8a8_accepts_int32_bit_pattern_too():
    weight = torch.tensor([[8, -4]], dtype=torch.int8)
    weight_scale = torch.tensor([0.125], dtype=torch.float32)
    encoded = weight_scale.contiguous().view(torch.int32)

    actual, had_input_scale = dequantize_weight_static(
        weight, encoded, dtype=torch.float32
    )

    assert not had_input_scale
    torch.testing.assert_close(actual, weight.float() * weight_scale.unsqueeze(1))


class _Reader:
    def __init__(self, tensors):
        self.tensors = tensors

    def get_tensor(self, name):
        return self.tensors.get(name)


def test_grouped_dual_streaming_expert_dequantizes_static_w8a8():
    prefix = "model.layers.0.mlp.experts"
    quant_name = f"{prefix}.0.gate_proj"
    weight = torch.tensor([[10, -20], [30, 40]], dtype=torch.int8)
    input_scale = torch.tensor([0.5, 0.25], dtype=torch.float32)
    weight_scale = torch.tensor([0.02, 0.04], dtype=torch.float32)
    reader = _Reader({
        f"{quant_name}.weight": weight,
        f"{quant_name}.deq_scale": _encode_ascend_deq_scale(
            input_scale * weight_scale
        ),
        f"{quant_name}.input_scale": input_scale,
    })
    comparator = object.__new__(ShardedBlockComparator)
    comparator.dtype = torch.float32

    actual = comparator._dequant_streaming_proj(
        reader, prefix, 0, "gate_proj", "W8A8", "cpu"
    )

    expected = weight.float() * weight_scale.unsqueeze(1)
    torch.testing.assert_close(actual, expected)


def test_grouped_dual_streaming_expert_dequantizes_sparse_w8a8s():
    prefix = "model.layers.0.mlp.experts"
    quant_name = f"{prefix}.0.up_proj"
    weight = torch.tensor([[6, -3]], dtype=torch.int8)
    input_scale = torch.tensor([0.5], dtype=torch.float32)
    weight_scale = torch.tensor([0.02], dtype=torch.float32)
    reader = _Reader({
        f"{quant_name}.weight": weight,
        f"{quant_name}.deq_scale": _encode_ascend_deq_scale(
            input_scale * weight_scale
        ),
        f"{quant_name}.input_scale": input_scale,
    })
    comparator = object.__new__(ShardedBlockComparator)
    comparator.dtype = torch.float32

    actual = comparator._dequant_streaming_proj(
        reader, prefix, 0, "up_proj", "W8A8S", "cpu"
    )

    torch.testing.assert_close(
        actual, weight.float() * weight_scale.unsqueeze(1)
    )


def test_grouped_dual_streaming_expert_dequantizes_dynamic_w8a8():
    prefix = "model.layers.0.mlp.experts"
    quant_name = f"{prefix}.0.down_proj"
    weight = torch.tensor([[4, -2], [1, 3]], dtype=torch.int8)
    scale = torch.tensor([0.25, 0.5], dtype=torch.float32)
    offset = torch.tensor([1.0, -1.0], dtype=torch.float32)
    reader = _Reader({
        f"{quant_name}.weight": weight,
        f"{quant_name}.weight_scale": scale,
        f"{quant_name}.weight_offset": offset,
    })
    comparator = object.__new__(ShardedBlockComparator)
    comparator.dtype = torch.float32

    actual = comparator._dequant_streaming_proj(
        reader, prefix, 0, "down_proj", "W8A8_DYNAMIC", "cpu"
    )

    expected = (weight.float() - offset.unsqueeze(1)) * scale.unsqueeze(1)
    torch.testing.assert_close(actual, expected)
