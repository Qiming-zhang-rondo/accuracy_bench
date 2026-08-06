"""CPU-only regression tests for msModelSlim W8A8 dequantization."""

from types import SimpleNamespace

import torch

from accuracy_checker.layer1_block_compare import ShardedBlockComparator
from accuracy_checker.model_loader import ShardWeightReader, dequantize_weight_static


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


class _PackedReader:
    def __init__(self, tensors):
        self.tensors = tensors
        self.weight_map = {key: "model.safetensors" for key in tensors}
        self.sliced = []

    def get_tensor(self, name):
        return self.tensors.get(name)

    def get_tensor_shape(self, name):
        tensor = self.tensors.get(name)
        return tuple(tensor.shape) if tensor is not None else None

    def get_tensor_slice(self, name, index, expected_first_dim=None):
        tensor = self.tensors.get(name)
        if tensor is None:
            return None
        if (
            tensor.dim() > 0
            and tensor.shape[0] == expected_first_dim
        ):
            self.sliced.append((name, index))
            return tensor[index]
        return tensor


def _packed_comparator():
    comparator = object.__new__(ShardedBlockComparator)
    comparator.dtype = torch.float32
    comparator.activation_quant = False
    return comparator


def _packed_mlp():
    return SimpleNamespace(config=SimpleNamespace(hidden_act="silu"))


def test_qwen36_packed_bf16_expert_streams_one_slice():
    prefix = "model.language_model.layers.0.mlp.experts"
    gate_up_key = f"{prefix}.gate_up_proj"
    down_key = f"{prefix}.down_proj"
    gate_up = torch.tensor([
        [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 0.]],
        [[2., 0., 0.], [0., 2., 0.], [0., 0., 2.], [1., 0., 1.]],
    ])
    down = torch.tensor([
        [[1., 0.], [0., 1.], [1., 1.]],
        [[2., 0.], [0., 2.], [1., -1.]],
    ])
    reader = _PackedReader({gate_up_key: gate_up, down_key: down})
    comparator = _packed_comparator()
    x = torch.tensor([[0.5, -1.0, 2.0]])

    actual = comparator._streaming_expert_forward(
        1, x, "cpu", prefix, reader, None, False, False, _packed_mlp()
    )

    gate = torch.nn.functional.linear(x, gate_up[1, :2])
    up = torch.nn.functional.linear(x, gate_up[1, 2:])
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(gate) * up, down[1]
    )
    torch.testing.assert_close(actual, expected)
    assert (gate_up_key, 1) in reader.sliced
    assert (down_key, 1) in reader.sliced


def test_qwen36_packed_w8a8_expert_slices_scales_before_dequant():
    prefix = "model.language_model.layers.0.mlp.experts"
    gate_up_key = f"{prefix}.gate_up_proj"
    down_key = f"{prefix}.down_proj"
    gate_up = torch.tensor([
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]],
        [[2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 0, 1]],
    ], dtype=torch.int8)
    down = torch.tensor([
        [[1, 0], [0, 1], [1, 1]],
        [[2, 0], [0, 2], [1, -1]],
    ], dtype=torch.int8)
    gate_scale = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
    ])
    down_scale = torch.tensor([
        [0.2, 0.3, 0.4],
        [0.6, 0.7, 0.8],
    ])
    gate_input_scale = torch.tensor([0.25, 0.5])
    down_input_scale = torch.tensor([0.5, 0.25])
    reader = _PackedReader({
        gate_up_key: gate_up,
        f"{gate_up_key}.deq_scale": _encode_ascend_deq_scale(
            gate_scale * gate_input_scale.unsqueeze(1)
        ),
        f"{gate_up_key}.input_scale": gate_input_scale,
        down_key: down,
        f"{down_key}.deq_scale": _encode_ascend_deq_scale(
            down_scale * down_input_scale.unsqueeze(1)
        ),
        f"{down_key}.input_scale": down_input_scale,
    })
    comparator = _packed_comparator()
    x = torch.tensor([[0.5, -1.0, 2.0]])

    actual = comparator._streaming_expert_forward(
        1, x, "cpu", prefix, reader,
        {gate_up_key: "W8A8", down_key: "W8A8"},
        True, False, _packed_mlp(),
    )

    gate_up_fp = gate_up[1].float() * gate_scale[1].unsqueeze(1)
    down_fp = down[1].float() * down_scale[1].unsqueeze(1)
    gate = torch.nn.functional.linear(x, gate_up_fp[:2])
    up = torch.nn.functional.linear(x, gate_up_fp[2:])
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(gate) * up, down_fp
    )
    torch.testing.assert_close(actual, expected)
    assert (f"{gate_up_key}.deq_scale", 1) in reader.sliced
    assert (f"{down_key}.deq_scale", 1) in reader.sliced


def test_shard_reader_closes_safe_open_context_without_close_method():
    class _Handle:
        def __init__(self):
            self.exited = False

        def __exit__(self, exc_type, exc_value, traceback):
            self.exited = True

    handle = _Handle()
    reader = object.__new__(ShardWeightReader)
    reader._sf_cache = {"model.safetensors": handle}

    reader.close()

    assert handle.exited
    assert reader._sf_cache == {}
