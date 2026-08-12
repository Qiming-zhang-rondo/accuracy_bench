"""Regression tests for packed shared-expert loading and activation hooks."""

import pytest
import torch
import torch.nn as nn

from accuracy_checker.layer1_block_compare import ShardedBlockComparator
from accuracy_checker.model_loader import _load_msslim_quant_param
from accuracy_checker.utils import normalize_quant_desc_values, normalize_quant_type


class _Reader:
    def __init__(self, tensors):
        self.tensors = tensors

    def get_tensor(self, name):
        return self.tensors.get(name)


def test_missing_split_descriptor_recovers_dim0_packed_w4_weight():
    name = "model.model.layers.0.mlp.shared_expert.gate_proj.weight"
    quant_name = name.rsplit(".", 1)[0]
    packed = torch.tensor([
        [0x21, 0x43, 0x65],
        [0x12, 0x34, 0x56],
    ], dtype=torch.int8)
    reader = _Reader({
        name: packed,
        f"{quant_name}.weight_scale": torch.ones(4),
    })
    param = nn.Parameter(torch.empty(4, 3))
    quant_desc = {
        "model.model.layers.0.mlp.shared_expert.down_proj": "W4A4_LAOS",
    }

    loaded = _load_msslim_quant_param(
        name, param, reader, quant_desc, torch.float32, False, False
    )

    assert loaded
    assert tuple(param.shape) == (4, 3)
    assert param.dtype == torch.float32
    assert param._acc_quant_type == "W4A4_LAOS"


def test_msmodelslim_int4_dynamic_alias_loads_normal_weight():
    name = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
    quant_name = name.rsplit(".", 1)[0]
    reader = _Reader({
        name: torch.tensor([
            [0x21, 0x43, 0x65],
            [0x12, 0x34, 0x56],
        ], dtype=torch.int8),
        f"{quant_name}.weight_scale": torch.ones(4),
    })
    param = nn.Parameter(torch.empty(4, 3))

    loaded = _load_msslim_quant_param(
        name, param, reader, {name: "W4A4_INT4_DYNAMIC"},
        torch.float32, False, False,
    )

    assert loaded
    assert tuple(param.shape) == (4, 3)
    assert param._acc_quant_type == "W4A4_DYNAMIC"


def test_msmodelslim_int4_dynamic_alias_loads_streaming_expert():
    prefix = "model.language_model.layers.0.mlp.experts"
    quant_name = f"{prefix}.0.gate_proj"
    reader = _Reader({
        f"{quant_name}.weight": torch.tensor([
            [0x21, 0x43, 0x65],
            [0x12, 0x34, 0x56],
        ], dtype=torch.int8),
        f"{quant_name}.weight_scale": torch.ones(4),
    })
    comparator = object.__new__(ShardedBlockComparator)
    comparator.dtype = torch.float32

    resolved = comparator._streaming_quant_type(
        {f"{quant_name}.weight": "W4A4_INT4_DYNAMIC"},
        f"{quant_name}.weight",
    )
    actual = comparator._dequant_streaming_proj(
        reader, prefix, 0, "gate_proj", resolved, "cpu"
    )

    assert resolved == "W4A4_DYNAMIC"
    assert tuple(actual.shape) == (4, 3)


def test_quant_description_value_alias_is_canonicalized():
    assert normalize_quant_type("W4A4_INT4_DYNAMIC") == "W4A4_DYNAMIC"
    desc = normalize_quant_desc_values({
        "layer.weight": "W4A4_INT4_DYNAMIC",
        "metadata": {"version": 1},
    })
    assert desc["layer.weight"] == "W4A4_DYNAMIC"
    assert desc["metadata"] == {"version": 1}


def test_missing_split_descriptor_recovers_dim1_packed_mxfp4_weight():
    name = "model.model.layers.0.mlp.shared_expert.down_proj.weight"
    quant_name = name.rsplit(".", 1)[0]
    reader = _Reader({
        name: torch.zeros(2, 16, dtype=torch.uint8),
        f"{quant_name}.weight_scale": torch.full(
            (2, 1), 127, dtype=torch.uint8
        ),
    })
    param = nn.Parameter(torch.empty(2, 32))

    loaded = _load_msslim_quant_param(
        name, param, reader, {}, torch.float32, False, False
    )

    assert loaded
    assert tuple(param.shape) == (2, 32)
    torch.testing.assert_close(param, torch.zeros_like(param))


def test_integer_float_misclassification_fails_before_accelerator_forward():
    name = "model.model.layers.0.mlp.shared_expert.gate_proj.weight"
    reader = _Reader({name: torch.ones(2, 3, dtype=torch.int8)})
    param = nn.Parameter(torch.empty(4, 3))

    with pytest.raises(ValueError, match="classified as FLOAT"):
        _load_msslim_quant_param(
            name, param, reader, {}, torch.float32, False, False
        )


def test_explicit_quant_shape_mismatch_fails_during_loading():
    name = "model.model.layers.0.mlp.shared_expert.down_proj.weight"
    quant_name = name.rsplit(".", 1)[0]
    reader = _Reader({
        name: torch.ones(2, 3, dtype=torch.int8),
        f"{quant_name}.weight_scale": torch.ones(4),
    })
    param = nn.Parameter(torch.empty(5, 3))

    with pytest.raises(ValueError, match="quantized weight shape mismatch"):
        _load_msslim_quant_param(
            name, param, reader, {name: "W4A4_DYNAMIC"},
            torch.float32, False, False,
        )


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)
        self.float_proj = nn.Linear(4, 4, bias=False)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer()])


def test_activation_quant_hooks_are_registered_on_quant_model_only():
    comparator = object.__new__(ShardedBlockComparator)
    comparator.activation_quant = True
    comparator.activation_quant_type = "AUTO"
    comparator.verbose = False
    comparator._activation_hooks = []
    comparator._quant_quant_desc = {
        "model.layers.0.proj.weight": "W4A4_INT4_DYNAMIC",
        "model.layers.0.float_proj.weight": "FLOAT",
    }
    ref_model = _Model()
    quant_model = _Model()

    comparator._register_quant_activation_hooks(quant_model, [0])

    assert len(ref_model.model.layers[0].proj._forward_pre_hooks) == 0
    assert len(quant_model.model.layers[0].proj._forward_pre_hooks) == 1
    assert len(quant_model.model.layers[0].float_proj._forward_pre_hooks) == 0
    assert len(comparator._activation_hooks) == 1
    comparator._clear_activation_quant_hooks()


def test_explicit_activation_type_does_not_touch_other_schemes():
    comparator = object.__new__(ShardedBlockComparator)
    comparator.activation_quant = True
    comparator.activation_quant_type = "W4A4_DYNAMIC"
    comparator.verbose = False
    comparator._activation_hooks = []
    comparator._quant_quant_desc = {
        "model.layers.0.proj.weight": "W4A4_INT4_DYNAMIC",
        "model.layers.0.float_proj.weight": "W8A8_MXFP8",
    }
    quant_model = _Model()

    comparator._register_quant_activation_hooks(quant_model, [0])

    assert len(quant_model.model.layers[0].proj._forward_pre_hooks) == 1
    assert len(quant_model.model.layers[0].float_proj._forward_pre_hooks) == 0
    comparator._clear_activation_quant_hooks()


def test_activation_quant_requires_checkpoint_descriptors():
    comparator = object.__new__(ShardedBlockComparator)
    comparator.activation_quant = True
    comparator.activation_quant_type = "AUTO"
    comparator.verbose = False
    comparator._activation_hooks = []
    comparator._quant_quant_desc = None

    with pytest.raises(ValueError, match="quant_model_description.json"):
        comparator._register_quant_activation_hooks(_Model(), [0])


def test_shared_expert_shape_guard_reports_before_matmul():
    class _SharedExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(8, 2, bias=False)
            self.up_proj = nn.Linear(8, 2, bias=False)
            self.down_proj = nn.Linear(4, 8, bias=False)

    mlp = nn.Module()
    mlp.shared_expert = _SharedExpert()

    with pytest.raises(RuntimeError, match="projection shapes are inconsistent"):
        ShardedBlockComparator._forward_shared_expert(
            mlp, torch.randn(1, 3, 8)
        )
