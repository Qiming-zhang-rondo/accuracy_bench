"""Tests for resident packed routed-expert storage."""

import torch

from accuracy_checker.resident_experts import (
    _build_resident_store,
    _dequant_resident_chunk,
    _dequant_expert_triplet,
    _group_expert_keys,
)


def _tiny_w4_weights(prefix, experts=2):
    weights = {}
    for expert_id in range(experts):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = f"{prefix}.{expert_id}.{projection}"
            weights[f"{name}.weight"] = torch.tensor(
                [[0x21, 0x43, 0x65], [0x12, 0x34, 0x56]], dtype=torch.int8)
            weights[f"{name}.weight_scale"] = torch.ones(4)
    return weights


def test_resident_store_stacks_experts_and_serves_slices():
    prefix = "model.layers.0.mlp.experts"
    weights = _tiny_w4_weights(prefix)
    groups = _group_expert_keys(weights, {prefix})
    store = _build_resident_store(prefix, groups[prefix], weights, "cpu", 2)

    assert store.tensors["gate_proj.weight"].shape == (2, 2, 3)
    torch.testing.assert_close(
        store.get(f"{prefix}.1.gate_proj.weight"),
        weights[f"{prefix}.1.gate_proj.weight"],
    )


def test_resident_w4_dequantizes_only_selected_expert():
    prefix = "model.layers.0.mlp.experts"
    weights = _tiny_w4_weights(prefix)
    store = _build_resident_store(prefix, list(weights), weights, "cpu", 2)
    desc = {
        f"{prefix}.{expert_id}.{projection}.weight": "W4A8_DYNAMIC"
        for expert_id in range(2)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }

    gate_up, down = _dequant_expert_triplet(
        prefix, 1, store, desc, torch.float32, "cpu")

    assert gate_up.shape == (8, 3)
    assert down.shape == (4, 3)


def test_resident_chunk_matches_individual_w4_dequantization():
    prefix = "model.layers.0.mlp.experts"
    weights = _tiny_w4_weights(prefix)
    store = _build_resident_store(prefix, list(weights), weights, "cpu", 2)
    desc = {
        f"{prefix}.{expert_id}.{projection}.weight": "W4A8_DYNAMIC"
        for expert_id in range(2)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }

    batched = _dequant_resident_chunk(
        prefix, [0, 1], store, desc, torch.float32, "cpu", False)
    individual = [
        _dequant_expert_triplet(
            prefix, expert_id, store, desc, torch.float32, "cpu")
        for expert_id in range(2)
    ]

    for actual, expected in zip(batched, individual):
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
