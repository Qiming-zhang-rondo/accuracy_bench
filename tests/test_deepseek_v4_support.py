"""Structural regressions for the official DeepSeek-V4 integration."""

import json

from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from torch import nn


class DeepseekV4Attention(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("q_a_proj", "q_a_norm", "q_b_proj", "q_b_norm",
                     "kv_proj", "kv_norm", "o_a_proj", "o_b_proj"):
            setattr(self, name, nn.Identity())
        scorer = nn.Module()
        scorer.weights_proj = nn.Identity()
        indexer = nn.Module()
        indexer.kv_proj = nn.Identity()
        indexer.gate_proj = nn.Identity()
        indexer.kv_norm = nn.Identity()
        indexer.q_b_proj = nn.Identity()
        indexer.scorer = scorer
        compressor = nn.Module()
        compressor.kv_proj = nn.Identity()
        compressor.gate_proj = nn.Identity()
        compressor.kv_norm = nn.Identity()
        compressor.indexer = indexer
        self.compressor = compressor


class _Moe(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Identity()
        self.shared_experts = nn.Identity()
        self.experts = nn.Identity()


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DeepseekV4Attention()
        self.mlp = _Moe()


def test_v4_detection_and_fine_subgraphs():
    from accuracy_checker.subgraph_locate import detect_model_type, get_subgraph_names

    layer = DeepseekV4DecoderLayer()
    assert detect_model_type(layer) == "deepseek_v4"
    names = get_subgraph_names("deepseek_v4", layer, mla_fine=True)
    assert "self_attn.compressor.indexer" in names
    assert "self_attn.compressor.indexer.scorer.weights_proj" in names
    assert "mlp.gate" in names
    assert "mlp.shared_experts" in names
    assert "mlp.experts" in names


def test_v4_streaming_detection_accepts_packed_experts():
    from accuracy_checker.inference_check import _hf_detect_streaming_mode

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="deepseek_v4")
            self.experts = nn.Module()
            self.experts.gate_up_proj = nn.Parameter(torch.empty(2, 4, 3))
            self.experts.down_proj = nn.Parameter(torch.empty(2, 3, 2))

    assert _hf_detect_streaming_mode(Model(), "/unused", True) == (True, True)


def test_v4_hash_router_receives_cached_input_ids():
    from accuracy_checker.layer1_block_compare import ShardedBlockComparator

    class HashGate(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("tid2eid", torch.zeros(8, 2, dtype=torch.long))
            self.seen = None

        def forward(self, hidden, input_ids):
            self.seen = input_ids.detach().clone()
            rows = hidden.numel() // hidden.shape[-1]
            logits = torch.zeros(rows, 4)
            weights = torch.full((rows, 2), 0.5)
            indices = self.tid2eid[input_ids.reshape(-1)]
            return logits, weights, indices

    comparator = ShardedBlockComparator.__new__(ShardedBlockComparator)
    comparator._input_ids = torch.tensor([[1, 2, 3]])
    gate = HashGate()
    _, scores, indices = comparator._resolve_gate_output(
        gate, torch.zeros(1, 3, 4)
    )
    assert torch.equal(gate.seen, comparator._input_ids)
    assert scores.shape == (3, 2)
    assert indices.shape == (3, 2)


def test_cache_v4_payload_keeps_input_ids(tmp_path, monkeypatch):
    from accuracy_checker import cache

    monkeypatch.setattr(cache, "_cache_dir_override", str(tmp_path))
    hidden = torch.randn(1, 3, 4)
    ids = torch.tensor([[4, 5, 6]])
    cache.save_cache("m", "p", 3, 1, "ref", "dequantize", hidden, input_ids=ids)
    loaded = cache.load_cache("m", "p", 3, 1, "ref", "dequantize", "cpu")
    assert torch.equal(loaded["hidden_states"], hidden)
    assert torch.equal(loaded["input_ids"], ids)


def test_official_v4_reader_maps_and_packs_split_experts(tmp_path):
    from accuracy_checker.model_loader import (
        ShardWeightReader,
        load_layer_weights_indexed,
    )

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.empty(2, 4, 3))
            self.down_proj = nn.Parameter(torch.empty(2, 3, 2))

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = Experts()

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = MLP()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Layer()])

    tensors = {}
    for expert_id in range(2):
        prefix = f"model.layers.0.ffn.experts.{expert_id}"
        tensors[f"{prefix}.w1.weight"] = torch.full((2, 3), expert_id + 1.0)
        tensors[f"{prefix}.w3.weight"] = torch.full((2, 3), expert_id + 3.0)
        tensors[f"{prefix}.w2.weight"] = torch.full((3, 2), expert_id + 5.0)
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    weight_map = {key: path.name for key in tensors}
    reader = ShardWeightReader(str(tmp_path), weight_map)
    model = Model()
    loaded = load_layer_weights_indexed(
        model, str(tmp_path), [0], "cpu", torch.float32, weight_map, reader,
        is_quant=False, verbose=False,
    )
    assert loaded == 2
    gate_up = model.model.layers[0].mlp.experts.gate_up_proj
    assert torch.equal(gate_up[1, :2], tensors["model.layers.0.ffn.experts.1.w1.weight"])
    assert torch.equal(gate_up[1, 2:], tensors["model.layers.0.ffn.experts.1.w3.weight"])
    assert torch.equal(
        model.model.layers[0].mlp.experts.down_proj[1],
        tensors["model.layers.0.ffn.experts.1.w2.weight"],
    )
    reader.close()


def test_modelscope_v4_bare_prefixes_map_to_runtime_names(tmp_path):
    from accuracy_checker.model_loader import ShardWeightReader

    tensors = {
        "embed.weight": torch.ones(2, 2),
        "norm.weight": torch.ones(2),
        "head.weight": torch.ones(2, 2),
        "layers.0.ffn.experts.0.w1.weight": torch.ones(2, 2),
    }
    path = tmp_path / "quant_model_weights-00001-of-00001.safetensors"
    save_file(tensors, str(path))
    reader = ShardWeightReader(
        str(tmp_path), {key: path.name for key in tensors}
    )
    assert reader.get_tensor("model.embed_tokens.weight") is not None
    assert reader.get_tensor("model.norm.weight") is not None
    assert reader.get_tensor("lm_head.weight") is not None
    assert reader.get_tensor(
        "model.layers.0.mlp.experts.0.gate_proj.weight"
    ) is not None
    reader.close()


def test_v4_bare_expert_prefix_index_and_mtp_filter():
    from accuracy_checker.layer1_block_compare import ShardedBlockComparator
    from accuracy_checker.model_loader import _decide_should_load

    reader = SimpleNamespace(weight_map={
        "layers.0.ffn.experts.0.w1.weight": "shard.safetensors",
    })
    index = ShardedBlockComparator._expert_prefix_index(reader)
    assert index[(0, "ffn")] == "layers.0.ffn.experts"
    assert not _decide_should_load(
        "model.mtp.0.attn.wq_a.weight", None, set(),
        load_embed_only=True, load_norm_head_only=False, verbose=False,
    )
    assert _decide_should_load(
        "model.mtp.0.attn.wq_a.weight", None, set(),
        load_embed_only=True, load_norm_head_only=False, verbose=False,
        include_auxiliary=True,
    )


def test_v4_descriptor_lookup_accepts_runtime_and_native_expert_names():
    from accuracy_checker.layer1_block_compare import _lookup_quant_descriptor

    assert _lookup_quant_descriptor(
        {"model.layers.0.mlp.experts.0.gate_proj.weight": "W8A8_DYNAMIC"},
        "layers.0.ffn.experts.0.w1.weight",
    ) == "W8A8_DYNAMIC"


def test_official_v4_fp4_config_and_decode(tmp_path):
    from accuracy_checker.model_loader import (
        dequantize_deepseek_v4_fp4,
        is_quantized_model,
        native_quant_description,
    )

    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "deepseek_v4", "expert_dtype": "fp4",
        "quantization_config": {"quant_method": "fp8"},
    }))
    assert is_quantized_model(str(tmp_path))
    assert native_quant_description(str(tmp_path)) == {
        "__acc_deepseek_v4_fp4__": "DEEPSEEK_FP4"
    }

    # low nibble=+0.5, high nibble=-1.0; scale=2 produces [+1, -2].
    # V4 scales one 32-logical-column group, thus 16 packed int8 columns.
    packed = torch.zeros((1, 16), dtype=torch.int8)
    packed[0, 0] = -95
    scale = torch.tensor([[2.0]])
    decoded = dequantize_deepseek_v4_fp4(packed, scale, torch.float32)
    expected = torch.zeros((1, 32))
    expected[0, :2] = torch.tensor([1.0, -2.0])
    torch.testing.assert_close(decoded, expected)
    assert _lookup_quant_descriptor(
        {"layers.0.ffn.experts.0.w2.weight": "W8A8_DYNAMIC"},
        "model.layers.0.mlp.experts.0.down_proj.weight",
    ) == "W8A8_DYNAMIC"


def test_official_v4_reader_uses_native_quant_descriptor(tmp_path):
    from accuracy_checker.model_loader import (
        ShardWeightReader,
        load_layer_weights_indexed,
    )

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_a_proj = nn.Linear(3, 2, bias=False)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Layer()])

    native = "model.layers.0.attn.wq_a"
    tensors = {
        f"{native}.weight": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int8),
        f"{native}.weight_scale": torch.tensor([0.5, 0.25]),
    }
    path = tmp_path / "quant_model_weights.safetensors"
    save_file(tensors, str(path))
    weight_map = {key: path.name for key in tensors}
    reader = ShardWeightReader(str(tmp_path), weight_map)
    model = Model()
    loaded = load_layer_weights_indexed(
        model, str(tmp_path), [0], "cpu", torch.float32, weight_map, reader,
        is_quant=True,
        quant_desc={f"{native}.weight": "W8A8_DYNAMIC"},
        verbose=False,
    )
    assert loaded == 1
    expected = tensors[f"{native}.weight"].float() * torch.tensor([[0.5], [0.25]])
    torch.testing.assert_close(model.model.layers[0].self_attn.q_a_proj.weight, expected)
    reader.close()


def test_unindexed_v4_shards_are_discovered_from_metadata(tmp_path):
    from accuracy_checker.model_loader import build_weight_index, load_quant_weights

    first = tmp_path / "DeepSeek-V4-00001-of-00002.safetensors"
    second = tmp_path / "DeepSeek-V4-00002-of-00002.safetensors"
    save_file({"model.embed_tokens.weight": torch.zeros(2, 2)}, str(first))
    save_file({"model.layers.0.ffn.experts.0.w1.weight": torch.zeros(2, 2)}, str(second))
    index = build_weight_index(str(tmp_path))
    assert index["model.embed_tokens.weight"] == first.name
    assert index["model.layers.0.ffn.experts.0.w1.weight"] == second.name

    quant_dir = tmp_path / "quant"
    quant_dir.mkdir()
    q1 = quant_dir / "quant_model_weights-00001-of-00002.safetensors"
    q2 = quant_dir / "quant_model_weights-00002-of-00002.safetensors"
    save_file({"a.weight": torch.ones(2, 2)}, str(q1))
    save_file({"b.weight": torch.ones(2, 2)}, str(q2))
    loaded = load_quant_weights(str(quant_dir))
    assert set(loaded) == {"a.weight", "b.weight"}


def test_boundary_aliases_official_v4_nonexpert_weights():
    from accuracy_checker.model_loader import add_deepseek_v4_checkpoint_aliases

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_a_proj = nn.Linear(3, 2, bias=False)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Layer()])

    native = "model.layers.0.attn.wq_a.weight"
    runtime = "model.layers.0.self_attn.q_a_proj.weight"
    weights = {
        native: torch.randn(2, 3),
        "model.layers.0.attn.wq_a.weight_scale": torch.ones(2),
    }
    desc = {native: "W8A8_DYNAMIC"}
    assert add_deepseek_v4_checkpoint_aliases(Model(), weights, desc) == 2
    assert weights[runtime] is weights[native]
    assert desc[runtime] == "W8A8_DYNAMIC"
    assert weights[f"{runtime[:-7]}.weight_scale"] is weights[
        "model.layers.0.attn.wq_a.weight_scale"
    ]


def test_boundary_v4_split_expert_store_preserves_forward():
    from accuracy_checker.resident_experts import (
        _build_v4_split_store,
        _dequant_resident_chunk,
    )

    prefix = "model.layers.0.mlp.experts"
    native = "model.layers.0.ffn.experts"
    weights = {}
    for expert_id in range(2):
        weights[f"{native}.{expert_id}.w1.weight"] = torch.randn(2, 3)
        weights[f"{native}.{expert_id}.w3.weight"] = torch.randn(2, 3)
        weights[f"{native}.{expert_id}.w2.weight"] = torch.randn(3, 2)
    store, desc = _build_v4_split_store(prefix, weights, "cpu", True, {})
    triplets = _dequant_resident_chunk(
        prefix, [0, 1], store, desc, torch.float32, "cpu", False
    )
    for expert_id, (gate_up, down) in enumerate(triplets):
        assert torch.equal(gate_up[:2], weights[f"{native}.{expert_id}.w1.weight"])
        assert torch.equal(gate_up[2:], weights[f"{native}.{expert_id}.w3.weight"])
        assert torch.equal(down, weights[f"{native}.{expert_id}.w2.weight"])


def test_glm_query_parallel_topk_is_exact_and_ordered_on_cpu():
    """Query shards must match one-device causal running-topk exactly."""
    from accuracy_checker.glm_dsa_blockwise import (
        _query_block_assignments,
        _tp_query_parallel_topk,
    )

    torch.manual_seed(7)
    batch, seq_len, heads, dim = 1, 11, 3, 5
    q = torch.randn(batch, seq_len, heads, dim, dtype=torch.bfloat16)
    k = torch.randn(batch, seq_len, dim, dtype=torch.bfloat16)
    weights = torch.randn(batch, seq_len, heads, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long).view(1, -1)
    scale = 0.7

    scores = torch.matmul(
        q.float(), k.float().transpose(-1, -2).unsqueeze(1)
    ) * scale
    scores = torch.relu(scores)
    expected_scores = torch.matmul(
        weights.float().unsqueeze(-2), scores
    ).squeeze(-2)
    causal = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1
    )
    expected_scores = expected_scores.masked_fill(causal, float("-inf"))
    causal_mask = torch.zeros(seq_len, seq_len, dtype=torch.float32)
    causal_mask = causal_mask.masked_fill(
        torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1),
        float("-inf"),
    ).unsqueeze(0)
    for topk, key_block, query_block in ((2, 5, 3), (4, 3, 2)):
        expected = torch.topk(
            expected_scores, topk, dim=-1, sorted=True
        ).indices.to(torch.int32)
        expected = torch.where(
            expected <= position_ids[:, :, None], expected, -1
        )
        for mask in (None, causal_mask):
            actual = _tp_query_parallel_topk(
                q, k, weights, query_block, key_block, topk,
                position_ids, mask, scale,
                ["cpu", "cpu", "cpu"], "cpu",
            )
            torch.testing.assert_close(actual, expected)


def test_glm_query_block_assignment_preserves_block_boundaries_and_balance():
    from accuracy_checker.glm_dsa_blockwise import _query_block_assignments

    for seq_len, query_block, num_devices, expected_blocks in (
        (65536, 1024, 8, 64),
        (10000, 1024, 8, 10),
    ):
        assignments = _query_block_assignments(
            seq_len, query_block, num_devices
        )
        assert len(assignments) == num_devices
        blocks = [block for device_blocks in assignments for block in device_blocks]
        assert len(blocks) == expected_blocks
        assert all(q_end - q_start <= query_block for q_start, q_end in blocks)
        assert sorted(blocks) == [
            (start, min(seq_len, start + query_block))
            for start in range(0, seq_len, query_block)
        ]
        assert all(
            len(device_blocks) in {
                expected_blocks // num_devices,
                (expected_blocks + num_devices - 1) // num_devices,
            }
            for device_blocks in assignments
        )
        assert blocks[-1][1] <= seq_len
        if seq_len == 10000:
            assert (9216, 10000) in blocks


def test_v4_query_block_assignment_preserves_block_boundaries_and_balance():
    from accuracy_checker.deepseek_v4_blockwise import _query_block_assignments

    assignments = _query_block_assignments(65536, 1024, 8)
    blocks = [block for device_blocks in assignments for block in device_blocks]
    assert len(blocks) == 64
    assert all(end - start <= 1024 for start, end in blocks)
    assert sorted(blocks) == [
        (start, min(65536, start + 1024))
        for start in range(0, 65536, 1024)
    ]
    assert {len(device_blocks) for device_blocks in assignments} == {8}

    tail = _query_block_assignments(10000, 1024, 8)
    tail_blocks = [block for device_blocks in tail for block in device_blocks]
    assert len(tail_blocks) == 10
    assert (9216, 10000) in tail_blocks


def test_v4_query_parallel_indexer_blockwise_topk_is_exact_on_cpu():
    from accuracy_checker.deepseek_v4_blockwise import (
        _compute_indexer_query_block_topk,
    )

    torch.manual_seed(31)
    batch, query_len, heads, dim, compressed_len = 2, 7, 3, 5, 11
    q = torch.randn(batch, query_len, heads, dim, dtype=torch.bfloat16)
    k = torch.randn(batch, compressed_len, dim, dtype=torch.bfloat16)
    weights = torch.randn(batch, query_len, heads, dtype=torch.bfloat16)
    positions = torch.tensor([[0, 1, 3, 5, 7, 9, 10], [0, 2, 4, 6, 8, 9, 10]])
    compress_rate = 2
    causal = (positions + 1) // compress_rate
    scale = 0.37
    topk = 6  # larger than one key block, forcing running-top-k merge
    actual = _compute_indexer_query_block_topk(
        q, k, weights, causal, key_block=4, top_k=topk, scale=scale,
    )
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2).unsqueeze(1))
    scores = torch.relu(scores) * scale
    scores = torch.matmul(weights.float().unsqueeze(-2), scores).squeeze(-2)
    key_ids = torch.arange(compressed_len)
    scores = scores.masked_fill(key_ids.view(1, 1, -1) >= causal.unsqueeze(-1), float("-inf"))
    expected = torch.topk(scores, topk, dim=-1, sorted=True).indices.to(torch.int32)
    expected = torch.where(
        expected < causal.unsqueeze(-1), expected,
        torch.full_like(expected, -1),
    )
    torch.testing.assert_close(actual, expected)


def test_glm_sparse_attention_online_softmax_matches_dense_selected_reference():
    from accuracy_checker.glm_dsa_blockwise import (
        _compute_sparse_attention_query_block,
        _gather_selected_attention_states,
    )

    torch.manual_seed(19)
    batch, heads, seq_len, key_dim, value_dim = 2, 3, 7, 4, 5
    query = torch.randn(batch, heads, seq_len, key_dim)
    keys = torch.randn(batch, heads, seq_len, key_dim)
    values = torch.randn(batch, heads, seq_len, value_dim)
    positions = torch.arange(seq_len).view(1, -1).expand(batch, -1)
    # Deliberately include future entries.  They are -inf candidates used to
    # fill top-k at early positions and must not affect sparse attention.
    selected = torch.tensor(
        [
            [[0, 3, 1, 6, 2]] * seq_len,
            [[0, 4, 2, 6, 1]] * seq_len,
        ],
        dtype=torch.long,
    )
    scaling = 0.63

    gathered_k = _gather_selected_attention_states(keys, selected)
    gathered_v = _gather_selected_attention_states(values, selected)
    dense_scores = torch.matmul(
        query.unsqueeze(-2), gathered_k.transpose(-1, -2)
    ).squeeze(-2) * scaling
    valid = selected[:, None] <= positions[:, None, :, None]
    dense_scores = dense_scores.masked_fill(~valid, float("-inf"))
    expected = torch.matmul(
        dense_scores.softmax(dim=-1).unsqueeze(-2), gathered_v
    ).squeeze(-2)

    # selected_block=2 forces three online-softmax tiles (2 + 2 + 1).
    actual = _compute_sparse_attention_query_block(
        query,
        keys.contiguous(),
        values.contiguous(),
        selected,
        positions,
        None,
        0,
        scaling,
        2,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_glm_sparse_attention_respects_compact_padding_mask():
    from accuracy_checker.glm_dsa_blockwise import (
        _compute_sparse_attention_query_block,
        _gather_selected_attention_states,
    )

    torch.manual_seed(23)
    batch, heads, seq_len, dim = 1, 2, 6, 3
    query = torch.randn(batch, heads, seq_len, dim)
    keys = torch.randn(batch, heads, seq_len, dim)
    values = torch.randn(batch, heads, seq_len, dim)
    positions = torch.arange(seq_len).view(1, -1)
    selected = torch.arange(seq_len).view(1, 1, -1).expand(
        batch, seq_len, -1
    )
    padding = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)

    gathered_k = _gather_selected_attention_states(keys, selected)
    gathered_v = _gather_selected_attention_states(values, selected)
    scores = torch.matmul(
        query.unsqueeze(-2), gathered_k.transpose(-1, -2)
    ).squeeze(-2)
    valid = selected[:, None] <= positions[:, None, :, None]
    valid = valid & padding[:, None, None, :].bool()
    expected = torch.matmul(
        scores.masked_fill(~valid, float("-inf"))
        .softmax(dim=-1).unsqueeze(-2),
        gathered_v,
    ).squeeze(-2)
    actual = _compute_sparse_attention_query_block(
        query, keys.contiguous(), values.contiguous(), selected, positions,
        padding, 0, 1.0, 4,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_glm_sparse_attention_query_parallel_restores_query_order_on_cpu():
    from accuracy_checker.glm_dsa_blockwise import (
        _compute_sparse_attention_query_block,
        _tp_sparse_query_parallel_attention,
    )

    torch.manual_seed(29)
    batch, heads, seq_len, dim = 1, 2, 11, 4
    query = torch.randn(batch, heads, seq_len, dim)
    keys = torch.randn(batch, heads, seq_len, dim)
    values = torch.randn(batch, heads, seq_len, dim)
    positions = torch.arange(seq_len).view(1, -1)
    selected = torch.arange(seq_len).view(1, 1, -1).expand(
        batch, seq_len, -1
    )
    expected = _compute_sparse_attention_query_block(
        query, keys.contiguous(), values.contiguous(), selected, positions,
        None, 0, 0.5, 3,
    )
    actual = _tp_sparse_query_parallel_attention(
        query,
        keys.contiguous(),
        values.contiguous(),
        selected,
        positions,
        None,
        0.5,
        2,
        3,
        ["cpu", "cpu", "cpu"],
        "cpu",
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_boundary_v4_native_w8_experts_dequantize_per_projection():
    from accuracy_checker.resident_experts import (
        _build_v4_split_store,
        _dequant_resident_chunk,
    )

    prefix = "model.layers.0.mlp.experts"
    native = "model.layers.0.ffn.experts"
    weights = {}
    desc = {}
    for expert_id in range(2):
        for projection, shape in (("w1", (2, 3)), ("w3", (2, 3)), ("w2", (3, 2))):
            key = f"{native}.{expert_id}.{projection}.weight"
            weights[key] = torch.full(shape, expert_id + 1, dtype=torch.int8)
            weights[f"{key[:-7]}.weight_scale"] = torch.arange(
                1, shape[0] + 1, dtype=torch.float32
            )
            desc[key] = "W8A8_DYNAMIC"
    store, runtime_desc = _build_v4_split_store(
        prefix, weights, "cpu", True, desc
    )
    (gate_up, down), = _dequant_resident_chunk(
        prefix, [1], store, runtime_desc, torch.float32, "cpu", False
    )
    expected_gate = weights[f"{native}.1.w1.weight"].float() * torch.tensor([[1.], [2.]])
    expected_up = weights[f"{native}.1.w3.weight"].float() * torch.tensor([[1.], [2.]])
    expected_down = weights[f"{native}.1.w2.weight"].float() * torch.tensor([[1.], [2.], [3.]])
    torch.testing.assert_close(gate_up, torch.cat((expected_gate, expected_up)))
    torch.testing.assert_close(down, expected_down)
