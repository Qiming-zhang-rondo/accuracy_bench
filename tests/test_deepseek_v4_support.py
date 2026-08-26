"""Structural regressions for the official DeepSeek-V4 integration."""

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
