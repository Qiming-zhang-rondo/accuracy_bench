"""CPU-only structural tests for Kimi K3 support."""

import json

import torch
import torch.nn as nn


class FakeKdaAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.mode = "chunk"
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 8, bias=False)
        self.v_proj = nn.Linear(8, 8, bias=False)
        self.q_conv1d = nn.Conv1d(8, 8, 1, bias=False)
        self.f_a_proj = nn.Linear(8, 4, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)


class FakeKimiGate(nn.Module):
    def forward(self, hidden_states):
        tokens = hidden_states.numel() // hidden_states.shape[-1]
        indices = torch.tensor([[1, 3]], dtype=torch.long).expand(tokens, -1)
        scores = torch.tensor([[0.6, 0.4]], dtype=hidden_states.dtype).expand(tokens, -1)
        return indices, scores


class FakeKimiMoe(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 4
        self.top_k = 2
        self.gate = FakeKimiGate()
        self.experts = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(4)])
        self.routed_expert_down_proj = nn.Linear(8, 4, bias=False)
        self.routed_expert_norm = nn.LayerNorm(4)
        self.routed_expert_up_proj = nn.Linear(4, 8, bias=False)
        self.shared_experts = nn.Linear(8, 8, bias=False)


class FakeKimiLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_attn_residuals = True
        self.self_attn = FakeKdaAttention()
        self.block_sparse_moe = FakeKimiMoe()
        self.self_attention_res_proj = nn.Linear(8, 1, bias=False)
        self.mlp_res_proj = nn.Linear(8, 1, bias=False)


def test_kimi_wrapper_model_components():
    """Kimi K3 uses language_model.model.layers and nests lm_head one level up."""
    from accuracy_checker.model_structure import get_model_components

    class TextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([FakeKimiLayer()])
            self.embed_tokens = nn.Embedding(16, 8)
            self.norm = nn.LayerNorm(8)

    class CausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = TextModel()
            self.lm_head = nn.Linear(8, 16, bias=False)

    class ConditionalGeneration(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = CausalLM()

    model = ConditionalGeneration()
    components = get_model_components(model)
    assert components.text_model is model.language_model.model
    assert components.layers is model.language_model.model.layers
    assert components.embed is model.language_model.model.embed_tokens
    assert components.lm_head is model.language_model.lm_head


def test_kimi_layer_detection_and_subgraphs():
    from accuracy_checker.subgraph_locate import detect_model_type, get_subgraph_names

    layer = FakeKimiLayer()
    assert detect_model_type(layer) == 'kimi_k3'
    names = get_subgraph_names('kimi_k3', layer, mla_fine=True)
    for expected in (
        'self_attn', 'self_attn.q_proj', 'self_attn.q_conv1d',
        'block_sparse_moe', 'block_sparse_moe.gate',
        'block_sparse_moe.routed_expert_down_proj',
        'block_sparse_moe.experts',
        'block_sparse_moe.routed_expert_up_proj',
        'block_sparse_moe.shared_experts',
        'self_attention_res_proj', 'mlp_res_proj',
    ):
        assert expected in names


def test_kimi_router_tuple_and_attnres_state():
    from accuracy_checker.layer1_block_compare import ShardedBlockComparator

    hidden = torch.randn(1, 3, 8)
    layer = FakeKimiLayer()
    router_logits, scores, indices = ShardedBlockComparator._resolve_gate_output(
        object(), layer.block_sparse_moe.gate, hidden)
    assert router_logits is None
    assert scores.shape == (3, 2)
    assert indices.dtype == torch.long

    kwargs = ShardedBlockComparator._build_cross_layer_state_kwargs(
        layer, None, hidden)
    assert set(kwargs) == {'block_residual'}
    assert kwargs['block_residual'].shape == (3, 0, 8)


def test_kimi_replay_mask_and_indexed_expert_detection():
    from accuracy_checker.model_structure import (
        build_replay_attention_mask,
        has_indexed_routed_experts,
    )

    hidden = torch.randn(1, 3, 8)
    layer = FakeKimiLayer()
    mask = build_replay_attention_mask(layer, hidden)
    assert mask.shape == (1, 1, 3, 3)
    assert mask[0, 0, 2, 0] == 0
    assert mask[0, 0, 0, 2] < 0
    assert has_indexed_routed_experts(layer)


def test_non_stateful_layer_does_not_receive_glm_state_kwarg():
    from accuracy_checker.model_structure import get_layer_state_kwarg

    class PlainLayer(nn.Module):
        def forward(self, hidden_states, position_ids=None):
            return hidden_states

    assert get_layer_state_kwarg(PlainLayer()) is None


def test_kimi_nested_compressed_tensors_detection(tmp_path):
    from accuracy_checker.model_loader import (
        is_compressed_tensors_model,
        is_quantized_model,
    )

    config = {
        'model_type': 'kimi_k3',
        'text_config': {
            'model_type': 'kimi_linear',
            'quantization_config': {
                'quant_method': 'compressed-tensors',
                'format': 'mxfp4-pack-quantized',
            },
        },
    }
    (tmp_path / 'config.json').write_text(json.dumps(config), encoding='utf-8')
    assert is_quantized_model(str(tmp_path))
    assert is_compressed_tensors_model(str(tmp_path))


def test_cache_entry_preserves_attnres_state():
    from accuracy_checker.subgraph_locate import _unpack_l1_cache_entry

    hidden = torch.randn(1, 3, 8)
    state = torch.randn(3, 2, 8)
    unpacked_hidden, unpacked_state = _unpack_l1_cache_entry({
        'hidden_states': hidden,
        'layer_state': state,
    })
    assert unpacked_hidden is hidden
    assert unpacked_state is state


def test_kimi_npu_auto_backend_avoids_triton_kda():
    """Ascend auto mode patches both remote-code KDA branches to eager torch."""
    from accuracy_checker.kimi_kda import torch_recurrent_kda
    from accuracy_checker.layer1_block_compare import ShardedBlockComparator

    def original_chunk(x):
        return x

    def original_recurrent(x):
        return x
    namespace = {
        "chunk_kda": original_chunk,
        "fused_recurrent_kda": original_recurrent,
    }
    exec("def forward(self, x):\n    return chunk_kda(x)", namespace)
    attention_type = type(
        "BackendKdaAttention",
        (FakeKdaAttention,),
        {"forward": namespace["forward"]},
    )
    layer = FakeKimiLayer()
    layer.self_attn = attention_type()
    comparator = object.__new__(ShardedBlockComparator)
    comparator.kimi_kda_backend = "torch"
    comparator.verbose = False

    assert comparator._resolve_kimi_kda_backend("auto", "npu:0") == "torch"
    assert comparator._resolve_kimi_kda_backend("auto", "cuda:0") is None
    assert comparator._configure_kimi_kda_backend(
        layer, torch.zeros(1, 2, 8)
    ) == "torch"
    forward_globals = layer.self_attn.forward.__func__.__globals__
    assert forward_globals["chunk_kda"] is torch_recurrent_kda
    assert forward_globals["fused_recurrent_kda"] is torch_recurrent_kda

    comparator.kimi_kda_backend = "chunk"
    comparator._configure_kimi_kda_backend(layer, torch.zeros(1, 2, 8))
    assert forward_globals["chunk_kda"] is original_chunk
    assert forward_globals["fused_recurrent_kda"] is original_recurrent


def test_torch_kda_recurrence_matches_one_dimensional_delta_rule():
    """The portable KDA path performs decay, correction, update, then readout."""
    from accuracy_checker.kimi_kda import torch_recurrent_kda

    q = torch.ones(1, 2, 1, 1)
    k = torch.ones_like(q)
    v = torch.tensor([[[[1.0]], [[2.0]]]])
    g = torch.zeros_like(q)
    beta = torch.ones(1, 2, 1)

    output, state = torch_recurrent_kda(
        q, k, v, g, beta, output_final_state=True
    )

    assert torch.allclose(output.flatten(), torch.tensor([1.0, 2.0]))
    assert torch.allclose(state.flatten(), torch.tensor([2.0]))
