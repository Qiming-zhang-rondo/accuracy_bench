"""CPU-only structural tests for Kimi K3 support."""

import json
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

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


def test_kimi_torch_import_shim_replaces_complete_fla_surface():
    """The portable path must satisfy remote imports before model creation."""
    import sys
    from accuracy_checker.kimi_fla_shim import (
        PortableFusedRMSNormGated,
        PortableShortConvolution,
        ensure_kimi_torch_import_path,
    )

    with patch.dict(sys.modules):
        assert ensure_kimi_torch_import_path(
            requested_backend="auto",
            devices=("npu:0", "npu:1"),
            model_type="kimi_k3",
            model_paths=(),
        )
        from fla.modules import FusedRMSNormGated, ShortConvolution
        from fla.ops.kda import chunk_kda, fused_recurrent_kda
        from fla.ops.utils.index import (
            prepare_cu_seqlens_from_mask,
            prepare_lens_from_mask,
        )
        from fla.utils import tensor_cache

        assert FusedRMSNormGated is PortableFusedRMSNormGated
        assert ShortConvolution is PortableShortConvolution
        assert chunk_kda is fused_recurrent_kda
        assert tensor_cache(lambda: None)() is None
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
        assert torch.equal(prepare_lens_from_mask(mask), torch.tensor([2, 1]))
        assert torch.equal(
            prepare_cu_seqlens_from_mask(mask), torch.tensor([0, 2, 3])
        )


def test_portable_fla_modules_preserve_kimi_weight_contracts():
    """Shim modules retain FLA parameter names and eager forward semantics."""
    from accuracy_checker.kimi_fla_shim import (
        PortableFusedRMSNormGated,
        PortableShortConvolution,
    )

    convolution = PortableShortConvolution(2, 2, activation=None)
    with torch.no_grad():
        convolution.weight.fill_(1.0)
    x = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])
    output, state = convolution(x, output_final_state=True)
    assert torch.equal(
        output,
        torch.tensor([[[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]]]),
    )
    assert tuple(state.shape) == (1, 2, 2)
    assert set(convolution.state_dict()) == {"weight"}

    norm = PortableFusedRMSNormGated(2, eps=0.0, activation="sigmoid")
    normalized = norm(torch.tensor([[[3.0, 4.0]]]), torch.zeros(1, 1, 2))
    expected = torch.tensor([[[0.6, 0.8]]]) * (2.0 ** 0.5) * 0.5
    assert torch.allclose(normalized, expected)
    assert set(norm.state_dict()) == {"weight"}


def test_kimi_streaming_skeleton_discards_routed_expert_parameters():
    """grouped_dual keeps routing metadata but no routed expert weights."""
    from accuracy_checker.model_loader import _finalize_kimi_streaming_skeleton

    class FakeStreamingMoe(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_experts=896)
            self.num_experts = 1
            self.experts_per_rank = 1
            self.experts = nn.ModuleList([nn.Linear(4, 4, bias=False)])

    model = nn.Module()
    model.moe = FakeStreamingMoe()
    collapsed = _finalize_kimi_streaming_skeleton(model, {896})

    assert collapsed == 1
    assert len(model.moe.experts) == 1
    assert isinstance(model.moe.experts[0], nn.Identity)
    assert model.moe.num_experts == 896
    assert model.moe.experts_per_rank == 896
    assert not list(model.moe.experts.parameters())


def test_kimi_streaming_skeleton_forces_eager_attention():
    from accuracy_checker.model_loader import _force_kimi_eager_attention

    text_config = SimpleNamespace(_attn_implementation="flash_attention_2")
    model = nn.Module()
    model.config = SimpleNamespace(
        _attn_implementation="flash_attention_2",
        text_config=text_config,
    )
    model.text = nn.Module()
    model.text.config = text_config
    model.text._use_flash_attention_2 = True

    _force_kimi_eager_attention(model)

    assert model.config._attn_implementation == "eager"
    assert text_config._attn_implementation == "eager"
    assert model.text._use_flash_attention_2 is False


def test_kimi_streaming_construction_collapses_only_expert_range():
    from accuracy_checker.model_loader import _kimi_streaming_construction

    fake_module = ModuleType("fake_kimi.modeling_kimi_linear")

    class KimiFakeModel:
        def post_init(self):
            return "original"

    class FakeDynamicModel:
        pass

    fake_module.KimiFakeModel = KimiFakeModel
    fake_module.KimiBlockSparseMLP = object
    config = SimpleNamespace(
        text_config=SimpleNamespace(num_experts=896),
    )
    original_post_init = KimiFakeModel.post_init

    with patch(
        "accuracy_checker.model_loader._load_kimi_modeling_modules",
        return_value=(FakeDynamicModel, [fake_module]),
    ):
        with _kimi_streaming_construction(config, "/fake/kimi", True) as state:
            counts, model_cls = state
            assert counts == {896}
            assert model_cls is FakeDynamicModel
            assert list(fake_module.range(896)) == [0]
            assert list(fake_module.range(3)) == [0, 1, 2]
            assert KimiFakeModel().post_init() is None

    assert "range" not in fake_module.__dict__
    assert KimiFakeModel.post_init is original_post_init


def test_grouped_dual_visits_only_router_selected_kimi_experts():
    """896-expert Kimi layers must not scan every inactive expert."""
    from types import MethodType

    from accuracy_checker.layer1_block_compare import ShardedBlockComparator

    comparator = object.__new__(ShardedBlockComparator)
    visited = []
    synchronized = []

    def _record_expert(self, expert_id, *args):
        visited.append(expert_id)
        return None

    comparator._forward_single_routed_expert = MethodType(
        _record_expert, comparator
    )
    comparator._sync_chunk_device = synchronized.append
    reader = SimpleNamespace(weight_map={
        "model.layers.0.block_sparse_moe.experts.1.w1.weight": "part-1",
    })
    hidden = torch.zeros(1, 2, 4)
    scores = torch.tensor([[[0.6, 0.4], [0.7, 0.3]]])
    indices = torch.tensor([[[1, 300], [700, 300]]])

    output = comparator._run_expert_chunks(
        mlp=None,
        experts_mod=None,
        hidden_states=hidden,
        devices=["npu:0", "npu:1"],
        chunk_size=256,
        layer_idx=0,
        topk_scores=scores,
        topk_indices=indices,
        num_experts_per_tok=2,
        num_experts=896,
        is_packed=False,
        is_module_list=True,
        use_streaming=True,
        sf_reader=reader,
        quant_desc_str=None,
        is_quant=False,
        is_ct=False,
        primary_device="npu:0",
    )

    assert visited == [1, 300, 700]
    assert synchronized == ["npu:0", "npu:1", "npu:0"]
    assert torch.equal(output, torch.zeros_like(hidden.view(-1, 4)).float())


def test_kimi_expert_prefix_index_is_cached_per_reader():
    from accuracy_checker.layer1_block_compare import ShardedBlockComparator

    reader = SimpleNamespace(weight_map={
        "model.layers.2.block_sparse_moe.experts.7.w1.weight": "part-1",
        "model.layers.3.block_sparse_moe.experts.9.w1.weight": "part-2",
    })

    first = ShardedBlockComparator._expert_prefix_index(reader)
    reader.weight_map.clear()
    second = ShardedBlockComparator._expert_prefix_index(reader)

    assert first is second
    assert first[(2, "block_sparse_moe")] == (
        "model.layers.2.block_sparse_moe.experts"
    )


def test_kimi_chunk_sizing_uses_real_expert_count():
    from accuracy_checker.layer1_block_compare import ShardedBlockComparator

    comparator = object.__new__(ShardedBlockComparator)
    comparator._model_config = {
        "text_config": {"num_experts": 896},
    }

    assert comparator._configured_num_experts() == 896
    assert comparator._auto_expert_chunk_size(896, 4) == 224
    assert comparator._auto_expert_chunk_size(896, 2) == 256
