"""CPU contract tests against the real Transformers Qwen4 implementation."""
import json
from types import SimpleNamespace
import pytest
import torch
from safetensors.torch import save_file

hf = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from accuracy_checker.qwen4_exp import expand_embedding, layer_kwargs, MMapNGramEmbedding
from accuracy_checker.model_loader import (create_model_skeleton, build_weight_index,
    ShardWeightReader, load_layer_weights_indexed, _initialize_rotary_modules)
from accuracy_checker.layer1_block_compare import ShardedBlockComparator
from accuracy_checker.model_structure import get_layer_state_kwarg


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(42)
    cfg = Qwen4ExpTextConfig(
        vocab_size=32, hidden_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8,
        moe_intermediate_size=8, shared_expert_intermediate_size=8,
        num_experts=4, num_experts_per_tok=2, hc_count=4, hc_lowrank=4,
        ple_layer_ids=[1], ple_embed_dim=16, ngram_size=3,
        heads_per_ngram=1, ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=8, split_ngram_parts=2,
        layer_types=['linear_attention', 'indexed_attention'],
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8,
        indexer_budget=4, indexer_compress_ratio=2, eos_token_id=2,
        rope_parameters={'rope_type':'default', 'rope_theta':10000., 'mrope_section':[1,1,2]},
    )
    model = hf.Qwen4ExpForCausalLM._from_config(cfg, attn_implementation='eager').eval()
    cfg.save_pretrained(tmp_path)
    state = {}
    for key, value in model.state_dict().items():
        # Exercise actual MM checkpoint naming and split PLE tables.
        key = key.replace('model.', 'model.language_model.', 1) if key.startswith('model.') else key
        if key.endswith('ngram_embedding.weight'):
            for i, chunk in enumerate(value.chunk(2)):
                state[key.replace('.weight', f'.shard_{i}.weight')] = chunk.contiguous()
        else:
            state[key] = value.contiguous()
    save_file(state, str(tmp_path / 'model.safetensors'))
    return tmp_path, model


def load_tiny(path):
    model = create_model_skeleton(str(path), dtype=torch.float32, verbose=False)
    model.to_empty(device='cpu')
    reader = ShardWeightReader(str(path), build_weight_index(str(path)))
    for layers in ([], [0, 1], [-1]):
        load_layer_weights_indexed(model, str(path), layers, 'cpu', torch.float32,
            reader.weight_map, reader, verbose=False)
    _initialize_rotary_modules(model)
    return model.eval(), reader


def test_lazy_ple_and_layer_replay_match_official(checkpoint):
    path, official = checkpoint
    model, _ = load_tiny(path)
    ids = torch.tensor([[3, 4, 7, 2, 8, 9, 5]])
    assert isinstance(model.model.layers[0].ple.ple_embedding.ngram_embedding, MMapNGramEmbedding)
    assert not any(n.endswith('ngram_embedding.weight') for n, _ in model.named_parameters())
    with torch.no_grad():
        expected = official(ids, use_cache=False).logits
        torch.testing.assert_close(model(ids, use_cache=False).logits, expected)
        hidden = expand_embedding(model, model.model.embed_tokens(ids))
        pos = torch.arange(ids.shape[1]).unsqueeze(0)
        pe = model.model.rotary_emb(hidden, pos)
        for layer in model.model.layers:
            assert get_layer_state_kwarg(layer) is None
            hidden = layer(hidden, **layer_kwargs(layer, hidden, pos, pe, ids))
        collapsed = ShardedBlockComparator._collapse_hc_streams(model, hidden)
        torch.testing.assert_close(model.lm_head(collapsed), expected)


def test_decode_cache_matches_full_forward(checkpoint):
    path, _ = checkpoint
    model, _ = load_tiny(path)
    ids = torch.tensor([[3, 4, 7, 8, 9]])
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits[:, -1]
        cached = model(ids[:, :-1], use_cache=True)
        actual = model(ids[:, -1:], past_key_values=cached.past_key_values, use_cache=True).logits[:, -1]
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_ple_requires_original_ids(checkpoint):
    _, model = checkpoint
    layer = model.model.layers[0]
    with pytest.raises(ValueError, match='original input_ids'):
        layer_kwargs(layer, torch.zeros(1, 3, 64), None, None, None)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_boundary_streaming_matches_official(checkpoint, dtype):
    from accuracy_checker.inference_check import (_hf_create_skeleton, distribute_model,
        _hf_load_weights)
    path, official = checkpoint
    model, is_quant, streaming = _hf_create_skeleton(str(path), dtype, False)
    assert streaming
    distribute_model(model, ['cpu'])
    _hf_load_weights(model, str(path), dtype, ['cpu'], is_quant, False, streaming, False)
    _initialize_rotary_modules(model)
    official.to(dtype)
    ids = torch.tensor([[3, 4, 7, 8, 9]])
    with torch.no_grad():
        tolerance = dict(atol=1e-6, rtol=1e-4) if dtype == torch.float32 else dict(atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(model(ids, use_cache=False).logits,
                                   official(ids, use_cache=False).logits, **tolerance)
        torch.testing.assert_close(model.generate(ids, max_new_tokens=2, do_sample=False),
                                   official.generate(ids, max_new_tokens=2, do_sample=False))


@pytest.mark.parametrize('mode', ['grouped_dual', 'dual'])
def test_grouped_l1_and_l2(checkpoint, mode):
    from accuracy_checker.replay_provider import ReplayProvider
    path, official = checkpoint
    ids = torch.tensor([[3, 4, 7, 8, 9]])
    tokenizer = SimpleNamespace(decode=lambda ids: str(ids[0]))
    comparator = ShardedBlockComparator(str(path), str(path), tokenizer,
        ref_device='cpu', quant_device='cpu', dtype=torch.float32,
        use_fake_quant=False, compare_mode=mode, verbose=False)
    report = comparator.compare_ids(ids, layers_per_shard=1)
    assert len(report.results) == 3
    assert report.topk_result is not None
    assert report.topk_result.top1_match
    with torch.no_grad():
        expected_topk = official(ids, use_cache=False).logits[0, -1].topk(10).indices.tolist()
    assert report.topk_result.ref_topk_ids == expected_topk
    assert not report.logits_error
    provider = ReplayProvider(str(path), str(path), dtype=torch.float32, verbose=False)
    h = expand_embedding(official, official.model.embed_tokens(ids)).detach()
    handle = provider.get_layer_handle(0, device='cpu', quant_device='cpu',
        ref_hidden_override=h, quant_hidden_override=h, input_ids_override=ids)
    with torch.no_grad():
        pos = torch.arange(ids.shape[1]).unsqueeze(0)
        pe = official.model.rotary_emb(h, pos)
        expected = official.model.layers[0](h, **layer_kwargs(official.model.layers[0], h, pos, pe, ids))
        actual = handle.ref_layer(h, **handle.ref_layer_kwargs)
    torch.testing.assert_close(actual, expected)
    from accuracy_checker.subgraph_locate import diagnose_layer
    handle.quant_hidden = h * 1.1
    diagnosis = diagnose_layer(handle, model_type='auto')
    assert diagnosis['model_type'] == 'qwen4_exp'
    assert 'ple' in diagnosis['subgraphs']


def test_expert_count_above_300():
    from accuracy_checker.model_loader import _count_experts_in_map
    keys = {f'model.language_model.layers.0.mlp.experts.{i}.gate_proj.weight': 'x' for i in range(512)}
    reader = ShardWeightReader('/unused', keys)
    assert _count_experts_in_map(reader, 'model.layers.0.mlp.experts.{i}.gate_proj.weight') == 512


def test_qsa_does_not_claim_glm_tp(checkpoint):
    from accuracy_checker.qwen4_exp import check_parallel_mode
    path, _ = checkpoint
    check_parallel_mode(str(path), 'pp')
    with pytest.raises(ValueError, match='not compatible'):
        check_parallel_mode(str(path), 'tp')


def test_missing_weight_fails_before_forward(checkpoint):
    from safetensors.torch import load_file
    path, _ = checkpoint
    tensors = load_file(str(path / 'model.safetensors'))
    del tensors['model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight']
    save_file(tensors, str(path / 'model.safetensors'))
    with pytest.raises(ValueError, match='required parameter'):
        load_tiny(path)


def test_quantized_alias_and_split_experts(checkpoint):
    from safetensors.torch import load_file
    from accuracy_checker.inference_check import _hf_create_skeleton, distribute_model, _hf_load_weights, _hf_replay_logits
    path, expected_model = checkpoint
    state = load_file(str(path / 'model.safetensors'))
    quant_key = 'model.language_model.layers.1.self_attn.indexer.index_qk_proj.weight'
    w = state[quant_key]
    scale = w.abs().amax(dim=1, keepdim=True) / 127
    state[quant_key] = (w / scale).round().clamp(-127, 127).to(torch.int8)
    state[quant_key.replace('.weight', '.weight_scale')] = scale
    state[quant_key.replace('.weight', '.weight_offset')] = torch.zeros_like(scale)
    expected_model.model.layers[1].self_attn.indexer.index_qk_proj.weight.data = state[quant_key].float() * scale
    for key in list(state):
        if key.endswith('experts.gate_up_proj'):
            for i, expert in enumerate(state.pop(key)):
                gate, up = expert.chunk(2)
                prefix = key.removesuffix('gate_up_proj') + str(i)
                state[prefix + '.gate_proj.weight'] = gate.contiguous()
                state[prefix + '.up_proj.weight'] = up.contiguous()
        elif key.endswith('experts.down_proj'):
            for i, expert in enumerate(state.pop(key)):
                prefix = key.removesuffix('down_proj') + str(i)
                state[prefix + '.down_proj.weight'] = expert.contiguous()
    desc = {key: ('W8A8_DYNAMIC' if key == quant_key else 'FLOAT') for key in state}
    (path / 'quant_model_description.json').write_text(json.dumps(desc))
    save_file(state, str(path / 'model.safetensors'))
    model, is_quant, streaming = _hf_create_skeleton(str(path), torch.float32, False)
    assert is_quant
    distribute_model(model, ['cpu'])
    _hf_load_weights(model, str(path), torch.float32, ['cpu'], is_quant, False, streaming, False)
    _initialize_rotary_modules(model)
    ids = torch.tensor([[3, 4, 7, 8, 9]])
    capture = _hf_replay_logits(model, ids, [0, 2, 4], 'cpu', verbose=False)
    with torch.no_grad():
        expected = expected_model(ids, use_cache=False).logits[0, [0, 2, 4]]
    torch.testing.assert_close(capture.logits, expected, atol=1e-6, rtol=1e-4)
    assert capture.token_positions == [0, 2, 4]
    assert torch.equal(capture.input_ids, ids)


def test_split_ple_bounds_and_sparse_rows(checkpoint):
    path, official = checkpoint
    model, _ = load_tiny(path)
    table = model.model.layers[0].ple.ple_embedding.ngram_embedding
    original = official.model.layers[0].ple.ple_embedding.ngram_embedding
    ids = torch.tensor([[table.shape[0]-1, 0, 0, table.shape[0]//2]])
    torch.testing.assert_close(table(ids), original(ids))
    table.to_empty(device='meta')
    assert table.weight.device.type == 'cpu'
    with pytest.raises(IndexError):
        table(torch.tensor([table.shape[0]]))


def test_outer_multimodal_config_loads_text_only(checkpoint):
    path, official = checkpoint
    cfg = json.loads((path / 'config.json').read_text())
    (path / 'config.json').write_text(json.dumps(dict(
        model_type='qwen4_exp', text_config=cfg,
        architectures=['Qwen4ExpForConditionalGeneration'])))
    model, _ = load_tiny(path)
    assert model.config.model_type == 'qwen4_exp_text'
    assert not hasattr(model.model, 'visual')
    ids = torch.tensor([[3, 4, 5]])
    with torch.no_grad():
        torch.testing.assert_close(model(ids, use_cache=False).logits, official(ids, use_cache=False).logits)


def test_missing_ple_part_fails_at_skeleton_creation(checkpoint):
    from safetensors.torch import load_file
    path, _ = checkpoint
    tensors = load_file(str(path / 'model.safetensors'))
    key = next(k for k in tensors if 'ngram_embedding.shard_1.weight' in k)
    del tensors[key]
    save_file(tensors, str(path / 'model.safetensors'))
    with pytest.raises(ValueError, match='incomplete PLE'):
        create_model_skeleton(str(path), verbose=False)
