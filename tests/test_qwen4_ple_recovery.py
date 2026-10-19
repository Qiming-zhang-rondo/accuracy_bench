"""PLE hash metadata recovery must preserve integer semantics on every load.

Only the three deterministic hash buffers are recoverable.  Never repair
arbitrary weights or a checkpoint whose metadata disagrees with its config.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

hf = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
from tests.test_qwen4_exp import checkpoint, load_tiny
from accuracy_checker.model_loader import load_layer_weights_indexed, unload_layers_to_meta
from accuracy_checker.qwen4_exp import expand_embedding, layer_kwargs


BUFFER_NAMES = ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")
PLE_PREFIX = "model.language_model.layers.0.ple.ple_embedding."


def _official_buffers(model):
    ple = model.model.layers[0].ple.ple_embedding
    return {name: getattr(ple, name).detach().cpu().clone() for name in BUFFER_NAMES}


def _export_float_buffers(path, dtype=torch.bfloat16):
    """Simulate an exporter casting floating and integer state indiscriminately."""
    state = load_file(str(path / "model.safetensors"))
    for name in BUFFER_NAMES:
        state[PLE_PREFIX + name] = state[PLE_PREFIX + name].to(dtype)
    save_file(state, str(path / "model.safetensors"))


def test_rebuild_matches_hf_and_uses_ple_ordinal_not_decoder_number(checkpoint):
    from accuracy_checker.qwen4_exp import rebuild_ple_hash_buffers
    _, model = checkpoint
    config = deepcopy(model.config)
    config.ple_layer_ids = [2, 7]
    for decoder_layer, ple_ordinal in ((1, 0), (6, 1)):
        official = hf.Qwen4ExpTextNGramEmbedding(config, config.ple_embed_dim, decoder_layer, ple_ordinal)
        actual = rebuild_ple_hash_buffers(config, decoder_layer)
        assert set(actual) == set(BUFFER_NAMES)
        for name in BUFFER_NAMES:
            assert actual[name].dtype == torch.int64
            assert actual[name].device.type == "cpu"
            assert torch.equal(actual[name], getattr(official, name))


def test_rebuild_does_not_allocate_embedding_table(checkpoint, monkeypatch):
    from accuracy_checker.qwen4_exp import rebuild_ple_hash_buffers
    _, model = checkpoint
    expected = _official_buffers(model)

    original_embedding = torch.nn.Embedding

    def meta_only(*args, **kwargs):
        table = original_embedding(*args, **kwargs)
        assert table.weight.is_meta, "Metadata recovery must not allocate PLE table storage"
        return table

    monkeypatch.setattr(torch.nn, "Embedding", meta_only)
    actual = rebuild_ple_hash_buffers(model.config, 0)
    for name in BUFFER_NAMES:
        assert torch.equal(actual[name], expected[name])


@pytest.mark.parametrize("field", ["seed", "eos_token_id", "ple_layer_ids", "heads_per_ngram"])
def test_paired_ple_config_mismatch_is_rejected(checkpoint, field):
    from accuracy_checker.qwen4_exp import validate_ple_pair_configs
    _, model = checkpoint
    quant = deepcopy(model.config)
    if field == "ple_layer_ids":
        quant.ple_layer_ids = [2]
    else:
        setattr(quant, field, getattr(quant, field) + 1)
    with pytest.raises(ValueError, match=field):
        validate_ple_pair_configs(model.config, quant)


def test_pair_accepts_raw_outer_config_and_installed_defaults(checkpoint):
    from accuracy_checker.qwen4_exp import validate_ple_pair_configs
    _, model = checkpoint
    raw = model.config.to_dict()
    assert model.config.seed == 1234
    raw.pop("seed")
    outer = {"model_type": "qwen4_exp", "text_config": raw}
    validate_ple_pair_configs(outer, model.config)
    # HF accepts a list, while PLE uses the first EOS entry.
    quant = deepcopy(model.config)
    quant.eos_token_id = [model.config.eos_token_id]
    validate_ple_pair_configs(outer, quant)


@pytest.mark.parametrize("dtype", [torch.int64, torch.bfloat16, torch.float32, torch.float64])
def test_resolve_only_recovers_exact_export_cast(checkpoint, dtype):
    from accuracy_checker.qwen4_exp import resolve_ple_hash_buffers
    _, model = checkpoint
    expected = _official_buffers(model)
    raw = {name: value.to(dtype) for name, value in expected.items()}
    raw_before = {name: value.clone() for name, value in raw.items()}
    effective, recovered = resolve_ple_hash_buffers(model.config, 0, raw)
    assert set(recovered) == (set() if dtype == torch.int64 else set(BUFFER_NAMES))
    for name in BUFFER_NAMES:
        assert effective[name].dtype == torch.int64
        assert torch.equal(effective[name], expected[name])
        assert raw[name].dtype == dtype
        assert torch.equal(raw[name], raw_before[name])


def test_finite_fp16_export_is_recoverable(checkpoint):
    from accuracy_checker.qwen4_exp import rebuild_ple_hash_buffers, resolve_ple_hash_buffers
    _, model = checkpoint
    config = deepcopy(model.config)
    # Keep deterministic multipliers small enough for FP16. Normal model
    # multipliers overflow FP16 and must be rejected, tested separately below.
    config.vocab_size = 2 ** 60
    expected = rebuild_ple_hash_buffers(config, 0)
    raw = {name: value.half() for name, value in expected.items()}
    assert all(torch.isfinite(value).all() for value in raw.values())
    effective, recovered = resolve_ple_hash_buffers(config, 0, raw)
    assert set(recovered) == set(BUFFER_NAMES)
    assert all(torch.equal(effective[name], expected[name]) for name in BUFFER_NAMES)


@pytest.mark.parametrize("mutation", ["missing", "shape", "nan", "inf", "float_mismatch", "int64_mismatch", "fp16_overflow"])
def test_conflicting_metadata_is_never_silently_repaired(checkpoint, mutation):
    from accuracy_checker.qwen4_exp import resolve_ple_hash_buffers
    _, model = checkpoint
    raw = _official_buffers(model)
    name = "layer_multipliers"
    if mutation == "missing":
        del raw[name]
    elif mutation == "shape":
        raw[name] = raw[name].unsqueeze(0)
    elif mutation in {"nan", "inf"}:
        raw[name] = raw[name].float()
        raw[name][0] = float(mutation)
    elif mutation == "float_mismatch":
        raw[name] = raw[name].bfloat16()
        raw[name][0] = torch.nextafter(raw[name][0], torch.tensor(float("inf"), dtype=torch.bfloat16))
    elif mutation == "int64_mismatch":
        # A float comparison would miss this one-unit change in a large int64.
        raw[name][0] += 1
    elif mutation == "fp16_overflow":
        raw[name] = raw[name].half()
    with pytest.raises(ValueError):
        resolve_ple_hash_buffers(model.config, 0, raw)


def test_loader_recovery_survives_repeated_meta_unload_and_matches_logits(checkpoint):
    path, official = checkpoint
    expected_buffers = _official_buffers(official)
    _export_float_buffers(path)
    model, reader = load_tiny(path)
    ids = torch.tensor([[3, 4, 7, 2, 8, 9, 5]])
    with torch.no_grad():
        expected = official(ids, use_cache=False).logits
        for _ in range(3):
            ple = model.model.layers[0].ple.ple_embedding
            for name in BUFFER_NAMES:
                assert getattr(ple, name).dtype == torch.int64
                assert torch.equal(getattr(ple, name).cpu(), expected_buffers[name])
            torch.testing.assert_close(model(ids, use_cache=False).logits, expected)
            unload_layers_to_meta(model, [0], cleanup=False)
            load_layer_weights_indexed(model, str(path), [0], "cpu", torch.float32,
                                       reader.weight_map, reader, verbose=False)
        # Verify the fix also preserves the official cached/decode hash path.
        cached = model(ids[:, :-1], use_cache=True)
        actual = model(ids[:, -1:], past_key_values=cached.past_key_values, use_cache=True).logits[:, -1]
        torch.testing.assert_close(actual, expected[:, -1], atol=1e-5, rtol=1e-4)
    stored = load_file(str(path / "model.safetensors"))
    assert all(stored[PLE_PREFIX + name].dtype == torch.bfloat16 for name in BUFFER_NAMES)


def test_loader_rejects_unknown_metadata_instead_of_casting_to_long(checkpoint):
    path, _ = checkpoint
    _export_float_buffers(path)
    state = load_file(str(path / "model.safetensors"))
    state[PLE_PREFIX + "ngram_heads_offsets"][-1] += 128
    save_file(state, str(path / "model.safetensors"))
    with pytest.raises(ValueError):
        load_tiny(path)


def test_l2_replay_uses_recovered_hashes(checkpoint):
    from accuracy_checker.replay_provider import ReplayProvider
    path, official = checkpoint
    _export_float_buffers(path)
    ids = torch.tensor([[3, 4, 7, 2, 8, 9]])
    hidden = expand_embedding(official, official.model.embed_tokens(ids)).detach()
    provider = ReplayProvider(str(path), str(path), dtype=torch.float32, verbose=False)
    handle = provider.get_layer_handle(0, device="cpu", quant_device="cpu",
                                      ref_hidden_override=hidden, quant_hidden_override=hidden,
                                      input_ids_override=ids)
    with torch.no_grad():
        pos = torch.arange(ids.shape[1]).unsqueeze(0)
        pe = official.model.rotary_emb(hidden, pos)
        expected = official.model.layers[0](hidden, **layer_kwargs(official.model.layers[0], hidden, pos, pe, ids))
        torch.testing.assert_close(handle.ref_layer(hidden, **handle.ref_layer_kwargs), expected)
        torch.testing.assert_close(handle.quant_layer(hidden, **handle.quant_layer_kwargs), expected)


def test_boundary_replay_recovers_bfloat16_exported_hashes(checkpoint):
    from accuracy_checker.inference_check import _hf_create_skeleton, distribute_model, _hf_load_weights
    from accuracy_checker.model_loader import _initialize_rotary_modules
    path, official = checkpoint
    _export_float_buffers(path)
    model, is_quant, streaming = _hf_create_skeleton(str(path), torch.float32, False)
    distribute_model(model, ["cpu"])
    _hf_load_weights(model, str(path), torch.float32, ["cpu"], is_quant, False, streaming, False)
    _initialize_rotary_modules(model)
    ids = torch.tensor([[3, 4, 7, 2, 8, 9]])
    with torch.no_grad():
        torch.testing.assert_close(model(ids, use_cache=False).logits, official(ids, use_cache=False).logits,
                                   atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("config", [
    {"model_type": "qwen4_exp_text"},
    {"model_type": "qwen4_exp", "text_config": {"model_type": "qwen4_exp_text"}},
])
def test_old_qwen_hidden_cache_and_latest_manifest_are_ignored_not_deleted(tmp_path, monkeypatch, config):
    from accuracy_checker import cache
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(config))
    model_path = str(model_path)
    monkeypatch.setattr(cache, "_cache_dir_override", str(tmp_path / "cache"))
    prompt, quant_mode = "test prompt", "dequantize"
    ids = torch.tensor([[3, 4, 7]])
    old_hidden, new_hidden = torch.zeros(1, 3, 16), torch.ones(1, 3, 16)
    with monkeypatch.context() as old_runtime:
        # Reproduce the pre-fix path-only identity through the same serializer.
        old_runtime.setattr(cache, "_model_cache_identity", lambda path: str(path))
        old_cache = Path(cache.save_cache(model_path, prompt, 3, 1, "quant", quant_mode,
                                         old_hidden, input_ids=ids))
        old_manifest = Path(cache.save_latest_l1_cache_manifest(model_path, model_path, prompt,
                                                               quant_mode, [0, 1]))
    old_cache_bytes, old_manifest_bytes = old_cache.read_bytes(), old_manifest.read_bytes()
    assert cache.load_cache(model_path, prompt, 3, 1, "quant", quant_mode, "cpu") is None
    assert cache.load_latest_l1_cache_manifest(model_path, model_path, prompt, quant_mode) is None
    new_cache = Path(cache.save_cache(model_path, prompt, 3, 1, "quant", quant_mode,
                                     new_hidden, input_ids=ids))
    new_manifest = Path(cache.save_latest_l1_cache_manifest(model_path, model_path, prompt,
                                                           quant_mode, [1]))
    assert new_cache != old_cache
    assert new_manifest != old_manifest
    restored = cache.load_cache(model_path, prompt, 3, 1, "quant", quant_mode, "cpu")
    assert torch.equal(restored["hidden_states"], new_hidden)
    assert torch.equal(restored["input_ids"], ids)
    assert cache.load_latest_l1_cache_manifest(model_path, model_path, prompt, quant_mode) == [1]
    assert old_cache.read_bytes() == old_cache_bytes
    assert old_manifest.read_bytes() == old_manifest_bytes


@pytest.mark.parametrize("model_type", ["glm_moe_dsa", "qwen3_5_moe", "deepseek_v4"])
def test_other_architectures_keep_existing_cache_identity(tmp_path, monkeypatch, model_type):
    from accuracy_checker import cache
    model_path = tmp_path / model_type
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    model_path = str(model_path)
    monkeypatch.setattr(cache, "_cache_dir_override", str(tmp_path / "cache"))
    assert cache.model_hash(model_path) == hashlib.sha256(model_path.encode()).hexdigest()[:8]
    prompt, quant_mode = "test prompt", "dequantize"
    legacy_manifest_digest = hashlib.sha256("\0".join(
        (model_path, model_path, prompt, quant_mode)).encode()).hexdigest()[:20]
    actual_manifest = cache._latest_manifest_path(model_path, model_path, prompt, quant_mode)
    assert Path(actual_manifest).name == f"latest_l1_{legacy_manifest_digest}.json"
