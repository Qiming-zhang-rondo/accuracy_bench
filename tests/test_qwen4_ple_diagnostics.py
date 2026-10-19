"""Small checkpoint diagnostics: no accelerator or full Qwen model required."""
import json

import pytest
import torch
from safetensors.torch import save_file

from accuracy_checker.qwen4_ple_diagnostics import (
    audit_lookup,
    compute_ngram_ids,
    inspect_checkpoint,
    tensor_comparison,
)


def _write_checkpoint(path, *, multimodal=False, multipliers=None, changed_row=None):
    path.mkdir()
    text_config = {
        "model_type": "qwen4_exp_text", "vocab_size": 32, "hidden_size": 8,
        "num_hidden_layers": 2, "ple_layer_ids": [2], "ple_embed_dim": 8,
        "ngram_size": 3, "heads_per_ngram": 1, "ngram_vocab_size_base": 5,
        "make_ngram_vocab_size_divisible_by": 1, "split_ngram_parts": 3,
        "eos_token_id": 2,
    }
    config = {"model_type": "qwen4_exp", "text_config": text_config} if multimodal else text_config
    (path / "config.json").write_text(json.dumps(config))
    prefix = "model.language_model.layers.1" if multimodal else "model.layers.1"
    stem = prefix + ".ple.ple_embedding"
    table = torch.arange(48, dtype=torch.float32).reshape(12, 4)
    if changed_row is not None:
        table[changed_row] += 3
    metadata = {
        stem + ".layer_multipliers": torch.tensor([3, 5, 7]) if multipliers is None else multipliers,
        stem + ".ngram_heads_vocab_sizes": torch.tensor([5, 7]),
        stem + ".ngram_heads_offsets": torch.tensor([0, 5]),
    }
    # Separate files and uneven shards expose both file/key and row-offset errors.
    weight_map = {}
    for i, (start, stop) in enumerate(((0, 2), (2, 7), (7, 12))):
        filename = f"model-{i + 1:05d}-of-00003.safetensors"
        tensors = {stem + f".ngram_embedding.shard_{i}.weight": table[start:stop].contiguous()}
        if i == 0:
            tensors.update(metadata)
        save_file(tensors, str(path / filename))
        weight_map.update({key: filename for key in tensors})
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return path, table


def _hash(ids, multipliers=None):
    return compute_ngram_ids(
        ids,
        multipliers=torch.tensor([3, 5, 7]) if multipliers is None else multipliers,
        head_vocab_sizes=torch.tensor([5, 7]),
        head_offsets=torch.tensor([0, 5]),
        ngram_size=3, heads_per_ngram=1, eos_token_id=2,
    )


def test_known_ngram_hash_and_eos_reset():
    # EOS still belongs to the ending segment; the following token starts anew.
    actual = _hash(torch.tensor([[1, 4, 2, 6, 7]]))
    expected = torch.tensor([[[4, 5], [4, 5], [3, 5], [4, 6], [1, 10]]])
    assert actual.dtype == torch.int64
    assert torch.equal(actual, expected)


def test_hash_short_sequences_multiple_batches_and_consecutive_eos():
    ids = torch.tensor([[2, 2, 6], [4, 2, 6]])
    rows = _hash(ids)
    assert rows.shape == (2, 3, 2)
    # Prefix from the preceding segment cannot leak across EOS.
    assert torch.equal(rows[0, 2], rows[1, 2])
    assert torch.equal(_hash(torch.tensor([[6]]))[0, 0], rows[0, 2])


def test_hash_large_integer_multipliers_do_not_round_through_float():
    ids = torch.tensor([[1, 31, 2, 29, 7], [2, 2, 3, 25, 5]])
    multipliers = torch.tensor([2**60 + 3, 2**58 + 5, 2**56 + 7])
    actual = _hash(ids, multipliers)
    expected = []
    for batch in ids.tolist():
        batch_rows = []
        history = []
        for token in batch:
            shifted = [token] + [history[-i] if len(history) >= i else 2 for i in (1, 2)]
            products = [(t * m) & ((1 << 64) - 1) for t, m in zip(shifted, multipliers.tolist())]
            mixed = products[0]
            heads = []
            for i, (size, offset) in enumerate(((5, 0), (7, 5)), start=1):
                mixed ^= products[i]
                signed = mixed if mixed < 2**63 else mixed - 2**64
                heads.append(signed % size + offset)
            batch_rows.append(heads)
            history.append(token)
            if token == 2:
                history = []
        expected.append(batch_rows)
    assert torch.equal(actual, torch.tensor(expected))


@pytest.mark.parametrize("multimodal", [False, True])
def test_uneven_shard_lookup_matches_independent_scalar_reads(tmp_path, multimodal):
    path, expected_table = _write_checkpoint(tmp_path / "checkpoint", multimodal=multimodal)
    checkpoint = inspect_checkpoint(str(path), 1, dtype=torch.float32)
    try:
        assert not [issue for issue in checkpoint.issues if issue["severity"] == "error"]
        # Include first/last row of every shard, duplicates and non-monotonic IDs.
        ids = torch.tensor([[11, 0, 2, 1], [6, 7, 11, 2]])
        assert torch.equal(checkpoint.table(ids), expected_table[ids])
        audit = audit_lookup(checkpoint, ids)
        assert audit["comparison"]["equal"]
        assert audit["comparison"]["max_abs_diff"] == 0
        assert audit["checked_rows"] == ids.numel()
        assert len(checkpoint.table.parts) == 3
        assert checkpoint.buffers["layer_multipliers"].dtype == torch.int64
    finally:
        checkpoint.close()


def test_same_checkpoint_comparison_is_exact(tmp_path):
    path, _ = _write_checkpoint(tmp_path / "checkpoint")
    ref = inspect_checkpoint(str(path), 1, dtype=torch.bfloat16)
    quant = inspect_checkpoint(str(path), 1, dtype=torch.bfloat16)
    try:
        ids = _hash(torch.tensor([[1, 4, 2, 6, 7]]))
        for key in ref.buffers:
            assert tensor_comparison(ref.buffers[key], quant.buffers[key])["equal"]
        comparison = tensor_comparison(ref.table(ids), quant.table(ids))
        assert comparison["equal"]
        assert comparison["rel_l2"] == 0
    finally:
        ref.close()
        quant.close()


def test_changed_hash_buffer_value_is_detectable(tmp_path):
    ref_path, _ = _write_checkpoint(tmp_path / "ref")
    quant_path, _ = _write_checkpoint(tmp_path / "quant", multipliers=torch.tensor([3, 5, 9]))
    ref = inspect_checkpoint(str(ref_path), 1)
    quant = inspect_checkpoint(str(quant_path), 1)
    try:
        comparison = tensor_comparison(ref.buffers["layer_multipliers"], quant.buffers["layer_multipliers"])
        assert not comparison["equal"]
        assert comparison["max_abs_diff"] == 2
        ids = torch.tensor([[1, 4, 2, 6, 7]])
        assert not torch.equal(_hash(ids, ref.buffers["layer_multipliers"]),
                               _hash(ids, quant.buffers["layer_multipliers"]))
    finally:
        ref.close()
        quant.close()


def test_float_hash_metadata_is_reported_not_silently_repaired(tmp_path):
    path, _ = _write_checkpoint(tmp_path / "checkpoint", multipliers=torch.tensor([3., 5., 7.]))
    checkpoint = inspect_checkpoint(str(path), 1)
    try:
        assert checkpoint.buffers["layer_multipliers"].dtype == torch.float32
        errors = [issue for issue in checkpoint.issues if issue["severity"] == "error"]
        assert errors
        assert "layer_multipliers" in json.dumps(errors)
    finally:
        checkpoint.close()


def test_changed_checkpoint_row_is_not_misreported_as_reader_error(tmp_path):
    ref_path, _ = _write_checkpoint(tmp_path / "ref")
    quant_path, _ = _write_checkpoint(tmp_path / "quant", changed_row=7)
    ref = inspect_checkpoint(str(ref_path), 1, dtype=torch.float32)
    quant = inspect_checkpoint(str(quant_path), 1, dtype=torch.float32)
    try:
        ids = torch.tensor([0, 1, 2, 6, 7, 11])
        assert audit_lookup(ref, ids)["comparison"]["equal"]
        assert audit_lookup(quant, ids)["comparison"]["equal"]
        comparison = tensor_comparison(ref.table(ids), quant.table(ids))
        assert not comparison["equal"]
        assert comparison["max_abs_diff"] == 3
        assert comparison["rel_l2"] > 0
    finally:
        ref.close()
        quant.close()


def test_audit_really_uses_an_independent_row_read(tmp_path, monkeypatch):
    path, _ = _write_checkpoint(tmp_path / "checkpoint")
    checkpoint = inspect_checkpoint(str(path), 1, dtype=torch.float32)
    try:
        original_forward = checkpoint.table.forward
        monkeypatch.setattr(checkpoint.table, "forward", lambda ids: original_forward(ids) + 1)
        audit = audit_lookup(checkpoint, torch.tensor([0, 1, 2, 6, 7, 11]))
        assert not audit["comparison"]["equal"]
        assert audit["comparison"]["max_abs_diff"] == 1
    finally:
        checkpoint.close()


def test_tensor_comparison_nonfinite_and_shape_mismatch_do_not_pass():
    mismatch = tensor_comparison(torch.zeros(2), torch.zeros(3))
    assert not mismatch["shape_match"]
    assert not mismatch["equal"]
    nonfinite = tensor_comparison(torch.ones(2), torch.tensor([1., float("nan")]))
    assert not nonfinite["finite"]
    assert not nonfinite["equal"]
    assert nonfinite["rel_l2"] is None


def test_tensor_comparison_zero_reference_has_no_infinite_json_metric():
    result = tensor_comparison(torch.zeros(2), torch.ones(2))
    assert not result["equal"]
    assert result["rel_l2"] is None
    json.dumps(result, allow_nan=False)


def test_pair_report_uses_full_hash_prefix_before_sampling(tmp_path):
    from accuracy_checker.qwen4_ple_diagnostics import diagnose_pair
    path, _ = _write_checkpoint(tmp_path / "checkpoint")
    ref = inspect_checkpoint(str(path), 1, dtype=torch.float32)
    quant = inspect_checkpoint(str(path), 1, dtype=torch.float32)
    try:
        ids = torch.tensor([[1, 4, 5, 8, 7, 9, 5, 4, 3, 2, 6, 7]])
        report = diagnose_pair(ref, quant, ids, max_positions=2)
        assert report["checks"]["ngram_ids"]["equal"]
        assert report["checked_hash_positions"] == ids.numel()
        assert report["checks"]["sampled_lookup_outputs"]["equal"]
        assert report["checks"]["same_row_table_values"]["equal"]
        assert report["checks"]["ref_runtime_hash_cpu"]["equal"]
        assert report["status"] == "SAMPLED_LOOKUP_ALIGNED"
        json.dumps(report, allow_nan=False)
    finally:
        ref.close()
        quant.close()


def test_pair_report_fails_fast_on_float_hash_metadata(tmp_path):
    from accuracy_checker.qwen4_ple_diagnostics import diagnose_pair
    ref_path, _ = _write_checkpoint(tmp_path / "ref")
    quant_path, _ = _write_checkpoint(tmp_path / "quant", multipliers=torch.tensor([3., 5., 7.]))
    ref = inspect_checkpoint(str(ref_path), 1)
    quant = inspect_checkpoint(str(quant_path), 1)
    try:
        report = diagnose_pair(ref, quant, torch.tensor([[1, 4, 2, 6, 7]]))
        assert report["status"] == "INVALID_METADATA"
        assert "sampled_lookup_outputs" not in report["checks"]
        assert any(finding["code"] == "INVALID_METADATA" for finding in report["findings"])
    finally:
        ref.close()
        quant.close()


@pytest.mark.parametrize("ngram_size,heads_per_ngram", [(2, 1), (3, 1), (3, 2), (4, 2)])
def test_hash_matches_official_ngram_forward_when_available(ngram_size, heads_per_ngram):
    hf = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    config_module = pytest.importorskip("transformers.models.qwen4_exp.configuration_qwen4_exp")
    embedding_dim = (ngram_size - 1) * heads_per_ngram * 4
    config = config_module.Qwen4ExpTextConfig(
        vocab_size=32, hidden_size=8, ple_embed_dim=embedding_dim, ple_layer_ids=[2],
        ngram_size=ngram_size, heads_per_ngram=heads_per_ngram, ngram_vocab_size_base=5,
        make_ngram_vocab_size_divisible_by=1, eos_token_id=2,
    )
    official = hf.Qwen4ExpTextNGramEmbedding(config, embedding_dim=embedding_dim, layer_idx=1)
    seen = []
    hook = official.ngram_embedding.register_forward_pre_hook(lambda _module, args: seen.append(args[0].clone()))
    ids = torch.tensor([[1, 4, 2, 6, 7], [2, 2, 3, 4, 5]])
    try:
        with torch.no_grad():
            official(ids, past_key_values=None)
    finally:
        hook.remove()
    actual = compute_ngram_ids(
        ids, multipliers=official.layer_multipliers,
        head_vocab_sizes=official.ngram_heads_vocab_sizes,
        head_offsets=official.ngram_heads_offsets,
        ngram_size=config.ngram_size, heads_per_ngram=config.heads_per_ngram,
        eos_token_id=config.eos_token_id,
    )
    assert torch.equal(actual, seen[0])


@pytest.mark.parametrize("value", [[1, True, 2], [1, 4.0, 2], {"input_ids": [1, False]}, {"prompt": [1, 2.5]}])
def test_cli_rejects_float_or_bool_recorded_ids(tmp_path, value):
    from diagnose_qwen4_ple import build_parser, load_input
    path = tmp_path / "ids.json"
    path.write_text(json.dumps(value))
    args = build_parser().parse_args([
        "--ref_model", "unused-ref", "--quant_model", "unused-quant",
        "--input_ids_json", str(path),
    ])
    with pytest.raises(ValueError, match="integer"):
        load_input(args)


def test_cli_l1_cache_ids_bypass_tokenizer(tmp_path, monkeypatch):
    from diagnose_qwen4_ple import build_parser, load_input
    from transformers import AutoTokenizer
    ids = torch.tensor([[1, 4, 2, 6, 7]])
    path = tmp_path / "layer_1.pt"
    torch.save({"input_ids": ids, "ref_hidden": torch.zeros(1, 5, 8)}, path)

    def unexpected_tokenizer(*args, **kwargs):
        pytest.fail("Recorded L1 IDs must not load or invoke a tokenizer")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", unexpected_tokenizer)
    args = build_parser().parse_args([
        "--ref_model", "unused-ref", "--quant_model", "unused-quant", "--l1_cache", str(path),
    ])
    actual, source = load_input(args)
    assert torch.equal(actual, ids)
    assert source["kind"] == "l1_cache"
    assert source["retokenized"] is False


def _cli_args(tmp_path, ref_path, quant_path):
    ids_path = tmp_path / "ids.json"
    ids_path.write_text(json.dumps({"input_ids": [1, 4, 2, 6, 7]}))
    output = tmp_path / "diagnostic.json"
    return ["--ref_model", str(ref_path), "--quant_model", str(quant_path),
            "--input_ids_json", str(ids_path), "--output", str(output)], output


def test_cli_same_checkpoint_cpu_replay_exits_zero_and_writes_json(tmp_path, capsys):
    from diagnose_qwen4_ple import main
    path, _ = _write_checkpoint(tmp_path / "checkpoint")
    argv, output = _cli_args(tmp_path, path, path)
    assert main(argv) == 0
    report = json.loads(output.read_text())
    assert report["status"] == "SAMPLED_LOOKUP_ALIGNED"
    assert report["input_source"]["retokenized"] is False
    assert report["checks"]["ref_runtime_hash_cpu"]["equal"]
    assert "Status: SAMPLED_LOOKUP_ALIGNED" in capsys.readouterr().out


def test_cli_invalid_hash_dtype_exits_two_and_writes_nan_safe_report(tmp_path):
    from diagnose_qwen4_ple import main
    ref_path, _ = _write_checkpoint(tmp_path / "ref")
    quant_path, _ = _write_checkpoint(tmp_path / "quant", multipliers=torch.tensor([float("nan"), 5., 7.]))
    argv, output = _cli_args(tmp_path, ref_path, quant_path)
    assert main(argv) == 2

    def reject_constant(value):
        pytest.fail(f"JSON must not contain a nonstandard numeric constant: {value}")

    report = json.loads(output.read_text(), parse_constant=reject_constant)
    assert report["status"] == "INVALID_METADATA"
    assert "sampled_lookup_outputs" not in report["checks"]
    assert not report["checks"]["buffer_layer_multipliers"]["finite"]


def test_cli_unavailable_runtime_hash_is_inconclusive_not_pass(tmp_path, monkeypatch, capsys):
    import accuracy_checker.qwen4_ple_diagnostics as diagnostics
    from diagnose_qwen4_ple import main
    path, _ = _write_checkpoint(tmp_path / "checkpoint")
    argv, output = _cli_args(tmp_path, path, path)

    def unavailable(*args, **kwargs):
        raise RuntimeError("Test runtime integer kernel unavailable")

    monkeypatch.setattr(diagnostics, "capture_runtime_ngram_ids", unavailable)
    assert main(argv) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "NEEDS_REVIEW"
    assert report["checks"]["ref_runtime_hash_cpu"]["status"] == "unavailable"
    assert any(finding["code"] == "RUNTIME_CHECK_UNAVAILABLE" for finding in report["findings"])
    assert "NOT CHECKED ref_runtime_hash_cpu" in capsys.readouterr().out
