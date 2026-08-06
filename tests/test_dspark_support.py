"""CPU-only tests for DSpark parameter and verifier-cache contracts."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from accuracy_checker.dspark import (
    is_dspark_config,
    is_dspark_checkpoint,
    load_dspark_sample,
    normalize_dspark_contract,
    validate_dspark_pair,
)


def _official_config(**overrides):
    config = {
        "architectures": ["Qwen3DSparkModel"],
        "block_size": 7,
        "num_anchors": 512,
        "num_hidden_layers": 5,
        "target_layer_ids": [1, 9, 17, 25, 33],
        "hidden_size": 8,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 151936,
        "mask_token_id": 151669,
        "markov_rank": 256,
        "markov_head_type": "vanilla",
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
    }
    config.update(overrides)
    return config


def test_official_deepspec_contract_is_normalized():
    contract = normalize_dspark_contract(_official_config())
    assert contract.flavor == "deepspec"
    assert contract.architecture == "Qwen3DSparkModel"
    assert contract.block_size == 7
    assert contract.num_anchors == 512
    assert contract.draft_layers == 5
    assert contract.target_layer_ids == (1, 9, 17, 25, 33)
    assert contract.target_hidden_width == 40
    assert contract.markov_rank == 256
    assert contract.enable_confidence_head


def test_speculators_contract_aliases_are_normalized():
    config = {
        "architectures": ["DSparkDraftModel"],
        "speculators_model_type": "dspark",
        "block_size": 8,
        "num_layers": 3,
        "aux_hidden_state_layer_ids": [2, 20, 39, 58, 75],
        "draft_vocab_size": 154880,
        "transformer_layer_config": {
            "hidden_size": 6144,
            "vocab_size": 154880,
            "mask_token_id": 154877,
        },
        "markov_rank": 256,
        "markov_head_type": "vanilla",
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "speculators_config": {
            "algorithm": "dspark",
            "verifier": {"name_or_path": "RedHatAI/GLM-5.2"},
        },
    }
    contract = normalize_dspark_contract(config)
    assert contract.flavor == "speculators"
    assert contract.draft_layers == 3
    assert contract.target_layer_ids == (2, 20, 39, 58, 75)
    assert contract.verifier_model == "RedHatAI/GLM-5.2"
    assert contract.target_hidden_width == 5 * 6144


def test_ref_quant_contract_mismatch_fails_before_loading():
    ref = normalize_dspark_contract(_official_config())
    quant = normalize_dspark_contract(_official_config(block_size=8))
    with pytest.raises(ValueError, match="block_size"):
        validate_dspark_pair(ref, quant)


def test_backbone_attention_parameters_are_aligned():
    ref = normalize_dspark_contract(_official_config())
    quant = normalize_dspark_contract(_official_config(num_attention_heads=4))
    with pytest.raises(ValueError, match=r"backbone\.num_attention_heads"):
        validate_dspark_pair(ref, quant)


def test_confidence_markov_dependency_is_validated():
    with pytest.raises(ValueError, match="markov_rank"):
        normalize_dspark_contract(_official_config(markov_rank=0))


def test_mask_token_and_target_layer_order_are_validated():
    with pytest.raises(ValueError, match="mask_token_id"):
        normalize_dspark_contract(_official_config(mask_token_id=None))
    with pytest.raises(ValueError, match="strictly increasing"):
        normalize_dspark_contract(_official_config(target_layer_ids=[1, 9, 9]))


def test_speculators_verifier_identity_is_part_of_pair_contract():
    config = {
        "architectures": ["DSparkDraftModel"],
        "speculators_model_type": "dspark",
        "block_size": 7,
        "num_layers": 2,
        "aux_hidden_state_layer_ids": [1, 9],
        "draft_vocab_size": 32,
        "mask_token_id": 31,
        "transformer_layer_config": {"hidden_size": 8, "vocab_size": 32},
        "speculators_config": {
            "algorithm": "dspark",
            "verifier": {"name_or_path": "org/verifier-a"},
        },
    }
    ref = normalize_dspark_contract(config)
    config["speculators_config"]["verifier"]["name_or_path"] = "org/verifier-b"
    quant = normalize_dspark_contract(config)
    with pytest.raises(ValueError, match="verifier_model"):
        validate_dspark_pair(ref, quant)


def test_dspark_sample_shape_and_aliases(tmp_path):
    contract = normalize_dspark_contract(_official_config())
    sample_path = tmp_path / "sample.pt"
    torch.save({
        "input_ids": torch.arange(6),
        "target_hidden_states": torch.randn(6, 5, 8),
        "loss_mask": torch.ones(6),
        "target_last_hidden_states": torch.randn(6, 8),
    }, sample_path)
    sample = load_dspark_sample(str(sample_path), contract)
    assert sample.input_ids.shape == (1, 6)
    assert sample.hidden_states.shape == (1, 6, 40)
    assert sample.verifier_last_hidden_states.shape == (1, 6, 8)
    assert sample.document_ids.shape == (1, 6)


def test_dspark_sample_hidden_width_mismatch_is_rejected(tmp_path):
    contract = normalize_dspark_contract(_official_config())
    sample_path = tmp_path / "bad.pt"
    torch.save({
        "input_ids": torch.arange(6).unsqueeze(0),
        "hidden_states": torch.randn(1, 6, 32),
        "loss_mask": torch.ones(1, 6),
    }, sample_path)
    with pytest.raises(ValueError, match="width mismatch"):
        load_dspark_sample(str(sample_path), contract)


def test_dspark_checkpoint_detection(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(_official_config()), encoding="utf-8"
    )
    assert is_dspark_checkpoint(str(tmp_path))


def test_k3_vllm_dspark_is_recognized_but_not_misloaded_as_speculators():
    config = {
        "architectures": ["K3DSparkModel"],
        "model_type": "k3_dspark",
        "hidden_size": 7168,
        "num_hidden_layers": 5,
        "num_target_layers": 5,
        "target_layer_ids": [2, 23, 47, 71, 89],
        "vocab_size": 163840,
        "mask_token_id": 163837,
        "_torchspec_version": "0.1.0",
    }
    assert is_dspark_config(config)
    with pytest.raises(NotImplementedError, match="vllm_k3"):
        normalize_dspark_contract(config)


def test_cli_exposes_dspark_sample_and_seed(monkeypatch):
    import run_accuracy_check

    monkeypatch.setattr(sys, "argv", [
        "run_accuracy_check.py",
        "--ref_model", "ref",
        "--quant_model", "quant",
        "--model_type", "dspark",
        "--dspark_sample", "sample.pt",
        "--dspark_seed", "17",
        "--dspark_max_anchors", "4",
    ])
    args = run_accuracy_check.parse_args()
    assert args.model_type == "dspark"
    assert args.dspark_sample == "sample.pt"
    assert args.dspark_seed == 17
    assert args.dspark_max_anchors == 4


def _dspark_cli_args(**overrides):
    values = {
        "quant_method": "dequantize",
        "compare_mode": "dual",
        "ref_devices": None,
        "quant_devices": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_dspark_cli_rejects_fake_quant_and_grouped_dual_early():
    import run_accuracy_check

    with pytest.raises(ValueError, match="dequantize"):
        run_accuracy_check._validate_standalone_dspark_args(
            _dspark_cli_args(quant_method="fake_quant"), "l1", True
        )
    with pytest.raises(ValueError, match="dual"):
        run_accuracy_check._validate_standalone_dspark_args(
            _dspark_cli_args(compare_mode="grouped_dual"), "l1", True
        )


def test_dspark_cli_rejects_multi_device_aliases():
    import run_accuracy_check

    with pytest.raises(ValueError, match="ref_device"):
        run_accuracy_check._validate_standalone_dspark_args(
            _dspark_cli_args(ref_devices="npu:0,1"), "l1", True
        )
