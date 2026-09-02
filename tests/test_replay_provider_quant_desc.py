"""L2 must keep reference and target quantization descriptions separate."""

import accuracy_checker.replay_provider as replay_provider_module
from accuracy_checker.replay_provider import ReplayProvider


def test_layer_loader_forwards_the_explicit_side_descriptor(monkeypatch):
    provider = object.__new__(ReplayProvider)
    provider.dtype = "bf16-sentinel"
    captured = []

    def fake_load(*args, **kwargs):
        captured.append(kwargs["quant_desc"])

    monkeypatch.setattr(
        replay_provider_module, "load_layer_weights_indexed", fake_load
    )
    monkeypatch.setattr(
        replay_provider_module, "move_layers_to_device", lambda *a, **k: None
    )

    ref_fp8_desc = {"__acc_native_fp8__": "FP8_E4M3"}
    provider._load_layer_weights(
        object(), "/models/ref", [0], "cpu", {}, object(), True, False,
        ref_fp8_desc,
    )

    assert captured == [ref_fp8_desc]
