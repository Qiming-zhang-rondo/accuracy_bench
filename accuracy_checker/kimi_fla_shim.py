"""Portable import-time FLA shim for the Kimi K3 accuracy path.

Kimi K3's Hugging Face remote code imports a small FLA surface while the
module is being imported.  Selecting an eager KDA implementation after model
creation is therefore too late when ``fla-core`` is absent, and still leaves
the short convolution and gated RMSNorm on Triton when FLA is present.

This module installs only the names used by Kimi K3 and implements them with
ordinary PyTorch operations.  It is intentionally activated only for a Kimi
checkpoint whose resolved KDA backend is ``torch``.
"""

from __future__ import annotations

import importlib.machinery
import json
import os
import sys
import types
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kimi_kda import torch_recurrent_kda


_SHIM_MARKER = "__acc_bench_kimi_fla_shim__"


def _activation(x: torch.Tensor, name: Optional[str]) -> torch.Tensor:
    if name is None:
        return x
    if name in {"silu", "swish"}:
        return F.silu(x)
    if name == "sigmoid":
        return torch.sigmoid(x)
    raise ValueError(f"Unsupported portable Kimi activation: {name}")


class PortableShortConvolution(nn.Conv1d):
    """FLA ``ShortConvolution`` compatible causal depthwise convolution."""

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: Optional[str] = "silu",
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
            device=device,
            dtype=dtype,
        )
        if activation not in {None, "silu", "swish"}:
            raise ValueError(f"Unsupported ShortConvolution activation: {activation}")
        self.hidden_size = hidden_size
        self.activation = activation
        self.backend = "torch"

    def _forward_sequence(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = x.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"ShortConvolution hidden size {hidden_size} != {self.hidden_size}"
            )
        width = self.kernel_size[0]
        x_channels = x.transpose(1, 2)
        if cache is not None:
            expected = (batch_size, hidden_size, width)
            if tuple(cache.shape) != expected:
                raise ValueError(
                    f"ShortConvolution cache shape {tuple(cache.shape)} != {expected}"
                )
            prefix = cache[..., -(width - 1):] if width > 1 else cache[..., :0]
            history = torch.cat((prefix, x_channels), dim=-1)
            state_history = torch.cat((cache, x_channels), dim=-1)
        else:
            history = F.pad(x_channels, (width - 1, 0))
            state_history = F.pad(x_channels, (width, 0))

        if seq_len:
            output = F.conv1d(
                history,
                self.weight,
                self.bias,
                groups=self.hidden_size,
            ).transpose(1, 2)
            output = _activation(output, self.activation)
        else:
            output = x.new_empty(batch_size, 0, hidden_size)
        return output, state_history[..., -width:].contiguous()

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x.dim() != 3:
            raise ValueError(f"ShortConvolution expects [B,T,D], got {tuple(x.shape)}")
        if mask is not None:
            if cu_seqlens is not None:
                raise ValueError("mask and cu_seqlens cannot be used together")
            x = x * mask.unsqueeze(-1).to(x.dtype)

        if cu_seqlens is None:
            output, final_state = self._forward_sequence(x, cache)
        else:
            if x.shape[0] != 1:
                raise ValueError("cu_seqlens ShortConvolution input must have batch size 1")
            boundaries = cu_seqlens.detach().cpu().tolist()
            sequence_count = len(boundaries) - 1
            if cache is not None and cache.shape[0] != sequence_count:
                raise ValueError(
                    f"ShortConvolution cache batch {cache.shape[0]} != {sequence_count}"
                )
            outputs = []
            final_states = []
            for sequence_idx, (start, end) in enumerate(
                zip(boundaries[:-1], boundaries[1:])
            ):
                sequence_cache = (
                    cache[sequence_idx:sequence_idx + 1]
                    if cache is not None else None
                )
                sequence_output, sequence_state = self._forward_sequence(
                    x[:, start:end], sequence_cache
                )
                outputs.append(sequence_output)
                final_states.append(sequence_state)
            output = torch.cat(outputs, dim=1) if outputs else x[:, :0]
            final_state = torch.cat(final_states, dim=0)

        if residual is not None:
            if residual.shape != output.shape:
                raise ValueError(
                    f"ShortConvolution residual shape {tuple(residual.shape)} != "
                    f"{tuple(output.shape)}"
                )
            output = output + residual
        return output, final_state if output_final_state else None

    @property
    def state_size(self) -> int:
        return self.hidden_size * self.kernel_size[0]


class PortableFusedRMSNormGated(nn.Module):
    """Inference-compatible PyTorch form of FLA ``FusedRMSNormGated``."""

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        activation: str = "swish",
        device=None,
        dtype=None,
    ):
        super().__init__()
        if activation not in {"swish", "silu", "sigmoid"}:
            raise ValueError(f"Unsupported gated RMSNorm activation: {activation}")
        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        self.activation = activation
        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(hidden_size, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ):
        if x.shape != g.shape or x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"gated RMSNorm expects matching [...,{self.hidden_size}] tensors; "
                f"got x={tuple(x.shape)}, g={tuple(g.shape)}"
            )
        residual_out = x if residual is None else x + residual
        source = residual_out.float()
        normalized = source * torch.rsqrt(
            source.square().mean(dim=-1, keepdim=True) + self.eps
        )
        if self.weight is not None:
            normalized = normalized * self.weight.float()
        output = normalized * _activation(g.float(), self.activation)
        output = output.to(x.dtype)
        if not prenorm:
            return output
        residual_dtype = torch.float32 if residual_in_fp32 else x.dtype
        return output, residual_out.to(residual_dtype)


def _tensor_cache(function=None, **kwargs):
    """Correctness path: preserve the decorator contract without caching."""
    if function is None:
        return lambda wrapped: wrapped
    return function


@_tensor_cache
def _prepare_lens_from_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask.sum(dim=-1, dtype=torch.int32)


@_tensor_cache
def _prepare_cu_seqlens_from_mask(
    mask: torch.Tensor,
    dtype: Optional[torch.dtype] = torch.int32,
) -> torch.Tensor:
    lengths = _prepare_lens_from_mask(mask)
    return F.pad(lengths.cumsum(dim=0, dtype=dtype), (1, 0))


def _new_module(name: str, package: bool = False) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(
        name=name,
        loader=None,
        is_package=package,
    )
    if package:
        module.__path__ = []
    setattr(module, _SHIM_MARKER, True)
    return module


def kimi_fla_shim_active() -> bool:
    return bool(getattr(sys.modules.get("fla"), _SHIM_MARKER, False))


def install_kimi_fla_shim() -> bool:
    """Install the minimal module hierarchy imported by Kimi remote code.

    Returns ``True`` only when this call installed the shim.  The operation is
    process-local and idempotent; it does not change site-packages.
    """
    if kimi_fla_shim_active():
        return False

    fla = _new_module("fla", package=True)
    modules = _new_module("fla.modules", package=True)
    ops = _new_module("fla.ops", package=True)
    kda = _new_module("fla.ops.kda")
    ops_utils = _new_module("fla.ops.utils", package=True)
    index = _new_module("fla.ops.utils.index")
    utils = _new_module("fla.utils", package=True)

    modules.ShortConvolution = PortableShortConvolution
    modules.FusedRMSNormGated = PortableFusedRMSNormGated
    kda.chunk_kda = torch_recurrent_kda
    kda.fused_recurrent_kda = torch_recurrent_kda
    index.prepare_lens_from_mask = _prepare_lens_from_mask
    index.prepare_cu_seqlens_from_mask = _prepare_cu_seqlens_from_mask
    utils.tensor_cache = _tensor_cache

    fla.modules = modules
    fla.ops = ops
    fla.utils = utils
    fla.__version__ = "acc-bench-portable"
    ops.kda = kda
    ops.utils = ops_utils
    ops_utils.index = index

    replacements = {
        "fla": fla,
        "fla.modules": modules,
        "fla.ops": ops,
        "fla.ops.kda": kda,
        "fla.ops.utils": ops_utils,
        "fla.ops.utils.index": index,
        "fla.utils": utils,
    }
    sys.modules.update(replacements)
    return True


def is_kimi_k3_checkpoint(model_path: Optional[str]) -> bool:
    """Identify local Kimi K3 checkpoints without importing remote code."""
    if not model_path:
        return False
    normalized_name = os.path.basename(os.path.normpath(model_path)).lower()
    if "kimi-k3" in normalized_name or "kimi_k3" in normalized_name:
        return True
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, ValueError):
        return False
    candidates: Iterable[object] = (
        config.get("model_type"),
        *(config.get("architectures") or []),
        (config.get("text_config") or {}).get("model_type"),
    )
    return any("kimi_k3" in str(value).lower() for value in candidates if value)


def ensure_kimi_torch_import_path(
    requested_backend: str,
    devices: Iterable[object],
    model_type: Optional[str],
    model_paths: Iterable[Optional[str]],
) -> bool:
    """Install the shim when the CLI policy resolves Kimi KDA to torch."""
    explicit_kimi = str(model_type or "").lower() in {"kimi_k3", "kimi-k3"}
    if not explicit_kimi and not any(is_kimi_k3_checkpoint(path) for path in model_paths):
        return False
    use_torch = requested_backend == "torch"
    if requested_backend == "auto":
        use_torch = any(
            str(getattr(device, "type", str(device).split(":", 1)[0])).lower()
            == "npu"
            for device in devices
        )
    if not use_torch:
        return False
    install_kimi_fla_shim()
    return True
