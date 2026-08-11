"""Ascend-safe eager Kimi Delta Attention recurrence.

The official Kimi K3 reference model dispatches KDA to FLA Triton kernels.
Some Triton-Ascend/CANN combinations cannot compile the chunk output kernel.
For an accuracy checker, correctness and portability matter more than fused
kernel throughput, so this module expresses the same recurrence with ordinary
PyTorch tensor operations supported by torch-npu.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _gate_parameter_view(
    value: Optional[torch.Tensor],
    value_heads: int,
    key_dim: int,
    name: str,
) -> Optional[torch.Tensor]:
    """Broadcast official head-wise and older channel-wise KDA parameters."""
    if value is None:
        return None
    value = value.float()
    if value.numel() == 1:
        return value.reshape(1, 1, 1, 1)
    if value.numel() == value_heads * key_dim:
        return value.reshape(1, 1, value_heads, key_dim)
    if value.numel() == value_heads:
        return value.reshape(1, 1, value_heads, 1)
    if value.numel() == key_dim:
        return value.reshape(1, 1, 1, key_dim)
    raise ValueError(
        f"Kimi KDA {name} has {value.numel()} elements; expected 1, "
        f"{value_heads}, {key_dim}, or {value_heads * key_dim}"
    )


def _canonical_state(
    initial_state: Optional[torch.Tensor],
    batch_size: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
    state_v_first: bool,
    like: torch.Tensor,
) -> torch.Tensor:
    """Return an fp32 state in canonical ``[B, HV, K, V]`` layout."""
    if initial_state is None:
        return torch.zeros(
            batch_size,
            value_heads,
            key_dim,
            value_dim,
            dtype=torch.float32,
            device=like.device,
        )
    state = initial_state.float()
    if state_v_first:
        state = state.transpose(-1, -2)
    expected = (batch_size, value_heads, key_dim, value_dim)
    if tuple(state.shape) != expected:
        raise ValueError(
            f"Kimi KDA initial_state shape {tuple(state.shape)} != {expected}"
        )
    return state.clone()


def _run_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run KDA sequentially using NPU-native elementwise and matmul ops."""
    outputs = []
    for token_idx in range(q.shape[1]):
        q_t = q[:, token_idx] * scale
        k_t = k[:, token_idx]
        v_t = v[:, token_idx]
        decay_t = decay[:, token_idx]

        state = state * torch.exp(decay_t).unsqueeze(-1)
        predicted_v = torch.matmul(k_t.unsqueeze(-2), state).squeeze(-2)
        beta_t = beta[:, token_idx]
        if beta_t.dim() == 2:
            beta_t = beta_t.unsqueeze(-1)
        delta = (v_t - predicted_v) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append(torch.matmul(q_t.unsqueeze(-2), state).squeeze(-2))

    if outputs:
        output = torch.stack(outputs, dim=1)
    else:
        output = v.new_empty(v.shape, dtype=torch.float32)
    return output, state


def torch_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    lower_bound: Optional[float] = None,
    state_v_first: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
    return_intermediate_states: bool = False,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Drop-in inference replacement for FLA ``chunk_kda``.

    This implements the recurrence used by FLA's fused recurrent kernel:
    decay state, apply the delta-rule correction, update the state, then read
    it with the normalized/scaled query.  It intentionally supports inference
    only, which is the only mode used by acc_bench.
    """
    if "transpose_state_layout" in kwargs:
        if state_v_first:
            raise ValueError(
                "Cannot pass both state_v_first and transpose_state_layout"
            )
        state_v_first = bool(kwargs.pop("transpose_state_layout"))
    if kwargs.get("cp_context") is not None:
        raise NotImplementedError("torch KDA fallback does not support context parallelism")
    if return_intermediate_states:
        raise NotImplementedError(
            "torch KDA fallback does not expose chunk intermediate states"
        )
    if q.shape != k.shape or q.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"Kimi KDA expects q=k=[B,T,H,K], v=[B,T,HV,V]; "
            f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )

    output_dtype = q.dtype
    batch_size, seq_len, query_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2], v.shape[3]
    if value_heads % query_heads != 0:
        raise ValueError(
            f"Kimi KDA value heads {value_heads} must be divisible by "
            f"query heads {query_heads}"
        )
    if tuple(g.shape) != (batch_size, seq_len, value_heads, key_dim):
        raise ValueError(f"Kimi KDA gate shape is invalid: {tuple(g.shape)}")
    valid_beta_shapes = {
        (batch_size, seq_len, value_heads),
        (batch_size, seq_len, value_heads, value_dim),
    }
    if tuple(beta.shape) not in valid_beta_shapes:
        raise ValueError(
            f"Kimi KDA beta shape is invalid: {tuple(beta.shape)}; expected "
            f"[B,T,HV] or [B,T,HV,V]"
        )

    q_fp = q.float()
    k_fp = k.float()
    v_fp = v.float()
    if use_qk_l2norm_in_kernel:
        q_fp = q_fp * torch.rsqrt(q_fp.square().sum(dim=-1, keepdim=True) + 1e-6)
        k_fp = k_fp * torch.rsqrt(k_fp.square().sum(dim=-1, keepdim=True) + 1e-6)
    if value_heads != query_heads:
        repeat = value_heads // query_heads
        q_fp = q_fp.repeat_interleave(repeat, dim=2)
        k_fp = k_fp.repeat_interleave(repeat, dim=2)

    if scale is None:
        scale = key_dim ** -0.5
    if use_gate_in_kernel:
        if A_log is None:
            raise ValueError("A_log is required when use_gate_in_kernel=True")
        a_view = _gate_parameter_view(A_log, value_heads, key_dim, "A_log")
        bias_view = _gate_parameter_view(dt_bias, value_heads, key_dim, "dt_bias")
        gate_input = g.float()
        if bias_view is not None:
            gate_input = gate_input + bias_view
        rate = torch.exp(a_view)
        if safe_gate:
            if lower_bound is None or not (-5 <= lower_bound < 0):
                raise ValueError("safe KDA gate requires lower_bound in [-5, 0)")
            decay = float(lower_bound) * torch.sigmoid(rate * gate_input)
        else:
            decay = -rate * F.softplus(gate_input)
    else:
        decay = g.float()

    beta_fp = beta.float()
    if use_beta_sigmoid_in_kernel:
        beta_fp = torch.sigmoid(beta_fp)
        if allow_neg_eigval:
            beta_fp = beta_fp * 2.0
    elif allow_neg_eigval:
        raise ValueError(
            "allow_neg_eigval=True requires use_beta_sigmoid_in_kernel=True"
        )

    if cu_seqlens is None:
        state = _canonical_state(
            initial_state,
            batch_size,
            value_heads,
            key_dim,
            value_dim,
            state_v_first,
            q,
        )
        output, final_state = _run_recurrence(
            q_fp, k_fp, v_fp, decay, beta_fp, state, float(scale)
        )
    else:
        if batch_size != 1:
            raise ValueError("cu_seqlens KDA input must have batch size 1")
        boundaries_source = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
        boundaries = boundaries_source.detach().cpu().tolist()
        sequence_count = len(boundaries) - 1
        states = _canonical_state(
            initial_state,
            sequence_count,
            value_heads,
            key_dim,
            value_dim,
            state_v_first,
            q,
        )
        output = torch.empty_like(v_fp)
        final_states = []
        for sequence_idx, (start, end) in enumerate(
            zip(boundaries[:-1], boundaries[1:])
        ):
            seq_output, seq_state = _run_recurrence(
                q_fp[:, start:end],
                k_fp[:, start:end],
                v_fp[:, start:end],
                decay[:, start:end],
                beta_fp[:, start:end],
                states[sequence_idx:sequence_idx + 1],
                float(scale),
            )
            output[:, start:end] = seq_output
            final_states.append(seq_state)
        final_state = torch.cat(final_states, dim=0)

    if state_v_first:
        final_state = final_state.transpose(-1, -2).contiguous()
    return output.to(output_dtype), final_state if output_final_state else None
