"""
SubgraphLocate: L2 sub-graph counterfactual diagnosis.

For each candidate layer from V1 L1, locate which sub-graph
(attention vs MLP/MoE) is the dominant error source by patching
ref sub-graph output into quant model and measuring recovery.

Reads hidden_states from V1 L1 cache (produced by --cache_top_k /
--auto_cache_bad_layer). This avoids re-forwarding prefix layers
and naturally handles rotated models (quant hidden is already in
rotated space from V1 L1's sequential forward).

Sub-graph decomposition:
  Dense:     self_attn | mlp
  MoE:       self_attn | moe.gate | moe.shared_expert | moe.experts
  GLM MoE:   self_attn | mlp.gate | mlp.shared_experts | mlp.experts
  GLM MLA:   self_attn | mlp  (dense MLP layers, e.g. first_k_density)

Usage:
    from accuracy_checker.subgraph_locate import diagnose_layers
    results = diagnose_layers(ref_model_path=..., quant_model_path=...,
                               candidate_layers=[3,4,5], prompt="你好", ...)
    # CLI: python3 run_accuracy_check.py --l2 --target_layers 3 4 5 ...
"""
from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import logging


logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accuracy_checker.replay_provider import ReplayProvider
from accuracy_checker.v2_metrics import rel_l2, recovery_ratio
from accuracy_checker.operator_patcher import ReplacementHook, _resolve_path
from accuracy_checker.utils import (
    load_rotation_matrix,
    normalize_quant_desc_values,
    rotate_hidden,
    unrotate_hidden,
)
from accuracy_checker.cache import (
    CACHE_FORMAT_VERSION,
    get_cache_dir,
    model_hash,
    prompt_hash,
)
from accuracy_checker.model_structure import get_text_config, is_kimi_k3_layer


# 多旋转矩阵支持 (GLM-5 DSA QuaRot)

# MLA 子图 → 旋转矩阵 key 映射
# 规则：子图输出的 hidden 维度落在哪个旋转空间，就用哪个 R 做右乘。
#   q_a_proj:     输入 R1 空间 → 输出 R3 空间 (权重融合了 R1^T @ W @ R3)
#   q_b_proj:     输入 R3 空间 → 输出恒等空间 (权重融合了 R3^T @ W)
#   kv_a_proj:    输入 R1 空间 → 输出 R4 空间 (权重融合了 R1^T @ W @ R4)
#   kv_b_proj:    输入 R4 空间 → 输出 R2 空间 (权重融合了 R4^T @ W @ R2)
#   o_proj:       输入 R2 空间 → 输出 R1 空间 (权重融合了 R2^T @ W @ R1)
# 非细粒度子图 (self_attn 整体 / mlp 子图): 内部成对抵消 → 输出 R1 空间
#
# 注意: attention 内部子图 (q_a/q_b/kv_a/kv_b) 的 patch 可能产生负 recovery，
# 因为 attention 有 softmax 非线性 — 如果 Q 仍有量化误差但 K/V 被 patch 为 ref，
# Q-K 不匹配导致 attention 分布错误，比全部保持量化误差更差。
# 只有 o_proj (attention 之后) 的 patch 是可靠的正 recovery。
_MLA_SUBGRAPH_ROT_KEY = {
    'self_attn.q_a_proj': 'rot_b_proj',        # R3 (2048)
    'self_attn.q_b_proj': None,                 # 恒等空间，不旋转
    'self_attn.kv_a_proj_with_mqa': 'rot_kv_b_proj',  # R4 (512), split: [R4, I(64)]
    'self_attn.kv_a_proj': 'rot_kv_b_proj',     # R4 (512), split: [R4, I(64)]
    'self_attn.kv_b_proj': 'rot_uv',            # R2 (256), split per-head: [I(192), R2]
    'self_attn.o_proj': 'rot',                   # R1 (6144)
    'self_attn.indexer': None,                   # 不旋转 (返回整数 topk_indices)
    'self_attn.indexer.wq_b': None,             # indexer 内部, 在 q_a_LN 输出空间 (R3)
    'self_attn.indexer.wk': None,               # indexer 内部, 输入是 layer input (R1), 输出进 k_norm
    'self_attn.indexer.weights_proj': None,      # indexer 内部, 输入是 layer input (R1)
    'self_attn.indexer.k_norm': None,           # indexer 内部 LN, wk 输出空间 (恒等)
}

# Attention 内部子图: patch 这些子图时，Q 仍为 quant 而 K/V 被 ref 替换，
# 导致 Q-K attention 分布错位 (softmax 非线性)。对这些子图用 RotBErr + SelfRotErr 代替 Recovery。
_MLA_ATTN_INNER_SUBGRAPHS = {
    'self_attn.q_a_proj', 'self_attn.q_b_proj',
    'self_attn.kv_a_proj_with_mqa', 'self_attn.kv_a_proj',
    'self_attn.kv_b_proj',
}

# Indexer 内部子图: 这些是 indexer 内部的线性层，patch 无意义 (fp8_index 是 NPU 自定义算子)
# 但可以计算 RotBErr 和 SelfRotErr 来定位 indexer 误差来源
_INDEXER_INNER_SUBGRAPHS = {
    'self_attn.indexer.wq_b', 'self_attn.indexer.wk',
    'self_attn.indexer.weights_proj', 'self_attn.indexer.k_norm',
}

# Attention 串联子链: 每条链内部的 op 是串联的，SelfRotErr 可视为 delta
# Q chain: input_layernorm → q_a_proj → q_a_layernorm → q_b_proj
# KV chain: input_layernorm → kv_a_proj → kv_a_layernorm → kv_b_proj
# Post-softmax: attn_output → o_proj
# Indexer chains (DSA): 独立于主 attention 的 sparse token selection
#   Indexer_Q: q_a_layernorm → wq_b (wq_b 输入 = qr, 同 q_b 输入源)
#   Indexer_K: layer_input → wk → k_norm (wk+LN 对, k_norm 后进 rope)
#   Indexer_W: layer_input → weights_proj (权重投影, 不经 LN)
_MLA_CHAINS = {
    'Q_chain': ['self_attn.q_a_proj', 'self_attn.q_b_proj'],
    'KV_chain': ['self_attn.kv_a_proj_with_mqa', 'self_attn.kv_b_proj'],
    'post_softmax': ['self_attn.o_proj'],
    'Indexer_Q': ['self_attn.indexer.wq_b'],
    'Indexer_K': ['self_attn.indexer.wk'],    # wk + k_norm 作为对, SelfRotErr 测 wk→k_norm
    'Indexer_W': ['self_attn.indexer.weights_proj'],
}

# 分支级组合 patch: 一起 patch 整个 Q 或 KV 路径的所有 op
# 同一分支内 Q 和 KV 都用 ref 值 → 不产生 softmax 错位
_MLA_BRANCH_PATCHES = {
    'Q_path': ['self_attn.q_a_proj', 'self_attn.q_b_proj'],
    'KV_path': ['self_attn.kv_a_proj_with_mqa', 'self_attn.kv_b_proj'],
    'Q+KV_all': ['self_attn.q_a_proj', 'self_attn.q_b_proj',
                 'self_attn.kv_a_proj_with_mqa', 'self_attn.kv_b_proj'],
}

# SelfRotErr 输入映射: 每个内部子图 → (输入来源子图, 输入空间旋转 key)
# q_a_proj: 输入来自 input_layernorm 输出 (R1 空间, q_a 的 right_rot=R1)
# q_b_proj: 输入来自 q_a_layernorm 输出 (R3 空间, q_a 的 left_rot=R3 → 输出 R3 空间)
# kv_a_proj: 输入来自 input_layernorm 输出 (R1 空间, kv_a 的 right_rot=R1)
# kv_b_proj: 输入来自 kv_a_layernorm 输出 (R4 空间, kv_a 的 left_rot=[R4,I] → KV 部分 R4 空间)
#
# 对于 SelfRotErr，我们需要用 ref forward 中该子图的输入，旋转到 quant 空间，
# 然后单独喂给 quant 子图，比较输出和 ref 输出（旋转后）的差异。
# "输入来源"标记用于从 ref forward hook 中捕获正确的输入。
_MLA_INNER_INPUT_SOURCE = {
    'self_attn.q_a_proj': 'input_layernorm',   # layer input → q_a_proj input (R1 space)
    'self_attn.q_b_proj': 'q_a_layernorm',      # q_a_proj → q_a_LN → q_b_proj (R3 space)
    'self_attn.kv_a_proj_with_mqa': 'input_layernorm',  # layer input → kv_a_proj input (R1 space)
    'self_attn.kv_a_proj': 'input_layernorm',
    'self_attn.kv_b_proj': 'kv_a_layernorm',    # kv_a_proj → kv_a_LN → kv_b_proj (R4 space)
}

# 子图输入的旋转 key: 将 ref 输入旋转到 quant 子图期望的输入空间
_MLA_INNER_INPUT_ROT_KEY = {
    'self_attn.q_a_proj': 'rot',             # R1: q_a right_rot=R1, input should be R1-rotated
    'self_attn.q_b_proj': 'rot_b_proj',       # R3: q_a left_rot=R3 → q_a output in R3 → q_b input in R3
    'self_attn.kv_a_proj_with_mqa': 'rot',    # R1: kv_a right_rot=R1
    'self_attn.kv_a_proj': 'rot',
    'self_attn.kv_b_proj': 'rot_kv_b_proj',   # R4: kv_a left_rot=[R4,I] → KV part in R4 → kv_b right_rot=R4
    'self_attn.indexer.wq_b': 'rot_b_proj',   # R3: wq_b 输入 = q_a_LN 输出, 同 q_b 输入空间
    'self_attn.indexer.wk': 'rot',            # R1: wk 输入 = layer input
    'self_attn.indexer.weights_proj': 'rot',  # R1: weights_proj 输入 = layer input
    'self_attn.indexer.k_norm': None,         # k_norm 输入 = wk 输出, 恒等空间 (不旋转)
}

# RotBErr 不适用的子图 (indexer 整体+内部/experts: 局部误差大但不反映真实影响，用专用指标)
# indexer 内部: wk/k_norm/weights_proj/wq_b 的 RotBErr 不直观 (非线性后处理 + fp8_index)
# 改用 SelfRotErr 隔离各自量化误差 + flip rate 看整体影响
_ROTBERR_SKIP_SUBGRAPHS = {
    'self_attn.indexer',
    'self_attn.indexer.wq_b', 'self_attn.indexer.wk',
    'self_attn.indexer.weights_proj', 'self_attn.indexer.k_norm',
    'mlp.experts',
}

# SelfRotErr: ref 输入来源
# 'layer_input' = 使用 ref_input (层输入, 在 input_layernorm 之前)
# 子图名 = 使用该子图的 ref 输出 (已在旋转后的 ref_sub 中)
_MLA_SELFROTERR_INPUT = {
    'self_attn.q_a_proj': 'layer_input',
    'self_attn.q_b_proj': 'self_attn.q_a_proj',
    'self_attn.kv_a_proj_with_mqa': 'layer_input',
    'self_attn.kv_a_proj': 'layer_input',
    'self_attn.kv_b_proj': 'self_attn.kv_a_proj_with_mqa',
    'self_attn.indexer.wq_b': 'self_attn.q_a_proj',  # wq_b 输入 = q_a_layernorm 输出 ≈ q_a_proj 输出
    'self_attn.indexer.wk': 'layer_input',            # wk 输入 = layer input x
    'self_attn.indexer.weights_proj': 'layer_input',  # weights_proj 输入 = layer input x
    'self_attn.indexer.k_norm': 'self_attn.indexer.wk',  # k_norm 输入 = wk 输出
}

# SelfRotErr: 每个内部子图输入前的 layernorm 路径
# fuse_ln_linear 后这些 LN weight=1/bias=0, 但仍做 (x-mean)/std 归一化。
# SelfRotErr 需要先将 ref input 旋转到 quant 空间, 再过 quant LN, 再喂 quant 子图,
# 才能正确隔离量化误差 (否则融合的 LN 权重会 double-apply)。
_MLA_SELFROTERR_LN_PATH = {
    'self_attn.q_a_proj': 'input_layernorm',
    'self_attn.q_b_proj': 'self_attn.q_a_layernorm',
    'self_attn.kv_a_proj_with_mqa': 'input_layernorm',
    'self_attn.kv_a_proj': 'input_layernorm',
    'self_attn.kv_b_proj': 'self_attn.kv_a_layernorm',
    'self_attn.indexer.wq_b': 'self_attn.q_a_layernorm',  # wq_b 输入 = q_a_LN 输出, 同 q_b
    'self_attn.indexer.wk': 'input_layernorm',            # wk 输入 = attn 输入 = input_LN 输出
    'self_attn.indexer.weights_proj': 'input_layernorm',  # weights_proj 输入 = attn 输入 = input_LN 输出
}

# SelfRotErr: 每个内部子图输出后的 layernorm 路径 (post-LN)
# 与 _MLA_SELFROTERR_LN_PATH (pre-LN) 相反 — 有些 op 后面紧跟 LN,
# SelfRotErr 应测 linear→LN 整体的误差, 比较目标是 LN 的输出而非 linear 输出。
# indexer.wk → k_norm: wk 输出经过 k_norm 归一化后才进 rope/fp8_index,
# 所以 wk 的 SelfRotErr = feed ref input → quant_wk → quant_k_norm, compare vs ref k_norm output
_MLA_SELFROTERR_POST_LN = {
    'self_attn.indexer.wk': 'self_attn.indexer.k_norm',
}


def load_all_rotation_matrices(
    rotation_matrix_path: str,
) -> Optional[Dict[str, torch.Tensor]]:
    """加载全部旋转矩阵 (R1/R2/R3/R4).

    搜索顺序:
      1. 直接作为文件加载 (rotation_matrices.pt 含 R1-R4, 或其他 .pt/.safetensors)
      2. 作为目录 → 搜索目录下 rotation_matrices.pt
      3. 文件不存在 → 搜索父目录下 rotation_matrices.pt
      4. Fallback: load_rotation_matrix (单 R1)

    Args:
        rotation_matrix_path: 可以是 .pt 文件、.safetensors 文件、或目录

    Returns:
        {key: R_tensor}, keys: 'rot'(R1), 'rot_b_proj'(R3), 'rot_uv'(R2), 'rot_kv_b_proj'(R4)
        None if no rotation matrix found.
    """
    import os
    if not rotation_matrix_path:
        return None

    def _try_load_pt(pt_path):
        """Try loading a .pt file with R1-R4 rotation matrices."""
        if not pt_path.endswith('.pt') or not os.path.exists(pt_path):
            return None
        data = torch.load(pt_path, weights_only=True)
        if isinstance(data, dict) and 'rot' in data:
            mats = {}
            for key in ['rot', 'rot_b_proj', 'rot_uv', 'rot_kv_b_proj']:
                if key in data:
                    mats[key] = data[key]
                    logger.info(f"Loaded rotation matrix {key}: shape={mats[key].shape}")
            if mats:
                return mats
        return None

    # Direct file path
    if os.path.isfile(rotation_matrix_path):
        result = _try_load_pt(rotation_matrix_path)
        if result is not None:
            return result
        # It's a file but not rotation_matrices.pt format — fall through to load_rotation_matrix

    # Directory path → search for rotation_matrices.pt
    if os.path.isdir(rotation_matrix_path):
        for candidate in ['rotation_matrices.pt', 'optional/rotation_matrices.pt']:
            result = _try_load_pt(os.path.join(rotation_matrix_path, candidate))
            if result is not None:
                return result

    # File doesn't exist → try parent directory (user may give .../model/rotate_matrix.pt
    # but actual file is .../model/rotation_matrices.pt)
    parent = os.path.dirname(rotation_matrix_path)
    if os.path.isdir(parent):
        result = _try_load_pt(os.path.join(parent, 'rotation_matrices.pt'))
        if result is not None:
            return result

    # Fallback: single rotation matrix
    if os.path.exists(rotation_matrix_path) or os.path.isdir(parent):
        R1 = load_rotation_matrix(rotation_matrix_path)
        if R1 is not None:
            return {'rot': R1}

    logger.warning(f"No rotation matrix found at or near: {rotation_matrix_path}")
    return None


def _rotate_kv_a_proj_block(
    ref_output: torch.Tensor,
    R_specific: torch.Tensor,
    out_dim: int,
    config: dict,
) -> Optional[torch.Tensor]:
    """Block rotation for kv_a_proj_with_mqa: split [R4(512), I(64)]."""
    kv_lora = config.get('kv_lora_rank', 512)
    rope_dim = config.get('qk_rope_head_dim', 64)
    if R_specific.shape[0] != kv_lora or out_dim != kv_lora + rope_dim:
        return None
    kv_dim = kv_lora
    ref_kv = ref_output[..., :kv_dim]
    ref_rope = ref_output[..., kv_dim:]
    rot_kv = rotate_hidden(ref_kv, R_specific)  # ref_kv @ R4
    rotated = torch.cat([rot_kv, ref_rope], dim=-1)
    return rotated.to(ref_output.dtype)


def _rotate_kv_b_proj_block(
    ref_output: torch.Tensor,
    R_specific: torch.Tensor,
    out_dim: int,
    config: dict,
) -> Optional[torch.Tensor]:
    """Block rotation for kv_b_proj: split per-head [I(192), R2(256)].

    Caller must verify R_specific.shape[0] == v_head_dim before calling.
    Returns None when out_dim does not match expected (caller falls through to warning).
    """
    qk_nope = config.get('qk_nope_head_dim', 192)
    v_head = config.get('v_head_dim', 256)
    num_kv_heads = config.get('num_key_value_heads', 64)
    head_dim = qk_nope + v_head  # 448
    expected_dim = num_kv_heads * head_dim  # 28672
    if out_dim != expected_dim:
        return None
    orig_shape = ref_output.shape
    reshaped = ref_output.view(*orig_shape[:-1], num_kv_heads, head_dim)
    ref_qk = reshaped[..., :qk_nope]
    ref_v = reshaped[..., qk_nope:]
    rot_v = rotate_hidden(ref_v, R_specific)  # ref_v @ R2 per head
    rotated = torch.cat([ref_qk, rot_v], dim=-1)
    rotated = rotated.view(orig_shape)
    return rotated.to(ref_output.dtype)


def _rotate_subgraph_output(
    ref_output: torch.Tensor,
    subgraph_name: str,
    rot_mats: Optional[Dict[str, torch.Tensor]],
    R1: Optional[torch.Tensor],
    config: Optional[dict] = None,
) -> Tuple[torch.Tensor, bool]:
    """对子图 ref output 施加正确的旋转矩阵.

    Args:
        ref_output: ref 子图输出 (original space)
        subgraph_name: 子图名称
        rot_mats: 全部旋转矩阵 dict (from load_all_rotation_matrices)
        R1: R1 旋转矩阵 (backwards compat, used when rot_mats is None)
        config: model config dict (for head structure, needed for block rotation)

    Returns:
        (rotated_output, can_patch): can_patch=False 表示旋转不匹配，不应 patch

    规则:
      - 在 _MLA_SUBGRAPH_ROT_KEY 中且 rot_key=None → 不旋转 (恒等空间, 如 q_b_proj)
      - 在 _MLA_SUBGRAPH_ROT_KEY 中且 rot_key=str → 用对应旋转矩阵
        - kv_a_proj_with_mqa: split [R4(512), I(64)] — 前 512 维 @ R4, 后 64 维不动
        - kv_b_proj: split per-head [I(192), R2(256)] — 每个 head 前 192 不动, 后 256 @ R2
        - 其他: 直接整维 @ R
      - 不在 _MLA_SUBGRAPH_ROT_KEY 中 → 用 R1 (大子图, 如 self_attn 整体, mlp 子图)
    """
    # Not in MLA fine-grained dict: large sub-graph (self_attn, mlp.*)
    if subgraph_name not in _MLA_SUBGRAPH_ROT_KEY:
        if R1 is not None and ref_output.shape[-1] == R1.shape[0]:
            return rotate_hidden(ref_output, R1), True
        return ref_output, True

    rot_key = _MLA_SUBGRAPH_ROT_KEY[subgraph_name]
    # Identity space — no rotation
    if rot_key is None:
        return ref_output, True

    # Missing rotation matrix — skip rotation but allow patch
    if rot_mats is None or rot_key not in rot_mats:
        return ref_output, True

    R_specific = rot_mats[rot_key]
    out_dim = ref_output.shape[-1]

    # Exact match — simple full rotation
    if out_dim == R_specific.shape[0]:
        return rotate_hidden(ref_output, R_specific), True

    # --- Block rotation for kv_a_proj_with_mqa ---
    # QuaRot left_rot = [R4, I(64)] on weight:
    #   W_quant = block_diag(R4, I)^T @ W_ref = block_diag(R4^T, I) @ W_ref
    #   output_quant = input @ W_quant^T = input @ W_ref^T @ block_diag(R4, I)
    #     = ref_output @ block_diag(R4, I)
    # So KV part: ref_kv @ R4 = rotate_hidden(ref_kv, R4)
    if 'kv_a_proj' in subgraph_name and config is not None:
        rotated = _rotate_kv_a_proj_block(ref_output, R_specific, out_dim, config)
        if rotated is not None:
            return rotated, True

    # --- Block rotation for kv_b_proj ---
    # QuaRot left_rot = [I(192), R2(256)] on weight:
    #   W_quant = block_diag(I, R2)^T @ W_ref = block_diag(I, R2^T) @ W_ref
    #   output_quant V part per head: ref_v @ R2 = rotate_hidden(ref_v, R2)
    if 'kv_b_proj' in subgraph_name and config is not None:
        v_head = config.get('v_head_dim', 256)
        if R_specific.shape[0] != v_head:
            return ref_output, True
        rotated = _rotate_kv_b_proj_block(ref_output, R_specific, out_dim, config)
        if rotated is not None:
            return rotated, True

    # Dim mismatch that we can't handle
    logger.warning(f"  Subgraph '{subgraph_name}' output dim {out_dim} "
          f"!= {rot_key} dim {R_specific.shape[0]}, skipping patch (unhandled block rotation)")
    return ref_output, False


# V1 L1 cache 读取

# Cache 目录通过 accuracy_checker.cache 模块统一管理
# (CLI --cache_dir > ACC_CACHE_DIR env > ./.acc_cache/)

# V1 L1 cache may use different quant_method suffixes depending on how it was run.
# Common values: "fake_quant" (V1 with --use_fake_quant), "dequantize" (V1 default).
_QUANT_METHOD_FALLBACKS = ["fake_quant", "dequantize"]


def _try_cache_match(
    base: str,
    mh: str,
    ph: str,
    target_layer: int,
    side: str,
    quant_method: str,
    device: str,
) -> Optional[torch.Tensor]:
    """Try a single quant_method for cache matching."""
    import glob

    # Strategy 1: exact model + prompt
    pattern = (
        f"{mh}_{ph}_s*_L{target_layer}_{side}_"
        f"{CACHE_FORMAT_VERSION}_*_{quant_method}.pt"
    )
    matches = glob.glob(base + pattern)
    if matches:
        if len(matches) > 1:
            logger.info(f"  [CACHE] Multiple matches (exact) for L{target_layer} {side}, "
                  f"using first: {os.path.basename(matches[0])}")
        return torch.load(matches[0], weights_only=True, map_location=device)

    # Strategy 2: compatibility with one legacy cache whose prompt identity
    # was only "N_tokens". Never choose arbitrarily when multiple samples are
    # present: that can silently replay another prompt's hidden states.
    pattern = (
        f"{mh}_*_s*_L{target_layer}_{side}_"
        f"{CACHE_FORMAT_VERSION}_*_{quant_method}.pt"
    )
    matches = glob.glob(base + pattern)
    if len(matches) == 1:
        logger.warning(
            f"  [CACHE] Exact sample key not found; using sole legacy/model-only "
            f"cache for L{target_layer} {side}: {os.path.basename(matches[0])}"
        )
        return torch.load(matches[0], weights_only=True, map_location=device)
    if len(matches) > 1:
        logger.warning(
            f"  [CACHE] {len(matches)} model-only matches for L{target_layer} {side}; "
            "refusing an ambiguous cross-prompt cache match"
        )
        return None

    # Strategy 3: layer + side + method only (loosest)
    pattern = (
        f"*_s*_L{target_layer}_{side}_"
        f"{CACHE_FORMAT_VERSION}_*_{quant_method}.pt"
    )
    matches = glob.glob(base + pattern)
    if len(matches) == 1:
        logger.info(f"  [CACHE] Matched by layer/side/method only: {os.path.basename(matches[0])}")
        return torch.load(matches[0], weights_only=True, map_location=device)
    if len(matches) > 1:
        logger.info(f"  [CACHE] {len(matches)} ambiguous matches for L{target_layer} {side}, "
              f"cannot resolve without model_path. Skip.")
        return None

    return None


def _load_l1_cache(
    model_path: str,
    prompt: str,
    target_layer: int,
    side: str,
    quant_method: str,
    device: str,
) -> Optional[torch.Tensor]:
    """Load hidden_states from V1 L1 cache.

    Tries the specified quant_method first, then falls back to other common
    suffixes (fake_quant, dequantize) since V1 L1's --use_fake_quant flag
    determines the cache suffix and may differ from the current --quant_method.

    Cache key: {model_hash}_{prompt_hash}_s{seqlen}_L{layer}_{side}_{ver}_{int4ver}_{method}.pt
    """
    mh = model_hash(model_path)
    ph = prompt_hash(prompt)
    base = os.path.join(get_cache_dir(), "")

    # Try specified method first, then fallbacks
    methods_to_try = [quant_method]
    for fb in _QUANT_METHOD_FALLBACKS:
        if fb != quant_method:
            methods_to_try.append(fb)

    for qm in methods_to_try:
        result = _try_cache_match(base, mh, ph, target_layer, side, qm, device)
        if result is not None:
            if qm != quant_method:
                logger.info(f"  [CACHE] quant_method '{quant_method}' not found, "
                      f"matched with fallback '{qm}'")
            return result

    return None


def _unpack_l1_cache_entry(entry):
    """Return ``(hidden_states, layer_state)`` for legacy/v4 caches."""
    if isinstance(entry, dict):
        return entry.get("hidden_states"), entry.get("layer_state")
    return entry, None


# V1 L1 报告解析
# 子图 patch 逻辑

def _load_quant_desc(quant_model_path: str) -> Optional[dict]:
    """加载 quant_model_description.json"""
    import json
    desc_path = os.path.join(quant_model_path, "quant_model_description.json")
    if os.path.exists(desc_path):
        with open(desc_path) as f:
            return normalize_quant_desc_values(json.load(f))
    return None


def _classify_subgraph_quant_type(
    quant_desc: Optional[dict],
    layer_idx: int,
    subgraph_name: str,
) -> str:
    """根据 quant_model_description.json 判断子图是否为 FLOAT.

    规则: 如果子图下所有 .weight 键都是 FLOAT → 返回 "FLOAT",
    否则返回子图内最主要的量化类型 (如 "W8A8_DYNAMIC").

    Returns: "FLOAT" / "W8A8" / "W8A8_DYNAMIC" / "MIXED" / "UNKNOWN"
    """
    if quant_desc is None:
        return "UNKNOWN"

    prefix = f"model.layers.{layer_idx}.{subgraph_name}."
    weight_types = {}
    for key, val in quant_desc.items():
        if not key.startswith(prefix):
            continue
        if not key.endswith(".weight"):
            continue
        # Skip scale/offset/quant_bias sub-keys
        op_path = key[len(prefix):]
        weight_types[op_path] = val

    if not weight_types:
        return "UNKNOWN"

    # If ALL weights are FLOAT, the subgraph is FLOAT
    non_float = {k: v for k, v in weight_types.items()
                 if v != "FLOAT" and isinstance(v, str)}
    if not non_float:
        return "FLOAT"

    # Return the most common quant type among non-FLOAT ops
    from collections import Counter
    type_counts = Counter(v for v in non_float.values() if isinstance(v, str))
    return type_counts.most_common(1)[0][0] if type_counts else "MIXED"


def _move_kwargs(kwargs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move all tensors in kwargs to target device."""
    moved = {}
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device)
        elif isinstance(v, tuple) and all(isinstance(t, torch.Tensor) for t in v):
            moved[k] = tuple(t.to(device) for t in v)
        else:
            moved[k] = v
    return moved


def _compute_topk_flip_rate(
    ref_output: torch.Tensor,
    quant_output: torch.Tensor,
    topk: int = 8,
) -> float:
    """Compute expert flip rate between ref and quant top-k selections.

    For each position, measure how many selected experts differ.
    Returns average fraction of experts that differ per position.
    E.g. top8, 1 expert different → flip = 1/8 = 12.5%
    """
    if ref_output.shape != quant_output.shape:
        return float('nan')

    # Integer outputs are already selected indices (Kimi K3 returns top-16).
    # Floating outputs are logits/scores unless they already have top-k width.
    outputs_are_indices = not ref_output.dtype.is_floating_point
    if not outputs_are_indices and ref_output.shape[-1] != topk:
        k = min(topk, ref_output.shape[-1])
        ref_idx = ref_output.float().topk(k, dim=-1).indices
        quant_idx = quant_output.float().topk(k, dim=-1).indices
    else:
        ref_idx = ref_output
        quant_idx = quant_output

    k = ref_idx.shape[-1]
    ref_flat = ref_idx.reshape(-1, k).cpu()
    quant_flat = quant_idx.reshape(-1, k).cpu()

    n_positions = ref_flat.shape[0]
    total_diff = 0
    for i in range(n_positions):
        ref_set = set(ref_flat[i].tolist())
        quant_set = set(quant_flat[i].tolist())
        total_diff += len(ref_set.symmetric_difference(quant_set))

    # Each differing expert is counted in both symmetric difference,
    # but we want "how many unique experts are different" = |symmetric_diff| / 2 per side,
    # expressed as fraction of k: avg |symmetric_diff| / (2 * k)
    return (total_diff / (2 * k * n_positions)) if n_positions > 0 else float('nan')


def _capture_subgraph_outputs(
    layer: torch.nn.Module,
    hidden: torch.Tensor,
    subgraph_names: List[str],
    layer_kwargs: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """在 ref forward 中用 hook 捕获子图 output.

    支持嵌套属性名如 'mlp.gate', 'mlp.shared_experts', 'mlp.experts'.
    """
    captured = {}
    device = next(layer.parameters()).device

    def make_hook(store, name):
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            store[name] = t.detach()
            if (
                name.endswith(".gate") and isinstance(out, tuple)
                and len(out) >= 3 and isinstance(out[2], torch.Tensor)
            ):
                store[f"{name}.__indices"] = out[2].detach()
        return hook

    hooks = []
    for name in subgraph_names:
        mod = _resolve_path(layer, name)
        if mod is not None:
            hooks.append(mod.register_forward_hook(make_hook(captured, name)))

    kwargs = _move_kwargs(layer_kwargs, device)
    with torch.no_grad():
        _ = layer(hidden.to(device), **kwargs)

    for h in hooks:
        h.remove()

    return captured


def _patch_subgraph(
    layer: torch.nn.Module,
    hidden: torch.Tensor,
    subgraph_name: str,
    ref_output: torch.Tensor,
    layer_kwargs: Dict[str, Any],
) -> torch.Tensor:
    """在 quant forward 中 patch 单个子图，返回 layer output.

    Uses ReplacementHook from operator_patcher for safe active/inactive control.
    """
    mod = _resolve_path(layer, subgraph_name)
    if mod is None:
        raise AttributeError(f"Cannot resolve subgraph '{subgraph_name}' on layer")

    rh = ReplacementHook(ref_output)
    h = mod.register_forward_hook(rh)
    device = next(layer.parameters()).device
    kwargs = _move_kwargs(layer_kwargs, device)
    try:
        with rh.active():
            with torch.no_grad():
                patched_out = layer(hidden.to(device), **kwargs)
    finally:
        h.remove()

    t = patched_out[0] if isinstance(patched_out, tuple) else patched_out
    return t.detach()


def detect_model_type(layer: torch.nn.Module) -> str:
    """检测层结构类型: dense / moe / glm_mla / glm_moe_dsa / deepseek_v4 / qwen3_5_moe.

    GLM-5 (glm_moe_dsa) 的 MoE 挂在 layer.mlp 下:
      - layer.mlp 是 GlmMoeDsaMoE (有 gate/shared_experts/experts)
      - layer.self_attn 是 MLA (有 q_a_proj/kv_a_proj_with_mqa 等)

    Qwen3.5/3.6 MoE (qwen3_5_moe) 的 MoE 也挂在 layer.mlp 下:
      - layer.mlp 是 Qwen3_5MoeSparseMoeBlock (有 gate/experts, shared_expert 单数)
      - layer.self_attn 是标准注意力 (q_proj/k_proj/v_proj/o_proj, 非 MLA)
      - 或 layer.linear_attn (线性注意力, 含 conv1d)

    通用 MoE (如 Qwen3MoE) 挂在 layer.moe / layer.block_sparse_moe 下.
    """
    if is_kimi_k3_layer(layer):
        return 'kimi_k3'

    # DeepSeek-V4 uses HCA/CSA attention and hash/router MoE modules; do not
    # mistake its ``mlp`` for GLM-MoE-DSA merely because both are sparse MoE.
    layer_name = type(layer).__name__.lower()
    attn_name = type(getattr(layer, 'self_attn', None)).__name__.lower()
    if 'deepseekv4' in layer_name or 'deepseekv4' in attn_name \
            or hasattr(getattr(layer, 'self_attn', None), 'compressor'):
        return 'deepseek_v4'

    mlp = getattr(layer, 'mlp', None)
    has_mlp_gate_experts = (
        mlp is not None and hasattr(mlp, 'gate') and hasattr(mlp, 'experts')
    )

    # Qwen3.5/3.6 MoE: mlp.gate+experts 但 self_attn 是标准注意力 (有 q_proj, 无 q_a_proj)
    # 或 layer 有 linear_attn (Qwen3.6 独有)
    if has_mlp_gate_experts:
        has_linear_attn = hasattr(layer, 'linear_attn')
        attn = getattr(layer, 'self_attn', None)
        has_standard_attn = (
            attn is not None and hasattr(attn, 'q_proj')
            and not hasattr(attn, 'q_a_proj')
            and not hasattr(attn, 'kv_a_proj_with_mqa')
        )
        # Qwen3.6 特征: shared_expert 单数 OR linear_attn OR 标准注意力 (无 MLA)
        has_qwen36_marker = (
            hasattr(mlp, 'shared_expert')  # 单数, GLM 是 shared_experts 复数
            or has_linear_attn
            or has_standard_attn
        )
        if has_qwen36_marker:
            return 'qwen3_5_moe'
        # 否则 fallthrough 到 glm_moe_dsa (GLM 有 shared_experts 复数 + MLA)

    # GLM MoE DSA: mlp 本身就是 MoE (有 gate + shared_experts + experts, MLA attention)
    if has_mlp_gate_experts:
        return 'glm_moe_dsa'

    # 通用 MoE
    has_moe = hasattr(layer, 'moe') or hasattr(layer, 'block_sparse_moe')
    if has_moe:
        return 'moe'

    # GLM MLA (dense MLP + MLA attention)
    attn = getattr(layer, 'self_attn', None)
    if attn is not None:
        has_mla = (
            hasattr(attn, 'q_a_proj') or hasattr(attn, 'q_a')
            or hasattr(attn, 'kv_a_proj_with_mqa') or hasattr(attn, 'kv_a_proj')
        )
        if has_mla:
            return 'glm_mla'

    return 'dense'


def _get_mla_subgraph_names(attn: torch.nn.Module) -> List[str]:
    """GLM MLA attention 子图拆分.

    MLA 投影路径:
      Query:  q_a_proj → q_a_layernorm → q_b_proj
      KV:     kv_a_proj_with_mqa → kv_a_layernorm → kv_b_proj
      Output: o_proj
      Indexer: indexer (DSA sparse token selection)

    注意: q_a→q_b, kv_a→kv_b 是 SmoothQuant pair,
    patch pair middle (q_a, kv_a) 的 output 存在 scale 空间不匹配风险.
    pair exit (q_b, kv_b, o_proj) 的 output patch 更可靠.
    """
    names = []
    if hasattr(attn, 'q_a_proj'):
        names.append('self_attn.q_a_proj')
    if hasattr(attn, 'q_b_proj'):
        names.append('self_attn.q_b_proj')
    if hasattr(attn, 'kv_a_proj_with_mqa'):
        names.append('self_attn.kv_a_proj_with_mqa')
    elif hasattr(attn, 'kv_a_proj'):
        names.append('self_attn.kv_a_proj')
    if hasattr(attn, 'kv_b_proj'):
        names.append('self_attn.kv_b_proj')
    if hasattr(attn, 'o_proj'):
        names.append('self_attn.o_proj')
    # Indexer: 整体 + 内部细粒度子图
    if hasattr(attn, 'indexer'):
        names.append('self_attn.indexer')
        indexer = attn.indexer
        if hasattr(indexer, 'wq_b'):
            names.append('self_attn.indexer.wq_b')
        if hasattr(indexer, 'wk'):
            names.append('self_attn.indexer.wk')
        if hasattr(indexer, 'k_norm'):
            names.append('self_attn.indexer.k_norm')
        if hasattr(indexer, 'weights_proj'):
            names.append('self_attn.indexer.weights_proj')
    return names


def _get_glm_moe_dsa_subgraph_names(layer: torch.nn.Module, mla_fine: bool) -> List[str]:
    """子图名 for GLM-5 MoE DSA (MLA attention + mlp MoE)."""
    attn = getattr(layer, 'self_attn', None)
    if mla_fine and attn is not None:
        # 大子图 + 小子图都输出: 先放 self_attn 整体, 再放细粒度子图
        names = ['self_attn']
        names.extend(_get_mla_subgraph_names(attn))
    else:
        names = ['self_attn']

    # GLM-5 MoE 子图: mlp 本身就是 MoE
    mlp = getattr(layer, 'mlp', None)
    if mlp is not None:
        if hasattr(mlp, 'gate'):
            names.append('mlp.gate')
        if hasattr(mlp, 'shared_experts'):
            names.append('mlp.shared_experts')
        if hasattr(mlp, 'experts'):
            names.append('mlp.experts')
    return names


def _get_deepseek_v4_subgraph_names(layer: torch.nn.Module, mla_fine: bool) -> List[str]:
    """DeepSeek-V4 HCA/CSA attention, mHC boundary and hash/top-k MoE."""
    names = ["self_attn"]
    attn = getattr(layer, "self_attn", None)
    if mla_fine and attn is not None:
        for child in (
            "q_a_proj", "q_a_norm", "q_b_proj", "q_b_norm",
            "kv_proj", "kv_norm", "o_a_proj", "o_b_proj",
        ):
            if hasattr(attn, child):
                names.append(f"self_attn.{child}")
        compressor = getattr(attn, "compressor", None)
        if compressor is not None:
            names.append("self_attn.compressor")
            for child in ("kv_proj", "gate_proj", "kv_norm"):
                if hasattr(compressor, child):
                    names.append(f"self_attn.compressor.{child}")
            indexer = getattr(compressor, "indexer", None)
            if indexer is not None:
                names.append("self_attn.compressor.indexer")
                for child in ("kv_proj", "gate_proj", "kv_norm", "q_b_proj", "scorer"):
                    if hasattr(indexer, child):
                        names.append(f"self_attn.compressor.indexer.{child}")
                if hasattr(getattr(indexer, "scorer", None), "weights_proj"):
                    names.append("self_attn.compressor.indexer.scorer.weights_proj")
    mlp = getattr(layer, "mlp", None)
    if mlp is not None:
        for child in ("gate", "shared_experts", "experts"):
            if hasattr(mlp, child):
                names.append(f"mlp.{child}")
    return names


def _get_qwen3_5_moe_subgraph_names(layer: torch.nn.Module, mla_fine: bool) -> List[str]:
    """子图名 for Qwen3.5/3.6 MoE (标准/线性注意力 + mlp 挂 MoE)."""
    # self_attn (整体) 或 linear_attn (整体), 取存在的那个
    names = []
    if hasattr(layer, 'self_attn'):
        names.append('self_attn')
    elif hasattr(layer, 'linear_attn'):
        names.append('linear_attn')

    # MoE 子图: mlp.gate / mlp.shared_expert (单数) / mlp.experts
    mlp = getattr(layer, 'mlp', None)
    if mlp is not None:
        if hasattr(mlp, 'gate'):
            names.append('mlp.gate')
        # Qwen3.6 用 shared_expert (单数); GLM 用 shared_experts (复数)
        if hasattr(mlp, 'shared_expert'):
            names.append('mlp.shared_expert')
        if hasattr(mlp, 'shared_experts'):
            names.append('mlp.shared_experts')
        if hasattr(mlp, 'experts'):
            names.append('mlp.experts')
    return names


def _get_kimi_k3_subgraph_names(layer: torch.nn.Module, mla_fine: bool) -> List[str]:
    """Subgraphs for Kimi K3 KDA/MLA + Stable LatentMoE + AttnRes."""
    names = ['self_attn']
    attn = getattr(layer, 'self_attn', None)
    if mla_fine and attn is not None:
        if hasattr(attn, 'q_a_proj') or hasattr(attn, 'kv_a_proj_with_mqa'):
            names.extend(_get_mla_subgraph_names(attn))
            if hasattr(attn, 'g_proj'):
                names.append('self_attn.g_proj')
        else:
            for child in (
                'q_proj', 'k_proj', 'v_proj',
                'q_conv1d', 'k_conv1d', 'v_conv1d',
                'f_a_proj', 'f_b_proj', 'b_proj',
                'g_proj', 'g_a_proj', 'g_b_proj', 'o_norm', 'o_proj',
            ):
                if hasattr(attn, child):
                    names.append(f'self_attn.{child}')

    moe = getattr(layer, 'block_sparse_moe', None)
    if moe is not None:
        names.append('block_sparse_moe')
        for child in (
            'gate', 'routed_expert_down_proj', 'routed_expert_norm',
            'experts', 'routed_expert_up_proj', 'shared_experts',
        ):
            if hasattr(moe, child):
                names.append(f'block_sparse_moe.{child}')
    elif hasattr(layer, 'mlp'):
        names.append('mlp')

    for child in (
        'self_attention_res_proj', 'self_attention_res_norm',
        'mlp_res_proj', 'mlp_res_norm',
    ):
        if hasattr(layer, child):
            names.append(child)
    return names


def _get_generic_moe_subgraph_names(layer: torch.nn.Module, mla_fine: bool) -> List[str]:
    """子图名 for 通用 MoE (layer.moe / layer.block_sparse_moe)."""
    names = ['self_attn']
    moe = getattr(layer, 'moe', None) or getattr(layer, 'block_sparse_moe', None)
    if moe is not None:
        if hasattr(moe, 'gate'):
            names.append('moe.gate')
        if hasattr(moe, 'shared_expert'):
            names.append('moe.shared_expert')
        if hasattr(moe, 'shared_experts'):
            names.append('moe.shared_experts')
        if hasattr(moe, 'experts'):
            names.append('moe.experts')
    if hasattr(layer, 'mlp'):
        names.append('mlp')
    return names


def _get_glm_mla_subgraph_names(layer: torch.nn.Module, mla_fine: bool) -> List[str]:
    """子图名 for GLM MLA (dense MLP + MLA attention)."""
    attn = getattr(layer, 'self_attn', None)
    if mla_fine and attn is not None:
        names = ['self_attn']
        names.extend(_get_mla_subgraph_names(attn))
    else:
        names = ['self_attn']
    names.append('mlp')
    return names


# Dispatch table for model-type → subgraph name generator.
# Each helper takes (layer, mla_fine) and returns the list of subgraph names.
_SUBGRAPH_NAME_DISPATCH = {
    'glm_moe_dsa': _get_glm_moe_dsa_subgraph_names,
    'deepseek_v4': _get_deepseek_v4_subgraph_names,
    'qwen3_5_moe': _get_qwen3_5_moe_subgraph_names,
    'qwen3_6': _get_qwen3_5_moe_subgraph_names,
    'qwen3_6_moe': _get_qwen3_5_moe_subgraph_names,
    'kimi_k3': _get_kimi_k3_subgraph_names,
    'moe': _get_generic_moe_subgraph_names,
    'glm_mla': _get_glm_mla_subgraph_names,
}


def get_subgraph_names(model_type: str, layer: torch.nn.Module, mla_fine: bool = False) -> List[str]:
    """根据模型类型返回子图名称列表.

    Args:
        model_type: 模型类型
        layer: decoder layer
        mla_fine: 是否对 MLA attention 做细粒度拆分 (q_a/q_b/kv_a/kv_b/o_proj)

    GLM-5 (glm_moe_dsa) 结构:
      layer.self_attn  — MLA attention (q_a_proj → q_b_proj / kv_a_proj → kv_b_proj → o_proj)
      layer.mlp.gate   — TopkRouter
      layer.mlp.shared_experts — shared MLP (gate_proj/up_proj/down_proj)
      layer.mlp.experts — routed experts (fused gate_up_proj + down_proj)

    通用 MoE (moe) 结构:
      layer.self_attn  — attention
      layer.moe.gate   — router
      layer.moe.shared_expert / moe.shared_experts
      layer.moe.experts — routed experts
    """
    builder = _SUBGRAPH_NAME_DISPATCH.get(model_type)
    if builder is not None:
        return builder(layer, mla_fine)

    # dense (and any unknown model_type fallback)
    return ['self_attn', 'mlp']


# 核心: 单层诊断

def _compute_input_recovery(
    handle, ref_input, ref_out, base_l2, R1,
) -> float:
    """Feed ref_hidden (rotated) into quant layer and measure input recovery."""
    # --- Input patch: feed ref_hidden (rotated) into quant layer ---
    if R1 is not None:
        ref_input_rotated = rotate_hidden(ref_input, R1)
    else:
        ref_input_rotated = ref_input
    input_patched_out = handle.forward_quant(ref_input_rotated).detach().cpu().float()
    if R1 is not None:
        input_patched_out = unrotate_hidden(input_patched_out, R1)
    input_patched_l2 = rel_l2(input_patched_out, ref_out, [-1]).item()
    return recovery_ratio(base_l2, input_patched_l2)


def _rotate_ref_subgraph_outputs(
    ref_sub, rot_mats, R1, model_config,
):
    """Rotate ref sub-graph outputs to match quant's internal (rotated) space.

    Returns (ref_sub_rotated, unpatchable_set).
    """
    unpatchable = set()
    if rot_mats is None and R1 is None:
        return ref_sub, unpatchable

    ref_sub_rotated = {}
    for name, out in ref_sub.items():
        rotated, can_patch = _rotate_subgraph_output(
            out, name, rot_mats, R1, config=model_config,
        )
        ref_sub_rotated[name] = rotated
        if not can_patch:
            unpatchable.add(name)
    return ref_sub_rotated, unpatchable


def _compute_rotberr(
    subgraph_names, ref_sub, quant_sub, quant_desc, layer_idx,
) -> Dict[str, float]:
    """Compute RotBErr (rotation-aligned boundary error) for each subgraph."""
    subgraph_rotberr = {}
    for name in subgraph_names:
        if name in _ROTBERR_SKIP_SUBGRAPHS:
            continue
        if quant_desc is not None:
            if _classify_subgraph_quant_type(quant_desc, layer_idx, name) == "FLOAT":
                continue
        ref_rot_val = ref_sub.get(name)
        quant_val = quant_sub.get(name)
        if ref_rot_val is None or quant_val is None:
            continue
        ref_f = ref_rot_val.detach().cpu().float()
        quant_f = quant_val.detach().cpu().float()
        if ref_f.shape != quant_f.shape:
            continue
        diff_norm = (ref_f - quant_f).norm().item()
        ref_norm = ref_f.norm().item()
        subgraph_rotberr[name] = diff_norm / ref_norm if ref_norm > 0 else float('nan')
    return subgraph_rotberr


def _resolve_selfroterr_input(name, ref_input, ref_sub, rot_mats):
    """Resolve the ref input tensor for a SelfRotErr computation.

    Returns the float ref input, or None if it cannot be resolved.
    """
    input_source = _MLA_SELFROTERR_INPUT.get(name)
    if input_source == 'layer_input':
        rot_key = _MLA_INNER_INPUT_ROT_KEY.get(name)
        if rot_key is None or rot_key not in rot_mats:
            return None
        R_input = rot_mats[rot_key]
        return rotate_hidden(ref_input.float(), R_input)
    ref_self_inp = ref_sub.get(input_source)
    if ref_self_inp is None:
        return None
    return ref_self_inp.float()


def _forward_quant_inner_op(quant_layer, name, ref_self_inp, model_config):
    """Run quant layer's inner op (with pre-LN) on the ref input.

    Returns (quant_self_out, device_quant) or None on failure.
    """
    ln_path = _MLA_SELFROTERR_LN_PATH.get(name)
    quant_ln = _resolve_path(quant_layer, ln_path) if ln_path else None
    quant_mod = _resolve_path(quant_layer, name)
    if quant_mod is None:
        return None

    device_quant = next(quant_layer.parameters()).device
    with torch.no_grad():
        try:
            x = ref_self_inp.to(device_quant).to(next(quant_mod.parameters()).dtype)
            if name == 'self_attn.kv_b_proj' and model_config:
                kv_lora = model_config.get('kv_lora_rank', 512)
                x = x[..., :kv_lora]
            if quant_ln is not None:
                x = quant_ln(x)
            quant_self_out = quant_mod(x)
        except Exception as e:
            return None
    if isinstance(quant_self_out, tuple):
        quant_self_out = quant_self_out[0]
    quant_self_out = quant_self_out.detach().cpu().float()
    return quant_self_out, device_quant


def _apply_post_ln_and_get_ref(quant_layer, name, quant_self_out, ref_sub, device_quant):
    """Apply post-LN if present, return (transformed_out, ref_rot_out) or None.

    For ops with a follow-up LN (e.g. wk → k_norm), the quant output is fed through
    the post-LN and compared against the post-LN's ref output.
    """
    post_ln_path = _MLA_SELFROTERR_POST_LN.get(name)
    if not post_ln_path:
        return quant_self_out, ref_sub.get(name)

    quant_post_ln = _resolve_path(quant_layer, post_ln_path)
    if quant_post_ln is not None:
        with torch.no_grad():
            try:
                quant_self_out = quant_post_ln(
                    quant_self_out.to(device_quant).to(next(quant_post_ln.parameters()).dtype)
                ).detach().cpu().float()
            except Exception:
                return None
    # Compare target: post-LN's ref output (not the linear's ref output)
    return quant_self_out, ref_sub.get(post_ln_path)


def _compute_single_selfroterr(
    name, quant_layer, ref_input, ref_sub, rot_mats, quant_desc, layer_idx, model_config,
):
    """Compute SelfRotErr for one inner subgraph; return (name, err) or None."""
    if quant_desc is not None:
        if _classify_subgraph_quant_type(quant_desc, layer_idx, name) == "FLOAT":
            return None

    ref_self_inp = _resolve_selfroterr_input(name, ref_input, ref_sub, rot_mats)
    if ref_self_inp is None:
        return None

    fwd_result = _forward_quant_inner_op(quant_layer, name, ref_self_inp, model_config)
    if fwd_result is None:
        return None
    quant_self_out, device_quant = fwd_result

    # --- post-LN: if this op has a follow-up LN (e.g. wk → k_norm) ---
    # Feed quant_self_out through the post-LN, compare with post-LN's ref output
    post_result = _apply_post_ln_and_get_ref(
        quant_layer, name, quant_self_out, ref_sub, device_quant,
    )
    if post_result is None:
        return None
    quant_self_out, ref_rot_out = post_result

    if ref_rot_out is None:
        return None
    ref_rot_out_f = ref_rot_out.detach().cpu().float()
    if quant_self_out.shape != ref_rot_out_f.shape:
        return None

    diff_norm = (quant_self_out - ref_rot_out_f).norm().item()
    ref_norm = ref_rot_out_f.norm().item()
    err = diff_norm / ref_norm if ref_norm > 0 else float('nan')
    return (name, err)


def _compute_selfroterr(
    subgraph_names, quant_layer, ref_input, ref_sub, rot_mats, quant_desc, layer_idx, model_config,
) -> Dict[str, float]:
    """Compute SelfRotErr for all inner subgraphs."""
    subgraph_selfroterr = {}
    inner_names_to_capture = [n for n in subgraph_names if n in _MLA_SELFROTERR_INPUT]
    if not inner_names_to_capture or not rot_mats:
        return subgraph_selfroterr
    for name in inner_names_to_capture:
        result = _compute_single_selfroterr(
            name, quant_layer, ref_input, ref_sub, rot_mats, quant_desc, layer_idx, model_config,
        )
        if result is not None:
            subgraph_selfroterr[result[0]] = result[1]
    return subgraph_selfroterr


def _compute_flip_rates(subgraph_names, ref_sub, quant_sub):
    """Compute indexer and experts routing flip rates. Returns (idx_flip, exp_flip)."""
    indexer_flip_rate = None
    experts_routing_flip = None

    indexer_name = next((name for name in (
        'self_attn.indexer', 'self_attn.compressor.indexer'
    ) if name in subgraph_names), None)
    if indexer_name is not None:
        ref_indexer_out = ref_sub.get(indexer_name)
        quant_indexer_out = quant_sub.get(indexer_name)
        if ref_indexer_out is not None and quant_indexer_out is not None:
            indexer_flip_rate = _compute_topk_flip_rate(ref_indexer_out, quant_indexer_out)

    gate_name = next(
        (name for name in ('mlp.gate', 'moe.gate',
                           'block_sparse_moe.gate')
         if name in subgraph_names),
        None,
    )
    if gate_name is not None:
        ref_gate_out = ref_sub.get(f"{gate_name}.__indices", ref_sub.get(gate_name))
        quant_gate_out = quant_sub.get(f"{gate_name}.__indices", quant_sub.get(gate_name))
        if ref_gate_out is not None and quant_gate_out is not None:
            experts_routing_flip = _compute_topk_flip_rate(ref_gate_out, quant_gate_out)

    return indexer_flip_rate, experts_routing_flip


def _patch_all_subgraphs(
    subgraph_names, quant_layer, quant_input, ref_sub, ref_out, base_l2, R1,
    layer_kwargs, unpatchable, quant_desc, layer_idx, mla_fine,
):
    """Patch each sub-graph and compute recovery. Returns (results, subgraph_quant_types)."""
    results = {}
    subgraph_quant_types = {}
    for name in subgraph_names:
        if mla_fine and name in _MLA_ATTN_INNER_SUBGRAPHS:
            results[name] = None  # skip patch, use RotBErr/SelfRotErr instead
            actual_qt = _classify_subgraph_quant_type(quant_desc, layer_idx, name) if quant_desc else "UNKNOWN"
            subgraph_quant_types[name] = actual_qt
            continue
        if mla_fine and name in _INDEXER_INNER_SUBGRAPHS:
            results[name] = None  # skip patch, use SelfRotErr instead
            actual_qt = _classify_subgraph_quant_type(quant_desc, layer_idx, name) if quant_desc else "UNKNOWN"
            subgraph_quant_types[name] = actual_qt
            continue
        if name in unpatchable:
            results[name] = None
            subgraph_quant_types[name] = "UNPATCHABLE"
            continue
        ref_output = ref_sub.get(name)
        if ref_output is None:
            results[name] = None
            continue

        patched_out = _patch_subgraph(
            quant_layer, quant_input, name, ref_output, layer_kwargs,
        ).cpu().float()

        if R1 is not None:
            patched_out = unrotate_hidden(patched_out, R1)
        p_l2 = rel_l2(patched_out, ref_out, [-1]).item()
        results[name] = recovery_ratio(base_l2, p_l2)

        subgraph_quant_types[name] = _classify_subgraph_quant_type(
            quant_desc, layer_idx, name,
        )
    return results, subgraph_quant_types


def _compute_chain_deltas(subgraph_selfroterr, results, mla_fine):
    """Compute chain delta recovery for attention serial chains."""
    chain_deltas = {}  # {chain_name: {op_name: delta_val, ...}}
    if not mla_fine:
        return chain_deltas
    for chain_name, ops in _MLA_CHAINS.items():
        chain_deltas[chain_name] = {}
        for op in ops:
            if op in subgraph_selfroterr:
                chain_deltas[chain_name][op] = subgraph_selfroterr[op]
            elif results.get(op) is not None:
                chain_deltas[chain_name][op] = results[op]
    return chain_deltas


def _register_branch_hooks(quant_layer, ref_vals):
    """Register ReplacementHooks for all ops in a branch.

    Returns (rhooks, handles) or (None, None) if any op cannot be resolved.
    On failure, any previously-registered handles are removed before returning.
    """
    rhooks = []
    handles = []
    for op, ref_val in ref_vals.items():
        mod = _resolve_path(quant_layer, op)
        if mod is None:
            # Clean up any hooks registered before the failure
            for h in handles:
                h.remove()
            return None, None
        rh = ReplacementHook(ref_val)
        h = mod.register_forward_hook(rh)
        rhooks.append(rh)
        handles.append(h)
    return rhooks, handles


def _run_branch_forward(quant_layer, quant_input, device_q, kwargs, rhooks, handles):
    """Run quant_layer forward with all branch hooks active; returns raw output."""
    try:
        for rh in rhooks:
            rh._active = True
        with torch.no_grad():
            out = quant_layer(quant_input.to(device_q), **kwargs)
        for rh in rhooks:
            rh._active = False
    finally:
        for h in handles:
            h.remove()
    return out


def _compute_branch_patches(
    quant_layer, quant_input, ref_sub, ref_out, base_l2, R1, layer_kwargs, mla_fine,
):
    """Compute branch patch recovery (full Q/KV path replacement)."""
    branch_patches = {}  # {branch_name: recovery_val}
    # Qwen3.6 可能没有 self_attn（用 linear_attn），且不是 MLA，跳过 branch patch
    has_self_attn = hasattr(quant_layer, 'self_attn')
    if not (mla_fine and has_self_attn and not hasattr(quant_layer.self_attn, 'indexer')):
        return branch_patches

    device_q = next(quant_layer.parameters()).device
    kwargs = _move_kwargs(layer_kwargs, device_q)
    for branch_name, ops in _MLA_BRANCH_PATCHES.items():
        # Check all ops have ref outputs and are patchable
        ref_vals = {}
        skip = False
        for op in ops:
            val = ref_sub.get(op)
            if val is None:
                skip = True
                break
            ref_vals[op] = val
        if skip:
            continue

        # Register hooks for all ops in this branch
        rhooks, handles = _register_branch_hooks(quant_layer, ref_vals)
        if rhooks is None:
            continue

        out = _run_branch_forward(quant_layer, quant_input, device_q, kwargs, rhooks, handles)

        out = (out[0] if isinstance(out, tuple) else out).detach().cpu().float()
        if R1 is not None:
            out = unrotate_hidden(out, R1)
        p_l2 = rel_l2(out, ref_out, [-1]).item()
        branch_patches[branch_name] = recovery_ratio(base_l2, p_l2)
    return branch_patches


def _find_root_suspect(
    subgraph_rotberr, subgraph_selfroterr, results, subgraph_quant_types,
    impact_boundary, indexer_flip_rate, mla_fine,
):
    """Find the root suspect op using priority-ordered heuristics."""
    coarse_names = {'self_attn'} if mla_fine else set()

    # Priority 0: if all SelfRotErrs are low (<10%) but boundary RotBErr is high (>10%),
    # the root cause is in an op WITHOUT SelfRotErr. Pick the op whose RotBErr ≈ boundary.
    if subgraph_rotberr and impact_boundary is not None:
        root_suspect = _find_root_suspect_priority0(
            subgraph_rotberr, subgraph_selfroterr, subgraph_quant_types,
            impact_boundary, coarse_names,
        )
        if root_suspect is not None:
            return root_suspect

    # Priority 1: highest SelfRotErr among attention-internal sub-graphs
    if subgraph_selfroterr:
        root_suspect = _find_root_suspect_by_selfroterr(
            subgraph_selfroterr, subgraph_quant_types, indexer_flip_rate, coarse_names,
        )
        if root_suspect is not None:
            return root_suspect

    # Priority 2: highest RotBErr for outer sub-graphs
    if subgraph_rotberr:
        root_suspect = _find_root_suspect_by_rotberr(
            subgraph_rotberr, subgraph_quant_types, coarse_names,
        )
        if root_suspect is not None:
            return root_suspect

    # Priority 3: last resort — use Recovery (lowest = most upstream problem)
    return _find_root_suspect_by_recovery(results, subgraph_quant_types, coarse_names)


def _find_root_suspect_priority0(
    subgraph_rotberr, subgraph_selfroterr, subgraph_quant_types,
    impact_boundary, coarse_names,
):
    """Priority 0: low SelfRotErr + high boundary RotBErr → op without SelfRotErr."""
    boundary_be = subgraph_rotberr.get(impact_boundary)
    max_serr = max(
        (v for v in subgraph_selfroterr.values() if v is not None),
        default=0,
    )
    if not (boundary_be is not None and max_serr < 0.10 and boundary_be > 0.10
            and impact_boundary in coarse_names):
        return None

    for op_name, op_be in subgraph_rotberr.items():
        if op_name in coarse_names or op_name == impact_boundary:
            continue
        if (op_be is not None and abs(op_be - boundary_be) < 0.02
                and subgraph_quant_types.get(op_name) != "FLOAT"
                and op_name not in _ROTBERR_SKIP_SUBGRAPHS
                and not op_name.endswith('.gate')):
            # op has no SelfRotErr (not in _MLA_SELFROTERR_INPUT)
            has_serr = subgraph_selfroterr.get(op_name) is not None
            if not has_serr:
                return op_name
    return None


def _find_root_suspect_by_selfroterr(
    subgraph_selfroterr, subgraph_quant_types, indexer_flip_rate, coarse_names,
):
    """Priority 1: highest SelfRotErr; exclude indexer internals if 0% flip."""
    non_float_serrs = {
        k: v for k, v in subgraph_selfroterr.items()
        if v is not None and subgraph_quant_types.get(k) != "FLOAT"
        and k not in coarse_names
    }
    if indexer_flip_rate is not None and indexer_flip_rate < 0.01:
        # Indexer has 0% flip — exclude indexer internals from root suspect
        non_float_serrs = {
            k: v for k, v in non_float_serrs.items()
            if k not in _INDEXER_INNER_SUBGRAPHS
        }
    if non_float_serrs:
        return max(non_float_serrs, key=non_float_serrs.get)
    return None


def _find_root_suspect_by_rotberr(subgraph_rotberr, subgraph_quant_types, coarse_names):
    """Priority 2: highest RotBErr for outer sub-graphs (skip gate, indexer, experts)."""
    non_gate_rotberr = {
        k: v for k, v in subgraph_rotberr.items()
        if v is not None and subgraph_quant_types.get(k) != "FLOAT"
        and k not in _ROTBERR_SKIP_SUBGRAPHS and not k.endswith('.gate')
        and k not in coarse_names
    }
    if non_gate_rotberr:
        return max(non_gate_rotberr, key=non_gate_rotberr.get)
    return None


def _find_root_suspect_by_recovery(results, subgraph_quant_types, coarse_names):
    """Priority 3: last resort — use Recovery (highest = least downstream correction)."""
    quantized_valid = {
        k: v for k, v in results.items()
        if v is not None and subgraph_quant_types.get(k) != "FLOAT"
        and not k.endswith('.gate')
    }
    if not quantized_valid:
        return None
    fine_valid = {k: v for k, v in quantized_valid.items() if k not in coarse_names}
    if fine_valid:
        return max(fine_valid, key=fine_valid.get)
    return max(quantized_valid, key=quantized_valid.get)


def _compute_impact_boundary(results, subgraph_quant_types):
    """Find the boundary with highest patch impact (downstream OK from here)."""
    quantized_valid = {
        k: v for k, v in results.items()
        if v is not None and subgraph_quant_types.get(k) != "FLOAT"
        and not k.endswith('.gate')
    }
    return max(quantized_valid, key=quantized_valid.get) if quantized_valid else None


def diagnose_layer(
    handle,
    model_type: str = 'auto',
    mla_fine: bool = True,
    R: Optional[torch.Tensor] = None,
    rot_mats: Optional[Dict[str, torch.Tensor]] = None,
    quant_desc: Optional[dict] = None,
    model_config: Optional[dict] = None,
) -> Dict[str, Any]:
    """诊断单层: baseline + 各子图 patch + recovery ranking.

    Args:
        handle: LayerReplayHandle
        model_type: 'auto' / 'dense' / 'moe' / 'glm_mla' / 'glm_moe_dsa' / 'qwen3_5_moe'
        mla_fine: 是否对 MLA attention 做细粒度拆分
        R: R1 rotation matrix (backwards compat)
        rot_mats: all rotation matrices dict {'rot': R1, 'rot_b_proj': R3, ...}
        quant_desc: quant_model_description.json dict, for FLOAT detection
        model_config: model config dict (for head structure in block rotation)

    Returns:
        diagnosis result dict
    """
    # Extract R1 from rot_mats or use standalone R
    if rot_mats is not None:
        R1 = rot_mats.get('rot')
    else:
        R1 = R

    ref_layer = handle.ref_layer
    quant_layer = handle.quant_layer
    ref_layer_kwargs = handle.ref_layer_kwargs or {}
    quant_layer_kwargs = handle.quant_layer_kwargs or {}

    # Detect model type
    if model_type == 'auto':
        model_type = detect_model_type(quant_layer)
    subgraph_names = get_subgraph_names(model_type, quant_layer, mla_fine=mla_fine)

    # --- Input ---
    # ref_hidden: original space (from V1 L1 cache)
    # quant_hidden: quant's natural space — rotated for QuaRot models (from V1 L1 cache)
    ref_input = handle.ref_hidden
    quant_input = handle.quant_hidden

    # Baseline — unrotate quant output if rotation model
    ref_out = handle.forward_ref(ref_input).detach().cpu().float()
    quant_out = handle.forward_quant(quant_input).detach().cpu().float()
    if R1 is not None:
        quant_out = unrotate_hidden(quant_out, R1)
    base_l2 = rel_l2(quant_out, ref_out, [-1]).item()

    if not torch.isfinite(torch.tensor(base_l2)):
        return {
            'layer_idx': handle.layer_idx,
            'baseline_l2': float('nan'),
            'subgraphs': {},
            'input_recovery': None,
            'dominant': 'unknown',
            'model_type': model_type,
        }

    # --- Input patch: feed ref_hidden (rotated) into quant layer ---
    input_recovery = _compute_input_recovery(handle, ref_input, ref_out, base_l2, R1)

    # Capture ref sub-graph outputs (in original space, from ref layer)
    ref_sub = _capture_subgraph_outputs(
        ref_layer, ref_input, subgraph_names, ref_layer_kwargs,
    )

    # Rotate ref sub-graph outputs so they match quant's internal (rotated) space.
    ref_sub, unpatchable = _rotate_ref_subgraph_outputs(
        ref_sub, rot_mats, R1, model_config,
    )

    # --- RotBErr / SelfRotErr / flip rates: only in mla_fine mode ---
    # These require an extra quant forward pass (for quant sub-graph outputs)
    # which costs NPU memory and is only meaningful for fine-grained attention
    # sub-graphs. In non-mla_fine mode, large sub-graph patch Recovery is sufficient.
    subgraph_rotberr = {}
    subgraph_selfroterr = {}
    indexer_flip_rate = None
    experts_routing_flip = None

    if mla_fine:
        # Capture quant sub-graph outputs (for RotBErr)
        quant_sub = _capture_subgraph_outputs(
            quant_layer, quant_input, subgraph_names, quant_layer_kwargs,
        )

        # --- RotBErr: rotation-aligned boundary error ---
        subgraph_rotberr = _compute_rotberr(
            subgraph_names, ref_sub, quant_sub, quant_desc, handle.layer_idx,
        )

        # --- SelfRotErr: controlled-input self quantization error ---
        subgraph_selfroterr = _compute_selfroterr(
            subgraph_names, quant_layer, ref_input, ref_sub, rot_mats,
            quant_desc, handle.layer_idx, model_config,
        )

        # --- Indexer / Experts flip rates ---
        indexer_flip_rate, experts_routing_flip = _compute_flip_rates(
            subgraph_names, ref_sub, quant_sub,
        )

    # Patch each sub-graph — in mla_fine mode, skip attention-internal sub-graphs
    # (Q-K mismatch makes Recovery meaningless). Use RotBErr/SelfRotErr instead.
    # Also skip indexer-internal sub-graphs (fp8_index is NPU custom op, partial patch invalid).
    # In non-mla_fine mode, all sub-graphs are large (self_attn, mlp.*) and patchable.
    results, subgraph_quant_types = _patch_all_subgraphs(
        subgraph_names, quant_layer, quant_input, ref_sub, ref_out, base_l2, R1,
        quant_layer_kwargs, unpatchable, quant_desc, handle.layer_idx, mla_fine,
    )

    # --- Chain Delta Recovery (串联子链增量分析) ---
    # 对 attention 内部串联子链，SelfRotErr 就是 delta 的等价物：
    # 每个op独立喂ref输入测自身误差，总和等于该链的累计误差。
    # 对 post-softmax (o_proj)，patch recovery 是可靠的，可以和 SelfRotErr 一起看。
    chain_deltas = _compute_chain_deltas(subgraph_selfroterr, results, mla_fine)

    # --- Branch Patch: Q/KV 全分支一起 patch ---
    # 仅对没有 indexer 的 MLA 模型有效。GLM-5 DSA 有 indexer，
    # indexer 用 quant 权重计算 top-k mask，即使 Q+KV 全替换
    # 也会因 indexer mask 不同导致 attention 分布错位 → 负 recovery。
    # 对 GLM-5 DSA，branch patch 无意义，跳过。
    branch_patches = _compute_branch_patches(
        quant_layer, quant_input, ref_sub, ref_out, base_l2, R1,
        quant_layer_kwargs, mla_fine,
    )

    # --- Impact Boundary: highest Recovery = "from this boundary onward, downstream is OK" ---
    # Recovery is Downstream Correctability, not root cause evidence.
    # High Recovery at a boundary means the chain AFTER that boundary can handle correct values,
    # so the problem is BEFORE or AT that boundary.
    impact_boundary = _compute_impact_boundary(results, subgraph_quant_types)

    # --- Root Suspect: the op most likely to be the root cause ---
    # Based on SelfRotErr (self quantization error) for attention internals,
    # RotBErr (boundary error including upstream) for outer sub-graphs,
    # and flip rates for discrete choices (gate, indexer).
    # SelfRotErr is strongest evidence because it isolates the op's own quant error.
    # However, indexer internals with high SelfRotErr but 0% flip rate are NOT
    # actually impacting output — deprioritize them.
    root_suspect = _find_root_suspect(
        subgraph_rotberr, subgraph_selfroterr, results, subgraph_quant_types,
        impact_boundary, indexer_flip_rate, mla_fine,
    )

    return {
        'layer_idx': handle.layer_idx,
        'baseline_l2': base_l2,
        'subgraphs': results,
        'subgraph_quant_types': subgraph_quant_types,
        'subgraph_rotberr': subgraph_rotberr,
        'subgraph_selfroterr': subgraph_selfroterr,
        'chain_deltas': chain_deltas,
        'branch_patches': branch_patches,
        'indexer_flip_rate': indexer_flip_rate,
        'experts_routing_flip': experts_routing_flip,
        'input_recovery': input_recovery,
        'impact_boundary': impact_boundary,
        'root_suspect': root_suspect,
        'model_type': model_type,
    }


# 多层批量诊断 (从 V1 L1 cache 加载 hidden_states)

def diagnose_layers(
    ref_model_path: str,
    quant_model_path: str,
    candidate_layers: List[int],
    prompt: str = "你好，今天天气怎么样？",
    quant_method: str = "dequantize",
    ref_device: str = "npu:0",
    quant_device: str = "npu:1",
    model_type: str = "auto",
    mla_fine: bool = True,
    rotation_matrix: str = None,
) -> List[Dict[str, Any]]:
    """对多个候选层批量跑 subgraph 诊断.

    从 V1 L1 cache 读取 hidden_states，不需要自己 forward prefix 层。
    Cache 由 V1 L1 的 --cache_top_k / --auto_cache_bad_layer 生成。

    Args:
        ref_model_path: ref 模型路径
        quant_model_path: quant 模型路径
        candidate_layers: V1 L1 给出的候选层列表
        prompt: 输入文本 (必须和 V1 L1 跑的时候一致，用于 cache key)
        quant_method: 量化方法 (必须和 V1 L1 一致，如 "dequantize")
        ref_device: ref 设备
        quant_device: quant 设备
        model_type: 'auto' / 'dense' / 'moe' / 'glm_mla' / 'glm_moe_dsa' / 'qwen3_5_moe'
        mla_fine: 是否对 MLA attention 做细粒度拆分
        rotation_matrix: 旋转矩阵路径 (.pt 或 .safetensors), 用于 QuaRot 旋转模型

    Returns:
        每层诊断结果列表
    """
    # Load rotation matrices if provided
    rot_mats = load_all_rotation_matrices(rotation_matrix) if rotation_matrix else None
    if rot_mats is not None:
        R = rot_mats.get('rot')
        n_mats = len(rot_mats)
        logger.info(f"Loaded {n_mats} rotation matrix(es), will use correct R per sub-graph")
    else:
        R = None

    # Load quant description for FLOAT detection
    quant_desc = _load_quant_desc(quant_model_path)
    if quant_desc is not None:
        logger.info(f"Loaded quant_model_description.json for FLOAT detection")

    provider = ReplayProvider(
        ref_model_path, quant_model_path,
        dtype=torch.bfloat16, verbose=True,
    )

    try:
        # Extract model config for block rotation
        model_config = {}
        if hasattr(provider.ref_model, 'config'):
            cfg = get_text_config(provider.ref_model.config)
            for key in ['qk_nope_head_dim', 'qk_rope_head_dim', 'v_head_dim',
                         'num_key_value_heads', 'kv_lora_rank', 'q_lora_rank',
                         'num_attention_heads', 'head_dim', 'hidden_size']:
                if hasattr(cfg, key):
                    model_config[key] = getattr(cfg, key)

        all_results = []
        for layer_idx in candidate_layers:
            logger.info(f"\nDiagnosing layer {layer_idx}...")

            # 从 V1 L1 cache 加载 hidden_states
            ref_cache = _load_l1_cache(
                ref_model_path, prompt, layer_idx, "ref", quant_method, ref_device)
            quant_cache = _load_l1_cache(
                quant_model_path, prompt, layer_idx, "quant", quant_method, quant_device)
            ref_hidden, ref_layer_state = _unpack_l1_cache_entry(ref_cache)
            quant_hidden, quant_layer_state = _unpack_l1_cache_entry(quant_cache)
            cached_input_ids = (
                ref_cache.get("input_ids") if isinstance(ref_cache, dict) else None
            )

            if ref_hidden is None or quant_hidden is None:
                logger.info(f"  [SKIP] Layer {layer_idx}: no V1 L1 cache found. "
                      f"Run V1 L1 with --cache_top_k or --auto_cache_bad_layer first.")
                continue

            # Validate hidden_size matches the model config
            # ForConditionalGeneration (如 Qwen3.6): hidden_size 在 text_config 里
            config = get_text_config(provider.ref_model.config)
            expected_hidden = getattr(config, 'hidden_size', None)
            if expected_hidden is None:
                logger.warning(f"  [SKIP] Layer {layer_idx}: cannot find hidden_size in config")
                continue
            ref_hidden_size = ref_hidden.shape[-1]
            if ref_hidden_size != expected_hidden:
                logger.info(f"  [SKIP] Layer {layer_idx}: cache hidden_size={ref_hidden_size} "
                      f"!= model hidden_size={expected_hidden}. Cache is from a different model.")
                continue

            seqlen = ref_hidden.shape[1]
            logger.info(f"  [CACHE] Loaded: ref shape={list(ref_hidden.shape)}, quant shape={list(quant_hidden.shape)}")

            # 获取 handle (只加载目标层权重，hidden_states 从 cache 读)
            handle = provider.get_layer_handle(
                layer_idx=layer_idx,
                device=ref_device,
                quant_device=quant_device,
                ref_hidden_override=ref_hidden,
                quant_hidden_override=quant_hidden,
                ref_layer_state_override=ref_layer_state,
                quant_layer_state_override=quant_layer_state,
                input_ids_override=cached_input_ids,
            )

            r = diagnose_layer(handle, model_type=model_type, mla_fine=mla_fine, R=R, rot_mats=rot_mats, quant_desc=quant_desc, model_config=model_config)
            all_results.append(r)

            handle.cleanup()

        return all_results
    finally:
        provider.close()


# ============================================================================


def _short_name(name: str) -> str:
    """Shorten a subgraph name for compact column headers."""
    s = name.replace('self_attn', 'attn').replace('moe.', '').replace('mlp.', '')
    s = s.replace('indexer.', 'idx.')
    return s


def _collect_all_subgraphs(results: List[Dict[str, Any]]) -> List[str]:
    """Collect the union of all subgraph names across results, preserving order."""
    all_subgraphs = []
    for r in results:
        for name in r.get('subgraphs', {}):
            if name not in all_subgraphs:
                all_subgraphs.append(name)
    return all_subgraphs


def _format_patch_impact_cell(name, val, qt, r, is_mla_fine,
                              is_inner, is_idx_inner) -> str:
    """Format one cell of the Patch Impact table for a single (result, subgraph)."""
    if is_mla_fine and is_inner:
        if qt == "FLOAT":
            return f" | {'FLOAT†':>12}"
        rb = r.get('subgraph_rotberr', {}).get(name)
        if rb is not None:
            return f" | {rb*100:>11.1f}⚠"
        return f" | {'N/A⚠':>12}"
    if is_mla_fine and is_idx_inner:
        if qt == "FLOAT":
            return f" | {'FLOAT†':>12}"
        se = r.get('subgraph_selfroterr', {}).get(name)
        if se is not None:
            return f" | {se*100:>11.1f}⚠"
        return f" | {'N/A⚠':>12}"
    if name == 'self_attn.indexer':
        idx_flip = r.get('indexer_flip_rate')
        if idx_flip is not None:
            return f" | {idx_flip*100:>10.1f}⚠"
        return f" | {'N/A':>12}"
    if val is not None:
        if qt == "FLOAT":
            return f" | {val*100:>10.1f}†"
        return f" | {val*100:>11.1f}%"
    if qt == "UNPATCHABLE":
        return f" | {'SKIP':>12}"
    return f" | {'N/A':>12}"


def _print_patch_impact_section(results, all_subgraphs, is_mla_fine):
    """Section 1: Patch Impact (Boundary Cut Diagnosis) table."""
    logger.info(f"--- Patch Impact (Boundary Cut Diagnosis) ---")
    logger.info(f"  High Recovery = downstream is OK, problem is before/at this boundary")
    header = f"{'Layer':>5} | {'Base L2':>9} | {'Type':>8} | {'Input Rec':>9}"
    for name in all_subgraphs:
        s = _short_name(name)
        if is_mla_fine and name in _MLA_ATTN_INNER_SUBGRAPHS:
            header += f" | {s + ' RotBE':>12}"
        elif is_mla_fine and name in _INDEXER_INNER_SUBGRAPHS:
            header += f" | {s + ' SErr':>12}"
        elif name == 'self_attn.indexer':
            header += f" | {'idx Flip':>12}"
        else:
            header += f" | {s + ' Impact':>12}"
    header += f" | {'ImpactBnd':>12}"
    logger.info(header)
    logger.info("-" * len(header))

    for r in results:
        input_rec = r.get('input_recovery')
        input_rec_str = f"{input_rec*100:>8.1f}%" if input_rec is not None else "     N/A"
        row = f"{r['layer_idx']:>5} | {r['baseline_l2']:>9.6f} | {r['model_type']:>8} | {input_rec_str}"
        qtypes = r.get('subgraph_quant_types', {})
        for name in all_subgraphs:
            val = r['subgraphs'].get(name)
            qt = qtypes.get(name, "")
            is_inner = name in _MLA_ATTN_INNER_SUBGRAPHS
            is_idx_inner = name in _INDEXER_INNER_SUBGRAPHS
            row += _format_patch_impact_cell(name, val, qt, r, is_mla_fine, is_inner, is_idx_inner)
        ib = r.get('impact_boundary')
        row += f" | {_short_name(ib) if ib else 'N/A':>12}"
        logger.info(row)


def _print_selfroterr_section(results, all_subgraphs, is_mla_fine):
    """Section 2: SelfRotErr (Op-Level Quantization Error), only when mla_fine."""
    if not is_mla_fine:
        return
    has_selfroterr = any(r.get('subgraph_selfroterr') for r in results)
    has_inner_float = any(
        r.get('subgraph_quant_types', {}).get(n) == "FLOAT"
        for r in results for n in all_subgraphs
        if n in _MLA_ATTN_INNER_SUBGRAPHS or n in _INDEXER_INNER_SUBGRAPHS
    )
    if not (has_selfroterr or has_inner_float):
        return

    inner_names = [n for n in all_subgraphs
                   if n in _MLA_ATTN_INNER_SUBGRAPHS or n in _INDEXER_INNER_SUBGRAPHS]
    logger.info(f"\n--- SelfRotErr (Op-Level Quantization Error) ---")
    logger.info(f"  Isolates each op's own quant error from upstream propagation")
    header2 = f"{'Layer':>5}"
    for name in inner_names:
        header2 += f" | {_short_name(name) + ' SErr':>12}"
    logger.info(header2)
    logger.info("-" * len(header2))

    for r in results:
        serrs = r.get('subgraph_selfroterr', {})
        qtypes = r.get('subgraph_quant_types', {})
        row = f"{r['layer_idx']:>5}"
        for name in inner_names:
            if qtypes.get(name) == "FLOAT":
                row += f" | {'FLOAT†':>12}"
            else:
                se = serrs.get(name)
                if se is not None:
                    row += f" | {se*100:>11.1f}%"
                else:
                    row += f" | {'N/A':>12}"
        logger.info(row)


def _print_flip_rates_section(results):
    """Section 3: Flip Rate (Discrete Choice Divergence)."""
    has_special = any(
        r.get('indexer_flip_rate') is not None or r.get('experts_routing_flip') is not None
        for r in results
    )
    if not has_special:
        return
    logger.info(f"\n--- Flip Rate (Discrete Choice Divergence) ---")
    logger.info(f"  Note: gate is FLOAT — flip reflects input divergence, not gate quantization error")
    logger.info(f"{'Layer':>5} | {'Indexer Flip':>12} | {'Experts Flip':>12}")
    logger.info("-" * 38)
    for r in results:
        idx_flip = r.get('indexer_flip_rate')
        exp_flip = r.get('experts_routing_flip')
        idx_str = f"{idx_flip*100:>10.1f}%" if idx_flip is not None else "N/A"
        exp_str = f"{exp_flip*100:>10.1f}%" if exp_flip is not None else "N/A"
        logger.info(f"{r['layer_idx']:>5} | {idx_str:>12} | {exp_str:>12}")


def _print_rotberr_section(results, all_subgraphs, is_mla_fine):
    """Section 4: RotBErr (Boundary Error, includes upstream propagation)."""
    if not is_mla_fine:
        return
    has_rotberr = any(r.get('subgraph_rotberr') for r in results)
    if not has_rotberr:
        return
    logger.info(f"\n--- RotBErr (Boundary Error, includes upstream propagation) ---")
    header3 = f"{'Layer':>5}"
    for name in all_subgraphs:
        if name not in _ROTBERR_SKIP_SUBGRAPHS:
            header3 += f" | {_short_name(name) + ' BErr':>12}"
    logger.info(header3)
    logger.info("-" * len(header3))

    for r in results:
        rotberrs = r.get('subgraph_rotberr', {})
        if not rotberrs:
            continue
        row = f"{r['layer_idx']:>5}"
        for name in all_subgraphs:
            if name in _ROTBERR_SKIP_SUBGRAPHS:
                continue
            rb = rotberrs.get(name)
            if rb is not None:
                row += f" | {rb*100:>11.1f}%"
            else:
                row += f" | {'N/A':>12}"
        logger.info(row)


def _print_chain_delta_section(results):
    """Section 5a: Chain Delta Recovery (Attention 串联子链自误差)."""
    has_chain = any(r.get('chain_deltas') for r in results)
    if not has_chain:
        return
    logger.info(f"\n--- Chain Delta Recovery (Attention 串联子链自误差) ---")
    logger.info(f"  SelfRotErr per op = independent quant error ≈ delta in serial chain")
    for r in results:
        chains = r.get('chain_deltas', {})
        if not chains:
            continue
        logger.info(f"  Layer {r['layer_idx']}:")
        for chain_name, ops in chains.items():
            if not ops:
                continue
            parts = []
            max_op = None
            max_val = -1
            for op, val in ops.items():
                parts.append(f"{_short_name(op)}={val*100:.1f}%")
                if val is not None and val > max_val:
                    max_val = val
                    max_op = op
            total = sum(v for v in ops.values() if v is not None)
            logger.info(f"    {chain_name}: {' → '.join(parts)}  (total={total*100:.1f}%)")
            if max_op:
                logger.info(f"      largest delta: {_short_name(max_op)} ({max_val*100:.1f}%)")


def _print_branch_patch_section(results):
    """Section 5b: Branch Patch Recovery (全分支替换)."""
    has_branch = any(r.get('branch_patches') for r in results)
    if not has_branch:
        return
    logger.info(f"\n--- Branch Patch Recovery (全分支替换) ---")
    logger.info(f"  Negative = patched branch conflicts with unpatched branch (softmax coupling)")
    logger.info(f"  Negative KV/Q patch confirms: Q-K relative error is the attention bottleneck")
    logger.info(f"{'Layer':>5} | {'Q_path':>8} | {'KV_path':>8} | {'Q+KV_all':>9}")
    logger.info("-" * 40)
    for r in results:
        bp = r.get('branch_patches', {})
        if not bp:
            continue
        q_val = bp.get('Q_path')
        kv_val = bp.get('KV_path')
        qkv_val = bp.get('Q+KV_all')
        q_str = f"{q_val*100:>6.1f}%" if q_val is not None else "N/A"
        kv_str = f"{kv_val*100:>6.1f}%" if kv_val is not None else "N/A"
        qkv_str = f"{qkv_val*100:>7.1f}%" if qkv_val is not None else "N/A"
        logger.info(f"{r['layer_idx']:>5} | {q_str:>8} | {kv_str:>8} | {qkv_str:>9}")


def _print_branch_interpretation(bp):
    """Step 5 of Diagnosis Summary: branch patch interpretation for one layer."""
    q_val = bp.get('Q_path')
    kv_val = bp.get('KV_path')
    qkv_val = bp.get('Q+KV_all')
    if q_val is not None and kv_val is not None:
        if kv_val < 0:
            logger.info(f"    Branch: KV patch negative ({kv_val*100:.1f}%) — Q-K mismatch confirms softmax coupling")
            if q_val > 0:
                logger.info(f"      Q patch alone positive ({q_val*100:.1f}%) — Q error is smaller, KV patch breaks Q-K balance")
        elif q_val < 0:
            logger.info(f"    Branch: Q patch negative ({q_val*100:.1f}%) — Q-K mismatch confirms softmax coupling")
        elif q_val > kv_val * 1.5:
            logger.info(f"    Branch: Q path dominates (Q={q_val*100:.1f}% > KV={kv_val*100:.1f}%)")
        elif kv_val > q_val * 1.5:
            logger.info(f"    Branch: KV path dominates (KV={kv_val*100:.1f}% > Q={q_val*100:.1f}%)")
        else:
            logger.info(f"    Branch: Q & KV comparable (Q={q_val*100:.1f}%, KV={kv_val*100:.1f}%)")
    if qkv_val is not None and qkv_val < 0:
        logger.info(f"    Q+KV full patch: {qkv_val*100:.1f}% — negative = Q-K relative error dominates")
    elif qkv_val is not None:
        logger.info(f"    Q+KV full patch: {qkv_val*100:.1f}% (vs o_proj patch for comparison)")


def _print_root_suspect_line(r, rs):
    """Step 2 of Diagnosis Summary: root suspect interpretation for one layer."""
    if not rs:
        return
    rs_qt = r.get('subgraph_quant_types', {}).get(rs, "")
    if rs in (r.get('subgraph_selfroterr') or {}):
        se_val = r['subgraph_selfroterr'][rs]
        logger.info(f"    Root Suspect: {_short_name(rs)} (SelfRotErr={se_val*100:.1f}%)")
    elif rs in (r.get('subgraph_rotberr') or {}):
        rb_val = r['subgraph_rotberr'][rs]
        logger.info(f"    Root Suspect: {_short_name(rs)} (RotBErr={rb_val*100:.1f}%)")
    elif rs_qt:
        logger.info(f"    Root Suspect: {_short_name(rs)} (quant_type={rs_qt})")


def _print_flip_alerts(r):
    """Step 4 of Diagnosis Summary: flip alerts for indexer and experts."""
    idx_flip = r.get('indexer_flip_rate')
    exp_flip = r.get('experts_routing_flip')
    if idx_flip is not None and idx_flip > 0.1:
        logger.info(f"    Indexer Flip: {idx_flip*100:.1f}% — sparse token selection diverges")
        chains = r.get('chain_deltas', {})
        idx_chains = {k: v for k, v in chains.items() if k.startswith('Indexer_')}
        if idx_chains:
            for cname, cops in idx_chains.items():
                if cops:
                    max_op = max(cops, key=cops.get) if cops else None
                    if max_op:
                        logger.info(f"      {cname}: {_short_name(max_op)} SelfRotErr={cops[max_op]*100:.1f}% (highest)")
    if exp_flip is not None and exp_flip > 0.1:
        logger.info(f"    Experts Flip: {exp_flip*100:.1f}% — routing diverges (gate is FLOAT, reflects input error)")


def _print_layer_diagnosis(r):
    """Section 6: Diagnosis Summary for a single layer."""
    idx = r['layer_idx']
    ib = r.get('impact_boundary')
    rs = r.get('root_suspect')
    input_rec = r.get('input_recovery')

    logger.info(f"  Layer {idx}:")

    # Step 1: Impact boundary interpretation
    if ib:
        ib_val = r['subgraphs'].get(ib)
        if ib_val is not None:
            logger.info(f"    Impact Boundary: {_short_name(ib)} (Recovery={ib_val*100:.1f}%)")
            logger.info(f"      → downstream from {_short_name(ib)} is OK, problem is before/at this boundary")

    # Step 2: Root suspect
    _print_root_suspect_line(r, rs)

    # Step 3: Input pollution
    if input_rec is not None and input_rec > 0.3:
        logger.info(f"    Input Recovery: {input_rec*100:.1f}% — upstream error is significant")

    # Step 4: Flip alerts
    _print_flip_alerts(r)

    # Step 5: Branch patch interpretation
    bp = r.get('branch_patches', {})
    if bp:
        _print_branch_interpretation(bp)


def _print_interpretation_footer(is_mla_fine):
    """Print the legend / interpretation footer."""
    logger.info(f"\nInterpretation:")
    logger.info(f"  Impact:     Downstream Correctability — high = downstream OK, problem before boundary")
    logger.info(f"  ⚠ RotBE:   RotBErr for ATTN INNER sub-graphs (boundary error with upstream)")
    logger.info(f"  ⚠ SErr:    SelfRotErr — op's own quant error, isolated from upstream (= chain delta)")
    logger.info(f"  idx Flip:   Indexer top-k selection divergence rate (symmetric_difference ratio)")
    logger.info(f"  †:          FLOAT (unquantized) subgraph")
    logger.info(f"  ImpactBnd:  boundary with highest patch impact (downstream is OK from here)")
    logger.info(f"  RootSuspect: op most likely to be root cause (based on SelfRotErr > RotBErr > Impact)")
    logger.info(f"  ChainDelta: SelfRotErr along serial chains (Q/KV/Indexer), largest delta = biggest error source")
    logger.info(f"  Indexer chains: Indexer_Q=idx.wq_b, Indexer_K=idx.wk→k_norm, Indexer_W=idx.weights_proj")
    logger.info(f"  BranchPatch: full Q/KV path replacement → which branch matters more")
    if not is_mla_fine:
        logger.info(f"\n  Use --mla_fine for op-level SelfRotErr + Chain Delta + Branch Patch diagnosis.")


def print_report(results: List[Dict[str, Any]]):
    """打印 subgraph 诊断报告.

    分层诊断逻辑:
      1. Patch Recovery — 切点排查: 高 Recovery = 从该边界往后没问题, 问题在边界前
      2. SelfRotErr — 精确定位: 哪个 op 自身量化误差最大
      3. RotBErr — 边界误差: 真实链路下哪个边界偏得大
      4. Flip Rate — 离散选择: routing/indexer 有没有翻
      5. Input Rec — 上游污染: 输入本身已经偏了多少
    """
    if not results:
        logger.info("No results.")
        return

    # 收集所有子图名
    all_subgraphs = _collect_all_subgraphs(results)

    # Detect mode: mla_fine if any inner sub-graphs present
    is_mla_fine = any(n in _MLA_ATTN_INNER_SUBGRAPHS or n in _INDEXER_INNER_SUBGRAPHS
                      for n in all_subgraphs)

    # --- Section 1: Patch Impact (切点排查) ---
    _print_patch_impact_section(results, all_subgraphs, is_mla_fine)

    # --- Section 2: SelfRotErr (精确定位, only mla_fine) ---
    _print_selfroterr_section(results, all_subgraphs, is_mla_fine)

    # --- Section 3: Flip Rates (离散选择偏差) ---
    _print_flip_rates_section(results)

    # --- Section 4: RotBErr (边界误差, only mla_fine) ---
    _print_rotberr_section(results, all_subgraphs, is_mla_fine)

    # --- Section 5: Chain Delta & Branch Patch (串联子链增量 + 分支定位) ---
    if is_mla_fine:
        _print_chain_delta_section(results)
        _print_branch_patch_section(results)

    # --- Section 6: Diagnosis Summary ---
    logger.info(f"\n--- Diagnosis ---")
    for r in results:
        _print_layer_diagnosis(r)

    _print_interpretation_footer(is_mla_fine)
