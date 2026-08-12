"""
通用工具函数
"""

import torch
import torch.nn as nn

from .model_structure import get_model_components
from torch import Tensor
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Tensor 操作
# ============================================================================

def to_cpu_fp32(t: Tensor) -> Tensor:
    """tensor -> CPU float32，统一入口，避免设备/精度干扰对比结果"""
    return t.detach().cpu().float()


def flatten_for_compare(a: Tensor, b: Tensor, use_cpu: bool = True) -> Tuple[Tensor, Tensor]:
    """两个tensor flatten后对齐，用于逐元素对比"""
    if use_cpu:
        a, b = to_cpu_fp32(a), to_cpu_fp32(b)
    else:
        a, b = a.detach().float(), b.detach().float()
    return a.flatten(), b.flatten()


# ============================================================================
# 模型结构探测
# ============================================================================

def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """兼容 CausalLM、多模态 wrapper 和 Kimi K3，获取 decoder layers。"""
    if isinstance(model, nn.ModuleList):
        return model
    return get_model_components(model).layers


def get_embed_module(model: nn.Module) -> Optional[nn.Module]:
    """获取embedding模块"""
    return get_model_components(model).embed


def get_norm_module(model: nn.Module) -> Optional[nn.Module]:
    """获取final layer norm模块"""
    return get_model_components(model).final_norm


def get_lm_head_module(model: nn.Module) -> Optional[nn.Module]:
    """获取lm_head模块"""
    return get_model_components(model).lm_head


def get_rotary_emb_module(model: nn.Module) -> Optional[nn.Module]:
    """获取rotary embedding模块"""
    return get_model_components(model).rotary_emb


def get_num_layers(model: nn.Module) -> int:
    """获取模型层数"""
    try:
        return len(get_decoder_layers(model))
    except ValueError:
        if hasattr(model, 'config'):
            return getattr(model.config, 'num_hidden_layers', 0)
        return 0


# ============================================================================
# Key / 名称工具
# ============================================================================

def parse_base_name(weight_key: str) -> str:
    """从权重key中提取base_name

    "model.layers.0.self_attn.q_proj.weight" -> "model.layers.0.self_attn.q_proj"
    "model.layers.0.self_attn.q_proj.weight_scale" -> "model.layers.0.self_attn.q_proj"
    """
    quant_suffixes = [
        ".weight_scale", ".weight_offset",
        ".deq_scale", ".input_scale", ".input_offset", ".quant_bias",
        ".kv_cache_scale", ".kv_cache_offset",
        ".scale_bias",
        ".weight",
    ]
    for suffix in quant_suffixes:
        if weight_key.endswith(suffix):
            return weight_key[:-len(suffix)]
    return weight_key


_QUANT_TYPE_ALIASES = {
    # msModelSlim A4 exports use both spellings for the same packed INT4
    # dynamic format. Keep one internal name across indexed and streaming paths.
    "W4A4_INT4_DYNAMIC": "W4A4_DYNAMIC",
}


def normalize_quant_type(quant_type: str) -> str:
    """Return the canonical internal spelling for a checkpoint quant type."""
    if not isinstance(quant_type, str):
        return quant_type
    normalized = quant_type.strip()
    return _QUANT_TYPE_ALIASES.get(normalized.upper(), normalized)


def normalize_quant_desc_values(quant_desc: dict) -> dict:
    """Canonicalize string quant types while preserving metadata values."""
    if not quant_desc:
        return quant_desc
    return {
        key: normalize_quant_type(value) if isinstance(value, str) else value
        for key, value in quant_desc.items()
    }


def normalize_quant_desc_keys(quant_desc: dict, model) -> dict:
    """修正 quant_desc key 与 model.named_modules() 的前缀不匹配

    msmodelslim 的 quant_model_description.json key 前缀:
      - ForConditionalGeneration: "model.language_model.layers..."
      - ForCausalLM: "model.layers..."

    model.named_modules() 的前缀:
      - ForConditionalGeneration: "model.model.language_model.layers..."
      - ForCausalLM: "model.model.layers..."
    """
    if not quant_desc:
        return quant_desc

    module_names = tuple(name for name, _ in model.named_modules())
    has_nested_language_model = any(
        name.endswith("language_model") or ".language_model." in name
        for name in module_names
    )

    new_desc = {}
    for key, value in quant_desc.items():
        if not isinstance(value, str):
            new_desc[key] = value
            continue
        value = normalize_quant_type(value)

        if has_nested_language_model and key.startswith("model.language_model."):
            new_key = "model.model." + key[len("model."):]
            new_desc[new_key] = value
            new_desc[key] = value
        elif not has_nested_language_model and key.startswith("model.layers."):
            new_key = "model.model." + key[len("model."):]
            new_desc[new_key] = value
            new_desc[key] = value
        else:
            new_desc[key] = value

    return new_desc


# ============================================================================
# 设备/精度工具
# ============================================================================

def clear_device_cache(devices: List[str] = None):
    """清空 NPU/CUDA 显存缓存"""
    import gc
    gc.collect()
    if hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def auto_device() -> str:
    """自动选择可用设备"""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return "npu:0"
    return "cpu"


DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def parse_dtype(dtype_str: str) -> torch.dtype:
    """字符串转torch.dtype"""
    return DTYPE_MAP.get(dtype_str, torch.bfloat16)


# ============================================================================
# Rotation matrix
# ============================================================================

def _check_orthogonal(R: torch.Tensor, tol: float = 0.05) -> bool:
    """Check if R is approximately orthogonal: R @ R^T ≈ I.

    For large matrices (>2048), checks row norms (diagonal) and a sample
    of off-diagonal dot products. Full R@R^T would OOM for 6144x6144.

    Always runs on CPU to avoid NPU/CUDA device mismatch with default device.
    """
    if R.dim() != 2 or R.shape[0] != R.shape[1]:
        return False
    # Force CPU: rotation matrix is small, and default device may be NPU
    # causing torch.eye(n) (CPU) vs R_f @ R_f.t() (NPU) mismatch.
    R_f = R.float().cpu()
    n = R.shape[0]

    if n <= 2048:
        # Small enough for full check
        prod = R_f @ R_f.t()
        identity = torch.eye(n)
        diff = (prod - identity).abs().max().item()
    else:
        # Large matrix: check row norms + sampled off-diagonal
        # Row norms (diagonal of R@R^T): should all be ≈ 1
        row_norms_sq = (R_f * R_f).sum(dim=1)
        diag_diff = (row_norms_sq - 1.0).abs().max().item()

        # Sample 100 random off-diagonal pairs: R[i] @ R[j] should be ≈ 0
        idx = torch.randperm(n)[:100]
        R_sub = R_f[idx]  # [100, n]
        off_prod = R_sub @ R_sub.t()  # [100, 100]
        off_mask = ~torch.eye(100, dtype=torch.bool)
        off_diff = off_prod[off_mask].abs().max().item()

        diff = max(diag_diff, off_diff)

    if diff > tol:
        logger.warning(
            f"Rotation matrix FAILED orthogonality check: max_diff={diff:.6f} (tol={tol}). "
            f"This is NOT a valid rotation matrix — using it will produce garbage results. "
            f"shape={R.shape}, dtype={R.dtype}, min={R.min():.6f}, max={R.max():.6f}"
        )
        return False
    logger.info(f"Rotation matrix orthogonality check PASSED: max_diff={diff:.6f}")
    return True


def load_rotation_matrix(path: str) -> Optional[torch.Tensor]:
    """Load rotation matrix from .pt or .safetensors file.

    Supports:
      - .pt file with key R1 (V3.2 style)
      - .safetensors with key global_rotation (GLM5/QuaRot style)
    Returns None if file not found, key missing, or matrix fails orthogonality check.
    """
    import os
    if not path or not os.path.exists(path):
        return None

    def _try_load(R, source_desc):
        """Validate and return R, or None if not a valid rotation matrix."""
        if R is None:
            return None
        if not _check_orthogonal(R):
            logger.error(
                f"Loaded tensor from {source_desc} is NOT orthogonal — refusing to use it as rotation matrix. "
                f"Check that you are using optional/quarot.safetensors (key=global_rotation), "
                f"NOT rot.safetensors (key=rot.weight, which is NOT a rotation matrix)."
            )
            return None
        return R

    if path.endswith(".safetensors"):
        from safetensors.torch import safe_open
        sf = safe_open(path, framework="pt", device="cpu")
        for key in ["global_rotation", "R1", "rot.weight", "rot"]:
            if key in sf.keys():
                R = sf.get_tensor(key)
                logger.info(f"Loaded rotation matrix from {path} key={key}: shape={R.shape}, dtype={R.dtype}")
                return _try_load(R, f"{path} key={key}")
        logger.warning(f"No rotation key (global_rotation/R1/rot.weight) found in {path}, keys={list(sf.keys())}")
        return None
    else:
        data = torch.load(path, weights_only=True)
        for key in ["R1", "global_rotation", "rot.weight", "rot"]:
            if key in data:
                R = data[key]
                logger.info(f"Loaded rotation matrix from {path} key={key}: shape={R.shape}, dtype={R.dtype}")
                return _try_load(R, f"{path} key={key}")
        logger.warning(f"No rotation key (R1/global_rotation/rot.weight) found in {path}, keys={list(data.keys())}")
        return None


def unrotate_hidden(hidden: torch.Tensor, R: torch.Tensor, device: str = None) -> torch.Tensor:
    """Apply R^T to quant hidden states to align with ref representation space.

    quant_hidden @ R^T -> aligned (same space as ref)
    """
    if R is None:
        return hidden
    target_device = device or hidden.device
    R_dev = R.to(target_device)
    result = hidden.float() @ R_dev.t().float()
    return result.to(hidden.dtype)


def rotate_hidden(hidden: torch.Tensor, R: torch.Tensor, device: str = None) -> torch.Tensor:
    """Apply R to hidden states: original space -> rotated space.

    hidden @ R -> rotated (same space as quant model internals)
    Inverse of unrotate_hidden (which applies R^T).
    """
    if R is None:
        return hidden
    target_device = device or hidden.device
    R_dev = R.to(target_device)
    result = hidden.float() @ R_dev.float()
    return result.to(hidden.dtype)
