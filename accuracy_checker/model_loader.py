"""
量化模型加载器

借鉴 msmodelslim_inference_large.py 的方案:
  1. symlink trick: 让 transformers 能识别 msmodelslim 的权重文件
  2. AutoModelForCausalLM.from_pretrained() 加载模型结构 + INT8权重
  3. 逐层反量化: w_fp = (w_int8 - offset) * scale, 写回 module.weight

支持的量化类型:
"""

import builtins
import os
import json
import gc
import sys
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from safetensors.torch import load_file
from safetensors import safe_open

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

from .utils import parse_base_name, normalize_quant_desc_keys, normalize_quant_type
import logging

logger = logging.getLogger(__name__)
_INFERRED_QUANT_WARNINGS = set()
_DEEPSEEK_V4_FP4_MARKER = "__acc_deepseek_v4_fp4__"
_NATIVE_FP8_MARKER = "__acc_native_fp8__"

# ============================================================================
# 运行时注册 glm_moe_dsa (transformers < 5.5 可能没有)
# ============================================================================

def _ensure_glm_moe_dsa_registered():
    """确保 glm_moe_dsa 架构注册到 HF Auto 映射，兼容旧版 transformers。"""
    if "glm_moe_dsa" in CONFIG_MAPPING:
        return
    try:
        from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM
        CONFIG_MAPPING.register("glm_moe_dsa", GlmMoeDsaConfig)
        MODEL_FOR_CAUSAL_LM_MAPPING.register(GlmMoeDsaConfig, GlmMoeDsaForCausalLM)
    except ImportError:
        # transformers 版本没有 glm_moe_dsa 模块 — 尝试复制 glm4_moe 作为 fallback
        pass

_ensure_glm_moe_dsa_registered()


def _ensure_deepseek_v4_registered():
    """Register official DeepSeek-V4 classes when an older Transformers build omitted the mapping."""
    if "deepseek_v4" in CONFIG_MAPPING:
        return
    try:
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM
        CONFIG_MAPPING.register("deepseek_v4", DeepseekV4Config)
        MODEL_FOR_CAUSAL_LM_MAPPING.register(DeepseekV4Config, DeepseekV4ForCausalLM)
        logger.info("  registered Transformers model_type=deepseek_v4")
    except ImportError:
        # The installed Transformers package predates DeepSeek-V4; report the
        # actionable error at AutoConfig time instead of masking the import.
        pass


_ensure_deepseek_v4_registered()


def _read_raw_model_config(model_path: str) -> Dict[str, Any]:
    config_path = os.path.join(model_path, "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _looks_like_deepseek_v4_config(raw_config: Dict[str, Any]) -> bool:
    model_type = str(raw_config.get("model_type", "")).lower()
    architectures = [
        str(name).lower() for name in (raw_config.get("architectures") or [])
    ]
    if model_type == "deepseek_v4" or any("deepseekv4" in name for name in architectures):
        return True
    # Some converted/rotated checkpoints retain ``deepseek_v3`` as the
    # model_type even though the actual graph is V4.  Require multiple V4-only
    # architecture fields so ordinary V3 checkpoints are never upgraded by
    # path-name guesswork.
    v4_markers = (
        "hc_mult", "compress_ratios", "compress_rate_csa",
        "compress_rate_hca", "mlp_layer_types",
    )
    return sum(key in raw_config for key in v4_markers) >= 2


def _load_model_config(model_path: str):
    """Load AutoConfig, correcting converted V4 checkpoints misread as V3."""
    raw_config = _read_raw_model_config(model_path)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if not _looks_like_deepseek_v4_config(raw_config):
        return config

    try:
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
            DeepseekV4Config,
        )
    except ImportError:
        return config

    if not isinstance(config, DeepseekV4Config) or not hasattr(config, "layer_types"):
        logger.warning(
            "  DeepSeek-V4 checkpoint 被 AutoConfig 解析为 %s；"
            "按原始 V4 config.json 重新构造 DeepseekV4Config",
            type(config).__name__,
        )
        config = DeepseekV4Config.from_dict(raw_config)
    if not getattr(config, "layer_types", None):
        raise RuntimeError(
            "DeepSeek-V4 config 重建后仍缺少 layer_types；请确认 config.json "
            "包含 V4 的 compress_ratios/layer_types"
        )
    config.architectures = ["DeepseekV4ForCausalLM"]
    return config


def require_model_runtime_support(model_path: str) -> None:
    """Fail early with an actionable error for architectures missing in HF.

    ``trust_remote_code`` cannot help DeepSeek-V4 checkpoints that only ship a
    standard ``config.json``: the actual implementation must exist in the
    installed Transformers package.
    """
    raw_config = _read_raw_model_config(model_path)
    if not _looks_like_deepseek_v4_config(raw_config):
        return
    try:
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import (  # noqa: F401
            DeepseekV4Config,
        )
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (  # noqa: F401
            DeepseekV4ForCausalLM,
        )
    except ImportError as exc:
        import transformers
        raise RuntimeError(
            "检测到 model_type=deepseek_v4，但当前 Transformers "
            f"{transformers.__version__} 不包含官方 DeepSeek-V4 实现。"
            "请升级到包含 transformers.models.deepseek_v4 的版本"
            "（建议 transformers>=5.12.0），然后重新运行；本工具不会把 V4 "
            "错误地回退成 V3/GLM 结构。"
        ) from exc


# ============================================================================
# Symlink 工具
# ============================================================================

def setup_quant_weight_links(model_path: str) -> bool:
    """
    为量化模型创建标准权重文件名的符号链接

    msmodelslim 导出的文件名是 quant_model_weights-*.safetensors,
    transformers 期望的是 model-*.safetensors.
    创建符号链接让 from_pretrained 能找到权重。
    """
    index_path = os.path.join(model_path, "quant_model_weights.safetensors.index.json")
    link_path = os.path.join(model_path, "model.safetensors.index.json")

    if os.path.exists(index_path) and not os.path.exists(link_path):
        os.symlink(index_path, link_path)

        with open(index_path, 'r') as f:
            index_data = json.load(f)

        weight_map = index_data.get("weight_map", {})
        shard_files = set(weight_map.values())

        for shard_file in shard_files:
            shard_path = os.path.join(model_path, shard_file)
            if shard_file.startswith("quant_model_weights-"):
                new_name = shard_file.replace("quant_model_weights-", "model-")
                new_path = os.path.join(model_path, new_name)
                if os.path.exists(shard_path) and not os.path.exists(new_path):
                    os.symlink(shard_path, new_path)

        return True
    return False


def cleanup_quant_weight_links(model_path: str):
    """清理创建的符号链接"""
    link_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.islink(link_path):
        os.unlink(link_path)

    for f in os.listdir(model_path):
        if f.startswith("model-") and f.endswith(".safetensors"):
            path = os.path.join(model_path, f)
            if os.path.islink(path):
                os.unlink(path)


def is_quantized_model(model_path: str) -> bool:
    """检查是否是量化模型 (msmodelslim 或 compressed-tensors)"""
    # msmodelslim
    if os.path.exists(os.path.join(model_path, "quant_model_description.json")):
        return True
    # compressed-tensors: config.json 里 quantization_config.quant_method == "compressed-tensors"
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                config = json.load(f)
            qconfig = config.get("quantization_config", {})
            if not qconfig:
                qconfig = config.get("text_config", {}).get("quantization_config", {})
            if qconfig.get("quant_method") == "compressed-tensors":
                return True
            # GLM-5.3 and other native checkpoints advertise FP8 directly
            # (quant_method=fp8, fmt=e4m3) rather than shipping an
            # msModelSlim descriptor.  They still need to be decoded to
            # BF16 on A3, whose runtime does not execute native FP8 weights.
            if _config_declares_native_fp8(config):
                return True
            # The official DeepSeek-V4-Flash reference checkpoint stores
            # routed experts as packed E2M1 FP4 (with ``.scale`` tensors),
            # despite advertising an FP8 conversion config for deployment.
            if (
                str(config.get("model_type", "")).lower() == "deepseek_v4"
                and str(config.get("expert_dtype", "")).lower() == "fp4"
            ):
                return True
        except Exception as e:
            logger.debug(f"is_quantized_model config parse failed: {e}")
    return False


def _config_declares_native_fp8(config: Dict[str, Any]) -> bool:
    """Return whether a checkpoint config declares native FP8 weights.

    Keep this deliberately scoped to quantization/dtype fields so an
    unrelated string containing ``fp8`` cannot make a BF16 model look
    quantized.  ``text_config`` is handled because multimodal GLM exports
    place ``quantization_config`` there.
    """
    configs = [config]
    nested = config.get("text_config")
    if isinstance(nested, dict):
        configs.append(nested)
    for item in configs:
        qconfig = item.get("quantization_config")
        if isinstance(qconfig, dict):
            method = str(qconfig.get("quant_method", "")).lower()
            fmt = str(qconfig.get("fmt", qconfig.get("format", ""))).lower()
            weight_dtype = str(qconfig.get("weight_dtype", "")).lower()
            if method in {"fp8", "mxfp8", "float8"}:
                return True
            if any(token in value for value in (fmt, weight_dtype)
                   for token in ("fp8", "float8", "e4m3", "e5m2")):
                return True
        for key in ("torch_dtype", "dtype", "weight_dtype", "quant_dtype"):
            value = str(item.get(key, "")).lower()
            if "float8" in value or "fp8" in value:
                return True
    return False


def is_deepseek_v4_fp4_model(model_path: str) -> bool:
    """Whether this is the official DeepSeek-V4 packed-FP4 reference format."""
    config_path = os.path.join(model_path, "config.json")
    try:
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        str(config.get("model_type", "")).lower() == "deepseek_v4"
        and str(config.get("expert_dtype", "")).lower() == "fp4"
    )


def native_quant_description(model_path: str) -> Optional[dict]:
    """Return a synthetic descriptor for native checkpoint formats.

    Most exports carry ``quant_model_description.json``.  The official V4
    reference does not: its routed experts are packed FP4 and use per-group
    ``.scale`` keys.  Native GLM FP8 exports likewise omit the msModelSlim
    descriptor and use ``weight_scale_inv``; small markers keep the generic
    indexed loader on the correct decode path.
    """
    if is_deepseek_v4_fp4_model(model_path):
        return {_DEEPSEEK_V4_FP4_MARKER: "DEEPSEEK_FP4"}
    if _config_declares_native_fp8(_read_raw_model_config(model_path)):
        return {_NATIVE_FP8_MARKER: "FP8_E4M3"}
    return None


def is_compressed_tensors_model(model_path: str) -> bool:
    """检查是否是 compressed-tensors 量化模型"""
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                config = json.load(f)
            qconfig = config.get("quantization_config", {})
            if not qconfig:
                qconfig = config.get("text_config", {}).get("quantization_config", {})
            return qconfig.get("quant_method") == "compressed-tensors"
        except Exception as e:
            logger.debug(f"is_compressed_tensors_model config parse failed: {e}")
    return False


# ============================================================================
# INT4 解包
# ============================================================================
def unpack_int4_to_int8(packed_weight: Tensor) -> Tensor:
    """
    将INT4打包的INT8权重解包为INT8

    msmodelslim 的打包方式:
      1个INT8 = 2个INT4
      低4位 = 第1个值, 高4位 = 第2个值

    Args:
        packed_weight: [out_features/2, in_features] INT8 tensor

    Returns:
        [out_features, in_features] INT8 tensor (实际值范围 -8~7)
    """
    # 转为uint8来方便位操作
    packed = packed_weight.to(torch.uint8)

    # 低4位
    low = (packed & 0x0F).to(torch.int8)
    # 高4位
    high = ((packed >> 4) & 0x0F).to(torch.int8)

    # INT4是4位有符号: 0-7表示0-7, 8-15表示-8到-1
    # 需要将 unsigned 4-bit 转为 signed
    low = torch.where(low > 7, low - 16, low)
    high = torch.where(high > 7, high - 16, high)

    # 交错排列: [out_features/2] 的 low 和 high 交错为 [out_features]
    # 对每行: [low_0, high_0, low_1, high_1, ...]
    out_features_half, in_features = packed.shape
    out_features = out_features_half * 2

    # 堆叠: [2, out_features/2, in_features]
    unpacked = torch.stack([low, high], dim=0)
    # 转置为 [out_features/2, 2, in_features] 再 reshape
    unpacked = unpacked.permute(1, 0, 2).reshape(out_features, in_features)

    return unpacked.to(torch.int8)


# ============================================================================
# 反量化
# ============================================================================

# MX (Microscaling) 量化类型
# W8A8_MXFP8:  weight=float8_e4m3fn, scale=uint8 (E8M0), block_size=32
# W4A8_MXFP:   weight=uint8 (2×FP4 packed), scale=uint8 (E8M0), block_size=32
# W4A4_MXFP4:  weight=uint8 (2×FP4 packed), scale=uint8 (E8M0), block_size=32
_MX_QUANT_TYPES = frozenset({"W8A8_MXFP8", "W4A8_MXFP", "W4A4_MXFP4"})
_MXFP4_QUANT_TYPES = frozenset({"W4A8_MXFP", "W4A4_MXFP4"})


def _build_fp8_e4m3fn_lut():
    """Return the 256-value E4M3FN decode table.

    The table is intentionally built with Python arithmetic rather than by
    casting a native float8 tensor.  Several torch_npu releases reject that
    cast on NPU (aclnnInplaceCopy/561103), while integer indexing into a small
    float32 table is supported.
    """
    import math

    values = []
    for raw in range(256):
        sign = -1.0 if raw & 0x80 else 1.0
        exponent = (raw >> 3) & 0x0F
        mantissa = raw & 0x07
        if exponent == 0:
            value = (mantissa / 8.0) * math.ldexp(1.0, -6)
        elif exponent == 0x0F:
            # E4M3FN has no infinities.  The top mantissa code is NaN;
            # mantissas 0..6 are finite (up to 448).
            value = float("nan") if mantissa == 0x07 else (1.0 + mantissa / 8.0) * 256.0
        else:
            value = (1.0 + mantissa / 8.0) * math.ldexp(1.0, exponent - 7)
        values.append(sign * value)
    return tuple(values)


_FP8_E4M3FN_LUT = _build_fp8_e4m3fn_lut()
_FP8_E4M3FN_LUT_BY_DEVICE = {}


def dequantize_weight_mxfp8_npu(
    weight_e4m3: Tensor,
    weight_scale_u8: Tensor,
    block_size: int = 32,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Decode native E4M3 FP8 bytes on NPU without a float8 cast.

    ``view(torch.uint8)`` is a byte reinterpretation (not a numeric cast),
    so it avoids the unsupported native-float8 ``.float()`` path.  The actual
    E4M3FN conversion is a 256-entry lookup on the destination NPU.  If the
    runtime cannot reinterpret/index the native dtype, the caller can catch
    the RuntimeError and use the CPU compatibility path.
    """
    if weight_e4m3.dim() != 2 or weight_scale_u8.dim() != 2:
        raise ValueError(
            "MXFP8 requires 2-D weight and scale; got "
            f"{tuple(weight_e4m3.shape)} and {tuple(weight_scale_u8.shape)}"
        )
    if weight_e4m3.device != weight_scale_u8.device:
        raise ValueError("MXFP8 weight and scale must be on the same device")

    # This must remain a view: ``to(uint8)`` would be a numeric conversion and
    # loses the FP8 payload.  The caller may already have performed this view
    # on CPU before transferring to an NPU that has no native FP8 support.
    raw = weight_e4m3 if weight_e4m3.dtype == torch.uint8 else weight_e4m3.view(torch.uint8)
    key = str(weight_e4m3.device)
    lut = _FP8_E4M3FN_LUT_BY_DEVICE.get(key)
    if lut is None:
        lut = torch.tensor(
            _FP8_E4M3FN_LUT, dtype=torch.float32, device=weight_e4m3.device
        )
        _FP8_E4M3FN_LUT_BY_DEVICE[key] = lut
    w_fp = lut[raw.to(torch.long)]

    shared_exp = weight_scale_u8.to(torch.float32) - 127.0
    scale_value = torch.pow(2.0, shared_exp)
    scale_broad = scale_value.repeat_interleave(block_size, dim=1)
    if scale_broad.shape != w_fp.shape:
        raise ValueError(
            "MXFP8 scale shape is incompatible with weight: "
            f"weight={tuple(weight_e4m3.shape)}, scale={tuple(weight_scale_u8.shape)}, "
            f"expanded={tuple(scale_broad.shape)}"
        )
    return (w_fp * scale_broad).to(dtype)


def dequantize_weight_mxfp8(
    weight_e4m3: Tensor,
    weight_scale_u8: Tensor,
    block_size: int = 32,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """W8A8_MXFP8 per-block 反量化"""
    shared_exp = weight_scale_u8.float() - 127.0
    scale_value = torch.pow(2.0, shared_exp)
    scale_broad = scale_value.repeat_interleave(block_size, dim=1)
    w_fp = weight_e4m3.float() * scale_broad
    return w_fp.to(dtype)


def unpack_fp4_from_uint8(packed: Tensor) -> Tensor:
    """将 uint8 packed 的 FP4 (E2M1) 解包为 float32。"""
    E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                dtype=torch.float32, device=packed.device)
    m, n_half = packed.shape
    packed_flat = packed.reshape(-1).to(torch.uint8)

    low_nibble = (packed_flat & 0x0F).long()
    high_nibble = ((packed_flat >> 4) & 0x0F).long()

    low_sign = 1.0 - 2.0 * (low_nibble >> 3).float()
    low_idx = low_nibble & 0x07
    low_val = E2M1_VALUES[low_idx] * low_sign

    high_sign = 1.0 - 2.0 * (high_nibble >> 3).float()
    high_idx = high_nibble & 0x07
    high_val = E2M1_VALUES[high_idx] * high_sign

    out = torch.stack([low_val, high_val], dim=-1).reshape(-1)
    return out.reshape(m, n_half * 2)


def dequantize_weight_mxfp4(
    packed_weight_u8: Tensor,
    weight_scale_u8: Tensor,
    block_size: int = 32,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """W4A8_MXFP / W4A4_MXFP4 per-block 反量化。"""
    unpacked = unpack_fp4_from_uint8(packed_weight_u8)

    shared_exp = weight_scale_u8.float() - 127.0
    scale_value = torch.pow(2.0, shared_exp)
    scale_broad = scale_value.repeat_interleave(block_size, dim=1)
    w_fp = unpacked * scale_broad
    return w_fp.to(dtype)


def dequantize_weight_mx(
    weight_data: Tensor,
    weight_scale_u8: Tensor,
    quant_type: str,
    block_size: int = 32,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """MX 量化权重统一反量化入口，根据 quant_type 自动选择 MXFP8 或 MXFP4 逻辑。"""
    if quant_type in _MXFP4_QUANT_TYPES:
        return dequantize_weight_mxfp4(weight_data, weight_scale_u8, block_size, dtype)
    else:
        return dequantize_weight_mxfp8(weight_data, weight_scale_u8, block_size, dtype)


_DEEPSEEK_FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                         0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _decode_deepseek_v4_e8m0_scale(scale: Tensor, device=None) -> Tensor:
    """Decode a V4 UE8M0 scale to float32.

    ModelScope safetensors may expose ``F8_E8M0`` as a native float8 dtype,
    while older torch/safetensors/NPU combinations expose the same bytes as
    uint8. The latter are exponent bytes, not numeric multipliers: byte ``b``
    means ``2 ** (b - 127)``.
    """
    if device is None:
        device = scale.device
    dtype_name = str(scale.dtype).lower()
    if "e8m0" in dtype_name:
        try:
            return scale.to(device=device, dtype=torch.float32)
        except (RuntimeError, TypeError):
            raw = scale.view(torch.uint8)
    elif not scale.dtype.is_floating_point:
        raw = scale.to(device=device, dtype=torch.int32)
    else:
        return scale.to(device=device, dtype=torch.float32)
    raw = raw.to(device=device, dtype=torch.float32)
    return torch.pow(2.0, raw - 127.0)


def dequantize_deepseek_v4_fp4(
    packed_weight: Tensor,
    scale: Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Decode DeepSeek-V4's packed E2M1 expert matrix to BF16/FP16.

    The official checkpoint stores two E2M1 values per int8 and a floating
    E8M0 multiplier for every 32 logical input columns.  This is distinct
    from msModelSlim W4A8: metadata is named ``.scale`` and no zero-point is
    present.  The formula follows the vendor's released ``inference/convert``
    implementation before its optional FP8 conversion step.
    """
    if packed_weight.dim() != 2 or scale.dim() != 2:
        raise ValueError(
            "DeepSeek-V4 FP4 requires 2-D packed weight and 2-D scale; got "
            f"{tuple(packed_weight.shape)} and {tuple(scale.shape)}"
        )
    out_features, packed_columns = packed_weight.shape
    logical_columns = packed_columns * 2
    if scale.shape[0] != out_features or logical_columns % scale.shape[1] != 0:
        raise ValueError(
            "DeepSeek-V4 FP4 scale shape is incompatible with packed weight: "
            f"weight={tuple(packed_weight.shape)}, scale={tuple(scale.shape)}"
        )
    group_size = logical_columns // scale.shape[1]
    if group_size != 32:
        raise ValueError(
            "DeepSeek-V4 FP4 expects 32-column scale groups; got "
            f"group_size={group_size} for weight={tuple(packed_weight.shape)}, "
            f"scale={tuple(scale.shape)}"
        )
    packed = packed_weight.to(torch.uint8)
    values = torch.tensor(
        _DEEPSEEK_FP4_VALUES, dtype=torch.float32, device=packed.device
    )
    low = values[(packed & 0x0F).long()]
    high = values[((packed >> 4) & 0x0F).long()]
    unpacked = torch.stack((low, high), dim=-1).flatten(1)
    expanded_scale = _decode_deepseek_v4_e8m0_scale(scale, device=packed.device)
    expanded_scale = expanded_scale.repeat_interleave(group_size, dim=1)
    return (unpacked * expanded_scale).to(dtype)


def dequantize_deepseek_v4_fp8(
    weight: Tensor,
    scale: Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Decode V4 block-FP8 matrices carrying E8M0 ``.scale`` tensors."""
    if weight.dim() != 2 or scale.dim() != 2:
        raise ValueError(
            "DeepSeek-V4 FP8 requires 2-D weight and 2-D scale; got "
            f"{tuple(weight.shape)} and {tuple(scale.shape)}"
        )
    out_features, in_features = weight.shape
    if out_features % scale.shape[0] or in_features % scale.shape[1]:
        raise ValueError(
            "DeepSeek-V4 FP8 scale shape is incompatible with weight: "
            f"weight={tuple(weight.shape)}, scale={tuple(scale.shape)}"
        )
    expanded_scale = _decode_deepseek_v4_e8m0_scale(scale, device=weight.device)
    expanded_scale = expanded_scale.repeat_interleave(
        out_features // scale.shape[0], dim=0
    ).repeat_interleave(in_features // scale.shape[1], dim=1)
    return (weight.to(torch.float32) * expanded_scale).to(dtype)


def dequantize_deepseek_v4_native_weight(
    weight: Tensor,
    scale: Optional[Tensor],
    dtype: torch.dtype = torch.bfloat16,
) -> Optional[Tensor]:
    """Decode an official V4 FP4 expert or FP8 dense matrix when applicable."""
    if scale is None or weight.dim() != 2:
        return None
    if weight.dtype == torch.int8:
        return dequantize_deepseek_v4_fp4(weight, scale, dtype=dtype)
    if str(weight.dtype).startswith("torch.float8"):
        return dequantize_deepseek_v4_fp8(weight, scale, dtype=dtype)
    return None


def dequantize_native_fp8_weight(
    weight: Tensor,
    scale: Optional[Tensor] = None,
    dtype: torch.dtype = torch.bfloat16,
    block_size: Tuple[int, int] = (128, 128),
    scale_name: str = "weight_scale_inv",
) -> Tensor:
    """Decode a native E4M3 FP8 matrix into the requested runtime dtype.

    GLM-5.3's public checkpoint uses E4M3 payloads and ``weight_scale_inv``
    block scales, while A3 torch_npu installations generally cannot execute
    a native float8 parameter.  Decode the payload through the same byte LUT
    used by the MXFP8 path, then expand either [out, in] or block [out/bo,
    in/bi] scales.  ``weight_scale_inv`` is multiplied during dequantization;
    uint8 E8M0 scales retain the MXFP8 exponent interpretation.
    """
    if weight.dim() < 2:
        raise ValueError(f"native FP8 weight must be at least 2-D, got {tuple(weight.shape)}")
    raw = weight if weight.dtype == torch.uint8 else weight.view(torch.uint8)
    lut = _FP8_E4M3FN_LUT_BY_DEVICE.get(str(weight.device))
    if lut is None:
        lut = torch.tensor(_FP8_E4M3FN_LUT, dtype=torch.float32, device=weight.device)
        _FP8_E4M3FN_LUT_BY_DEVICE[str(weight.device)] = lut
    decoded = lut[raw.to(torch.long)]
    if scale is None:
        return decoded.to(dtype)

    scale = scale.to(device=weight.device)
    scale_name = str(scale_name).lower()
    if scale.dtype == torch.uint8:
        scale_value = torch.pow(2.0, scale.to(torch.float32) - 127.0)
    elif "e8m0" in str(scale.dtype).lower():
        scale_value = _decode_deepseek_v4_e8m0_scale(scale, device=weight.device)
    else:
        scale_value = scale.to(torch.float32)

    target_shape = tuple(weight.shape)
    matrix_shape = target_shape[-2:]
    if scale_value.dim() == 0:
        scale_value = scale_value.reshape((1, 1))
    elif scale_value.dim() == 1:
        count = scale_value.numel()
        if count == 1:
            scale_value = scale_value.reshape(1, 1)
        elif count == matrix_shape[0]:
            scale_value = scale_value.reshape(matrix_shape[0], 1)
        elif count == matrix_shape[1]:
            scale_value = scale_value.reshape(1, matrix_shape[1])
    # Packed MoE experts may carry a leading expert dimension.  A 2-D scale
    # is shared by all experts; a same-rank scale keeps the leading dimensions.
    if scale_value.dim() == 2 and weight.dim() > 2:
        scale_value = scale_value.reshape((1,) * (weight.dim() - 2) + tuple(scale_value.shape))
    if tuple(scale_value.shape) != target_shape:
        if scale_value.dim() != weight.dim():
            raise ValueError(
                f"{scale_name} must match the matrix rank for native FP8 weight; got "
                f"{tuple(scale_value.shape)}"
            )
        scale_prefix = tuple(scale_value.shape[:-2])
        target_prefix = target_shape[:-2]
        if len(scale_prefix) == len(target_prefix) and all(
            source in (1, target) for source, target in zip(scale_prefix, target_prefix)
        ):
            scale_value = scale_value.expand(target_prefix + tuple(scale_value.shape[-2:]))
        elif scale_prefix != target_prefix:
            raise ValueError(
                f"native FP8 scale leading shape {scale_prefix} does not match "
                f"weight {target_prefix}"
            )
        out, width = matrix_shape
        rows, cols = tuple(scale_value.shape[-2:])
        if rows == out and cols > 0 and width % cols == 0:
            scale_value = scale_value.repeat_interleave(width // cols, dim=-1)
        elif cols == width and rows > 0 and out % rows == 0:
            scale_value = scale_value.repeat_interleave(out // rows, dim=-2)
        elif rows > 0 and cols > 0 and out % rows == 0 and width % cols == 0:
            scale_value = scale_value.repeat_interleave(out // rows, dim=-2)
            scale_value = scale_value.repeat_interleave(width // cols, dim=-1)
        else:
            raise ValueError(
                f"native FP8 scale shape {tuple(scale_value.shape)} is incompatible "
                f"with weight shape {target_shape}"
            )
    return (decoded * scale_value).to(dtype)


def _canonical_dynamic_qparam(
    value: Tensor,
    weight_shape: Tuple[int, int],
    name: str,
    device: torch.device,
) -> Tensor:
    """Normalize a dynamic weight qparam without expanding it in memory."""
    value = value.to(device=device, dtype=torch.float32)
    while value.dim() > 2 and value.shape[-1] == 1:
        value = value.squeeze(-1)

    if value.dim() == 0:
        return value
    if value.dim() > 2:
        raise ValueError(
            f"{name} must be scalar, 1-D, or 2-D after removing trailing "
            f"singleton dimensions; got {tuple(value.shape)}"
        )

    out_features, in_features = weight_shape
    if value.dim() == 1:
        count = value.numel()
        if count == out_features:
            value = value.reshape(out_features, 1)
        elif count == in_features:
            value = value.reshape(1, in_features)
        elif count % out_features == 0:
            columns = count // out_features
            if columns > in_features or in_features % columns != 0:
                raise ValueError(
                    f"{name} shape {tuple(value.shape)} is incompatible with "
                    f"weight shape {weight_shape}"
                )
            value = value.reshape(out_features, columns)
        elif count < in_features and in_features % count == 0:
            value = value.reshape(1, count)
        else:
            raise ValueError(
                f"{name} shape {tuple(value.shape)} is incompatible with "
                f"weight shape {weight_shape}"
            )

    rows, columns = value.shape
    if rows not in (1, out_features):
        raise ValueError(
            f"{name} first dimension must be 1 or {out_features}, got "
            f"{tuple(value.shape)}"
        )
    if columns > in_features or in_features % columns != 0:
        raise ValueError(
            f"{name} second dimension must divide input width {in_features}, "
            f"got {tuple(value.shape)}"
        )
    return value


def _reshape_grouped_dynamic_qparam(
    value: Tensor,
    weight_shape: Tuple[int, int],
    num_groups: int,
    name: str,
) -> Tensor:
    """View a qparam so it broadcasts over [out, groups, group_size]."""
    if value.dim() == 0:
        return value

    _, in_features = weight_shape
    rows, columns = value.shape
    group_size = in_features // num_groups
    if columns == 1:
        return value.reshape(rows, 1, 1)
    if columns == num_groups:
        return value.unsqueeze(-1)
    if columns == in_features:
        return value.reshape(rows, num_groups, group_size)
    raise ValueError(
        f"{name} shape {tuple(value.shape)} cannot broadcast across "
        f"{num_groups} weight groups for weight shape {weight_shape}"
    )


def dequantize_weight_dynamic(
    weight_data: Tensor,
    weight_scale: Tensor,
    weight_offset: Tensor = None,
    dtype: torch.dtype = torch.float16,
) -> Tensor:
    """Dequantize dynamic weights, including per-channel and per-group layouts.

    Per-group checkpoints store scale/offset as ``[out_features, num_groups]``.
    The weight group size is inferred from those tensors; it is independent of
    the activation fake-quant group size selected by the CLI.
    """
    if weight_data.dim() != 2:
        raise ValueError(
            "dynamic weight dequantization expects a 2-D matrix, got "
            f"{tuple(weight_data.shape)}"
        )

    weight_shape = tuple(weight_data.shape)
    weight_fp = weight_data.to(torch.float32)
    scale = _canonical_dynamic_qparam(
        weight_scale, weight_shape, "weight_scale", weight_data.device
    )
    offset = None
    if weight_offset is not None:
        offset = _canonical_dynamic_qparam(
            weight_offset, weight_shape, "weight_offset", weight_data.device
        )

    scale_columns = scale.shape[1] if scale.dim() == 2 else 1
    in_features = weight_shape[1]
    is_grouped = 1 < scale_columns < in_features
    if is_grouped:
        num_groups = scale_columns
        group_size = in_features // num_groups
        grouped_weight = weight_fp.reshape(
            weight_shape[0], num_groups, group_size
        )
        grouped_scale = _reshape_grouped_dynamic_qparam(
            scale, weight_shape, num_groups, "weight_scale"
        )
        grouped_offset = 0.0
        if offset is not None:
            grouped_offset = _reshape_grouped_dynamic_qparam(
                offset, weight_shape, num_groups, "weight_offset"
            )
        result = (grouped_weight - grouped_offset) * grouped_scale
        return result.reshape(weight_shape).to(dtype)

    try:
        result = weight_fp if offset is None else weight_fp - offset
        return (result * scale).to(dtype)
    except RuntimeError as exc:
        offset_shape = None if offset is None else tuple(offset.shape)
        raise ValueError(
            "dynamic weight qparams are not broadcast-compatible: "
            f"weight={weight_shape}, scale={tuple(scale.shape)}, "
            f"offset={offset_shape}"
        ) from exc


def dequantize_weight_static(
    weight_data: Tensor,
    deq_scale: Tensor,
    input_scale: Tensor = None,
    quant_bias: Tensor = None,
    dtype: torch.dtype = torch.float16,
) -> Tuple[Tensor, bool]:
    """
    W8A8 静态量化反量化

    deq_scale = input_scale * weight_scale
    所以: weight_scale = deq_scale / input_scale
    w_fp = w_int8 * weight_scale = w_int8 * (deq_scale / input_scale)

    如果没有 input_scale, 直接用 deq_scale: w_fp = w_int8 * deq_scale

    Returns:
        (dequantized_weight, had_input_scale)
    """
    # msModelSlim 为 FP16 模型导出静态 W8A8 时，会把每个 float32
    # deq_scale 的 32-bit 位模式先 reinterpret 成 int32，再数值扩展到
    # int64 供 Ascend 量化算子直接消费。这里必须先收窄回 int32，再按
    # float32 位模式解释；直接 ``view(float64)`` 或 ``.float()`` 都会把
    # scale 破坏，导致权重归零或放大到 1e9 量级。
    if deq_scale.dtype in (torch.int32, torch.int64):
        scale_device = deq_scale.device
        # The scale vector is tiny; bitcast on CPU so this path does not depend
        # on whether a particular torch_npu/CUDA build implements view(dtype).
        deq_scale = (
            deq_scale.to(device="cpu", dtype=torch.int32)
            .contiguous()
            .view(torch.float32)
            .to(scale_device)
        )

    deq_scale = deq_scale.float()

    if input_scale is not None:
        input_scale = input_scale.float()
        weight_scale = deq_scale / input_scale
        had_input_scale = True
    else:
        weight_scale = deq_scale
        had_input_scale = False

    if weight_scale.dim() == 1:
        weight_scale = weight_scale.unsqueeze(1)

    if not torch.isfinite(weight_scale).all():
        raise ValueError("W8A8 dequantization produced non-finite weight_scale")

    w_fp = weight_data.to(torch.float32) * weight_scale
    return w_fp.to(dtype), had_input_scale


# ============================================================================
# 逐层反量化
# ============================================================================

def load_quant_weights(model_path: str) -> Dict[str, Tensor]:
    """加载所有量化权重 (从 safetensors)"""
    # 从 weight_index 找到所有分片文件
    index_path = os.path.join(model_path, "quant_model_weights.safetensors.index.json")
    shard_files = set()
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        shard_files = set(index_data.get("weight_map", {}).values())
    else:
        # 可能只有一个标准命名的 safetensors 文件；如果没有则继续走
        # ModelScope 的无 index 多分片回退，而不是提前返回空字典。
        single = os.path.join(model_path, "quant_model_weights.safetensors")
        if os.path.exists(single):
            return load_file(single)

    weights = {}
    for shard_file in shard_files:
        file_path = os.path.join(model_path, shard_file)
        if os.path.exists(file_path):
            file_weights = load_file(file_path)
            weights.update(file_weights)

    if weights:
        return weights

    # ModelScope's DeepSeek-V4 W8A8 export contains 74
    # ``quant_model_weights-xxxxx-of-xxxxx.safetensors`` files but no index
    # json.  Load each shard lazily at the file level, preserving the existing
    # CPU dictionary contract used by the resident expert installer.
    try:
        shard_names = sorted(
            name for name in os.listdir(model_path)
            if name.startswith("quant_model_weights-")
            and name.endswith(".safetensors")
        )
    except OSError:
        shard_names = []
    for shard_name in shard_names:
        weights.update(load_file(os.path.join(model_path, shard_name)))
    return weights


def _resolve_target_device(param_name, device_map, device_list, default_device):
    """根据参数名确定目标设备 (从 device_map 中找最匹配的 key)。"""
    if device_map is None:
        return device_list[0] if device_list else default_device
    matched_device = device_list[0]
    matched_key_len = 0
    for map_key, map_dev in device_map.items():
        if param_name.startswith(map_key) and len(map_key) > matched_key_len:
            matched_device = map_dev
            matched_key_len = len(map_key)
    return matched_device


def _load_float_module_weights(model, quant_weights, quant_desc, dtype,
                                device_map, device_list, default_device):
    """加载 module 级别的 FLOAT 权重 (nn.Linear/nn.Embedding/norm 等)。"""
    from .utils import parse_base_name
    loaded = 0
    loaded_names = set()
    for name, module in model.named_modules():
        if not hasattr(module, 'weight'):
            continue
        weight_key = f"{name}.weight"
        base_name = parse_base_name(weight_key)
        quant_type = quant_desc.get(weight_key, quant_desc.get(base_name, None))
        if quant_type != "FLOAT" and quant_type is not None:
            continue
        if weight_key in quant_weights:
            target_device = _resolve_target_device(
                weight_key, device_map, device_list, default_device)
            module.weight.data = quant_weights[weight_key].to(dtype).to(target_device)
            loaded += 1
            loaded_names.add(weight_key)
        bias_key = f"{name}.bias"
        if (bias_key in quant_weights and hasattr(module, 'bias')
                and module.bias is not None):
            target_device = _resolve_target_device(
                bias_key, device_map, device_list, default_device)
            module.bias.data = quant_weights[bias_key].to(dtype).to(target_device)
            loaded += 1
            loaded_names.add(bias_key)
    return loaded, loaded_names


def _load_float_param_weights(model, quant_weights, quant_desc, dtype,
                               device_map, device_list, default_device,
                               loaded_names):
    """加载非 module 的 parameter (如 A_log, dt_bias 等)。"""
    from .utils import parse_base_name
    loaded = 0
    for name, param in model.named_parameters():
        if name in loaded_names or name not in quant_weights:
            continue
        base_name = parse_base_name(name)
        quant_type = quant_desc.get(name, quant_desc.get(base_name, None))
        if quant_type != "FLOAT" and quant_type is not None:
            continue
        target_device = _resolve_target_device(
            name, device_map, device_list, default_device)
        param.data = quant_weights[name].to(dtype).to(target_device)
        loaded += 1
    return loaded


def _load_float_weights(
    model, quant_weights, quant_desc, dtype, device, verbose=True,
):
    """
    加载非量化权重(embed_tokens, lm_head, norm等)到模型

    to_empty后参数不再是is_meta, 所以用quant_desc来识别FLOAT权重,
    直接从safetensors加载并写入对应module。
    """
    device_map = getattr(model, 'hf_device_map', None)
    device_list = parse_device_list(device) if isinstance(device, str) and ',' in device else [device]
    default_device = device if isinstance(device, str) else str(device)

    mod_loaded, loaded_names = _load_float_module_weights(
        model, quant_weights, quant_desc, dtype,
        device_map, device_list, default_device)
    param_loaded = _load_float_param_weights(
        model, quant_weights, quant_desc, dtype,
        device_map, device_list, default_device, loaded_names)

    loaded = mod_loaded + param_loaded
    if verbose and loaded > 0:
        logger.info(f"  加载了{loaded}个FLOAT/非量化权重到模型")



class PassThroughIdentity(nn.Module):
    """用于替换不需要的层的空层"""

    def forward(self, hidden_states, **kwargs):
        return hidden_states


def _get_shards_for_layers(model_path: str, layers: List[int]) -> Tuple[set, dict]:
    """
    根据layer范围确定需要加载的shard文件

    Returns:
        (shard_files, layer_to_shard) - 需要加载的shard文件集合, layer->shard映射

    同时也会加载 embed_tokens, norm, lm_head 等基础权重所在的 shard。
    """
    index_path = os.path.join(model_path, "quant_model_weights.safetensors.index.json")
    if not os.path.exists(index_path):
        return set(), {}

    with open(index_path, 'r') as f:
        index_data = json.load(f)
    weight_map = index_data.get("weight_map", {})

    layer_set = set(layers)
    shard_files = set()
    layer_to_shard = {}

    # 先收集所有 shard 文件
    all_shards = set(weight_map.values())

    for k, v in weight_map.items():
        # 收集层权重
        import re
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", k)
        if match is not None:
            layer_num = int(match.group(1))
            if layer_num in layer_set:
                shard_files.add(v)
                layer_to_shard[layer_num] = v

        # 收集基础权重 (embed_tokens, norm, lm_head)
        # 这些通常在第一个 shard 文件中
        if _is_non_layer_key(k):
            shard_files.add(v)

    # 如果没有找到任何 shard（层列表为空的情况），至少加载所有 shard
    if not shard_files and all_shards:
        shard_files = all_shards

    return shard_files, layer_to_shard


def _get_shards_for_layers_standard(model_path: str, layers: List[int]) -> Tuple[set, dict]:
    """
    根据layer范围确定需要加载的shard文件 (标准 model-*.safetensors 格式)

    与 _get_shards_for_layers 相同逻辑，但读取 model.safetensors.index.json。

    Returns:
        (shard_files, layer_to_shard)
    """
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        # 没有 index 文件，可能只有一个 model.safetensors
        single = os.path.join(model_path, "model.safetensors")
        if os.path.exists(single):
            return {os.path.basename(single)}, {}
        return set(), {}

    with open(index_path, 'r') as f:
        index_data = json.load(f)
    weight_map = index_data.get("weight_map", {})

    layer_set = set(layers)
    shard_files = set()
    layer_to_shard = {}

    all_shards = set(weight_map.values())

    for k, v in weight_map.items():
        import re
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", k)
        if match is not None:
            layer_num = int(match.group(1))
            if layer_num in layer_set:
                shard_files.add(v)
                layer_to_shard[layer_num] = v

        if _is_non_layer_key(k):
            shard_files.add(v)

    if not shard_files and all_shards:
        shard_files = all_shards

    return shard_files, layer_to_shard


def build_weight_index(model_path: str) -> Dict[str, str]:
    """从 index.json 读取 weight_map (key→shard_file 映射)

    依次尝试:
      1. model.safetensors.index.json (标准HF或symlink后的msmodelslim)
      2. quant_model_weights.safetensors.index.json (msmodelslim原始)

    Returns:
        weight_map dict, key 是参数名, value 是 shard 文件名
        如果没有 index 文件，返回空 dict
    """
    index_names = [
        "model.safetensors.index.json",
        "quant_model_weights.safetensors.index.json",
    ]
    # Some official exports keep a model-specific index filename (or omit the
    # index entirely) while still using standard safetensors shards.
    try:
        index_names.extend(
            name for name in sorted(os.listdir(model_path))
            if name.endswith(".safetensors.index.json") and name not in index_names
        )
    except OSError:
        pass
    indexed_map = {}
    for index_name in index_names:
        index_path = os.path.join(model_path, index_name)
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            indexed_map.update(index_data.get("weight_map", {}))
    if indexed_map:
        # Keep the index fast path, then fill any omitted keys from shard
        # metadata.  This is needed for a few V4 exports whose index contains
        # only the converted/runtime subset while the native ffn expert keys
        # live in the same shard files.
        try:
            shard_names = sorted(
                name for name in os.listdir(model_path)
                if name.endswith(".safetensors")
            )
            for shard_name in shard_names:
                shard_path = os.path.join(model_path, shard_name)
                with safe_open(shard_path, framework="pt") as handle:
                    for key in handle.keys():
                        indexed_map.setdefault(key, shard_name)
        except OSError:
            pass
        return indexed_map
    # Unsharded HF checkpoints do not carry an index.json.  Build the same
    # key->file view from safetensors metadata so indexed loading remains
    # lazy and works for small standalone drafts such as DSpark.
    for single_name in ("model.safetensors", "quant_model_weights.safetensors"):
        single_path = os.path.join(model_path, single_name)
        if not os.path.exists(single_path):
            continue
        with safe_open(single_path, framework="pt") as handle:
            return {key: single_name for key in handle.keys()}
    # Last-resort discovery for unindexed multi-shard exports.  Opening a
    # safetensors file only reads metadata, not the large tensor payload.
    try:
        discovered = {}
        for shard_name in sorted(
            name for name in os.listdir(model_path)
            if name.endswith(".safetensors")
        ):
            with safe_open(os.path.join(model_path, shard_name), framework="pt") as handle:
                for key in handle.keys():
                    discovered.setdefault(key, shard_name)
        return discovered
    except OSError:
        return {}


class ShardWeightReader:
    """按需从 safetensors 文件读取单个 key，避免 load_file() 读取整个 shard

    用法:
        reader = ShardWeightReader(model_path, weight_map)
        tensor = reader.get_tensor(key)   # 只读这一个 key
        reader.close()                     # 关闭所有文件句柄
    """

    def __init__(self, model_path: str, weight_map: Dict[str, str]):
        self.model_path = model_path
        self.weight_map = weight_map
        self._sf_cache = {}  # fname -> safe_open handle

    def _get_sf(self, fname: str):
        if fname not in self._sf_cache:
            fpath = os.path.join(self.model_path, fname)
            self._sf_cache[fname] = safe_open(fpath, framework='pt')
        return self._sf_cache[fname]

    def _resolve_key(self, key: str):
        """Return ``(resolved_key, shard_file)`` including prefix fallback."""
        direct_candidates = [key]
        for prefix in ("model.model.", "model."):
            if key.startswith(prefix):
                direct_candidates.append(key[len(prefix):])
        # DeepSeek-V4 ModelScope exports use ``layers.N.*``, ``embed.*`` and
        # ``head.*`` without the Transformers ``model.`` container prefix.
        if key.startswith("model.layers."):
            direct_candidates.append(key[len("model."):])
        if key == "model.embed_tokens.weight":
            direct_candidates.append("embed.weight")
        if key == "model.norm.weight":
            direct_candidates.append("norm.weight")
        if key == "lm_head.weight":
            direct_candidates.append("head.weight")
        for alt_key in direct_candidates:
            fname = self.weight_map.get(alt_key)
            if fname is not None:
                return alt_key, fname
        native_key = self._deepseek_v4_native_key(key)
        if native_key is not None:
            fname = self.weight_map.get(native_key)
            if fname is not None:
                return native_key, fname
        return None, None

    def _deepseek_v4_native_key(self, key: str) -> Optional[str]:
        """Map the official in-memory V4 names to released checkpoint names."""
        if not any(".attn." in name or ".ffn." in name
                   for name in self.weight_map):
            return None
        if key == "model.embed_tokens.weight" and "embed.weight" in self.weight_map:
            return "embed.weight"
        if key == "model.norm.weight" and "norm.weight" in self.weight_map:
            return "norm.weight"
        if key == "lm_head.weight" and "head.weight" in self.weight_map:
            return "head.weight"
        # Released V4 checkpoints keep HyperHead tensors at the root, while
        # Transformers exposes them below model.hc_head.  They are required
        # by the final norm+head/logits pass.
        hc_head_keys = {
            "model.hc_head.hc_fn": "hc_head_fn",
            "model.hc_head.hc_base": "hc_head_base",
            "model.hc_head.hc_scale": "hc_head_scale",
            "model.model.hc_head.hc_fn": "hc_head_fn",
            "model.model.hc_head.hc_base": "hc_head_base",
            "model.model.hc_head.hc_scale": "hc_head_scale",
            "hc_head.hc_fn": "hc_head_fn",
            "hc_head.hc_base": "hc_head_base",
            "hc_head.hc_scale": "hc_head_scale",
        }
        native_hc_key = hc_head_keys.get(key)
        if native_hc_key is not None and native_hc_key in self.weight_map:
            return native_hc_key
        candidate = key
        replacements = (
            (".input_layernorm.", ".attn_norm."),
            (".post_attention_layernorm.", ".ffn_norm."),
            (".attn_hc.fn", ".hc_attn_fn"),
            (".attn_hc.base", ".hc_attn_base"),
            (".attn_hc.scale", ".hc_attn_scale"),
            (".ffn_hc.fn", ".hc_ffn_fn"),
            (".ffn_hc.base", ".hc_ffn_base"),
            (".ffn_hc.scale", ".hc_ffn_scale"),
        )
        for source, target in replacements:
            candidate = candidate.replace(source, target)
        if ".self_attn.compressor.indexer." in candidate:
            candidate = candidate.replace(
                ".self_attn.compressor.indexer.", ".attn.indexer."
            )
            for source, target in (
                ("scorer.weights_proj.", "weights_proj."),
                ("q_b_proj.", "wq_b."),
                ("kv_proj.", "compressor.wkv."),
                ("gate_proj.", "compressor.wgate."),
                ("kv_norm.", "compressor.norm."),
                ("position_bias", "compressor.ape"),
            ):
                candidate = candidate.replace(source, target)
        elif ".self_attn.compressor." in candidate:
            candidate = candidate.replace(
                ".self_attn.compressor.", ".attn.compressor."
            )
            for source, target in (
                ("kv_proj.", "wkv."), ("gate_proj.", "wgate."),
                ("kv_norm.", "norm."), ("position_bias", "ape"),
            ):
                candidate = candidate.replace(source, target)
        elif ".self_attn." in candidate:
            candidate = candidate.replace(".self_attn.", ".attn.")
            for source, target in (
                ("q_a_proj.", "wq_a."), ("q_b_proj.", "wq_b."),
                ("q_a_norm.", "q_norm."), ("kv_proj.", "wkv."),
                ("o_a_proj.", "wo_a."),
                ("o_b_proj.", "wo_b."), ("sinks", "attn_sink"),
            ):
                candidate = candidate.replace(source, target)
        if ".mlp." in candidate:
            candidate = candidate.replace(".mlp.", ".ffn.")
            for source, target in (
                ("shared_experts.gate_proj.", "shared_experts.w1."),
                ("shared_experts.down_proj.", "shared_experts.w2."),
                ("shared_experts.up_proj.", "shared_experts.w3."),
                ("gate.e_score_correction_bias", "gate.bias"),
            ):
                candidate = candidate.replace(source, target)
            # Released V4 checkpoints keep every routed expert as three
            # independent projections.  Transformers converts these names and
            # packs all experts into gate_up_proj/down_proj while loading.
            if ".experts." in candidate:
                candidate = candidate.replace(".gate_proj.", ".w1.")
                candidate = candidate.replace(".down_proj.", ".w2.")
                candidate = candidate.replace(".up_proj.", ".w3.")
        if candidate in self.weight_map:
            return candidate
        if candidate.startswith("model."):
            unprefixed = candidate[len("model."):]
            if unprefixed in self.weight_map:
                return unprefixed
        return None

    def get_tensor(self, key: str):
        """读取单个 key 的 tensor，如果 key 不在 weight_map 中返回 None

        自动处理 ForConditionalGeneration 的前缀不匹配:
          weight_map key: "model.language_model.layers..."
          query key:      "model.model.language_model.layers..."
        尝试去掉一个 "model." 前缀作为 fallback
        """
        resolved_key, fname = self._resolve_key(key)
        if fname is None:
            return None
        return self._get_sf(fname).get_tensor(resolved_key)

    def get_tensor_shape(self, key: str):
        """Return a tensor shape from safetensors metadata without loading it."""
        resolved_key, fname = self._resolve_key(key)
        if fname is None:
            return None
        sf = self._get_sf(fname)
        get_slice = getattr(sf, "get_slice", None)
        if callable(get_slice):
            return tuple(get_slice(resolved_key).get_shape())
        return tuple(sf.get_tensor(resolved_key).shape)

    def get_tensor_slice(self, key: str, index: int,
                         expected_first_dim: int = None):
        """Read one first-dimension slice without loading a packed tensor.

        Qwen3.5/3.6 stores routed experts in large 3D tensors.  Quantization
        metadata is sliced only when it carries the same expert dimension;
        scalar or shared metadata is returned unchanged.
        """
        resolved_key, fname = self._resolve_key(key)
        if fname is None:
            return None
        sf = self._get_sf(fname)
        get_slice = getattr(sf, "get_slice", None)
        if not callable(get_slice):
            tensor = sf.get_tensor(resolved_key)
            if (
                tensor.dim() > 0
                and 0 <= index < tensor.shape[0]
                and (expected_first_dim is None
                     or tensor.shape[0] == expected_first_dim)
            ):
                return tensor[index]
            return tensor
        tensor_slice = get_slice(resolved_key)
        shape = tuple(tensor_slice.get_shape())
        if (
            shape
            and 0 <= index < shape[0]
            and (expected_first_dim is None or shape[0] == expected_first_dim)
        ):
            return tensor_slice[index:index + 1][0]
        return sf.get_tensor(resolved_key)

    def get_keys_for_layers(self, layer_indices: List[int], model: nn.Module = None,
                            include_non_layer: bool = False) -> set:
        """返回指定层需要的所有 weight_map key

        Args:
            layer_indices: 层索引列表
            model: 可选，用于匹配参数名；如果为 None 则从 weight_map 的 key 模式推断
            include_non_layer: 是否包含 embed_tokens/norm/lm_head 等非层权重
        """
        needed = set()
        for key in self.weight_map:
            if '.layers.' in key or key.startswith('layers.'):
                _collect_layer_key(key, layer_indices, needed)
            elif include_non_layer and _is_non_layer_key(key):
                needed.add(key)
        return needed

    def close(self):
        """关闭所有 safetensors 文件句柄，释放 mmap 内存"""
        for sf in self._sf_cache.values():
            try:
                close = getattr(sf, "close", None)
                if callable(close):
                    close()
                else:
                    exit_context = getattr(sf, "__exit__", None)
                    if callable(exit_context):
                        exit_context(None, None, None)
            except Exception as e:
                logger.warning(f"Failed to close safetensors file handle: {e}")
        self._sf_cache.clear()

    def __del__(self):
        self.close()


def add_deepseek_v4_checkpoint_aliases(
    model: nn.Module,
    weights: Dict[str, torch.Tensor],
    quant_desc: Optional[dict] = None,
) -> int:
    """Expose official DeepSeek-V4 checkpoint names under runtime names.

    Transformers normally performs this conversion in its checkpoint loader.
    Boundary mode deliberately bypasses that loader, so it must expose the same
    aliases before generic dequantization.  Tensor storage is shared and routed
    experts remain compact for the streaming loader.
    """
    if not any(".attn." in key or ".ffn." in key for key in weights):
        return 0
    resolver = ShardWeightReader("", {key: "" for key in weights})
    runtime_names = {
        name for name, _ in model.named_parameters()
    } | {
        name for name, _ in model.named_buffers()
    }
    metadata_suffixes = (
        "weight_scale", "weight_offset", "deq_scale", "input_scale",
    )
    added = 0
    for runtime_name in runtime_names:
        if ".mlp.experts.gate_up_proj" in runtime_name or \
                ".mlp.experts.down_proj" in runtime_name:
            continue
        native_name = resolver._deepseek_v4_native_key(runtime_name)
        if native_name is None:
            continue
        if runtime_name not in weights:
            weights[runtime_name] = weights[native_name]
            added += 1
        if quant_desc is not None:
            qtype = quant_desc.get(
                native_name, quant_desc.get(parse_base_name(native_name))
            )
            if qtype is not None:
                quant_desc.setdefault(runtime_name, qtype)
        if not runtime_name.endswith(".weight"):
            continue
        runtime_base = runtime_name[:-7]
        native_base = native_name[:-7]
        for suffix in metadata_suffixes:
            runtime_meta = f"{runtime_base}.{suffix}"
            native_meta = f"{native_base}.{suffix}"
            if runtime_meta not in weights and native_meta in weights:
                weights[runtime_meta] = weights[native_meta]
                added += 1
    return added


def _collect_layer_key(key: str, layer_indices: List[int], needed: set):
    """Parse a "model.layers.N.*" key and add it to `needed` if N is in layer_indices.

    Helper extracted from ShardWeightReader.get_keys_for_layers to reduce block depth.
    """
    import re
    match = re.search(r'(?:^|\.)layers\.(\d+)\.', key)
    if match is None:
        return
    try:
        layer_num = int(match.group(1))
    except (ValueError, IndexError):
        return
    if layer_num in layer_indices:
        needed.add(key)


def _is_non_layer_key(key: str) -> bool:
    """True if key refers to embed_tokens / lm_head / norm (non-layer weight)."""
    return (
        any(x in key for x in ['embed_tokens', 'lm_head', '.norm.'])
        or key.startswith(('embed.', 'norm.', 'head.'))
    )


def _is_auxiliary_predictor_name(name: str) -> bool:
    """Whether a parameter belongs to optional V4 MTP/DSpark predictor blocks."""
    parts = name.split('.')
    return any(part in {"mtp", "nextn_predict_layers", "dspark"}
               for part in parts)


def _decide_should_load(name: str, layer_idx: Optional[int], layer_set: set,
                        load_embed_only: bool, load_norm_head_only: bool,
                        verbose: bool, param=None,
                        include_auxiliary: bool = False) -> bool:
    """Decide whether a parameter should be loaded given the layer selection mode.

    Extracted from load_layer_weights_indexed to reduce cyclomatic complexity.
    When `param` is provided and mode=load_embed_only, emits the original DEBUG EMBED log.
    """
    if not include_auxiliary and _is_auxiliary_predictor_name(name):
        return False
    if load_embed_only:
        if layer_idx is None and "norm" not in name and "lm_head" not in name:
            if verbose and 'embed' in name and param is not None:
                logger.info(f"  [DEBUG EMBED] will load: {name}, shape={param.shape}, device={param.device}")
            return True
        return False
    if load_norm_head_only:
        return layer_idx is None and ("norm" in name or "lm_head" in name)
    if include_auxiliary and _is_auxiliary_predictor_name(name):
        return True
    return layer_idx is not None and layer_idx in layer_set


def _load_ct_param(name: str, param, sf_reader: ShardWeightReader,
                   dtype: torch.dtype) -> bool:
    """Load one compressed-tensors parameter. Returns True if loaded."""
    tensor = sf_reader.get_tensor(name)
    scale_key = name.replace('.weight', '.weight_scale')
    scale_tensor = sf_reader.get_tensor(scale_key)
    if scale_tensor is None:
        scale_key = name.replace('.weight', '.weight_scale_inv')
        scale_tensor = sf_reader.get_tensor(scale_key)
    if tensor is None and name.endswith('.weight'):
        # compressed-tensors MXFP4 stores the packed nibbles separately.
        packed_key = name.replace('.weight', '.weight_packed')
        packed = sf_reader.get_tensor(packed_key)
        if packed is not None and scale_tensor is not None:
            param.data = dequantize_weight_mxfp4(
                packed, scale_tensor, dtype=dtype,
            )
            return True
    if tensor is None:
        return False
    if str(tensor.dtype).startswith("torch.float8"):
        param.data = dequantize_native_fp8_weight(
            tensor, scale_tensor, dtype=dtype, scale_name=scale_key.rsplit('.', 1)[-1]
        )
    elif tensor.dtype == torch.int8 and scale_tensor is not None:
        w_fp = tensor.to(dtype) * scale_tensor.to(dtype)
        param.data = w_fp
    else:
        param.data = tensor.to(dtype)
    return True


def _dequant_msslim_weight(weight_data: Tensor, quant_type: str, quant_name: str,
                           sf_reader: ShardWeightReader, dtype: torch.dtype) -> Tuple[Optional[Tensor], str]:
    """Dequantize a msmodelslim quantized weight per quant_type.

    Returns ``(weight_or_None, status)`` where status is:
      ``"loaded"``  — weight dequantized successfully
      ``"skipped"`` — quant_type recognized but required scales missing/invalid (skip silently)
      ``"unknown"`` — quant_type not in any handled branch (caller may warn)

    Extracted from load_layer_weights_indexed to reduce cyclomatic complexity.
    """
    quant_type = normalize_quant_type(quant_type)
    if quant_type == "DEEPSEEK_FP4":
        scale = sf_reader.get_tensor(f"{quant_name}.scale")
        if scale is None:
            return None, "skipped"
        decoded = dequantize_deepseek_v4_native_weight(
            weight_data, scale, dtype=dtype
        )
        return (decoded, "loaded") if decoded is not None else (None, "skipped")
    if quant_type in ("W8A8_DYNAMIC", "W8A8_MIX"):
        weight_scale = sf_reader.get_tensor(f"{quant_name}.weight_scale")
        weight_offset = sf_reader.get_tensor(f"{quant_name}.weight_offset")
        if weight_scale is None:
            return None, "skipped"
        return dequantize_weight_dynamic(
            weight_data, weight_scale, weight_offset, dtype
        ), "loaded"

    if quant_type in _MX_QUANT_TYPES:
        weight_scale_u8 = sf_reader.get_tensor(f"{quant_name}.weight_scale")
        if weight_scale_u8 is None:
            return None, "skipped"
        return dequantize_weight_mx(
            weight_data, weight_scale_u8, quant_type, dtype=dtype
        ), "loaded"

    if quant_type in ("W8A8", "W8A8S"):
        deq_scale = sf_reader.get_tensor(f"{quant_name}.deq_scale")
        input_scale = sf_reader.get_tensor(f"{quant_name}.input_scale")
        quant_bias = sf_reader.get_tensor(f"{quant_name}.quant_bias")
        if deq_scale is not None:
            w_fp, _ = dequantize_weight_static(
                weight_data, deq_scale, input_scale, quant_bias, dtype
            )
            return w_fp, "loaded"
        # fallback: 也可有 weight_scale
        weight_scale = sf_reader.get_tensor(f"{quant_name}.weight_scale")
        weight_offset = sf_reader.get_tensor(f"{quant_name}.weight_offset")
        if weight_scale is not None:
            return dequantize_weight_dynamic(
                weight_data, weight_scale, weight_offset, dtype
            ), "loaded"
        return None, "skipped"

    if quant_type in (
        "W4A16", "W4A8_DYNAMIC", "W4A8", "W4A4_DYNAMIC", "W4A4_LAOS",
        "W4A4_INT4_PER_GROUP",
    ):
        weight_scale = sf_reader.get_tensor(f"{quant_name}.weight_scale")
        weight_offset = sf_reader.get_tensor(f"{quant_name}.weight_offset")
        if weight_data.dtype == torch.int8 and weight_scale is not None:
            unpacked = unpack_int4_to_int8(weight_data)
            return dequantize_weight_dynamic(
                unpacked, weight_scale, weight_offset, dtype
            ), "loaded"
        return None, "skipped"

    return None, "unknown"


_KNOWN_QUANT_TYPES = (
    "W8A8", "W8A8S", "W8A8_DYNAMIC", "W8A8_MIX", "W8A8_MXFP8",
    "W4A8_MXFP", "W4A4_MXFP4",
    "W4A16", "W4A8_DYNAMIC", "W4A8",
    "W4A4_DYNAMIC", "W4A4_LAOS", "W4A4_INT4_PER_GROUP",
)


def _assign_param_checked(name: str, param, value: Tensor, source: str) -> None:
    """Assign a decoded tensor without silently changing the skeleton shape."""
    expected = tuple(param.shape)
    actual = tuple(value.shape)
    if actual != expected:
        raise ValueError(
            f"quantized weight shape mismatch for {name}: model expects "
            f"{expected}, but {source} produced {actual}. Check the checkpoint "
            "config and quant_model_description.json."
        )
    param.data = value


def _recover_misclassified_quant_weight(name: str, param, weight_data: Tensor,
                                        sf_reader: ShardWeightReader,
                                        quant_desc_str: dict,
                                        dtype: torch.dtype) -> Tuple[Optional[Tensor], Optional[str]]:
    """Recover a quantized tensor whose exact quant descriptor is absent.

    Some msModelSlim Qwen MoE exports omit descriptors for split shared-expert
    gate/up projections. Treating those tensors as FLOAT silently halves a
    Linear dimension. Only accept a recovery whose decoded shape exactly
    matches the model skeleton.
    """
    if weight_data.dim() != 2:
        return None, None

    weight_key = name if name.endswith(".weight") else f"{name}.weight"
    quant_name = weight_key.rsplit('.', 1)[0]
    expected = tuple(param.shape)
    raw_shape = tuple(weight_data.shape)
    candidates = []

    weight_scale = sf_reader.get_tensor(f"{quant_name}.weight_scale")
    deq_scale = sf_reader.get_tensor(f"{quant_name}.deq_scale")
    scale_is_e8m0 = (
        weight_scale is not None and weight_scale.dtype == torch.uint8
    )

    # Prefer a documented sibling projection when its scale representation is
    # compatible with this tensor. MX formats use uint8 E8M0 scales; dynamic
    # INT4/INT8 formats use ordinary floating-point scales.
    parent = quant_name.rsplit('.', 1)[0]
    for projection in ("gate_proj", "up_proj", "down_proj", "gate_up_proj"):
        sibling_key = f"{parent}.{projection}.weight"
        sibling_type = normalize_quant_type(quant_desc_str.get(
            sibling_key, quant_desc_str.get(parse_base_name(sibling_key))
        ))
        sibling_is_mx = sibling_type in _MX_QUANT_TYPES
        if (
            sibling_type in _KNOWN_QUANT_TYPES
            and sibling_is_mx == scale_is_e8m0
            and sibling_type not in candidates
        ):
            candidates.append(sibling_type)

    # Dynamic/LAOS INT4 is packed on out_features (dim 0); MXFP4 is packed on
    # in_features (dim 1). These checks distinguish the two without guessing.
    if weight_scale is not None:
        if not scale_is_e8m0 and (raw_shape[0] * 2, raw_shape[1]) == expected:
            candidates.append("W4A4_DYNAMIC")
        if scale_is_e8m0 and (raw_shape[0], raw_shape[1] * 2) == expected:
            candidates.append("W4A8_MXFP")
        if not scale_is_e8m0 and raw_shape == expected:
            candidates.append("W8A8_DYNAMIC")
        if scale_is_e8m0 and raw_shape == expected:
            candidates.append("W8A8_MXFP8")
    if deq_scale is not None and raw_shape == expected:
        candidates.append("W8A8")

    tried = set()
    for quant_type in candidates:
        if quant_type in tried:
            continue
        tried.add(quant_type)
        try:
            decoded, status = _dequant_msslim_weight(
                weight_data, quant_type, quant_name, sf_reader, dtype
            )
        except (RuntimeError, ValueError, IndexError):
            continue
        if (
            status == "loaded"
            and decoded is not None
            and tuple(decoded.shape) == expected
        ):
            return decoded, quant_type
    return None, None


def _load_msslim_quant_param(name: str, param, sf_reader: ShardWeightReader,
                             quant_desc_str: dict, dtype: torch.dtype,
                             use_fake_quant: bool, verbose: bool) -> bool:
    """Load one msmodelslim-quantized parameter. Returns True if loaded, False if skipped.

    Extracted from load_layer_weights_indexed to reduce cyclomatic complexity.
    """
    if name.endswith(".weight"):
        weight_key = name
    else:
        weight_key = f"{name}.weight"

    # Official DeepSeek-V4 reference weights have no msModelSlim descriptor.
    # Their packed-FP4 experts and block-FP8 dense projections both use a
    # sibling ``.scale`` tensor, so decode them before the generic FLOAT path.
    if quant_desc_str.get(_DEEPSEEK_V4_FP4_MARKER) == "DEEPSEEK_FP4":
        weight_data = sf_reader.get_tensor(name)
        if weight_data is None:
            weight_data = sf_reader.get_tensor(weight_key)
        if weight_data is not None:
            quant_name = weight_key.rsplit(".", 1)[0]
            decoded = dequantize_deepseek_v4_native_weight(
                weight_data,
                sf_reader.get_tensor(f"{quant_name}.scale"),
                dtype=dtype,
            )
            if decoded is not None:
                _assign_param_checked(
                    name, param, decoded, "official DeepSeek-V4 FP4/FP8 decode"
                )
                param._acc_quant_type = "DEEPSEEK_FP4"
                return True
    # Native FP8 checkpoints (for example GLM-5.3) carry no msModelSlim
    # descriptor.  Decode both the common ``weight_scale_inv`` spelling and
    # the vLLM-compatible ``weight_scale`` spelling before descriptor lookup.
    if (
        quant_desc_str.get(_NATIVE_FP8_MARKER) == "FP8_E4M3"
        and name.endswith(".weight")
    ):
        weight_data = sf_reader.get_tensor(name)
        if weight_data is None:
            weight_data = sf_reader.get_tensor(weight_key)
        if weight_data is not None:
            quant_name = weight_key.rsplit(".", 1)[0]
            scale_key = None
            scale = None
            for suffix in ("weight_scale_inv", "weight_scale", "scale_inv", "scale"):
                candidate = sf_reader.get_tensor(f"{quant_name}.{suffix}")
                if candidate is not None:
                    scale_key, scale = suffix, candidate
                    break
            # FP8 quantization configs commonly exempt embeddings, norms and
            # a few projections.  Do not reinterpret those ordinary BF16/
            # FP16 tensors as byte-level FP8 just because the model has a
            # native-FP8 config; only decode an actual FP8/uint8 payload.
            if not (
                str(weight_data.dtype).startswith("torch.float8")
                or weight_data.dtype == torch.uint8
            ):
                _assign_param_checked(name, param, weight_data.to(dtype), "FLOAT exempt tensor")
                param._acc_quant_type = "FLOAT"
                return True
            decoded = dequantize_native_fp8_weight(
                weight_data, scale, dtype=dtype, scale_name=scale_key or "scale"
            )
            _assign_param_checked(name, param, decoded, "native FP8 decode")
            param._acc_quant_type = "FP8_E4M3"
            return True
    quant_type = normalize_quant_type(
        _quant_desc_type_for_reader(quant_desc_str, weight_key, sf_reader)
    )

    # FLOAT 或未知类型: 直接读 tensor 赋值 (与原 FLOAT/unknown 合并分支一致)
    if quant_type == "FLOAT" or quant_type not in _KNOWN_QUANT_TYPES:
        weight_data = sf_reader.get_tensor(name)
        if weight_data is None:
            weight_data = sf_reader.get_tensor(weight_key)
        if weight_data is not None:
            looks_quantized = (
                not weight_data.dtype.is_floating_point
                or str(weight_data.dtype).startswith("torch.float8")
            )
            if looks_quantized:
                recovered, inferred_type = _recover_misclassified_quant_weight(
                    name, param, weight_data, sf_reader, quant_desc_str, dtype
                )
                if recovered is None:
                    raise ValueError(
                        f"quantized checkpoint tensor {weight_key} was classified "
                        "as FLOAT and its quantization format could not be "
                        "inferred safely; fix quant_model_description.json"
                    )
                projection = weight_key.rsplit('.', 2)[-2]
                warning_key = (inferred_type, projection)
                log = (
                    logger.warning
                    if warning_key not in _INFERRED_QUANT_WARNINGS
                    else logger.debug
                )
                log(
                    "quant descriptor missing for %s; inferred %s from "
                    "checkpoint metadata and model shape",
                    weight_key, inferred_type,
                )
                _INFERRED_QUANT_WARNINGS.add(warning_key)
                _assign_param_checked(
                    name, param, recovered,
                    f"inferred {inferred_type} dequantization",
                )
                param._acc_quant_type = inferred_type
                return True
            _assign_param_checked(
                name, param, weight_data.to(dtype), "FLOAT checkpoint tensor"
            )
            param._acc_quant_type = "FLOAT"
            return True
        return False

    if use_fake_quant:
        return False

    weight_data = sf_reader.get_tensor(weight_key)
    if weight_data is None:
        return False

    quant_name = weight_key.rsplit('.', 1)[0]
    w_fp, status = _dequant_msslim_weight(
        weight_data, quant_type, quant_name, sf_reader, dtype
    )
    if status == "unknown":
        # 与原 else 分支行为一致: 未知类型打印 warning
        if verbose:
            logger.warning(f"indexed加载: 未知量化类型 {quant_type}: {name}")
        return False
    if w_fp is None:
        # 已知类型但缺少 scale/bias: 静默跳过 (与原 skipped_count += 1 一致)
        return False
    _assign_param_checked(name, param, w_fp, f"{quant_type} dequantization")
    param._acc_quant_type = quant_type
    return True


def _load_named_buffers(model: nn.Module, sf_reader: ShardWeightReader,
                        layer_set: set, load_embed_only: bool,
                        load_norm_head_only: bool,
                        strict_embed_buffer_ids: Optional[set] = None,
                        strict_final_buffer_ids: Optional[set] = None,
                        include_auxiliary: bool = False) -> int:
    """Load register_buffer tensors (e.g. e_score_correction_bias). Returns loaded count.

    Extracted from load_layer_weights_indexed.
    """
    loaded_count = 0
    if not hasattr(model, 'named_buffers'):
        return loaded_count
    for bname, buf in model.named_buffers():
        if buf is None:
            continue
        if (
            load_embed_only
            and strict_embed_buffer_ids is not None
            and id(buf) not in strict_embed_buffer_ids
        ):
            continue
        if (
            load_norm_head_only
            and strict_final_buffer_ids is not None
            and id(buf) not in strict_final_buffer_ids
        ):
            continue
        b_layer_idx = _extract_layer_idx(bname)
        should_load = (
            id(buf) in strict_final_buffer_ids
            if load_norm_head_only and strict_final_buffer_ids is not None
            else _decide_should_load(
                bname, b_layer_idx, layer_set,
                load_embed_only, load_norm_head_only, False,
                include_auxiliary=include_auxiliary,
            )
        )
        if not should_load:
            continue
        # 从 weight_map 找 key (msmodelslim 量化场景 key 在 quant_model_weights-*.safetensors)
        try:
            b_data = sf_reader.get_tensor(bname)
        except Exception:
            b_data = None
        if b_data is None and bname.endswith('.e_score_correction_bias'):
            # 尝试 base name 解析路径兜底
            continue
        if b_data is not None and b_data.shape == buf.shape:
            # 直接搬运到目标 device (buffers 通常很小, ~256 float32)
            buf.data = b_data.to(buf.dtype).to(buf.device)
            loaded_count += 1
    return loaded_count


def _is_3d_quant_expert(name: str, param, is_quant: bool,
                        quant_desc_str: Optional[dict]) -> bool:
    """True if param is a 3D packed routed expert weight in a quantized model."""
    if param.dim() != 3 or not is_quant or quant_desc_str is None:
        return False
    if 'experts.gate_up_proj' in name:
        return True
    return 'experts.down_proj' in name and 'shared_expert' not in name


def _is_3d_nonquant_expert(name: str, param, is_quant: bool) -> bool:
    """True if param is a 3D packed routed expert weight in a non-quantized model."""
    if param.dim() != 3 or is_quant:
        return False
    if 'experts.gate_up_proj' in name:
        return True
    return 'experts.down_proj' in name and 'shared_expert' not in name


def _load_single_param(name: str, param, sf_reader: ShardWeightReader,
                       quant_desc_str: Optional[dict], is_quant: bool, is_ct: bool,
                       dtype: torch.dtype, use_fake_quant: bool,
                       skip_routed_experts: bool, verbose: bool) -> str:
    """Load a single parameter through the appropriate path.

    Returns:
      ``"loaded"``   — parameter was loaded
      ``"skipped"``  — parameter was skipped (not needed or load failed)

    Extracted from load_layer_weights_indexed to reduce CC from 21 to ≤20.
    """
    # compressed-tensors
    if is_ct:
        if _load_ct_param(name, param, sf_reader, dtype):
            return "loaded"
        return "skipped"

    # 3D packed expert params (quant model)
    result = _try_load_3d_quant_expert(
        name, param, sf_reader, quant_desc_str, is_quant, dtype,
        use_fake_quant, skip_routed_experts,
    )
    if result in ("loaded", "skipped"):
        return result

    # msmodelslim 量化
    if is_quant and quant_desc_str is not None:
        if _load_msslim_quant_param(
            name, param, sf_reader, quant_desc_str, dtype, use_fake_quant, verbose
        ):
            return "loaded"
        return "skipped"

    # 3D packed expert params (non-quant model)
    result = _try_load_3d_nonquant_expert(
        name, param, sf_reader, is_quant, dtype, skip_routed_experts,
    )
    if result in ("loaded", "skipped", "fallback_loaded"):
        return "loaded"

    # 非量化模型: 直接读 key
    if _load_raw_param(name, param, sf_reader, dtype, verbose):
        return "loaded"
    return "skipped"


def load_layer_weights_indexed(
    model: nn.Module,
    model_path: str,
    layers: List[int],
    device: str,
    dtype: torch.dtype,
    weight_map: Dict[str, str],
    sf_reader: ShardWeightReader,
    is_quant: bool = False,
    is_ct: bool = False,
    quant_desc: dict = None,
    use_fake_quant: bool = False,
    skip_routed_experts: bool = False,
    strict_embed_only: bool = False,
    strict_final_only: bool = False,
    include_auxiliary: bool = False,
    verbose: bool = True,
) -> int:
    """按需加载指定层权重 (使用 safe_open + get_tensor，避免 load_file 读整个 shard)

    与 load_layer_weights 功能相同，但 I/O 模式不同：
    - load_layer_weights: load_file() 读整个 shard 文件到内存 → 按 key 匹配
    - 本函数: 从 weight_map 查 key→file → safe_open 按 key 读取 → 立即赋值

    支持: 非量化 / compressed-tensors / msmodelslim (W8A8, W8A8_DYNAMIC, W8A8_MIX, W8A8_MXFP8, W4A8_MXFP, W4A4_MXFP4)
    对 MoE 模型（如 GLM5.1）大幅减少 I/O 和内存峰值。

    Args:
        model: 模型骨架
        model_path: 模型路径
        layers: 要加载的层索引列表
        device: 目标设备
        dtype: 数据类型
        weight_map: key→shard_file 映射 (from build_weight_index)
        sf_reader: ShardWeightReader 实例
        is_quant: 是否量化模型 (msmodelslim)
        is_ct: 是否 compressed-tensors
        quant_desc: 量化描述 dict (quant_model_description.json, 仅 msmodelslim 需要)
        use_fake_quant: 是否 fake quant
        skip_routed_experts: 跳过 routed experts 权重加载 (由 _moe_forward_chunked streaming 读取)
        strict_embed_only: layers=[] 时只加载实际 text embedding 模块，跳过视觉/projector
        strict_final_only: layers=[-1] 时只加载结构解析得到的 final norm/lm_head，
            避免 Kimi streaming 骨架中其他顶层 meta norm 被名称匹配误加载
        include_auxiliary: 完整模型加载时同时加载 V4 的 ``mtp``/DSpark
            predictor 模块；分片 L1 默认关闭，避免把可选预测头当作主干层
        verbose: 是否打印进度

    Returns:
        加载的权重数量
    """
    from .utils import parse_base_name

    layer_set = set(layers)
    load_embed_only = (layers == [])
    load_norm_head_only = (-1 in layers)

    loaded_count = 0
    skipped_count = 0

    strict_embed_param_ids = None
    strict_embed_buffer_ids = None
    if load_embed_only and strict_embed_only:
        from .utils import get_embed_module
        embed_module = get_embed_module(model)
        if embed_module is None:
            raise RuntimeError("Strict embedding load requested but no embedding module was found")
        strict_embed_param_ids = {id(param) for param in embed_module.parameters()}
        strict_embed_buffer_ids = {id(buf) for buf in embed_module.buffers()}

    strict_final_param_ids = None
    strict_final_buffer_ids = None
    if load_norm_head_only and strict_final_only:
        from .utils import get_norm_module, get_lm_head_module, get_model_components
        text_model = get_model_components(model).text_model
        final_modules = tuple(
            module for module in (
                get_norm_module(model), get_lm_head_module(model),
                getattr(text_model, "hc_head", None),
            ) if module is not None
        )
        if not final_modules:
            raise RuntimeError(
                "Strict final load requested but no final norm/lm_head was found"
            )
        strict_final_param_ids = {
            id(param) for module in final_modules
            for param in module.parameters()
        }
        strict_final_buffer_ids = {
            id(buf) for module in final_modules
            for buf in module.buffers()
        }

    # 修正 quant_desc key 与 named_modules() 的前缀不匹配
    if is_quant and quant_desc is not None:
        quant_desc = normalize_quant_desc_keys(quant_desc, model)

    # msmodelslim: 过滤 quant_desc 中非 string 的 metadata key
    if is_quant and quant_desc is not None:
        quant_desc_str = {k: v for k, v in quant_desc.items() if isinstance(v, str)}
    else:
        quant_desc_str = None

    for name, param in model.named_parameters():
        if (
            strict_embed_param_ids is not None
            and id(param) not in strict_embed_param_ids
        ):
            skipped_count += 1
            continue
        if (
            strict_final_param_ids is not None
            and id(param) not in strict_final_param_ids
        ):
            skipped_count += 1
            continue
        layer_idx = _extract_layer_idx(name)
        should_load = (
            id(param) in strict_final_param_ids
            if strict_final_param_ids is not None
            else _decide_should_load(
                name, layer_idx, layer_set, load_embed_only,
                load_norm_head_only, verbose, param,
                include_auxiliary=include_auxiliary,
            )
        )
        if not should_load:
            skipped_count += 1
            continue

        # grouped_dual streams routed experts directly from safetensors.
        # Covers both 3D packed experts and ModuleList experts (Kimi K3).
        if (
            skip_routed_experts
            and '.experts.' in name
            and 'shared_expert' not in name
        ):
            skipped_count += 1
            continue

        result = _load_single_param(
            name, param, sf_reader, quant_desc_str, is_quant, is_ct,
            dtype, use_fake_quant, skip_routed_experts, verbose,
        )
        if result == "loaded":
            loaded_count += 1
        else:
            skipped_count += 1

    # 加载 register_buffer 注册的非参数张量
    loaded_count += _load_named_buffers(
        model, sf_reader, layer_set, load_embed_only, load_norm_head_only,
        strict_embed_buffer_ids=strict_embed_buffer_ids,
        strict_final_buffer_ids=strict_final_buffer_ids,
        include_auxiliary=include_auxiliary,
    )

    if verbose:
        logger.info(f"  加载了 {loaded_count} 个权重 (含 buffer), 跳过了 {skipped_count} 个")

    return loaded_count


def _load_raw_param(name: str, param, sf_reader: ShardWeightReader,
                    dtype: torch.dtype, verbose: bool) -> bool:
    """非量化模型: 直接从 safetensors 读 key 并赋值到 param.

    Returns True if loaded, False if the key was missing. Preserves the original
    [DEBUG EMBED] info logs on both the loaded and failed paths.
    """
    weight_data = sf_reader.get_tensor(name)
    if weight_data is not None:
        param.data = weight_data.to(dtype)
        if verbose and 'embed' in name:
            logger.info(
                f"  [DEBUG EMBED] loaded: {name}, shape={weight_data.shape}, "
                f"norm={weight_data.norm():.6f}"
            )
        return True
    if 'embed' in name:
        logger.info(f"  [DEBUG EMBED] FAILED to load: {name}, weight_data=None")
    return False


def _try_load_3d_quant_expert(name: str, param, sf_reader: ShardWeightReader,
                              quant_desc_str: Optional[dict], is_quant: bool,
                              dtype: torch.dtype, use_fake_quant: bool,
                              skip_routed_experts: bool) -> str:
    """尝试加载 3D 量化 expert 参数 (GLM5.1 gate_up_proj / down_proj).

    Returns:
      ``"loaded"``       — successfully loaded into param.data
      ``"skipped"``      — explicitly skipped (skip_routed_experts or _load_3d_expert_indexed failed)
      ``"passthrough"``  — not a 3D quant expert; caller continues with other branches

    Extracted from load_layer_weights_indexed to reduce cyclomatic complexity.
    """
    if not _is_3d_quant_expert(name, param, is_quant, quant_desc_str):
        return "passthrough"
    # skip_routed_experts: 跳过 routed experts 加载，由 _moe_forward_chunked streaming 读取
    if 'shared_expert' not in name and skip_routed_experts:
        return "skipped"
    loaded = _load_3d_expert_indexed(
        name, param, sf_reader, quant_desc_str, dtype, use_fake_quant
    )
    return "loaded" if loaded else "skipped"


def _try_load_3d_nonquant_expert(name: str, param, sf_reader: ShardWeightReader,
                                 is_quant: bool, dtype: torch.dtype,
                                 skip_routed_experts: bool) -> str:
    """尝试加载 3D 非量化 expert 参数 (GLM5.1 ref model).

    Returns:
      ``"loaded"``         — loaded via _load_3d_expert_nonquant_indexed
      ``"fallback_loaded"`` — fallback to reading raw packed 3D tensor (Qwen3.5 case)
      ``"skipped"``        — explicitly skipped (skip_routed_experts, or none of the paths succeeded)
      ``"passthrough"``    — not a 3D non-quant expert; caller continues with other branches

    Extracted from load_layer_weights_indexed to reduce cyclomatic complexity.
    """
    if not _is_3d_nonquant_expert(name, param, is_quant):
        return "passthrough"
    # skip_routed_experts: 同上，跳过 routed experts
    if 'shared_expert' not in name and skip_routed_experts:
        return "skipped"
    loaded = _load_3d_expert_nonquant_indexed(name, param, sf_reader, dtype)
    if loaded:
        return "loaded"
    # Fallback: safetensors may already store packed 3D tensor (Qwen3.5)
    weight_data = sf_reader.get_tensor(name)
    if weight_data is not None:
        param.data = weight_data.to(dtype)
        return "fallback_loaded"
    return "skipped"


def _count_experts(sf_reader: ShardWeightReader, expert_prefix: str,
                   is_gate_up: bool, use_get_tensor_fallback: bool = True) -> int:
    """Count experts by probing weight_map keys (fast path).

    Args:
      use_get_tensor_fallback: when True and sf_reader has no weight_map, fall back
        to probing via get_tensor (matches _load_3d_expert_indexed behavior); when
        False, return 0 if weight_map is absent (matches _load_3d_expert_nonquant_indexed).

    Extracted from _load_3d_expert_indexed / _load_3d_expert_nonquant_indexed.
    """
    sub_name = "gate_proj" if is_gate_up else "down_proj"
    test_key_tmpl = f"{expert_prefix}.{{i}}.{sub_name}.weight"
    if hasattr(sf_reader, 'weight_map'):
        return _count_experts_in_map(sf_reader, test_key_tmpl)
    if not use_get_tensor_fallback:
        return 0
    # Fallback: probe via get_tensor
    num_experts = 0
    for i in range(300):
        test_key = test_key_tmpl.format(i=i)
        if sf_reader.get_tensor(test_key) is not None:
            num_experts = i + 1
        elif num_experts > 0:
            break
    return num_experts


def _count_experts_in_map(sf_reader: ShardWeightReader, test_key_tmpl: str) -> int:
    """Fast path: count experts by checking weight_map membership (no I/O)."""
    num_experts = 0
    for i in range(300):
        test_key = test_key_tmpl.format(i=i)
        resolved_key, _ = sf_reader._resolve_key(test_key)
        if resolved_key is not None:
            num_experts = i + 1
        elif num_experts > 0:
            break
    return num_experts


def _quant_desc_type_for_reader(quant_desc: dict, key: str,
                                sf_reader: ShardWeightReader) -> str:
    """Resolve a quant type through both runtime and official V4 names."""
    if quant_desc.get(_NATIVE_FP8_MARKER) == "FP8_E4M3":
        return "FP8_E4M3"
    base = parse_base_name(key)
    candidates = [key, base]
    for prefix in ("model.model.", "model."):
        if key.startswith(prefix):
            candidates.extend((key[len(prefix):], base[len(prefix):] if base.startswith(prefix) else base))
    for candidate in candidates:
        value = quant_desc.get(candidate)
        if value is not None:
            return value
    resolver = getattr(sf_reader, "_deepseek_v4_native_key", None)
    native = resolver(key) if callable(resolver) else None
    if native is None:
        return "FLOAT"
    native_base = parse_base_name(native)
    return quant_desc.get(native, quant_desc.get(native_base, "FLOAT"))


def _load_3d_expert_indexed_gate_up(name, param, sf_reader, expert_prefix,
                                    num_experts: int, quant_desc_str, dtype,
                                    use_fake_quant, is_w4_packed: bool,
                                    is_w4_unpack: bool) -> bool:
    """gate_up_proj path of _load_3d_expert_indexed. Returns True if loaded successfully."""
    first_gate = sf_reader.get_tensor(f"{expert_prefix}.0.gate_proj.weight")
    if first_gate is None:
        return False
    intermediate_dim = first_gate.shape[0]
    hidden_dim = first_gate.shape[1]
    if is_w4_packed:
        hidden_dim *= 2  # MXFP4: packed along in_features (dim 1)
    if is_w4_unpack:
        intermediate_dim *= 2  # W4A8_DYNAMIC: unpack_int4_to_int8 doubles out_features (dim 0)

    packed = torch.zeros(num_experts, 2 * intermediate_dim, hidden_dim, dtype=dtype)

    for i in range(num_experts):
        if not _fill_gate_up_slice_i(
            i, expert_prefix, quant_desc_str, sf_reader, dtype,
            use_fake_quant, packed, intermediate_dim,
        ):
            return False
    param.data = packed
    return True


def _fill_gate_up_slice_i(i: int, expert_prefix: str, quant_desc_str: dict,
                          sf_reader: ShardWeightReader, dtype: torch.dtype,
                          use_fake_quant: bool, packed: Tensor,
                          intermediate_dim: int) -> bool:
    """Fill packed[i, :intermediate_dim, :] (gate) and packed[i, intermediate_dim:, :] (up).

    Returns False on missing tensor. Note: u_type uses g_base (not u_base) — preserved
    from the original implementation.
    """
    g_key = f"{expert_prefix}.{i}.gate_proj.weight"
    g_base = parse_base_name(g_key)
    g_type = _quant_desc_type_for_reader(quant_desc_str, g_key, sf_reader)
    g_data = _dequant_single_expert_indexed(g_key, g_type, sf_reader, dtype, use_fake_quant)
    if g_data is None:
        return False
    packed[i, :intermediate_dim, :] = g_data

    u_key = f"{expert_prefix}.{i}.up_proj.weight"
    u_type = _quant_desc_type_for_reader(quant_desc_str, u_key, sf_reader)
    u_data = _dequant_single_expert_indexed(u_key, u_type, sf_reader, dtype, use_fake_quant)
    if u_data is None:
        return False
    packed[i, intermediate_dim:, :] = u_data
    return True


def _load_3d_expert_indexed_down(name, param, sf_reader, expert_prefix,
                                 num_experts: int, quant_desc_str, dtype,
                                 use_fake_quant, is_w4_packed: bool,
                                 is_w4_unpack: bool) -> bool:
    """down_proj path of _load_3d_expert_indexed. Returns True if loaded successfully."""
    first_down = sf_reader.get_tensor(f"{expert_prefix}.0.down_proj.weight")
    if first_down is None:
        return False
    hidden_dim = first_down.shape[0]
    intermediate_dim = first_down.shape[1]
    if is_w4_packed:
        intermediate_dim *= 2  # MXFP4: packed along in_features (dim 1)
    if is_w4_unpack:
        hidden_dim *= 2  # W4A8_DYNAMIC: unpack_int4_to_int8 doubles out_features (dim 0)

    packed = torch.zeros(num_experts, hidden_dim, intermediate_dim, dtype=dtype)

    for i in range(num_experts):
        d_key = f"{expert_prefix}.{i}.down_proj.weight"
        d_type = _quant_desc_type_for_reader(quant_desc_str, d_key, sf_reader)
        d_data = _dequant_single_expert_indexed(d_key, d_type, sf_reader, dtype, use_fake_quant)
        if d_data is None:
            return False
        packed[i] = d_data
    param.data = packed
    return True


def _load_3d_expert_indexed(name, param, sf_reader, quant_desc_str, dtype, use_fake_quant):
    """Load GLM5.1 3D packed expert params from per-expert quantized keys via sf_reader.

    Same logic as _load_3d_expert_quant but uses ShardWeightReader instead of all_weights dict.
    Returns True if loaded successfully.
    """
    if 'experts.gate_up_proj' in name:
        expert_prefix = name.rsplit('.gate_up_proj', 1)[0]
        is_gate_up = True
    elif 'experts.down_proj' in name:
        expert_prefix = name.rsplit('.down_proj', 1)[0]
        is_gate_up = False
    else:
        return False

    # Count experts from weight_map keys (fast, no I/O)
    num_experts = _count_experts(sf_reader, expert_prefix, is_gate_up)
    if num_experts == 0:
        return False

    # Detect W4 packed quant type to adjust dim
    g0_key = f"{expert_prefix}.0.gate_proj.weight" if is_gate_up else f"{expert_prefix}.0.down_proj.weight"
    g0_type = _quant_desc_type_for_reader(quant_desc_str, g0_key, sf_reader)
    is_w4_packed = g0_type in _MXFP4_QUANT_TYPES
    is_w4_unpack = g0_type in (
        "W4A8_DYNAMIC", "W4A16", "W4A8", "W4A4_DYNAMIC", "W4A4_LAOS",
        "W4A4_INT4_PER_GROUP",
    )

    if is_gate_up:
        return _load_3d_expert_indexed_gate_up(
            name, param, sf_reader, expert_prefix, num_experts,
            quant_desc_str, dtype, use_fake_quant,
            is_w4_packed, is_w4_unpack,
        )
    return _load_3d_expert_indexed_down(
        name, param, sf_reader, expert_prefix, num_experts,
        quant_desc_str, dtype, use_fake_quant,
        is_w4_packed, is_w4_unpack,
    )


def _load_3d_expert_nonquant_gate_up(name, param, sf_reader, expert_prefix,
                                     num_experts: int, dtype) -> bool:
    """gate_up_proj path of _load_3d_expert_nonquant_indexed."""
    first_gate = sf_reader.get_tensor(f"{expert_prefix}.0.gate_proj.weight")
    if first_gate is None:
        return False
    intermediate_dim = first_gate.shape[0]
    hidden_dim = first_gate.shape[1]

    packed = torch.zeros(num_experts, 2 * intermediate_dim, hidden_dim, dtype=dtype)

    for i in range(num_experts):
        g_data = sf_reader.get_tensor(f"{expert_prefix}.{i}.gate_proj.weight")
        if g_data is None:
            return False
        packed[i, :intermediate_dim, :] = g_data.to(dtype)

        u_data = sf_reader.get_tensor(f"{expert_prefix}.{i}.up_proj.weight")
        if u_data is None:
            return False
        packed[i, intermediate_dim:, :] = u_data.to(dtype)
    param.data = packed
    return True


def _load_3d_expert_nonquant_down(name, param, sf_reader, expert_prefix,
                                  num_experts: int, dtype) -> bool:
    """down_proj path of _load_3d_expert_nonquant_indexed."""
    first_down = sf_reader.get_tensor(f"{expert_prefix}.0.down_proj.weight")
    if first_down is None:
        return False
    hidden_dim = first_down.shape[0]
    intermediate_dim = first_down.shape[1]

    packed = torch.zeros(num_experts, hidden_dim, intermediate_dim, dtype=dtype)

    for i in range(num_experts):
        d_data = sf_reader.get_tensor(f"{expert_prefix}.{i}.down_proj.weight")
        if d_data is None:
            return False
        packed[i] = d_data.to(dtype)
    param.data = packed
    return True


def _load_3d_expert_nonquant_indexed(name, param, sf_reader, dtype):
    """Load GLM5.1 3D packed expert params for non-quantized (BF16) model.

    Ref model stores experts as per-expert keys (experts.0.gate_proj.weight etc.)
    but the model structure expects 3D packed (experts.gate_up_proj [N, 2*inter, hidden]).
    This function reads per-expert keys and repacks into 3D.
    """
    if 'experts.gate_up_proj' in name:
        expert_prefix = name.rsplit('.gate_up_proj', 1)[0]
        is_gate_up = True
    elif 'experts.down_proj' in name:
        expert_prefix = name.rsplit('.down_proj', 1)[0]
        is_gate_up = False
    else:
        return False

    # Count experts from weight_map (fast)
    num_experts = _count_experts(sf_reader, expert_prefix, is_gate_up,
                                 use_get_tensor_fallback=False)
    if num_experts == 0:
        return False

    if is_gate_up:
        return _load_3d_expert_nonquant_gate_up(
            name, param, sf_reader, expert_prefix, num_experts, dtype
        )
    return _load_3d_expert_nonquant_down(
        name, param, sf_reader, expert_prefix, num_experts, dtype
    )


def _dequant_single_expert_indexed(weight_key, quant_type, sf_reader, dtype, use_fake_quant):
    """Dequantize a single expert weight using ShardWeightReader."""
    quant_type = normalize_quant_type(quant_type)
    if quant_type == "FLOAT":
        weight_data = sf_reader.get_tensor(weight_key)
        if weight_data is not None:
            return weight_data.to(dtype)
        return None

    if quant_type == "FP8_E4M3":
        weight_data = sf_reader.get_tensor(weight_key)
        if weight_data is None:
            return None
        if not (
            str(weight_data.dtype).startswith("torch.float8")
            or weight_data.dtype == torch.uint8
        ):
            return weight_data.to(dtype)
        quant_name = weight_key.rsplit('.', 1)[0]
        scale_key = None
        scale = None
        for suffix in ("weight_scale_inv", "weight_scale", "scale_inv", "scale"):
            candidate = sf_reader.get_tensor(f"{quant_name}.{suffix}")
            if candidate is not None:
                scale_key, scale = suffix, candidate
                break
        return dequantize_native_fp8_weight(
            weight_data, scale, dtype=dtype, scale_name=scale_key or "scale"
        )

    if use_fake_quant:
        return None

    weight_data = sf_reader.get_tensor(weight_key)
    if weight_data is None:
        return None

    quant_name = weight_key.rsplit('.', 1)[0]

    if quant_type in ("W8A8_DYNAMIC", "W8A8_MIX"):
        weight_scale = sf_reader.get_tensor(f"{quant_name}.weight_scale")
        weight_offset = sf_reader.get_tensor(f"{quant_name}.weight_offset")
        if weight_scale is not None:
            return dequantize_weight_dynamic(
                weight_data, weight_scale, weight_offset, dtype
            )
        return None

    if quant_type in _MX_QUANT_TYPES:
        weight_scale_u8 = sf_reader.get_tensor(f"{quant_name}.weight_scale")
        if weight_scale_u8 is not None:
            return dequantize_weight_mx(weight_data, weight_scale_u8, quant_type, dtype=dtype)
        return None

    if quant_type in (
        "W4A8_DYNAMIC", "W4A16", "W4A8", "W4A4_DYNAMIC", "W4A4_LAOS",
        "W4A4_INT4_PER_GROUP",
    ):
        weight_scale = sf_reader.get_tensor(f"{quant_name}.weight_scale")
        weight_offset = sf_reader.get_tensor(f"{quant_name}.weight_offset")
        if weight_scale is not None and weight_data.dtype == torch.int8:
            unpacked = unpack_int4_to_int8(weight_data)
            return dequantize_weight_dynamic(unpacked, weight_scale, weight_offset, dtype)
        return None

    deq_scale = sf_reader.get_tensor(f"{quant_name}.deq_scale")
    input_scale = sf_reader.get_tensor(f"{quant_name}.input_scale")
    if deq_scale is not None:
        w_fp, _ = dequantize_weight_static(
            weight_data, deq_scale, input_scale, dtype=dtype
        )
        return w_fp

    return None


def _create_dspark_model_from_config(model_path: str, dtype: torch.dtype):
    """Instantiate a supported standalone DSpark draft on the current device.

    DeepSpec and Speculators intentionally use different model/config classes.
    Resolve those classes explicitly because ``AutoModelForCausalLM`` would
    otherwise instantiate the verifier architecture from ``model_type``.
    """
    from .dspark import load_dspark_contract

    contract = load_dspark_contract(model_path)
    try:
        if contract.architecture == "Qwen3DSparkModel":
            from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            config._attn_implementation = "eager"
            model = Qwen3DSparkModel(config)
        elif contract.architecture == "Gemma4DSparkModel":
            from deepspec.modeling.dspark.gemma4.modeling import Gemma4DSparkModel
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            config._attn_implementation = "eager"
            model = Gemma4DSparkModel(config)
        else:
            from speculators.models.dspark import (
                DSparkDraftModel,
                DSparkSpeculatorConfig,
            )
            config = DSparkSpeculatorConfig.from_pretrained(model_path)
            transformer_config = getattr(config, "transformer_layer_config", None)
            if transformer_config is not None:
                transformer_config._attn_implementation = "eager"
            model = DSparkDraftModel(config)
    except ImportError as exc:
        dependency = (
            "DeepSpec source on PYTHONPATH (clone the official repository and "
            "install its requirements)"
            if contract.flavor == "deepspec"
            else "speculators (pip install speculators)"
        )
        raise ImportError(
            f"Loading {contract.architecture} requires {dependency}"
        ) from exc

    # The constructor runs under a meta-device context.  Preserve the requested
    # comparison dtype before to_empty() allocates real storage.
    for parameter in model.parameters():
        if parameter.is_floating_point():
            parameter.data = parameter.data.to(dtype=dtype)
    return model, config


def _initialize_rotary_modules(model: nn.Module,
                               device: Optional[str] = None) -> None:
    """Rebuild non-persistent RoPE buffers after ``to_empty`` materialization.

    With ``device=None`` every rotary module remains on the device assigned by
    layer sharding.  DeepSeek-V4 owns several rotary modules and keeps separate
    ``main``/``compress`` frequency buffers, all of which must be rebuilt.
    """
    seen = set()
    for module in model.modules():
        has_frequency_buffer = (
            hasattr(module, "inv_freq")
            or bool(getattr(module, "layer_types", None))
        )
        if id(module) in seen or not has_frequency_buffer:
            continue
        seen.add(id(module))
        config = getattr(module, "config", None)
        compute = getattr(type(module), "compute_default_rope_parameters", None)
        rope_init_fn = getattr(module, "rope_init_fn", None)
        if config is None or (compute is None and not callable(rope_init_fn)):
            continue
        target_device = device
        if target_device is None:
            tensors = tuple(module.parameters()) + tuple(module.buffers())
            materialized = [tensor for tensor in tensors if not tensor.is_meta]
            target_device = str(materialized[0].device) if materialized else "cpu"
        module.to(target_device)
        layer_types = getattr(module, "layer_types", None)
        if layer_types:
            for layer_type in layer_types:
                rope_type = getattr(module, "rope_type", {}).get(layer_type, "default")
                if rope_type == "default":
                    initializer = compute
                else:
                    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
                    initializer = ROPE_INIT_FUNCTIONS[rope_type]
                inv_freq, attention_scaling = initializer(
                    config, device=target_device, layer_type=layer_type
                )
                setattr(module, f"{layer_type}_inv_freq", inv_freq)
                if hasattr(module, f"{layer_type}_original_inv_freq"):
                    setattr(module, f"{layer_type}_original_inv_freq", inv_freq.clone())
                setattr(module, f"{layer_type}_attention_scaling", attention_scaling)
            continue
        initializer = compute if compute is not None else rope_init_fn
        inv_freq, attention_scaling = initializer(config, device=target_device)
        module.inv_freq = inv_freq
        if hasattr(module, "attention_scaling"):
            module.attention_scaling = attention_scaling
        if hasattr(module, "original_inv_freq"):
            module.original_inv_freq = inv_freq.clone()


def _load_speculators_verifier_weights(
    model: nn.Module,
    weight_map: Dict[str, str],
    verifier_model: Optional[str],
) -> None:
    """Populate verifier-owned embeddings/heads omitted by Speculators drafts.

    Standard Speculators checkpoints intentionally do not duplicate these large
    tensors. Their custom ``from_pretrained`` path fetches them from the
    configured verifier; acc_bench's indexed loader must preserve that contract.
    """
    if not verifier_model:
        raise ValueError(
            "Speculators DSpark requires "
            "speculators_config.verifier.name_or_path"
        )
    loader = getattr(model, "load_verifier_weights", None)
    if not callable(loader):
        raise RuntimeError(
            "Installed speculators DSpark model has no load_verifier_weights() API"
        )

    # Upstream only reloads some shared tensors when their placeholder contains
    # NaNs. ``to_empty`` destroys the constructor's NaN sentinel, so restore it
    # only for tensors that really are absent from the draft checkpoint.
    verifier_owned = (
        "embed_tokens.weight",
        "lm_head.weight",
        "verifier_lm_head.weight",
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not any(name.endswith(suffix) for suffix in verifier_owned):
                continue
            if any(key == name or key.endswith(f".{name}") for key in weight_map):
                continue
            if parameter.is_floating_point():
                parameter.fill_(float("nan"))

    try:
        loader()
    except Exception as exc:
        raise RuntimeError(
            "Failed to load Speculators verifier-owned DSpark weights from "
            f"{verifier_model!r}; make that model/path resolvable in this environment"
        ) from exc


_MISSING_GLOBAL = object()


def _kimi_expert_counts(config) -> set:
    """Return routed-expert counts from outer and nested Kimi configs."""
    counts = set()
    for candidate in (config, getattr(config, "text_config", None)):
        value = getattr(candidate, "num_experts", None)
        if isinstance(value, int) and value > 1:
            counts.add(value)
    return counts


def _load_kimi_modeling_modules(config, model_path: str) -> tuple:
    """Import Kimi remote code without constructing the model yet."""
    auto_map = getattr(config, "auto_map", None) or {}
    class_ref = auto_map.get("AutoModelForCausalLM") or auto_map.get("AutoModel")
    if isinstance(class_ref, (list, tuple)):
        class_ref = class_ref[0] if class_ref else None
    if not class_ref:
        raise RuntimeError(
            "Kimi streaming skeleton requires an AutoModelForCausalLM entry "
            "in config.auto_map"
        )

    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        model_cls = get_class_from_dynamic_module(class_ref, model_path)
    except Exception as exc:
        raise RuntimeError(
            "Unable to import Kimi remote code before streaming skeleton creation"
        ) from exc

    package_prefix = model_cls.__module__.rsplit(".", 1)[0]
    modules = []
    for name, module in tuple(sys.modules.items()):
        if module is None or not name.startswith(package_prefix + "."):
            continue
        if (
            name == model_cls.__module__
            or name.endswith(".modeling_kimi_linear")
            or hasattr(module, "KimiBlockSparseMLP")
        ):
            modules.append(module)
    has_linear_modeling = any(
        getattr(module, "__name__", "").endswith(".modeling_kimi_linear")
        or hasattr(module, "KimiBlockSparseMLP")
        for module in modules
    )
    if not has_linear_modeling:
        raise RuntimeError(
            "Kimi remote code was imported but modeling_kimi_linear was not found; "
            "refusing to fall back to a full 896-expert skeleton"
        )
    return model_cls, modules


@contextmanager
def _kimi_streaming_construction(config, model_path: str, enabled: bool):
    """Collapse Kimi's per-layer expert construction while building on meta.

    Kimi's remote code uses ``range(config.num_experts)`` to instantiate one
    Python module per routed expert. grouped_dual never executes those modules;
    it streams the selected expert tensors directly from safetensors. During
    skeleton construction, replace only that module-local range with range(1)
    and suppress Kimi post_init weight initialization, which is unnecessary for
    a checkpoint-backed meta skeleton.
    """
    if not enabled:
        yield set(), None
        return

    expert_counts = _kimi_expert_counts(config)
    if not expert_counts:
        raise RuntimeError(
            "Kimi streaming skeleton requested but num_experts was not found"
        )
    model_cls, modules = _load_kimi_modeling_modules(config, model_path)

    patched_globals = []
    patched_post_init = []

    def streaming_range(*args):
        if len(args) == 1 and args[0] in expert_counts:
            return builtins.range(1)
        return builtins.range(*args)

    try:
        for module in modules:
            old_range = module.__dict__.get("range", _MISSING_GLOBAL)
            module.__dict__["range"] = streaming_range
            patched_globals.append((module, old_range))

            for value in tuple(module.__dict__.values()):
                if not isinstance(value, type) or not value.__name__.startswith("Kimi"):
                    continue
                if not hasattr(value, "post_init"):
                    continue
                old_post_init = value.__dict__.get("post_init", _MISSING_GLOBAL)
                value.post_init = lambda self: None
                patched_post_init.append((value, old_post_init))
        yield expert_counts, model_cls
    finally:
        for cls, old_post_init in reversed(patched_post_init):
            if old_post_init is _MISSING_GLOBAL:
                delattr(cls, "post_init")
            else:
                cls.post_init = old_post_init
        for module, old_range in reversed(patched_globals):
            if old_range is _MISSING_GLOBAL:
                module.__dict__.pop("range", None)
            else:
                module.__dict__["range"] = old_range


def _finalize_kimi_streaming_skeleton(model: nn.Module, expert_counts: set) -> int:
    """Replace the one prototype expert per Kimi MoE layer with Identity."""
    collapsed_layers = 0
    for module in model.modules():
        experts = getattr(module, "experts", None)
        module_config = getattr(module, "config", None)
        configured_count = getattr(module_config, "num_experts", None)
        if configured_count not in expert_counts or not isinstance(experts, nn.ModuleList):
            continue
        if len(experts) != 1:
            raise RuntimeError(
                "Kimi streaming skeleton failed to collapse routed experts: "
                f"expected 1 prototype, found {len(experts)}"
            )
        experts[0] = nn.Identity()
        module.num_experts = configured_count
        if hasattr(module, "experts_per_rank"):
            module.experts_per_rank = configured_count
        module._acc_bench_streaming_experts = True
        collapsed_layers += 1

    if collapsed_layers == 0:
        raise RuntimeError(
            "Kimi streaming skeleton did not find any routed MoE layers; "
            "refusing to materialize the full model"
        )
    return collapsed_layers


def _force_kimi_eager_attention(model: nn.Module) -> None:
    """Undo Kimi remote code's forced FlashAttention choice before L1 forward."""
    seen_configs = set()
    for candidate in (
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
    ):
        if candidate is not None:
            seen_configs.add(id(candidate))
            candidate._attn_implementation = "eager"
    for module in model.modules():
        module_config = getattr(module, "config", None)
        if module_config is not None and id(module_config) not in seen_configs:
            module_config._attn_implementation = "eager"
            seen_configs.add(id(module_config))
        if hasattr(module, "_use_flash_attention_2"):
            module._use_flash_attention_2 = False


def create_model_skeleton(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = True,
    streaming_experts: bool = False,
) -> nn.Module:
    """
    只创建模型骨架（meta device），不加载任何权重。

    用于分片加载时复用模型结构，避免每次重建模型导致的不稳定。

    Args:
        model_path: 模型路径
        dtype: 数据类型
        verbose: 是否打印进度
        streaming_experts: Kimi grouped_dual 使用轻量 expert 占位并保持整模 meta

    Returns:
        模型骨架（meta device）
    """
    from .dspark import is_dspark_checkpoint
    from .kimi_fla_shim import is_kimi_k3_checkpoint

    require_model_runtime_support(model_path)
    is_dspark = is_dspark_checkpoint(model_path)
    use_kimi_streaming_skeleton = (
        streaming_experts and is_kimi_k3_checkpoint(model_path)
    )
    # 加载config。DSpark 不能交给 AutoModelForCausalLM，否则会按 verifier 的
    # model_type 创建普通 Qwen/Gemma 模型并丢掉 Markov/confidence head。
    if is_dspark:
        config = None
    else:
        config = _load_model_config(model_path)

    # 多模态 ForConditionalGeneration 模型 (如 Qwen3.6): num_hidden_layers 在 text_config 里
    if config is not None and not hasattr(config, 'num_hidden_layers'):
        if hasattr(config, 'text_config') and hasattr(config.text_config, 'num_hidden_layers'):
            config.num_hidden_layers = config.text_config.num_hidden_layers
        elif hasattr(config, 'num_layer'):
            config.num_hidden_layers = config.num_layer
        else:
            raise ValueError("config 中缺少 num_hidden_layers 和 num_layer，无法确定模型层数")

    if verbose:
        if is_dspark:
            from .dspark import load_dspark_contract
            logger.info(
                "  创建 DSpark 模型骨架 (meta device): %s 层",
                load_dspark_contract(model_path).draft_layers,
            )
        else:
            logger.info(f"  创建模型骨架 (meta device): {config.num_hidden_layers} 层")

    # 创建模型结构 (meta device)
    # 多模态 ForConditionalGeneration 模型不能用 AutoModelForCausalLM，需用 architectures 指定的类
    with _kimi_streaming_construction(
        config, model_path, enabled=use_kimi_streaming_skeleton
    ) as kimi_streaming_context, torch.device('meta'):
        kimi_expert_counts, kimi_model_cls = kimi_streaming_context
        if is_dspark:
            model, config = _create_dspark_model_from_config(model_path, dtype)
        elif use_kimi_streaming_skeleton:
            # Reuse the class imported before the scoped range/post_init patch;
            # calling AutoModel again could reload the dynamic module and lose it.
            config._attn_implementation = "eager"
            model = kimi_model_cls._from_config(
                config,
                torch_dtype=dtype,
                attn_implementation="eager",
            )
        else:
            try:
                model = AutoModelForCausalLM.from_config(
                    config,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    attn_implementation='eager',
                )
            except (AttributeError, TypeError) as e:
                # Fallback: 用 architectures 中指定的模型类
                architectures = getattr(config, 'architectures', []) or []
                if not architectures:
                    raise ValueError(f"AutoModelForCausalLM 失败且无 architectures: {e}")
                cls_name = architectures[0]
                # 从 transformers 顶层获取模型类
                import transformers
                model_cls = getattr(transformers, cls_name, None)
                if model_cls is None:
                    raise ValueError(f"transformers 中找不到 {cls_name}: {e}")
                model = model_cls(config)
                # 设置 dtype
                for p in model.parameters():
                    p.data = p.data.to(dtype)
    if use_kimi_streaming_skeleton:
        collapsed_layers = _finalize_kimi_streaming_skeleton(
            model, kimi_expert_counts
        )
        _force_kimi_eager_attention(model)
        model._acc_bench_lazy_meta = True
        if verbose:
            logger.info(
                "  Kimi streaming skeleton: %d MoE layers; routed experts stay "
                "in safetensors and inactive layers stay on meta",
                collapsed_layers,
            )
    else:
        model = model.to_empty(device='cpu')

    return model


def _load_dspark_model_for_comparison(
    model_path: str,
    device: str,
    dtype: torch.dtype,
    verbose: bool,
) -> Tuple[nn.Module, bool]:
    """Load a standalone DSpark draft through acc_bench indexed dequantization."""
    from .dspark import load_dspark_contract

    is_quant = is_quantized_model(model_path)
    is_ct = is_compressed_tensors_model(model_path)
    model = create_model_skeleton(model_path, dtype=dtype, verbose=verbose)
    contract = load_dspark_contract(model_path)
    weight_map = build_weight_index(model_path)
    if not weight_map:
        raise FileNotFoundError(
            f"No safetensors weights/index found in DSpark checkpoint: {model_path}"
        )
    reader = ShardWeightReader(model_path, weight_map)
    quant_desc = None
    quant_desc_path = os.path.join(model_path, "quant_model_description.json")
    if is_quant and not is_ct and os.path.isfile(quant_desc_path):
        with open(quant_desc_path, "r", encoding="utf-8") as handle:
            quant_desc = json.load(handle)
    try:
        common = dict(
            model=model,
            model_path=model_path,
            device=device,
            dtype=dtype,
            weight_map=weight_map,
            sf_reader=reader,
            is_quant=is_quant,
            is_ct=is_ct,
            quant_desc=quant_desc,
            verbose=verbose,
        )
        # [] loads embeddings/projections/heads except final norm/lm_head;
        # layer IDs load the draft backbone; -1 loads final norm/lm_head.
        load_layer_weights_indexed(layers=[], **common)
        load_layer_weights_indexed(
            layers=list(range(contract.draft_layers)), **common
        )
        load_layer_weights_indexed(layers=[-1], **common)
    finally:
        reader.close()
    if contract.flavor == "speculators":
        _load_speculators_verifier_weights(
            model, weight_map, contract.verifier_model
        )
    model = model.to(device)
    _initialize_rotary_modules(model, device)
    return model, is_quant


def _extract_layer_idx(param_name: str) -> Optional[int]:
    """从参数名提取层索引"""
    parts = param_name.split('.')
    for i, p in enumerate(parts):
        if p.isdigit():
            if i > 0 and parts[i-1] in ('layers', 'h', 'block'):
                return int(p)
            if i > 0 and parts[i-1] in ('model', 'transformer'):
                if i + 1 < len(parts) and parts[i+1].isdigit():
                    return int(p)
    return None


def _replace_unloaded_layers(model: nn.Module, layers: List[int], verbose: bool = True):
    """替换不在加载列表中的层为PassThroughIdentity"""
    layer_set = set(layers)

    # 找到decoder layers
    from .utils import get_decoder_layers
    decoder_layers = get_decoder_layers(model)

    if decoder_layers:
        replaced = 0
        for i, layer in enumerate(decoder_layers):
            if i not in layer_set:
                decoder_layers[i] = PassThroughIdentity()
                replaced += 1

        if verbose and replaced > 0:
            logger.info(f"  [逐层加载] 替换了 {replaced} 个层为PassThroughIdentity")



def load_model_for_comparison(
    model_path: str,
    device: str = "npu:0",
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = True,
    use_fake_quant: bool = True,
    layers: Optional[List[int]] = None,
    keep_all_layers: bool = False,
) -> Tuple[nn.Module, bool]:
    """Load model for accuracy comparison.

    For sharded mode (layers != None): returns skeleton model (no weights loaded).
    For full mode: loads all weights using from_pretrained.
    """
    from .dspark import is_dspark_checkpoint

    is_quant = is_quantized_model(model_path)
    native_fp8 = _config_declares_native_fp8(_read_raw_model_config(model_path))

    if is_dspark_checkpoint(model_path):
        if layers is not None:
            return create_model_skeleton(model_path, dtype, verbose=verbose), is_quant
        if use_fake_quant:
            raise NotImplementedError(
                "DSpark fake_quant loading is not supported yet; use "
                "--quant_method dequantize so ref/quant run the same draft API"
            )
        single_device = (
            device.split(',')[0]
            if isinstance(device, str) and ',' in device
            else device
        )
        return _load_dspark_model_for_comparison(
            model_path, single_device, dtype, verbose
        )

    # Sharded mode: return skeleton only
    if layers is not None:
        model = create_model_skeleton(model_path, dtype, verbose=verbose)
        return model, is_quant

    # Full-load mode (single device)
    single_device = device.split(',')[0] if isinstance(device, str) and ',' in device else device

    if is_quant and use_fake_quant and not native_fp8:
        if verbose:
            logger.info(f"  量化模型, FakeQuant 模式加载")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation='eager',
        )
        model = model.to(single_device)
        return model, True

    if is_quant:
        if verbose:
            mode = "原生 FP8→BF16 反量化" if native_fp8 else "反量化"
            logger.info(f"  量化模型, {mode}模式加载")
        # Load with dequantize: skeleton + load all weights + dequant
        model = create_model_skeleton(model_path, dtype, verbose=verbose)
        weight_map = build_weight_index(model_path)
        reader = ShardWeightReader(model_path, weight_map)
        num_layers = model.config.num_hidden_layers
        all_layers = list(range(num_layers))
        quant_desc = native_quant_description(model_path)
        load_layer_weights_indexed(model, model_path, all_layers, single_device, dtype,
                                   weight_map, reader, is_quant=is_quant,
                                   quant_desc=quant_desc,
                                   include_auxiliary=True, verbose=verbose)
        reader.close()
        model = model.to(single_device)
        return model, True

    # Non-quantized model
    if verbose:
        logger.info(f"  非量化模型, 直接加载")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation='eager',
    )
    model = model.to(single_device)
    return model, False


def move_layers_to_device(model: nn.Module, layer_indices: List[int], device: str,
                          clear_others: bool = True, skip_3d_experts: bool = True):
    """只移动指定层的参数到设备

    Args:
        model: 模型
        layer_indices: 要移动的层索引列表
        device: 目标设备
        clear_others: True=先把其他层的设备清空（移到CPU），避免内存累积
        skip_3d_experts: True=跳过3D expert参数(L1 grouped_dual, expert按需读取);
                         False=移动3D expert到device(L2需要完整前向)
    """
    from .utils import get_decoder_layers
    layers = get_decoder_layers(model)
    layer_set = set(layer_indices)

    # 如果需要清空其他层，先把非当前shard的层移回CPU
    if clear_others:
        cpu_device = 'cpu'
        for i, layer in enumerate(layers):
            if i not in layer_set:
                # Skip layers on meta device — they have no data and cannot be moved
                try:
                    if next(layer.parameters(), None) is not None and next(layer.parameters()).is_meta:
                        continue
                except StopIteration:
                    continue
                layer.to(cpu_device)

    # 移动当前shard的层到目标设备
    for i, layer in enumerate(layers):
        if i in layer_set:
            # Skip 3D routed-expert params (L1 grouped_dual: expert按需读取, 3D expert params留在CPU)
            # 但不跳过非expert的3D参数 (如 Qwen3.6 linear_attn.conv1d.weight [out, 1, kernel])
            # L2需要完整前向, 3D expert params必须上device
            def _is_routed_expert_param(name: str, p: nn.Parameter) -> bool:
                # Packed experts are 3D; ModuleList experts use ``experts.N`` 2D weights.
                # Exclude shared experts and unrelated Conv1d parameters.
                parts = name.split('.')
                has_indexed_expert = any(
                    part == 'experts' and i + 1 < len(parts) and parts[i + 1].isdigit()
                    for i, part in enumerate(parts)
                )
                return (
                    'shared' not in name
                    and (
                        (p.dim() == 3 and 'experts' in name)
                        or has_indexed_expert
                    )
                )

            has_routed_expert = any(
                _is_routed_expert_param(name, p)
                for name, p in layer.named_parameters()
            )
            if has_routed_expert and skip_3d_experts:
                for name, param in layer.named_parameters():
                    if not _is_routed_expert_param(name, param):
                        param.data = param.to(device)
                for buf in layer.buffers():
                    buf.data = buf.to(device)
            else:
                layer.to(device)


def unload_layers_to_meta(
    model: nn.Module,
    layer_indices: List[int],
    *,
    cleanup: bool = True,
):
    """将指定 decoder layers 移回 meta device，释放 CPU + NPU 内存

    与 move_layers_to_device(layers, 'cpu') 不同，to_empty(device='meta') 会彻底释放
    权重占用的内存，而非仅移到 CPU 保留。这对 MoE 模型（如 GLM5.1，每层 256 experts）
    至关重要，否则已处理层权重堆积在 CPU 会导致 OOM。

    注意：只卸载 decoder layers，不影响 embed_tokens、rotary_emb、norm、lm_head。

    Args:
        model: 模型
        layer_indices: 要卸载的层索引列表
        cleanup: 是否立即执行 GC、设备缓存回收和 malloc_trim。连续卸载
            ref/quant 两侧时，第一侧可传 False，由最后一侧统一回收。
    """
    import gc
    import ctypes
    from .utils import get_decoder_layers
    layers = get_decoder_layers(model)
    for i in layer_indices:
        if i < len(layers):
            old_layer = layers[i]
            try:
                layers[i] = old_layer.to_empty(device='meta')
            except Exception:
                # fallback: 如果 to_empty 失败，逐个参数处理
                for p in old_layer.parameters():
                    p.data = torch.empty(0, dtype=p.dtype, device=p.device)
                layers[i] = old_layer.to_empty(device='meta')
            del old_layer
    if cleanup:
        gc.collect()
        if hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.empty_cache()
        # 尝试归还内存给OS
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception as e:
            logger.debug(f"malloc_trim failed: {e}")
