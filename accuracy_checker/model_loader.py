"""
量化模型加载器

借鉴 msmodelslim_inference_large.py 的方案:
  1. symlink trick: 让 transformers 能识别 msmodelslim 的权重文件
  2. AutoModelForCausalLM.from_pretrained() 加载模型结构 + INT8权重
  3. 逐层反量化: w_fp = (w_int8 - offset) * scale, 写回 module.weight

支持的量化类型:
"""

import os
import json
import gc
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from safetensors.torch import load_file
from safetensors import safe_open

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

from .utils import parse_base_name, normalize_quant_desc_keys
import logging

logger = logging.getLogger(__name__)

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
        except Exception as e:
            logger.debug(f"is_quantized_model config parse failed: {e}")
    return False


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


def dequantize_weight_dynamic(
    weight_data: Tensor,
    weight_scale: Tensor,
    weight_offset: Tensor = None,
    dtype: torch.dtype = torch.float16,
) -> Tensor:
    """
    W8A8_DYNAMIC / W8A8_MIX 反量化

    公式: w_fp = (w_data - offset) * scale
    """
    if weight_offset is None:
        weight_offset = torch.zeros_like(weight_scale)

    # 确保 shape 匹配: scale/offset 是 [out_features], 需要扩展到 [out_features, 1]
    if weight_scale.dim() == 1:
        weight_scale = weight_scale.unsqueeze(1)
        weight_offset = weight_offset.unsqueeze(1)

    w_fp = (weight_data.to(torch.float32) - weight_offset) * weight_scale
    return w_fp.to(dtype)


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
    # deq_scale 可能是 INT64 (NPU加速格式), 需要 reinterpret 为 FP32
    if deq_scale.dtype == torch.int64:
        deq_scale = deq_scale.view(torch.float64).float()

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

    w_fp = weight_data.to(torch.float32) * weight_scale
    return w_fp.to(dtype), had_input_scale


# ============================================================================
# 逐层反量化
# ============================================================================

def load_quant_weights(model_path: str) -> Dict[str, Tensor]:
    """加载所有量化权重 (从 safetensors)"""
    # 从 weight_index 找到所有分片文件
    index_path = os.path.join(model_path, "quant_model_weights.safetensors.index.json")
    if not os.path.exists(index_path):
        # 可能只有一个safetensors
        single = os.path.join(model_path, "quant_model_weights.safetensors")
        if os.path.exists(single):
            return load_file(single)
        return {}

    with open(index_path, 'r') as f:
        index_data = json.load(f)

    weight_map = index_data.get("weight_map", {})
    shard_files = set(weight_map.values())

    weights = {}
    for shard_file in shard_files:
        file_path = os.path.join(model_path, shard_file)
        if os.path.exists(file_path):
            file_weights = load_file(file_path)
            weights.update(file_weights)

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
        if ".layers." in k:
            parts = k.split(".layers.")
            if len(parts) > 1:
                try:
                    layer_num = int(parts[1].split(".")[0])
                    if layer_num in layer_set:
                        shard_files.add(v)
                        layer_to_shard[layer_num] = v
                except (ValueError, IndexError):
                    pass

        # 收集基础权重 (embed_tokens, norm, lm_head)
        # 这些通常在第一个 shard 文件中
        if any(x in k for x in ['embed_tokens', 'lm_head', '.norm.']):
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
        if ".layers." in k:
            parts = k.split(".layers.")
            if len(parts) > 1:
                try:
                    layer_num = int(parts[1].split(".")[0])
                    if layer_num in layer_set:
                        shard_files.add(v)
                        layer_to_shard[layer_num] = v
                except (ValueError, IndexError):
                    pass

        if any(x in k for x in ['embed_tokens', 'lm_head', '.norm.']):
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
    for index_name in ("model.safetensors.index.json",
                       "quant_model_weights.safetensors.index.json"):
        index_path = os.path.join(model_path, index_name)
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            return index_data.get("weight_map", {})
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

    def get_tensor(self, key: str):
        """读取单个 key 的 tensor，如果 key 不在 weight_map 中返回 None

        自动处理 ForConditionalGeneration 的前缀不匹配:
          weight_map key: "model.language_model.layers..."
          query key:      "model.model.language_model.layers..."
        尝试去掉一个 "model." 前缀作为 fallback
        """
        fname = self.weight_map.get(key)
        if fname is not None:
            sf = self._get_sf(fname)
            return sf.get_tensor(key)

        # Fallback: 去掉一层 "model." 前缀重试
        # "model.model.language_model.layers..." → "model.language_model.layers..."
        if key.startswith("model.model."):
            alt_key = key[len("model."):]
            fname = self.weight_map.get(alt_key)
            if fname is not None:
                sf = self._get_sf(fname)
                return sf.get_tensor(alt_key)

        return None

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
            if '.layers.' in key:
                _collect_layer_key(key, layer_indices, needed)
            elif include_non_layer and _is_non_layer_key(key):
                needed.add(key)
        return needed

    def close(self):
        """关闭所有 safetensors 文件句柄，释放 mmap 内存"""
        for sf in self._sf_cache.values():
            try:
                sf.close()
            except Exception as e:
                logger.warning(f"Failed to close safetensors file handle: {e}")
        self._sf_cache.clear()

    def __del__(self):
        self.close()


def _collect_layer_key(key: str, layer_indices: List[int], needed: set):
    """Parse a "model.layers.N.*" key and add it to `needed` if N is in layer_indices.

    Helper extracted from ShardWeightReader.get_keys_for_layers to reduce block depth.
    """
    parts = key.split('.layers.')
    if len(parts) <= 1:
        return
    try:
        layer_num = int(parts[1].split('.')[0])
    except (ValueError, IndexError):
        return
    if layer_num in layer_indices:
        needed.add(key)


def _is_non_layer_key(key: str) -> bool:
    """True if key refers to embed_tokens / lm_head / norm (non-layer weight)."""
    return any(x in key for x in ['embed_tokens', 'lm_head', '.norm.'])


def _decide_should_load(name: str, layer_idx: Optional[int], layer_set: set,
                        load_embed_only: bool, load_norm_head_only: bool,
                        verbose: bool, param=None) -> bool:
    """Decide whether a parameter should be loaded given the layer selection mode.

    Extracted from load_layer_weights_indexed to reduce cyclomatic complexity.
    When `param` is provided and mode=load_embed_only, emits the original DEBUG EMBED log.
    """
    if load_embed_only:
        if layer_idx is None and "norm" not in name and "lm_head" not in name:
            if verbose and 'embed' in name and param is not None:
                logger.info(f"  [DEBUG EMBED] will load: {name}, shape={param.shape}, device={param.device}")
            return True
        return False
    if load_norm_head_only:
        return layer_idx is None and ("norm" in name or "lm_head" in name)
    return layer_idx is not None and layer_idx in layer_set


def _load_ct_param(name: str, param, sf_reader: ShardWeightReader,
                   dtype: torch.dtype) -> bool:
    """Load one compressed-tensors parameter. Returns True if loaded."""
    tensor = sf_reader.get_tensor(name)
    scale_key = name.replace('.weight', '.weight_scale')
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
    if tensor.dtype == torch.int8 and scale_tensor is not None:
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

    if quant_type == "W8A8":
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

    if quant_type in ("W4A16", "W4A8_DYNAMIC", "W4A8", "W4A4_DYNAMIC", "W4A4_LAOS"):
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
    "W8A8", "W8A8_DYNAMIC", "W8A8_MIX", "W8A8_MXFP8",
    "W4A8_MXFP", "W4A4_MXFP4",
    "W4A16", "W4A8_DYNAMIC", "W4A8",
    "W4A4_DYNAMIC", "W4A4_LAOS",
)


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
    base_name = parse_base_name(weight_key)
    quant_type = quant_desc_str.get(weight_key, quant_desc_str.get(base_name, "FLOAT"))

    # FLOAT 或未知类型: 直接读 tensor 赋值 (与原 FLOAT/unknown 合并分支一致)
    if quant_type == "FLOAT" or quant_type not in _KNOWN_QUANT_TYPES:
        weight_data = sf_reader.get_tensor(name)
        if weight_data is None:
            weight_data = sf_reader.get_tensor(weight_key)
        if weight_data is not None:
            param.data = weight_data.to(dtype)
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
    param.data = w_fp
    return True


def _load_named_buffers(model: nn.Module, sf_reader: ShardWeightReader,
                        layer_set: set, load_embed_only: bool,
                        load_norm_head_only: bool) -> int:
    """Load register_buffer tensors (e.g. e_score_correction_bias). Returns loaded count.

    Extracted from load_layer_weights_indexed.
    """
    loaded_count = 0
    if not hasattr(model, 'named_buffers'):
        return loaded_count
    for bname, buf in model.named_buffers():
        if buf is None:
            continue
        b_layer_idx = _extract_layer_idx(bname)
        if not _decide_should_load(bname, b_layer_idx, layer_set,
                                   load_embed_only, load_norm_head_only, False):
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

    # 修正 quant_desc key 与 named_modules() 的前缀不匹配
    if is_quant and quant_desc is not None:
        quant_desc = normalize_quant_desc_keys(quant_desc, model)

    # msmodelslim: 过滤 quant_desc 中非 string 的 metadata key
    if is_quant and quant_desc is not None:
        quant_desc_str = {k: v for k, v in quant_desc.items() if isinstance(v, str)}
    else:
        quant_desc_str = None

    for name, param in model.named_parameters():
        layer_idx = _extract_layer_idx(name)
        should_load = _decide_should_load(
            name, layer_idx, layer_set, load_embed_only, load_norm_head_only,
            verbose, param,
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
        model, sf_reader, layer_set, load_embed_only, load_norm_head_only
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
        if test_key in sf_reader.weight_map:
            num_experts = i + 1
        elif num_experts > 0:
            break
    return num_experts


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
    g_type = quant_desc_str.get(g_key, quant_desc_str.get(g_base, "FLOAT"))
    g_data = _dequant_single_expert_indexed(g_key, g_type, sf_reader, dtype, use_fake_quant)
    if g_data is None:
        return False
    packed[i, :intermediate_dim, :] = g_data

    u_key = f"{expert_prefix}.{i}.up_proj.weight"
    u_type = quant_desc_str.get(u_key, quant_desc_str.get(g_base, "FLOAT"))
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
        d_base = parse_base_name(d_key)
        d_type = quant_desc_str.get(d_key, quant_desc_str.get(d_base, "FLOAT"))
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
    g0_type = quant_desc_str.get(g0_key, quant_desc_str.get(parse_base_name(g0_key), "FLOAT"))
    is_w4_packed = g0_type in _MXFP4_QUANT_TYPES
    is_w4_unpack = g0_type in ("W4A8_DYNAMIC", "W4A16", "W4A8", "W4A4_DYNAMIC", "W4A4_LAOS")

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
    if quant_type == "FLOAT":
        weight_data = sf_reader.get_tensor(weight_key)
        if weight_data is not None:
            return weight_data.to(dtype)
        return None

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

    if quant_type in ("W4A8_DYNAMIC", "W4A16", "W4A8", "W4A4_DYNAMIC", "W4A4_LAOS"):
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
            weight_data, deq_scale, input_scale, dtype
        )
        return w_fp

    return None


def create_model_skeleton(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = True,
) -> nn.Module:
    """
    只创建模型骨架（meta device），不加载任何权重。

    用于分片加载时复用模型结构，避免每次重建模型导致的不稳定。

    Args:
        model_path: 模型路径
        dtype: 数据类型
        verbose: 是否打印进度

    Returns:
        模型骨架（meta device）
    """
    # 加载config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # 多模态 ForConditionalGeneration 模型 (如 Qwen3.6): num_hidden_layers 在 text_config 里
    if not hasattr(config, 'num_hidden_layers'):
        if hasattr(config, 'text_config') and hasattr(config.text_config, 'num_hidden_layers'):
            config.num_hidden_layers = config.text_config.num_hidden_layers
        elif hasattr(config, 'num_layer'):
            config.num_hidden_layers = config.num_layer
        else:
            raise ValueError("config 中缺少 num_hidden_layers 和 num_layer，无法确定模型层数")

    if verbose:
        logger.info(f"  创建模型骨架 (meta device): {config.num_hidden_layers} 层")

    # 创建模型结构 (meta device)
    # 多模态 ForConditionalGeneration 模型不能用 AutoModelForCausalLM，需用 architectures 指定的类
    with torch.device('meta'):
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
            import importlib
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
    model = model.to_empty(device='cpu')

    return model


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
    is_quant = is_quantized_model(model_path)

    # Sharded mode: return skeleton only
    if layers is not None:
        model = create_model_skeleton(model_path, dtype, verbose=verbose)
        return model, is_quant

    # Full-load mode (single device)
    single_device = device.split(',')[0] if isinstance(device, str) and ',' in device else device

    if is_quant and use_fake_quant:
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
            logger.info(f"  量化模型, 反量化模式加载")
        # Load with dequantize: skeleton + load all weights + dequant
        model = create_model_skeleton(model_path, dtype, verbose=verbose)
        weight_map = build_weight_index(model_path)
        reader = ShardWeightReader(model_path, weight_map)
        num_layers = model.config.num_hidden_layers
        all_layers = list(range(num_layers))
        load_layer_weights_indexed(model, model_path, all_layers, single_device, dtype,
                                   weight_map, reader, is_quant=is_quant, verbose=verbose)
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


def unload_layers_to_meta(model: nn.Module, layer_indices: List[int]):
    """将指定 decoder layers 移回 meta device，释放 CPU + NPU 内存

    与 move_layers_to_device(layers, 'cpu') 不同，to_empty(device='meta') 会彻底释放
    权重占用的内存，而非仅移到 CPU 保留。这对 MoE 模型（如 GLM5.1，每层 256 experts）
    至关重要，否则已处理层权重堆积在 CPU 会导致 OOM。

    注意：只卸载 decoder layers，不影响 embed_tokens、rotary_emb、norm、lm_head。

    Args:
        model: 模型
        layer_indices: 要卸载的层索引列表
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
    gc.collect()
    if hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.empty_cache()
    # 尝试归还内存给OS
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception as e:
        logger.debug(f"malloc_trim failed: {e}")
