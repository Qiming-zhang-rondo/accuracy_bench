"""
HF 推理检查工具 — 排除框架影响

加载 HF 量化模型 → 反量化 → 多卡分布 → generate → 输出结果。
区分精度问题来自量化本身还是推理框架:
  - 通顺 → 量化没问题, 精度问题在推理框架
  - 乱码/重复 → 量化有问题, 需 V2 逐层定位

v2 加载流程 (NPU 加速反量化):
  1. create_model_skeleton (meta → CPU)
  2. distribute_model (skeleton → 各 NPU)
  3. 逐层加载 INT8 → .to(npu) → NPU 反量化 → 写回 module
  4. 释放 CPU INT8 权重

用法:
    python3 -m accuracy_checker.inference_check \\
        --model_path /path/to/model --devices npu:0,1,2,3 \\
        --prompt_file prompts.json --max_new_tokens 4096
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .model_loader import (
    create_model_skeleton,
    dequantize_weight_static,
    dequantize_weight_dynamic,
    dequantize_weight_mxfp8,
    dequantize_weight_mx,
    unpack_int4_to_int8,
    _load_float_weights,
    load_quant_weights,
    is_quantized_model,
    _MX_QUANT_TYPES,
)
from .utils import (
    parse_base_name,
    normalize_quant_desc_keys,
    normalize_quant_desc_values,
    load_rotation_matrix,
)
from .model_structure import get_model_components


# ============================================================================
# 工具函数
# ============================================================================

def parse_devices(device_str: str) -> List[str]:
    """解析设备字符串，自动补 npu: 前缀"""
    devices = []
    for d in device_str.split(","):
        d = d.strip()
        if not d:
            continue
        if d.startswith("npu:") or d.startswith("cuda:"):
            devices.append(d)
        else:
            devices.append(f"npu:{d}")
    return devices


def preprocess_messages(messages: List[Dict]) -> List[Dict]:
    """预处理 messages: tool_calls.arguments 从 JSON string 转 dict"""
    processed = []
    for msg in messages:
        m = dict(msg)
        if "tool_calls" in m and m["tool_calls"]:
            new_tc = []
            for tc in m["tool_calls"]:
                tc2 = dict(tc)
                if "function" in tc2:
                    func = dict(tc2["function"])
                    if isinstance(func.get("arguments"), str):
                        func["arguments"] = json.loads(func["arguments"])
                    tc2["function"] = func
                new_tc.append(tc2)
            m["tool_calls"] = new_tc
        processed.append(m)
    return processed


def load_prompt_file(path: str) -> Tuple[List[List[Dict]], Optional[List]]:
    """加载 JSON 请求；.txt/.prompt 文件按已渲染的原始 prompt 原样读取。"""
    if path.lower().endswith((".txt", ".text", ".prompt")):
        with open(path, encoding="utf-8") as f:
            raw_prompt = f.read()
        if not raw_prompt.strip():
            raise ValueError(f"原始 prompt 文件为空: {path}")
        logger.info(f"  从 {path} 加载了 1 个原始 prompt (不再套 chat template)")
        return [[{"role": "user", "content": raw_prompt, "_raw_prompt": True}]], None

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        conversations = [data.get("messages", [])]
        tools = data.get("tools", None)
        logger.info(f"  从 {path} 加载了 1 个请求 (tools={'有' if tools else '无'})")
        return conversations, tools

    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], list):
            return data, None
        return [data], None

    raise ValueError(f"不支持的 prompt 文件格式: {path}")


def decode_output(tokenizer, new_tokens) -> Dict[str, Any]:
    """解码生成 token，分离思维链和正文"""
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    think_end = raw_text.find("</think>")
    if think_end != -1:
        thinking = raw_text[:think_end].replace("<think>", "").strip()
        answer = raw_text[think_end + len("</think>"):].strip()
        return {
            "thinking": thinking,
            "answer": answer,
            "raw_text": raw_text,
            "thinking_truncated": False,
        }
    # 没有 </think> — 可能在思维链中被截断
    clean = raw_text.replace("<think>", "").strip()
    is_truncated = "<think>" in raw_text and "</think>" not in raw_text
    return {
        "thinking": clean if is_truncated else "",
        "answer": clean if not is_truncated else "",
        "raw_text": raw_text,
        "thinking_truncated": is_truncated,
    }

def _detect_model_layers_and_modules(model):
    """Detect model layers, embed, final_norm and lm_head structurally.

    Returns:
        (layers, embed, final_norm, lm_head)
    """
    components = get_model_components(model)
    return (components.layers, components.embed,
            components.final_norm, components.lm_head)


def _get_inner_model_for_rotary(model):
    """Get the inner model containing rotary_emb."""
    return get_model_components(model).text_model


def distribute_model(model, device_list: List[str]) -> List[nn.Module]:
    """
    将模型分布到多卡，注册跨卡 hidden_states 迁移 hook。

    使用统一结构探测获取文本层和特殊模块路径。

    Returns:
        layers 列表
    """
    components = get_model_components(model)
    layers, embed, final_norm, lm_head = (
        components.layers, components.embed,
        components.final_norm, components.lm_head,
    )

    n_layers = len(layers)
    n_devices = len(device_list)
    layers_per_device = (n_layers + n_devices - 1) // n_devices

    # embed + rotary_emb → 第一张卡
    if embed is not None:
        embed.to(device_list[0])

    if components.rotary_emb is not None:
        components.rotary_emb.to(device_list[0])

    # layers → 均分
    for i, layer in enumerate(layers):
        target = device_list[min(i // layers_per_device, n_devices - 1)]
        layer.to(target)

    # DeepSeek-V4 ModelScope checkpoints append three optional MTP/DSpark
    # blocks.  Newer vendor runtimes expose them as a ModuleList; official HF
    # V4 currently ignores the extra tensors.  If present, shard them as a
    # small auxiliary layer stack instead of leaving them on CPU or silently
    # pinning all three to one device.
    auxiliary = None
    auxiliary_name = None
    text_model = components.text_model
    for candidate in ("mtp", "nextn_predict_layers", "dspark"):
        value = getattr(text_model, candidate, None)
        if isinstance(value, (nn.ModuleList, list, tuple)) and value:
            auxiliary = value
            auxiliary_name = candidate
            break
    if auxiliary is not None:
        aux_per_device = (len(auxiliary) + n_devices - 1) // n_devices
        for i, block in enumerate(auxiliary):
            block.to(device_list[min(i // aux_per_device, n_devices - 1)])
        _register_layer_input_hooks(
            auxiliary, device_list, aux_per_device, n_devices
        )
        logger.info(
            "  DeepSeek-V4 %s: %d 个 MTP/DSpark block 已分布到 %d 张卡",
            auxiliary_name, len(auxiliary), min(len(auxiliary), n_devices),
        )

    # norm + lm_head → 最后一张卡
    if final_norm is not None:
        final_norm.to(device_list[-1])
        if n_devices > 1:
            # The HF model applies the final norm inside ``model.forward``;
            # unlike decoder layers it is not reached by the layer hooks
            # below.  Move the last hidden state explicitly before the norm,
            # otherwise a PP-sharded V4 run ends with npu[N-2] vs npu[N-1]
            # matmul errors.
            _register_module_input_hook(final_norm, device_list[-1])
    # DeepSeek-V4 collapses its hc_mult residual streams before final norm.
    hc_head = getattr(components.text_model, "hc_head", None)
    if hc_head is not None:
        hc_head.to(device_list[-1])
        if n_devices > 1:
            _register_module_input_hook(hc_head, device_list[-1])
    if lm_head is not None:
        lm_head.to(device_list[-1])
        if n_devices > 1:
            _register_lm_head_device_hook(lm_head, device_list[0])

    # visual → 第一张卡
    if components.visual is not None:
        components.visual.to(device_list[0])

    logger.info(f"  分布完成: {n_layers} 层, 每设备约 {layers_per_device} 层")

    # Register pre-forward hooks for cross-device hidden_states migration
    _register_layer_input_hooks(layers, device_list, layers_per_device, n_devices)

    return layers


def _register_lm_head_device_hook(lm_head, first_device: str):
    """Register hook to move logits to first device (multi-card only)."""
    def _logits_to_first_device(module, args, output):
        if isinstance(output, torch.Tensor) and str(output.device) != first_device:
            return output.to(first_device)
    lm_head.register_forward_hook(_logits_to_first_device)


def _register_module_input_hook(module: nn.Module, target_device: str):
    """Move a terminal module's tensor inputs to its owning PP device."""
    def _hook(_module, args, kwargs):
        new_args = tuple(_move_to_device(a, target_device) for a in args)
        new_kwargs = {k: _move_to_device(v, target_device)
                      for k, v in kwargs.items()}
        return new_args, new_kwargs

    module.register_forward_pre_hook(_hook, with_kwargs=True)


def _move_to_device(obj, dev):
    """Recursively move tensors in nested structures to device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(dev)
    if isinstance(obj, (tuple, list)):
        moved = [_move_to_device(o, dev) for o in obj]
        return type(obj)(moved)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, dev) for k, v in obj.items()}
    return obj


def _register_layer_input_hooks(layers, device_list, layers_per_device, n_devices):
    """Register pre-forward hooks on each layer for cross-device tensor migration."""
    def _make_input_hook(target_dev):
        def _hook(module, args, kwargs):
            new_args = tuple(_move_to_device(a, target_dev) for a in args)
            new_kwargs = {k: _move_to_device(v, target_dev) for k, v in kwargs.items()}
            return new_args, new_kwargs
        return _hook

    for i, layer in enumerate(layers):
        target = device_list[min(i // layers_per_device, n_devices - 1)]
        layer.register_forward_pre_hook(_make_input_hook(target), with_kwargs=True)


def _get_module_device(module: nn.Module) -> str:
    """获取 module 所在设备 (从第一个参数推断)"""
    for p in module.parameters():
        return str(p.device)
    for b in module.buffers():
        return str(b.device)
    return 'cpu'


# ============================================================================
# NPU 加速反量化: 先 distribute skeleton, 再在 NPU 上逐层反量化
# ============================================================================

def _dequant_mx_single(weight_data, weight_scale, quant_type, dtype, target_device):
    """MX quantization dequant helper."""
    if weight_scale is None:
        return None
    weight_data_dev = weight_data.to(target_device)
    weight_scale_dev = weight_scale.to(target_device)
    return dequantize_weight_mx(weight_data_dev, weight_scale_dev, quant_type, dtype=dtype)


def _dequant_w8a8_dynamic(weight_data, weight_scale, weight_offset, dtype, target_device):
    """W8A8_DYNAMIC dequant helper."""
    if weight_scale is None:
        return None
    weight_data_dev = weight_data.to(target_device)
    weight_scale_dev = weight_scale.to(target_device)
    weight_offset_dev = weight_offset.to(target_device) if weight_offset is not None else None
    return dequantize_weight_dynamic(weight_data_dev, weight_scale_dev, weight_offset_dev, dtype)


def _dequant_w8a8_static(weight_data, deq_scale, input_scale, dtype, target_device):
    """W8A8 static dequant helper."""
    if deq_scale is None:
        return None
    weight_data_dev = weight_data.to(target_device)
    deq_scale_dev = deq_scale.to(target_device)
    input_scale_dev = input_scale.to(target_device) if input_scale is not None else None
    w_fp, _ = dequantize_weight_static(weight_data_dev, deq_scale_dev, input_scale_dev, dtype=dtype)
    return w_fp


def _dequant_w4_dynamic(weight_data, weight_scale, weight_offset, dtype, target_device):
    """W4A8/W4A16 dequant helper with int8 unpack."""
    if weight_data.dtype != torch.int8 or weight_scale is None:
        return None
    weight_data_dev = weight_data.to(target_device)
    unpacked = unpack_int4_to_int8(weight_data_dev)
    weight_scale_dev = weight_scale.to(target_device)
    weight_offset_dev = weight_offset.to(target_device) if weight_offset is not None else None
    return dequantize_weight_dynamic(unpacked, weight_scale_dev, weight_offset_dev, dtype)


def _dequant_single_weight_on_device(
    name: str,
    module: nn.Module,
    quant_weights: Dict[str, torch.Tensor],
    quant_type: str,
    dtype: torch.dtype,
) -> bool:
    """
    在 module 所在设备上反量化单个 Linear 的权重。

    Returns True if dequantized successfully.
    """
    weight_key = f"{name}.weight"
    weight_data = quant_weights.get(weight_key)
    if weight_data is None:
        return False

    target_device = _get_module_device(module)
    quant_name = name

    if quant_type in ("FLOAT", "C8"):
        return False

    # MX quantization (MXFP8, MXFP4, etc.)
    if quant_type in _MX_QUANT_TYPES:
        weight_scale = quant_weights.get(f"{quant_name}.weight_scale")
        w_fp = _dequant_mx_single(weight_data, weight_scale, quant_type, dtype, target_device)
        if w_fp is not None:
            module.weight.data = w_fp
            return True
        return False

    # W8A8_DYNAMIC / W8A8_MIX
    if quant_type in ("W8A8_DYNAMIC", "W8A8_MIX"):
        weight_scale = quant_weights.get(f"{quant_name}.weight_scale")
        weight_offset = quant_weights.get(f"{quant_name}.weight_offset")
        w_fp = _dequant_w8a8_dynamic(weight_data, weight_scale, weight_offset, dtype, target_device)
        if w_fp is not None:
            module.weight.data = w_fp
            return True
        # fallback: static dequant
        deq_scale = quant_weights.get(f"{quant_name}.deq_scale")
        input_scale = quant_weights.get(f"{quant_name}.input_scale")
        w_fp = _dequant_w8a8_static(weight_data, deq_scale, input_scale, dtype, target_device)
        if w_fp is not None:
            module.weight.data = w_fp
            return True
        return False

    # W8A8 static
    if quant_type == "W8A8":
        deq_scale = quant_weights.get(f"{quant_name}.deq_scale")
        input_scale = quant_weights.get(f"{quant_name}.input_scale")
        w_fp = _dequant_w8a8_static(weight_data, deq_scale, input_scale, dtype, target_device)
        if w_fp is not None:
            module.weight.data = w_fp
            return True
        # fallback: dynamic dequant
        weight_scale = quant_weights.get(f"{quant_name}.weight_scale")
        weight_offset = quant_weights.get(f"{quant_name}.weight_offset")
        w_fp = _dequant_w8a8_dynamic(weight_data, weight_scale, weight_offset, dtype, target_device)
        if w_fp is not None:
            module.weight.data = w_fp
            return True
        return False

    # W4A16 / W4A8 / W4A8_DYNAMIC
    if quant_type in ("W4A16", "W4A8_DYNAMIC", "W4A8"):
        weight_scale = quant_weights.get(f"{quant_name}.weight_scale")
        weight_offset = quant_weights.get(f"{quant_name}.weight_offset")
        w_fp = _dequant_w4_dynamic(weight_data, weight_scale, weight_offset, dtype, target_device)
        if w_fp is not None:
            module.weight.data = w_fp
            return True
        return False

    return False


def _dequant_3d_expert_on_device(
    name: str,
    param: nn.Parameter,
    quant_weights: Dict[str, torch.Tensor],
    quant_desc: Dict[str, str],
    dtype: torch.dtype,
) -> bool:
    """
    在参数所在设备上反量化 3D packed expert。

    逐 expert 反量化，但每次搬 INT8 到 NPU → NPU 反量化 → 拼接到 NPU packed tensor。
    """
    if 'experts.gate_up_proj' in name:
        expert_prefix = name.rsplit('.gate_up_proj', 1)[0]
        is_gate_up = True
    elif 'experts.down_proj' in name:
        expert_prefix = name.rsplit('.down_proj', 1)[0]
        is_gate_up = False
    else:
        return False

    # 统计 expert 数量 (从 quant_weights 的 key 动态提取)
    prefix_pattern = f"{expert_prefix}."
    expert_indices = set()
    for key in quant_weights.keys():
        if key.startswith(prefix_pattern):
            # 提取 expert index: "{expert_prefix}.{idx}.gate_proj.weight" 或 "{expert_prefix}.{idx}.down_proj.weight"
            suffix = key[len(prefix_pattern):]
            parts = suffix.split('.', 1)
            if parts[0].isdigit():
                expert_indices.add(int(parts[0]))

    if not expert_indices:
        return False

    num_experts = max(expert_indices) + 1

    if num_experts == 0:
        return False

    target_device = str(param.device)

    if is_gate_up:
        gate_int8 = quant_weights[f"{expert_prefix}.0.gate_proj.weight"]
        intermediate_dim = gate_int8.shape[0]
        hidden_dim = gate_int8.shape[1]

        packed = torch.zeros(num_experts, 2 * intermediate_dim, hidden_dim,
                             dtype=dtype, device=target_device)

        for i in range(num_experts):
            # gate_proj
            g_key = f"{expert_prefix}.{i}.gate_proj.weight"
            g_base = parse_base_name(g_key)
            g_type = quant_desc.get(g_key, quant_desc.get(g_base, "FLOAT"))
            g_data = _dequant_single_expert_on_device(
                g_key, g_base, g_type, quant_weights, dtype, target_device
            )
            if g_data is None:
                return False
            packed[i, :intermediate_dim, :] = g_data

            # up_proj
            u_key = f"{expert_prefix}.{i}.up_proj.weight"
            u_base = parse_base_name(u_key)
            u_type = quant_desc.get(u_key, quant_desc.get(u_base, "FLOAT"))
            u_data = _dequant_single_expert_on_device(
                u_key, u_base, u_type, quant_weights, dtype, target_device
            )
            if u_data is None:
                return False
            packed[i, intermediate_dim:, :] = u_data

        param.data = packed
        return True

    else:  # down_proj
        down_int8 = quant_weights[f"{expert_prefix}.0.down_proj.weight"]
        hidden_dim = down_int8.shape[0]
        intermediate_dim = down_int8.shape[1]

        packed = torch.zeros(num_experts, hidden_dim, intermediate_dim,
                             dtype=dtype, device=target_device)

        for i in range(num_experts):
            d_key = f"{expert_prefix}.{i}.down_proj.weight"
            d_base = parse_base_name(d_key)
            d_type = quant_desc.get(d_key, quant_desc.get(d_base, "FLOAT"))
            d_data = _dequant_single_expert_on_device(
                d_key, d_base, d_type, quant_weights, dtype, target_device
            )
            if d_data is None:
                return False
            packed[i] = d_data

        param.data = packed
        return True


def _dequant_single_expert_on_device(
    weight_key: str,
    base_name: str,
    quant_type: str,
    quant_weights: Dict[str, torch.Tensor],
    dtype: torch.dtype,
    target_device: str,
) -> Optional[torch.Tensor]:
    """在目标设备上反量化单个 expert 权重"""
    if quant_type == "FLOAT":
        weight_data = quant_weights.get(weight_key)
        if weight_data is not None:
            return weight_data.to(dtype).to(target_device)
        return None

    weight_data = quant_weights.get(weight_key)
    if weight_data is None:
        return None

    quant_name = weight_key.rsplit('.', 1)[0]

    if quant_type in ("W8A8_DYNAMIC", "W8A8_MIX"):
        weight_scale = quant_weights.get(f"{quant_name}.weight_scale")
        weight_offset = quant_weights.get(f"{quant_name}.weight_offset")
        if weight_scale is not None:
            return dequantize_weight_dynamic(
                weight_data.to(target_device),
                weight_scale.to(target_device),
                weight_offset.to(target_device) if weight_offset is not None else None,
                dtype,
            )
        return None

    if quant_type in ("W4A8_DYNAMIC", "W4A8", "W4A16"):
        weight_scale = quant_weights.get(f"{quant_name}.weight_scale")
        weight_offset = quant_weights.get(f"{quant_name}.weight_offset")
        if weight_data.dtype == torch.int8 and weight_scale is not None:
            unpacked = unpack_int4_to_int8(weight_data.to(target_device))
            return dequantize_weight_dynamic(
                unpacked,
                weight_scale.to(target_device),
                weight_offset.to(target_device) if weight_offset is not None else None,
                dtype,
            )
        return None

    if quant_type in _MX_QUANT_TYPES:
        weight_scale_u8 = quant_weights.get(f"{quant_name}.weight_scale")
        if weight_scale_u8 is not None:
            return dequantize_weight_mx(
                weight_data.to(target_device),
                weight_scale_u8.to(target_device),
                quant_type,
                dtype=dtype,
            )
        return None

    deq_scale = quant_weights.get(f"{quant_name}.deq_scale")
    input_scale = quant_weights.get(f"{quant_name}.input_scale")
    if deq_scale is not None:
        w_fp, _ = dequantize_weight_static(
            weight_data.to(target_device),
            deq_scale.to(target_device),
            input_scale.to(target_device) if input_scale is not None else None,
            dtype=dtype,
        )
        return w_fp

    return None


def _dequant_pass1_linears(model: nn.Module, quant_weights: Dict[str, torch.Tensor],
                           quant_desc: Dict[str, str], dtype: torch.dtype,
                           verbose: bool) -> Tuple[Dict[str, str], int, int, int]:
    """Pass 1: 反量化所有 nn.Linear 权重"""
    dequant_log: Dict[str, str] = {}
    skipped = 0
    failed = 0

    named_modules_list = list(model.named_modules())
    total_linears = sum(1 for _, m in named_modules_list if isinstance(m, nn.Linear))

    if verbose:
        logger.info(f"  NPU 反量化: {total_linears} 个 Linear 层...")

    done = 0
    for name, module in named_modules_list:
        if not isinstance(module, nn.Linear):
            continue
        done += 1

        weight_key = f"{name}.weight"
        base_name = parse_base_name(weight_key)
        quant_type = quant_desc.get(weight_key, quant_desc.get(base_name, "FLOAT"))

        if quant_type in ("FLOAT", "C8"):
            skipped = _load_float_linear(name, module, quant_weights,
                                         weight_key, dtype, skipped)
        else:
            ok = _dequant_single_weight_on_device(
                name, module, quant_weights, quant_type, dtype)
            if ok:
                dequant_log[name] = quant_type
            else:
                failed += 1
                if verbose:
                    logger.warning(f"{name} ({quant_type}) NPU 反量化失败")

        # 进度打印: 每 10 层或每个 completed 百分比
        if verbose and (done % 10 == 0 or done == total_linears):
            logger.info(f"    Linear 进度: {done}/{total_linears} "
                  f"(反量化 {len(dequant_log)}, FLOAT {skipped}, 失败 {failed})")

    return dequant_log, total_linears, skipped, failed


def _load_float_linear(name: str, module: nn.Module,
                       quant_weights: Dict[str, torch.Tensor],
                       weight_key: str, dtype: torch.dtype,
                       skipped: int) -> int:
    """加载 FLOAT 类型 Linear 的 weight 和 bias 到设备"""
    skipped += 1
    # FLOAT 权重也需要加载到设备
    if weight_key in quant_weights:
        target_device = _get_module_device(module)
        module.weight.data = quant_weights[weight_key].to(dtype).to(target_device)
    # 加载 bias
    bias_key = f"{name}.bias"
    if bias_key in quant_weights and module.bias is not None:
        target_device = _get_module_device(module)
        module.bias.data = quant_weights[bias_key].to(dtype).to(target_device)
    return skipped


def _is_3d_expert_param(name: str, param: nn.Parameter) -> bool:
    """判断参数是否是 3D packed expert (gate_up_proj 或非 shared down_proj)"""
    return (param.dim() == 3 and (
        'experts.gate_up_proj' in name or
        ('experts.down_proj' in name and 'shared_expert' not in name)
    ))


def _dequant_pass2_3d_experts(model: nn.Module, quant_weights: Dict[str, torch.Tensor],
                              quant_desc: Dict[str, str], dtype: torch.dtype,
                              verbose: bool, skip_3d_experts: bool,
                              dequant_log: Dict[str, str]) -> int:
    """Pass 2: 反量化 3D packed expert 参数"""
    if skip_3d_experts and verbose:
        # 仅在 verbose 时打印跳过信息
        named_params_list = list(model.named_parameters())
        total_3d = sum(1 for n, p in named_params_list if _is_3d_expert_param(n, p))
        if total_3d > 0:
            logger.info(f"  跳过 {total_3d} 个 3D packed expert (流式模式)")
        return 0

    expert_3d_count = 0
    named_params_list = list(model.named_parameters())
    total_3d = sum(1 for n, p in named_params_list if _is_3d_expert_param(n, p))

    if total_3d > 0 and verbose and not skip_3d_experts:
        logger.info(f"  NPU 反量化: {total_3d} 个 3D packed expert 参数...")

    if skip_3d_experts:
        return 0

    for name, param in named_params_list:
        if not _is_3d_expert_param(name, param):
            continue

        ok = _dequant_3d_expert_on_device(name, param, quant_weights, quant_desc, dtype)
        if ok:
            dequant_log[name] = "3D_PACKED_EXPERT"
            expert_3d_count += 1
        elif verbose:
            logger.warning(f"3D expert {name} NPU 反量化失败")

        if verbose:
            logger.info(f"    3D expert 进度: {expert_3d_count}/{total_3d} "
                  f"(latest: {name.split('.')[-2] if '.' in name else name})")

    return expert_3d_count


def _dequant_pass3_float_params(model: nn.Module, quant_weights: Dict[str, torch.Tensor],
                                quant_desc: Dict[str, str], dtype: torch.dtype
                                ) -> Tuple[int, int]:
    """Pass 3: 加载剩余 FLOAT 参数和 buffer (RMSNorm, Embedding, A_log 等)"""
    # Pass 1 只加载了 nn.Linear 的 FLOAT 权重, 这里加载其余 FLOAT 参数
    # (RMSNorm weight, embed_tokens weight, e_score_correction_bias 等)
    loaded_module_names = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue  # 只跳过 nn.Linear (Pass 1 已加载)
        weight_key = f"{name}.weight"
        if hasattr(module, 'weight') and weight_key in quant_weights:
            loaded_module_names.add(weight_key)
        bias_key = f"{name}.bias"
        if hasattr(module, 'bias') and bias_key in quant_weights and module.bias is not None:
            loaded_module_names.add(bias_key)

    float_param_count = 0
    for name, param in model.named_parameters():
        if name in loaded_module_names:
            continue
        if name in quant_weights:
            weight_key = name
            base_name = parse_base_name(weight_key)
            quant_type = quant_desc.get(weight_key, quant_desc.get(base_name, None))
            if quant_type == "FLOAT" or quant_type is None:
                param.data = quant_weights[name].to(dtype).to(str(param.device))
                float_param_count += 1

    # ---- Pass 3b: FLOAT buffers (e_score_correction_bias 等) ----
    float_buffer_count = 0
    for name, buf in model.named_buffers():
        if name in quant_weights:
            weight_key = name
            base_name = parse_base_name(weight_key)
            quant_type = quant_desc.get(weight_key, quant_desc.get(base_name, None))
            if quant_type == "FLOAT" or quant_type is None:
                # e_score_correction_bias 需保持 fp32
                target_dtype = torch.float32 if "e_score_correction_bias" in name else dtype
                buf.data = quant_weights[name].to(target_dtype).to(str(buf.device))
                float_buffer_count += 1

    return float_param_count, float_buffer_count


def dequantize_model_on_devices(
    model: nn.Module,
    quant_weights: Dict[str, torch.Tensor],
    quant_desc: Dict[str, str],
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = True,
    skip_3d_experts: bool = False,
) -> Dict[str, str]:
    """
    NPU 加速反量化: 模型已 distribute 到各 NPU, 逐模块在目标设备上反量化。

    对比 dequantize_model(): 后者全量 CPU 反量化, 本函数直接在 NPU 上做。
    INT8 权重临时搬到 NPU → 反量化 → 写回 module.weight (已在 NPU)。

    Args:
        model: 已 distribute 到 NPU 的模型
        quant_weights: 量化权重字典 (在 CPU)
        quant_desc: quant_model_description.json 内容
        dtype: 反量化后的数据类型
        verbose: 是否打印进度
        skip_3d_experts: 跳过 3D packed expert 反量化 (用于流式 expert 模式)

    Returns:
        反量化记录 {module_name: quant_type}
    """
    quant_desc = normalize_quant_desc_keys(quant_desc, model)

    # ---- Pass 1: nn.Linear 权重 ----
    dequant_log, total_linears, skipped, failed = _dequant_pass1_linears(
        model, quant_weights, quant_desc, dtype, verbose)

    # ---- Pass 2: 3D packed expert (nn.Parameter, not nn.Linear) ----
    expert_3d_count = _dequant_pass2_3d_experts(
        model, quant_weights, quant_desc, dtype, verbose,
        skip_3d_experts, dequant_log)

    # ---- Pass 3: FLOAT 参数 (RMSNorm, Embedding, A_log 等) ----
    float_param_count, float_buffer_count = _dequant_pass3_float_params(
        model, quant_weights, quant_desc, dtype)

    if verbose:
        expert_msg = f", {expert_3d_count} 3D packed expert" if expert_3d_count > 0 else ""
        logger.info(f"  NPU 反量化完成: {len(dequant_log)}/{total_linears} 层反量化, "
              f"{skipped} FLOAT Linear, {float_param_count} FLOAT param, "
              f"{float_buffer_count} FLOAT buffer, "
              f"{failed} 失败{expert_msg}")

    return dequant_log


# ============================================================================
# 流式 Expert 反量化: 743B 等超大 MoE 模型, expert 权重不全量加载到 NPU
# ============================================================================

def _set_param_by_name(model: nn.Module, name: str, new_param: nn.Parameter):
    """通过 dotted name 导航到 parent module 并替换 parameter"""
    parts = name.split('.')
    obj = model
    for p in parts[:-1]:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    setattr(obj, parts[-1], new_param)


def _prepare_expert_placeholders(model: nn.Module) -> Dict[str, tuple]:
    """
    将 3D expert 参数 (gate_up_proj, down_proj) 替换为 [1,1,1] placeholder。

    这样 distribute_model 的 layer.to(target) 只实体化非 expert 权重 (~10GB),
    避免把 753B expert 全量转 BF16 导致 OOM。

    Returns:
        {param_name: original_shape} — 用于后续流式 forward 的信息
    """
    original_shapes = {}
    for name, param in list(model.named_parameters()):
        is_3d_expert = param.dim() == 3 and (
            'experts.gate_up_proj' in name or
            ('experts.down_proj' in name and 'shared_expert' not in name)
        )
        if not is_3d_expert:
            continue
        original_shapes[name] = tuple(param.shape)
        placeholder = nn.Parameter(torch.empty(1, 1, 1, dtype=param.dtype))
        _set_param_by_name(model, name, placeholder)

    return original_shapes


# ============================================================================
# QuaRot 运行时旋转
# ============================================================================

def _get_quarot_side(name: str) -> Optional[str]:
    """判断模块的 QuaRot 旋转方向 (基于 r1r2_get_rotate_map 的 rot 组).

    Pre-run 只右旋了 embed_tokens (已 baked 到权重).
    Runtime 需要对以下模块应用 R:
      右旋 (W' = W @ R): q_a_proj, kv_a_proj_with_mqa, gate_proj, up_proj, gate(router), lm_head
      左旋 (W' = R^T @ W): o_proj, down_proj

    MLA 专用旋转 (R2/R3/R4) 是成对的 — 两边都不做时计算仍正确, 故只应用 R.
    """
    # lm_head
    if name == 'lm_head' or name.endswith('.lm_head'):
        return 'right'
    # q_a_proj (不含 q_b_proj)
    if name.endswith('.q_a_proj'):
        return 'right'
    # kv_a_proj_with_mqa
    if name.endswith('.kv_a_proj_with_mqa'):
        return 'right'
    # o_proj
    if name.endswith('.o_proj'):
        return 'left'
    # indexer 模块 (DSA indexer, 右旋 by R)
    if name.endswith('.indexer.wk') or name.endswith('.indexer.weights_proj'):
        return 'right'
    # mlp.gate (router) — 必须在 gate_proj 之前检查
    if name.endswith('.mlp.gate'):
        return 'right'
    # gate_proj, up_proj (dense + shared_experts, 不含 experts)
    if name.endswith('.gate_proj') or name.endswith('.up_proj'):
        return 'right'
    # down_proj (dense + shared_experts, 不含 experts)
    if name.endswith('.down_proj'):
        return 'left'
    return None


def _detect_tied_word_embeddings(model: nn.Module) -> bool:
    """检查 model.embed_tokens 与 lm_head 是否共享权重 (tie_word_embeddings)"""
    embed_mod = None
    lm_head_mod = None
    for name, module in model.named_modules():
        if name in ('model.embed_tokens', 'embed_tokens'):
            embed_mod = module
        elif name == 'lm_head':
            lm_head_mod = module
    return (embed_mod is not None and lm_head_mod is not None
            and embed_mod.weight is lm_head_mod.weight)


def _rotate_module_weight(module: nn.Module, side: str, R: torch.Tensor) -> int:
    """对单个 Module 应用旋转, 返回 1 表示已旋转 (right 或 left), 否则 0"""
    device = module.weight.device
    if device.type == 'meta':
        return 0

    R_dev = R.to(device=device, dtype=module.weight.dtype)
    with torch.no_grad():
        if side == 'right':
            # W' = W @ R
            module.weight.data = torch.matmul(module.weight.data, R_dev)
            return 1
        if side == 'left':
            # W' = R^T @ W
            module.weight.data = torch.matmul(R_dev.t(), module.weight.data)
            return 1
    return 0


def _apply_quarot_rotation(model: nn.Module, R: torch.Tensor, verbose: bool = True):
    """对非 expert 权重应用 QuaRot 运行时旋转.

    Pre-run 旋转 (embed_tokens 右旋) 已 baked 到保存的权重中.
    Runtime 旋转 (q_a_proj, o_proj, gate_proj 等) 未 baked, 需推理时应用.

    右旋: W' = W @ R  (输入投影, lm_head, router)
    左旋: W' = R^T @ W (输出投影)

    3D expert 权重由流式 forward 处理, 此处跳过.
    """
    if R is None:
        return

    tied = _detect_tied_word_embeddings(model)

    right_count = 0
    left_count = 0
    skip_expert = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # 跳过 3D expert 模块 (流式 forward 处理)
        if '.experts.' in name:
            skip_expert += 1
            continue

        # 跳过 tied lm_head (已通过 embed_tokens pre-run 旋转)
        if tied and (name == 'lm_head' or name.endswith('.lm_head')):
            continue

        side = _get_quarot_side(name)
        if side is None:
            continue

        rotated = _rotate_module_weight(module, side, R)
        if side == 'right':
            right_count += rotated
        elif side == 'left':
            left_count += rotated

    if verbose:
        logger.info(f"  QuaRot 旋转: {right_count} 右旋, {left_count} 左旋, "
                     f"{skip_expert} expert 跳过(流式)")


def _collect_active_experts(expert_hit, num_experts):
    """从 expert_hit 中收集活跃 expert 索引列表 (跳过 padding)。"""
    active_experts = []
    for expert_idx_t in expert_hit:
        expert_idx = expert_idx_t[0].item()
        if expert_idx == num_experts:
            continue
        active_experts.append(expert_idx)
    return active_experts


def _batch_dequant_experts(active_experts, prefix, qw, qd, dt):
    """批量 CPU 反量化 expert 权重, 返回 (gate_up_list, down_list)。"""
    gate_up_list = []
    down_list = []
    for expert_idx in active_experts:
        gate_w = _dequant_expert_weight_cpu(
            f"{prefix}.{expert_idx}.gate_proj", qw, qd, dt
        )
        up_w = _dequant_expert_weight_cpu(
            f"{prefix}.{expert_idx}.up_proj", qw, qd, dt
        )
        down_w = _dequant_expert_weight_cpu(
            f"{prefix}.{expert_idx}.down_proj", qw, qd, dt
        )
        if gate_w is None or up_w is None or down_w is None:
            logger.warning(f"  expert {expert_idx} 反量化失败, 跳过")
            gate_up_list.append(None)
            down_list.append(None)
            continue
        gate_up_list.append(torch.cat([gate_w, up_w], dim=0))
        down_list.append(down_w)
    return gate_up_list, down_list


def _apply_quarot_rotation(gate_up_stack, down_stack, R_mat, target_device):
    """应用 QuaRot 运行时旋转: gate_up 右旋 (W @ R), down 左旋 (R^T @ W)。"""
    if R_mat is None:
        return gate_up_stack, down_stack
    R_dev = R_mat.to(device=target_device, dtype=gate_up_stack.dtype)
    gate_up_stack = torch.matmul(gate_up_stack, R_dev)
    down_stack = torch.matmul(R_dev.t(), down_stack)
    return gate_up_stack, down_stack


def _install_streaming_moe_forward(
    model: nn.Module,
    quant_weights: Dict[str, torch.Tensor],
    quant_desc: Dict[str, str],
    dtype: torch.dtype,
    verbose: bool = True,
    R: Optional[torch.Tensor] = None,
):
    """
    Monkey-patch GlmMoeDsaNaiveMoe.forward — 逐 expert 从 CPU dict 读 W4A8 →
    NPU 反量化 → F.linear, 不全量加载 3D expert 权重到 NPU。

    必须在 _prepare_expert_placeholders + distribute_model + dequantize_model_on_devices
    (skip_3d_experts=True) 之后调用。
    """
    try:
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaNaiveMoe
    except ImportError:
        if verbose:
            logger.warning("  无法导入 GlmMoeDsaNaiveMoe, 流式 expert 未安装")
        return

    # 找到所有 expert 模块及其 full name (作为 quant_weights key 前缀)
    expert_modules = []
    for name, module in model.named_modules():
        if isinstance(module, GlmMoeDsaNaiveMoe):
            expert_modules.append((module, name))

    if not expert_modules:
        if verbose:
            logger.info("  未找到 GlmMoeDsaNaiveMoe 模块, 无需流式 expert")
        return

    # 在每个 expert 模块上存储流式配置
    for module, prefix in expert_modules:
        module._streaming_config = {
            'quant_weights': quant_weights,
            'quant_desc': quant_desc,
            'dtype': dtype,
            'expert_prefix': prefix,
            'R': R,  # QuaRot 旋转矩阵 (None = 无旋转)
        }

    # 保存原始 forward (用于 fallback)
    _original_moe_forward = GlmMoeDsaNaiveMoe.forward

    # 进度计数器
    _forward_call_count = [0]

    def streaming_forward(self, hidden_states, top_k_index, top_k_weights):
        cfg = getattr(self, '_streaming_config', None)
        if cfg is None:
            return _original_moe_forward(self, hidden_states, top_k_index, top_k_weights)

        qw = cfg['quant_weights']
        qd = cfg['quant_desc']
        dt = cfg['dtype']
        prefix = cfg['expert_prefix']
        target_device = str(hidden_states.device)

        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = nn.functional.one_hot(
                top_k_index, num_classes=self.num_experts
            ).permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()

        # 收集活跃 expert 列表
        active_experts = _collect_active_experts(expert_hit, self.num_experts)
        if not active_experts:
            return final_hidden_states

        # 批量 CPU 反量化 + 拼接, 只做 2 次 NPU 传输 (gate_up_stack, down_stack)
        gate_up_list, down_list = _batch_dequant_experts(
            active_experts, prefix, qw, qd, dt
        )

        # 批量传输到 NPU (仅 2 次 sync)
        valid = [(i, gu, d) for i, (gu, d) in enumerate(zip(gate_up_list, down_list)) if gu is not None]
        if not valid:
            return final_hidden_states

        gate_up_stack = torch.stack([gu for _, gu, _ in valid]).to(target_device)
        down_stack = torch.stack([d for _, _, d in valid]).to(target_device)

        # QuaRot 运行时旋转
        gate_up_stack, down_stack = _apply_quarot_rotation(
            gate_up_stack, down_stack, cfg.get('R'), target_device
        )

        # 逐 expert F.linear (权重已在 NPU)
        for stack_idx, (active_idx, _, _) in enumerate(valid):
            expert_idx = active_experts[active_idx]

            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            gate, up = nn.functional.linear(
                current_state, gate_up_stack[stack_idx]
            ).chunk(2, dim=-1)
            current_hidden = self.act_fn(gate) * up
            current_hidden = nn.functional.linear(current_hidden, down_stack[stack_idx])
            current_hidden = current_hidden * top_k_weights[token_idx, top_k_pos, None]

            final_hidden_states.index_add_(
                0, token_idx, current_hidden.to(final_hidden_states.dtype)
            )

        del gate_up_stack, down_stack

        # 进度日志 (每 100 次调用)
        _forward_call_count[0] += 1
        if _forward_call_count[0] % 100 == 0:
            logger.info(f"  [streaming] {_forward_call_count[0]} calls, "
                  f"last layer: {prefix}, {len(active_experts)} experts active")

        return final_hidden_states

    GlmMoeDsaNaiveMoe.forward = streaming_forward

    # 在 model 上保存引用, 防止 quant_weights 被 GC
    model._streaming_quant_weights = quant_weights

    if verbose:
        logger.info(f"  流式 expert forward 已安装: {len(expert_modules)} 个 MoE 层")


def _dequant_expert_weight_cpu(
    quant_name: str,
    quant_weights: Dict[str, torch.Tensor],
    quant_desc: Dict[str, str],
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """CPU-only 反量化单个 expert 权重 (不搬 NPU, 用于批量传输优化)"""
    weight_key = f"{quant_name}.weight"
    base_name = parse_base_name(weight_key)
    quant_type = quant_desc.get(weight_key, quant_desc.get(base_name, "FLOAT"))

    weight_data = quant_weights.get(weight_key)
    if weight_data is None:
        return None

    if quant_type == "FLOAT":
        return weight_data.to(dtype)

    if quant_type in ("W8A8_DYNAMIC", "W8A8_MIX"):
        ws = quant_weights.get(f"{quant_name}.weight_scale")
        wo = quant_weights.get(f"{quant_name}.weight_offset")
        if ws is not None:
            return dequantize_weight_dynamic(weight_data, ws, wo, dtype)
        return None

    if quant_type in ("W4A8_DYNAMIC", "W4A8", "W4A16"):
        ws = quant_weights.get(f"{quant_name}.weight_scale")
        wo = quant_weights.get(f"{quant_name}.weight_offset")
        if weight_data.dtype == torch.int8 and ws is not None:
            unpacked = unpack_int4_to_int8(weight_data)
            return dequantize_weight_dynamic(unpacked, ws, wo, dtype)
        return None

    if quant_type in _MX_QUANT_TYPES:
        ws = quant_weights.get(f"{quant_name}.weight_scale")
        if ws is not None:
            return dequantize_weight_mx(weight_data, ws, quant_type, dtype=dtype)
        return None

    deq_scale = quant_weights.get(f"{quant_name}.deq_scale")
    input_scale = quant_weights.get(f"{quant_name}.input_scale")
    if deq_scale is not None:
        w_fp, _ = dequantize_weight_static(
            weight_data, deq_scale, input_scale, dtype=dtype
        )
        return w_fp

    return None


DEFAULT_CONVERSATIONS = [
    [
        {"role": "user", "content": "你好，请介绍一下你自己。"},
    ],
    [
        {"role": "user", "content": "1+1等于几？"},
    ],
]


def _remove_layer_hooks(model):
    """移除 decoder layer 上的 pre-forward hooks (手动 forward 不需要)。

    distribute_model 注册的 hooks 会自动迁移 hidden_states 跨设备,
    但在多卡 + model.generate() 下可能导致乱码。
    手动 forward 显式管理设备迁移, 不需要 hooks。
    保留 lm_head 的 post-forward hook (logits 迁移到 device[0])。
    """
    try:
        layers = get_model_components(model).layers
    except ValueError:
        return

    count = 0
    for layer in layers:
        if hasattr(layer, '_forward_pre_hooks'):
            hooks_to_remove = list(layer._forward_pre_hooks.keys())
            for hook_id in hooks_to_remove:
                layer._forward_pre_hooks.pop(hook_id, None)
                count += 1
        if hasattr(layer, '_forward_pre_hooks_with_kwargs'):
            layer._forward_pre_hooks_with_kwargs.clear()
    logger.info(f"  移除 {count} 个 pre-forward hooks (使用手动设备迁移)")


def _manual_move_position_embeddings(position_embeddings, target_dev: str):
    if position_embeddings is None:
        return None
    """将 position_embeddings (tuple 或单一 tensor) 移动到目标设备"""
    if isinstance(position_embeddings, tuple):
        return tuple(t.to(target_dev) for t in position_embeddings)
    return position_embeddings.to(target_dev)


def _manual_forward_layers(layers, hidden_states, position_ids, position_embeddings,
                           past_kv, get_layer_device, n_layers: int):
    """手动逐层 forward with KV cache, 显式移动 hidden_states 跨设备"""
    topk_indices = None
    for layer_idx in range(n_layers):
        target_dev = get_layer_device(layer_idx)

        # 移动到目标设备
        hidden_states = hidden_states.to(target_dev)
        if topk_indices is not None:
            topk_indices = topk_indices.to(target_dev)
        pe_dev = _manual_move_position_embeddings(position_embeddings, target_dev)
        pos_ids_dev = position_ids.to(target_dev)

        out = layers[layer_idx](
            hidden_states,
            position_ids=pos_ids_dev,
            position_embeddings=pe_dev,
            prev_topk_indices=topk_indices,
            past_key_values=past_kv,
            use_cache=True,
        )

        if isinstance(out, tuple):
            hidden_states = out[0]
            if len(out) > 1 and out[1] is not None:
                topk_indices = out[1]
        else:
            hidden_states = out

    return hidden_states


def _manual_generate(model, tokenizer, input_ids, device_list,
                      max_new_tokens, thinking="chat"):
    """手动逐层 forward with KV cache, 绕过 model.generate() 和 pre-forward hooks。

    显式移动 hidden_states 跨设备, 逐层调用 forward。
    使用 DynamicCache 进行 KV cache, 每层更新自己的 cache entry。
    不传 attention_mask (GLM 层 forward 内部处理 causal attention)。
    """
    components = get_model_components(model)
    layers = components.layers
    embed = components.embed
    rotary_emb = components.rotary_emb
    norm = components.final_norm
    lm_head = components.lm_head
    if any(component is None for component in (embed, norm, lm_head)):
        raise RuntimeError("手动生成要求 embedding/norm/lm_head 均可探测")

    n_layers = len(layers)
    n_devices = len(device_list)
    layers_per_device = (n_layers + n_devices - 1) // n_devices

    def get_layer_device(i):
        return device_list[min(i // layers_per_device, n_devices - 1)]

    embed_device = device_list[0]
    final_device = device_list[-1]

    from transformers import DynamicCache
    past_kv = DynamicCache()
    generated = []
    current_ids = input_ids

    for step in range(max_new_tokens):
        # Embedding
        ids_dev = current_ids.to(embed_device)
        hidden_states = embed(ids_dev)

        # Position IDs
        seq_len = current_ids.shape[1]
        if step == 0:
            past_len = 0
        else:
            try:
                past_len = past_kv.get_seq_length()
            except Exception:
                past_len = step
        position_ids = torch.arange(
            past_len, past_len + seq_len, device=embed_device
        ).unsqueeze(0)

        # Rotary embeddings (computed once on embed_device, moved per-layer)
        position_embeddings = (
            rotary_emb(hidden_states, position_ids=position_ids)
            if rotary_emb is not None else None
        )

        # 逐层 forward (手动移动 hidden_states 跨设备)
        hidden_states = _manual_forward_layers(
            layers, hidden_states, position_ids, position_embeddings,
            past_kv, get_layer_device, n_layers)

        # Final norm + lm_head
        hidden_states = hidden_states.to(final_device)
        hidden_states = norm(hidden_states)
        logits = lm_head(hidden_states)

        # Greedy sampling
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        next_id = next_token.item()
        generated.append(next_id)

        if next_id == tokenizer.eos_token_id:
            break

        # 下一步只处理新 token
        current_ids = next_token.to(embed_device)

        if (step + 1) % 5 == 0 or step == 0:
            partial = tokenizer.decode(generated, skip_special_tokens=True)
            logger.info(f"  step {step+1}/{max_new_tokens}: '{partial}'")

    return generated


def _model_loop_generate(model, tokenizer, input_ids, device_list,
                          max_new_tokens, thinking="chat"):
    """用 model() 直接循环生成, 绕过 model.generate()。

    保留 pre-forward hooks (处理跨卡设备迁移), 但不使用 model.generate()。
    传 2D attention_mask (全 1), 让 model.forward() 内部转换为 causal mask。
    model.generate() 会创建 4D causal mask 并传给 model, 可能与 GLM MLA 不兼容。
    """
    from transformers import DynamicCache
    first_device = device_list[0]
    past_kv = DynamicCache()
    generated = []
    current_ids = input_ids

    for step in range(max_new_tokens):
        # 2D attention_mask (全 1, 无 padding)
        attention_mask = torch.ones_like(current_ids)
        with torch.no_grad():
            out = model(
                current_ids,
                attention_mask=attention_mask,
                past_key_values=past_kv,
                use_cache=True,
            )

        logits = out.logits
        past_kv = out.past_key_values

        # Greedy sampling
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        next_id = next_token.item()
        generated.append(next_id)

        if next_id == tokenizer.eos_token_id:
            break

        current_ids = next_token.to(first_device)

        if (step + 1) % 5 == 0 or step == 0:
            partial = tokenizer.decode(generated, skip_special_tokens=True)
            logger.info(f"  step {step+1}/{max_new_tokens}: '{partial}'")

    return generated


def _hf_log_header(model_path: str, device_list: List[str], dtype: str,
                   use_cpu_dequant: bool, verbose: bool):
    """打印 HF 推理检查头部日志"""
    if not verbose:
        return
    logger.info("=" * 70)
    logger.info("HF Inference Check — 排除框架影响")
    logger.info("=" * 70)
    logger.info(f"  模型: {model_path}")
    logger.info(f"  设备: {device_list}")
    logger.info(f"  dtype: {dtype}")
    if use_cpu_dequant:
        logger.info(f"  反量化模式: CPU (旧流程)")
    else:
        logger.info(f"  反量化模式: NPU 加速 (v2)")


def _hf_load_tokenizer(model_path: str, verbose: bool):
    """[1/6] 加载 tokenizer，处理 GLM-5 TokenizersBackend 回退"""
    if verbose:
        logger.info("[1/6] 加载 tokenizer...")
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except ValueError:
        # GLM-5 的 tokenizer_class="TokenizersBackend" 不在 transformers 里，fallback 到 fast tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
            tokenizer_class="PreTrainedTokenizerFast")
    # Some converted checkpoints carry the tokenizer files but omit the
    # optional Jinja file.  Load it only into this in-memory tokenizer object;
    # never write into the quantized model directory.  DeepSeek-V4 official
    # releases commonly omit this file and are handled by the encoder-format
    # fallback in input_resolver.py.
    if not getattr(tokenizer, "chat_template", None):
        template_path = os.path.join(model_path, "chat_template.jinja")
        if os.path.isfile(template_path):
            try:
                with open(template_path, encoding="utf-8") as f:
                    tokenizer.chat_template = f.read()
                if verbose:
                    logger.info("  已加载模型目录 chat_template.jinja（仅运行时，不修改模型文件）")
            except OSError as exc:
                logger.warning("  chat_template.jinja 读取失败: %s", exc)
        elif "deepseek" in str(model_path).lower() and "v4" in str(model_path).lower():
            if verbose:
                logger.info("  DeepSeek-V4 未提供 Jinja chat_template，将使用官方 encoder 兼容格式")
    if verbose:
        logger.info(f"  vocab_size: {tokenizer.vocab_size}")
    return tokenizer


def _hf_detect_streaming_mode(model: nn.Module, model_path: str,
                              is_quant: bool) -> Tuple[bool, bool]:
    """检测 3D expert 参数和 GLM MoE 类型，返回 (has_3d_experts, use_streaming)"""
    has_3d_experts = any(
        p.dim() == 3 and (
            'experts.gate_up_proj' in n or
            ('experts.down_proj' in n and 'shared_expert' not in n)
        )
        for n, p in model.named_parameters()
    )
    # GLM and DeepSeek-V4 both use large 3D packed experts.  Expanding all
    # expert weights to BF16 would defeat layer sharding and usually OOM.
    supports_streaming = (
        'glm' in type(model).__name__.lower() or
        'glm' in str(getattr(model.config, 'model_type', '')).lower() or
        'deepseekv4' in type(model).__name__.lower() or
        str(getattr(model.config, 'model_type', '')).lower() == 'deepseek_v4'
    )
    use_streaming = is_quant and has_3d_experts and supports_streaming
    return has_3d_experts, use_streaming


def _hf_create_skeleton(model_path: str, torch_dtype: torch.dtype,
                        verbose: bool) -> Tuple[nn.Module, bool, bool]:
    """[2/6] 创建模型骨架，处理 streaming expert placeholders"""
    if verbose:
        logger.info("\n[2/6] 创建模型骨架...")
    model = create_model_skeleton(model_path, dtype=torch_dtype, verbose=verbose)

    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        logger.info(f"  骨架: {total_params/1e9:.2f}B 参数")

    components = get_model_components(model)
    if verbose:
        logger.info(
            f"  结构探测: {type(components.text_model).__name__}, "
            f"{len(components.layers)} layers"
        )

    is_quant = is_quantized_model(model_path)
    has_3d_experts, use_streaming = _hf_detect_streaming_mode(model, model_path, is_quant)

    if use_streaming:
        if verbose:
            logger.info("  检测到 3D packed expert → 流式 expert 模式")
        expert_shapes = _prepare_expert_placeholders(model)
        if verbose:
            logger.info(f"  替换 {len(expert_shapes)} 个 3D expert 参数为 placeholder")

    return model, is_quant, use_streaming


def _hf_load_quant_weights(model: nn.Module, model_path: str, torch_dtype: torch.dtype,
                           device_list: List[str], use_cpu_dequant: bool,
                           use_streaming: bool, verbose: bool,
                           expert_chunk_size: Optional[int] = None):
    """[4/6] 量化分支: 加载量化权重 + 描述 + NPU 反量化 + 流式安装"""
    if verbose:
        logger.info("  [v2] NPU 加速反量化..." if not use_cpu_dequant
                    else "  [旧流程] CPU 全量反量化...")

    quant_weights = load_quant_weights(model_path)
    if verbose:
        logger.info(f"  加载 {len(quant_weights)} 个量化权重到 CPU")

    desc_path = os.path.join(model_path, "quant_model_description.json")
    with open(desc_path) as f:
        quant_desc_raw = json.load(f)
    meta_keys = {"metadata", "optional", "group_size", "version",
                 "kv_cache_type", "kv_quant_type", "model_quant_type"}
    quant_desc = {k: v for k, v in quant_desc_raw.items()
                  if k not in meta_keys and isinstance(v, str)}
    quant_desc = normalize_quant_desc_values(quant_desc)

    # Official DeepSeek-V4 files use native attn/ffn names.  Our boundary
    # loader works directly from the tensor dictionary, so expose the runtime
    # aliases that Transformers' regular loader would otherwise create.
    from .model_loader import add_deepseek_v4_checkpoint_aliases
    alias_count = add_deepseek_v4_checkpoint_aliases(
        model, quant_weights, quant_desc
    )
    if verbose and alias_count:
        logger.info("  DeepSeek-V4 checkpoint 映射: %d 个 runtime alias", alias_count)

    if use_cpu_dequant:
        # 旧流程: CPU 全量反量化 → 再搬 NPU
        raise NotImplementedError(
            "CPU 全量反量化 (dequantize_model) 未在 clean 分支恢复；"
            "请用默认 NPU 加速路径 (不加 --use_cpu_dequant)"
        )
        _load_float_weights(model, quant_weights, quant_desc, torch_dtype,
                            device=device_list[0], verbose=verbose)
    else:
        # v2: NPU 加速反量化 (模型已在 NPU 上)
        dequantize_model_on_devices(
            model, quant_weights, quant_desc, torch_dtype,
            verbose=verbose, skip_3d_experts=use_streaming,
        )

    # QuaRot: msmodelslim QuaRotProcessor.post_run() 已通过 rotate_linear()
    # 将旋转烘焙到权重中 (W' = W @ R 或 W' = R^T @ W)。
    # optional/quarot.safetensors 中的 global_rotation 仅供推理框架参考,
    # 不应在 runtime 再次旋转权重 (会导致双重旋转 → 乱码)。
    R = None  # 权重已预旋转, runtime 不再旋转

    if use_streaming:
        from .resident_experts import install_resident_streaming_moe
        install_resident_streaming_moe(
            model, quant_weights, quant_desc, torch_dtype,
            chunk_size=expert_chunk_size or 8, verbose=verbose,
        )
    else:
        del quant_weights

    return R


def _hf_load_nonquant_weights(model: nn.Module, model_path: str,
                              torch_dtype: torch.dtype, verbose: bool):
    """[4/6] 非量化分支: 直接加载 safetensors"""
    from safetensors.torch import load_file
    all_weights = {}
    for f_name in sorted(os.listdir(model_path)):
        if f_name.endswith(".safetensors") and (
            f_name.startswith("model-") or
            f_name.startswith("model.safetensors") or
            f_name == "model.safetensors"
        ):
            shard = load_file(os.path.join(model_path, f_name))
            all_weights.update(shard)
    if verbose:
        logger.info(f"  加载 {len(all_weights)} 个权重 (非量化)")

    # DeepSeek-V4 ModelScope exports use native bare names (``layers.*``,
    # ``embed.*``, ``head.*``), while the Transformers module tree uses
    # ``model.layers.*``.  Apply the same aliasing used by the quantized
    # boundary path so a reference model cannot fail solely on key prefixes.
    from .model_loader import add_deepseek_v4_checkpoint_aliases
    alias_count = add_deepseek_v4_checkpoint_aliases(model, all_weights)
    if verbose and alias_count:
        logger.info("  DeepSeek-V4 checkpoint 映射: %d 个 runtime alias", alias_count)

    # 加载 parameters
    loaded_params = 0
    for name, param in model.named_parameters():
        if name in all_weights:
            param.data = all_weights[name].to(torch_dtype).to(str(param.device))
            loaded_params += 1

    # 加载 buffers (A_log, dt_bias, conv1d.weight 等)
    loaded_buffers = 0
    for name, buf in model.named_buffers():
        if name in all_weights:
            buf.data = all_weights[name].to(buf.dtype).to(str(buf.device))
            loaded_buffers += 1

    if verbose:
        logger.info(f"  加载 {loaded_params} 个参数, {loaded_buffers} 个 buffer")
    del all_weights


def _hf_load_weights(model: nn.Module, model_path: str, torch_dtype: torch.dtype,
                     device_list: List[str], is_quant: bool, use_cpu_dequant: bool,
                     use_streaming: bool, verbose: bool,
                     expert_chunk_size: Optional[int] = None):
    """[4/6] 加载权重 + 反量化, 返回耗时"""
    if verbose:
        logger.info(f"\n[4/6] 加载权重 ({'量化' if is_quant else '非量化'})...")

    t0 = time.time()
    if is_quant:
        _hf_load_quant_weights(model, model_path, torch_dtype, device_list,
                               use_cpu_dequant, use_streaming, verbose,
                               expert_chunk_size=expert_chunk_size)
    else:
        _hf_load_nonquant_weights(model, model_path, torch_dtype, verbose)

    gc.collect()
    load_time = time.time() - t0
    if verbose:
        logger.info(f"  权重加载耗时: {load_time:.1f}s")
    return load_time


def _hf_verify_model_loaded(model: nn.Module, verbose: bool):
    """[5/6] 验证模型完整性, 列出仍处于 meta 设备的参数/buffer"""
    if verbose:
        logger.info("\n[5/6] 验证模型完整性...")
    still_meta = sum(1 for _, p in model.named_parameters() if p.device.type == 'meta')
    still_meta += sum(1 for _, b in model.named_buffers() if b.device.type == 'meta')
    if still_meta > 0:
        logger.warning(f"仍有 {still_meta} 个参数/buffer 在 meta 设备上!")
        if verbose:
            for n, p in model.named_parameters():
                if p.device.type == 'meta':
                    logger.info(f"    param: {n} shape={list(p.shape)}")
            for n, b in model.named_buffers():
                if b.device.type == 'meta':
                    logger.info(f"    buffer: {n} shape={list(b.shape)}")
    else:
        logger.info("  所有参数已加载到设备 ✅")


def _hf_generation_kwargs(generation_config):
    """Translate supported OpenAI/vLLM sampling fields to ``generate`` kwargs.

    An explicitly positive temperature means sampling, including the common
    ``temperature=1, top_p=1`` request.  Temperature zero remains greedy.
    Unsupported OpenAI penalties fail fast instead of silently changing the
    distribution used by Boundary.
    """
    config = dict(generation_config or {})
    unsupported = []
    for name in ("frequency_penalty", "presence_penalty"):
        value = config.get(name)
        if value not in (None, 0, 0.0):
            unsupported.append(f"{name}={value}")
    if unsupported:
        raise ValueError(
            "Transformers sampling 暂不支持严格对齐: " + ", ".join(unsupported)
        )

    temperature = config.get("temperature")
    explicit_do_sample = config.get("do_sample")
    if explicit_do_sample is not None:
        do_sample = bool(explicit_do_sample)
    elif temperature is not None:
        do_sample = float(temperature) > 0
    else:
        top_p = config.get("top_p")
        top_k = config.get("top_k")
        do_sample = (
            (top_p is not None and float(top_p) < 1.0)
            or (top_k is not None and int(top_k) > 0)
        )

    kwargs = {"do_sample": do_sample}
    if do_sample:
        if temperature is not None and float(temperature) > 0:
            kwargs["temperature"] = float(temperature)
        if config.get("top_p") is not None:
            kwargs["top_p"] = float(config["top_p"])
        if config.get("top_k") is not None:
            # vLLM uses -1/0 for disabled top-k; HF uses 0.
            kwargs["top_k"] = max(0, int(config["top_k"]))
    repetition_penalty = config.get("repetition_penalty")
    if repetition_penalty not in (None, 1, 1.0):
        kwargs["repetition_penalty"] = float(repetition_penalty)
    return kwargs


def _seed_hf_generation(generation_config):
    seed = (generation_config or {}).get("seed")
    if seed is None:
        return None
    seed = int(seed)
    torch.manual_seed(seed)
    npu = getattr(torch, "npu", None)
    if npu is not None and hasattr(npu, "manual_seed_all"):
        npu.manual_seed_all(seed)
    return seed


def _hf_generate_conversation_batch(model, tokenizer, messages, request_tools, thinking,
                                    max_new_tokens, first_device, conv_idx, batch_size,
                                    run_offset, chat_template_mode="auto",
                                    generation_config=None,
                                    print_full_output=False):
    from .input_resolver import resolve_model_input

    raw_prompt = (len(messages) == 1 and messages[0].get("_raw_prompt") is True)
    if raw_prompt:
        resolved = resolve_model_input(
            tokenizer, prompt=messages[0]["content"], source_kind="text",
            thinking=thinking, chat_template_mode=chat_template_mode,
        )
    else:
        processed = preprocess_messages(messages)
        resolved = resolve_model_input(
            tokenizer, messages=processed, source_kind="messages",
            request_tools=request_tools, thinking=thinking,
            chat_template_mode=chat_template_mode,
        )
    text = resolved["rendered_text"]
    input_ids = resolved["input_ids"].to(first_device)
    batched_ids = input_ids.repeat(batch_size, 1)
    attention_mask = torch.ones_like(input_ids).repeat(batch_size, 1)

    last_user = [m["content"] for m in messages if m["role"] == "user"][-1] \
        if any(m["role"] == "user" for m in messages) else f"对话{conv_idx+1}"
    logger.info(f"\n--- 对话 {conv_idx+1} (last user: {last_user[:60]}...) ---")
    logger.info(f"  输入 token 数: {input_ids.shape[1]}, batch={batch_size}, "
                f"runs={run_offset + 1}-{run_offset + batch_size}")

    sampling_kwargs = _hf_generation_kwargs(generation_config)
    seed = _seed_hf_generation(generation_config)
    sampling_desc = "sampling" if sampling_kwargs["do_sample"] else "greedy"
    logger.info(
        f"  生成策略: {sampling_desc}, "
        f"temperature={sampling_kwargs.get('temperature', 'n/a')}, "
        f"top_p={sampling_kwargs.get('top_p', 'n/a')}, "
        f"top_k={sampling_kwargs.get('top_k', 'n/a')}, seed={seed}"
    )

    with torch.no_grad():
        t0 = time.time()
        from transformers import LogitsProcessorList
        from .generation_progress import GenerationProgressProcessor
        progress = GenerationProgressProcessor(input_ids.shape[1], logger=logger)
        output = model.generate(batched_ids, attention_mask=attention_mask,
                                max_new_tokens=max_new_tokens,
                                use_cache=True, return_dict_in_generate=True,
                                logits_processor=LogitsProcessorList([progress]),
                                **sampling_kwargs)
        gen_time = time.time() - t0

    results = []
    for batch_idx in range(batch_size):
        new_tokens = output.sequences[batch_idx][input_ids.shape[1]:]
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            while len(new_tokens) and new_tokens[-1].item() == pad_id:
                new_tokens = new_tokens[:-1]
        decoded = decode_output(tokenizer, new_tokens)
        run_index = run_offset + batch_idx + 1
        preview = decoded["answer"] or decoded["thinking"]
        logger.info(f"  run {run_index}: {len(new_tokens)} tokens, "
                    f"{preview[:160] if preview else '(empty)'}")
        if print_full_output:
            logger.info(f"  [full output run {run_index}]\n{decoded['raw_text']}")
        results.append({
            "run_index": run_index, "batch_index": batch_idx, "messages": messages,
            "generated": decoded["answer"], "thinking": decoded["thinking"],
            "raw_text": decoded["raw_text"], "input_tokens": input_ids.shape[1],
            "output_tokens": len(new_tokens), "batch_time": gen_time, "time": gen_time,
            "thinking_truncated": decoded["thinking_truncated"],
            "generation_config": {**sampling_kwargs, "seed": seed},
        })
    return results


def _hf_run_generation(model, tokenizer, prompt_file, thinking, max_new_tokens,
                       first_device, verbose: bool, num_runs: int = 1,
                       concurrency: int = 1,
                       stop_predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
                       conversations_override=None, request_tools_override=None,
                       chat_template_mode: str = "auto",
                       generation_config=None,
                       print_full_output: bool = False):
    if verbose:
        logger.info("\n[6/6] 推理测试...")
    if conversations_override is not None:
        conversations = conversations_override
        request_tools = request_tools_override
    elif prompt_file:
        conversations, request_tools = load_prompt_file(prompt_file)
    else:
        conversations = DEFAULT_CONVERSATIONS
        request_tools = None

    results = []
    for i, messages in enumerate(conversations):
        completed = 0
        while completed < num_runs:
            # A fixed seed denotes one deterministic request.  Run seeded
            # requests individually so every repetition starts from that same
            # RNG state instead of sharing a batch-level random stream.
            seeded = (generation_config or {}).get("seed") is not None
            batch_size = 1 if seeded else min(concurrency, num_runs - completed)
            batch_results = _hf_generate_conversation_batch(
                model, tokenizer, messages, request_tools, thinking,
                max_new_tokens, first_device, i, batch_size, completed,
                chat_template_mode, generation_config,
                print_full_output,
            )
            results.extend(batch_results)
            completed += batch_size
            if stop_predicate and any(stop_predicate(r) for r in batch_results):
                logger.info(f"  检测到 bad case，提前停止于 run {completed}")
                break
    return results


def _hf_replay_logits(model, input_ids, positions, first_device, verbose=True,
                      attention_mask=None):
    """Replay one captured input-id sequence and return logits at exact positions.

    This path deliberately bypasses tokenizer/chat-template/generate.  It is
    used by Boundary intermittent mode, where the captured vLLM input_ids are
    the source of truth and only sampler-input vocabulary logits are needed.
    """
    from .logits_compare import LogitsCollection

    ids = torch.as_tensor(input_ids, dtype=torch.long)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    if ids.dim() != 2 or ids.shape[0] != 1:
        raise ValueError("captured replay input_ids must have shape [1,S]")
    pos = [int(p) for p in positions]
    if not pos:
        raise ValueError("captured replay positions must be non-empty")
    if min(pos) < 0 or max(pos) >= ids.shape[1]:
        raise ValueError(
            f"captured replay position outside input sequence: S={ids.shape[1]}, positions={pos}"
        )
    ids = ids.to(first_device)
    if attention_mask is None:
        attention_mask = torch.ones_like(ids)
    else:
        attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if attention_mask.shape != ids.shape:
            raise ValueError("captured replay attention_mask must match input_ids")
        attention_mask = attention_mask.to(first_device)
    if verbose:
        logger.info(
            "  [intermittent replay] direct input_ids forward: S=%d, positions=%s",
            ids.shape[1], pos if len(pos) <= 12 else f"{pos[:6]}...{pos[-3:]}",
        )
    with torch.no_grad():
        try:
            output = model(ids, attention_mask=attention_mask,
                           use_cache=False, return_dict=True)
        except TypeError:
            output = model(ids, attention_mask=attention_mask, use_cache=False)
    logits = output.logits if hasattr(output, "logits") else output[0]
    if logits is None or logits.dim() != 3:
        raise RuntimeError("Transformers replay did not return [B,S,V] vocabulary logits")
    selected = logits[0, pos, :].detach().to("cpu", dtype=torch.float32)
    return LogitsCollection(
        token_positions=pos,
        logits=selected,
        input_ids=ids.detach().to("cpu"),
        position_mode="captured_replay",
    )


def _hf_run_ppl(model, tokenizer, first_device, skip_ppl: bool):
    """[可选] PPL 计算"""
    if skip_ppl:
        return
    logger.info("\n--- Perplexity ---")
    test_text = "自然语言处理是人工智能领域的一个重要分支，它研究计算机与人类语言之间的交互方式。"
    try:
        encodings = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = encodings["input_ids"].to(first_device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        ppl = torch.exp(loss).item()
        logger.info(f"  PPL = {ppl:.2f} (loss = {loss:.4f})")
    except Exception as e:
        logger.info(f"  PPL 计算失败: {e}")


def _hf_log_summary(results: List[Dict[str, Any]]):
    """打印结果汇总"""
    logger.info("\n" + "=" * 70)
    logger.info("结果汇总:")
    logger.info("=" * 70)
    for i, r in enumerate(results):
        last_user = [m["content"] for m in r['messages'] if m['role'] == 'user'][-1][:40] \
            if any(m['role'] == 'user' for m in r['messages']) else f"对话{i+1}"
        gen_preview = r['generated'][:100].replace('\n', ' ') if r['generated'] else "(思维链截断，无正文)"
        logger.info(f"  [{i+1}] Q: {last_user}")
        logger.info(f"      A: {gen_preview}")
        logger.info(f"      tokens: {r['input_tokens']}+{r['output_tokens']}, time: {r['time']:.1f}s")

    logger.info("\n如果生成结果通顺合理 → 量化本身没问题，精度问题在推理框架")
    logger.info("如果生成结果乱码/重复 → 量化本身有问题，需要用 V2 逐层定位")
    logger.info("=" * 70)


def hf_inference_check(
    model_path: str,
    devices: str = "npu:0",
    dtype: str = "bfloat16",
    max_new_tokens: int = 2048,
    prompt_file: str = None,
    skip_ppl: bool = False,
    thinking: str = "chat",
    verbose: bool = True,
    use_cpu_dequant: bool = False,
    noquit: bool = False,
    num_runs: int = 1,
    concurrency: int = 1,
    stop_predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
    conversations: Optional[List[List[Dict[str, Any]]]] = None,
    request_tools: Optional[List[Dict[str, Any]]] = None,
    expert_chunk_size: Optional[int] = None,
    prefill_parallel: str = "pp",
    glm_attn_query_block: Optional[int] = None,
    glm_attn_selected_block: Optional[int] = None,
    deepseek_v4_query_block: Optional[int] = None,
    deepseek_v4_key_block: Optional[int] = None,
    chat_template_mode: str = "auto",
    generation_config: Optional[Dict[str, Any]] = None,
    print_full_output: bool = False,
    replay_input_ids=None,
    replay_positions=None,
    replay_attention_mask=None,
) -> Any:
    """
    通用 HF 推理检查入口。

    v2 默认流程: skeleton → distribute NPU → NPU 上反量化 (快, 省内存)
    use_cpu_dequant=True: 回退到旧流程 (CPU 全量反量化 → 搬 NPU)
    noquit=True: 推理完成后进入交互模式，模型留在 NPU 上可反复推理

    Args:
        model_path: 量化模型路径
        devices: 设备列表，如 "npu:0,1,2,3"
        dtype: 数据类型
        max_new_tokens: 最大生成 token 数
        prompt_file: prompt 文件路径（vLLM 请求格式或对话列表）
        skip_ppl: 跳过 PPL 计算
        thinking: thinking 模式, "chat" (开思维链) / "none" (关闭)
        verbose: 是否打印详细进度
        use_cpu_dequant: 回退到旧 CPU 全量反量化流程
        noquit: 推理完成后不退出，进入交互模式

    Returns:
        结果列表
    """
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map[dtype]
    if num_runs < 1 or concurrency < 1:
        raise ValueError("num_runs 和 concurrency 必须为正整数")
    device_list = parse_devices(devices)

    # Install the GLM DSA long-prefill runtime before model construction.  In
    # TP mode keeps the PP-sharded model layers on their owning cards while
    # dispatching bounded query blocks across the configured device group.
    from .glm_dsa_blockwise import install_glm_dsa_blockwise_indexer
    glm_blockwise_ok = install_glm_dsa_blockwise_indexer(
        parallel_mode=prefill_parallel,
        device_groups=[device_list],
        attention_query_block=glm_attn_query_block,
        attention_selected_block=glm_attn_selected_block,
    )
    if not glm_blockwise_ok:
        logger.warning(
            "  GLM DSA blockwise indexer 未安装；长 prompt 将使用 Transformers "
            "eager indexer，可能产生超大 score tensor"
        )
    if verbose and str(prefill_parallel).lower() == "tp":
        logger.info("  GLM 长 prefill: TP indexer 已启用（PP 层分片保留）")

    from .deepseek_v4_blockwise import install_deepseek_v4_blockwise_runtime
    install_deepseek_v4_blockwise_runtime(
        query_block=deepseek_v4_query_block,
        key_block=deepseek_v4_key_block,
        parallel_mode=prefill_parallel,
        device_groups=[device_list],
    )
    if verbose and str(prefill_parallel).lower() == "tp":
        logger.info("  DeepSeek-V4 长 prefill: TP query-parallel 已启用（PP 层分片保留）")

    _hf_log_header(model_path, device_list, dtype, use_cpu_dequant, verbose)

    # ---- Step 1: tokenizer ----
    tokenizer = _hf_load_tokenizer(model_path, verbose)

    # ---- Step 2: 创建骨架 ----
    model, is_quant, use_streaming = _hf_create_skeleton(
        model_path, torch_dtype, verbose)

    # ---- Step 3: 多卡分布骨架 (先于反量化!) ----
    if verbose:
        logger.info(f"\n[3/6] 分布模型骨架到 {len(device_list)} 个设备...")
    distribute_model(model, device_list)
    # ``to_empty`` leaves non-persistent RoPE buffers uninitialized.  Rebuild
    # them after sharding so global, compressor and indexer rotary modules stay
    # on their owning devices (DeepSeek-V4 has main + compress RoPE for each).
    from .model_loader import _initialize_rotary_modules
    _initialize_rotary_modules(model)

    # ---- Step 4: 加载权重 + 反量化 ----
    _hf_load_weights(model, model_path, torch_dtype, device_list,
                     is_quant, use_cpu_dequant, use_streaming, verbose,
                     expert_chunk_size=expert_chunk_size)

    # ---- Step 5: 验证 ----
    _hf_verify_model_loaded(model, verbose)

    model.eval()
    first_device = device_list[0]

    if replay_input_ids is not None:
        if replay_positions is None:
            raise ValueError("replay_positions is required with replay_input_ids")
        return _hf_replay_logits(
            model, replay_input_ids, replay_positions, first_device, verbose=verbose,
            attention_mask=replay_attention_mask,
        )

    # ---- Step 6: 推理 ----
    results = _hf_run_generation(
        model, tokenizer, prompt_file, thinking, max_new_tokens, first_device,
        verbose, num_runs=num_runs, concurrency=concurrency,
        stop_predicate=stop_predicate, conversations_override=conversations,
        request_tools_override=request_tools,
        chat_template_mode=chat_template_mode,
        generation_config=generation_config,
        print_full_output=print_full_output,
    )

    # PPL
    _hf_run_ppl(model, tokenizer, first_device, skip_ppl)

    # 总结
    _hf_log_summary(results)

    if noquit:
        _interactive_loop(model, tokenizer, first_device, device_list,
                          thinking, max_new_tokens, verbose)

    return results


def _interactive_load_conversations(user_input: str):
    """根据用户输入加载对话: .json 文件或直接文本"""
    if user_input.endswith('.json') and os.path.exists(user_input):
        return load_prompt_file(user_input)
    return [[{"role": "user", "content": user_input}]], None


def _interactive_log_decoded(decoded: Dict[str, Any], gen_time: float,
                            new_tokens_len: int):
    """打印单次交互的解码结果"""
    if decoded["thinking_truncated"]:
        logger.info(f"  生成 ({gen_time:.1f}s, {new_tokens_len} tokens):")
        logger.info(f"  [Thinking (截断)]: {decoded['thinking'][:500]}")
        return
    if decoded["thinking"]:
        logger.info(f"  生成 ({gen_time:.1f}s, {new_tokens_len} tokens):")
        logger.info(f"  [Thinking]: {decoded['thinking'][:500]}")
        logger.info(f"  [Answer]: {decoded['answer'][:500]}")
        return
    logger.info(f"  生成 ({gen_time:.1f}s, {new_tokens_len} tokens): {decoded['answer'][:500]}")


def _interactive_generate_one(model, tokenizer, first_device,
                             thinking, max_new_tokens, messages, request_tools):
    """对单条对话执行 generate 并打印结果"""
    processed = preprocess_messages(messages)

    template_kwargs = dict(
        conversation=processed, tokenize=False, add_generation_prompt=True,
        enable_thinking=(thinking == "chat"),
    )
    if request_tools:
        template_kwargs["tools"] = request_tools

    text = tokenizer.apply_chat_template(**template_kwargs)
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(first_device)

    logger.info(f"  输入 token 数: {input_ids.shape[1]}")

    with torch.no_grad():
        t0 = time.time()
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        gen_time = time.time() - t0

    new_tokens = output[0][input_ids.shape[1]:]
    decoded = decode_output(tokenizer, new_tokens)
    _interactive_log_decoded(decoded, gen_time, len(new_tokens))


def _interactive_loop(model, tokenizer, first_device, device_list,
                      thinking, max_new_tokens, verbose):
    """交互式推理循环，模型已加载在 NPU 上，反复接收输入"""
    logger.info("\n" + "=" * 70)
    logger.info("进入交互模式 (模型已加载，无需重新加载)")
    logger.info("输入 prompt 进行推理，输入 quit/exit 退出")
    logger.info("=" * 70)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            logger.info("\n退出交互模式")
            break

        if user_input.lower() in ('quit', 'exit', 'q'):
            logger.info("退出交互模式")
            break

        if not user_input:
            continue

        conversations, request_tools = _interactive_load_conversations(user_input)

        for messages in conversations:
            _interactive_generate_one(model, tokenizer, first_device,
                                     thinking, max_new_tokens,
                                     messages, request_tools)

    logger.info("模型仍在 NPU 上，进程退出后自动释放")


# ============================================================================
# Qwen3.5 快捷函数
# ============================================================================

def qwen35_inference_check(
    model_path: str,
    devices: str = "npu:0,1,2,3",
    dtype: str = "bfloat16",
    max_new_tokens: int = 4096,
    prompt_file: str = None,
    skip_ppl: bool = True,
    thinking: str = "chat",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Qwen3.5-35B-A3B W8A8 HF 推理检查快捷函数。

    用法:
        from accuracy_checker.inference_check import qwen35_inference_check
        results = qwen35_inference_check(
            model_path="/path/to/Qwen3.5-35B-A3B-306",
            devices="npu:0,1,2,3",
            prompt_file="scripts/qwen35_badcase_prompt.json",
        )

    或命令行:
        ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python3 scripts/qwen35_inference_check.py \\
            --model_path /path/to/Qwen3.5-35B-A3B-306 \\
            --devices npu:0,1,2,3 --prompt_file scripts/qwen35_badcase_prompt.json
    """
    return hf_inference_check(
        model_path=model_path,
        devices=devices,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        prompt_file=prompt_file,
        skip_ppl=skip_ppl,
        thinking=thinking,
        verbose=verbose,
    )


# ============================================================================
# Boundary 定界 (L0.5) — 已拆分到 boundary_check.py (保持向后兼容 re-export)
# ============================================================================
# 原本 760 行 boundary 代码 (BoundaryResult/detect_badcase/classify_boundary/
# run_boundary/run_boundary_cli/print_boundary_verdict_explain 等) 已移至
# boundary_check.py 以满足 CI Huge Non Headerfile (≤2000行) 约束。
# 此处 re-export 保持所有历史 import 路径不变:
#   from accuracy_checker.inference_check import run_boundary / BoundaryResult / ...
from .boundary_check import (
    BoundaryResult,
    WEIGHT_OR_QUANTIZATION, INFERENCE_FRAMEWORK, BOTH, INCONCLUSIVE, INVALID_RUN,
    INTERMITTENT_LOGITS_ALIGNED, INTERMITTENT_LOGITS_MISMATCH,
    INTERMITTENT_RANKING_SENSITIVE,
    repeat_4gram_ratio, nonprintable_ratio,
    detect_badcase,
    classify_boundary, run_boundary, boundary_result_to_dict,
    run_boundary_cli as _run_boundary_cli_impl,
    print_boundary_verdict_explain as _print_boundary_verdict_explain_impl,
)


def _run_boundary_cli(args):
    """--mode boundary CLI: 委托给 boundary_check.run_boundary_cli (拆分后薄包装)。"""
    _run_boundary_cli_impl(args)


def _print_boundary_verdict_explain(kind: str) -> None:
    """委托给 boundary_check.print_boundary_verdict_explain (拆分后薄包装)。"""
    _print_boundary_verdict_explain_impl(kind)


# ============================================================================
# CLI — 已拆分到 cli.py (保持向后兼容 re-export)
# ============================================================================
from .cli import main as _cli_main


def main():
    """CLI 入口 (委托给 cli.main, 拆分后薄包装)。"""
    _cli_main()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
