"""
L2 Cache 机制

L2 把 target_layer 的 input hidden_states 缓存到磁盘。
同 model/prompt/layer/version 再次运行时直接加载，跳过前置层。

V1 L1 和 V2 subgraph_locate 共用此模块。

Cache 目录:
  优先级: --cache_dir 参数 > ACC_CACHE_DIR 环境变量 > 当前目录下的 .acc_cache/
  目录不存在时自动创建。
"""

import hashlib
import json
import os
import time

import torch


_cache_dir_override = None


def set_cache_dir(path: str):
    """设置 cache 目录（由 --cache_dir 参数调用）"""
    global _cache_dir_override
    _cache_dir_override = path


def get_cache_dir() -> str:
    if _cache_dir_override is not None:
        return _cache_dir_override
    env = os.environ.get("ACC_CACHE_DIR", "")
    if env:
        return env
    return os.path.join(os.getcwd(), ".acc_cache")


def get_report_dir() -> str:
    return os.path.join(get_cache_dir(), "reports")


CACHE_FORMAT_VERSION = "4"  # v4 also stores token ids needed by hash-routed MoE replay
INT4_UNPACK_VERSION = "2"  # bump when _decode_int4_packed changes


def model_hash(model_path: str) -> str:
    return hashlib.sha256(model_path.encode()).hexdigest()[:8]


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]


def _cache_key(model_path, prompt, seqlen, target_layer, side, quant_mode):
    mh = model_hash(model_path)
    ph = prompt_hash(prompt)
    return f"{mh}_{ph}_s{seqlen}_L{target_layer}_{side}_{CACHE_FORMAT_VERSION}_{INT4_UNPACK_VERSION}_{quant_mode}.pt"


def _latest_manifest_path(ref_model_path, quant_model_path, prompt, quant_mode):
    identity = "\0".join((ref_model_path, quant_model_path, prompt, quant_mode))
    digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
    return os.path.join(get_cache_dir(), f"latest_l1_{digest}.json")


def save_latest_l1_cache_manifest(ref_model_path, quant_model_path, prompt,
                                  quant_mode, layers):
    """Publish the cache layers produced by the latest successful L1 run."""
    cache_dir = get_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    path = _latest_manifest_path(
        ref_model_path, quant_model_path, prompt, quant_mode
    )
    payload = {
        "version": 1,
        "created_at": time.time(),
        "ref_model_hash": model_hash(ref_model_path),
        "quant_model_hash": model_hash(quant_model_path),
        "prompt_hash": prompt_hash(prompt),
        "quant_mode": quant_mode,
        "layers": sorted({int(layer) for layer in layers}),
    }
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    return path


def load_latest_l1_cache_manifest(ref_model_path, quant_model_path, prompt,
                                  quant_mode):
    """Return the latest L1 layer set, or None for legacy/no manifest."""
    path = _latest_manifest_path(
        ref_model_path, quant_model_path, prompt, quant_mode
    )
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError):
        return None
    layers = payload.get("layers")
    if not isinstance(layers, list):
        return None
    try:
        return sorted({int(layer) for layer in layers})
    except (TypeError, ValueError):
        return None


def save_cache(model_path, prompt, seqlen, target_layer, side, quant_mode,
               tensor, layer_state=None, input_ids=None):
    cache_dir = get_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(model_path, prompt, seqlen, target_layer, side, quant_mode)
    path = os.path.join(cache_dir, key)
    if layer_state is None and input_ids is None:
        payload = tensor.cpu()
    else:
        payload = {
            "hidden_states": tensor.cpu(),
            "layer_state": layer_state.cpu() if layer_state is not None else None,
            "input_ids": input_ids.cpu() if input_ids is not None else None,
        }
    torch.save(payload, path)
    return path


def load_cache(model_path, prompt, seqlen, target_layer, side, quant_mode, device):
    cache_dir = get_cache_dir()
    key = _cache_key(model_path, prompt, seqlen, target_layer, side, quant_mode)
    path = os.path.join(cache_dir, key)
    if not os.path.exists(path):
        return None
    tensor = torch.load(path, weights_only=True, map_location=device)
    return tensor
