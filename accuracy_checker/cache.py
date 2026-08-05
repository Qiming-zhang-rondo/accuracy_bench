"""
L2 Cache 机制

L2 把 target_layer 的 input hidden_states 缓存到磁盘。
同 model/prompt/layer/version 再次运行时直接加载，跳过前置层。

V1 L1 和 V2 subgraph_locate 共用此模块。

Cache 目录:
  优先级: --cache_dir 参数 > ACC_CACHE_DIR 环境变量 > 当前目录下的 .acc_cache/
  目录不存在时自动创建。
"""

import os
import hashlib

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


ADAPTER_VERSION = "2"  # bump when adapter logic changes
INT4_UNPACK_VERSION = "2"  # bump when _decode_int4_packed changes


def model_hash(model_path: str) -> str:
    return hashlib.sha256(model_path.encode()).hexdigest()[:8]


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]


def _cache_key(model_path, prompt, seqlen, target_layer, side, quant_mode):
    mh = model_hash(model_path)
    ph = prompt_hash(prompt)
    return f"{mh}_{ph}_s{seqlen}_L{target_layer}_{side}_{ADAPTER_VERSION}_{INT4_UNPACK_VERSION}_{quant_mode}.pt"


def save_cache(model_path, prompt, seqlen, target_layer, side, quant_mode, tensor):
    cache_dir = get_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(model_path, prompt, seqlen, target_layer, side, quant_mode)
    path = os.path.join(cache_dir, key)
    torch.save(tensor.cpu(), path)
    return path


def load_cache(model_path, prompt, seqlen, target_layer, side, quant_mode, device):
    cache_dir = get_cache_dir()
    key = _cache_key(model_path, prompt, seqlen, target_layer, side, quant_mode)
    path = os.path.join(cache_dir, key)
    if not os.path.exists(path):
        return None
    tensor = torch.load(path, weights_only=True, map_location=device)
    return tensor
