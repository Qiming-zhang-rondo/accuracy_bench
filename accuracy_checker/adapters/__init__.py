"""
模型 Adapter 层

将框架层代码与具体模型结构解耦。
支持多级模型检测 + 能力声明。
"""

import os
import json
from typing import Optional

from .base import BaseModelAdapter
from .qwen3 import Qwen3Adapter
from .qwen3_moe import Qwen3MoEAdapter
from .qwen3vl import Qwen3VLAdapter
from .qwen35 import Qwen35MoEAdapter
from .glm import GLMAdapter, GLM5MoEAdapter


# 模型家族 -> Adapter 的映射
# 注意：顺序重要！更具体的类型要放在前面
MODEL_FAMILY_ADAPTERS = {
    # Qwen3 MoE 系列
    "qwen3_moe": Qwen3MoEAdapter,
    # Qwen3.5 MoE 系列 (优先，因为 qwen3_5_moe 包含 qwen)
    "qwen3_5_moe": Qwen35MoEAdapter,
    "qwen3.5": Qwen35MoEAdapter,
    # Qwen3-VL 系列 (优先于 qwen3)
    "qwen3_vl": Qwen3VLAdapter,
    "qwen3vl": Qwen3VLAdapter,
    # Qwen 系列
    "qwen": Qwen3Adapter,
    "qwen2": Qwen3Adapter,
    "qwen3": Qwen3Adapter,
    # GLM-5.1 MoE DSA 系列 (优先于老 GLM)
    "glm_moe_dsa": GLM5MoEAdapter,
    "glm5": GLM5MoEAdapter,
    # GLM 系列 (老版本)
    "glm": GLMAdapter,
    "chatglm": GLMAdapter,
    "chatglm2": GLMAdapter,
    "chatglm3": GLMAdapter,
}


# 高优先级模式检测 (按优先级排序, 先匹配先返回)
_NAME_PRIORITY_PATTERNS = [
    ("qwen3_5", "qwen3_5_moe"),
    ("qwen3_vl", "qwen3_vl"),
    ("qwen3vl", "qwen3_vl"),
]

def _detect_from_name(name: str) -> Optional[str]:
    """从字符串 (arch/model_type/class_name/path_name) 检测模型家族。"""
    # 高优先级模式 (精确子串匹配)
    for pattern, family in _NAME_PRIORITY_PATTERNS:
        if pattern in name:
            return family
    # MoE 检测 (GLM DSA 优先于通用 MoE)
    if "moe" in name:
        if "glm" in name and "dsa" in name:
            return "glm_moe_dsa"
        if "qwen3" in name:
            return "qwen3_moe"
        return "qwen3_5_moe"
    # GLM MoE DSA (无 "moe" 关键字但有 "dsa")
    if "glm" in name and "dsa" in name:
        return "glm_moe_dsa"
    # 字典兜底匹配
    for family in MODEL_FAMILY_ADAPTERS:
        if family in name:
            return family
    return None


def _detect_from_config_file(model_path: str) -> Optional[str]:
    """从 model_path/config.json 检测模型家族。"""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path) as f:
        config = json.load(f)
    architectures = config.get("architectures", [])
    if architectures:
        family = _detect_from_name(architectures[0].lower())
        if family:
            return family
    model_type = config.get("model_type", "").lower()
    return _detect_from_name(model_type)


def _detect_from_model_config(model) -> Optional[str]:
    """从 model.config.model_type 检测模型家族。"""
    if not hasattr(model, "config"):
        return None
    config = model.config
    if not hasattr(config, "model_type"):
        return None
    return _detect_from_name(config.model_type.lower())


def _detect_from_class_name(model) -> Optional[str]:
    """从模型类名检测模型家族。"""
    return _detect_from_name(type(model).__name__.lower())


def _detect_from_path_name(model_path: str) -> Optional[str]:
    """从 model_path 目录名检测模型家族 (兜底)。"""
    return _detect_from_name(os.path.basename(model_path).lower())


def _detect_model_family(model, model_path: str = None) -> Optional[str]:
    """
    多级检测模型家族

    检测优先级：
    1. model_path/config.json 的 model_type
    2. model.config.model_type (HuggingFace格式)
    3. model.__class__.__name__
    4. model_path 目录名

    Returns:
        模型家族名 (如 "qwen3", "glm"), 找不到返回 None
    """
    if model_path:
        family = _detect_from_config_file(model_path)
        if family:
            return family
    family = _detect_from_model_config(model)
    if family:
        return family
    family = _detect_from_class_name(model)
    if family:
        return family
    if model_path:
        family = _detect_from_path_name(model_path)
        if family:
            return family
    return None


def get_model_adapter(model, model_path: str = None) -> Optional[BaseModelAdapter]:
    """
    根据模型自动选择合适的 Adapter

    多级检测优先级：
      1. model_path/config.json 的 model_type
      2. model.config.model_type
      3. model.__class__.__name__
      4. model_path 目录名

    Args:
        model: PyTorch 模型实例
        model_path: 模型路径（可选）

    Returns:
        对应的 ModelAdapter 实例，找不到返回 None (走 legacy fallback)
    """
    # 检测模型家族
    family = _detect_model_family(model, model_path)

    if family and family in MODEL_FAMILY_ADAPTERS:
        adapter_class = MODEL_FAMILY_ADAPTERS[family]
        return adapter_class(model, model_path)

    # 找不到合适的 adapter，返回 None 让主框架走 legacy
    return None


__all__ = [
    "BaseModelAdapter",
    "Qwen3Adapter",
    "Qwen3VLAdapter",
    "Qwen3MoEAdapter",
    "Qwen35MoEAdapter",
    "GLMAdapter",
    "GLM5MoEAdapter",
    "get_model_adapter",
]
