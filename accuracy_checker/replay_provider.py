"""
ReplayProvider: 为 V2 LocalPatchRunner 提供单层 replay handle。

复用 V1 的模型骨架创建、分片加载、position_embeddings 逻辑。
只提供 get_layer_handle() 和 forward_ref/quant() 接口，不关心 patch 细节。

Usage:
    provider = ReplayProvider(ref_model_path, quant_model_path)
    handle = provider.get_layer_handle(layer_idx=2, device="npu:0", quant_device="npu:1")
    ref_out = handle.forward_ref(hidden)
    quant_out = handle.forward_quant(hidden)
    handle.cleanup()
    provider.close()
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

import logging

from accuracy_checker.model_loader import (
    ShardWeightReader,
    build_weight_index,
    create_model_skeleton,
    is_quantized_model,
    is_compressed_tensors_model,
    load_layer_weights_indexed,
    move_layers_to_device,
    unload_layers_to_meta,
)
from accuracy_checker.utils import get_decoder_layers, get_rotary_emb_module

logger = logging.getLogger(__name__)


class ReplayProvider:
    """
    管理 ref/quant 两个模型的 skeleton、weight_index、quant_desc。

    职责:
      1. 创建两个模型的 skeleton（meta device）
      2. 构建 weight_index 和 ShardWeightReader
      3. 解析量化描述（quant_model_description.json）
      4. 提供 get_layer_handle() 创建 LayerReplayHandle
    """

    def __init__(
        self,
        ref_model_path: str,
        quant_model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        verbose: bool = True,
    ):
        self.ref_model_path = ref_model_path
        self.quant_model_path = quant_model_path
        self.dtype = dtype
        self.verbose = verbose

        if verbose:
            logger.info(f"[ReplayProvider] Creating model skeletons...")

        # --- ref skeleton ---
        self.ref_model = create_model_skeleton(ref_model_path, dtype, verbose)
        self.ref_model.eval()

        # --- quant skeleton ---
        self.quant_model = create_model_skeleton(quant_model_path, dtype, verbose)
        self.quant_model.eval()

        # --- weight index ---
        self.ref_weight_map = build_weight_index(ref_model_path)
        self.quant_weight_map = build_weight_index(quant_model_path)

        # --- ShardWeightReader ---
        self.ref_reader = ShardWeightReader(ref_model_path, self.ref_weight_map)
        self.quant_reader = ShardWeightReader(quant_model_path, self.quant_weight_map)

        # --- quant description (msmodelslim) ---
        self.quant_desc = self._load_quant_desc(quant_model_path)
        self.is_quant = is_quantized_model(quant_model_path)
        self.is_ct = is_compressed_tensors_model(quant_model_path)

        # --- ref is_quant ---
        self.ref_is_quant = is_quantized_model(ref_model_path)
        self.ref_is_ct = is_compressed_tensors_model(ref_model_path)

        # --- 获取 decoder layers 引用 ---
        self.ref_layers = get_decoder_layers(self.ref_model)
        self.quant_layers = get_decoder_layers(self.quant_model)
        self.num_layers = len(self.ref_layers)

        # --- rotary_emb modules (用于生成 position_embeddings) ---
        self.ref_rotary = get_rotary_emb_module(self.ref_model)
        self.quant_rotary = get_rotary_emb_module(self.quant_model)

        if verbose:
            logger.info(f"[ReplayProvider] Model has {self.num_layers} layers")
            logger.info(f"[ReplayProvider] ref_rotary: {self.ref_rotary is not None}")
            logger.info(f"[ReplayProvider] quant_rotary: {self.quant_rotary is not None}")
            logger.info(f"[ReplayProvider] quant is_quant={self.is_quant} is_ct={self.is_ct}")

    def _load_quant_desc(self, model_path: str) -> Optional[dict]:
        """加载 quant_model_description.json（msmodelslim 量化描述）"""
        import os
        desc_path = os.path.join(model_path, "quant_model_description.json")
        if os.path.exists(desc_path):
            import json
            with open(desc_path) as f:
                return json.load(f)
        return None

    def get_layer_handle(
        self,
        layer_idx: int,
        device: str = "npu:0",
        quant_device: Optional[str] = None,
        prompt: Optional[str] = None,
        ref_prefix_layers: Optional[List[int]] = None,
        ref_hidden_override: Optional[torch.Tensor] = None,
        quant_hidden_override: Optional[torch.Tensor] = None,
    ) -> "LayerReplayHandle":
        """
        获取目标层的 replay handle。

        Args:
            layer_idx: 目标层索引
            device: ref 模型设备
            quant_device: quant 模型设备（None = 与 ref 同设备）
            prompt: 输入文本（如果为 None，使用随机 dummy hidden_states）
            ref_prefix_layers: 如果提供，逐层 forward ref 的 prefix 层得到真实输入
            ref_hidden_override: 直接提供 ref hidden_states（如从 V1 L1 cache 加载）
            quant_hidden_override: 直接提供 quant hidden_states（如从 V1 L1 cache 加载）
                提供 override 时，prompt/ref_prefix_layers 被忽略。

        Returns:
            LayerReplayHandle
        """
        if quant_device is None:
            quant_device = device

        if self.verbose:
            logger.info(f"  [get_layer_handle] Layer {layer_idx}: ref={device}, quant={quant_device}")

        # 加载 embed_tokens 权重（两个模型各加载一次）
        self._load_embed_tokens(
            self.ref_model, self.ref_model_path, device,
            self.ref_weight_map, self.ref_reader,
            self.ref_is_quant, self.ref_is_ct,
        )
        self._load_embed_tokens(
            self.quant_model, self.quant_model_path, quant_device,
            self.quant_weight_map, self.quant_reader,
            self.is_quant, self.is_ct,
        )

        # 加载目标层权重
        self._load_layer_weights(
            self.ref_model, self.ref_model_path, [layer_idx], device,
            self.ref_weight_map, self.ref_reader,
            self.ref_is_quant, self.ref_is_ct,
        )
        self._load_layer_weights(
            self.quant_model, self.quant_model_path, [layer_idx], quant_device,
            self.quant_weight_map, self.quant_reader,
            self.is_quant, self.is_ct,
        )

        # 获取 hidden_states
        if ref_hidden_override is not None and quant_hidden_override is not None:
            # 从外部提供 hidden_states（如 V1 L1 cache）
            ref_hidden = ref_hidden_override
            quant_hidden = quant_hidden_override
            # Move to device for position_embeddings computation
            layer_kwargs = self._build_layer_kwargs(ref_hidden.to(device), device)
        elif ref_prefix_layers:
            # 逐层 forward ref prefix 层得到真实 hidden
            if prompt is None:
                raise ValueError("ref_prefix_layers requires a prompt")
            ref_hidden, _, _ = self._prepare_from_prompt(prompt, device, quant_device)
            ref_hidden = self._forward_prefix(ref_hidden, ref_prefix_layers, device)
            quant_hidden = ref_hidden.to(quant_device)
            layer_kwargs = self._build_layer_kwargs(ref_hidden, device)
            # prefix forward 后目标层可能被 clear_others 清掉，重新加载
            self._load_layer_weights(
                self.ref_model, self.ref_model_path, [layer_idx], device,
                self.ref_weight_map, self.ref_reader,
                self.ref_is_quant, self.ref_is_ct,
                clear_others=False,
            )
            self._load_layer_weights(
                self.quant_model, self.quant_model_path, [layer_idx], quant_device,
                self.quant_weight_map, self.quant_reader,
                self.is_quant, self.is_ct,
                clear_others=False,
            )
        else:
            ref_hidden, quant_hidden, layer_kwargs = self._prepare_inputs(
                prompt, device, quant_device, layer_idx,
            )

        # 获取 ref/quant 的目标层 module
        ref_layer = self.ref_layers[layer_idx]
        quant_layer = self.quant_layers[layer_idx]

        return LayerReplayHandle(
            layer_idx=layer_idx,
            ref_layer=ref_layer,
            quant_layer=quant_layer,
            ref_model=self.ref_model,
            quant_model=self.quant_model,
            ref_hidden=ref_hidden,
            quant_hidden=quant_hidden,
            layer_kwargs=layer_kwargs,
            ref_device=device,
            quant_device=quant_device,
            ref_layers=self.ref_layers,
            quant_layers=self.quant_layers,
        )

    def _forward_prefix(
        self,
        hidden: torch.Tensor,
        prefix_layers: List[int],
        device: str,
    ) -> torch.Tensor:
        """Forward prefix layers one-by-one (load→forward→unload) to avoid OOM.

        Supports GLM-5 DSA: decoder layer forward returns (hidden, topk_indices),
        topk_indices is passed to the next layer as prev_topk_indices.
        """
        rotary = get_rotary_emb_module(self.ref_model)
        batch_size, seq_len = hidden.shape[:2]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = None
        if rotary is not None:
            position_embeddings = rotary(hidden, position_ids)

        h = hidden
        topk_indices = None
        for pl in prefix_layers:
            # Load only this single layer
            load_layer_weights_indexed(
                self.ref_model, self.ref_model_path, layers=[pl],
                device=device, dtype=self.dtype,
                weight_map=self.ref_weight_map, sf_reader=self.ref_reader,
                is_quant=self.ref_is_quant, is_ct=self.ref_is_ct,
                quant_desc=None,
                use_fake_quant=False, verbose=False,
            )
            move_layers_to_device(self.ref_model, [pl], device, clear_others=False, skip_3d_experts=False)

            layer = self.ref_layers[pl]
            layer.eval()
            with torch.no_grad():
                out = layer(
                    h,
                    position_embeddings=position_embeddings,
                    prev_topk_indices=topk_indices,
                )
            if isinstance(out, tuple):
                h = out[0]
                topk_indices = out[1] if len(out) > 1 else None
            else:
                h = out
                topk_indices = None

            # Unload this layer immediately
            unload_layers_to_meta(self.ref_model, [pl])
            gc.collect()

        return h

    def _build_layer_kwargs(
        self, hidden: torch.Tensor, device: str,
    ) -> Dict[str, Any]:
        """为 decoder layer forward 构建 layer_kwargs（position_embeddings）。"""
        from accuracy_checker.utils import get_rotary_emb_module
        rotary = get_rotary_emb_module(self.ref_model)
        batch_size, seq_len = hidden.shape[:2]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = None
        if rotary is not None:
            # Qwen3.6 ForConditionalGeneration: rotary_emb 参数可能在 CPU，需要移到 device
            rotary = rotary.to(device)
            position_embeddings = rotary(hidden, position_ids)
        return {
            "position_embeddings": position_embeddings,
            "position_ids": position_ids,
        }

    def _load_embed_tokens(
        self, model, model_path, device, weight_map, reader, is_quant, is_ct,
    ):
        """加载 embed_tokens + norm + lm_head 等非层权重。

        load_layer_weights_indexed:
          layers=[]   — 只加载 embed_tokens、rotary_emb 等
          layers=[-1] — 只加载 norm、lm_head
        需要两次调用才能加载所有非层权重。
        """
        for special_layers in ([], [-1]):
            load_layer_weights_indexed(
                model, model_path, layers=special_layers,
                device=device, dtype=self.dtype,
                weight_map=weight_map, sf_reader=reader,
                is_quant=is_quant, is_ct=is_ct,
                quant_desc=self.quant_desc if is_quant else None,
                use_fake_quant=False, verbose=False,
            )

    def _load_layer_weights(
        self, model, model_path, layers, device, weight_map, reader, is_quant, is_ct,
        clear_others: bool = True,
    ):
        """加载指定层的权重。"""
        load_layer_weights_indexed(
            model, model_path, layers=layers,
            device=device, dtype=self.dtype,
            weight_map=weight_map, sf_reader=reader,
            is_quant=is_quant, is_ct=is_ct,
            quant_desc=self.quant_desc if is_quant else None,
            use_fake_quant=False, verbose=False,
        )
        # 把目标层移到 device (L2需要完整前向, 不跳过3D expert)
        move_layers_to_device(model, layers, device, clear_others=clear_others, skip_3d_experts=False)

    def _prepare_inputs(
        self, prompt: Optional[str], device: str, quant_device: str, layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        准备 ref_hidden, quant_hidden, layer_kwargs。

        如果 prompt 为 None，使用随机 dummy hidden_states（只用于验证 patch 机制）。
        """
        if prompt is not None:
            return self._prepare_from_prompt(prompt, device, quant_device)
        else:
            return self._prepare_dummy(device, quant_device, layer_idx)

    def _prepare_dummy(
        self, device: str, quant_device: str, layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """用随机 dummy hidden_states，适合测试 patch 机制。"""
        # 获取 hidden_size
        sample_param = next(self.ref_layers[layer_idx].parameters())
        if hasattr(self.ref_model.config, 'hidden_size'):
            hidden_size = self.ref_model.config.hidden_size
        else:
            hidden_size = sample_param.shape[-1]

        batch_size, seq_len = 1, 8
        ref_hidden = torch.randn(batch_size, seq_len, hidden_size, dtype=self.dtype, device=device)
        quant_hidden = torch.randn(batch_size, seq_len, hidden_size, dtype=self.dtype, device=quant_device)

        # 生成 position_ids + position_embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = None
        if self.ref_rotary is not None:
            position_embeddings = self.ref_rotary(ref_hidden, position_ids)

        layer_kwargs = {
            "position_embeddings": position_embeddings,
            "position_ids": position_ids,
        }
        return ref_hidden, quant_hidden, layer_kwargs

    def _prepare_from_prompt(
        self, prompt: str, device: str, quant_device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        通过模型的 embed_tokens 处理 prompt，得到 hidden_states。
        复用 V1 的方式：embed_tokens(input_ids) → hidden_states。
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.ref_model_path, trust_remote_code=True
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        batch_size, seq_len = input_ids.shape

        # ref hidden — 确保 embed module 在目标 device
        ref_embed = self.ref_model.get_input_embeddings()
        ref_embed = ref_embed.to(device)
        ref_hidden = ref_embed(input_ids.to(device))

        # quant hidden
        quant_embed = self.quant_model.get_input_embeddings()
        quant_embed = quant_embed.to(quant_device)
        quant_hidden = quant_embed(input_ids.to(quant_device))

        # position_ids + position_embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = None
        if self.ref_rotary is not None:
            position_embeddings = self.ref_rotary(ref_hidden, position_ids)

        layer_kwargs = {
            "position_embeddings": position_embeddings,
            "position_ids": position_ids,
        }
        return ref_hidden, quant_hidden, layer_kwargs

    def close(self):
        """关闭 ShardWeightReader 文件句柄。"""
        self.ref_reader.close()
        self.quant_reader.close()


@dataclass
class LayerReplayHandle:
    """
    单层 replay handle。

    管理 ref/quant 双卡 device、forward_ref/quant、cleanup。
    LocalPatchRunner 通过这个 handle 操作，不直接接触模型加载逻辑。

    属性:
        layer_idx: 目标层索引
        ref_layer: ref 目标层 module
        quant_layer: quant 目标层 module
        ref_model: ref 完整模型 skeleton
        quant_model: quant 完整模型 skeleton
        ref_hidden: ref 层输入 hidden_states
        quant_hidden: quant 层输入 hidden_states（从 V1 L1 cache 时在 rotated 空间）
        layer_kwargs: forward 额外参数（position_embeddings, position_ids）
        ref_device: ref 模型设备
        quant_device: quant 模型设备
    """
    layer_idx: int
    ref_layer: nn.Module
    quant_layer: nn.Module
    ref_model: nn.Module
    quant_model: nn.Module
    ref_hidden: torch.Tensor
    quant_hidden: torch.Tensor
    layer_kwargs: Dict[str, Any] = field(default_factory=dict)
    ref_device: str = "npu:0"
    quant_device: str = "npu:1"
    ref_layers: Optional[nn.ModuleList] = None
    quant_layers: Optional[nn.ModuleList] = None

    def __post_init__(self):
        self.ref_layer.eval()
        self.quant_layer.eval()

    def forward_ref(
        self,
        hidden: torch.Tensor,
        hooks: Optional[List[Callable]] = None,
    ) -> torch.Tensor:
        """
        ref 模型单层 forward。

        Args:
            hidden: 输入 hidden_states
            hooks: 可选的 forward hook 列表

        Returns:
            layer output (detached, CPU)
        """
        device = next(self.ref_layer.parameters()).device
        input_hidden = hidden.to(device)

        # 确保 layer_kwargs 中的 tensor 也在 ref device 上
        kwargs = {}
        for k, v in self.layer_kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.to(device)
            elif isinstance(v, tuple) and all(isinstance(t, torch.Tensor) for t in v):
                kwargs[k] = tuple(t.to(device) for t in v)
            else:
                kwargs[k] = v

        # 注册 hooks（如果有）
        hook_handles = []
        if hooks:
            for hook_fn in hooks:
                handle = self.ref_layer.register_forward_hook(hook_fn)
                hook_handles.append(handle)

        try:
            with torch.no_grad():
                out = self.ref_layer(input_hidden, **kwargs)
            output = out[0] if isinstance(out, tuple) else out
            return output.detach().cpu()
        finally:
            for h in hook_handles:
                h.remove()

    def forward_quant(
        self,
        hidden: torch.Tensor,
        hooks: Optional[List[Callable]] = None,
    ) -> torch.Tensor:
        """
        quant 模型单层 forward。

        Args:
            hidden: 输入 hidden_states
            hooks: 可选的 forward hook 列表

        Returns:
            layer output (detached, CPU)
        """
        device = next(self.quant_layer.parameters()).device
        input_hidden = hidden.to(device)

        # 确保 layer_kwargs 中的 tensor 也在 quant device 上
        kwargs = {}
        for k, v in self.layer_kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.to(device)
            elif isinstance(v, tuple) and all(isinstance(t, torch.Tensor) for t in v):
                kwargs[k] = tuple(t.to(device) for t in v)
            else:
                kwargs[k] = v

        hook_handles = []
        if hooks:
            for hook_fn in hooks:
                handle = self.quant_layer.register_forward_hook(hook_fn)
                hook_handles.append(handle)

        try:
            with torch.no_grad():
                out = self.quant_layer(input_hidden, **kwargs)
            output = out[0] if isinstance(out, tuple) else out
            return output.detach().cpu()
        finally:
            for h in hook_handles:
                h.remove()

    def cleanup(self):
        """释放目标层的 NPU + CPU 内存。"""
        if self.ref_layers is not None:
            unload_layers_to_meta(self.ref_model, [self.layer_idx])
        if self.quant_layers is not None:
            unload_layers_to_meta(self.quant_model, [self.layer_idx])
        gc.collect()
        if hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.empty_cache()
