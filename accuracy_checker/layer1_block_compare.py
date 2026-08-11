"""
L1: 逐block对比

粗粒度数值定位器: 找出 first bad block 在哪一层。

判定逻辑:
  - first_bad_block = 首个"显著局部突降层" (rolling-window delta + MAD 检测)
  - 不再用绝对阈值 0.99 做根因定位 (W4A8 逐层累积会导致中间层误报)
  - 绝对阈值 0.99 保留为辅助告警 (first_threshold_crossing)
"""

import os
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import statistics as stat

from .metrics import compute_all_metrics, cos_sim
from .model_structure import (
    build_replay_attention_mask,
    get_layer_state_kwarg,
    get_moe_module,
    has_indexed_routed_experts,
    is_kimi_k3_layer,
)
from .report_schema import LogitsData
import logging


logger = logging.getLogger(__name__)

# ============================================================================
# 数据类型与报告类已拆分到 block_compare_types.py (保持向后兼容 re-export)
# ============================================================================
from .block_compare_types import (
    DELTA_WINDOW, DELTA_K, DELTA_MIN_DROP,
    PERSISTENCE_CHECK_LAYERS, PERSISTENCE_RECOVERY_TOL, MAD_EPS,
    _layer_idx_from_name,
    TopKCompareResult, BlockCompareResult, LayerDeltaInfo,
    BadLayerDetection, BlockCompareReport,
)

def _dispatch_act_fake_quant(x, quant_type: str):
    """按 quant_type 分派激活伪量化函数。
    W8A8_MXFP8 / W4A8_MXFP → MXFP8 per-block (A8 microscaling)
    W4A4_MXFP4            → MXFP4 per-block (A4 microscaling)
    W4A4_DYNAMIC          → INT4 per-token sym (A4 线性)
    W4A8_DYNAMIC / W8A8_DYNAMIC / W4A8 / W8A8 → INT8 per-token sym (A8 线性)
    """
    if quant_type == "W4A4_MXFP4":
        from .mxfp4_fake_quant import mxfp4_fake_quant_per_block
        return mxfp4_fake_quant_per_block(x)
    if quant_type in ("W4A4_DYNAMIC", "W4A4_LAOS"):
        from .int4_fake_quant import int4_fake_quant_per_token_sym
        return int4_fake_quant_per_token_sym(x)
    if quant_type in ("W4A8_DYNAMIC", "W8A8_DYNAMIC", "W4A8", "W8A8"):
        from .int8_fake_quant import int8_fake_quant_per_token_sym
        return int8_fake_quant_per_token_sym(x)
    # 默认: MXFP8 (W8A8_MXFP8, W4A8_MXFP)
    from .mxfp8_fake_quant import mxfp8_fake_quant_per_block
    return mxfp8_fake_quant_per_block(x)


def _make_act_fake_quant_hook(quant_type: str):
    """构造捕获 quant_type 的 pre-forward hook (闭包)。"""
    def _hook(module, input):
        x = input[0]
        x_fq = _dispatch_act_fake_quant(x, quant_type)
        return (x_fq,) + input[1:]
    return _hook


class _ExpertSliceReader:
    """Reader view that slices packed expert tensors on their first axis."""

    def __init__(self, reader, expert_id: int, num_experts: int):
        self.reader = reader
        self.expert_id = expert_id
        self.num_experts = num_experts
        self.weight_map = getattr(reader, "weight_map", {})

    def get_tensor(self, key: str):
        get_slice = getattr(self.reader, "get_tensor_slice", None)
        if callable(get_slice):
            return get_slice(
                key, self.expert_id,
                expected_first_dim=self.num_experts,
            )
        tensor = self.reader.get_tensor(key)
        if (
            tensor is not None
            and tensor.dim() > 0
            and tensor.shape[0] == self.num_experts
        ):
            return tensor[self.expert_id]
        return tensor

class ShardedBlockComparator:
    """
    多卡Block对比器

    使用hf_device_map实现多卡并行加载，适用于可用设备不足的场景。
    和普通全量加载的区别是：明确使用多卡device_map加载。
    """

    def __init__(
        self,
        ref_model_path: str,
        quant_model_path: str,
        tokenizer,
        ref_device: str = "npu:0",
        quant_device: str = "npu:1",
        dtype: torch.dtype = torch.bfloat16,
        use_fake_quant: bool = True,
        per_device_memory_gb: float = 64.0,
        verbose: bool = True,
        auto_cache_bad_layer: bool = False,
        bad_layer_threshold: float = 0.99,
        cache_top_k: int = 0,
        quant_method: str = "dequantize",
        rotation_matrix: str = None,
        l1_target_layers: List[int] = None,
        compare_mode: str = "dual",
        ref_devices: Optional[List[str]] = None,
        quant_devices: Optional[List[str]] = None,
        expert_chunk_size: Optional[int] = None,
        kimi_kda_backend: str = "auto",
        activation_quant: bool = False,
        activation_quant_type: str = "W8A8_MXFP8",
        # Full logits capture on the L1 forward pass (no extra model reload).
        # Default on; compare_logits 4-panel data feeds ReportData.logits.
        collect_full_logits: bool = True,
        # OOM guard for large vocabs (GLM-5.1 vocab=154880): keep only last N
        # positions of full-vocab logits. 32 positions × 154880 × 4B (fp32) ≈ 20MB.
        logits_max_positions: int = 32,
    ):
        self.ref_model_path = ref_model_path
        self.quant_model_path = quant_model_path
        self.tokenizer = tokenizer
        self.ref_device = ref_device
        self.quant_device = quant_device
        self.dtype = dtype
        self.use_fake_quant = use_fake_quant
        self.per_device_memory_gb = per_device_memory_gb
        self.verbose = verbose
        self.auto_cache_bad_layer = auto_cache_bad_layer
        self.bad_layer_threshold = bad_layer_threshold
        self.cache_top_k = cache_top_k
        self.quant_method = quant_method
        self.l1_target_layers = l1_target_layers  # [5,6,7] 等
        self.compare_mode = compare_mode
        self.ref_devices = ref_devices or [ref_device]
        self.quant_devices = quant_devices or [quant_device]
        self.expert_chunk_size = expert_chunk_size
        if kimi_kda_backend not in {"auto", "torch", "chunk", "fused_recurrent"}:
            raise ValueError(
                "kimi_kda_backend must be auto, torch, chunk, or fused_recurrent"
            )
        self.kimi_kda_backend = kimi_kda_backend
        self._kimi_kda_backend_logged = False
        self.activation_quant = activation_quant
        self.activation_quant_type = activation_quant_type
        self.collect_full_logits = collect_full_logits
        self.logits_max_positions = logits_max_positions
        self._activation_hooks = []  # track registered hooks for cleanup

        # Load rotation matrix (optional, for rotated quant models like GLM5.1 QuaRot)
        self.R = None
        if rotation_matrix:
            from .utils import load_rotation_matrix
            self.R = load_rotation_matrix(rotation_matrix)

        # 解析设备数
        self.num_ref_devices = len(ref_device.split(',')) if ',' in ref_device else 1
        self.num_quant_devices = len(quant_device.split(',')) if ',' in quant_device else 1

        # 加载器
        from .model_loader import load_model_for_comparison
        self._load_model = load_model_for_comparison

        # weight reader 缓存 (由 _init_dual_skeleton_and_embed 延迟初始化)
        self._ref_weight_map = None
        self._quant_weight_map = None

        # 解析实际设备（第一个）
        self.actual_ref_device = ref_device.split(',')[0]
        self.actual_quant_device = quant_device.split(',')[0]

        # 解析num_layers
        import json
        config_path = os.path.join(self.ref_model_path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.num_layers = config.get('num_hidden_layers', None)
        # 嵌套 config (如 Qwen3.5 MoE): num_hidden_layers 在 text_config 里
        if self.num_layers is None:
            text_config = config.get('text_config', {})
            self.num_layers = text_config.get('num_hidden_layers', 64)
        # NOTE: num_nextn_predict_layers (MTP) 在 config 里有, 但模型权重里
        # 没有对应的 MTP 层 (safetensors 里无 mtp/nextn 相关 key).
        # L1 只对比 model.layers ModuleList 中的层 (0..num_hidden_layers-1).
        # MTP 层是推理框架在 decode 阶段动态加的, 不在静态模型权重里.

    def _plan_shards(self, layers_per_shard: int = None) -> List[Tuple[int, int]]:
        """规划分片：返回 [(layer_start, layer_end), ...]"""
        if layers_per_shard is None:
            # 默认每批8层
            layers_per_shard = 8

        if self.l1_target_layers is not None:
            # --l1_target_layers 模式: 需要从 layer 0 forward 到 max target 层，
            # 保证前序层的 hidden_states 是正确的
            max_target = max(self.l1_target_layers)
            # 从 0 到 max_target+1 按 layers_per_shard 分片
            shards = []
            for start in range(0, max_target + 1, layers_per_shard):
                end = min(start + layers_per_shard, max_target + 1)
                shards.append((start, end))
            if self.verbose:
                logger.info(f"[Sharded L1] 目标层: {sorted(set(self.l1_target_layers))}, "
                      f"需 forward 0-{max_target}, 规划了 {len(shards)} 个shard: {shards}")
            return shards

        shards = []
        for start in range(0, self.num_layers, layers_per_shard):
            end = min(start + layers_per_shard, self.num_layers)
            shards.append((start, end))

        if self.verbose:
            logger.info(f"[Sharded L1] 规划了 {len(shards)} 个shard: {shards}")
        return shards

    # ------------------------------------------------------------------ #
    #  共用逻辑提取 (3 个 compare 方法共享)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_rss_gb() -> float:
        """读取当前进程 RSS (GB)。"""
        try:
            with open('/proc/self/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) / 1024**2
        except Exception:
            return -1

    def _build_weight_index_and_reader(self):
        """Build weight index and reader for ref/quant models."""
        from .model_loader import (build_weight_index, ShardWeightReader,
                                   is_quantized_model, is_compressed_tensors_model)
        import json as _json

        self._ref_is_quant = is_quantized_model(self.ref_model_path)
        self._ref_is_ct = is_compressed_tensors_model(self.ref_model_path)
        self._quant_is_quant = is_quantized_model(self.quant_model_path)
        self._quant_is_ct = is_compressed_tensors_model(self.quant_model_path)
        self._ref_weight_map = build_weight_index(self.ref_model_path)
        self._quant_weight_map = build_weight_index(self.quant_model_path)
        self._ref_reader = ShardWeightReader(self.ref_model_path, self._ref_weight_map)
        self._quant_reader = ShardWeightReader(self.quant_model_path, self._quant_weight_map)

        # quant_desc
        self._ref_quant_desc = None
        self._quant_quant_desc = None
        if self._ref_is_quant and not self._ref_is_ct:
            _desc_path = os.path.join(self.ref_model_path, "quant_model_description.json")
            if os.path.exists(_desc_path):
                with open(_desc_path, 'r') as _f:
                    self._ref_quant_desc = _json.load(_f)
        if self._quant_is_quant and not self._quant_is_ct:
            _desc_path = os.path.join(self.quant_model_path, "quant_model_description.json")
            if os.path.exists(_desc_path):
                with open(_desc_path, 'r') as _f:
                    self._quant_quant_desc = _json.load(_f)

    def _init_rotary_emb(self, model, device: str):
        """Initialize rotary embedding for a model."""
        from .utils import get_rotary_emb_module
        rotary = get_rotary_emb_module(model)
        if rotary is None:
            return
        single_dev = device.split(',')[0] if ',' in device else device
        rotary.to(single_dev)
        if hasattr(rotary, 'config') and hasattr(rotary, 'compute_default_rope_parameters'):
            inv_freq, attn_scaling = type(rotary).compute_default_rope_parameters(rotary.config, device=single_dev)
            rotary.inv_freq = inv_freq
            if hasattr(rotary, 'attention_scaling'):
                rotary.attention_scaling = attn_scaling
            if hasattr(rotary, 'original_inv_freq'):
                rotary.original_inv_freq = inv_freq.clone()

    def _init_dual_skeleton_and_embed(self, ref_device: str, quant_device: str):
        """创建 ref/quant 模型骨架 + 加载 embed_tokens + 初始化 rotary_emb。

        Returns:
            (ref_model, quant_model)
        """
        from .model_loader import create_model_skeleton, load_layer_weights_indexed

        if self.verbose:
            logger.info(f"\n[Sharded L1] 创建模型骨架...")

        ref_model = create_model_skeleton(self.ref_model_path, self.dtype, verbose=False)
        quant_model = create_model_skeleton(self.quant_model_path, self.dtype, verbose=False)

        if self.verbose:
            logger.info(f"  ref/quant 模型骨架已创建")
            logger.info(f"  加载 embed_tokens, rotary_emb 到各自设备...")

        # 构建 weight_map + reader，缓存到 self 供后续 shard 加载复用
        if not hasattr(self, '_ref_weight_map') or self._ref_weight_map is None:
            self._build_weight_index_and_reader()

        layers = []
        load_layer_weights_indexed(ref_model, self.ref_model_path, layers, ref_device, self.dtype,
                                   self._ref_weight_map, self._ref_reader,
                                   is_quant=self._ref_is_quant, is_ct=self._ref_is_ct,
                                   quant_desc=self._ref_quant_desc,
                                   verbose=True)
        load_layer_weights_indexed(quant_model, self.quant_model_path, layers, quant_device, self.dtype,
                                   self._quant_weight_map, self._quant_reader,
                                   is_quant=self._quant_is_quant, is_ct=self._quant_is_ct,
                                   quant_desc=self._quant_quant_desc,
                                   verbose=True)

        # rotary_emb 初始化
        self._init_rotary_emb(ref_model, ref_device)
        self._init_rotary_emb(quant_model, quant_device)

        if self.verbose:
            logger.info("  embed_tokens, rotary_emb 已加载")

        return ref_model, quant_model

    def _materialize_meta_layers(self, model, layers: List[int]):
        """将 meta device 上的层移回 CPU (to_empty)。"""
        from .utils import get_decoder_layers
        decoder_layers = get_decoder_layers(model)
        for i in layers:
            if i < len(decoder_layers):
                layer = decoder_layers[i]
                try:
                    dev = next(layer.parameters()).device
                    is_meta = dev.type == 'meta'
                except StopIteration:
                    is_meta = False
                if is_meta:
                    decoder_layers[i] = layer.to_empty(device='cpu')

    def _prepare_embed_input(self, input_ids: Tensor, ref_model, quant_model,
                             ref_device: str, quant_device: str, need_embed: bool,
                             ref_hidden_states, quant_hidden_states, all_results: list):
        """准备输入: embed forward (首个 shard) 或使用上一 shard 的 hidden_states。

        Returns:
            (ref_hidden, quant_hidden, need_embed_updated)
        """
        if need_embed:
            ref_input = input_ids.cpu()
            quant_input = input_ids.cpu()

            from .utils import get_embed_module
            ref_embed = get_embed_module(ref_model)
            quant_embed = get_embed_module(quant_model)

            with torch.no_grad():
                ref_emb_out = ref_embed(ref_input)
                quant_emb_out = quant_embed(quant_input)

            # Embedding IS rotated in QuaRot: quant_embed ≈ ref_embed @ R.
            # So quant_emb_out is in rotated space; apply R^T to align with ref.
            # Ref (FP16) is never rotated — no unrotate on ref side.
            from .utils import unrotate_hidden
            ref_emb_for_metrics = ref_emb_out
            quant_emb_for_metrics = unrotate_hidden(quant_emb_out, self.R)
            metrics = compute_all_metrics(ref_emb_for_metrics, quant_emb_for_metrics)
            all_results.append(BlockCompareResult(
                layer_name="embedding",
                metrics=metrics,
            ))

            ref_hidden = ref_emb_out.to(ref_device)
            quant_hidden = quant_emb_out.to(quant_device)
            return ref_hidden, quant_hidden, False
        else:
            ref_hidden = ref_hidden_states.to(ref_device) if ref_hidden_states is not None else None
            quant_hidden = quant_hidden_states.to(quant_device) if quant_hidden_states is not None else None
            return ref_hidden, quant_hidden, need_embed

    @staticmethod
    def _save_hidden_states_for_next_shard(ref_output, quant_output):
        """保存当前 shard 输出的 hidden_states 到 CPU (跨 shard 传递)。"""
        ref_hs = ref_output.detach().clone().cpu() if ref_output is not None else None
        quant_hs = quant_output.detach().clone().cpu() if quant_output is not None else None
        return ref_hs, quant_hs

    def _should_cache_layer(self, cos_sim_val: float) -> bool:
        """判断当前层是否需要缓存 hidden_states (L2 诊断用)。"""
        if self.l1_target_layers is not None:
            return True
        if self.auto_cache_bad_layer:
            is_bad = cos_sim_val < self.bad_layer_threshold
            is_cliff = (hasattr(self, '_prev_cos_sim') and
                        self._prev_cos_sim is not None and
                        self._prev_cos_sim - cos_sim_val > 0.1)
            should = is_bad or is_cliff
            self._prev_cos_sim = cos_sim_val
            return should
        if self.cache_top_k > 0:
            return True
        return False

    def _save_layer_cache(self, ref_hidden: Tensor, quant_hidden: Tensor,
                           layer_idx: int, cos_sim_val: float,
                           layer_cos_sims: dict, layer_inputs: dict,
                           ref_layer_state: Optional[Tensor] = None,
                           quant_layer_state: Optional[Tensor] = None):
        """缓存目标层输入及其可选跨层状态，并记录路径。"""
        ref_input = ref_hidden.detach().clone().cpu()
        quant_input = quant_hidden.detach().clone().cpu()
        from .cache import save_cache as l2_save_cache
        seqlen = ref_input.shape[1]
        p1 = l2_save_cache(self.ref_model_path, self._prompt_text,
                           seqlen, layer_idx, "ref", self.quant_method, ref_input,
                           layer_state=ref_layer_state)
        p2 = l2_save_cache(self.quant_model_path, self._prompt_text,
                           seqlen, layer_idx, "quant", self.quant_method, quant_input,
                           layer_state=quant_layer_state)
        del ref_input, quant_input

        if self.l1_target_layers is not None:
            logger.info(f"  [L1 TARGET CACHE] Layer {layer_idx} input saved (cos_sim={cos_sim_val:.4f})")
        elif self.auto_cache_bad_layer:
            logger.info(f"  [L2 CACHE] Layer {layer_idx} input saved (cos_sim={cos_sim_val:.4f}): "
                        f"{os.path.basename(p1)}, {os.path.basename(p2)}")

        if self.cache_top_k > 0:
            layer_cos_sims[layer_idx] = cos_sim_val
            layer_inputs[layer_idx] = (p1, p2)

    @staticmethod
    def _cache_top_k_cleanup(layer_cos_sims: dict, layer_inputs: dict, cache_top_k: int,
                             verbose: bool = False):
        """按 cos_sim 排序，保留最低 top-k 层的 cache，删除其余临时文件。"""
        if cache_top_k <= 0 or not layer_cos_sims:
            return
        sorted_layers = sorted(layer_cos_sims.items(), key=lambda x: x[1])
        keep_set = set(layer_idx for layer_idx, _ in sorted_layers[:cache_top_k])
        keep_count = 0
        for layer_idx, cs in sorted_layers:
            if layer_idx in layer_inputs:
                p1, p2 = layer_inputs[layer_idx]
                if layer_idx in keep_set:
                    keep_count += 1
                    if verbose:
                        logger.info(f"  [L2 CACHE] Layer {layer_idx}: cos_sim={cs:.6f} -> "
                                    f"{os.path.basename(p1)}, {os.path.basename(p2)}")
                else:
                    for p in [p1, p2]:
                        if p and os.path.exists(p):
                            os.remove(p)
        if verbose and keep_count > 0:
            logger.info(f"  [L2 CACHE] 保留了 {keep_count} 层最低 cos_sim 的 cache")
        layer_inputs.clear()

    def _compute_layer_metrics(self, ref_hidden: Tensor, quant_hidden: Tensor,
                               layer_idx: int) -> Tuple[Dict, 'BlockCompareResult']:
        """unrotate quant only → compute_all_metrics.

        Ref model is NEVER rotated (always FP16/BF16 original space).
        Quant model (W4A8 QuaRot) has rotated layer outputs — apply R^T to
        bring them back to original space for comparison.
        Embedding is NOT rotated (handled separately, no unrotate).
        """
        from .utils import unrotate_hidden
        ref_for_metrics = ref_hidden  # ref is never rotated
        quant_for_metrics = unrotate_hidden(quant_hidden, self.R)
        metrics = compute_all_metrics(ref_for_metrics, quant_for_metrics)
        result = BlockCompareResult(
            layer_name=f"layer.{layer_idx}.block_output",
            metrics=metrics,
        )
        return metrics, result

    def _compute_layer_metrics_and_cache(
        self,
        ref_hidden: Tensor,
        quant_hidden: Tensor,
        layer_idx: int,
        layer_cos_sims: dict,
        layer_inputs: dict,
        ref_layer_input: Optional[Tensor] = None,
        quant_layer_input: Optional[Tensor] = None,
        ref_layer_state: Optional[Tensor] = None,
        quant_layer_state: Optional[Tensor] = None,
    ):
        """单层: compute metrics + append result + 决定是否 cache。"""
        metrics, result = self._compute_layer_metrics(ref_hidden, quant_hidden, layer_idx)

        cs = metrics.get('cos_sim', 1.0)
        should_cache = self._should_cache_layer(cs)
        if should_cache:
            self._save_layer_cache(
                ref_layer_input if ref_layer_input is not None else ref_hidden,
                quant_layer_input if quant_layer_input is not None else quant_hidden,
                layer_idx, cs, layer_cos_sims, layer_inputs,
                ref_layer_state=ref_layer_state,
                quant_layer_state=quant_layer_state,
            )
        return metrics, result

    def _compute_logits_topk(
        self,
        ref_model,
        quant_model,
        ref_hidden_states,
        quant_hidden_states,
        ref_device: str,
        quant_device: str,
        top_k: int = 5,
    ) -> TopKCompareResult:
        """加载 norm+lm_head，跑 logits，比较 top-k token 对齐"""
        from .model_loader import load_layer_weights_indexed
        from .utils import get_norm_module, get_lm_head_module

        # 加载 norm 和 lm_head 权重
        load_layer_weights_indexed(ref_model, self.ref_model_path, [-1], ref_device, self.dtype,
                                   self._ref_weight_map, self._ref_reader,
                                   is_quant=self._ref_is_quant, is_ct=self._ref_is_ct,
                                   quant_desc=self._ref_quant_desc,
                                   verbose=False)
        load_layer_weights_indexed(quant_model, self.quant_model_path, [-1], quant_device, self.dtype,
                                   self._quant_weight_map, self._quant_reader,
                                   is_quant=self._quant_is_quant, is_ct=self._quant_is_ct,
                                   quant_desc=self._quant_quant_desc,
                                   verbose=False)

        # 移到设备
        for model, device in [(ref_model, ref_device), (quant_model, quant_device)]:
            norm_mod = get_norm_module(model)
            head_mod = get_lm_head_module(model)
            if norm_mod is not None:
                norm_mod.to(device)
            if head_mod is not None:
                head_mod.to(device)
        self._restore_tied_lm_head(
            ref_model, self._ref_weight_map, "ref", ref_device)
        self._restore_tied_lm_head(
            quant_model, self._quant_weight_map, "quant", quant_device)

        ref_hidden = ref_hidden_states.to(ref_device)
        quant_hidden = quant_hidden_states.to(quant_device)

        with torch.no_grad():
            # final_norm
            ref_norm_mod = get_norm_module(ref_model)
            quant_norm_mod = get_norm_module(quant_model)
            if ref_norm_mod is not None:
                ref_hidden = ref_norm_mod(ref_hidden)
            if quant_norm_mod is not None:
                quant_hidden = quant_norm_mod(quant_hidden)

            # lm_head
            ref_head = get_lm_head_module(ref_model)
            quant_head = get_lm_head_module(quant_model)
            if ref_head is None or quant_head is None:
                raise RuntimeError(
                    "[L1 logits] output projection is missing; cannot compute token logits"
                )
            ref_logits = ref_head(ref_hidden)
            quant_logits = quant_head(quant_hidden)
            self._validate_logits(ref_logits, "ref")
            self._validate_logits(quant_logits, "quant")

            # logits cos_sim (取最后一个 token)
            last_idx = -1
            logits_cs = cos_sim(ref_logits[:, last_idx], quant_logits[:, last_idx], use_cpu=True)

            # top-k token ids (最后一个 token)
            ref_topk = ref_logits[:, last_idx].topk(top_k).indices.tolist()[0]
            quant_topk = quant_logits[:, last_idx].topk(top_k).indices.tolist()[0]
            match_count = len(set(ref_topk) & set(quant_topk))

            # 熵对比
            import torch.nn.functional as F
            ref_probs = F.softmax(ref_logits[:, last_idx].float(), dim=-1)
            quant_probs = F.softmax(quant_logits[:, last_idx].float(), dim=-1)
            ref_ent = -(ref_probs * ref_probs.clamp(min=1e-10).log()).sum().item()
            quant_ent = -(quant_probs * quant_probs.clamp(min=1e-10).log()).sum().item()

            # KL散度: KL(quant || ref) = sum(quant * log(quant/ref))
            kl_div = (quant_probs * (quant_probs.clamp(min=1e-10).log() - ref_probs.clamp(min=1e-10).log())).sum().item()

        # 卸载 norm/lm_head 回 cpu
        for model in [ref_model, quant_model]:
            norm_mod = get_norm_module(model)
            head_mod = get_lm_head_module(model)
            if norm_mod is not None:
                norm_mod.to('cpu')
            if head_mod is not None:
                head_mod.to('cpu')

        import gc
        gc.collect()
        from .utils import clear_device_cache
        clear_device_cache()

        # Decode tokens if tokenizer is available
        ref_topk_tokens = None
        quant_topk_tokens = None
        if self.tokenizer is not None:
            try:
                ref_topk_tokens = [self.tokenizer.decode([tid]) for tid in ref_topk[:top_k]]
                quant_topk_tokens = [self.tokenizer.decode([tid]) for tid in quant_topk[:top_k]]
            except Exception as e:
                logger.warning(f"Token decoding failed: {e}")

        return TopKCompareResult(
            top_k=top_k,
            ref_topk_ids=ref_topk,
            quant_topk_ids=quant_topk,
            match_count=match_count,
            logits_cos_sim=logits_cs,
            top1_match=(ref_topk[0] == quant_topk[0]),
            ref_entropy=ref_ent,
            quant_entropy=quant_ent,
            entropy_diff=quant_ent - ref_ent,
            kl_divergence=kl_div,
            ref_topk_tokens=ref_topk_tokens,
            quant_topk_tokens=quant_topk_tokens,
        )

    @staticmethod
    def _has_explicit_lm_head(model, head_mod, weight_map) -> bool:
        """Whether the checkpoint stores the resolved output module's weight.

        Match the actual module path instead of any key ending in
        ``output_proj.weight``; the latter can accidentally match a decoder
        layer projection or DSpark's verifier head.
        """
        if not weight_map or head_mod is None:
            return False
        try:
            named_modules = model.named_modules(remove_duplicate=False)
        except TypeError:  # older torch
            named_modules = model.named_modules()
        module_names = [
            f"{name}.weight" if name else "weight"
            for name, module in named_modules if module is head_mod
        ]
        for key in weight_map:
            key = str(key)
            if any(key == name or key.endswith(f".{name}") for name in module_names):
                return True
        # Common top-level CausalLM layout; keep this exact so
        # ``verifier_lm_head.weight`` is not treated as the target head.
        return "lm_head.weight" in {str(key) for key in weight_map}

    def _restore_tied_lm_head(self, model, weight_map, label: str, device) -> bool:
        """Restore an omitted lm_head from this model's embedding table.

        Hugging Face checkpoints may omit ``lm_head.weight`` when it is tied to
        ``embed_tokens.weight``.  The sharded skeleton is materialized with
        ``to_empty()``, so relying on the constructor-time alias leaves an empty
        output head.  Create an explicit copy for the final logits pass instead.
        """
        from .utils import get_embed_module, get_lm_head_module
        embed_mod = get_embed_module(model)
        head_mod = get_lm_head_module(model)
        if not weight_map or self._has_explicit_lm_head(model, head_mod, weight_map):
            return False
        if embed_mod is None or head_mod is None:
            raise RuntimeError(
                f"[L1 logits] {label} checkpoint has no explicit lm_head.weight "
                "and its tied embedding/output modules could not be resolved"
            )
        if not hasattr(embed_mod, "weight") or not hasattr(head_mod, "weight"):
            raise RuntimeError(
                f"[L1 logits] {label} tied embedding/output module has no weight"
            )
        if tuple(embed_mod.weight.shape) != tuple(head_mod.weight.shape):
            raise RuntimeError(
                f"[L1 logits] {label} cannot tie lm_head to embedding: "
                f"embed={tuple(embed_mod.weight.shape)}, head={tuple(head_mod.weight.shape)}"
            )
        if embed_mod.weight.is_meta:
            raise RuntimeError(
                f"[L1 logits] {label} embedding is meta; cannot restore tied lm_head"
            )

        copied = embed_mod.weight.detach().to(device=device, dtype=self.dtype).clone()
        head_mod.weight = nn.Parameter(copied, requires_grad=False)
        logger.info(
            "[L1 logits] %s checkpoint omits lm_head.weight; using tied "
            "embed_tokens.weight (%s)", label, tuple(copied.shape)
        )
        return True

    @staticmethod
    def _validated_lm_head_weight(model, label: str) -> Tensor:
        """Return CPU fp32 lm_head weight, rejecting empty/meta placeholders."""
        from .utils import get_lm_head_module
        head_mod = get_lm_head_module(model)
        if head_mod is None or not hasattr(head_mod, "weight"):
            raise RuntimeError(f"[L1 logits] {label} lm_head module is missing")
        weight = head_mod.weight
        if weight.is_meta:
            raise RuntimeError(f"[L1 logits] {label} lm_head.weight is still meta")
        flat = weight.detach().reshape(-1)
        step = max(1, flat.numel() // 4096)
        sample = flat[::step][:4096].float().cpu()
        if sample.numel() == 0 or not torch.isfinite(sample).all():
            raise RuntimeError(f"[L1 logits] {label} lm_head.weight is empty or non-finite")
        if sample.abs().max().item() == 0:
            raise RuntimeError(
                f"[L1 logits] {label} lm_head.weight is all-zero; refusing invalid logits"
            )
        return weight.detach().cpu().float()

    @staticmethod
    def _apply_real_final_norm(hidden_states: Tensor, norm_mod, dtype) -> Tensor:
        """Run the model's real final norm (RMSNorm/LayerNorm), not a substitute."""
        hidden = hidden_states.detach().cpu().to(dtype)
        if norm_mod is not None:
            hidden = norm_mod(hidden)
        return hidden.float()

    @staticmethod
    def _validate_logits(logits: Tensor, label: str) -> None:
        """Reject non-finite or constant logits instead of reporting cos_sim=0."""
        values = logits.detach().float()
        if not torch.isfinite(values).all():
            raise RuntimeError(f"[L1 logits] {label} logits contain NaN/Inf")
        if values.shape[-1] > 1:
            spread = values.amax(dim=-1) - values.amin(dim=-1)
            if spread.max().item() <= 1e-12:
                raise RuntimeError(
                    f"[L1 logits] {label} logits are constant (spread=0); "
                    "lm_head was not loaded correctly"
                )

    def _load_norm_and_head_to_cpu(self, ref_model, quant_model):
        """Load norm and lm_head modules to CPU."""
        from .model_loader import load_layer_weights_indexed
        from .utils import get_norm_module, get_lm_head_module

        load_layer_weights_indexed(ref_model, self.ref_model_path, [-1], 'cpu', self.dtype,
                                   self._ref_weight_map, self._ref_reader,
                                   is_quant=self._ref_is_quant, is_ct=self._ref_is_ct,
                                   quant_desc=self._ref_quant_desc,
                                   verbose=False)
        load_layer_weights_indexed(quant_model, self.quant_model_path, [-1], 'cpu', self.dtype,
                                   self._quant_weight_map, self._quant_reader,
                                   is_quant=self._quant_is_quant, is_ct=self._quant_is_ct,
                                   quant_desc=self._quant_quant_desc,
                                   verbose=False)

        for model in [ref_model, quant_model]:
            norm_mod = get_norm_module(model)
            head_mod = get_lm_head_module(model)
            if norm_mod is not None:
                norm_mod.to('cpu')
            if head_mod is not None:
                head_mod.to('cpu')
        self._restore_tied_lm_head(ref_model, self._ref_weight_map, "ref", 'cpu')
        self._restore_tied_lm_head(quant_model, self._quant_weight_map, "quant", 'cpu')

    def _unload_norm_and_head(self, ref_model, quant_model):
        """Unload norm and lm_head modules back to meta device."""
        from .utils import get_norm_module, get_lm_head_module
        for model in [ref_model, quant_model]:
            norm_mod = get_norm_module(model)
            head_mod = get_lm_head_module(model)
            if norm_mod is not None:
                norm_mod.to_empty(device='meta')
            if head_mod is not None:
                head_mod.to_empty(device='meta')

    def _compute_logits_topk_cpu(
        self,
        ref_model,
        quant_model,
        ref_hidden_states,
        quant_hidden_states,
        top_k: int = 5,
    ) -> TopKCompareResult:
        """在 CPU 上加载 norm+lm_head 权重并计算 logits，绕过 NPU 状态异常。"""
        from .utils import get_norm_module, get_lm_head_module

        self._load_norm_and_head_to_cpu(ref_model, quant_model)

        ref_norm_mod = get_norm_module(ref_model)
        quant_norm_mod = get_norm_module(quant_model)
        ref_head_w = self._validated_lm_head_weight(ref_model, "ref")
        quant_head_w = self._validated_lm_head_weight(quant_model, "quant")

        with torch.no_grad():
            ref_h = self._apply_real_final_norm(
                ref_hidden_states, ref_norm_mod, self.dtype)
            quant_h = self._apply_real_final_norm(
                quant_hidden_states, quant_norm_mod, self.dtype)

            # lm_head: 取 last token
            ref_last = ref_h[:, -1]
            quant_last = quant_h[:, -1]
            ref_logits = torch.nn.functional.linear(ref_last, ref_head_w)
            quant_logits = torch.nn.functional.linear(quant_last, quant_head_w)
            self._validate_logits(ref_logits, "ref")
            self._validate_logits(quant_logits, "quant")

            logits_cs = cos_sim(ref_logits, quant_logits, use_cpu=True)

            ref_topk = ref_logits.topk(top_k).indices.tolist()[0]
            quant_topk = quant_logits.topk(top_k).indices.tolist()[0]
            match_count = len(set(ref_topk) & set(quant_topk))

            import torch.nn.functional as F
            ref_probs = F.softmax(ref_logits.float(), dim=-1)
            quant_probs = F.softmax(quant_logits.float(), dim=-1)
            ref_ent = -(ref_probs * ref_probs.clamp(min=1e-10).log()).sum().item()
            quant_ent = -(quant_probs * quant_probs.clamp(min=1e-10).log()).sum().item()
            kl_div = (quant_probs * (quant_probs.clamp(min=1e-10).log() - ref_probs.clamp(min=1e-10).log())).sum().item()

        self._unload_norm_and_head(ref_model, quant_model)

        import gc; gc.collect()
        from .utils import clear_device_cache
        clear_device_cache()

        # Decode token IDs to actual tokens if tokenizer is available
        ref_tokens_decoded = None
        quant_tokens_decoded = None
        if self.tokenizer:
            try:
                ref_tokens_decoded = [self.tokenizer.decode([tid]) for tid in ref_topk]
                quant_tokens_decoded = [self.tokenizer.decode([tid]) for tid in quant_topk]
            except Exception as e:
                logger.warning(f"Failed to decode tokens: {e}")

        return TopKCompareResult(
            top_k=top_k,
            ref_topk_ids=ref_topk,
            quant_topk_ids=quant_topk,
            match_count=match_count,
            logits_cos_sim=logits_cs,
            top1_match=(ref_topk[0] == quant_topk[0]),
            ref_entropy=ref_ent,
            quant_entropy=quant_ent,
            entropy_diff=quant_ent - ref_ent,
            kl_divergence=kl_div,
            ref_topk_tokens=ref_tokens_decoded,
            quant_topk_tokens=quant_tokens_decoded,
        )

    def _materialize_norm_lm_head(self, ref_model, quant_model):
        """materialize norm+lm_head 到 CPU + 加载权重。"""
        from .model_loader import load_layer_weights_indexed
        from .utils import get_norm_module, get_lm_head_module
        for model in [ref_model, quant_model]:
            norm_mod = get_norm_module(model)
            head_mod = get_lm_head_module(model)
            for mod in [norm_mod, head_mod]:
                if mod is not None:
                    try:
                        p = next(mod.parameters())
                        if p.device.type == 'meta':
                            mod.to_empty(device='cpu')
                    except StopIteration:
                        pass
        load_layer_weights_indexed(ref_model, self.ref_model_path, [-1], 'cpu', self.dtype,
                                   self._ref_weight_map, self._ref_reader,
                                   is_quant=self._ref_is_quant, is_ct=self._ref_is_ct,
                                   quant_desc=self._ref_quant_desc,
                                   verbose=False)
        load_layer_weights_indexed(quant_model, self.quant_model_path, [-1], 'cpu', self.dtype,
                                   self._quant_weight_map, self._quant_reader,
                                   is_quant=self._quant_is_quant, is_ct=self._quant_is_ct,
                                   quant_desc=self._quant_quant_desc,
                                   verbose=False)
        for model in [ref_model, quant_model]:
            norm_mod = get_norm_module(model)
            head_mod = get_lm_head_module(model)
            if norm_mod is not None:
                norm_mod.to('cpu')
            if head_mod is not None:
                head_mod.to('cpu')
        self._restore_tied_lm_head(ref_model, self._ref_weight_map, "ref", 'cpu')
        self._restore_tied_lm_head(quant_model, self._quant_weight_map, "quant", 'cpu')

    def _get_norm_modules_and_lm_head_weights(self, ref_model, quant_model):
        """提取真实 final norm 模块和已校验的 lm_head 权重。"""
        from .utils import get_norm_module
        ref_norm_mod = get_norm_module(ref_model)
        quant_norm_mod = get_norm_module(quant_model)
        ref_head_w = self._validated_lm_head_weight(ref_model, "ref")
        quant_head_w = self._validated_lm_head_weight(quant_model, "quant")
        return ref_norm_mod, quant_norm_mod, ref_head_w, quant_head_w

    def _compute_logits_np(self, ref_hidden_states, quant_hidden_states,
                           ref_norm_mod, quant_norm_mod,
                           ref_head_w, quant_head_w, max_positions):
        """norm → slice last-N → lm_head → [N, vocab] fp32 CPU。"""
        with torch.no_grad():
            ref_h = self._apply_real_final_norm(
                ref_hidden_states, ref_norm_mod, self.dtype)
            quant_h = self._apply_real_final_norm(
                quant_hidden_states, quant_norm_mod, self.dtype)
            seq_len = ref_h.shape[1]
            N = min(max_positions, seq_len)
            ref_h_last = ref_h[:, -N:, :].contiguous()
            quant_h_last = quant_h[:, -N:, :].contiguous()
            del ref_h, quant_h
            ref_logits_full = torch.nn.functional.linear(ref_h_last, ref_head_w)
            quant_logits_full = torch.nn.functional.linear(quant_h_last, quant_head_w)
            self._validate_logits(ref_logits_full, "ref")
            self._validate_logits(quant_logits_full, "quant")
            ref_logits_np = ref_logits_full.squeeze(0).to(torch.float32).contiguous()
            quant_logits_np = quant_logits_full.squeeze(0).to(torch.float32).contiguous()
            del ref_h_last, quant_h_last, ref_logits_full, quant_logits_full
        if ref_logits_np.shape[0] != quant_logits_np.shape[0]:
            n = min(ref_logits_np.shape[0], quant_logits_np.shape[0])
            ref_logits_np = ref_logits_np[:n]
            quant_logits_np = quant_logits_np[:n]
            N = n
        return ref_logits_np, quant_logits_np, N

    @staticmethod
    def _unload_norm_lm_head(ref_model, quant_model):
        """卸载 norm+lm_head 回 meta, 释放 CPU 内存。"""
        from .utils import get_norm_module, get_lm_head_module
        for model in [ref_model, quant_model]:
            norm_mod = get_norm_module(model)
            head_mod = get_lm_head_module(model)
            if norm_mod is not None:
                norm_mod.to_empty(device='meta')
            if head_mod is not None:
                head_mod.to_empty(device='meta')
        import gc; gc.collect()
        from .utils import clear_device_cache
        clear_device_cache()

    def _collect_full_logits(
        self,
        ref_model,
        quant_model,
        ref_hidden_states,
        quant_hidden_states,
        max_positions: Optional[int] = None,
        top_k: int = 10,
    ) -> Optional[LogitsData]:
        """采集 last-N 位置的全词表 logits, 跑 compare_logits 出 4 类可视化数据。"""
        if self.tokenizer is None:
            if self.verbose:
                logger.warning("[L1 logits] 跳过 full logits 采集: 未配置 tokenizer")
            return None
        if not self.collect_full_logits:
            return None
        if max_positions is None:
            max_positions = self.logits_max_positions

        from .logits_compare import LogitsCollection, compare_logits

        self._materialize_norm_lm_head(ref_model, quant_model)
        ref_norm_mod, quant_norm_mod, ref_head_w, quant_head_w = (
            self._get_norm_modules_and_lm_head_weights(ref_model, quant_model)
        )

        try:
            ref_logits_np, quant_logits_np, N = self._compute_logits_np(
                ref_hidden_states, quant_hidden_states,
                ref_norm_mod, quant_norm_mod,
                ref_head_w, quant_head_w, max_positions)
            ref_c = LogitsCollection(token_positions=list(range(N)), logits=ref_logits_np)
            quant_c = LogitsCollection(token_positions=list(range(N)), logits=quant_logits_np)
            if self.verbose:
                logger.info(f"[L1 logits] full logits 采集完成: {N} positions × "
                            f"{ref_logits_np.shape[1]} vocab → compare_logits")
            comparison = compare_logits(ref_c, quant_c, self.tokenizer, top_k=top_k)
            return comparison.to_logits_data()
        except Exception as e:
            logger.warning(f"[L1 logits] full logits 采集失败: {e}")
            return None
        finally:
            self._unload_norm_lm_head(ref_model, quant_model)

    def _compare_dual_sharded(
        self,
        input_ids: Tensor,
        layers_per_shard: int = 8,
        **forward_kwargs,
    ) -> 'BlockCompareReport':
        """双设备分片：ref在ref_device，quant在quant_device"""
        if self.verbose:
            logger.info(f"[Sharded L1] 开始双设备分片...")
            logger.info(f"[Sharded L1] ref_device: {self.ref_device}, quant_device: {self.quant_device}")
            logger.info(f"[Sharded L1] 每批层数: {layers_per_shard}")
            if self.activation_quant:
                logger.info(f"  activation_quant: {self.activation_quant_type} 激活伪量化已启用 (ref+quant 双侧)")
            logger.info(f"[Sharded L1] 输入长度: {input_ids.shape[1]} tokens")

        # 清理 NPU 缓存
        from .utils import clear_device_cache
        clear_device_cache()

        forward_kwargs.setdefault("use_cache", False)

        shard_plans = self._plan_shards(layers_per_shard)
        all_results = []

        ref_device = self.actual_ref_device
        quant_device = self.actual_quant_device

        from .model_loader import (load_layer_weights_indexed,
                                   move_layers_to_device, unload_layers_to_meta)

        # 创建骨架 + 加载 embed + 初始化 rotary_emb
        ref_model, quant_model = self._init_dual_skeleton_and_embed(ref_device, quant_device)
        from .utils import get_decoder_layers
        if any(
            is_kimi_k3_layer(layer) and has_indexed_routed_experts(layer)
            for layer in get_decoder_layers(ref_model)
        ):
            raise RuntimeError(
                "Kimi K3 uses one module per routed expert; L1 dual mode would "
                "materialize all experts. Re-run with --compare_mode grouped_dual."
            )

        # weight_map/reader 已在 _init_dual_skeleton_and_embed 中缓存到 self
        ref_reader = self._ref_reader
        quant_reader = self._quant_reader
        ref_weight_map = self._ref_weight_map
        quant_weight_map = self._quant_weight_map
        ref_is_quant = self._ref_is_quant
        ref_is_ct = self._ref_is_ct
        quant_is_quant = self._quant_is_quant
        quant_is_ct = self._quant_is_ct
        ref_quant_desc = self._ref_quant_desc
        quant_quant_desc = self._quant_quant_desc

        ref_hidden_states = None
        quant_hidden_states = None
        need_embed = True

        # GLM MoE DSA: topk_indices persists across shards
        ref_prev_topk = None
        quant_prev_topk = None

        # cache_top_k: 收集每层 cos_sim + input hidden_states，跑完后缓存最低 N 层
        layer_cos_sims = {}
        layer_inputs = {}

        for shard_idx, (layer_start, layer_end) in enumerate(shard_plans):
            layers = list(range(layer_start, layer_end))

            if self.verbose:
                logger.info(f"\n[Sharded L1] 处理 shard {shard_idx+1}/{len(shard_plans)}: layers {layer_start}-{layer_end-1}")
                logger.info(f"  [RSS] before load: {self._get_rss_gb():.1f}GB")

            # meta→CPU materialize
            for model in [ref_model, quant_model]:
                self._materialize_meta_layers(model, layers)

            shard_succeeded = False
            try:
                load_layer_weights_indexed(ref_model, self.ref_model_path, layers, ref_device, self.dtype,
                                           ref_weight_map, ref_reader,
                                           is_quant=ref_is_quant, is_ct=ref_is_ct,
                                           quant_desc=ref_quant_desc,
                                           verbose=False)
                load_layer_weights_indexed(quant_model, self.quant_model_path, layers, quant_device, self.dtype,
                                           quant_weight_map, quant_reader,
                                           is_quant=quant_is_quant, is_ct=quant_is_ct,
                                           quant_desc=quant_quant_desc,
                                           verbose=False)

                # 移动层到设备
                # dual 模式需要完整前向, 3D expert 参数也必须上 NPU
                # grouped_dual 模式 expert 按需读取, 跳过 3D 参数
                skip_3d = self.compare_mode == 'grouped_dual'
                move_layers_to_device(ref_model, layers, ref_device, clear_others=False, skip_3d_experts=skip_3d)
                move_layers_to_device(quant_model, layers, quant_device, clear_others=False, skip_3d_experts=skip_3d)

                # 注册 MXFP8 激活伪量化 hook (dense 层)
                if self.activation_quant:
                    self._register_activation_quant_hooks(ref_model, layers)
                    self._register_activation_quant_hooks(quant_model, layers)

                if self.verbose:
                    logger.info(f"  [RSS] after load: {self._get_rss_gb():.1f}GB")
                    if hasattr(torch, 'npu') and torch.npu.is_available():
                        ref_mem = torch.npu.memory_allocated(ref_device) / 1024**3
                        quant_mem = torch.npu.memory_allocated(quant_device) / 1024**3
                        logger.info(f"  [NPU MEM] ref: {ref_mem:.1f}GB, quant: {quant_mem:.1f}GB")

                # 准备输入 hidden_states
                ref_hidden, quant_hidden, need_embed = self._prepare_embed_input(
                    input_ids, ref_model, quant_model, ref_device, quant_device,
                    need_embed, ref_hidden_states, quant_hidden_states, all_results)

                # 执行 transformer 层
                ref_output = ref_hidden
                quant_output = quant_hidden
                if ref_hidden is not None and quant_hidden is not None:
                    ref_model.eval()
                    quant_model.eval()
                    with torch.no_grad():
                        ref_pos_ids, quant_pos_ids, ref_pe, quant_pe = self._setup_dual_position_embeddings(
                            ref_hidden, quant_hidden, ref_model, quant_model)
                        ref_hidden, quant_hidden, ref_prev_topk, quant_prev_topk = self._forward_dual_layers(
                            ref_hidden, quant_hidden, ref_model, quant_model,
                            layer_start, layer_end, ref_pos_ids, quant_pos_ids,
                            ref_pe, quant_pe, ref_prev_topk, quant_prev_topk,
                            shard_idx, layer_cos_sims, layer_inputs, all_results)
                    ref_output = ref_hidden
                    quant_output = quant_hidden

                # 保存下一 shard 的 hidden_states
                ref_hidden_states, quant_hidden_states = self._save_hidden_states_for_next_shard(ref_output, quant_output)

                if self.verbose:
                    logger.info(f"  [RSS] after forward: {self._get_rss_gb():.1f}GB")
                    if hasattr(torch, 'npu') and torch.npu.is_available():
                        ref_mem = torch.npu.memory_allocated(ref_device) / 1024**3
                        quant_mem = torch.npu.memory_allocated(quant_device) / 1024**3
                        logger.info(f"  [NPU MEM] ref: {ref_mem:.1f}GB, quant: {quant_mem:.1f}GB")
                shard_succeeded = True
            finally:
                # 清理激活伪量化 hook (在 unload 前移除，避免 hook 引用已释放的模块)
                self._clear_activation_quant_hooks()

                # A failed asynchronous NPU kernel poisons the device context.
                # Synchronizing in empty_cache then masks the original error
                # with 507014.  The process is terminating, so skip device
                # cleanup and preserve the actionable root exception.
                if shard_succeeded:
                    unload_layers_to_meta(ref_model, layers)
                    unload_layers_to_meta(quant_model, layers)

                    from .utils import clear_device_cache
                    clear_device_cache()

            if self.verbose:
                logger.info(f"  [RSS] after unload+gc: {self._get_rss_gb():.1f}GB")
                if hasattr(torch, 'npu') and torch.npu.is_available():
                    ref_mem = torch.npu.memory_allocated(ref_device) / 1024**3
                    quant_mem = torch.npu.memory_allocated(quant_device) / 1024**3
                    logger.info(f"  [NPU MEM] ref: {ref_mem:.1f}GB, quant: {quant_mem:.1f}GB")

        # 收尾: close readers → cache cleanup → topk → logits → report
        report = self._finalize_dual_report(
            ref_model, quant_model, ref_hidden_states, quant_hidden_states,
            all_results, ref_reader, quant_reader, layer_cos_sims, layer_inputs,
            use_cpu_topk=False)
        if self.verbose:
            logger.info(f"\n[Sharded L1] 完成，共 {len(all_results)} 个结果")
        return report

    # ------------------------------------------------------------------ #
    #  Grouped Dual: MoE expert chunk 跨卡分发
    # ------------------------------------------------------------------ #

    def _compare_grouped_dual(
        self,
        input_ids: Tensor,
        layers_per_shard: int = 8,
        **forward_kwargs,
    ) -> 'BlockCompareReport':
        """Grouped dual: MoE expert chunk 跨卡分发。

        ref 侧使用 self.ref_devices 多卡分摊 expert chunk，
        quant 侧使用 self.quant_devices 多卡分摊 expert chunk。

        非 MoE 层正常 forward (和 _compare_dual_sharded 一样)，
        MoE 层路由到 _moe_forward_chunked 做 expert chunk 跨卡计算。
        """
        if self.verbose:
            logger.info(f"[Sharded L1] 开始 grouped dual 模式 (streaming expert)...")
            logger.info(f"  ref devices:   {self.ref_devices}")
            logger.info(f"  quant devices: {self.quant_devices}")
            logger.info(f"  expert chunk size: {self.expert_chunk_size}")
            logger.info(f"  每批层数: {layers_per_shard}")
            logger.info(f"  输入长度: {input_ids.shape[1]} tokens")
            logger.info(f"  streaming: routed experts 从 safetensors 实时读取，不在 CPU 预构造 3D tensor")
            if self.activation_quant:
                logger.info(f"  activation_quant: {self.activation_quant_type} 激活伪量化已启用 (ref+quant 双侧)")

        forward_kwargs.setdefault("use_cache", False)

        # grouped_dual 需要 expert_chunk_size; None 时按卡数自动设默认
        if self.expert_chunk_size is None:
            n_dev = max(len(self.ref_devices), len(self.quant_devices))
            self.expert_chunk_size = max(1, 256 // n_dev)  # 256 experts / n_dev
            if self.verbose:
                logger.info(f"  [grouped_dual] expert_chunk_size auto-set to {self.expert_chunk_size} (256/{n_dev})")

        from .utils import clear_device_cache
        clear_device_cache(set(self.ref_devices + self.quant_devices))

        shard_plans = self._plan_shards(layers_per_shard)
        all_results = []

        ref_device = self.ref_devices[0]
        quant_device = self.quant_devices[0]

        from .model_loader import (load_layer_weights_indexed,
                                   move_layers_to_device, unload_layers_to_meta)

        ref_model, quant_model = self._init_dual_skeleton_and_embed(ref_device, quant_device)

        # weight_map/reader 缓存
        ref_reader = self._ref_reader
        quant_reader = self._quant_reader
        ref_weight_map = self._ref_weight_map
        quant_weight_map = self._quant_weight_map
        ref_is_quant = self._ref_is_quant
        ref_is_ct = self._ref_is_ct
        quant_is_quant = self._quant_is_quant
        quant_is_ct = self._quant_is_ct
        ref_quant_desc = self._ref_quant_desc
        quant_quant_desc = self._quant_quant_desc

        ref_hidden_states = None
        quant_hidden_states = None
        need_embed = True

        # GLM MoE DSA: topk_indices persists across shards
        ref_prev_topk = None
        quant_prev_topk = None

        layer_cos_sims = {}
        layer_inputs = {}

        for shard_idx, (layer_start, layer_end) in enumerate(shard_plans):
            layers = list(range(layer_start, layer_end))

            if self.verbose:
                logger.info(f"\n[Sharded L1] 处理 shard {shard_idx+1}/{len(shard_plans)}: layers {layer_start}-{layer_end-1}")
                logger.info(f"  [RSS] before load: {self._get_rss_gb():.1f}GB")

            if shard_idx > 0:
                from .utils import clear_device_cache
                clear_device_cache(set(self.ref_devices + self.quant_devices))

            # materialize meta layers + 替换 3D routed expert params 为 placeholder
            self._materialize_and_replace_experts(ref_model, quant_model, layers)

            shard_succeeded = False
            try:
                # streaming 模式跳过 routed experts
                load_layer_weights_indexed(ref_model, self.ref_model_path, layers, ref_device, self.dtype,
                                           ref_weight_map, ref_reader,
                                           is_quant=ref_is_quant, is_ct=ref_is_ct,
                                           quant_desc=ref_quant_desc,
                                           skip_routed_experts=True,
                                           verbose=False)
                load_layer_weights_indexed(quant_model, self.quant_model_path, layers, quant_device, self.dtype,
                                           quant_weight_map, quant_reader,
                                           is_quant=quant_is_quant, is_ct=quant_is_ct,
                                           quant_desc=quant_quant_desc,
                                           skip_routed_experts=True,
                                           verbose=False)

                move_layers_to_device(ref_model, layers, ref_device, clear_others=False)
                move_layers_to_device(quant_model, layers, quant_device, clear_others=False)

                if self.activation_quant:
                    self._register_activation_quant_hooks(ref_model, layers)
                    self._register_activation_quant_hooks(quant_model, layers)

                if self.verbose:
                    logger.info(f"  [RSS] after load: {self._get_rss_gb():.1f}GB")

                ref_hidden, quant_hidden, need_embed = self._prepare_embed_input(
                    input_ids, ref_model, quant_model, ref_device, quant_device,
                    need_embed, ref_hidden_states, quant_hidden_states, all_results)

                ref_output = ref_hidden
                quant_output = quant_hidden
                if ref_hidden is not None and quant_hidden is not None:
                    from .utils import get_decoder_layers, get_rotary_emb_module
                    ref_model.eval()
                    quant_model.eval()
                    with torch.no_grad():
                        ref_pos_ids, quant_pos_ids, ref_pe, quant_pe = self._setup_dual_position_embeddings(
                            ref_hidden, quant_hidden, ref_model, quant_model)
                        ref_hidden, quant_hidden, ref_prev_topk, quant_prev_topk = self._forward_grouped_dual_layers(
                            ref_hidden, quant_hidden, ref_model, quant_model,
                            layer_start, layer_end, ref_pos_ids, quant_pos_ids,
                            ref_pe, quant_pe, ref_prev_topk, quant_prev_topk,
                            shard_idx, layer_cos_sims, layer_inputs, all_results,
                            ref_reader, quant_reader,
                            ref_quant_desc, quant_quant_desc,
                            ref_is_quant, quant_is_quant,
                            ref_is_ct, quant_is_ct)
                    ref_output = ref_hidden
                    quant_output = quant_hidden

                ref_hidden_states, quant_hidden_states = self._save_hidden_states_for_next_shard(ref_output, quant_output)

                if self.verbose:
                    logger.info(f"  [RSS] after forward: {self._get_rss_gb():.1f}GB")
                shard_succeeded = True
            finally:
                self._clear_activation_quant_hooks()

                if shard_succeeded:
                    unload_layers_to_meta(ref_model, layers)
                    unload_layers_to_meta(quant_model, layers)

                    from .utils import clear_device_cache
                    clear_device_cache(set(self.ref_devices + self.quant_devices))

            if self.verbose:
                logger.info(f"  [RSS] after unload+gc: {self._get_rss_gb():.1f}GB")

        report = self._finalize_dual_report(
            ref_model, quant_model, ref_hidden_states, quant_hidden_states,
            all_results, ref_reader, quant_reader, layer_cos_sims, layer_inputs,
            use_cpu_topk=True)
        if self.verbose:
            logger.info(f"\n[Sharded L1] grouped dual 完成，共 {len(all_results)} 个结果")
        return report

    # ---- grouped_dual 特有辅助方法 ----

    def _materialize_and_replace_experts(self, ref_model, quant_model, layers):
        """materialize meta layers + 替换 3D routed expert params 为 placeholder。"""
        from .utils import get_decoder_layers
        for model in [ref_model, quant_model]:
            decoder_layers = get_decoder_layers(model)
            for i in layers:
                if i >= len(decoder_layers):
                    continue
                layer = decoder_layers[i]
                _, mlp_mod = get_moe_module(layer)
                experts_mod = getattr(mlp_mod, 'experts', None) if mlp_mod else None

                # Replace routed expert storage while it is still on meta.  Doing
                # this after to_empty would allocate the full Kimi/GLM expert set
                # on CPU before immediately discarding it.
                if isinstance(experts_mod, nn.ModuleList):
                    for expert_idx in range(len(experts_mod)):
                        experts_mod[expert_idx] = nn.Identity()
                elif experts_mod is not None:
                    for attr_name in ['gate_up_proj', 'down_proj']:
                        p = getattr(experts_mod, attr_name, None)
                        if p is not None and p.dim() == 3 and p.shape[0] > 1:
                            orig_shape = tuple(p.shape)
                            placeholder = torch.nn.Parameter(
                                torch.zeros(1, 1, 1, dtype=self.dtype),
                                requires_grad=False)
                            placeholder._orig_shape = orig_shape
                            setattr(experts_mod, attr_name, placeholder)

                try:
                    dev = next(layer.parameters()).device
                    is_meta = dev.type == 'meta'
                except StopIteration:
                    is_meta = False
                if is_meta:
                    decoder_layers[i] = layer.to_empty(device='cpu')

    def _make_chunked_mlp_forward(self, orig_fwd, layer, devices, chunk_size,
                                    lidx, sf_rdr, q_desc, is_q, is_ct):
        """创建 chunked MLP forward 闭包。"""
        _self = self
        def _chunked_mlp_forward(hidden_states, *args, **kwargs):
            return _self._moe_forward_chunked(
                layer, hidden_states, devices, chunk_size,
                layer_idx=lidx, sf_reader=sf_rdr,
                quant_desc_str=q_desc, is_quant=is_q, is_ct=is_ct)
        return _chunked_mlp_forward

    def _patch_moe_forward_for_layer(self, ref_layer, quant_layer, layer_idx,
                                       ref_reader, quant_reader,
                                       ref_quant_desc, quant_quant_desc,
                                       ref_is_quant, quant_is_quant,
                                       ref_is_ct, quant_is_ct):
        """Patch the routed MoE block and return modules/original forwards."""
        _, ref_moe = get_moe_module(ref_layer)
        _, quant_moe = get_moe_module(quant_layer)
        if ref_moe is None or quant_moe is None:
            raise RuntimeError(f"Layer {layer_idx} was classified as MoE but has no MoE module")
        ref_orig_mlp_fwd = ref_moe.forward
        quant_orig_mlp_fwd = quant_moe.forward
        _ref_qds = {k: v for k, v in ref_quant_desc.items() if isinstance(v, str)} if ref_quant_desc else None
        _quant_qds = {k: v for k, v in quant_quant_desc.items() if isinstance(v, str)} if quant_quant_desc else None
        ref_moe.forward = self._make_chunked_mlp_forward(
            ref_orig_mlp_fwd, ref_layer, self.ref_devices,
            self.expert_chunk_size, layer_idx, ref_reader, _ref_qds,
            ref_is_quant, ref_is_ct)
        quant_moe.forward = self._make_chunked_mlp_forward(
            quant_orig_mlp_fwd, quant_layer, self.quant_devices,
            self.expert_chunk_size, layer_idx, quant_reader, _quant_qds,
            quant_is_quant, quant_is_ct)
        return ref_moe, ref_orig_mlp_fwd, quant_moe, quant_orig_mlp_fwd

    def _log_norm_debug(self, layer_idx, ref_hidden, quant_hidden, stage):
        """Track hidden-state explosion at DEBUG level only."""
        if (
            layer_idx not in (0, 1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 76, 77)
            or not logger.isEnabledFor(logging.DEBUG)
        ):
            return
        ref_n = ref_hidden.float().norm().item()
        quant_n = quant_hidden.float().norm().item()
        ref_max = ref_hidden.float().abs().max().item()
        quant_max = quant_hidden.float().abs().max().item()
        logger.debug(f"  [NORM L{layer_idx}] {stage}: ref_norm={ref_n:.4f} quant_norm={quant_n:.4f} ref_absmax={ref_max:.6f} quant_absmax={quant_max:.6f}")
        if stage == "IN":
            logger.debug(f"  [NORM L{layer_idx}] IN: same_ptr={ref_hidden.data_ptr() == quant_hidden.data_ptr()}")

    def _log_layer77_debug(self, layer_idx, ref_layer, quant_layer, ref_hidden, quant_hidden):
        """layer 77 weight + hidden state check。"""
        if layer_idx != 77 or not logger.isEnabledFor(logging.DEBUG):
            return
        try:
            ref_o = dict(ref_layer.named_parameters()).get('self_attn.o_proj.weight')
            quant_o = dict(quant_layer.named_parameters()).get('self_attn.o_proj.weight')
            if ref_o is not None and quant_o is not None:
                logger.debug(f"  [DEBUG L77] ref o_proj: min={ref_o.float().min().item():.4f}, max={ref_o.float().max().item():.4f}")
                logger.debug(f"  [DEBUG L77] quant o_proj: min={quant_o.float().min().item():.4f}, max={quant_o.float().max().item():.4f}")
                logger.debug(f"  [DEBUG L77] o_proj same storage? {ref_o.data_ptr() == quant_o.data_ptr()}")
            logger.debug(f"  [DEBUG L77] ref_hidden: shape={ref_hidden.shape}, min={ref_hidden.float().min().item():.6f}, max={ref_hidden.float().max().item():.6f}")
            logger.debug(f"  [DEBUG L77] quant_hidden: shape={quant_hidden.shape}, min={quant_hidden.float().min().item():.6f}, max={quant_hidden.float().max().item():.6f}")
            logger.debug(f"  [DEBUG L77] hidden same storage? {ref_hidden.data_ptr() == quant_hidden.data_ptr()}")
            diff = (ref_hidden.float().cpu() - quant_hidden.float().cpu()).abs()
            logger.debug(f"  [DEBUG L77] hidden diff: mean={diff.mean().item():.8f}, max={diff.max().item():.8f}")
        except Exception as e:
            logger.debug(f"  [DEBUG L77] error: {e}")

    def _forward_grouped_dual_layers(self, ref_hidden, quant_hidden, ref_model, quant_model,
                                       layer_start, layer_end, ref_pos_ids, quant_pos_ids,
                                       ref_pe, quant_pe, ref_prev_topk, quant_prev_topk,
                                       shard_idx, layer_cos_sims, layer_inputs, all_results,
                                       ref_reader, quant_reader,
                                       ref_quant_desc, quant_quant_desc,
                                       ref_is_quant, quant_is_quant,
                                       ref_is_ct, quant_is_ct):
        """grouped_dual 模式逐层 forward (含 MoE patch) + 对比 + 缓存。"""
        from .utils import get_decoder_layers
        ref_layers = get_decoder_layers(ref_model)
        quant_layers = get_decoder_layers(quant_model)
        for layer_idx in range(layer_start, layer_end):
            ref_layer = ref_layers[layer_idx]
            quant_layer = quant_layers[layer_idx]
            ref_layer_input = ref_hidden
            quant_layer_input = quant_hidden
            ref_state_input = ref_prev_topk
            quant_state_input = quant_prev_topk
            is_moe = self._is_moe_layer(ref_layer)
            if layer_idx == 3 and logger.isEnabledFor(logging.DEBUG):
                self._debug_layer3_expert0 = True
                self._debug_layer3_out = True
            self._log_norm_debug(layer_idx, ref_hidden, quant_hidden, "IN")

            ref_moe = quant_moe = None
            ref_orig_mlp_fwd = quant_orig_mlp_fwd = None
            if is_moe:
                ref_moe, ref_orig_mlp_fwd, quant_moe, quant_orig_mlp_fwd = self._patch_moe_forward_for_layer(
                    ref_layer, quant_layer, layer_idx, ref_reader, quant_reader,
                    ref_quant_desc, quant_quant_desc, ref_is_quant, quant_is_quant,
                    ref_is_ct, quant_is_ct)

            ref_hidden, quant_hidden, ref_prev_topk, quant_prev_topk = self._forward_single_dual_layer(
                ref_layer, quant_layer, ref_hidden, quant_hidden,
                ref_pos_ids, quant_pos_ids, ref_pe, quant_pe,
                ref_prev_topk, quant_prev_topk)

            self._log_norm_debug(layer_idx, ref_hidden, quant_hidden, "OUT")
            if is_moe:
                ref_moe.forward = ref_orig_mlp_fwd
                quant_moe.forward = quant_orig_mlp_fwd

            is_target = (self.l1_target_layers is not None and layer_idx in self.l1_target_layers)
            if not is_target and self.l1_target_layers is not None:
                continue
            self._log_layer77_debug(layer_idx, ref_layer, quant_layer, ref_hidden, quant_hidden)
            metrics, result = self._compute_layer_metrics_and_cache(
                ref_hidden, quant_hidden, layer_idx, layer_cos_sims, layer_inputs,
                ref_layer_input=ref_layer_input,
                quant_layer_input=quant_layer_input,
                ref_layer_state=ref_state_input,
                quant_layer_state=quant_state_input,
            )
            all_results.append(result)
        return ref_hidden, quant_hidden, ref_prev_topk, quant_prev_topk

    @staticmethod
    def _is_moe_layer(layer: nn.Module) -> bool:
        """检测 HF decoder layer 是否是 MoE 层"""
        _, moe = get_moe_module(layer)
        return moe is not None

    def _register_activation_quant_hooks(self, model, layers: List[int]):
        """在指定层的所有 nn.Linear (除 router/gate 外) 上注册 MXFP8 激活伪量化 hook。"""
        self._clear_activation_quant_hooks()
        if not self.activation_quant:
            return
        from .utils import get_decoder_layers
        decoder_layers = get_decoder_layers(model)
        for i in layers:
            if i >= len(decoder_layers):
                continue
            layer = decoder_layers[i]
            for name, mod in layer.named_modules():
                if not isinstance(mod, nn.Linear):
                    continue
                # 跳过 MoE router/gate (选专家不应量化)
                if 'gate' in name and 'gate_up' not in name and 'gate_proj' not in name:
                    continue
                if 'router' in name:
                    continue
                # 跳过 MoE expert 内部的 linear (streaming 模式已单独处理)
                if 'experts' in name:
                    continue
                h = mod.register_forward_pre_hook(_make_act_fake_quant_hook(self.activation_quant_type))
                self._activation_hooks.append(h)
        if self.verbose and self._activation_hooks:
            logger.info(f"  [ACT FAKE QUANT] 注册了 {len(self._activation_hooks)} 个 {self.activation_quant_type} 激活伪量化 hook")

    def _clear_activation_quant_hooks(self):
        """移除所有激活伪量化 hook。"""
        for h in self._activation_hooks:
            h.remove()
        self._activation_hooks.clear()

    def _moe_forward_chunked(
        self,
        layer: nn.Module,
        hidden_states: Tensor,
        devices: List[str],
        chunk_size: int,
        layer_idx: int = None,
        sf_reader=None,
        quant_desc_str: dict = None,
        is_quant: bool = False,
        is_ct: bool = False,
    ) -> Tensor:
        """手动拆解 MoE forward，expert chunk 轮询分发到 devices。

        支持 streaming dequant 模式: 当 sf_reader + quant_desc_str 提供时，
        直接从 safetensors 读取 FP8/BF16 权重，在 NPU 上反量化+计算，
        避免在 CPU 上预构造 108GB BF16 tensor。

        Args:
            layer: HF decoder layer (含 mlp.gate/mlp.experts)
            hidden_states: [batch, seq, hidden] 在 primary device
            devices: 设备列表 (ref_devices 或 quant_devices)
            chunk_size: 每 chunk 的 expert 数
            layer_idx: 当前层索引 (用于 streaming 模式构造 key)
            sf_reader: ShardWeightReader (用于 streaming 模式)
            quant_desc_str: 量化描述 dict (用于 streaming 模式)
            is_quant: 是否量化模型

        Returns:
            MoE output tensor，在 primary device
        """
        _, mlp = get_moe_module(layer)
        if mlp is None:
            raise RuntimeError(f"Cannot find routed MoE module on {type(layer).__name__}")
        primary_device = devices[0]

        # ---- 1. Router: 手动计算 top-k expert ----
        gate = getattr(mlp, 'gate', None) or getattr(mlp, 'router', None) or getattr(mlp, 'gate_proj', None)
        if gate is None:
            out = mlp(hidden_states)
            return out[0] if isinstance(out, tuple) else out

        router_logits, precomputed_scores, precomputed_indices = self._resolve_gate_output(gate, hidden_states)
        num_experts_per_tok = self._resolve_num_experts_per_tok(mlp, gate)

        # ---- MoE router scoring ----
        topk_scores, topk_indices, routed_scaling_factor = self._compute_moe_topk_routing(
            mlp, gate, router_logits, precomputed_scores, precomputed_indices, num_experts_per_tok)

        # ---- 2. Detect expert format & streaming mode ----
        experts_mod = mlp.experts
        is_packed = hasattr(experts_mod, 'gate_up_proj') or hasattr(experts_mod, 'down_proj')
        is_module_list = isinstance(experts_mod, nn.ModuleList)
        use_streaming = (
            sf_reader is not None and layer_idx is not None
            and (is_packed or is_module_list)
            and (quant_desc_str is not None or not is_quant or is_ct)
        )

        if layer_idx == 3 and logger.isEnabledFor(logging.DEBUG):
            self._log_moe_l3_format(experts_mod, is_packed, is_module_list, use_streaming,
                                    sf_reader, quant_desc_str, is_quant)

        if not is_packed and not is_module_list:
            out = mlp(hidden_states)
            return out[0] if isinstance(out, tuple) else out

        # ---- 3. Shared expert + optional Kimi LatentMoE projection ----
        shared_output = self._forward_shared_expert(mlp, hidden_states)
        routed_input = hidden_states
        routed_down = getattr(mlp, 'routed_expert_down_proj', None)
        if routed_down is not None:
            routed_input = routed_down(hidden_states)

        # ---- 4. 逐 chunk 处理 routed experts ----
        num_experts = (
            router_logits.shape[-1] if router_logits is not None
            else getattr(mlp, 'num_experts', None)
            or len(experts_mod)
        )
        y = self._run_expert_chunks(
            mlp, experts_mod, routed_input, devices, chunk_size, layer_idx,
            topk_scores, topk_indices, num_experts_per_tok, num_experts,
            is_packed, is_module_list, use_streaming,
            sf_reader, quant_desc_str, is_quant, is_ct, primary_device)

        # Kimi K3 Stable LatentMoE maps routed output back to hidden_size.
        routed_norm = getattr(mlp, 'routed_expert_norm', None)
        routed_up = getattr(mlp, 'routed_expert_up_proj', None)
        if routed_norm is not None:
            y = routed_norm(y.to(hidden_states.dtype))
        if routed_up is not None:
            y = routed_up(y.to(hidden_states.dtype))

        # ---- 5. 加上 shared expert + debug ----
        return self._finalize_moe_output(
            y, shared_output, hidden_states, layer_idx,
            topk_scores, topk_indices, routed_scaling_factor)

    @staticmethod
    def _resolve_packed_expert_keys(sf_reader, expert_prefix):
        """Resolve Qwen3.5/3.6 fused 3D expert storage, if present."""
        weight_map = getattr(sf_reader, "weight_map", {})
        candidates = (
            (f"{expert_prefix}.gate_up_proj", f"{expert_prefix}.down_proj"),
            (f"{expert_prefix}.gate_up_proj.weight",
             f"{expert_prefix}.down_proj.weight"),
        )
        for gate_up_key, down_key in candidates:
            if weight_map and gate_up_key in weight_map and down_key in weight_map:
                get_shape = getattr(sf_reader, "get_tensor_shape", None)
                gate_up_shape = get_shape(gate_up_key) if callable(get_shape) else None
                down_shape = get_shape(down_key) if callable(get_shape) else None
                if gate_up_shape is None or down_shape is None:
                    gate_up_shape = tuple(sf_reader.get_tensor(gate_up_key).shape)
                    down_shape = tuple(sf_reader.get_tensor(down_key).shape)
                if len(gate_up_shape) == 3 and len(down_shape) == 3:
                    if gate_up_shape[0] != down_shape[0]:
                        raise ValueError(
                            "packed expert tensors disagree on num_experts: "
                            f"{gate_up_shape} vs {down_shape}"
                        )
                    return gate_up_key, down_key, gate_up_shape[0]

        if not weight_map:
            for gate_up_key, down_key in candidates:
                gate_up = sf_reader.get_tensor(gate_up_key)
                down = sf_reader.get_tensor(down_key)
                if (
                    gate_up is not None and down is not None
                    and gate_up.dim() == 3 and down.dim() == 3
                ):
                    if gate_up.shape[0] != down.shape[0]:
                        raise ValueError(
                            "packed expert tensors disagree on num_experts: "
                            f"{tuple(gate_up.shape)} vs {tuple(down.shape)}"
                        )
                    return gate_up_key, down_key, gate_up.shape[0]
        return None

    @staticmethod
    def _streaming_quant_type(quant_desc_str, weight_key, default="FLOAT"):
        """Look up quant type for both Parameter and Linear-style keys."""
        if not quant_desc_str:
            return default
        from .utils import parse_base_name
        candidates = [weight_key, parse_base_name(weight_key)]
        if weight_key.endswith(".weight"):
            candidates.append(weight_key[:-len(".weight")])
        else:
            candidates.extend((
                f"{weight_key}.weight",
                parse_base_name(f"{weight_key}.weight"),
            ))
        for candidate in candidates:
            quant_type = quant_desc_str.get(candidate)
            if isinstance(quant_type, str):
                return quant_type
        return default

    def _dequant_streaming_weight(self, sf_reader, weight_key, w_type, device,
                                    is_ct=False, expert_id=None,
                                    num_experts=None):
        """Read/dequantize one expert weight from split or fused storage."""
        from .model_loader import dequantize_weight_mx, _dequant_msslim_weight
        reader = (
            _ExpertSliceReader(sf_reader, expert_id, num_experts)
            if expert_id is not None and num_experts is not None
            else sf_reader
        )
        w = reader.get_tensor(weight_key)
        quant_name = (
            weight_key.rsplit('.', 1)[0]
            if weight_key.endswith(".weight")
            else weight_key
        )
        if w is None and is_ct:
            packed = reader.get_tensor(f"{quant_name}.weight_packed")
            scale = reader.get_tensor(f"{quant_name}.weight_scale")
            if packed is not None and scale is not None:
                fp = dequantize_weight_mx(
                    packed, scale, "W4A4_MXFP4", dtype=self.dtype,
                ).to(device)
                del packed, scale
                return fp
        if w is None:
            raise KeyError(f"streaming expert weight not found: {weight_key}")
        if w_type == "FLOAT":
            if is_ct and not w.dtype.is_floating_point:
                scale = reader.get_tensor(f"{quant_name}.weight_scale")
                if scale is not None:
                    fp = (w.to(self.dtype) * scale.to(self.dtype)).to(device)
                    del w, scale
                    return fp
            if not w.dtype.is_floating_point:
                raise ValueError(
                    "integer streaming expert was classified as FLOAT; "
                    f"quantization metadata is missing for {weight_key}"
                )
            fp = w.to(device=device, dtype=self.dtype)
            del w
            return fp

        fp, status = _dequant_msslim_weight(
            w, w_type, quant_name, reader, self.dtype
        )
        del w
        if status == "unknown":
            raise NotImplementedError(
                f"streaming expert quant type is not supported: {w_type} "
                f"({weight_key})"
            )
        if fp is None:
            raise ValueError(
                "streaming expert is missing dequantization parameters: "
                f"{w_type} ({weight_key})"
            )
        return fp.to(device)

    def _dequant_streaming_proj(self, sf_reader, expert_prefix, expert_id, proj_name,
                                  w_type, device, is_ct=False):
        """反量化单个 streaming proj 权重 (gate/up/down)。返回 fp tensor 或 None。"""
        key = f"{expert_prefix}.{expert_id}.{proj_name}.weight"
        return self._dequant_streaming_weight(
            sf_reader, key, w_type, device, is_ct=is_ct,
        )

    def _streaming_expert_forward(
        self,
        expert_id: int,
        x_chunk: Tensor,
        device: str,
        expert_prefix: str,
        sf_reader,
        quant_desc_str: dict,
        is_quant: bool,
        is_ct: bool,
        mlp: nn.Module,
    ) -> Optional[Tensor]:
        """Streaming expert forward: 从 safetensors 读取权重，反量化后传 NPU 计算。

        每个 expert 的权重用完立即释放，不在 CPU 上构造完整 3D tensor。
        对量化模型: CPU 反量化到目标 dtype，传 NPU 做 F.linear
        对非量化模型: 直接读 BF16，传 NPU 做 F.linear

        Returns:
            expert output tensor on device, 或 None
        """
        packed_keys = self._resolve_packed_expert_keys(sf_reader, expert_prefix)
        if packed_keys is not None:
            gate_up_key, down_key, num_experts = packed_keys
            g_type = self._streaming_quant_type(
                quant_desc_str, gate_up_key, "FLOAT"
            ) if is_quant else "FLOAT"
            u_type = g_type
            d_type = self._streaming_quant_type(
                quant_desc_str, down_key, g_type
            ) if is_quant else "FLOAT"
        else:
            gate_name, up_name, down_name = self._resolve_expert_proj_names(
                sf_reader, expert_prefix, expert_id)
            gate_key = f"{expert_prefix}.{expert_id}.{gate_name}.weight"
            up_key = f"{expert_prefix}.{expert_id}.{up_name}.weight"
            down_key = f"{expert_prefix}.{expert_id}.{down_name}.weight"
            if is_quant:
                g_type = self._streaming_quant_type(
                    quant_desc_str, gate_key, "FLOAT"
                )
                u_type = self._streaming_quant_type(
                    quant_desc_str, up_key, g_type
                )
                d_type = self._streaming_quant_type(
                    quant_desc_str, down_key, g_type
                )
            else:
                g_type = u_type = d_type = "FLOAT"

        # ---- gate/up/down proj: 读+反量化(CPU)→传NPU ----
        if packed_keys is not None:
            gate_up_fp = self._dequant_streaming_weight(
                sf_reader, gate_up_key, g_type, device, is_ct=is_ct,
                expert_id=expert_id, num_experts=num_experts,
            )
            down_fp = self._dequant_streaming_weight(
                sf_reader, down_key, d_type, device, is_ct=is_ct,
                expert_id=expert_id, num_experts=num_experts,
            )
            intermediate_dim = down_fp.shape[-1]
            if gate_up_fp.shape[0] != 2 * intermediate_dim:
                raise ValueError(
                    "packed gate_up_proj shape is incompatible with down_proj: "
                    f"{tuple(gate_up_fp.shape)} vs {tuple(down_fp.shape)}"
                )
            gate_fp = gate_up_fp[:intermediate_dim]
            up_fp = gate_up_fp[intermediate_dim:]
            del gate_up_fp
        else:
            gate_fp = self._dequant_streaming_proj(
                sf_reader, expert_prefix, expert_id, gate_name, g_type,
                device, is_ct=is_ct,
            )
            up_fp = self._dequant_streaming_proj(
                sf_reader, expert_prefix, expert_id, up_name, u_type,
                device, is_ct=is_ct,
            )
            down_fp = self._dequant_streaming_proj(
                sf_reader, expert_prefix, expert_id, down_name, d_type,
                device, is_ct=is_ct,
            )

        # ---- gate_up_proj → SiLU(gate) * up → down_proj ----
        # DEBUG: print expert weight stats for first expert processed at layer 3
        if logger.isEnabledFor(logging.DEBUG) and getattr(self, '_debug_layer3_expert0', False):
            logger.debug(f"  [STREAM EXPERT {expert_id}] g_type={g_type} u_type={u_type} d_type={d_type}")
            logger.debug(f"  [STREAM EXPERT {expert_id}] gate_fp: shape={gate_fp.shape} min={gate_fp.float().min().item():.6f} max={gate_fp.float().max().item():.6f} norm={gate_fp.float().norm().item():.6f}")
            logger.debug(f"  [STREAM EXPERT {expert_id}] up_fp: shape={up_fp.shape} min={up_fp.float().min().item():.6f} max={up_fp.float().max().item():.6f} norm={up_fp.float().norm().item():.6f}")
            logger.debug(f"  [STREAM EXPERT {expert_id}] down_fp: shape={down_fp.shape} min={down_fp.float().min().item():.6f} max={down_fp.float().max().item():.6f} norm={down_fp.float().norm().item():.6f}")
            logger.debug(f"  [STREAM EXPERT {expert_id}] x_chunk: shape={x_chunk.shape} min={x_chunk.float().min().item():.6f} max={x_chunk.float().max().item():.6f}")
            self._debug_layer3_expert0 = False  # only once

        # activation fake quant: 模拟 per-block 激活量化 (按 activation_quant_type 分派)
        x_for_gate = x_chunk
        x_for_up = x_chunk
        if self.activation_quant:
            x_for_gate = _dispatch_act_fake_quant(x_chunk, self.activation_quant_type)
            x_for_up = _dispatch_act_fake_quant(x_chunk, self.activation_quant_type)

        gate_out = torch.nn.functional.linear(x_for_gate, gate_fp)
        up_out = torch.nn.functional.linear(x_for_up, up_fp)
        act_out = self._apply_expert_activation(mlp, gate_out, up_out)

        if self.activation_quant:
            act_out = _dispatch_act_fake_quant(act_out, self.activation_quant_type)

        expert_out = torch.nn.functional.linear(act_out, down_fp)

        # DEBUG: print expert output stats at layer 3
        if logger.isEnabledFor(logging.DEBUG) and getattr(self, '_debug_layer3_out', False):
            logger.debug(f"  [STREAM EXPERT OUT {expert_id}] gate_out: norm={gate_out.float().norm().item():.6f} absmax={gate_out.float().abs().max().item():.6f}")
            logger.debug(f"  [STREAM EXPERT OUT {expert_id}] up_out: norm={up_out.float().norm().item():.6f} absmax={up_out.float().abs().max().item():.6f}")
            logger.debug(f"  [STREAM EXPERT OUT {expert_id}] act_out: norm={act_out.float().norm().item():.6f} absmax={act_out.float().abs().max().item():.6f}")
            logger.debug(f"  [STREAM EXPERT OUT {expert_id}] expert_out: norm={expert_out.float().norm().item():.6f} absmax={expert_out.float().abs().max().item():.6f}")
            self._debug_layer3_out = False

        del gate_fp, up_fp, down_fp, gate_out, up_out, act_out
        return expert_out

    @staticmethod
    def _resolve_expert_proj_names(sf_reader, expert_prefix, expert_id):
        """Resolve Qwen/GLM gate-up-down names or Kimi w1-w3-w2 names."""
        weight_map = getattr(sf_reader, 'weight_map', {})
        for names in (("gate_proj", "up_proj", "down_proj"), ("w1", "w3", "w2")):
            gate = f"{expert_prefix}.{expert_id}.{names[0]}"
            if (
                f"{gate}.weight" in weight_map
                or f"{gate}.weight_packed" in weight_map
            ):
                return names
        return "gate_proj", "up_proj", "down_proj"

    @staticmethod
    def _apply_expert_activation(mlp, gate_out, up_out):
        """Apply the expert GLU activation, including Kimi K3 SiTU."""
        config = getattr(mlp, 'config', None)
        hidden_act = getattr(config, 'hidden_act', 'silu')
        if hidden_act == 'situ':
            beta = float(getattr(config, 'activation_situ_beta', 1.0) or 1.0)
            linear_beta = getattr(config, 'activation_situ_linear_beta', None)
            gate = gate_out.float()
            up = up_out.float()
            gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
            if linear_beta is not None:
                up = float(linear_beta) * torch.tanh(up / float(linear_beta))
            return (gate * up).to(gate_out.dtype)
        return torch.nn.functional.silu(gate_out) * up_out

    # ---- MoE chunked forward 辅助方法 (Extract Method 降圈复杂度) ----

    def _resolve_gate_output(self, gate, hidden_states):
        """解析 GLM/Qwen3.6/Kimi K3 router 的不同返回约定。"""
        precomputed_scores = None
        precomputed_indices = None
        gate_out = gate(hidden_states)
        if isinstance(gate_out, tuple):
            # Kimi K3 returns (topk_indices, topk_weights), without logits.
            if (
                len(gate_out) == 2
                and isinstance(gate_out[0], torch.Tensor)
                and not gate_out[0].dtype.is_floating_point
            ):
                router_logits = None
                precomputed_indices = gate_out[0]
                precomputed_scores = gate_out[1]
            else:
                router_logits = gate_out[0]
            if len(gate_out) >= 3:
                precomputed_scores = gate_out[1]
                precomputed_indices = gate_out[2]
            elif len(gate_out) == 2 and router_logits is not None:
                precomputed_indices = gate_out[1]
        else:
            router_logits = gate_out
        return router_logits, precomputed_scores, precomputed_indices

    def _resolve_num_experts_per_tok(self, mlp, gate):
        """解析 num_experts_per_tok，多处 fallback。"""
        num_experts_per_tok = (getattr(mlp, 'top_k', None)
                               or getattr(mlp, 'num_experts_per_tok', None)
                               or getattr(gate, 'top_k', None)
                               or getattr(gate, 'num_experts_per_tok', None))
        if num_experts_per_tok is not None:
            return num_experts_per_tok
        import json
        config_path = os.path.join(self.ref_model_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                num_experts_per_tok = (cfg.get('num_experts_per_tok')
                                       or cfg.get('num_experts_per_token')
                                       or cfg.get('moe', {}).get('num_experts_per_tok')
                                       or cfg.get('text_config', {}).get('num_experts_per_tok')
                                       or 8)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to read config.json for num_experts_per_tok: {e}")
                num_experts_per_tok = 8
        else:
            num_experts_per_tok = 8
        return num_experts_per_tok

    def _compute_sigmoid_routing(self, mlp, gate, router_logits, num_experts_per_tok):
        """GLM-5.1 / DeepSeek-V3 风格 sigmoid + noaux_tc + renorm + scaling 路由。"""
        n_routed_experts = router_logits.shape[-1]
        n_group = getattr(mlp, 'n_group', None) or 1
        topk_group = getattr(mlp, 'topk_group', None) or 1
        norm_topk_prob = getattr(mlp, 'norm_topk_prob', None)
        norm_topk_prob = True if norm_topk_prob is None else bool(norm_topk_prob)
        routed_scaling_factor = getattr(mlp, 'routed_scaling_factor', None)
        routed_scaling_factor = (float(routed_scaling_factor)
                                 if routed_scaling_factor is not None else 1.0)

        sigmoid_scores = router_logits.float().sigmoid()
        score_correction_bias = getattr(gate, 'e_score_correction_bias', None)
        scores_for_choice = sigmoid_scores
        if score_correction_bias is not None and score_correction_bias.numel() > 0:
            scores_for_choice = sigmoid_scores + score_correction_bias.to(
                device=sigmoid_scores.device, dtype=sigmoid_scores.dtype)

        if n_group > 1 and n_routed_experts >= n_group:
            scores_for_choice = self._apply_noaux_tc_group_filter(
                scores_for_choice, n_routed_experts, n_group, topk_group)

        _, topk_indices = scores_for_choice.topk(num_experts_per_tok, dim=-1, sorted=False)
        topk_scores = sigmoid_scores.gather(-1, topk_indices)
        if norm_topk_prob:
            denom = topk_scores.sum(dim=-1, keepdim=True) + 1e-20
            topk_scores = topk_scores / denom
        topk_scores = topk_scores * routed_scaling_factor
        return topk_scores, topk_indices, routed_scaling_factor

    @staticmethod
    def _apply_noaux_tc_group_filter(scores_for_choice, n_routed_experts, n_group, topk_group):
        """noaux_tc 组过滤: n_group>1 时按 group top-2 sum 选 topk_group 个 group。"""
        group_size = n_routed_experts // n_group
        group_scores = (
            scores_for_choice.view(-1, n_group, group_size)
            .topk(2, dim=-1)[0].sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, n_group, group_size)
            .reshape(-1, n_routed_experts)
        )
        return scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))

    def _compute_softmax_routing(self, router_logits, precomputed_scores,
                                  precomputed_indices, num_experts_per_tok):
        """legacy softmax 评分 (Qwen MoE 等); Qwen3.6 用预计算结果。"""
        if precomputed_scores is not None and precomputed_indices is not None:
            return precomputed_scores, precomputed_indices
        router_scores = torch.nn.functional.softmax(router_logits.float(), dim=-1)
        topk_scores, topk_indices = router_scores.topk(num_experts_per_tok, dim=-1)
        return topk_scores, topk_indices

    def _compute_moe_topk_routing(self, mlp, gate, router_logits,
                                   precomputed_scores, precomputed_indices,
                                   num_experts_per_tok):
        """MoE router scoring 总入口: sigmoid 或 softmax(fallback)。"""
        routed_scaling_factor = getattr(mlp, 'routed_scaling_factor', None)
        if router_logits is None:
            if precomputed_scores is None or precomputed_indices is None:
                raise RuntimeError("Router returned no logits and no precomputed top-k")
            return precomputed_scores, precomputed_indices, routed_scaling_factor
        use_sigmoid_routing = (
            routed_scaling_factor is not None
            or getattr(mlp, 'norm_topk_prob', None) is not None
            or getattr(mlp, 'n_group', None) is not None
            or hasattr(gate, 'e_score_correction_bias')
        )
        if use_sigmoid_routing:
            return self._compute_sigmoid_routing(mlp, gate, router_logits, num_experts_per_tok)
        topk_scores, topk_indices = self._compute_softmax_routing(
            router_logits, precomputed_scores, precomputed_indices, num_experts_per_tok)
        return topk_scores, topk_indices, routed_scaling_factor

    def _log_moe_l3_format(self, experts_mod, is_packed, is_module_list,
                            use_streaming, sf_reader, quant_desc_str, is_quant):
        """layer 3 MoE format debug 日志。"""
        logger.debug(f"  [MOE L3] is_packed={is_packed} is_module_list={is_module_list} use_streaming={use_streaming}")
        logger.debug(f"  [MOE L3] sf_reader={sf_reader is not None} quant_desc_str={quant_desc_str is not None} is_quant={is_quant}")
        logger.debug(f"  [MOE L3] experts type={type(experts_mod).__name__}")
        for attr in ['gate_up_proj', 'down_proj', 'gate_proj', 'up_proj']:
            v = getattr(experts_mod, attr, None)
            if v is not None:
                logger.debug(f"  [MOE L3] experts.{attr}: type={type(v).__name__} shape={getattr(v, 'shape', None)} dtype={getattr(v, 'dtype', None)}")

    @staticmethod
    def _forward_shared_expert(mlp, hidden_states):
        """shared expert forward (GLM-5.1: shared_experts, others: shared_expert)。"""
        shared_expert = getattr(mlp, 'shared_experts', None) or getattr(mlp, 'shared_expert', None)
        if shared_expert is None:
            return None
        with torch.no_grad():
            shared_out = shared_expert(hidden_states)
            return shared_out[0] if isinstance(shared_out, tuple) else shared_out

    def _forward_single_expert_packed(self, experts_mod, eid, x_chunk, chunk_device):
        """Legacy: 从预构造的 3D packed tensor 取 expert 权重并 forward。"""
        gate_up_w = getattr(experts_mod, 'gate_up_proj', None)
        down_w = getattr(experts_mod, 'down_proj', None)
        if gate_up_w is None or down_w is None:
            return None
        gate_up_e = gate_up_w.data[eid].to(chunk_device, non_blocking=True)
        down_e = down_w.data[eid].to(chunk_device, non_blocking=True)
        intermediate_dim = down_e.shape[1]
        x_for_gate_up = x_chunk
        if self.activation_quant:
            x_for_gate_up = _dispatch_act_fake_quant(x_chunk, self.activation_quant_type)
        gate_up_out = torch.nn.functional.linear(x_for_gate_up, gate_up_e)
        gate_out = gate_up_out[..., :intermediate_dim]
        up_out = gate_up_out[..., intermediate_dim:]
        act_out = torch.nn.functional.silu(gate_out) * up_out
        if self.activation_quant:
            act_out = _dispatch_act_fake_quant(act_out, self.activation_quant_type)
        expert_out = torch.nn.functional.linear(act_out, down_e)
        del gate_up_e, down_e
        return expert_out

    @staticmethod
    def _forward_single_expert_module_list(experts_mod, eid, x_chunk, chunk_device):
        """ModuleList 风格 expert forward。"""
        expert_mod = experts_mod[eid]
        if expert_mod is None:
            return None
        curr_dev = next(expert_mod.parameters()).device
        need_move = (str(curr_dev) != chunk_device)
        if need_move:
            expert_mod.to(chunk_device)
        expert_out = expert_mod(x_chunk)
        if isinstance(expert_out, tuple):
            expert_out = expert_out[0]
        if need_move:
            expert_mod.to('cpu')
            if hasattr(torch, 'npu') and torch.npu.is_available():
                torch.npu.empty_cache()
        return expert_out

    @staticmethod
    def _resolve_expert_prefix(sf_reader, layer_idx: int, moe_attr: str) -> str:
        """Resolve the checkpoint prefix for routed experts from its weight map."""
        marker = f".layers.{layer_idx}.{moe_attr}.experts."
        for key in getattr(sf_reader, 'weight_map', {}):
            if marker in key:
                return key.split(marker, 1)[0] + marker[:-1]
        return f"model.layers.{layer_idx}.{moe_attr}.experts"

    def _run_expert_chunks(self, mlp, experts_mod, hidden_states, devices, chunk_size,
                            layer_idx, topk_scores, topk_indices, num_experts_per_tok,
                            num_experts, is_packed, is_module_list, use_streaming,
                            sf_reader, quant_desc_str, is_quant, is_ct, primary_device):
        """逐 chunk 处理 routed experts，返回累积输出 y。"""
        orig_shape = hidden_states.shape
        h_flat = hidden_states.view(-1, orig_shape[-1])
        scores_flat = topk_scores.view(-1, num_experts_per_tok)
        indices_flat = topk_indices.view(-1, num_experts_per_tok)
        y = torch.zeros_like(h_flat, dtype=torch.float32)
        # Infer the owning layer attribute from the checkpoint.  This supports
        # both Qwen/GLM ``mlp.experts`` and Kimi ``block_sparse_moe.experts``.
        if sf_reader is not None:
            candidate_attrs = ('block_sparse_moe', 'mlp', 'moe')
            expert_prefix = None
            for attr in candidate_attrs:
                prefix = self._resolve_expert_prefix(sf_reader, layer_idx, attr)
                if any(
                    key.startswith(prefix + '.')
                    for key in getattr(sf_reader, 'weight_map', {})
                ):
                    expert_prefix = prefix
                    break
            if expert_prefix is None:
                expert_prefix = self._resolve_expert_prefix(sf_reader, layer_idx, 'mlp')
        else:
            expert_prefix = f"model.layers.{layer_idx}.mlp.experts"

        for chunk_start in range(0, num_experts, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_experts)
            chunk_ids = list(range(chunk_start, chunk_end))
            chunk_idx = chunk_start // chunk_size
            chunk_device = devices[chunk_idx % len(devices)]
            for eid in chunk_ids:
                expert_out = self._forward_single_routed_expert(
                    eid, h_flat, scores_flat, indices_flat, chunk_device,
                    experts_mod, is_packed, is_module_list, use_streaming,
                    expert_prefix, sf_reader, quant_desc_str, is_quant, is_ct, mlp,
                    y, primary_device, layer_idx)
                if expert_out is not None:
                    del expert_out
            self._sync_chunk_device(chunk_device)
        return y

    def _forward_single_routed_expert(self, eid, h_flat, scores_flat, indices_flat,
                                       chunk_device, experts_mod, is_packed,
                                       is_module_list, use_streaming, expert_prefix,
                                       sf_reader, quant_desc_str, is_quant, is_ct, mlp,
                                       y, primary_device, layer_idx):
        """处理单个 routed expert: 取 token mask → forward → 累积到 y。"""
        eid_mask = (indices_flat == eid)
        token_mask = eid_mask.any(dim=-1)
        if not token_mask.any():
            return None
        s_for_eid = (scores_flat * eid_mask.float()).sum(dim=-1)
        x_chunk = h_flat[token_mask].to(chunk_device, non_blocking=True)
        s_chunk = s_for_eid[token_mask].to(chunk_device, non_blocking=True)

        expert_out = None
        if use_streaming:
            expert_out = self._streaming_expert_forward(
                eid, x_chunk, chunk_device, expert_prefix,
                sf_reader, quant_desc_str, is_quant, is_ct, mlp)
        elif is_packed:
            expert_out = self._forward_single_expert_packed(experts_mod, eid, x_chunk, chunk_device)
        elif is_module_list:
            expert_out = self._forward_single_expert_module_list(experts_mod, eid, x_chunk, chunk_device)
        if expert_out is None:
            return None

        out_weighted = (expert_out.to(torch.float32) *
                        s_chunk.unsqueeze(-1).to(torch.float32))
        y[token_mask] += out_weighted.to(primary_device)
        if layer_idx == 3 and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"  [MOE L3] after expert {eid}: y_norm={y.float().norm().item():.6f} y_absmax={y.float().abs().max().item():.6f} expert_out_norm={expert_out.float().norm().item():.6f} s={s_chunk.tolist()}")
        return expert_out

    @staticmethod
    def _sync_chunk_device(chunk_device):
        """同步 chunk 设备并清理缓存。"""
        if hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.synchronize(chunk_device)
            torch.npu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.synchronize(chunk_device)
            torch.cuda.empty_cache()

    def _finalize_moe_output(self, y, shared_output, hidden_states, layer_idx,
                              topk_scores, topk_indices, routed_scaling_factor):
        """加 shared expert + debug 日志，返回最终输出。"""
        orig_shape = hidden_states.shape
        debug_l3 = layer_idx == 3 and logger.isEnabledFor(logging.DEBUG)
        if shared_output is not None:
            shared_flat = shared_output.view(-1, orig_shape[-1]).to(torch.float32)
            if debug_l3:
                logger.debug(f"  [MOE L3] shared_output: norm={shared_flat.float().norm().item():.6f} absmax={shared_flat.float().abs().max().item():.6f}")
                logger.debug(f"  [MOE L3] routed y (pre-shared): norm={y.float().norm().item():.6f} absmax={y.float().abs().max().item():.6f}")
            y = y + shared_flat.to(hidden_states.device)
        elif debug_l3:
            logger.debug(f"  [MOE L3] routed y (no shared): norm={y.float().norm().item():.6f} absmax={y.float().abs().max().item():.6f}")
        if debug_l3:
            logger.debug(f"  [MOE L3] topk_scores: {topk_scores[0].tolist()}")
            logger.debug(f"  [MOE L3] topk_indices: {topk_indices[0].tolist()}")
            logger.debug(f"  [MOE L3] routed_scaling_factor={routed_scaling_factor}")
            logger.debug(f"  [MOE L3] final y: norm={y.float().norm().item():.6f} absmax={y.float().abs().max().item():.6f}")
        return y.view(orig_shape).to(hidden_states.dtype)

    # ---- dual sharded forward 辅助方法 (Extract Method 降圈复杂度) ----

    def _setup_dual_position_embeddings(self, ref_hidden, quant_hidden,
                                          ref_model, quant_model):
        """为 dual 模式创建 position_ids + position_embeddings。"""
        from .utils import get_rotary_emb_module
        batch_size, seq_len = ref_hidden.shape[:2]
        ref_position_ids = torch.arange(seq_len, dtype=torch.long, device=ref_hidden.device)
        ref_position_ids = ref_position_ids.unsqueeze(0).expand(batch_size, -1)
        quant_position_ids = torch.arange(seq_len, dtype=torch.long, device=quant_hidden.device)
        quant_position_ids = quant_position_ids.unsqueeze(0).expand(batch_size, -1)
        ref_rotary = get_rotary_emb_module(ref_model)
        quant_rotary = get_rotary_emb_module(quant_model)
        if ref_rotary is not None and quant_rotary is not None:
            ref_pe = ref_rotary(ref_hidden, ref_position_ids)
            quant_pe = quant_rotary(quant_hidden, quant_position_ids)
        else:
            ref_pe = None
            quant_pe = None
        return ref_position_ids, quant_position_ids, ref_pe, quant_pe

    @staticmethod
    def _build_cross_layer_state_kwargs(layer, state, hidden_states):
        """Move/init decoder cross-layer state and return it as forward kwargs."""
        state_name = get_layer_state_kwarg(layer)
        if state_name is None:
            return {}
        if state_name == "block_residual" and state is None:
            batch, seq_len, hidden_size = hidden_states.shape
            state = hidden_states.new_zeros(batch * seq_len, 0, hidden_size)
        if state is not None:
            state = state.to(hidden_states.device)
        return {state_name: state}

    @staticmethod
    def _resolve_kimi_kda_backend(requested: str, device) -> Optional[str]:
        """Resolve the Kimi KDA execution mode for the current device."""
        if requested != "auto":
            return requested
        device_type = getattr(device, "type", str(device).split(":", 1)[0])
        if str(device_type).lower() == "npu":
            # Kimi's chunk_kda reaches chunk_gla kernels that are not compiled
            # by the CANN 8.5.1 Triton-Ascend stack.  Use an equivalent eager
            # torch recurrence so this correctness tool does not depend on a
            # Triton KDA compiler path at all.
            return "torch"
        return None

    def _configure_kimi_kda_backend(self, layer, hidden_states) -> Optional[str]:
        """Select Kimi K3's KDA backend without touching MLA layers."""
        attn = getattr(layer, "self_attn", None)
        is_kda = bool(
            attn is not None
            and getattr(attn, "q_conv1d", None) is not None
            and getattr(attn, "f_a_proj", None) is not None
            and hasattr(attn, "mode")
        )
        if not is_kda:
            return None
        backend = self._resolve_kimi_kda_backend(
            getattr(self, "kimi_kda_backend", "auto"), hidden_states.device
        )
        if backend is None:
            return getattr(attn, "mode", None)
        forward_fn = getattr(attn.forward, "__func__", attn.forward)
        forward_globals = getattr(forward_fn, "__globals__", None)
        if forward_globals is None:
            raise RuntimeError(
                "[Kimi K3] cannot configure KDA backend: forward globals unavailable"
            )
        original_chunk_key = "__acc_bench_original_chunk_kda"
        original_recurrent_key = "__acc_bench_original_fused_recurrent_kda"
        if original_chunk_key not in forward_globals:
            forward_globals[original_chunk_key] = forward_globals.get("chunk_kda")
        if original_recurrent_key not in forward_globals:
            forward_globals[original_recurrent_key] = forward_globals.get(
                "fused_recurrent_kda"
            )

        if backend == "torch":
            from .kimi_kda import torch_recurrent_kda
            forward_globals["chunk_kda"] = torch_recurrent_kda
            forward_globals["fused_recurrent_kda"] = torch_recurrent_kda
            # Both branches are patched; use chunk for compatibility with old
            # Kimi remote-code versions that only recognize this mode.
            attn.mode = "chunk"
        else:
            original_chunk = forward_globals.get(original_chunk_key)
            original_recurrent = forward_globals.get(original_recurrent_key)
            if original_chunk is not None:
                forward_globals["chunk_kda"] = original_chunk
            if original_recurrent is not None:
                forward_globals["fused_recurrent_kda"] = original_recurrent
            attn.mode = backend
        if (getattr(self, "verbose", False)
                and not getattr(self, "_kimi_kda_backend_logged", False)):
            logger.info(
                "[Kimi K3] KDA backend: %s%s",
                backend,
                " (auto-selected for Ascend NPU)"
                if getattr(self, "kimi_kda_backend", "auto") == "auto" else "",
            )
            self._kimi_kda_backend_logged = True
        return backend

    def _forward_single_dual_layer(self, ref_layer, quant_layer, ref_hidden, quant_hidden,
                                     ref_pos_ids, quant_pos_ids, ref_pe, quant_pe,
                                     ref_prev_topk, quant_prev_topk):
        """forward 单层 ref+quant，返回更新后的 hidden + prev_topk。"""
        self._configure_kimi_kda_backend(ref_layer, ref_hidden)
        self._configure_kimi_kda_backend(quant_layer, quant_hidden)
        ref_fwd_kwargs = {}
        if ref_pe is not None:
            ref_fwd_kwargs['position_embeddings'] = ref_pe
        ref_fwd_kwargs['position_ids'] = ref_pos_ids
        ref_attention_mask = build_replay_attention_mask(ref_layer, ref_hidden)
        if ref_attention_mask is not None:
            ref_fwd_kwargs['attention_mask'] = ref_attention_mask
        ref_fwd_kwargs.update(self._build_cross_layer_state_kwargs(
            ref_layer, ref_prev_topk, ref_hidden))
        ref_out = ref_layer(ref_hidden, **ref_fwd_kwargs)
        ref_hidden = ref_out[0] if isinstance(ref_out, tuple) else ref_out
        if isinstance(ref_out, tuple) and len(ref_out) > 1 and ref_out[1] is not None:
            ref_prev_topk = ref_out[1]

        quant_fwd_kwargs = {}
        if quant_pe is not None:
            quant_fwd_kwargs['position_embeddings'] = quant_pe
        quant_fwd_kwargs['position_ids'] = quant_pos_ids
        quant_attention_mask = build_replay_attention_mask(quant_layer, quant_hidden)
        if quant_attention_mask is not None:
            quant_fwd_kwargs['attention_mask'] = quant_attention_mask
        quant_fwd_kwargs.update(self._build_cross_layer_state_kwargs(
            quant_layer, quant_prev_topk, quant_hidden))
        quant_out = quant_layer(quant_hidden, **quant_fwd_kwargs)
        quant_hidden = quant_out[0] if isinstance(quant_out, tuple) else quant_out
        if isinstance(quant_out, tuple) and len(quant_out) > 1 and quant_out[1] is not None:
            quant_prev_topk = quant_out[1]
        return ref_hidden, quant_hidden, ref_prev_topk, quant_prev_topk

    def _log_dual_layer_debug(self, layer_idx, layer_start, shard_idx,
                                ref_hidden, quant_hidden, ref_layers, quant_layers):
        """dual 模式逐层 debug 日志 (NaN 检查 + 设备检查)。"""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        ref_input_nan = torch.isnan(ref_hidden).any() if ref_hidden is not None else False
        quant_input_nan = torch.isnan(quant_hidden).any() if quant_hidden is not None else False
        if layer_idx == layer_start:
            logger.debug(f"shard {shard_idx+1} input: ref_input_nan={ref_input_nan}, quant_input_nan={quant_input_nan}")
            if ref_input_nan:
                logger.info(f"    ref input shape: {ref_hidden.shape}, max: {ref_hidden.abs().max()}")
            if quant_input_nan:
                logger.info(f"    quant input shape: {quant_hidden.shape}, max: {quant_hidden.abs().max()}")
            logger.debug(f"    ref_layer.{layer_idx} first param device: {next(ref_layers[layer_idx].parameters()).device}")
            logger.debug(f"    quant_layer.{layer_idx} first param device: {next(quant_layers[layer_idx].parameters()).device}")

    @staticmethod
    def _log_dual_post_forward_nan(layer_idx, ref_hidden, quant_hidden):
        """forward 后 NaN 检查日志。"""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        ref_has_nan = torch.isnan(ref_hidden).any()
        quant_has_nan = torch.isnan(quant_hidden).any()
        if ref_has_nan or quant_has_nan:
            logger.debug(f"layer.{layer_idx}: ref_has_nan={ref_has_nan}, quant_has_nan={quant_has_nan}")
            if ref_has_nan:
                logger.info(f"    ref NaN count: {torch.isnan(ref_hidden).sum()}, max: {ref_hidden.abs().max()}")
            if quant_has_nan:
                logger.info(f"    quant NaN count: {torch.isnan(quant_hidden).sum()}, max: {quant_hidden.abs().max()}")

    def _forward_dual_layers(self, ref_hidden, quant_hidden, ref_model, quant_model,
                               layer_start, layer_end, ref_pos_ids, quant_pos_ids,
                               ref_pe, quant_pe, ref_prev_topk, quant_prev_topk,
                               shard_idx, layer_cos_sims, layer_inputs, all_results):
        """dual 模式逐层 forward + 对比 + 缓存。"""
        from .utils import get_decoder_layers
        ref_layers = get_decoder_layers(ref_model)
        quant_layers = get_decoder_layers(quant_model)
        for layer_idx in range(layer_start, layer_end):
            self._log_dual_layer_debug(layer_idx, layer_start, shard_idx,
                                         ref_hidden, quant_hidden, ref_layers, quant_layers)
            ref_layer = ref_layers[layer_idx]
            quant_layer = quant_layers[layer_idx]
            ref_layer_input = ref_hidden
            quant_layer_input = quant_hidden
            ref_state_input = ref_prev_topk
            quant_state_input = quant_prev_topk
            ref_hidden, quant_hidden, ref_prev_topk, quant_prev_topk = self._forward_single_dual_layer(
                ref_layer, quant_layer, ref_hidden, quant_hidden,
                ref_pos_ids, quant_pos_ids, ref_pe, quant_pe,
                ref_prev_topk, quant_prev_topk)
            self._log_dual_post_forward_nan(layer_idx, ref_hidden, quant_hidden)
            is_target = (self.l1_target_layers is not None and layer_idx in self.l1_target_layers)
            if not is_target and self.l1_target_layers is not None:
                continue
            metrics, result = self._compute_layer_metrics_and_cache(
                ref_hidden, quant_hidden, layer_idx, layer_cos_sims, layer_inputs,
                ref_layer_input=ref_layer_input,
                quant_layer_input=quant_layer_input,
                ref_layer_state=ref_state_input,
                quant_layer_state=quant_state_input,
            )
            all_results.append(result)
        return ref_hidden, quant_hidden, ref_prev_topk, quant_prev_topk

    def _log_shard_memory(self, ref_device, quant_device, stage, devices=None):
        """shard 各阶段内存日志。"""
        if not self.verbose:
            return
        logger.info(f"  [RSS] {stage}: {self._get_rss_gb():.1f}GB")
        if hasattr(torch, 'npu') and torch.npu.is_available():
            devs = devices if devices else [ref_device, quant_device]
            for d in devs:
                mem = torch.npu.memory_allocated(d) / 1024**3
                logger.info(f"  [NPU MEM] {d}: {mem:.1f}GB")

    def _finalize_dual_report(self, ref_model, quant_model, ref_hidden_states,
                                quant_hidden_states, all_results, ref_reader,
                                quant_reader, layer_cos_sims, layer_inputs,
                                use_cpu_topk=False):
        """dual 模式收尾: close readers → cache cleanup → topk → logits → report。"""
        ref_reader.close()
        quant_reader.close()
        self._cache_top_k_cleanup(layer_cos_sims, layer_inputs, self.cache_top_k, self.verbose)
        report = BlockCompareReport()
        report.results = all_results
        if self.l1_target_layers is not None:
            logger.info(f"\n[Sharded L1] 目标层 {sorted(self.l1_target_layers)} 已全部缓存完毕。")
            logger.info(f"  现在可以运行 L2: --l2 --target_layers {' '.join(str(l) for l in sorted(self.l1_target_layers))}")
            return report
        if self.verbose:
            logger.info(f"\n[Sharded L1] 计算 top-k token 对齐{' (CPU)' if use_cpu_topk else ''}...")
        try:
            if use_cpu_topk:
                report.topk_result = self._compute_logits_topk_cpu(
                    ref_model, quant_model, ref_hidden_states, quant_hidden_states)
            else:
                report.topk_result = self._compute_logits_topk(
                    ref_model, quant_model, ref_hidden_states, quant_hidden_states,
                    self.actual_ref_device, self.actual_quant_device,
                )
            if self.verbose:
                logger.info(report.topk_result)
        except Exception as e:
            if self.verbose:
                logger.warning(f"top-k 对齐检查失败: {e}")
        try:
            report.logits_data = self._collect_full_logits(
                ref_model, quant_model, ref_hidden_states, quant_hidden_states)
        except Exception as e:
            if self.verbose:
                logger.warning(f"full logits 采集失败: {e}")
        return report

    def compare(self, prompt: str, layers_per_shard: int = 8, **kwargs) -> 'BlockCompareReport':
        tokenizer = self.tokenizer
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        return self.compare_ids(input_ids, layers_per_shard=layers_per_shard, **kwargs)

    def compare_ids(self, input_ids: Tensor, layers_per_shard: int = 8, **kwargs) -> 'BlockCompareReport':
        self._prompt_text = str(input_ids.shape[1]) + "_tokens"
        if self.compare_mode == "grouped_dual":
            return self._compare_grouped_dual(input_ids, layers_per_shard=layers_per_shard, **kwargs)
        return self._compare_dual_sharded(input_ids, layers_per_shard=layers_per_shard, **kwargs)
