"""
L1 Block 对比的数据类型与报告类 (从 layer1_block_compare.py 拆分)。

包含:
  - Delta 检测参数 (DELTA_WINDOW / DELTA_K / ...)
  - _layer_idx_from_name
  - TopKCompareResult / BlockCompareResult / LayerDeltaInfo / BadLayerDetection
  - BlockCompareReport (rolling-window delta + MAD 检测)
"""
from __future__ import annotations

import re
import statistics as stat
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .report_schema import LogitsData


# ============================================================================
# Delta 检测参数
# ============================================================================
DELTA_WINDOW = 10              # rolling window 大小
DELTA_K = 5                    # z/mad score 倍数
DELTA_MIN_DROP = 0.005         # 最小绝对下降 (0.5%)
PERSISTENCE_CHECK_LAYERS = 3   # 检查后续 N 层
PERSISTENCE_RECOVERY_TOL = 0.001  # 恢复到突降前 ±0.1% 内 = 已恢复
MAD_EPS = 1e-8                 # MAD 为 0 时的 epsilon


def _layer_idx_from_name(name: str, default: int = -1) -> int:
    """从 layer_name 提取 idx: 'layer.33.block_output' -> 33"""
    m = re.search(r'layer\.(\d+)', name or "")
    return int(m.group(1)) if m else default


@dataclass
class TopKCompareResult:
    """Top-K token 对齐结果"""
    top_k: int
    ref_topk_ids: List[int]
    quant_topk_ids: List[int]
    match_count: int
    logits_cos_sim: float
    top1_match: bool
    ref_entropy: float = 0.0
    quant_entropy: float = 0.0
    entropy_diff: float = 0.0
    kl_divergence: float = 0.0
    ref_topk_tokens: Optional[List[str]] = None
    quant_topk_tokens: Optional[List[str]] = None

    def __str__(self):
        ref_tokens = str(self.ref_topk_ids[:self.top_k])
        quant_tokens = str(self.quant_topk_ids[:self.top_k])
        match_str = f"{self.match_count}/{self.top_k}"
        top1 = "YES" if self.top1_match else "NO"
        result = (f"  Top-K Token: top-{self.top_k} match={match_str}, "
                f"top-1 match={top1}, logits_cos_sim={self.logits_cos_sim:.6f}\n"
                f"    ref_topk={ref_tokens}\n"
                f"    quant_topk={quant_tokens}\n"
                f"    ref_entropy={self.ref_entropy:.4f}, quant_entropy={self.quant_entropy:.4f}, "
                f"diff={self.entropy_diff:+.4f}\n"
                f"    kl_divergence={self.kl_divergence:.6f}")
        if self.ref_topk_tokens and self.quant_topk_tokens:
            result += f"\n    ref_tokens={[repr(t) for t in self.ref_topk_tokens[:self.top_k]]}"
            result += f"\n    quant_tokens={[repr(t) for t in self.quant_topk_tokens[:self.top_k]]}"
        return result


@dataclass
class BlockCompareResult:
    """单层对比结果"""
    layer_name: str
    metrics: Dict[str, float]

    @property
    def is_aligned(self) -> bool:
        """cos_sim > 0.99: 辅助告警, 不做根因定位"""
        return self.metrics.get("cos_sim", 0) > 0.99

    @property
    def is_suspicious(self) -> bool:
        return self.metrics.get("cos_sim", 0) < 0.99

    def __str__(self):
        status = "OK" if self.is_aligned else "BAD"
        cs = self.metrics.get("cos_sim", 0)
        snr_val = self.metrics.get("snr", 0)
        re = self.metrics.get("relative_error", 0)
        return f"[{status}] {self.layer_name}: cos_sim={cs:.6f}, snr={snr_val:.2f}dB, rel_err={re:.6f}"


@dataclass
class LayerDeltaInfo:
    """单层 delta 分析结果"""
    layer_name: str
    layer_idx: int
    cos_sim: float
    prev_cos_sim: Optional[float] = None
    delta_cos: Optional[float] = None        # cos_sim[i] - cos_sim[i-1], 负=下降
    drop_percent: Optional[float] = None     # delta_cos * 100
    baseline_delta: Optional[float] = None  # rolling median of deltas
    mad: Optional[float] = None              # median absolute deviation
    z_mad_score: Optional[float] = None      # (delta - baseline) / (mad + eps)
    is_statistical_jump: bool = False        # z_mad > k
    is_absolute_jump: bool = False           # delta < -min_drop
    is_persistent: bool = False              # 后续 N 层未恢复
    is_bad_jump: bool = False               # statistical AND absolute

    def debug_str(self) -> str:
        lines = [f"Layer {self.layer_idx} ({self.layer_name}):"]
        if self.prev_cos_sim is not None:
            lines.append(f"  cos_sim_prev = {self.prev_cos_sim:.6f}")
        lines.append(f"  cos_sim_curr = {self.cos_sim:.6f}")
        if self.delta_cos is not None:
            lines.append(f"  delta_cos    = {self.delta_cos:+.6f}")
            lines.append(f"  drop_percent = {self.drop_percent:+.4f}%")
        if self.baseline_delta is not None:
            lines.append(f"  baseline     = {self.baseline_delta:+.6f}")
            lines.append(f"  MAD          = {self.mad:.6f}")
            lines.append(f"  z/mad score  = {self.z_mad_score:.2f}")
        lines.append(f"  statistical  = {self.is_statistical_jump}")
        lines.append(f"  absolute     = {self.is_absolute_jump}")
        lines.append(f"  persistent   = {self.is_persistent}")
        lines.append(f"  detected_bad = {self.is_bad_jump}")
        return "\n".join(lines)


@dataclass
class BadLayerDetection:
    """first_bad_block 检测结果"""
    layer_deltas: List[LayerDeltaInfo] = field(default_factory=list)
    bad_layers: List[LayerDeltaInfo] = field(default_factory=list)
    first_bad: Optional[LayerDeltaInfo] = None
    first_threshold_crossing: Optional[str] = None   # 第一个 cos_sim < 0.99 的层名


@dataclass
class BlockCompareReport:
    """L1对比总报告"""
    results: List[BlockCompareResult] = field(default_factory=list)
    topk_result: Optional['TopKCompareResult'] = None
    # Full logits (last-N positions x full vocab) captured on the SAME L1 forward
    # pass. Populated by ShardedBlockComparator._collect_full_logits(); streamed
    # straight into assemble_report(logits_comparison=...) so the L1 panel and
    # the logits panel share one forward pass (no extra model reload).
    logits_data: Optional[LogitsData] = None
    # 缓存的检测结果 (首次调用 _detect_bad_layers 后填充)
    _detection_cache: Optional[BadLayerDetection] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # 核心检测: rolling-window delta + MAD
    # ------------------------------------------------------------------

    def _analyze_single_layer_delta(self, i, layer_results, cos_sims, deltas):
        """分析单层 delta: baseline/MAD/z-score/jump/persistence。"""
        r = layer_results[i]
        info = LayerDeltaInfo(
            layer_name=r.layer_name,
            layer_idx=_layer_idx_from_name(r.layer_name),
            cos_sim=cos_sims[i],
        )
        if i == 0:
            return info
        info.prev_cos_sim = cos_sims[i - 1]
        info.delta_cos = deltas[i - 1]
        info.drop_percent = deltas[i - 1] * 100
        hist_start = max(0, i - 1 - DELTA_WINDOW)
        history = deltas[hist_start:i - 1]
        if len(history) >= 2:
            info.baseline_delta = stat.median(history)
            abs_devs = [abs(d - info.baseline_delta) for d in history]
            info.mad = stat.median(abs_devs)
            denom = info.mad + MAD_EPS
            info.z_mad_score = (info.delta_cos - info.baseline_delta) / denom
            info.is_statistical_jump = info.z_mad_score < -DELTA_K
        else:
            info.is_statistical_jump = False
        info.is_absolute_jump = info.delta_cos < -DELTA_MIN_DROP
        info.is_bad_jump = info.is_statistical_jump and info.is_absolute_jump
        if info.is_bad_jump:
            info.is_persistent = self._check_persistence(i, cos_sims)
        return info

    @staticmethod
    def _check_persistence(i, cos_sims):
        """持续性检查: 后续 N 层 cos_sim 未恢复到突降前水平。"""
        pre_cos = cos_sims[i - 1]
        if i + 1 >= len(cos_sims):
            return True
        post_end = min(i + 1 + PERSISTENCE_CHECK_LAYERS, len(cos_sims))
        post_max = max(cos_sims[i + 1:post_end]) if post_end > i + 1 else cos_sims[i]
        return post_max < pre_cos - PERSISTENCE_RECOVERY_TOL

    def _detect_bad_layers(self) -> BadLayerDetection:
        """检测显著局部突降层。

        算法:
        1. delta_cos[layer] = cos_sim[layer] - cos_sim[layer-1]  (负=下降)
        2. 滚动基线: baseline = median(delta_cos[max(1,i-window):i])
           MAD = median(|delta_cos[...] - baseline|)
        3. 双条件:
           is_statistical_jump = delta < baseline - k * MAD
           is_absolute_jump    = delta < -min_drop
           is_bad_jump         = statistical AND absolute
        4. 持续性: 后 PERSISTENCE_CHECK_LAYERS 层 cos_sim 未恢复到突降前水平
        """
        if self._detection_cache is not None:
            return self._detection_cache

        layer_results = [r for r in self.results
                         if r.layer_name.startswith("layer.")]
        detection = BadLayerDetection()

        if len(layer_results) < 3:
            # 层太少, 只做绝对阈值
            for r in self.results:
                if not r.is_aligned:
                    detection.first_threshold_crossing = r.layer_name
                    break
            self._detection_cache = detection
            return detection

        # 提取 cos_sim 序列
        cos_sims = []
        for r in layer_results:
            cs = (r.metrics or {}).get("cos_sim")
            cos_sims.append(cs if cs is not None else 0.0)

        # 计算 delta_cos (负=下降)
        deltas = []
        for i in range(1, len(cos_sims)):
            deltas.append(cos_sims[i] - cos_sims[i - 1])

        # 逐层分析
        for i in range(len(layer_results)):
            info = self._analyze_single_layer_delta(
                i, layer_results, cos_sims, deltas)
            detection.layer_deltas.append(info)
            if info.is_bad_jump:
                detection.bad_layers.append(info)

        # 首个 bad layer: 优先 persistent, 否则取第一个 bad_jump
        persistent_bads = [b for b in detection.bad_layers if b.is_persistent]
        if persistent_bads:
            detection.first_bad = persistent_bads[0]
        elif detection.bad_layers:
            detection.first_bad = detection.bad_layers[0]

        # 绝对阈值首次跨过 (辅助信息, 不做根因定位)
        for r in layer_results:
            cs = (r.metrics or {}).get("cos_sim", 0)
            if cs < 0.99:
                detection.first_threshold_crossing = r.layer_name
                break

        self._detection_cache = detection
        return detection

    # ------------------------------------------------------------------
    # 公开属性
    # ------------------------------------------------------------------

    @property
    def first_bad_block(self) -> Optional[str]:
        """首个显著局部突降层 (delta + MAD 检测)。

        不用绝对阈值 0.99 — W4A8 逐层累积会导致中间层误报 (如 layer 33)。
        改用 rolling window + MAD 找真正的跳变点。

        回退链:
        1. 首个 is_bad_jump AND is_persistent 的层
        2. 首个 is_bad_jump 的层
        3. 首个 cos_sim < 0.99 的层 (绝对阈值, 仅兜底)
        """
        detection = self._detect_bad_layers()
        if detection.first_bad:
            return detection.first_bad.layer_name
        # Fallback: absolute threshold
        for r in self.results:
            if not r.is_aligned:
                return r.layer_name
        return None

    @property
    def first_threshold_crossing(self) -> Optional[str]:
        """首个 cos_sim < 0.99 的层 (辅助告警, 非根因定位)"""
        detection = self._detect_bad_layers()
        return detection.first_threshold_crossing

    @property
    def all_aligned(self) -> bool:
        return all(r.is_aligned for r in self.results)

    def summary(self) -> str:
        lines = [f"{'='*70}", "L1 逐Block对比", f"{'='*70}"]
        for r in self.results:
            lines.append(f"  {r}")
        lines.append(f"{'='*70}")
        if self.topk_result:
            lines.append(str(self.topk_result))
            lines.append(f"{'='*70}")
        if self.logits_data is not None:
            n_pos = len(self.logits_data.token_positions)
            n_cos = sum(1 for x in self.logits_data.token_wise_cos if x is not None)
            lines.append(f"  Full logits: {n_pos} positions captured ({n_cos} cos metrics), "
                        f"via L1 forward (compare_logits 4-panel data ready)")
            lines.append(f"{'='*70}")

        detection = self._detect_bad_layers()

        if self.all_aligned:
            lines.append("  全部对齐! 量化精度无损")
        else:
            # 绝对阈值跨过 (辅助)
            if detection.first_threshold_crossing:
                lines.append(f"  First threshold crossing (cos<0.99): {detection.first_threshold_crossing}")

            # 显著局部突降 (根因定位)
            if detection.first_bad:
                fb = detection.first_bad
                lines.append(f"  First significant local drop: {fb.layer_name}")
                lines.append(f"    delta={fb.delta_cos:+.6f} ({fb.drop_percent:+.4f}%), "
                            f"baseline={fb.baseline_delta:+.6f}, MAD={fb.mad:.6f}, "
                            f"z/mad={fb.z_mad_score:.2f}, persistent={fb.is_persistent}")
            else:
                lines.append("  无显著局部突降 (delta 检测未命中)")

            # Debug: 打印所有 bad jump 候选
            if detection.bad_layers:
                lines.append(f"  Bad jump candidates ({len(detection.bad_layers)}):")
                for b in detection.bad_layers:
                    lines.append(f"    {b.debug_str()}")

        return "\n".join(lines)

    def _find_cliff(self) -> str:
        """兼容旧接口, 现已整合到 summary() 的 delta 分析中"""
        detection = self._detect_bad_layers()
        if detection.first_bad:
            fb = detection.first_bad
            return (f"  断崖下降: {fb.layer_name}, delta={fb.delta_cos:+.6f} "
                    f"({fb.drop_percent:+.4f}%), persistent={fb.is_persistent}")
        return "  无明显断崖，精度渐进下降"


__all__ = [
    "DELTA_WINDOW", "DELTA_K", "DELTA_MIN_DROP",
    "PERSISTENCE_CHECK_LAYERS", "PERSISTENCE_RECOVERY_TOL", "MAD_EPS",
    "_layer_idx_from_name",
    "TopKCompareResult", "BlockCompareResult", "LayerDeltaInfo",
    "BadLayerDetection", "BlockCompareReport",
]
