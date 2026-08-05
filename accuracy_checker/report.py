"""
对齐报告生成

汇总 L0/Boundary/L1/L2/Logits/InferenceCompare 的检查结果, 生成可读报告。
AlignmentReport 是流程中各模块产物的内存聚合, 可被 report_data.assemble_report
消费转成统一 ReportData, 再由 generate_product_html_report 渲染。
"""

from typing import Optional, List, Any, Dict
from .layer1_block_compare import BlockCompareReport
from .validity_checks import RunStatus, aggregate_overall
import logging


logger = logging.getLogger(__name__)


class AlignmentReport:
    """
    精度对齐总报告 (内存聚合器)

    用法:
        report = AlignmentReport()
        report.set_l0(l0_result)            # L0SanityResult
        report.set_boundary(boundary_dict)  # inference_check.boundary_result_to_dict
        report.set_l1(block_report)         # BlockCompareReport
        report.add_l2(l2_dict)             # diagnose_layer dict
        report.set_logits(logits_cmp)       # LogitsComparison 或 LogitsData
        report.set_inference_compare(inf_cmp)  # InferenceCompareData 或 dict
        logger.info(report.summary())
    """

    def __init__(self):
        # L0
        self.l0: Optional[Any] = None  # L0SanityResult
        # Boundary (定界)
        self.boundary: Optional[Dict[str, Any]] = None  # boundary_result_to_dict 输出
        # L1
        self.l1: Optional[BlockCompareReport] = None
        # L2
        self.l2_reports: List[Any] = []
        # Logits
        self.logits: Optional[Any] = None  # LogitsComparison / LogitsData
        # Inference Compare
        self.inference_compare: Optional[Any] = None  # InferenceCompareData / dict

    # ---- setters ----
    def set_l0(self, l0_result: Any) -> None:
        self.l0 = l0_result

    def set_boundary(self, boundary_result: Any) -> None:
        """接受 BoundaryResult 或 boundary_result_to_dict 的 dict。"""
        if hasattr(boundary_result, "boundary_result"):
            # BoundaryResult dataclass -> 转 dict 便于统一处理
            try:
                from .inference_check import boundary_result_to_dict
                self.boundary = boundary_result_to_dict(boundary_result)
                return
            except Exception:  # noqa: BLE001
                pass
        self.boundary = boundary_result

    def set_l1(self, report: BlockCompareReport) -> None:
        self.l1 = report

    def add_l2(self, report: Any) -> None:
        self.l2_reports.append(report)

    def set_logits(self, logits: Any) -> None:
        self.logits = logits

    def set_inference_compare(self, inf_compare: Any) -> None:
        self.inference_compare = inf_compare

    # ---- run_status 推导 ----
    def run_status(self) -> RunStatus:
        """根据各模块结果推导整体运行状态 (与 report_schema._decide_status 同口径)。"""
        # L0 INVALID_RUN 直接判无效
        if self.l0 is not None:
            ov = getattr(self.l0, "overall_status", None)
            if ov == RunStatus.INVALID_RUN.value:
                return RunStatus.INVALID_RUN
            if ov == RunStatus.FAILED.value:
                return RunStatus.FAILED
        # boundary 乱码 + L1 全对齐 -> 不可信结论
        bnd_ok = self._boundary_clean()
        if self.l1 is not None:
            results = getattr(self.l1, "results", None) or []
            if results and all(
                (getattr(r, "metrics", {}).get("cos_sim") or 0) > 0.99 for r in results
            ) and self.boundary is not None and not bnd_ok:
                return RunStatus.INCONCLUSIVE
        # 缺模块 -> PARTIAL
        have_modules = [self.l1 is not None, bool(self.l2_reports),
                        self.boundary is not None]
        if all(have_modules):
            return RunStatus.SUCCESS
        if not any(have_modules) and self.l0 is None:
            return RunStatus.FAILED
        return RunStatus.PARTIAL

    def _boundary_clean(self) -> bool:
        """boundary 是否判定为干净 (非 GARBLED/TRUNCATED)。无 boundary -> True。"""
        if not self.boundary:
            return True
        if isinstance(self.boundary, dict):
            verdict = self.boundary.get("boundary_result")
        else:
            verdict = getattr(self.boundary, "boundary_result", None)
        if verdict in ("GARBLED", "TRUNCATED"):
            return False
        return True

    # ---- summary ----
    def _summarize_l0(self, lines):
        if self.l0 is not None:
            lines.append("")
            lines.append(f"  L0 预检: {getattr(self.l0, 'summary', self.l0)}")
        else:
            lines.append("\n  L0: 未执行")

    def _summarize_boundary(self, lines):
        if self.boundary is None:
            lines.append("\n  Boundary: 未执行")
            return
        b = self.boundary if isinstance(self.boundary, dict) else {}
        verdict = b.get("boundary_result", "?")
        fw = b.get("framework_name", "")
        fw_rep = b.get("framework_badcase_reproduced")
        tf_rep = b.get("transformers_badcase_reproduced")
        lines.append("")
        lines.append(f"  Boundary 定界: {verdict}"
                     f" (framework={fw or 'n/a'}, framework_repro={fw_rep}, "
                     f"transformers_quant_repro={tf_rep})")

    def _summarize_l1(self, lines):
        if self.l1 is not None:
            lines.append("")
            lines.append(self.l1.summary())
        else:
            lines.append("\n  L1: 未执行")

    def _summarize_l2(self, lines):
        if self.l2_reports:
            from .subgraph_locate import print_report
            print_report(self.l2_reports)
        else:
            lines.append("\n  L2: 未执行")

    def _summarize_logits(self, lines):
        if self.logits is not None:
            n = len(getattr(self.logits, "token_positions", []) or [])
            lines.append(f"\n  Logits 对比: 采集 {n} 个生成位置")
        else:
            lines.append("\n  Logits: 未执行")

    def _summarize_inference_compare(self, lines):
        if self.inference_compare is None:
            lines.append("\n  推理对比: 未执行")
            return
        ic = self.inference_compare
        rate = getattr(ic, "token_match_rate", None) if not isinstance(ic, dict) else ic.get("token_match_rate")
        exact = getattr(ic, "exact_match", None) if not isinstance(ic, dict) else ic.get("exact_match")
        lines.append(f"\n  推理对比: token_match_rate={rate}, exact_match={exact}")

    def _summarize_conclusion(self, lines, status):
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"  结论 (run_status={status.value})")
        lines.append("=" * 70)
        if self.l1 and self.l1.all_aligned:
            lines.append("  L1全部对齐，量化精度无损")
            return
        if not (self.l1 and self.l1.first_bad_block):
            return
        fbb = self.l1.first_bad_block
        lines.append(f"  L1发现精度问题: first bad block = {fbb}")
        if self.l2_reports:
            for l2 in self.l2_reports:
                if isinstance(l2, dict):
                    layer_idx = l2.get('layer_idx', '?')
                    dominant = l2.get('dominant_subgraph', 'unknown')
                    lines.append(f"    Layer {layer_idx} 主要问题子图: {dominant}")
                elif hasattr(l2, 'layer_idx') and hasattr(l2, 'bad_modules'):
                    lines.append(f"    Layer {l2.layer_idx} 问题算子: {l2.bad_modules}")
        else:
            lines.append(f"  建议对 {fbb} 执行L2逐算子对比，定位具体问题算子")

    def summary(self) -> str:
        lines = []
        lines.append("")
        lines.append("=" * 70)
        lines.append("  量化精度对齐报告")
        lines.append("=" * 70)

        status = self.run_status()

        self._summarize_l0(lines)
        self._summarize_boundary(lines)
        self._summarize_l1(lines)
        self._summarize_l2(lines)
        self._summarize_logits(lines)
        self._summarize_inference_compare(lines)
        self._summarize_conclusion(lines, status)

        lines.append("")
        return "\n".join(lines)


__all__ = ["AlignmentReport"]
