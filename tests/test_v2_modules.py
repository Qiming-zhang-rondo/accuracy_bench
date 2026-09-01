"""
Smoke tests for v2 productized modules.

Covers: report_schema, validity_checks, logits_compare, badcase_workflow,
html_report (v2), report_data (assemble_report), inference_check API.

Run: python3 -m pytest tests/test_v2_modules.py -q
  or: python3 -c "import tests.conftest; ..." (stub runner)
"""
import json
import os
import tempfile

import pytest


# ---------------------------------------------------------------------------
# report_schema
# ---------------------------------------------------------------------------
class TestReportSchema:
    def test_reportdata_roundtrip(self):
        from accuracy_checker.report_schema import (
            ReportData, OverviewData, L1LayerData, L2LayerData, SubgraphData,
        )
        rd = ReportData(
            overview=OverviewData(
                model_name="GLM-5.1",
                quant_format="W4A8 MXFP4",
                prompt="hello",
                boundary_result="good",
            ),
            l1_layers=[
                L1LayerData(
                    layer_idx=11, cos_sim=0.965, rel_l2=0.089, snr=21.3,
                    is_first_divergence=True, is_max_error=True,
                ),
            ],
            l2_results=[
                L2LayerData(
                    layer_idx=11, base_l2=0.089, input_recovery=0.743,
                    root_suspect="attn.q_a_proj",
                    impact_boundary="attn.o_proj",
                    subgraphs=[
                        SubgraphData(
                            name="attn.q_a_proj",
                            patch_recovery=0.743,
                            self_rot_err=0.024,
                            rot_b_err=0.011,
                            flip_rate=0.0,
                        ),
                    ],
                ),
            ],
            run_mode="full",
        )
        j = rd.to_json()
        rd2 = ReportData.from_dict(json.loads(j))
        assert rd2.overview.model_name == "GLM-5.1"
        assert rd2.l1_layers[0].cos_sim == 0.965
        assert rd2.l2_results[0].root_suspect == "attn.q_a_proj"
        assert rd2.run_mode == "full"

    def test_nan_inf_sanitized(self):
        from accuracy_checker.report_schema import L1LayerData, ReportData
        import math
        ld = L1LayerData(layer_idx=0, cos_sim=float("nan"), rel_l2=float("inf"))
        # NaN/Inf sanitized at ReportData level (to_json), not per-field
        rd = ReportData(l1_layers=[ld])
        j = rd.to_json()
        d = json.loads(j)
        assert d["l1_layers"][0]["cos_sim"] is None
        assert d["l1_layers"][0]["rel_l2"] is None


# ---------------------------------------------------------------------------
# validity_checks
# ---------------------------------------------------------------------------
class TestValidityChecks:
    def test_run_status_values(self):
        from accuracy_checker.validity_checks import RunStatus
        assert RunStatus.SUCCESS.value == "SUCCESS"
        assert RunStatus.FAILED.value == "FAILED"
        assert RunStatus.PARTIAL.value == "PARTIAL"

    def test_check_item_valid(self):
        from accuracy_checker.validity_checks import CheckItem
        ci = CheckItem(name="nan_check", status="PASS", detail="no NaN")
        assert ci.status == "PASS"

    def test_check_item_invalid_status(self):
        from accuracy_checker.validity_checks import CheckItem
        with pytest.raises(ValueError, match="must be one of"):
            CheckItem(name="bad", status="BOGUS", detail="x")

    def test_aggregate_overall(self):
        from accuracy_checker.validity_checks import aggregate_overall, RunStatus
        # No modules -> INCONCLUSIVE (can't determine)
        assert aggregate_overall({}) == RunStatus.INCONCLUSIVE


# ---------------------------------------------------------------------------
# logits_compare
# ---------------------------------------------------------------------------
class TestLogitsCompare:
    def test_logitsdata_construction(self):
        from accuracy_checker.logits_compare import LogitsData
        import dataclasses
        ld = LogitsData(
            token_positions=[0, 1],
            token_wise_cos=[0.99, 0.98],
            token_wise_kl=[0.01, 0.02],
            token_wise_topk_overlap=[1.0, 0.9],
            token_wise_top1_match=[1, 1],
        )
        d = dataclasses.asdict(ld)
        assert d["token_positions"] == [0, 1]


# ---------------------------------------------------------------------------
# badcase_workflow
# ---------------------------------------------------------------------------
class TestBadCaseWorkflow:
    def test_manifest_construction(self):
        from accuracy_checker.badcase_workflow import BadCaseManifest
        m = BadCaseManifest(
            model="glm5",
            baseline_config="w8a8",
            badcase_config="w4a8",
            injected_layer=11,
            expected_root_cause="attn.q_proj",
        )
        d = m.to_dict()
        assert d["model"] == "glm5"
        assert d["injected_layer"] == 11


# ---------------------------------------------------------------------------
# html_report (v2)
# ---------------------------------------------------------------------------
class TestHtmlReportV2:
    def test_generate_product_html_report(self, tmp_path):
        from accuracy_checker.html_report import generate_product_html_report
        from accuracy_checker.report_schema import (
            ReportData, OverviewData, L1LayerData, L2LayerData, SubgraphData,
        )
        rd = ReportData(
            overview=OverviewData(
                model_name="test-model",
                quant_format="W8A8",
                prompt="hello",
                input_mode="messages",
                comparison_scope="weight_plus_activation_qdq",
                quant_method="dequantize",
                activation_quant_enabled=True,
                activation_quant_type="AUTO",
                activation_quant_backend="npu",
            ),
            l1_layers=[
                L1LayerData(layer_idx=5, cos_sim=0.95, rel_l2=0.1, snr=20.0),
            ],
            l2_results=[
                L2LayerData(
                    layer_idx=5, base_l2=0.1, input_recovery=0.6,
                    root_suspect="attn.q_proj",
                    impact_boundary="mlp.down",
                    subgraphs=[
                        SubgraphData(name="attn.q_proj", patch_recovery=0.6),
                    ],
                ),
            ],
        )
        out = str(tmp_path / "report.html")
        generate_product_html_report(rd, out)
        assert os.path.getsize(out) > 5000
        with open(out, encoding="utf-8") as f:
            html = f.read()
        assert "test-model" in html
        assert "q_proj" in html
        # Self-contained (no external deps; SVG namespace URI is allowed)
        assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")
        assert "https://" not in html
        # Responsive/print contracts: charts and long tables stay readable.
        assert "@media (max-width:680px)" in html
        assert "@media print" in html
        assert "chart-scroll" in html
        assert "table-scroll" in html
        assert "let curPos=-1" in html
        assert 'L.position_mode==="prompt_prefill"' in html
        assert "首个 Decode Token" in html
        assert "权重 + 激活 QDQ 联合仿真" in html
        assert "ACT QDQ ON · QUANT SIDE" in html
        assert "L2 口径独立" in html
        assert "Prompt prefill 各位置" in html
        assert "l2MetricsTable" in html
        assert "position-btn" in html
        assert 'id="overview-section"' in html
        assert 'id="badcase-section"' in html
        # Metric help is keyboard-focusable rather than a click-only span.
        assert "<button type='button' class='help'" in html
        assert 'aria-label="指标说明"' in html

    def test_v1_backward_compat(self, tmp_path):
        from accuracy_checker.html_report import generate_html_report
        out = str(tmp_path / "v1_report.html")
        generate_html_report(boundary_results=[], l2_results=[], output_path=out)
        assert os.path.getsize(out) > 1000

    # ----- generate_index_html (sidebar history) -----

    @staticmethod
    def _write_report(report_dir, rd, age_sec=0):
        """Write report_data.json and set mtime to (now - age_sec)."""
        import time
        report_dir.mkdir(parents=True, exist_ok=True)
        out = report_dir / "report_data.json"
        out.write_text(rd.to_json(), encoding="utf-8")
        if age_sec > 0:
            ts = time.time() - age_sec
            os.utime(str(out), (ts, ts))

    def test_generate_index_html_multiple_reports(self, tmp_path):
        """Sidebar history index: scan multiple report_data.json, embed all, sort newest-first."""
        import re
        import time
        from accuracy_checker.html_report import generate_index_html
        from accuracy_checker.report_schema import (
            ReportData, OverviewData, L1LayerData, InferenceCompareData,
        )
        # run_a: older, run_status=PARTIAL, no inference_compare
        rd_a = ReportData(
            overview=OverviewData(
                model_name="model-A", quant_format="W8A8", prompt="hi",
                first_divergence_layer=3,
            ),
            l1_layers=[L1LayerData(layer_idx=3, cos_sim=0.97, rel_l2=0.1, snr=15.0)],
            run_status="PARTIAL",
            run_mode="l1",
        )
        self._write_report(tmp_path / "run_a", rd_a, age_sec=100)
        # run_b: newer, run_status=SUCCESS, boundary=CLEAN
        rd_b = ReportData(
            overview=OverviewData(
                model_name="model-B", quant_format="W4A8", prompt="yo",
                boundary_result="CLEAN", first_divergence_layer=None,
            ),
            run_status="SUCCESS",
            run_mode="boundary",
            inference_compare=InferenceCompareData(
                prompt="yo", ref_output="hello", quant_output="hello",
                ref_tokens=["hello"], quant_tokens=["hello"],
                token_match_rate=1.0, exact_match=True,
                first_divergence_pos=None, max_new_tokens=1,
            ),
        )
        self._write_report(tmp_path / "run_b", rd_b, age_sec=0)

        out = str(tmp_path / "index.html")
        generate_index_html(str(tmp_path), out)
        assert os.path.exists(out)
        with open(out, encoding="utf-8") as f:
            html = f.read()

        # Self-contained (no external deps; SVG namespace URI is allowed)
        assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")
        assert "https://" not in html

        # Sidebar + reports data embedded
        assert "window.__REPORTS__" in html
        assert "window.__SIDEBAR__" in html
        m = re.search(r"id='reports-data'>(.*?)</script>", html, re.DOTALL)
        assert m, "reports-data script tag not found"
        payload = json.loads(m.group(1))
        assert len(payload) == 2
        # Newest first → model-B (just written) is index 0
        assert payload[0]["overview"]["model_name"] == "model-B"
        assert payload[1]["overview"]["model_name"] == "model-A"
        # Sidebar summaries present + ordered consistently
        m2 = re.search(r"id='sidebar-data'>(.*?)</script>", html, re.DOTALL)
        assert m2
        sb = json.loads(m2.group(1))
        assert sb[0]["model_name"] == "model-B"
        assert sb[0]["run_status"] == "SUCCESS"
        assert sb[0]["boundary"] == "CLEAN"
        assert sb[0]["token_rate"] == 1.0
        assert sb[0]["run_mode"] == "BOUNDARY"
        assert sb[0]["report_relpath"] == "run_b"
        assert sb[1]["model_name"] == "model-A"
        assert sb[1]["run_status"] == "PARTIAL"
        assert sb[1]["run_mode"] == "L1"
        assert sb[1]["report_relpath"] == "run_a"
        # First-divergence layer propagates into sidebar summary
        assert sb[1]["first_div"] == 3
        # __switchReport wiring
        assert "window.__switchReport" in html
        assert "renderSidebar" in html
        assert "si-badge mode" in html
        assert "__openHistoryMenu" in html
        assert "__deleteHistoryReport" in html
        assert "/__accuracy_bench__/delete-report" in html
        assert "右键删除" in html
        # Polish: listener-accumulation guard in boot()
        assert "__accInitDone" in html
        # Polish: mobile responsive media query
        assert "@media (max-width:720px)" in html

    def test_generate_index_html_inconclusive_badge(self, tmp_path):
        """INCONCLUSIVE runs (not SUCCESS/INVALID/PARTIAL) must map to 'warn' badge."""
        import re
        from accuracy_checker.html_report import generate_index_html
        from accuracy_checker.report_schema import ReportData, OverviewData
        rd = ReportData(
            overview=OverviewData(model_name="model-X", quant_format="W8A8", prompt="hi"),
            run_status="INCONCLUSIVE",
        )
        self._write_report(tmp_path / "run_x", rd, age_sec=0)
        out = str(tmp_path / "index.html")
        generate_index_html(str(tmp_path), out)
        with open(out, encoding="utf-8") as f:
            html = f.read()
        # The badge-class picker in renderSidebar must include INCONCLUSIVE -> warn
        assert 'st.indexOf("INCONCLUSIVE")>=0' in html

    def test_generate_index_html_empty_dir(self, tmp_path):
        """No report_data.json anywhere → minimal 'no reports' HTML."""
        from accuracy_checker.html_report import generate_index_html
        out = str(tmp_path / "idx_empty.html")
        generate_index_html(str(tmp_path), out)
        with open(out, encoding="utf-8") as f:
            html = f.read()
        assert "暂无历史报告" in html
        # No __REPORTS__ bootstrap (no JS crashes if user opens)
        assert "window.__REPORTS__" not in html


# ---------------------------------------------------------------------------
# report_data (assemble_report)
# ---------------------------------------------------------------------------
class TestAssembleReport:
    def test_assemble_empty(self):
        from accuracy_checker.report_data import assemble_report
        from accuracy_checker.report_schema import ReportData
        rd = assemble_report(
            l1_report=None,
            l2_results=[],
            model_name="test",
            prompt="hi",
        )
        assert isinstance(rd, ReportData)
        assert rd.overview.model_name == "test"


# ---------------------------------------------------------------------------
# inference_check API
# ---------------------------------------------------------------------------
class TestInferenceCheckAPI:
    def test_boundary_result_fields(self):
        from accuracy_checker.inference_check import BoundaryResult
        br = BoundaryResult(
            framework_name="vllm",
            framework_badcase_reproduced=True,
            transformers_badcase_reproduced=False,
            ref_badcase_reproduced=False,
            boundary_result="WEIGHT",
            evidence="transformers dequant can't reproduce",
            limitations="",
        )
        assert br.boundary_result == "WEIGHT"

    def test_boundary_result_to_dict(self):
        from accuracy_checker.inference_check import (
            BoundaryResult, boundary_result_to_dict,
        )
        br = BoundaryResult(
            framework_name="vllm",
            framework_badcase_reproduced=True,
            transformers_badcase_reproduced=False,
            ref_badcase_reproduced=False,
            boundary_result="WEIGHT",
            evidence="test",
            limitations="",
        )
        d = boundary_result_to_dict(br)
        assert d["boundary_result"] == "WEIGHT"
        assert d["framework_name"] == "vllm"

    def test_classify_boundary(self):
        from accuracy_checker.inference_check import classify_boundary
        # framework reproduces, transformers doesn't -> INFERENCE_FRAMEWORK
        verdict = classify_boundary(
            framework_reproduced=True,
            transformers_quant_reproduced=False,
            ref_reproduced=False,
        )
        assert isinstance(verdict, str) and len(verdict) > 0


# ---------------------------------------------------------------------------
# AlignmentReport (report.py)
# ---------------------------------------------------------------------------
class TestAlignmentReport:
    def test_run_status_empty(self):
        from accuracy_checker.report import AlignmentReport
        from accuracy_checker.validity_checks import RunStatus
        r = AlignmentReport()
        assert r.run_status() == RunStatus.FAILED

    def test_set_and_summary(self):
        from accuracy_checker.report import AlignmentReport
        r = AlignmentReport()
        s = r.summary()
        assert "量化精度对齐报告" in s
        assert "L0: 未执行" in s


# ---------------------------------------------------------------------------
# RootSuspect Priority 0: RotBErr ≈ boundary when SelfRotErr missing
# ---------------------------------------------------------------------------
class TestRootSuspectPriority0:
    """Test that o_proj (no SelfRotErr) is picked as RootSuspect when
    its RotBErr ≈ boundary RotBErr and all SelfRotErrs are low.

    Bug: o_proj not in _MLA_SELFROTERR_INPUT → no SelfRotErr → RootSuspect
    incorrectly picked q_a_proj (SErr 2.5%) instead of o_proj (RotBErr 91.9%).
    """

    def _select_root_suspect(self, rotberr, selfroterr, quant_types,
                             quantized_valid, impact_boundary,
                             mla_fine=True):
        """Replicate the RootSuspect selection logic from subgraph_locate.py."""
        coarse_names = {'self_attn'} if mla_fine else set()
        root_suspect = None

        # Priority 0
        if rotberr and impact_boundary is not None:
            root_suspect = self._priority0_select(
                rotberr, selfroterr, quant_types,
                impact_boundary, coarse_names)

        # Priority 1: SelfRotErr
        if root_suspect is None and selfroterr:
            non_float = {
                k: v for k, v in selfroterr.items()
                if v is not None and quant_types.get(k) != "FLOAT"
                and k not in coarse_names
            }
            if non_float:
                root_suspect = max(non_float, key=non_float.get)

        # Priority 2: RotBErr fallback
        if root_suspect is None and rotberr:
            non_gate = {
                k: v for k, v in rotberr.items()
                if v is not None and quant_types.get(k) != "FLOAT"
                and not k.endswith('.gate') and k not in coarse_names
            }
            if non_gate:
                root_suspect = max(non_gate, key=non_gate.get)

        return root_suspect

    @staticmethod
    def _priority0_select(rotberr, selfroterr, quant_types,
                          impact_boundary, coarse_names):
        """Priority 0: find op with RotBErr ≈ boundary but no SelfRotErr."""
        boundary_be = rotberr.get(impact_boundary)
        max_serr = max(
            (v for v in selfroterr.values() if v is not None),
            default=0,
        )
        if not (boundary_be is not None and max_serr < 0.10
                and boundary_be > 0.10
                and impact_boundary in coarse_names):
            return None
        for op_name, op_be in rotberr.items():
            if op_name in coarse_names or op_name == impact_boundary:
                continue
            if (op_be is not None and abs(op_be - boundary_be) < 0.02
                    and quant_types.get(op_name) != "FLOAT"
                    and not op_name.endswith('.gate')):
                has_serr = selfroterr.get(op_name) is not None
                if not has_serr:
                    return op_name
        return None

    def test_oproj_picked_when_selfroterr_low(self):
        """o_proj (RotBErr=91.9%, no SelfRotErr) should beat q_a_proj (SErr=2.5%)."""
        rotberr = {
            'self_attn': 0.919,
            'self_attn.q_a_proj': 0.414,
            'self_attn.q_b_proj': 0.439,
            'self_attn.kv_a_proj_with_mqa': 0.675,
            'self_attn.o_proj': 0.919,
        }
        selfroterr = {
            'self_attn.q_a_proj': 0.025,
            'self_attn.q_b_proj': 0.018,
            'self_attn.kv_a_proj_with_mqa': 0.022,
        }
        quant_types = {k: 'W8A8_MXFP8' for k in rotberr}
        quantized_valid = {'self_attn': 0.16, 'self_attn.o_proj': 0.16}

        result = self._select_root_suspect(
            rotberr, selfroterr, quant_types, quantized_valid,
            impact_boundary='self_attn',
        )
        assert result == 'self_attn.o_proj', (
            f"Expected o_proj, got {result}"
        )

    def test_qa_proj_picked_when_selfroterr_high(self):
        """When q_a_proj has high SelfRotErr (50%), it should win over o_proj."""
        rotberr = {
            'self_attn': 0.919,
            'self_attn.q_a_proj': 0.414,
            'self_attn.o_proj': 0.919,
        }
        selfroterr = {
            'self_attn.q_a_proj': 0.50,
        }
        quant_types = {k: 'W8A8_MXFP8' for k in rotberr}
        quantized_valid = {'self_attn': 0.16, 'self_attn.o_proj': 0.16}

        result = self._select_root_suspect(
            rotberr, selfroterr, quant_types, quantized_valid,
            impact_boundary='self_attn',
        )
        # max_serr=0.50 > 0.10 → Priority 0 doesn't fire → SelfRotErr wins
        assert result == 'self_attn.q_a_proj'

    def test_no_priority0_when_boundary_not_coarse(self):
        """Priority 0 only fires when boundary is coarse (self_attn)."""
        rotberr = {
            'mlp.shared_experts': 0.80,
            'mlp.experts': 0.80,
        }
        selfroterr = {}
        quant_types = {k: 'W8A8' for k in rotberr}
        quantized_valid = {'mlp.shared_experts': 0.50}

        result = self._select_root_suspect(
            rotberr, selfroterr, quant_types, quantized_valid,
            impact_boundary='mlp.shared_experts',
        )
        # boundary not in coarse_names → Priority 0 doesn't fire
        # → falls to RotBErr fallback → picks shared_experts (highest)
        assert result == 'mlp.shared_experts'
