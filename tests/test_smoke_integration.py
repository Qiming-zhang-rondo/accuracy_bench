"""
工作项: 集成 smoke test — 验证 Agent B/C/D 各模块公开 API 端到端可用。

纯 CPU / mock 数据, 不依赖 NPU / 大模型:
  1. L0 sanity 基本流程 (路径非法 → 结构化结果)
  2. ValidityChecker 各项检查 (nan_inf / input_ids / meta_residual / weight_device_dtype)
  3. report_schema 序列化 ↔ 反序列化 roundtrip
  4. report_data.assemble_report (mock L1/L2/boundary → ReportData)
  5. html_report.generate_product_html_report (mock ReportData → 自包含 HTML)
  6. logits_compare.compare_logits 核心对比
  7. badcase_workflow.compare_with_ground_truth 命中 / 未命中

运行: python3 tests/run_tests.py tests/test_smoke_integration.py -v
"""

import math
import os
import types

import pytest
import torch
import torch.nn as nn

from accuracy_checker import (
    run_l0_sanity, L0SanityResult,
    ValidityChecker, RunStatus, CheckItem, aggregate_overall,
    ReportData, OverviewData, L1LayerData, L2LayerData, SubgraphData,
    LogitsData, InferenceCompareData, assemble_report, TokenProb,
    generate_product_html_report,
    LogitsCollection, compare_logits,
    BadCaseManifest, GroundTruthComparison, compare_with_ground_truth,
)
from accuracy_checker.layer1_block_compare import BlockCompareReport, BlockCompareResult


# ===========================================================================
# 1. L0 sanity 基本流程
# ===========================================================================

class TestL0Sanity:

    def test_invalid_path_returns_invalid_run(self):
        """非法路径 → L0SanityResult.overall_status == INVALID_RUN, 结构可序列化。"""
        result = run_l0_sanity(
            ref_model_path="/nonexistent/ref-model-smoke",
            quant_model_path="/nonexistent/quant-model-smoke",
        )
        assert isinstance(result, L0SanityResult)
        assert result.overall_status == RunStatus.INVALID_RUN.value
        assert len(result.checks) > 0
        # 路径检查应至少有一项 FAIL
        statuses = [c.status for c in result.checks]
        assert "FAIL" in statuses
        # to_dict 可序列化 (无异常即过)
        d = result.to_dict()
        assert "checks" in d and "overall_status" in d

    def test_summary_contains_counts(self, tmp_path):
        """summary 字符串含 PASS/WARN/FAIL/SKIP 计数。"""
        result = run_l0_sanity("/nope/ref", "/nope/quant")
        assert "PASS=" in result.summary or "FAIL=" in result.summary


# ===========================================================================
# 2. ValidityChecker 各项检查
# ===========================================================================

class TestValidityChecks:

    def test_nan_inf_detects_nan(self):
        vc = ValidityChecker()
        bad = torch.tensor([1.0, float("nan"), 2.0])
        item = vc.check_nan_inf(bad, "test")
        assert item.status == "FAIL"
        assert "NaN" in item.detail or "Inf" in item.detail

    def test_nan_inf_clean_passes(self):
        vc = ValidityChecker()
        item = vc.check_nan_inf(torch.randn(8), "clean")
        assert item.status == "PASS"

    def test_input_ids_consistency_identical(self):
        vc = ValidityChecker()
        ids = torch.tensor([[10, 20, 30, 40]])
        item = vc.check_input_ids_consistency(ids, ids.clone())
        assert item.status == "PASS"

    def test_input_ids_consistency_different(self):
        vc = ValidityChecker()
        a = torch.tensor([[1, 2, 3]])
        b = torch.tensor([[1, 2, 9]])
        item = vc.check_input_ids_consistency(a, b)
        assert item.status == "FAIL"

    def test_meta_residual_clean(self):
        vc = ValidityChecker()
        m = nn.Linear(3, 3)  # cpu, 无 meta
        item = vc.check_meta_residual(m)
        assert item.status == "PASS"

    def test_meta_residual_detects_meta(self):
        vc = ValidityChecker()
        m = nn.Linear(3, 3).to("meta")  # 参数仍在 meta
        item = vc.check_meta_residual(m)
        assert item.status == "FAIL"

    def test_weight_device_dtype_ok(self):
        vc = ValidityChecker()
        m = nn.Linear(3, 3, dtype=torch.float32)
        item = vc.check_weight_device_dtype(m, expected_device="cpu",
                                            expected_dtype=torch.float32)
        assert item.status in ("PASS", "WARN")  # PASS 优先

    def test_aggregate_overall_logic(self):
        # 全 PASS -> SUCCESS
        items = [CheckItem("a", "PASS", ""), CheckItem("b", "PASS", "")]
        assert aggregate_overall(items) == RunStatus.SUCCESS
        # 一个 FAIL -> INVALID_RUN
        items.append(CheckItem("c", "FAIL", ""))
        assert aggregate_overall(items) == RunStatus.INVALID_RUN


# ===========================================================================
# 3. report_schema 序列化 ↔ 反序列化
# ===========================================================================

class TestReportSchema:

    def test_roundtrip_preserves_fields(self):
        rd = ReportData(
            overview=OverviewData(model_name="glm-smoke",
                                  quant_format="W4A8_MXFP4",
                                  input_mode="messages",
                                  comparison_scope="weight_plus_activation_qdq",
                                  quant_method="dequantize",
                                  activation_quant_enabled=True,
                                  activation_quant_type="W4A4_INT4_PER_GROUP",
                                  activation_quant_backend="npu",
                                  activation_quant_group_size=128,
                                  first_divergence_layer=7,
                                  boundary_result="CLEAN"),
            l1_layers=[L1LayerData(layer_idx=7, layer_name="layer.7",
                                   cos_sim=0.42, is_first_divergence=True)],
            l2_results=[L2LayerData(layer_idx=7, impact_boundary="mlp",
                                    root_suspect="self_attn")],
            run_status="PARTIAL",
        )
        d = rd.to_dict()
        rd2 = ReportData.from_dict(d)
        assert rd2.overview.model_name == "glm-smoke"
        assert rd2.overview.first_divergence_layer == 7
        assert rd2.overview.input_mode == "messages"
        assert rd2.overview.comparison_scope == "weight_plus_activation_qdq"
        assert rd2.overview.activation_quant_enabled is True
        assert rd2.overview.activation_quant_type == "W4A4_INT4_PER_GROUP"
        assert rd2.overview.activation_quant_group_size == 128
        assert rd2.l1_layers[0].cos_sim == 0.42
        assert rd2.l1_layers[0].is_first_divergence is True
        assert rd2.l2_results[0].impact_boundary == "mlp"
        assert rd2.run_status == "PARTIAL"

    def test_nan_becomes_none_in_dict(self):
        """NaN 必须被 _clean_num 转成 None (JSON 合法)。"""
        rd = ReportData(l1_layers=[L1LayerData(layer_idx=0, cos_sim=float("nan"))])
        d = rd.to_dict()
        assert d["l1_layers"][0]["cos_sim"] is None

    def test_to_json_is_valid_json(self):
        rd = ReportData(overview=OverviewData(model_name="m"))
        s = rd.to_json()
        import json
        json.loads(s)  # 不抛异常即合法


# ===========================================================================
# 4. report_data.assemble_report
# ===========================================================================

def _mock_l1():
    """3 层 L1: 前两层对齐, 第 7 层发散 (first_bad)。"""
    return BlockCompareReport(results=[
        BlockCompareResult(layer_name="layer.0", metrics={"cos_sim": 0.999, "snr": 40, "relative_error": 0.001}),
        BlockCompareResult(layer_name="layer.1", metrics={"cos_sim": 0.998, "snr": 38, "relative_error": 0.002}),
        BlockCompareResult(layer_name="layer.7", metrics={"cos_sim": 0.42, "snr": 6, "relative_error": 0.31}),
    ])


def _mock_l2_results():
    return [{
        "layer_idx": 7,
        "baseline_l2": 0.31,
        "input_recovery": 0.62,
        "subgraphs": {"self_attn": 0.15, "mlp": 0.88},
        "subgraph_quant_types": {"self_attn": "INT8", "mlp": "INT8"},
        "subgraph_selfroterr": {"self_attn": 0.45, "mlp": 0.05},
        "subgraph_rotberr": {"self_attn": 0.50, "mlp": 0.08},
        "impact_boundary": "mlp",
        "root_suspect": "self_attn",
        "chain_deltas": {"attn_chain": {"q_proj": 0.1, "k_proj": 0.05}},
        "indexer_flip_rate": 0.02,
    }]


def _mock_boundary_clean():
    return [{
        "messages": [{"role": "user", "content": "你好"}],
        "generated": "你好，我是模型，很高兴为你服务。",
        "thinking_truncated": False,
        "output_tokens": 10,
    }]


class TestAssembleReport:

    def test_full_assemble_links_fields(self):
        rd = assemble_report(
            l1_report=_mock_l1(),
            l2_results=_mock_l2_results(),
            boundary_result=_mock_boundary_clean(),
            model_name="glm-smoke",
            ref_model_path="/ref", quant_model_path="/quant",
            quant_format="W4A8_MXFP4", device_mode="NPU",
            prompt="你好",
        )
        # overview 链路: first_divergence -> L2 at that layer
        assert rd.overview.first_divergence_layer == 7
        assert rd.overview.source_candidate == "self_attn"
        assert rd.overview.best_repair_point == "mlp"
        assert rd.overview.problem_path is not None
        assert "7" in rd.overview.problem_path
        # L1 3 层
        assert len(rd.l1_layers) == 3
        assert rd.l1_layers[2].is_first_divergence is True  # layer.7
        # L2 subgraphs 合并
        assert len(rd.l2_results) == 1
        sub = {s.name: s for s in rd.l2_results[0].subgraphs}
        assert "self_attn" in sub and "mlp" in sub
        assert sub["mlp"].is_repair_point is True   # impact_boundary
        assert sub["self_attn"].is_source_candidate is True  # root_suspect
        # boundary clean
        assert rd.overview.boundary_result == "CLEAN"
        # status: L1 发散 + boundary clean -> INCONCLUSIVE
        assert rd.run_status == "INCONCLUSIVE"

    def test_assemble_minimal_none_inputs(self):
        """全部 None 输入也不崩溃, run_status 合理。"""
        rd = assemble_report()
        assert rd.overview.model_name == ""
        assert len(rd.l1_layers) == 0
        assert rd.run_status in ("PARTIAL", "INCONCLUSIVE")

    def test_assemble_records_activation_comparison_scope(self):
        rd = assemble_report(
            l1_report=_mock_l1(),
            quant_method="dequantize",
            activation_quant_enabled=True,
            activation_quant_type="AUTO",
            activation_quant_backend="npu",
            input_mode="messages",
        )
        assert rd.overview.comparison_scope == "weight_plus_activation_qdq"
        assert rd.overview.activation_quant_enabled is True
        assert rd.overview.activation_quant_type == "AUTO"
        assert rd.overview.activation_quant_backend == "npu"
        assert rd.overview.input_mode == "messages"

    def test_assemble_records_int4_per_group_size(self):
        rd = assemble_report(
            l1_report=_mock_l1(),
            activation_quant_enabled=True,
            activation_quant_type="W4A4_INT4_PER_GROUP",
            activation_quant_backend="npu",
            activation_quant_group_size=64,
        )
        assert rd.overview.activation_quant_group_size == 64

    def test_old_report_scope_stays_unknown(self):
        rd = assemble_report(l1_report=_mock_l1())
        assert rd.overview.comparison_scope == "unknown"
        assert rd.overview.activation_quant_enabled is None


class TestL1L2CacheIdentity:

    def test_messages_identity_is_canonical_and_namespaced(self):
        from run_accuracy_check import _cache_input_identity

        left = types.SimpleNamespace(
            messages='[{"role":"user","content":"你好"}]',
            prompt=None,
        )
        right = types.SimpleNamespace(
            messages='[ { "content": "你好", "role": "user" } ]',
            prompt=None,
        )
        prompt = types.SimpleNamespace(messages=None, prompt="你好")

        assert _cache_input_identity(left) == _cache_input_identity(right)
        assert _cache_input_identity(left).startswith("messages:")
        assert _cache_input_identity(prompt) == "prompt:你好"
        assert _cache_input_identity(left) != _cache_input_identity(prompt)

    def test_compare_uses_sample_identity_instead_of_only_token_count(self):
        from accuracy_checker.layer1_block_compare import ShardedBlockComparator

        class Tokenizer:
            @staticmethod
            def encode(prompt, return_tensors=None):
                return torch.tensor([[1, 2, 3]])

        comparator = object.__new__(ShardedBlockComparator)
        comparator.tokenizer = Tokenizer()
        comparator.compare_mode = "grouped_dual"

        def capture_cache_prompt(self, input_ids, layers_per_shard=8, **kwargs):
            return self._prompt_text

        comparator._compare_grouped_dual = types.MethodType(
            capture_cache_prompt, comparator
        )

        assert comparator.compare("hello") == "hello"
        assert comparator.compare(
            "hello", cache_prompt="prompt:hello"
        ) == "prompt:hello"
        assert comparator.compare_ids(
            torch.tensor([[1, 2, 3]]), cache_prompt="messages:[]"
        ) == "messages:[]"
        assert comparator.compare_ids(torch.tensor([[1, 2, 3]])) == "3_tokens"

    def test_ambiguous_cross_prompt_cache_is_rejected(self, tmp_path, monkeypatch):
        from accuracy_checker.cache import CACHE_FORMAT_VERSION, INT4_UNPACK_VERSION
        from accuracy_checker.subgraph_locate import _try_cache_match

        model_hash = "deadbeef"
        for prompt_hash in ("11111111", "22222222"):
            name = (
                f"{model_hash}_{prompt_hash}_s3_L7_ref_"
                f"{CACHE_FORMAT_VERSION}_{INT4_UNPACK_VERSION}_dequantize.pt"
            )
            (tmp_path / name).touch()

        loaded = []
        monkeypatch.setattr(torch, "load", lambda *args, **kwargs: loaded.append(args))
        result = _try_cache_match(
            str(tmp_path) + os.sep,
            model_hash,
            "ffffffff",
            7,
            "ref",
            "dequantize",
            "cpu",
        )
        assert result is None
        assert loaded == []


# ===========================================================================
# 5. generate_product_html_report (mock ReportData)
# ===========================================================================

class TestProductHtmlReport:

    def test_generates_self_contained_html(self, tmp_path):
        rd = assemble_report(
            l1_report=_mock_l1(),
            l2_results=_mock_l2_results(),
            boundary_result=_mock_boundary_clean(),
            model_name="glm-smoke-html",
            prompt="你好",
        )
        out = str(tmp_path / "product_mock.html")
        path = generate_product_html_report(rd, output_path=out)
        assert path == out
        assert os.path.exists(out)
        content = open(out, encoding="utf-8").read()
        # 自包含: 无外网依赖 (SVG namespace URI 除外)
        assert "https://" not in content
        assert "http://" not in content.replace("http://www.w3.org/2000/svg", "")
        # 五大 section 容器存在
        for sec in ('id="overview"', 'id="l1"', 'id="l2"',
                    'id="logits"', 'id="badcase"'):
            assert sec in content, f"missing section {sec}"
        # 报告 JSON 嵌入
        assert "application/json" in content
        assert "window.__REPORT__" in content

    def test_none_logits_badcase_still_renders(self, tmp_path):
        rd = ReportData(overview=OverviewData(model_name="minimal-mock"))
        out = str(tmp_path / "minimal.html")
        generate_product_html_report(rd, output_path=out)
        assert os.path.getsize(out) > 0


# ===========================================================================
# 6. logits_compare 核心对比
# ===========================================================================

class _FakeTokenizer:
    """compare_logits 仅用到 .decode。"""
    def decode(self, ids, **kw):
        return f"tok{ids[0]}"


class TestLogitsCompare:

    def _collection(self, seed, n=3, vocab=40):
        torch.manual_seed(seed)
        return LogitsCollection(
            token_positions=list(range(n)),
            logits=torch.randn(n, vocab),
        )

    def test_compare_produces_metrics(self):
        ref = self._collection(0)
        quant = self._collection(1)
        comp = compare_logits(ref, quant, _FakeTokenizer(), top_k=5,
                              scatter_sample=50, hist_bins=10)
        # 维度一致
        assert comp.token_positions == [0, 1, 2]
        assert len(comp.token_wise_cos) == 3
        assert len(comp.token_wise_kl) == 3
        assert len(comp.ref_topk) == 3 and len(comp.quant_topk) == 3
        # 散点 / 直方图采样非空
        assert len(comp.scatter_ref) == len(comp.scatter_quant)
        assert len(comp.hist_bins) == 11  # bins+1 边界
        assert len(comp.hist_ref_counts) == 10
        # to_logits_data 转换可消费
        ld = comp.to_logits_data()
        assert isinstance(ld, LogitsData)
        assert ld.token_positions == [0, 1, 2]
        assert ld.position_mode == "generation"

    def test_compare_preserves_prompt_positions_and_mode(self):
        ref = LogitsCollection(
            token_positions=[7, 8, 9],
            logits=torch.randn(3, 20),
            position_mode="prompt_prefill",
        )
        quant = LogitsCollection(
            token_positions=[7, 8, 9],
            logits=torch.randn(3, 20),
            position_mode="prompt_prefill",
        )

        comp = compare_logits(ref, quant, _FakeTokenizer(), top_k=5)

        assert comp.token_positions == [7, 8, 9]
        assert comp.position_mode == "prompt_prefill"

    def test_identical_logits_top1_match(self):
        lc = self._collection(42)
        comp = compare_logits(lc, lc.clone_like() if hasattr(lc, "clone_like")
                              else LogitsCollection(lc.token_positions, lc.logits.clone()),
                              _FakeTokenizer(), top_k=5, scatter_sample=20)
        assert all(comp.token_wise_top1_match)
        # 完全相同 -> cos≈1
        assert comp.token_wise_cos[0] > 0.999


# ===========================================================================
# 6.5 ShardedBlockComparator._collect_full_logits + BlockCompareReport.logits_data
# ===========================================================================

class _MockNorm(nn.Module):
    """Tiny RMSNorm stand-in that records whether its real forward was used."""
    def __init__(self, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.calls = 0

    def forward(self, hidden_states):
        self.calls += 1
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + 1e-6)
        return (normalized * self.weight.float()).to(hidden_states.dtype)


class _MockLmHead(nn.Module):
    def __init__(self, hidden, vocab):
        super().__init__()
        # weight shape (vocab, hidden) — matches nn.Linear(in_features=hidden,
        # out_features=vocab).weight so F.linear(x, w) -> x @ w.T produces logits.
        self.weight = nn.Parameter(torch.randn(vocab, hidden) * 0.1)


class _MockModel(nn.Module):
    """Bare bones skeleton with norm + lm_head so get_norm_module/get_lm_head_module resolve."""
    def __init__(self, hidden, vocab):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity()])
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.norm = _MockNorm(hidden)
        self.lm_head = _MockLmHead(hidden, vocab)


class TestCollectFullLogits:
    """UT for the L1 → full-logits integration (the team-lead-requested feature).

    Verifies:
      * BlockCompareReport.logits_data field defaults to None and accepts a LogitsData.
      * _collect_full_logits produces a non-empty LogitsData from arbitrary
        ref/quant hidden_states without any disk I/O (load_layer_weights_indexed is
        patched to a no-op since the mock model already ships weights in-memory).
      * The captured positions equal logits_max_positions (or seq_len if shorter).
      * The returned LogitsData round-trips through assemble_report(logits_comparison=...).
    """

    def _make_comparator(self, hidden=8, vocab=12, max_positions=3):
        """Build a ShardedBlockComparator without calling __init__ heavy parts."""
        from accuracy_checker.layer1_block_compare import ShardedBlockComparator
        # Use object.__new__ to bypass filesystem-touching __init__
        cmp = object.__new__(ShardedBlockComparator)
        cmp.tokenizer = _FakeTokenizer()
        cmp.dtype = torch.float32
        cmp.verbose = False
        cmp.collect_full_logits = True
        cmp.logits_max_positions = max_positions
        cmp.ref_model_path = "/fake/ref"
        cmp.quant_model_path = "/fake/quant"
        cmp._ref_is_quant = False
        cmp._quant_is_quant = False
        cmp._ref_is_ct = False
        cmp._quant_is_ct = False
        cmp._ref_quant_desc = None
        cmp._quant_quant_desc = None
        cmp._ref_weight_map = {}
        cmp._quant_weight_map = {}
        cmp._ref_reader = None
        cmp._quant_reader = None
        cmp.R = None
        cmp.hidden = hidden
        cmp.vocab = vocab
        return cmp

    def test_block_compare_report_logits_data_field(self):
        """BlockCompareReport.logits_data defaults to None, accepts a LogitsData."""
        # default None
        r = BlockCompareReport()
        assert r.logits_data is None
        # field can be assigned a LogitsData
        ld = LogitsData(token_positions=[0, 1], ref_logits=[1.0, 2.0])
        r2 = BlockCompareReport(logits_data=ld)
        assert r2.logits_data is ld
        # summary() surfaces the new field when present
        s = r2.summary()
        assert "Full logits" in s and "2 positions" in s
        assert "对比口径: 未记录" in s
        joint = BlockCompareReport(
            comparison_scope="weight_plus_activation_qdq",
            quant_method="dequantize",
            activation_quant_enabled=True,
            activation_quant_type="AUTO",
            activation_quant_backend="npu",
        )
        joint_summary = joint.summary()
        assert "权重 + 激活 QDQ 联合仿真" in joint_summary
        assert "仅 Quant 侧启用" in joint_summary
        # summary is also valid when absent (no KeyError)
        _ = BlockCompareReport().summary()

    def test_collect_full_logits_produces_logits_data(self, monkeypatch):
        """End-to-end: hidden_states -> _collect_full_logits -> LogitsData."""
        # Patch the lazy import path so no disk I/O is needed
        import accuracy_checker.layer1_block_compare as L1
        from accuracy_checker.model_loader import load_layer_weights_indexed as _real_load

        def _no_op_load(*a, **kw):
            return None

        # Patch via module attribute on model_loader (lazy import inside method looks it up there)
        import accuracy_checker.model_loader as ML
        monkeypatch.setattr(ML, "load_layer_weights_indexed", _no_op_load)

        # Also patch model_loader reference reachable via `from .model_loader import ...`
        # The method does `from .model_loader import load_layer_weights_indexed`,
        # which Python evaluates at runtime — so the patched module attr is what's bound.
        cmp = self._make_comparator(hidden=8, vocab=12, max_positions=3)
        ref_model = _MockModel(8, 12)
        quant_model = _MockModel(8, 12)

        # hidden_states: batch=1, seq=5, hidden=8
        torch.manual_seed(0)
        ref_hs = torch.randn(1, 5, 8)
        quant_hs = torch.randn(1, 5, 8)

        ld = cmp._collect_full_logits(ref_model, quant_model, ref_hs, quant_hs)
        # sanity: returned a LogitsData with N positions (capped to max_positions=3)
        assert isinstance(ld, LogitsData)
        assert ld.token_positions == [2, 3, 4]   # absolute last 3 positions of seq=5
        assert ld.position_mode == "prompt_prefill"
        assert len(ld.token_wise_cos) == 3
        assert len(ld.ref_topk) == 3 and len(ld.quant_topk) == 3
        # scatter + histogram data populated (compare_logits contract)
        assert len(ld.scatter_ref) == len(ld.scatter_quant)
        assert len(ld.hist_bins) > 0
        # norm/lm_head unloaded back to meta after collection (consistent with topk_cpu)
        assert next(ref_model.norm.parameters()).device.type == 'meta'
        assert next(ref_model.lm_head.parameters()).device.type == 'meta'

    def test_tied_lm_head_is_restored_from_embedding(self):
        """Missing lm_head key must use this checkpoint's embedding, not an empty head."""
        cmp = self._make_comparator(hidden=8, vocab=12)
        model = _MockModel(8, 12)
        model.lm_head.weight = model.embed_tokens.weight
        expected = model.embed_tokens.weight.detach().clone()
        weight_map = {"model.embed_tokens.weight": "model.safetensors"}

        restored = cmp._restore_tied_lm_head(
            model, weight_map, "quant", torch.device("cpu"))

        assert restored is True
        assert torch.allclose(model.lm_head.weight, expected)
        assert model.lm_head.weight is not model.embed_tokens.weight
        assert model.lm_head.weight.abs().max().item() > 0

    def test_strict_final_load_skips_unrelated_meta_norm(self):
        """Kimi final logits must not assign CPU data to another meta norm."""
        from accuracy_checker.model_loader import load_layer_weights_indexed

        class _Reader:
            def __init__(self, tensors):
                self.tensors = tensors
                self.weight_map = {
                    name: "model.safetensors" for name in tensors
                }

            def get_tensor(self, name):
                return self.tensors.get(name)

        model = _MockModel(8, 12)
        model.aux_norm = nn.LayerNorm(8)
        for module in (model.norm, model.lm_head, model.aux_norm):
            module.to_empty(device="meta")
        # Mirrors the final-logits path: only resolved final modules are
        # materialized before indexed loading.
        model.norm.to_empty(device="cpu")
        model.lm_head.to_empty(device="cpu")
        reader = _Reader({
            "norm.weight": torch.full((8,), 2.0),
            "lm_head.weight": torch.full((12, 8), 3.0),
            "aux_norm.weight": torch.full((8,), 4.0),
            "aux_norm.bias": torch.full((8,), 5.0),
        })

        loaded = load_layer_weights_indexed(
            model, "/fake", [-1], "cpu", torch.float32,
            reader.weight_map, reader,
            strict_final_only=True,
            verbose=False,
        )

        assert loaded == 2
        assert torch.equal(model.norm.weight, torch.full((8,), 2.0))
        assert torch.equal(model.lm_head.weight, torch.full((12, 8), 3.0))
        assert model.aux_norm.weight.is_meta
        assert model.aux_norm.bias.is_meta

    def test_cpu_topk_uses_real_final_norm(self, monkeypatch):
        """Qwen RMSNorm forward must run; generic F.layer_norm is not equivalent."""
        import accuracy_checker.model_loader as ML
        monkeypatch.setattr(ML, "load_layer_weights_indexed", lambda *a, **kw: None)
        cmp = self._make_comparator(hidden=8, vocab=12)
        ref_model = _MockModel(8, 12)
        quant_model = _MockModel(8, 12)
        ref_hs = torch.randn(1, 3, 8)
        quant_hs = torch.randn(1, 3, 8)

        cmp._compute_logits_topk_cpu(ref_model, quant_model, ref_hs, quant_hs)

        assert ref_model.norm.calls == 1
        assert quant_model.norm.calls == 1

    def test_constant_logits_are_rejected(self):
        """An empty output head must not become a misleading logits_cos_sim=0 report."""
        cmp = self._make_comparator()
        with pytest.raises(RuntimeError, match="logits are constant"):
            cmp._validate_logits(torch.zeros(1, 12), "quant")

    def test_collect_full_logits_caps_to_seq_len(self, monkeypatch):
        """If seq_len < max_positions, only seq_len positions are captured."""
        import accuracy_checker.model_loader as ML
        monkeypatch.setattr(ML, "load_layer_weights_indexed", lambda *a, **kw: None)

        cmp = self._make_comparator(hidden=6, vocab=10, max_positions=32)
        ref_model = _MockModel(6, 10)
        quant_model = _MockModel(6, 10)
        # seq=2, but max=32 → should capture both positions only
        ref_hs = torch.randn(1, 2, 6)
        quant_hs = torch.randn(1, 2, 6)

        ld = cmp._collect_full_logits(ref_model, quant_model, ref_hs, quant_hs)
        assert ld.token_positions == [0, 1]

    def test_collect_full_logits_disabled_returns_none(self, monkeypatch):
        """When collect_full_logits=False, returns None even with valid inputs."""
        import accuracy_checker.model_loader as ML
        monkeypatch.setattr(ML, "load_layer_weights_indexed", lambda *a, **kw: None)

        cmp = self._make_comparator(hidden=8, vocab=12, max_positions=3)
        cmp.collect_full_logits = False
        ref_model = _MockModel(8, 12)
        quant_model = _MockModel(8, 12)
        ref_hs = torch.randn(1, 5, 8)
        quant_hs = torch.randn(1, 5, 8)
        assert cmp._collect_full_logits(ref_model, quant_model, ref_hs, quant_hs) is None

    def test_full_logits_flow_through_assemble_report(self, monkeypatch):
        """A BlockCompareReport.logits_data flows into ReportData.logits via assemble_report."""
        import accuracy_checker.model_loader as ML
        monkeypatch.setattr(ML, "load_layer_weights_indexed", lambda *a, **kw: None)

        cmp = self._make_comparator(hidden=8, vocab=12, max_positions=3)
        ref_model = _MockModel(8, 12)
        quant_model = _MockModel(8, 12)
        torch.manual_seed(1)
        ref_hs = torch.randn(1, 5, 8)
        quant_hs = torch.randn(1, 5, 8)
        ld = cmp._collect_full_logits(ref_model, quant_model, ref_hs, quant_hs)

        # Build a BlockCompareReport carrying this logits_data
        rep = BlockCompareReport(
            results=[BlockCompareResult(layer_name="layer.0", metrics={"cos_sim": 0.995})],
            logits_data=ld,
        )
        rd = assemble_report(l1_report=rep, model_name="ut-model")
        assert rd.logits is not None
        assert rd.logits.token_positions == [2, 3, 4]
        assert rd.logits.position_mode == "prompt_prefill"
        # assemble_report routes the data through, so L1's logits panel data is in ReportData


# ===========================================================================
# 7. badcase_workflow.compare_with_ground_truth
# ===========================================================================

class TestCompareWithGroundTruth:

    def _manifest(self, module, layer="5"):
        return BadCaseManifest(
            model="glm-9b",
            injected_layer=layer,
            injected_module=module,
            injected_change="scale 置零",
        )

    def _l2_hit(self):
        """定位 root_suspect == 注入 module -> 命中。"""
        return {
            "layer_idx": 5,
            "root_suspect": "self_attn.q_proj",
            "impact_boundary": "self_attn.q_proj",
            "subgraphs": {"self_attn.q_proj": 0.12, "mlp": 0.91},
            "subgraph_selfroterr": {"self_attn.q_proj": 0.88, "mlp": 0.03},
            "subgraph_rotberr": {"self_attn.q_proj": 0.90},
            "chain_deltas": {"attn": {"q_proj": 0.1}},
        }

    def test_hit_module_match(self):
        m = self._manifest("model.layers.5.self_attn.q_proj", layer="5")
        comp = compare_with_ground_truth(self._l2_hit(), m)
        assert isinstance(comp, GroundTruthComparison)
        assert comp.whether_hit_ground_truth is True
        assert "q_proj" in comp.source_candidate or "self_attn" in comp.source_candidate
        assert comp.ground_truth == "model.layers.5.self_attn.q_proj"
        # problem_path 含层号
        assert "5" in comp.problem_path

    def test_miss_disjoint_module(self):
        m = self._manifest("mlp.gate_proj", layer="5")
        comp = compare_with_ground_truth(self._l2_hit(), m)
        assert comp.whether_hit_ground_truth is False
        assert "MISS" in comp.hit_detail

    def test_list_input_picks_injected_layer(self):
        """list 输入: 选出与 injected_layer 匹配的层。"""
        wrong_layer = dict(self._l2_hit()); wrong_layer["layer_idx"] = 3
        right_layer = self._l2_hit()  # layer 5, root self_attn.q_proj
        m = self._manifest("self_attn.q_proj", layer="5")
        comp = compare_with_ground_truth([wrong_layer, right_layer], m)
        # 应选 layer 5 而非首层 3
        assert "layer=5" in comp.problem_path

    def test_manifest_save_load_roundtrip(self, tmp_path):
        m = self._manifest("self_attn.q_proj", layer="5")
        from accuracy_checker import save_manifest, load_manifest
        p = str(tmp_path / "manifest.json")
        save_manifest(m, p)
        m2 = load_manifest(p)
        assert m2.injected_module == "self_attn.q_proj"
        assert m2.injected_layer == "5"
