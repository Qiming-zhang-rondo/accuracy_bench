"""
L1 first_bad_block delta 检测测试

Case A: 正常 W4A8 累计缓慢下降 — 不应误判中间层
Case B: Layer 77 人为注入异常 — 应定位 Layer 77
Case C: 单层轻微自然波动后恢复 — 不应误判
Case D: 超大张量 cosine 数值稳定性 — 结果在 [-1, 1]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import unittest

from accuracy_checker.layer1_block_compare import (
    BlockCompareReport, BlockCompareResult,
    DELTA_MIN_DROP,
)
from accuracy_checker.metrics import cos_sim


def _make_results(cos_sims):
    """从 cos_sim 列表构造 BlockCompareReport"""
    results = []
    for i, cs in enumerate(cos_sims):
        results.append(BlockCompareResult(
            layer_name=f"layer.{i}.block_output",
            metrics={"cos_sim": cs},
        ))
    return BlockCompareReport(results=results)


class TestCaseA_NormalAccumulation(unittest.TestCase):
    """Case A: 正常 W4A8 累计缓慢下降

    cosine 在 Layer 33 跌破 0.99, 但每层 delta 都很小。
    不应把 Layer 33 判为 bad layer。
    """

    def test_no_bad_jump(self):
        # 78 层, 从 1.0 缓慢线性下降到 0.93
        # 每层 delta ≈ -0.0009 (0.09%), 远小于 min_drop=0.005
        cos_sims = [1.0 - i * 0.0009 for i in range(78)]
        report = _make_results(cos_sims)

        detection = report._detect_bad_layers()
        self.assertEqual(len(detection.bad_layers), 0,
                         f"不应有 bad layer, 但找到 {len(detection.bad_layers)} 个")

        # first_bad_block 应回退到绝对阈值 (layer 12 附近 cos<0.99)
        fb = report.first_bad_block
        # first_bad_block 应回退到绝对阈值
        self.assertIsNotNone(fb)

        # first_threshold_crossing 应在 layer 12 (1.0 - 12*0.0009 = 0.9892 < 0.99)
        tc = report.first_threshold_crossing
        self.assertIsNotNone(tc)
        self.assertIn("12", tc)

    def test_layer_33_not_bad(self):
        # 模拟实际 badcase 数据: layer 33 cos=0.989, 但 delta 很小
        cos_sims = [1.0]
        for i in range(1, 78):
            # 正常下降 ~0.0005/层, 到 layer 33 约 0.989
            cos_sims.append(cos_sims[-1] - 0.0005)
        report = _make_results(cos_sims)

        detection = report._detect_bad_layers()
        # Layer 33 的 delta 应该很小, 不应被判为 bad
        layer_33 = next((d for d in detection.layer_deltas if d.layer_idx == 33), None)
        self.assertIsNotNone(layer_33)
        self.assertFalse(layer_33.is_bad_jump,
                         f"Layer 33 不应被判为 bad (delta={layer_33.delta_cos})")


class TestCaseB_Layer77Injection(unittest.TestCase):
    """Case B: Layer 77 人为注入异常

    Layer 77 delta 约为 -0.01581, 应定位 Layer 77 为 first bad layer。
    """

    def test_layer_77_detected(self):
        # 基于实际 badcase 数据
        cos_sims = [1.0]
        for i in range(1, 78):
            if i == 77:
                # Layer 77: 大幅下降
                cos_sims.append(cos_sims[-1] - 0.01581)
            else:
                # 正常下降
                cos_sims.append(cos_sims[-1] - 0.0005)
        report = _make_results(cos_sims)

        detection = report._detect_bad_layers()
        self.assertEqual(len(detection.bad_layers), 1)
        self.assertEqual(detection.bad_layers[0].layer_idx, 77)

        fb = report.first_bad_block
        self.assertIsNotNone(fb)
        self.assertIn("77", fb)

    def test_layer_77_debug_output(self):
        """验证 debug 信息包含所有必需字段"""
        cos_sims = [1.0]
        for i in range(1, 78):
            if i == 77:
                cos_sims.append(cos_sims[-1] - 0.01581)
            else:
                cos_sims.append(cos_sims[-1] - 0.0005)
        report = _make_results(cos_sims)

        detection = report._detect_bad_layers()
        info = detection.bad_layers[0]

        debug_str = info.debug_str()
        for field in ["cos_sim_prev", "cos_sim_curr", "delta_cos",
                       "drop_percent", "baseline", "MAD", "z/mad score",
                       "statistical", "absolute", "persistent", "detected_bad"]:
            self.assertIn(field, debug_str, f"debug_str 缺少字段: {field}")

    def test_layer_77_persistent(self):
        """Layer 77 是最后一层, is_persistent 应为 True"""
        cos_sims = [1.0]
        for i in range(1, 78):
            if i == 77:
                cos_sims.append(cos_sims[-1] - 0.01581)
            else:
                cos_sims.append(cos_sims[-1] - 0.0005)
        report = _make_results(cos_sims)

        detection = report._detect_bad_layers()
        info = detection.bad_layers[0]
        self.assertTrue(info.is_persistent,
                         "最后一层的 bad_jump 应默认 persistent=True")


class TestCaseC_RecoveryAfterSpike(unittest.TestCase):
    """Case C: 单层轻微自然波动后恢复 — 不应误判

    某层有小幅下降, 但下一层立刻恢复, 不应被判为 bad。
    """

    def test_recovery_not_bad(self):
        cos_sims = [1.0]
        for i in range(1, 78):
            cos_sims.append(cos_sims[-1] - 0.0005)  # 正常下降

        # 在 layer 50 注入一个小 spike (delta=-0.003, < min_drop=0.005)
        # 下一层恢复
        cos_sims[50] = cos_sims[49] - 0.003
        cos_sims[51] = cos_sims[50] + 0.003  # 恢复

        report = _make_results(cos_sims)
        detection = report._detect_bad_layers()

        # Layer 50 的 delta=-0.003, 小于 min_drop=0.005, 不应被判为 bad
        layer_50 = next((d for d in detection.layer_deltas if d.layer_idx == 50), None)
        self.assertIsNotNone(layer_50)
        self.assertFalse(layer_50.is_absolute_jump,
                         "delta=-0.003 不应通过 absolute jump (需 < -0.005)")
        self.assertFalse(layer_50.is_bad_jump)

    def test_large_spike_with_recovery_still_detected(self):
        """大 spike 但下一层完全恢复: is_bad_jump=True, is_persistent=False"""
        cos_sims = [1.0]
        for i in range(1, 78):
            cos_sims.append(cos_sims[-1] - 0.0005)

        # Layer 50: 大 spike, 但下一层完全恢复
        cos_sims[50] = cos_sims[49] - 0.02  # delta=-0.02 > min_drop
        cos_sims[51] = cos_sims[50] + 0.02  # 完全恢复

        report = _make_results(cos_sims)
        detection = report._detect_bad_layers()

        layer_50 = next((d for d in detection.layer_deltas if d.layer_idx == 50), None)
        self.assertIsNotNone(layer_50)
        self.assertTrue(layer_50.is_bad_jump, "delta=-0.02 应通过 bad_jump")
        self.assertFalse(layer_50.is_persistent, "恢复后不应 persistent")

        # first_bad 应优先 persistent bad; 如果只有 non-persistent, 仍用它
        # 但这个 case 里 layer 77 也可能有 bad_jump (取决于 delta)
        # 关键: layer 50 的 is_persistent=False


class TestCaseD_NumericalStability(unittest.TestCase):
    """Case D: 超大张量 cosine 数值稳定性

    952M 元素 flatten 后 float32 点积溢出, 需 float64 分块。
    """

    def test_identical_large_tensor(self):
        """相同张量应得到约 0.999999"""
        n = 10_000_000  # 10M elements (足够大测试稳定性)
        a = torch.randn(n, dtype=torch.float32)
        b = a.clone()
        cs = cos_sim(a, b)
        self.assertGreater(cs, 0.9999, f"相同张量 cos_sim 应≈1.0, got {cs}")
        self.assertLessEqual(cs, 1.0 + 1e-6, f"cos_sim 不应超过 1.0, got {cs}")

    def test_orthogonal_large_tensor(self):
        """正交张量应得到约 0.0"""
        n = 1_000_000
        a = torch.randn(n, dtype=torch.float32)
        b = torch.randn(n, dtype=torch.float32)
        cs = cos_sim(a, b)
        self.assertLess(abs(cs), 0.01, f"随机正交张量 cos_sim 应≈0, got {cs}")

    def test_small_tensor_float32(self):
        """小张量仍用 float32 快速路径"""
        a = torch.randn(1000, dtype=torch.float32)
        b = a.clone()
        cs = cos_sim(a, b)
        self.assertGreater(cs, 0.9999)

    def test_opposite_large_tensor(self):
        """相反张量应得到约 -1.0

        float32 累积 1M 元素 dot product 有 ~0.05% 精度损失,
        所以阈值放宽到 -0.999 (而非 -0.9999)。
        关键验证: (1) 接近 -1.0, (2) 不超出 [-1, 1] 范围。
        """
        n = 1_000_000
        a = torch.randn(n, dtype=torch.float32)
        b = -a
        cs = cos_sim(a, b)
        self.assertLess(cs, -0.999, f"相反张量 cos_sim 应≈-1.0, got {cs}")
        self.assertGreaterEqual(cs, -1.0 - 1e-6, f"cos_sim < -1: {cs}")

    def test_zero_norm(self):
        """零向量应返回 0.0"""
        a = torch.zeros(1000, dtype=torch.float32)
        b = torch.randn(1000, dtype=torch.float32)
        cs = cos_sim(a, b)
        self.assertEqual(cs, 0.0)

    def test_range_validity(self):
        """cos_sim 必须在 [-1, 1] 范围内"""
        for _ in range(10):
            n = np.random.randint(100, 5_000_000)
            a = torch.randn(n, dtype=torch.float32)
            b = torch.randn(n, dtype=torch.float32)
            cs = cos_sim(a, b)
            self.assertGreaterEqual(cs, -1.0 - 1e-6, f"cos_sim < -1: {cs}")
            self.assertLessEqual(cs, 1.0 + 1e-6, f"cos_sim > 1: {cs}")


class TestCaseE_RealBadcaseData(unittest.TestCase):
    """Case E: 用实际 badcase report_data.json 验证

    Layer 77 应被定位为 first_bad_block。
    Layer 33 应只出现在 first_threshold_crossing。
    """

    def test_real_badcase_layer77(self):
        report_path = os.path.join(
            os.path.dirname(__file__), "..", "reports", "badcase_bf16_ref", "report_data.json"
        )
        if not os.path.exists(report_path):
            self.skipTest("report_data.json not found")

        import json
        with open(report_path) as f:
            d = json.load(f)

        results = []
        for l in d.get("l1_layers", []):
            if l.get("layer_idx", -1) >= 0:
                results.append(BlockCompareResult(
                    layer_name=l["layer_name"],
                    metrics={"cos_sim": l["cos_sim"]},
                ))
        report = BlockCompareReport(results=results)
        detection = report._detect_bad_layers()

        # Layer 77 应是 first_bad_block
        fb = report.first_bad_block
        self.assertIsNotNone(fb)
        self.assertIn("77", fb, f"first_bad_block 应是 layer 77, got {fb}")

        # Layer 33 应是 first_threshold_crossing (辅助, 非根因)
        tc = report.first_threshold_crossing
        self.assertIsNotNone(tc)
        self.assertIn("33", tc, f"first_threshold_crossing 应是 layer 33, got {tc}")

        # Layer 33 不应是 bad_jump
        layer_33 = next((d for d in detection.layer_deltas if d.layer_idx == 33), None)
        self.assertIsNotNone(layer_33)
        self.assertFalse(layer_33.is_bad_jump,
                         "Layer 33 不应被判为 bad (只是累计跌破 0.99)")

        # Layer 77 应是 bad_jump
        layer_77 = next((d for d in detection.layer_deltas if d.layer_idx == 77), None)
        self.assertIsNotNone(layer_77)
        self.assertTrue(layer_77.is_bad_jump, "Layer 77 应被判为 bad")
        self.assertTrue(layer_77.is_persistent, "Layer 77 (最后一层) 应 persistent=True")


if __name__ == "__main__":
    unittest.main(verbosity=2)
