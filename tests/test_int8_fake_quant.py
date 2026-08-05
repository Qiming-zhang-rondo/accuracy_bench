"""
INT8 per-token 对称线性激活伪量化 UT。

验证 int8_fake_quant_per_token_sym:
  1. 输出值 / scale 落在 {-128..127} 整数集
  2. scale = amax / 127 (对称 INT8)
  3. round = half-even (torch.round 默认)
  4. clamp [-128, 127]
  5. dtype/shape 保持
"""

import pytest
import torch

from accuracy_checker.int8_fake_quant import int8_fake_quant_per_token_sym, INT8_MAX


class TestINT8:

    def test_output_values_in_int8_set(self):
        """所有输出值 / scale 必须落在 {-128..127} 整数集"""
        torch.manual_seed(42)
        x = torch.randn(3, 64) * 10
        out = int8_fake_quant_per_token_sym(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

        # per-token scale
        x_2d = x.float().reshape(-1, 64)
        amax = torch.max(-x_2d.amin(1, keepdim=True), x_2d.amax(1, keepdim=True)).clamp(min=1e-12)
        scale = amax / 127.0
        normalized = out.float().reshape(-1, 64) / scale
        # 应接近 {-128..127} 中的整数
        rounded = torch.round(normalized)
        assert torch.allclose(normalized, rounded, atol=1e-4), \
            f"输出含非整数: max diff = {(normalized - rounded).abs().max().item()}"
        assert (rounded >= -128).all() and (rounded <= 127).all(), \
            f"输出超出 [-128, 127]: min={rounded.min().item()}, max={rounded.max().item()}"

    def test_scale_formula_amax_over_127(self):
        """scale = amax / 127, 构造已知 amax 验证"""
        # 构造 amax=5.0 的输入
        x = torch.tensor([[5.0, -3.0, 2.0, 1.0]])
        out = int8_fake_quant_per_token_sym(x)
        # amax = max(|-5|, |3|, |2|, |1|) = 5.0
        scale = 5.0 / 127.0
        # 5.0 / scale = 127 → clamp 127 → 反量化 127 * scale = 5.0
        assert abs(out[0, 0].item() - 5.0) < 1e-4, \
            f"amax=5.0 应精确表示, got {out[0,0].item()}"
        # -3.0 / scale = -76.2 → round -76 → 反量化 -76 * scale ≈ -2.99
        expected_neg3 = -76.0 * scale
        assert abs(out[0, 1].item() - expected_neg3) < 1e-4, \
            f"-3.0 应量化到 -76, got {out[0,1].item()}"

    def test_clamp_range(self):
        """超过 [-128, 127] 的值被 clamp"""
        # 构造极端值
        x = torch.tensor([[100.0, -100.0, 0.0, 1.0]])
        out = int8_fake_quant_per_token_sym(x)
        # amax=100, scale=100/127
        # 100 / (100/127) = 127 → clamp 127 → 反量化 127 * (100/127) = 100
        assert abs(out[0, 0].item() - 100.0) < 1e-4
        # -100 / (100/127) = -127 → clamp -127 → 反量化 -127 * (100/127) = -100
        assert abs(out[0, 1].item() - (-100.0)) < 1e-4

    def test_round_half_even(self):
        """half-even: 0.5 边界向偶数取整"""
        # 构造值在 0.5 边界
        # scale=1.0 (amax=127), 值 1.5 → round(1.5)=2 (偶数), 值 2.5 → round(2.5)=2 (偶数)
        x = torch.tensor([[127.0, 1.5, 2.5] + [0.0] * 124])
        out = int8_fake_quant_per_token_sym(x)
        # amax=127, scale=1.0
        # 1.5 / 1.0 = 1.5 → round half-even → 2.0
        assert abs(out[0, 1].item() - 2.0) < 1e-4, \
            f"1.5 (half-even) 应 → 2.0, got {out[0,1].item()}"
        # 2.5 / 1.0 = 2.5 → round half-even → 2.0
        assert abs(out[0, 2].item() - 2.0) < 1e-4, \
            f"2.5 (half-even) 应 → 2.0, got {out[0,2].item()}"

    def test_zero_input(self):
        """全零输入 → 全零输出"""
        x = torch.zeros(4, 64)
        out = int8_fake_quant_per_token_sym(x)
        assert torch.all(out == 0)

    def test_shape_dtype_preserved(self):
        """shape 和 dtype 保持不变"""
        x = torch.randn(2, 64, dtype=torch.float16) * 5
        out = int8_fake_quant_per_token_sym(x)
        assert out.shape == x.shape
        assert out.dtype == torch.float16

    def test_per_token_independence(self):
        """每个 token 独立计算 scale"""
        # 两个 token, amax 不同
        x = torch.tensor([[10.0, -5.0, 3.0, 1.0],
                          [1.0, -0.5, 0.3, 0.1]])
        out = int8_fake_quant_per_token_sym(x)
        # token 0: amax=10, scale=10/127
        # token 1: amax=1, scale=1/127
        # 两个 token 的 scale 不同, 输出应不同
        scale0 = 10.0 / 127.0
        scale1 = 1.0 / 127.0
        # token 0: 10 / scale0 = 127 → 反量化 10
        assert abs(out[0, 0].item() - 10.0) < 1e-4
        # token 1: 1 / scale1 = 127 → 反量化 1
        assert abs(out[1, 0].item() - 1.0) < 1e-4

    def test_int8_max_constant(self):
        """INT8_MAX = 127"""
        assert INT8_MAX == 127
