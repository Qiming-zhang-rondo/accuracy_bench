"""
工作项4 (A4 激活扩展) UT — 验证 mxfp4 + int4 fake quant 核心逻辑。

纯 torch 逻辑, 不依赖 NPU/大模型, 用小 tensor 验证:
  1. MXFP4: 输出值只含 E2M1 可表示集 × per-block scale
  2. MXFP4: scale 公式 = floor(log2(amax)) - 2 (D2 OCP even)
  3. MXFP4: round = half-up (D3)
  4. MXFP4: 饱和到 6.0 (max_norm)
  5. INT4: 输出值只含 {-8..7} × per-token scale
  6. INT4: scale = amax/7, clamp [-8,7] (D5)
  7. INT4: round = half-even (D4)
  8. dtype/shape 保持
"""

import numpy as np
import pytest
import torch

from accuracy_checker.mxfp4_fake_quant import mxfp4_fake_quant_per_block, E2M1_MAX
from accuracy_checker.int4_fake_quant import int4_fake_quant_per_token_sym, INT4_MAX
from accuracy_checker.layer1_block_compare import _dispatch_act_fake_quant

# E2M1 可表示正值 (与 model_loader.py E2M1_VALUES 一致)
E2M1_POSITIVE = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}


# ===========================================================================
# MXFP4 测试
# ===========================================================================

def _round_to_e2m1_set(val, scale):
    """把 fake quant 后的值 / scale 检查是否落在 E2M1 可表示集"""
    if scale == 0:
        return val == 0.0
    normalized = val / scale
    # 容差处理浮点误差
    for rep in E2M1_POSITIVE:
        if abs(normalized - rep) < 1e-5:
            return True
    return False


class TestMXFP4:

    def test_output_values_in_e2m1_set(self):
        """所有输出值 / scale 必须落在 E2M1 可表示集 {0,0.5,1,1.5,2,3,4,6}"""
        torch.manual_seed(42)
        x = torch.randn(2, 3, 32) * 10  # block_size=32 的倍数
        out = mxfp4_fake_quant_per_block(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

        # 逐 block 检查: out/block_scale 应在 E2M1 集
        x_blocked = x.float().view(2, 3, 1, 32)
        out_blocked = out.float().view(2, 3, 1, 32)
        amax = x_blocked.abs().amax(-1, keepdim=True).clamp(min=torch.finfo(torch.float32).eps)
        shared_scale = torch.pow(2.0, torch.floor(torch.log2(amax)) - 2)

        normalized = out_blocked / shared_scale
        allowed = torch.tensor(list(E2M1_POSITIVE))
        # 每个值的绝对值应接近 allowed 中的某个 (输出可正可负, E2M1 含正负)
        dists = (normalized.abs().unsqueeze(-1) - allowed).abs().min(-1).values
        assert (dists < 1e-4).all(), \
            f"输出含非 E2M1 可表示值: max dist = {dists.max().item()}"

    def test_scale_formula_floor_ocp_even(self):
        """D2: scale = 2^(floor(log2(amax)) - 2), 不是 ceil"""
        # 构造已知 amax 的 block
        x = torch.tensor([[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        # amax=3.0, floor(log2(3))=1, scale=2^(1-2)=2^(-1)=0.5
        # 若用 ceil: ceil(log2(3/6))=ceil(-1)=0? 不对, ceil(log2(3/6))=ceil(-1.0)=-1
        # 关键差异在 amax 落 (6,8] 时: floor 给 scale=2^(3-2)=2, ceil 给 scale=2^ceil(log2(amax/6))
        x_sat = torch.tensor([[7.0] + [0.0] * 31])
        out = mxfp4_fake_quant_per_block(x_sat)
        # amax=7.0, floor(log2(7))=2, scale=2^(2-2)=1.0
        # 7.0 / 1.0 = 7.0 → E2M1 max=6.0, 饱和到 6.0
        assert abs(out[0, 0].item() - 6.0) < 1e-4, \
            f"amax=7 应饱和到 6.0, got {out[0,0].item()}"

    def test_saturation_to_max_norm(self):
        """超过 6.0 的值 (scaled) 饱和到 6.0 (输出为反量化值 = 6.0 × scale)"""
        x = torch.tensor([[8.0] + [0.0] * 31])  # amax=8, scale=2^(3-2)=2, 8/2=4 → 4.0 (无饱和)
        out = mxfp4_fake_quant_per_block(x)
        # 8/2=4.0, 在 E2M1 集内, 不饱和 → 反量化 4.0 × scale(2) = 8.0
        assert abs(out[0, 0].item() - 8.0) < 1e-4, \
            f"8/2=4.0 无饱和, 反量化=4.0×2=8.0, got {out[0,0].item()}"

        # amax=15, scale=2^(3-2)=2, 15/2=7.5 → 饱和到 6.0 → 反量化 6.0 × 2 = 12.0
        x2 = torch.tensor([[15.0] + [0.0] * 31])
        out2 = mxfp4_fake_quant_per_block(x2)
        assert abs(out2[0, 0].item() - 12.0) < 1e-4, \
            f"15.0/2=7.5 饱和到 6.0, 反量化=6.0×2=12.0, got {out2[0,0].item()}"

    def test_round_half_up(self):
        """D3: half-up (不是 half-even)。
        构造值在 0.5 边界, half-up 总是向上, half-even 会到偶数"""
        # scale=1.0 (amax 在 (4,8]), step=0.5 (exp=0)
        # 值 0.75: half-up → floor(0.75/0.5+0.5)*0.5 = floor(2.0)*0.5 = 1.0
        #         half-even → round(0.75/0.5)*0.5 = round(1.5)*0.5 = 2*0.5 = 1.0 (偶数, 一致)
        # 值 1.25: half-up → floor(1.25/0.5+0.5)*0.5 = floor(3.0)*0.5 = 1.5
        #          half-even → round(1.25/0.5)*0.5 = round(2.5)*0.5 = 2*0.5 = 1.0 (偶数, 差异!)
        # 需 amax 落 (4,8] 让 scale=1.0
        x = torch.tensor([[5.0, 1.25] + [0.0] * 30])
        out = mxfp4_fake_quant_per_block(x)
        # amax=5.0, floor(log2(5))=2, scale=2^(2-2)=1.0
        # 1.25/1.0=1.25, exp=0, step=0.5
        # half-up: floor(1.25/0.5+0.5)*0.5 = floor(3.0)*0.5 = 1.5
        # half-even: round(1.25/0.5)*0.5 = round(2.5)*0.5 = 2*0.5 = 1.0
        assert abs(out[0, 1].item() - 1.5) < 1e-4, \
            f"1.25 (half-up) 应 → 1.5, got {out[0,1].item()} (half-even 会给 1.0)"

    def test_zero_input(self):
        """全零输入 → 全零输出"""
        x = torch.zeros(4, 32)
        out = mxfp4_fake_quant_per_block(x)
        assert torch.all(out == 0)

    def test_shape_dtype_preserved(self):
        x = torch.randn(2, 64, dtype=torch.float16) * 5
        out = mxfp4_fake_quant_per_block(x)
        assert out.shape == x.shape
        assert out.dtype == torch.float16

    def test_non_divisible_hidden_returns_original(self):
        """hidden_size 不是 block_size 倍数 → 返回原 tensor"""
        x = torch.randn(2, 33)  # 33 不是 32 倍数
        out = mxfp4_fake_quant_per_block(x)
        assert torch.equal(out, x)

    def test_block_size_param(self):
        """block_size=16 也应工作"""
        x = torch.randn(2, 32) * 5
        out = mxfp4_fake_quant_per_block(x, block_size=16)
        assert out.shape == x.shape


# ===========================================================================
# INT4 测试
# ===========================================================================

class TestINT4:

    def test_output_values_in_int4_set(self):
        """所有输出值 / scale 必须落在 {-8..7} 整数集"""
        torch.manual_seed(42)
        x = torch.randn(3, 64) * 10
        out = int4_fake_quant_per_token_sym(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

        # per-token scale
        x_2d = x.float().reshape(-1, 64)
        amax = torch.max(-x_2d.amin(1, keepdim=True), x_2d.amax(1, keepdim=True)).clamp(min=1e-12)
        scale = amax / 7.0
        normalized = out.float().reshape(-1, 64) / scale
        # 应接近 {-8..7} 中的整数
        rounded = torch.round(normalized)
        assert (torch.abs(normalized - rounded) < 1e-4).all(), "输出/scale 不接近整数"
        assert (rounded >= -8).all() and (rounded <= 7).all(), "输出/scale 超出 [-8,7]"

    def test_scale_formula_amax_over_7(self):
        """D5: scale = amax / 7 (不是 /8)"""
        # 单 token: [7, -7, 0, ...], amax=7, scale=7/7=1.0
        x = torch.tensor([[7.0, -7.0] + [0.0] * 62])
        out = int4_fake_quant_per_token_sym(x)
        # scale=1.0, 7/1=7.0 (在 [-8,7] 内), -7/1=-7.0
        assert abs(out[0, 0].item() - 7.0) < 1e-5
        assert abs(out[0, 1].item() - (-7.0)) < 1e-5

    def test_clamp_range(self):
        """D5: clamp [-8, 7], amax=8 时 8/scale=8 在范围内, amax>8 时饱和"""
        # amax=14, scale=14/7=2.0, 14/2=7.0 (在范围)
        x = torch.tensor([[14.0] + [0.0] * 63])
        out = int4_fake_quant_per_token_sym(x)
        assert abs(out[0, 0].item() - 14.0) < 1e-4  # 14/2=7, 7*2=14

        # amax=100, scale=100/7≈14.29, 100/14.29=7.0
        x2 = torch.tensor([[100.0] + [0.0] * 63])
        out2 = int4_fake_quant_per_token_sym(x2)
        assert abs(out2[0, 0].item() - 100.0) < 1e-2  # 100/scale=7, 7*scale=100

    def test_round_half_even(self):
        """D4: half-even (torch.round 默认)。
        值 2.5*step → half-even 给 2 (偶数), half-up 给 3"""
        # 构造 scale 使 x/scale = 2.5
        # scale = amax/7, 若 amax=7, scale=1.0, x=2.5 → 2.5/1.0=2.5
        # half-even: round(2.5)=2, half-up: floor(2.5+0.5)=3
        x = torch.tensor([[2.5, 7.0] + [0.0] * 62])  # amax=7, scale=1.0
        out = int4_fake_quant_per_token_sym(x)
        # torch.round(2.5) = 2.0 (half-even)
        assert abs(out[0, 0].item() - 2.0) < 1e-5, \
            f"2.5 (half-even) 应 → 2.0, got {out[0,0].item()} (half-up 会给 3.0)"

    def test_per_token_granularity(self):
        """per-token: 每行独立 scale"""
        # row 0 amax 大, row 1 amax 小
        x = torch.tensor([[10.0, -10.0] + [0.0] * 62,
                          [1.0, -1.0] + [0.0] * 62])
        out = int4_fake_quant_per_token_sym(x)
        # row 0: scale=10/7, 10/scale=7, out=7*10/7=10
        # row 1: scale=1/7, 1/scale=7, out=7*1/7=1
        assert abs(out[0, 0].item() - 10.0) < 1e-4
        assert abs(out[1, 0].item() - 1.0) < 1e-4

    def test_zero_input(self):
        x = torch.zeros(4, 64)
        out = int4_fake_quant_per_token_sym(x)
        assert torch.all(out == 0)

    def test_shape_dtype_preserved(self):
        x = torch.randn(2, 3, 64, dtype=torch.float16) * 8
        out = int4_fake_quant_per_token_sym(x)
        assert out.shape == x.shape
        assert out.dtype == torch.float16


def test_all_activation_quant_dispatches_preserve_tensor_contract():
    # transpose keeps the last dimension quantizable while making the tensor
    # non-contiguous, matching hidden states produced by some accelerator ops.
    x = torch.randn(2, 3, 64, dtype=torch.float16).transpose(0, 1)
    assert not x.is_contiguous()
    for quant_type in (
        "W8A8_MXFP8",
        "W4A8_MXFP",
        "W4A4_MXFP4",
        "W4A4_DYNAMIC",
        "W8A8_DYNAMIC",
        "W4A8_DYNAMIC",
        "W8A8",
        "W4A8",
    ):
        out = _dispatch_act_fake_quant(x, quant_type)
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert out.device == x.device


def test_legacy_laos_activation_alias_matches_dynamic():
    x = torch.randn(2, 64)
    torch.testing.assert_close(
        _dispatch_act_fake_quant(x, "W4A4_LAOS"),
        _dispatch_act_fake_quant(x, "W4A4_DYNAMIC"),
    )


def test_unknown_activation_quant_type_fails_closed():
    with pytest.raises(ValueError, match="unsupported activation quant type"):
        _dispatch_act_fake_quant(torch.randn(2, 64), "W4A4_UNKNOWN")


def test_auto_activation_type_must_be_resolved_before_dispatch():
    with pytest.raises(ValueError, match="must be resolved"):
        _dispatch_act_fake_quant(torch.randn(2, 64), "AUTO")
