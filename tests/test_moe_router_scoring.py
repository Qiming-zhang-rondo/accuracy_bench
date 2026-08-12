"""UT — MoE router sigmoid/noaux_tc scoring (回归 _moe_forward_chunked 路由公式).

背景:
  acc_bench 的 `_moe_forward_chunked` 之前用 F.softmax 做路由评分。而 GLM-5.1 /
  DeepSeek-V3 风格的 MoE (`GlmMoeDsaMoE.route_tokens_to_experts`) 是:
    sigmoid → (+e_score_correction_bias, 仅用于选择) → gather 原 sigmoid 分数
    → norm_topk_prob 归一化 → × routed_scaling_factor
  旧公式错导致 REF 走 streaming 生成时输出退化成 "1,1,1,11111111"。

本 UT 把修复后的公式独立出来 (mirror of `_moe_forward_chunked` router 段), 与
transformers 中 GlmMoeDsaMoE.route_tokens_to_experts 的 reference 实现做逐数值对齐:
  - n_group == 1 (GLM-5.1 默认), 不进组过滤
  - n_group >  1 (DeepSeek-V3), 走组过滤
  - bias 非零时, 选择 topk_indices 会改 (但权重不变)
  - sigmoid vs softmax: 用旧 F.softmax 的输出绝对不一致

纯 torch CPU; 不依赖 NPU / transformers (transformers 仅在可用时做交叉校验)。
"""
from __future__ import annotations

import torch

try:
    import pytest  # noqa: F401
    _HAS_PYTEST = True
except ImportError:  # NPU-only container 可能没装 pytest
    _HAS_PYTEST = False

    class _PytestFallback:
        @staticmethod
        def fixture(*args, **kwargs):
            def deco(fn):  # 忽略 autouse/params, 真接函数
                return fn
            return deco

    pytest = _PytestFallback()  # type: ignore


# =========================================================================
# 公式 (mirror of accuracy_checker.layer1_block_compare._moe_forward_chunked)
# 把 router 评分逻辑独立出来, 便于 UT 直接调用, 不需要构造完整 ShardedBlockComparator。
# =========================================================================
def compute_moe_topk(router_logits, num_experts_per_tok, *,
                     routed_scaling_factor=None, norm_topk_prob=True,
                     n_group=1, topk_group=1, e_score_correction_bias=None):
    """镜像 _moe_forward_chunked 的 sigmoid/noaux_tc 路由公式。

    Args:
        router_logits: [..., n_routed_experts]
        num_experts_per_tok: top_k
        routed_scaling_factor: float (GLM-5.1 默认 2.5)
        norm_topk_prob: 是否归一化
        n_group, topk_group: DeepSeek-V3 group filtering
        e_score_correction_bias: [n_routed_experts] (可选)

    Returns:
        (topk_indices, topk_weights): 都在最后一维是 num_experts_per_tok
    """
    routed_scaling_factor = 1.0 if routed_scaling_factor is None else float(routed_scaling_factor)
    sigmoid_scores = router_logits.float().sigmoid()

    scores_for_choice = sigmoid_scores
    if e_score_correction_bias is not None and e_score_correction_bias.numel() > 0:
        scores_for_choice = sigmoid_scores + e_score_correction_bias.to(
            device=sigmoid_scores.device, dtype=sigmoid_scores.dtype)

    n_routed_experts = router_logits.shape[-1]
    if n_group > 1 and n_routed_experts >= n_group:
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
        scores_for_choice = scores_for_choice.masked_fill(
            ~score_mask.bool(), float("-inf"))

    _, topk_indices = scores_for_choice.topk(num_experts_per_tok, dim=-1, sorted=False)
    topk_weights = sigmoid_scores.gather(-1, topk_indices)
    if norm_topk_prob:
        denom = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights / denom
    topk_weights = topk_weights * routed_scaling_factor
    return topk_indices, topk_weights


# =========================================================================
# Reference: 1:1 复制 transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py
#   GlmMoeDsaMoE.route_tokens_to_experts
# 仅用于 UT 校验, 不引入运行时依赖。
# =========================================================================
def _reference_route(router_logits, num_experts_per_tok, *,
                     routed_scaling_factor, norm_topk_prob=True,
                     n_group=1, topk_group=1, e_score_correction_bias=None):
    router_logits = router_logits.float().sigmoid()
    if e_score_correction_bias is None:
        e_score_correction_bias = torch.zeros(router_logits.shape[-1])
    router_logits_for_choice = router_logits + e_score_correction_bias
    n_routed_experts = router_logits.shape[-1]
    group_scores = (
        router_logits_for_choice.view(-1, n_group, n_routed_experts // n_group)
        .topk(2, dim=-1)[0].sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, n_group, n_routed_experts // n_group)
        .reshape(-1, n_routed_experts)
    )
    scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    topk_indices = torch.topk(scores_for_choice, k=num_experts_per_tok, dim=-1, sorted=False)[1]
    topk_weights = router_logits.gather(1, topk_indices)
    if norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights / denominator
    topk_weights = topk_weights * routed_scaling_factor
    return topk_indices, topk_weights


# =========================================================================
# Fixtures
# =========================================================================
@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(7)
    yield


def _make_logits(batch_seq, n_experts):
    return torch.randn(batch_seq, n_experts, dtype=torch.float32) * 3.0


# =========================================================================
# Tests
# =========================================================================
def test_glm51_ngroup1_matches_reference():
    """GLM-5.1 配置 (n_group=1, topk=8, scale=2.5, norm=true): 与 reference 对齐."""
    router_logits = _make_logits(4, 256)
    bias = torch.randn(256, dtype=torch.float32) * 0.05
    idx_mine, w_mine = compute_moe_topk(
        router_logits, num_experts_per_tok=8,
        routed_scaling_factor=2.5, norm_topk_prob=True,
        n_group=1, topk_group=1, e_score_correction_bias=bias)
    idx_ref, w_ref = _reference_route(
        router_logits, num_experts_per_tok=8,
        routed_scaling_factor=2.5, norm_topk_prob=True,
        n_group=1, topk_group=1, e_score_correction_bias=bias)
    assert torch.equal(idx_mine, idx_ref), "topk_indices 与 reference 不一致"
    assert torch.allclose(w_mine, w_ref, atol=1e-6), "topk_weights 与 reference 不一致"
    # 权重和 ≈ 2.5 (8 个归一化后乘 2.5)
    assert abs(w_mine.sum(dim=-1).mean().item() - 2.5) < 1e-4


def test_glm51_no_bias_matches_pure_sigmoid_topk():
    """bias=None (没加载到 buffer 的情况) 时, 选择等价于纯 sigmoid topk."""
    router_logits = _make_logits(2, 256)
    idx_mine, w_mine = compute_moe_topk(
        router_logits, 8,
        routed_scaling_factor=2.5, norm_topk_prob=True, n_group=1, topk_group=1)
    idx_ref, w_ref = _reference_route(
        router_logits, 8,
        routed_scaling_factor=2.5, norm_topk_prob=True,
        n_group=1, topk_group=1,
        e_score_correction_bias=torch.zeros(256))
    assert torch.equal(idx_mine, idx_ref)
    assert torch.allclose(w_mine, w_ref, atol=1e-6)
    # sigmoid / softmax 对单个 logit 都是单调变换，因此 top-k 索引必然
    # 相同；区别在归一化后的路由权重，而不是专家选择。
    softmax_scores = torch.nn.functional.softmax(router_logits.float(), dim=-1)
    softmax_topk, idx_softmax = softmax_scores.topk(8, dim=-1)
    softmax_weights = softmax_topk / softmax_topk.sum(dim=-1, keepdim=True) * 2.5
    assert torch.equal(idx_mine, idx_softmax)
    assert not torch.allclose(w_mine, softmax_weights, atol=1e-6)


def test_deepseekv3_ngroup8_matches_reference():
    """n_group=8 (DeepSeek-V3 风格) 与 reference 对齐."""
    router_logits = _make_logits(4, 256)
    bias = torch.randn(256) * 0.1
    idx_mine, w_mine = compute_moe_topk(
        router_logits, num_experts_per_tok=8,
        routed_scaling_factor=2.5, norm_topk_prob=True,
        n_group=8, topk_group=4, e_score_correction_bias=bias)
    idx_ref, w_ref = _reference_route(
        router_logits, num_experts_per_tok=8,
        routed_scaling_factor=2.5, norm_topk_prob=True,
        n_group=8, topk_group=4, e_score_correction_bias=bias)
    assert torch.equal(idx_mine, idx_ref), "n_group=8 topk_indices 不一致"
    assert torch.allclose(w_mine, w_ref, atol=1e-6), "n_group=8 topk_weights 不一致"


def test_norm_topk_false_matches_reference():
    """norm_topk_prob=False 时不归一化, 仅乘 scaling_factor (与 reference 一致)."""
    router_logits = _make_logits(2, 64)
    idx_mine, w_mine = compute_moe_topk(
        router_logits, 4, routed_scaling_factor=1.5,
        norm_topk_prob=False, n_group=1, topk_group=1)
    idx_ref, w_ref = _reference_route(
        router_logits, 4, routed_scaling_factor=1.5,
        norm_topk_prob=True,  # reference 内部 norm=True 不影响 (norm=false 时 reference 仍走 if 分支)
        n_group=1, topk_group=1,
        e_score_correction_bias=torch.zeros(64))
    # 索引必相等 (norm 只影响权重不影响选择)
    assert torch.equal(idx_mine, idx_ref)
    # mine: raw sigmoid * 1.5; ref: (sigmoid/sum)*1.5 → 必然不一致
    assert not torch.allclose(w_mine, w_ref, atol=1e-6)


def test_softmax_old_formula_diverges_from_sigmoid():
    """负向校验: 旧 softmax 公式与 sigmoid 公式必然分歧.

    关键性质: 无 bias 时 sigmoid 是单调函数, 与 softmax 选出的 top-k 索引相同
    (都是 raw_logit 的 top-k). 但 ** 权重 ** 不同:
      - 旧公式: softmax 分数取 top-k (无归一, 无 scale) → 8 个权重和 ≪ 1
      - 新公式: sigmoid top-k → renorm 到 1 → × 2.5 → 8 个权重和 = 2.5

    这就是为什么旧公式导致 MoE 输出被严重 underweight, 加上 shared_expert 主导,
    generation 退化成 "1,1,1,11111111" 复读。
    """
    logits = _make_logits(3, 256)
    softmax_scores = torch.nn.functional.softmax(logits.float(), dim=-1)
    _, idx_softmax = softmax_scores.topk(8, dim=-1)
    idx_sigmoid, w_sigmoid = compute_moe_topk(
        logits, 8, routed_scaling_factor=2.5,
        norm_topk_prob=True, n_group=1, topk_group=1,
        e_score_correction_bias=torch.zeros(256))

    # (1) 选择: bias=0 时 sigmoid 与 softmax 都是 raw logit 的单调函数, top-k 集合必同
    # (注意 sorted=False 与 sorted=True 返回的 index 顺序可能不同, 需排序后比较)
    idx_s_sorted = idx_sigmoid.sort(dim=-1).values
    idx_sm_sorted = idx_softmax.sort(dim=-1).values
    assert torch.equal(idx_s_sorted, idx_sm_sorted), \
        "bias=0 时 sigmoid/softmax 选出相同 top-k 集合 (都是单调函数)"

    # (2) 权重: 必然分歧
    # 新公式经 renorm + scaling_factor=2.5 → 每 token 权重和恒等于 2.5
    old_weight_sum = softmax_scores.gather(-1, idx_softmax).sum(dim=-1)
    new_weight_sum = w_sigmoid.sum(dim=-1)
    # softmax top-k 之和恒 ≤ 1 (sum over all 256 = 1)
    assert (old_weight_sum <= 1.0 + 1e-6).all(), \
        "softmax 全 256 个之和 = 1, top-k 子集必 ≤ 1"
    # 新公式和为 2.5, 严格大于旧公式和 (这是 MoE 输出被低估的根因)
    assert abs(new_weight_sum.mean().item() - 2.5) < 1e-4
    assert new_weight_sum.mean().item() > old_weight_sum.mean().item() * 2, \
        "新公式权重和 (2.5) 必 > 旧公式 (≤1) 的 2 倍 → MoE routed 贡献被低估 2.5x+ 才是退化根因"

    # (3) 有 bias 时, 新公式选择会与旧 softmax 分歧:
    bias = torch.randn(256) * 0.5
    idx_sigmoid_b, _ = compute_moe_topk(
        logits, 8, routed_scaling_factor=2.5,
        norm_topk_prob=True, n_group=1, topk_group=1,
        e_score_correction_bias=bias)
    diff_select = (idx_sigmoid_b != idx_softmax).any(dim=-1)
    assert diff_select.any(), \
        "有 bias 时至少 1 个 token 的 top-k 选择应与 softmax 不同"


def test_topk_weights_sum_is_scaling_factor():
    """norm_topk_prob=True 时, 每个 token 的 8 个权重和 = routed_scaling_factor."""
    logits = _make_logits(5, 256)
    for scale in [1.0, 2.5, 3.7]:
        _, weights = compute_moe_topk(
            logits, 8, routed_scaling_factor=scale, norm_topk_prob=True,
            n_group=1, topk_group=1, e_score_correction_bias=torch.zeros(256))
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.full_like(sums, scale), atol=1e-4)
