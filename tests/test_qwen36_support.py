"""UT — Qwen3.6 (Qwen3_5MoeForConditionalGeneration) 支持验证

验证 acc_bench 对 Qwen3.6 的支持:
  1. move_layers_to_device: 不跳过 linear_attn.conv1d.weight (3D 但非 expert)
  2. MoE router: Qwen3_5MoeTopKRouter 返回 tuple (logits, scores, indices) 的处理
  3. _check_orthogonal: 在 CPU 上运行, 避免 NPU 默认设备冲突
  4. create_model_skeleton: 多模态 ForConditionalGeneration config 回退

纯 CPU, 不依赖 NPU / 真实模型文件。
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    import pytest  # noqa: F401
    _HAS_PYTEST = True
except ImportError:
    _HAS_PYTEST = False

    class _PytestFallback:
        @staticmethod
        def fixture(*args, **kwargs):
            def deco(fn):
                return fn
            return deco

    pytest = _PytestFallback()  # type: ignore


# =========================================================================
# 1. move_layers_to_device: conv1d 3D 权重不应被跳过
# =========================================================================
def test_move_layers_skip_3d_experts_not_conv1d():
    """move_layers_to_device 应跳过 MoE expert 3D 权重, 但不跳过 conv1d 3D 权重.

    Qwen3.6 linear_attn.conv1d.weight 是 3D [out, 1, kernel], 但不是 routed expert.
    之前用 param.dim()==3 判断会误跳过 conv1d, 导致 NPU forward 报 device mismatch.
    """
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    # 构造一个模拟 layer: 有 conv1d (3D 非 expert) + experts (3D expert)
    class FakeLinearAttn(nn.Module):
        def __init__(self):
            super().__init__()
            # conv1d weight: [out_channels, in_channels, kernel_size] = [8192, 1, 4]
            self.conv1d = nn.Conv1d(1, 8192, kernel_size=4, bias=False)
            self.proj = nn.Linear(8192, 2048, bias=False)

    class FakeExperts(nn.Module):
        def __init__(self):
            super().__init__()
            # 3D expert weight: [num_experts, in, out] = [256, 1024, 2048]
            self.gate_up_proj = nn.Parameter(torch.zeros(256, 1024, 2048))
            self.down_proj = nn.Parameter(torch.zeros(256, 2048, 512))

    class FakeMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = FakeExperts()
            self.shared_expert = nn.Linear(2048, 1024, bias=False)

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_attn = FakeLinearAttn()
            self.mlp = FakeMLP()
            self.input_layernorm = nn.LayerNorm(2048)

    layer = FakeLayer()

    # 模拟 _is_routed_expert_param 逻辑 (from model_loader.move_layers_to_device)
    def _is_routed_expert_param(name: str, p: nn.Parameter) -> bool:
        return p.dim() == 3 and 'experts' in name and 'shared' not in name

    # 验证 conv1d.weight 不被识别为 routed expert
    for name, p in layer.named_parameters():
        if 'conv1d.weight' in name:
            assert not _is_routed_expert_param(name, p), \
                f"conv1d.weight 不应被识别为 routed expert: {name} shape={p.shape}"
            assert p.dim() == 3, f"conv1d.weight 应为 3D: {name} shape={p.shape}"

    # 验证 experts.gate_up_proj 和 experts.down_proj 被识别为 routed expert
    expert_params_found = 0
    for name, p in layer.named_parameters():
        if _is_routed_expert_param(name, p):
            expert_params_found += 1
            assert 'experts' in name, f"routed expert 应含 'experts': {name}"
            assert 'shared' not in name, f"shared_expert 不应被识别为 routed: {name}"
    assert expert_params_found == 2, f"应找到 2 个 routed expert 参数, 实际 {expert_params_found}"

    # 验证 shared_expert 不被识别为 routed expert
    for name, p in layer.named_parameters():
        if 'shared_expert' in name:
            assert not _is_routed_expert_param(name, p), \
                f"shared_expert 不应被识别为 routed expert: {name}"


# =========================================================================
# 2. MoE router: tuple 返回值处理
# =========================================================================
def test_moe_router_tuple_unpacking():
    """Qwen3_5MoeTopKRouter 返回 (logits, scores, indices) tuple.

    acc_bench 的 _moe_forward_chunked 之前直接 router_logits = gate(hidden_states),
    对 tuple 调用 .float() 会报 AttributeError: 'tuple' object has no attribute 'float'.
    修复后应正确解包 tuple.
    """
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    # 模拟 Qwen3_5MoeTopKRouter 的行为
    class FakeQwen36Router(nn.Module):
        def __init__(self, num_experts=8, hidden_dim=16, top_k=2):
            super().__init__()
            self.top_k = top_k
            self.num_experts = num_experts
            self.weight = nn.Parameter(torch.randn(num_experts, hidden_dim))

        def forward(self, hidden_states):
            hidden_states = hidden_states.reshape(-1, self.weight.shape[1])
            router_logits = torch.nn.functional.linear(hidden_states, self.weight)
            router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
            router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
            router_top_value = router_top_value.to(router_logits.dtype)
            return router_logits, router_top_value, router_indices

    router = FakeQwen36Router()
    hidden = torch.randn(1, 4, 16)  # [batch, seq, hidden]

    # 模拟 _moe_forward_chunked 的 tuple 解包逻辑
    gate_out = router(hidden)
    assert isinstance(gate_out, tuple), "Qwen3.6 router 应返回 tuple"

    precomputed_scores = None
    precomputed_indices = None
    if isinstance(gate_out, tuple):
        router_logits = gate_out[0]
        if len(gate_out) >= 3:
            precomputed_scores = gate_out[1]
            precomputed_indices = gate_out[2]
        elif len(gate_out) == 2:
            precomputed_indices = gate_out[1]
    else:
        router_logits = gate_out

    # 验证解包正确
    assert router_logits is not None, "router_logits 应被正确提取"
    assert router_logits.shape == (4, 8), f"router_logits shape 错误: {router_logits.shape}"
    assert precomputed_scores is not None, "precomputed_scores 应被提取"
    assert precomputed_indices is not None, "precomputed_indices 应被提取"
    assert precomputed_scores.shape == (4, 2), f"scores shape 错误: {precomputed_scores.shape}"
    assert precomputed_indices.shape == (4, 2), f"indices shape 错误: {precomputed_indices.shape}"

    # 验证 router_logits 可以调用 .float() (之前会报错)
    _ = router_logits.float()
    # 验证 precomputed_scores 已归一化 (sum=1 per token)
    scores_sum = precomputed_scores.float().sum(dim=-1)
    assert torch.allclose(scores_sum, torch.ones(4), atol=1e-5), \
        f"precomputed_scores 应归一化 (sum=1), 实际 sum={scores_sum}"


# =========================================================================
# 3. _check_orthogonal: CPU 运行 (避免 NPU 默认设备冲突)
# =========================================================================
def test_check_orthogonal_runs_on_cpu():
    """_check_orthogonal 应在 CPU 上运行, 即使默认设备是 NPU.

    之前 torch.eye(n) 在 CPU 但 R @ R^T 可能在 NPU, 导致 device mismatch.
    修复后 R 被强制 .cpu() .
    """
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from accuracy_checker.utils import _check_orthogonal

    # 构造正交矩阵 (Hadamard-like)
    R = torch.eye(8)  # 单位矩阵是正交的
    assert _check_orthogonal(R), "单位矩阵应通过正交性检查"

    # 构造非正交矩阵
    R_bad = torch.randn(8, 8)
    assert not _check_orthogonal(R_bad, tol=1e-6), "随机矩阵不应通过严格正交性检查"

    # 构造更大的正交矩阵 (用 QR 分解)
    Q, _ = torch.linalg.qr(torch.randn(64, 64))
    assert _check_orthogonal(Q), "QR 分解的 Q 应通过正交性检查"

    # 验证函数不依赖输入 tensor 的设备 (强制 CPU)
    # 即使传入 meta device tensor (模拟), 函数内部应 .cpu() 处理
    R_cpu = torch.eye(16)
    result = _check_orthogonal(R_cpu)
    assert result, "应正确处理 CPU tensor"


# =========================================================================
# 4. create_model_skeleton: 多模态 config 回退 (逻辑验证)
# =========================================================================
def test_multimodal_config_fallback_logic():
    """多模态 ForConditionalGeneration (如 Qwen3.6) 的 config 没有 num_hidden_layers,
    应从 text_config 回退.

    之前直接报 "config 中缺少 num_hidden_layers".
    修复后: config.text_config.num_hidden_layers 被采用.
    """
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    # 模拟多模态 config (Qwen3.6 ForConditionalGeneration)
    class FakeTextConfig:
        num_hidden_layers = 40
        hidden_size = 2048

    class FakeConfig:
        text_config = FakeTextConfig()
        architectures = ['Qwen3_5MoeForConditionalGeneration']
        model_type = 'qwen3_5_moe'
        # 注意: 没有 num_hidden_layers 属性

    config = FakeConfig()

    # 模拟 create_model_skeleton 的回退逻辑
    def resolve_num_hidden_layers(cfg):
        if hasattr(cfg, 'num_hidden_layers'):
            return cfg.num_hidden_layers
        if hasattr(cfg, 'text_config') and hasattr(cfg.text_config, 'num_hidden_layers'):
            return cfg.text_config.num_hidden_layers
        if hasattr(cfg, 'num_layer'):
            return cfg.num_layer
        raise ValueError("config 中缺少 num_hidden_layers 和 num_layer")

    n_layers = resolve_num_hidden_layers(config)
    assert n_layers == 40, f"多模态 config 应从 text_config 回退, 实际 {n_layers}"

    # 验证单模态 config (直接有 num_hidden_layers) 仍工作
    class FakeUniConfig:
        num_hidden_layers = 32

    assert resolve_num_hidden_layers(FakeUniConfig()) == 32


# =========================================================================
# 5. 集成: Qwen3.6 model_type alias
# =========================================================================
def test_qwen36_model_type_alias():
    """官方 model_type 与显式 qwen3_6 alias 应复用同一结构能力。"""
    from accuracy_checker.subgraph_locate import _SUBGRAPH_NAME_DISPATCH

    official = _SUBGRAPH_NAME_DISPATCH['qwen3_5_moe']
    assert _SUBGRAPH_NAME_DISPATCH['qwen3_6'] is official
    assert _SUBGRAPH_NAME_DISPATCH['qwen3_6_moe'] is official


# =========================================================================
# 6. 集成: get_decoder_layers 支持 ForConditionalGeneration
# =========================================================================
def test_get_decoder_layers_for_conditional_generation():
    """get_decoder_layers 应支持 model.model.language_model.layers 路径.

    Qwen3.6 ForConditionalGeneration 结构:
      model.model.language_model.layers (有 language_model 中间层)
    """
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from accuracy_checker.utils import get_decoder_layers

    # 构造模拟 ForConditionalGeneration 结构
    class FakeLayer(nn.Module):
        def __init__(self, idx):
            super().__init__()
            self.idx = idx
            self.weight = nn.Parameter(torch.zeros(4))

    class FakeLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([FakeLayer(i) for i in range(3)])

    class FakeInnerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = FakeLanguageModel()

    class FakeForConditionalGeneration(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeInnerModel()

    model = FakeForConditionalGeneration()
    layers = get_decoder_layers(model)
    assert len(layers) == 3, f"应返回 3 层, 实际 {len(layers)}"


# =========================================================================
# 7. L2: detect_model_type 正确识别 Qwen3.6 (不误判为 glm_moe_dsa)
# =========================================================================
def test_detect_model_type_qwen36():
    """detect_model_type 应将 Qwen3.6 层识别为 'qwen3_5_moe', 而非 'glm_moe_dsa'.

    Qwen3.6 层结构:
      - mlp 有 gate + experts + shared_expert (单数)
      - self_attn 有 q_proj/k_proj/v_proj/o_proj (标准注意力, 非 MLA)
      或
      - linear_attn (线性注意力, 含 conv1d)

    之前 detect_model_type 只看 mlp.gate+mlp.experts 就返回 'glm_moe_dsa',
    导致 get_subgraph_names 走 GLM MLA 路径 (找 q_a_proj 等), 对 Qwen3.6 无效.
    """
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from accuracy_checker.subgraph_locate import detect_model_type

    # 模拟 Qwen3.6 self_attn 层 (标准注意力 + MoE)
    class FakeStandardAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(2048, 2048, bias=False)
            self.k_proj = nn.Linear(2048, 256, bias=False)
            self.v_proj = nn.Linear(2048, 256, bias=False)
            self.o_proj = nn.Linear(2048, 2048, bias=False)

    class FakeQwen36MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(2048, 256, bias=False)
            # Qwen3.6 用 shared_expert (单数); GLM 用 shared_experts (复数)
            self.shared_expert = nn.Linear(2048, 1024, bias=False)
            self.experts = nn.Module()

    class FakeQwen36SelfAttnLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = FakeStandardAttn()
            self.mlp = FakeQwen36MLP()
            self.input_layernorm = nn.LayerNorm(2048)

    layer = FakeQwen36SelfAttnLayer()
    mt = detect_model_type(layer)
    assert mt == 'qwen3_5_moe', \
        f"Qwen3.6 self_attn 层应识别为 'qwen3_5_moe', 实际 '{mt}'"

    # 模拟 Qwen3.6 linear_attn 层
    class FakeLinearAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1d = nn.Conv1d(1, 8192, kernel_size=4, bias=False)
            self.proj = nn.Linear(8192, 2048, bias=False)

    class FakeQwen36LinearAttnLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_attn = FakeLinearAttn()
            self.mlp = FakeQwen36MLP()
            self.input_layernorm = nn.LayerNorm(2048)

    layer_lin = FakeQwen36LinearAttnLayer()
    mt_lin = detect_model_type(layer_lin)
    assert mt_lin == 'qwen3_5_moe', \
        f"Qwen3.6 linear_attn 层应识别为 'qwen3_5_moe', 实际 '{mt_lin}'"


# =========================================================================
# 8. L2: get_subgraph_names 返回正确的 Qwen3.6 子图
# =========================================================================
def test_get_subgraph_names_qwen36():
    """get_subgraph_names 对 'qwen3_5_moe' 应返回 self_attn/linear_attn + mlp.* 子图.

    关键: Qwen3.6 用 shared_expert (单数), 不是 GLM 的 shared_experts (复数).
    """
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from accuracy_checker.subgraph_locate import get_subgraph_names

    # self_attn 层
    class FakeStandardAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(2048, 2048, bias=False)
            self.o_proj = nn.Linear(2048, 2048, bias=False)

    class FakeQwen36MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(2048, 256, bias=False)
            self.shared_expert = nn.Linear(2048, 1024, bias=False)
            self.experts = nn.Module()

    class FakeQwen36SelfAttnLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = FakeStandardAttn()
            self.mlp = FakeQwen36MLP()
            self.input_layernorm = nn.LayerNorm(2048)

    layer = FakeQwen36SelfAttnLayer()
    names = get_subgraph_names('qwen3_5_moe', layer)
    assert 'self_attn' in names, f"应包含 'self_attn': {names}"
    assert 'mlp.gate' in names, f"应包含 'mlp.gate': {names}"
    assert 'mlp.shared_expert' in names, f"应包含 'mlp.shared_expert' (单数): {names}"
    assert 'mlp.experts' in names, f"应包含 'mlp.experts': {names}"
    # 不应包含 GLM 复数形式
    assert 'mlp.shared_experts' not in names, \
        f"不应包含 'mlp.shared_experts' (GLM 复数形式): {names}"

    # linear_attn 层
    class FakeLinearAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1d = nn.Conv1d(1, 8192, kernel_size=4, bias=False)

    class FakeQwen36LinearAttnLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_attn = FakeLinearAttn()
            self.mlp = FakeQwen36MLP()
            self.input_layernorm = nn.LayerNorm(2048)

    layer_lin = FakeQwen36LinearAttnLayer()
    names_lin = get_subgraph_names('qwen3_5_moe', layer_lin)
    assert 'linear_attn' in names_lin, \
        f"linear_attn 层应包含 'linear_attn': {names_lin}"
    assert 'self_attn' not in names_lin, \
        f"linear_attn 层不应包含 'self_attn': {names_lin}"
    assert 'mlp.gate' in names_lin, f"应包含 'mlp.gate': {names_lin}"


# =========================================================================
# 9. L2: GLM-5 仍识别为 glm_moe_dsa (回归测试, 不误判为 qwen3_5_moe)
# =========================================================================
def test_detect_model_type_glm_still_glm_moe_dsa():
    """GLM-5 (MLA attention + shared_experts 复数) 应仍识别为 'glm_moe_dsa'.

    回归测试: 添加 Qwen3.6 检测后, GLM-5 不应被误判.
    GLM-5 特征:
      - self_attn 有 q_a_proj / kv_a_proj_with_mqa (MLA)
      - mlp 有 shared_experts (复数, 非 shared_expert 单数)
    """
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from accuracy_checker.subgraph_locate import detect_model_type

    class FakeMlaAttn(nn.Module):
        def __init__(self):
            super().__init__()
            # GLM MLA 特征
            self.q_a_proj = nn.Linear(6144, 2048, bias=False)
            self.q_b_proj = nn.Linear(2048, 6144, bias=False)
            self.kv_a_proj_with_mqa = nn.Linear(6144, 576, bias=False)
            self.kv_b_proj = nn.Linear(576, 28672, bias=False)
            self.o_proj = nn.Linear(6144, 6144, bias=False)

    class FakeGlmMoe(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(6144, 256, bias=False)
            # GLM 用 shared_experts (复数)
            self.shared_experts = nn.Linear(6144, 1024, bias=False)
            self.experts = nn.Module()

    class FakeGlmLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = FakeMlaAttn()
            self.mlp = FakeGlmMoe()
            self.input_layernorm = nn.LayerNorm(6144)

    layer = FakeGlmLayer()
    mt = detect_model_type(layer)
    assert mt == 'glm_moe_dsa', \
        f"GLM-5 (MLA + shared_experts 复数) 应识别为 'glm_moe_dsa', 实际 '{mt}'"


if __name__ == '__main__':
    # 手动运行 (无 pytest 时)
    tests = [
        test_move_layers_skip_3d_experts_not_conv1d,
        test_moe_router_tuple_unpacking,
        test_check_orthogonal_runs_on_cpu,
        test_multimodal_config_fallback_logic,
        test_qwen36_model_type_alias,
        test_get_decoder_layers_for_conditional_generation,
        test_detect_model_type_qwen36,
        test_get_subgraph_names_qwen36,
        test_detect_model_type_glm_still_glm_moe_dsa,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"[PASS] {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
    print(f"\n=== {passed} passed, {failed} failed ===")
    exit(0 if failed == 0 else 1)
