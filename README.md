# acc_bench

量化模型精度对齐工具 — **L1 找坏层 → 定界区分框架/权重 → L2 反事实验证根因算子**。面向 Ascend NPU 量化工程师，从"精度掉了不知道哪层"到"L2 反事实定位根因算子"（8 卡 grouped_dual 典型 ~5min, 依赖 cache hit）。

## 谁该用

- **量化算法工程师**: 关心根因 → L2 反事实定位 RootSuspect op
- **部署工程师**: 关心哪层坏 → L1 找 cos_sim 最低层
- **框架开发**: 关心误差来自量化还是推理框架 → 定界 (反量化 → generate)

## TL;DR

> **前置**: `npu-smi info` 确认卡空闲; `OMP_NUM_THREADS=8` 防 futex livelock (streaming expert 必设)

```bash
# 一键全流程 (L1→Top-K→L2→logits→HTML, 推荐)
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=8 \
python3 run_accuracy_check.py --mode full \
  --ref_model <REF> --quant_model <QUANT> \
  --ref_devices npu:0,1,2,3 --quant_devices npu:4,5,6,7 \
  --compare_mode grouped_dual --layers_per_shard 2 --cache_top_k 1 --top_k 1 \
  --quant_method dequantize \
  --rotation_matrix <ROT>  # BF16 ref + 量化 quant 必传; 同量化类型不传

# L1 only (找坏层 + cache 供 L2 用)
python3 run_accuracy_check.py --l1 --ref_model <REF> --quant_model <QUANT> \
  --ref_devices npu:0,1,2,3 --quant_devices npu:4,5,6,7 \
  --compare_mode grouped_dual --layers_per_shard 2 --cache_top_k 1 \
  --quant_method dequantize --rotation_matrix <ROT>

# L2 only (L1 已跑过, 诊断候选层子图)
python3 run_accuracy_check.py --l2 --target_layers 11 20 33 \
  --ref_model <REF> --quant_model <QUANT> \
  --ref_devices npu:0,1,2,3 --quant_devices npu:4,5,6,7 \
  --compare_mode grouped_dual --quant_method dequantize \
  --rotation_matrix <ROT> [--mla_fine]
```

## 能力

| 能力 | 干什么 | 状态 |
|------|--------|------|
| **L1** | 逐 decoder block 比 hidden_states, 找 cos_sim 最低层 | ✅ |
| **定界** | 反量化 (BF16 round-trip, 非无损) → generate, 区分"量化坏" vs "推理框架坏" | ✅ |
| **L2** | 子图反事实替换, 测 Recovery Ratio, 输出 RootSuspect (假设: 候选子图为误差**充足原因**, 不作必要性主张) | ✅ |
| **HTML 报告** | 定界 banner + subgraph 表, 自包含, XSS 转义 | ✅ |
| **A4 激活量化** | MXFP4 (E2M1) + INT4 per-token, W4A4 全链路 | ✅ |
| **Bad Case 工作流** | manifest + 单点 clip + GT 对比 | ✅ |

## 对比

| 工具 | L1 找坏层 | 定界 (框架 vs 权重) | L2 反事实根因 | 端到端评测 |
|------|----------|---------------------|---------------|-----------|
| **acc_bench** | ✅ | ✅ | ✅ | — |
| vllm-ascend 自带 | basic | ✗ | ✗ | — |
| msmodelslim | ✗ | ✗ | ✗ | — |
| ais_bench | ✗ | ✗ | ✗ | ✅ |

> acc_bench 差异化: L2 反事实子图诊断（Recovery Ratio + RootSuspect）是其它工具没有的。

## 支持范围

- **模型**: GLM-5.1 (MLA + DSA + MoE, QuaRot) / Qwen3 / Qwen3MoE / Qwen3VL / Qwen3.5 MoE
  > GLM-5.2 (head_dim=192, indexer_types) 需按 `reference_glm_version_identification` 自行校验 adapter 兼容性
- **量化格式**: W8A8 / W4A8 / W4A4 / MXFP8 / MXFP4 / compressed-tensors (自动识别)
- **覆盖**: 5 模型族 × 6 量化格式 × 多个已验证 bad case (GLM-5.1 W4A8 GT HIT layer 77 o_proj)

## 使用约束

1. **必须提供 FP16/BF16 ref 模型** (不支持无 ref 比对)
2. **L2 前必须先跑 L1** (`--l1 --cache_top_k N` 或 `--l1 --l1_target_layers ...`)
3. **`--rotation_matrix`**: ref 与 quant 量化方案不同 (BF16 ref vs W8A8 quant) → 必传; 同方案 (W8A8 vs W8A8) → 不传 (RotBErr 兜底)
   > 前提: R 独立可逆且 ref/quant 用同一 R; 若 R 已融合进权重则必须显式传
4. **MoE 强烈推荐 `grouped_dual`**: 8 卡并行 expert chunk, L1 从 70min → 7min (10x)
5. **`dtype`** 仅 `bfloat16`/`float16`; NPU 推荐 `bfloat16` (Cube 原生), `float16` 注意激活溢出; ref 与 quant 必须一致; 模型自动 `eval()`
6. **L1 对 MoE router/DSA 层有已知 false-positive** (router softmax 附近 cos_sim 常偏低, 不一定是量化真正出错), 建议配 `v2_metrics` 的 `router_flip_risk` 信号交叉筛

> 详细 CLI 参数见 `cli_params_guide.html` 或 `python3 run_accuracy_check.py --help`
> Cache 机制: 默认 `./.acc_cache/`, 可 `--cache_dir` 或 `ACC_CACHE_DIR` 环境变量覆盖
> HTML 报告: `accuracy_checker.html_report.generate_html_report(boundary_results=..., l2_results=..., model_name=..., output_path=...)`

## 架构

```
run_accuracy_check.py (7 modes: screening/boundary/l1/l2/full/report/inference)
├── L1: ShardedBlockComparator (layer1_block_compare.py)
│   ├── model_loader.py — 分片加载 / 3D expert / 反量化
│   ├── *_fake_quant.py — MXFP8/MXFP4/INT4 激活伪量化
│   └── adapters/ — GLM5MoE / Qwen3 / Qwen3MoE / Qwen3VL / Qwen3.5MoE
├── 定界: inference_check.py — NPU 加速反量化 → generate → 重复检测
├── L2: subgraph_locate.py — 反事实子图诊断
│   ├── operator_patcher.py / weight_patcher.py — output/weight patch
│   ├── replay_provider.py — DSA topk_indices 传递
│   └── v2_metrics.py — 6+3 筛选指标
└── html_report.py / inference_html.py — 报告生成
```

> 详见 `docs/architecture.md`

## 不足 / 路标

- **L0 完整性校验**: 未合入 (路标)
- **Custom 模型** (DeepSeek V3.2/V4): 未合入 (路标)
- **`--boundary` CLI**: 占位, 通过 `scripts/glm5_inference_check.py` 或 Python API `from accuracy_checker.inference_check import hf_inference_check` 调用
- **多轮 generate N tokens 对齐**: 未支持 (路标)

> 详见 `docs/roadmap.md` 和 `docs/capability_gap.md`

## Contributing

- 新模型 adapter: 继承 `accuracy_checker/adapters/base.py:BaseModelAdapter`, 在 `adapters/__init__.py` 注册 (参考 `GLM5MoEAdapter` / `Qwen3Adapter`)
- 坏算子经验库共建: 欢迎提 issue 沉淀 "坏层 → 坏算子 → 修复建议" 案例

## 更新日志

见 `CHANGELOG.md`
