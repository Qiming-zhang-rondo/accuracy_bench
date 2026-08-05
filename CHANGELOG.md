# Changelog

## 2026-07-16 — grouped_dual 修复 + CLI 简化 + top-1 默认

- **grouped_dual bug 修复**: `expert_chunk_size=None` 导致 `range()` 报错; 修复为不传时自动设 `256//卡数`
- **device 参数简化**: `--ref_device`/`--quant_device` 不传时自动从 `--ref_devices[0]`/`--quant_devices[0]` 推导, 8 卡命令不再需要冗余单数参数
- **top_k 默认改 1**: 只诊断最差层, 省时
- **`--rotation_matrix` 使用条件明确**: BF16 ref + 量化 quant 必传; 同量化类型 ref vs quant 不需要传 (旋转自动消去, RotBErr 兜底)
- **GLM-5.1 MoE 推荐 `--layers_per_shard 2`**: BF16 反量化 8 层单卡 OOM
- **CLI 参数指南 HTML 更新**: grouped_dual 10x 加速标注, 对齐矩阵 GLM 列更新

## 2026-07-09 — 产品化 (定界恢复 + A4 + HTML + 入参规整)

- 定界恢复: `inference_check.py` + `adapters/` 从 v2 分支恢复, UT 17/17
- A4 激活扩展: `mxfp4_fake_quant.py` (E2M1, floor OCP even + half-up) + `int4_fake_quant.py` (amax/7 + half-even + [-8,7]), UT 15/15, NPU 实测通过
- HTML 报告 v1: `generate_html_report` 两 section (定界 banner + subgraph 表), UT 15/15
- 入参规整: `--quant_method` 默认改 `dequantize`; 删 `--available_devices` 死参数; 新增 `--model_type` / `--boundary`; `subgraph_locate.py` 删死 import + 改 docstring 为库 API
- CLI 新增 `--activation_quant_type` (5 choices: `W8A8_MXFP8` / `W4A8_MXFP` / `W4A4_MXFP4` / `W4A4_DYNAMIC` / `W4A4_LAOS`)

## 2026-06-26 — V1/V2 合并 + 开源清理

- V2 代码从 `accuracy_checker_v2/` 合并到 `accuracy_checker/` 统一包
- `--l2` 统一路由到 V2 `subgraph_locate` 反事实诊断
- Cache 目录可配置: `--cache_dir` / `ACC_CACHE_DIR` / `.acc_cache/`
- 代码开源清理: msmodelslim 可选化, `print`→`logging`, 阈值统一 0.99
- 暂不含 L0 / Custom / `inference_check` (后续恢复)

## 2026-06-02 — V2 Subgraph 因果诊断

- L2 子图反事实替换 (attn / `mlp.gate` / `shared_experts` / `experts`)
- Scale-aware output patch + combo patch
- GLM-5 MoE 子图路径 + MLA 细粒度拆分 (`--mla_fine`)
- DSA sparse attention 支持 (`topk_indices` 传递)

## 2026-05-27 — V2 Scale-Aware Patch

- SmoothQuant pair structure 分析 + scale-aware output patch
- Weight patch 修正 (按 op 类型选择 s 维度)

## 2026-05-09 — GLM-5.1 W8A8 L1/L2 + Rotation-Aligned Weight Comparison

- `RotationMatrices` 类: 4 矩阵加载
- 3D packed expert 加载 + Rotation-aware L2
- L2 诊断: Router / Indexer / KV Chain

## 2026-05-06 — GLM-5.1 MoE DSA 首次接入

- `GLM5MoEAdapter` 适配器
- HF 路径分片加载修复
