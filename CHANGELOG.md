# Changelog

## 2026-08-11 — Kimi K3 Ascend KDA fallback

- 新增 `--kimi_kda_backend auto|torch|chunk|fused_recurrent`；Ascend `auto` 默认使用无 Triton 依赖的 eager torch KDA recurrence
- torch fallback 对齐 FLA KDA 的 q/k L2Norm、gate、beta、delta-rule、GVA 与 state layout 契约
- shard forward 失败后不再调用会同步设备的 `empty_cache()`，避免 NPU 异常 context 将原始错误覆盖成 507014 timeout
- 参数页接入 KDA backend 推荐、预检状态和命令生成

## 2026-08-06 — W8A8 streaming 修复 + CLI 命令生成器

- 修复 `grouped_dual` streaming expert 将普通 W8A8/W8A8_DYNAMIC INT8 权重直接 cast 到浮点、未执行反量化的问题；统一复用普通层的反量化分派
- 修复 msModelSlim 静态 W8A8 `deq_scale` 的 Ascend int32→int64 位模式恢复（按 float32 位模式解释，而非错误的 float64）并补充 W8A8S
- 修正多个静态 W8A8 调用把 `dtype` 误传给 `quant_bias` 位置的问题
- 首层已灾难性失真时不再把后续随机局部波动报告成 first bad block，优先提示加载/反量化契约错误
- 新增交互式 `cli_params_guide.html`：输入 ref/quant 路径后推荐模型类型、设备拆分和命令，内置 Qwen3.6、Kimi K3、DSpark 与 boundary 场景
- Layer 3 expert、hidden norm 与专项调试默认移到 DEBUG；新增 `--debug`

## 2026-08-06 — DSpark standalone draft L1

- 运行 DeepSpec (`Qwen3DSparkModel` / `Gemma4DSparkModel`) 与标准 Speculators (`DSparkDraftModel`) checkpoint；识别并明确拒绝 forward 契约不同的 K3/TorchSpec/remote SpecForge 格式，避免误加载
- ref/quant 权重加载前强校验 block、目标 hidden 层、hidden/vocab、Markov/confidence head 等参数契约
- 新增 `--dspark_sample` / `--dspark_seed` / `--dspark_max_anchors`，使用 verifier 导出的真实 hidden-state cache 对齐 draft layers、Markov/confidence head 与 logits
- standalone DSpark 明确定界为专用 L1；普通 CausalLM L2、prompt-only、boundary/full 与 fake-quant fail-fast
- README 增加定界命令、判定结果解释和 DSpark cache/运行示例

## 2026-08-05 — Qwen3.6 / Kimi K3 + 结构探测统一

- 删除未完整接入主流程的两套 Adapter 抽象，统一由 `model_structure.py` 解析模型容器、文本层和跨层状态
- Qwen3.6: 保持官方 `qwen3_5_moe` 自动识别，新增 `qwen3_6` / `qwen3_6_moe` CLI alias
- Kimi K3: 支持 `language_model.model.layers`、KDA/MLA、`block_sparse_moe`、Stable LatentMoE、SiTU、top-16 router、AttnRes
- compressed-tensors: 支持从嵌套 `text_config` 识别量化配置及 MXFP4 `weight_packed` 反量化
- L1→L2 cache 修正为保存目标层输入，并可携带 AttnRes 跨层状态；cache format 升级到 v3
- eager replay 为 Kimi MLA / Qwen3.6 full attention 构建 causal mask；仅在层签名支持时传递跨层状态
- Kimi K3 官方 ModuleList experts: L1 强制 `grouped_dual` 流式读取；MoE 层 L2 在流式 replay 合入前 fail-fast，避免整层 OOM

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
