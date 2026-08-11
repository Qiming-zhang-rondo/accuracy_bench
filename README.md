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

## 定界使用实例

定界回答的是“坏输出来自量化权重，还是来自部署框架”。`--quant_model` 运行 Transformers 反量化基线；提供 `--ref_model` 后还会增加 BF16/FP16 基线，用于区分量化回归与 base 模型本身行为。

```bash
# 1. 本地复现：ref + 量化模型 + Transformers generate
ASCEND_RT_VISIBLE_DEVICES=0,1 \
python3 run_accuracy_check.py --mode boundary \
  --ref_model <BF16_REF> --quant_model <QUANT_MODEL> \
  --devices npu:0,1 --dtype bfloat16 \
  --prompt "请计算 17 * 23，并只输出结果" --max_new_tokens 64

# 2. 加入部署框架坏输出：判断 vLLM/MindIE 与 Transformers 是否同样复现
python3 run_accuracy_check.py --mode boundary \
  --ref_model <BF16_REF> --quant_model <QUANT_MODEL> \
  --devices npu:0,1 --framework_name vllm-ascend \
  --framework_bad_output "<从部署服务复制的完整坏输出>" \
  --prompt "<触发该坏输出的原始 prompt>" --max_new_tokens 128

# 3. Chat 模型：messages 必须是合法 JSON，工具会调用 apply_chat_template
python3 run_accuracy_check.py --mode boundary \
  --ref_model <BF16_REF> --quant_model <QUANT_MODEL> --devices npu:0,1 \
  --messages '[{"role":"user","content":"比较 0.1 和 0.01"}]' \
  --thinking none --max_new_tokens 128
```

结果含义：`WEIGHT_OR_QUANTIZATION` = Transformers 量化基线也复现、ref 正常；`INFERENCE_FRAMEWORK` = Transformers 正常、部署框架复现；`BOTH` = ref/量化/框架均复现；`INCONCLUSIVE` = 输出过短或证据不足；`INVALID_RUN` = 某次模型运行未完成。没有 `--framework_bad_output` 时只能验证本地 ref/quant，不能单独证明部署框架有问题。

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

- **模型**: GLM-5.1 (MLA + DSA + MoE, QuaRot) / Qwen3 / Qwen3MoE / Qwen3VL / Qwen3.5 MoE / **Qwen3.6** / **Kimi K3** / **DeepSpec、Speculators DSpark standalone draft（专用 L1）**
  > Kimi K3 已支持 text backbone 的 KDA/MLA、Stable LatentMoE、AttnRes、SiTU 和嵌套 MXFP4 配置。`--kimi_kda_backend auto` 在 Ascend 上会在加载 remote code 前安装纯 PyTorch import shim，并使用原生 PyTorch/NPU 短卷积、gated RMSNorm 和 KDA recurrence；该路径不要求安装 `fla-core`，也不会进入 CANN 8.5.1 无法编译的 FLA Triton 路径。确认环境内核兼容后可显式选 `chunk` 或 `fused_recurrent`。
  > Kimi K3 官方 896-expert `ModuleList` 必须用 `--compare_mode grouped_dual` 跑 L1；MoE 层 L2 暂不物化全部专家，会明确拒绝并提示使用流式 replay/内部 packed 模型。该边界待内网 NPU 回归后继续收敛。
  > GLM-5.2 (head_dim=192, indexer_types) 需按 `reference_glm_version_identification` 自行校验结构和子图兼容性
- **量化格式**: W8A8 / W4A8 / W4A4 / MXFP8 / MXFP4 / compressed-tensors (自动识别)
- **覆盖**: 多模型族 × 6 量化格式 × 多个已验证 bad case (GLM-5.1 W4A8 GT HIT layer 77 o_proj；Kimi K3/Qwen3.6 待内网 NPU 回归)

### DSpark：参数对齐与使用边界

DSpark 是 speculative decoding 的 draft/speculator，不是可独立 `generate` 的目标模型。acc_bench 会在加载权重前强校验 ref/quant 的 `block_size`、`num_anchors`（若配置固定）、draft 层数、目标 hidden 层 ID/顺序、hidden/vocab size、attention heads/head_dim、FFN、RMSNorm、RoPE、mask token、verifier 模型、Markov head、confidence head 和 `sample_from_anchor`；任何一项不一致都会终止，避免比较两套不同 draft 契约。

支持两种 standalone checkpoint 格式：

- DeepSpec 官方格式：`Qwen3DSparkModel` / `Gemma4DSparkModel`。官方仓库当前不是 pip package：clone `deepseek-ai/DeepSpec`、安装其 `requirements.txt`，并把仓库根目录加入 `PYTHONPATH`
- Speculators 标准格式：`DSparkDraftModel` / `speculators_model_type=dspark`，安装 `pip install speculators`。其 `config.json` 中 `speculators_config.verifier.name_or_path` 必须在内网可解析；draft checkpoint 通常不重复保存 embedding/LM head，工具会按官方加载路径从 verifier 补齐

`K3DSparkModel`（例如 Inferact/Kimi-K3-DSpark）、带 `_torchspec_version` 的 TorchSpec checkpoint，以及仅靠 `auto_map` 加载的 SpecForge checkpoint 会被识别，但不会被误当成上述两种格式运行。它们的 forward/runtime 契约不同：Kimi K3 原生格式依赖 vLLM MLA DSpark runtime，且 `block_size` 不在 checkpoint 中，而由部署参数 `speculative_config.num_speculative_tokens` 提供。acc_bench 当前会明确拒绝这类 standalone L1；部署定界请传 verifier/目标模型并提供 `--framework_bad_output`。截至本版本，vLLM Ascend 的 DSpark 仍处于 RFC/开发状态，不能把 CUDA/AMD 上可运行直接等同于 NPU 已可运行。

先用对应 verifier/目标模型导出一条 `.pt` 样本。这个文件不是 DSpark checkpoint 的标准附件，也不会由 acc_bench 从 prompt 猜测生成；它记录 verifier 对一条真实输入执行 forward 时的中间激活。必需字段为 `input_ids`、`hidden_states`（别名 `target_hidden_states`）和 `loss_mask`；Speculators 格式还必须包含 `verifier_last_hidden_states`。`hidden_states` 可为 `[B,S,N,H]` 或 `[B,S,N*H]`，其中 `N` 必须等于配置中的目标层数量。

```python
torch.save({
    "input_ids": input_ids,                       # [B, S]
    "hidden_states": auxiliary_hidden_states,    # [B, S, N, H] 或 [B, S, N*H]
    "loss_mask": loss_mask,                      # [B, S]
    "verifier_last_hidden_states": last_hidden,  # [B, S, H], Speculators 必需
    "document_ids": document_ids,                # 可选；缺省按单文档处理
}, "dspark_sample.pt")
```

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
python3 run_accuracy_check.py --mode l1 --model_type dspark \
  --ref_model <BF16_DSPARK_DRAFT> --quant_model <QUANT_DSPARK_DRAFT> \
  --dspark_sample /path/to/dspark_sample.pt --dspark_seed 0 --dspark_max_anchors 8 \
  --ref_device npu:0 --quant_device npu:1 \
  --compare_mode dual --quant_method dequantize --dtype bfloat16
```

`--dspark_max_anchors` 控制单次诊断的显存峰值，工具还会按样本有效 anchor 数和 checkpoint 上限自动下调。当前边界：standalone DSpark 支持 draft backbone、Markov head、confidence head 和 draft logits 的 L1 对齐；不支持普通 CausalLM L2、prompt-only、`full`/`boundary` 或 `fake_quant`。对“部署 DSpark 后是否仍有坏输出”做定界时，`--quant_model` 应传 verifier/目标模型（或可独立生成的融合 checkpoint），并把 DSpark 部署输出作为 `--framework_bad_output`；不要把 standalone draft 当成生成模型。

## 使用约束

1. **必须提供 FP16/BF16 ref 模型** (不支持无 ref 比对)
2. **L2 前必须先跑 L1** (`--l1 --cache_top_k N` 或 `--l1 --l1_target_layers ...`)
3. **`--rotation_matrix`**: 只在 checkpoint 确实使用 QuaRot 等旋转且 R 未融合进权重时传；普通非旋转 W8A8 不传。不能仅凭“BF16 ref vs W8A8 quant”判断存在旋转
   > 前提: R 独立可逆且 ref/quant 使用同一旋转契约；不确定时先检查量化配置与权重生成参数
4. **MoE 强烈推荐 `grouped_dual`**: 8 卡并行 expert chunk, L1 从 70min → 7min (10x)；Kimi K3 为必选
5. **`dtype`** 仅 `bfloat16`/`float16`; NPU 推荐 `bfloat16` (Cube 原生), `float16` 注意激活溢出; ref 与 quant 必须一致; 模型自动 `eval()`
6. **L1 对 MoE router/DSA 层有已知 false-positive** (router softmax 附近 cos_sim 常偏低, 不一定是量化真正出错), 建议配 `v2_metrics` 的 `router_flip_risk` 信号交叉筛
7. **DSpark 必须提供 verifier hidden-state 样本**: 不接受随机 hidden 或仅 prompt；`grouped_dual` 对 dense draft 无意义，使用专用 dual-device 路径

> Cache 机制: 默认 `./.acc_cache/`, 可 `--cache_dir` 或 `ACC_CACHE_DIR` 环境变量覆盖
> 
> HTML 报告: `l1` / `l2` / `full` 成功后都会生成 `report_data.json` 与 `product_report.html`，并按运行独立归档；未指定 `--output_dir` 时写入带时间戳的 `reports/<model>_<mode>_<time>/`。即使重复指定同一非空目录，也会自动使用带时间戳的同级目录，避免覆盖历史
> 
> 当前与历史报告统一入口: 在项目根目录运行 `python3 -m http.server 8765 --bind 0.0.0.0`，桌面浏览器固定打开 `http://[your_ip]:8765/latest.html`；页面默认显示最新结果，左侧边栏可切换全部历史记录
> 
> 交互式命令生成器与完整参数表在项目根目录运行 `python3 -m http.server 8765 --bind 0.0.0.0`，桌面浏览器固定打开 `http://[your_ip]:8765/cli_params_guide.html`；也可运行 `python3 run_accuracy_check.py --help`
> 
> 默认不打印 hidden norm / MoE L3 expert 等开发上下文；需要这些诊断时追加 `--debug`

## 架构

```
run_accuracy_check.py (7 modes: screening/boundary/l1/l2/full/report/inference)
├── L1: ShardedBlockComparator (layer1_block_compare.py)
│   ├── model_loader.py — 分片加载 / 3D expert / 反量化
│   ├── dspark.py — draft 参数契约 / verifier cache / 专用 L1
│   ├── *_fake_quant.py — MXFP8/MXFP4/INT4 激活伪量化
│   └── model_structure.py — 统一结构/能力探测（含多模态 wrapper、Kimi K3）
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
- **多轮 generate N tokens 对齐**: 未支持 (路标)

> 详见 `docs/roadmap.md` 和 `docs/capability_gap.md`

## Contributing

- 新模型接入: 优先扩展 `model_structure.py` 的稳定结构能力和 `subgraph_locate.py` 的子图能力；不要按模型名复制一套 Adapter
- 坏算子经验库共建: 欢迎提 issue 沉淀 "坏层 → 坏算子 → 修复建议" 案例

## 更新日志

见 `CHANGELOG.md`
