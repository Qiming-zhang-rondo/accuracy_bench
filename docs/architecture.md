# Accuracy Bench 架构文档

> 量化精度对齐工具 — 开源精简版
> 分支: `accuracy_checker_v2_clean`
> 最后更新: 2026-08-05

---

## 1. 概述

本工具对量化模型做**逐层精度对齐诊断**，采用 L1/L2 两级漏斗：

| 级别 | 问题 | 方法 |
|------|------|------|
| **L1** | "哪一层的 hidden_states 和 ref 不一致？" | 双卡分片逐层 forward，对比 hidden_states (cos_sim / rel_l2 / 熵 / KL) |
| **L2** | "问题层内哪个子图是误差源？" | 反事实替换：把 quant 子图 output 换成 ref 的，测恢复比例 |

L1 定位"哪一层有问题"，L2 在问题层内定位"哪个子图有问题"。L2 复用 L1 缓存的 hidden_states 作为层输入，避免重新 forward prefix 层，且天然支持旋转模型（quant hidden 已在旋转空间）。

### 与内部完整版的差异

本分支是开源精简版，相对内部完整版做了以下裁剪：

| 项 | 内部完整版 | 开源精简版 |
|---|---|---|
| V2 代码组织 | 独立 `accuracy_checker_v2/` 目录 | **合并进 `accuracy_checker/`** |
| PatchUnit / LocalPatchRunner | dataclass + Runner 类 | **重构为函数式** (`diagnose_layer(s)`) |
| Custom 路径 (DeepSeek V4/V3.2) | ✅ | ❌ 移除 |
| L0 模型完整性校验 | ✅ | ❌ 移除 |
| HF inference_check | ✅ | ❌ 移除 |
| msmodelslim 依赖 | ✅ | ❌ 移除 (third_party 可选) |

---

## 2. 目录结构

```
acc_bench/
├── run_accuracy_check.py          # 统一入口 (L1 + L2 自动衔接)
├── accuracy_checker/
│   ├── __init__.py
│   ├── subgraph_locate.py         # [1581行] L2 主逻辑: 反事实子图诊断
│   ├── replay_provider.py         # [547行]  双卡 ref/quant 单层 replay
│   ├── operator_patcher.py        # [65行]   ReplacementHook (output 替换)
│   ├── v2_metrics.py              # [51行]   rel_l2 / cos_sim / recovery_ratio
│   ├── report.py                  # [77行]   AlignmentReport (L1+L2 汇总)
│   ├── layer1_block_compare.py    #          L1: ShardedBlockComparator
│   ├── layer2_module_compare.py   #          V1 L2: rotation-aligned weight 对比
│   ├── model_loader.py            #          HF 模型加载 (3D expert, indexed)
│   ├── model_structure.py         #          统一结构/能力探测
│   ├── hooks.py                   #          HookManager / ActivationCollector
│   ├── metrics.py                 #          V1 指标 (cos_sim / snr / procrustes)
│   ├── utils.py                   #          auto_device / rotation 工具
│   └── cache.py                   #          L2 cache 目录管理
├── docs/
│   ├── architecture.md            # 本文档
│   └── sharded_l1_design.md       # L1 分片设计
└── README.md
```

---

## 3. L1: 逐 Block 对比

**入口**: `run_accuracy_check.py --l1`
**核心类**: `ShardedBlockComparator` (layer1_block_compare.py)

### 流程

1. 加载 ref / quant 模型 skeleton（meta device），逐 shard 加载权重到 NPU
2. 对同一 prompt 做 sequential forward，逐层对比 hidden_states
3. 旋转模型：quant 输出经 `unrotate_hidden(R1)` 回到原空间再对比
4. 指标：cos_sim / rel_l2 / 熵 / KL 散度 / top-k match
5. `--cache_top_k N`：把 cos_sim 最低的 N 层的 hidden_states 存入 L2 cache
6. `--l1_target_layers`：只跑指定层，每层 input 自动存 cache

### 关键参数

| 参数 | 作用 |
|------|------|
| `--layers_per_shard N` | 每 shard 层数（显存换速度） |
| `--cache_top_k N` | 缓存最差 N 层供 L2 使用 |
| `--rotation_matrix FILE` | QuaRot 旋转矩阵（R1） |
| `--quant_method dequantize\|fake_quant` | 量化模型加载方式 |
| `--compare_mode dual\|grouped_dual` | dual=双卡分片, grouped_dual=MoE expert 跨卡 |
| `--activation_quant` | 仅在 quant 侧启用 descriptor 驱动的激活伪量化 |
| `--activation_quant_type AUTO\|...` | 默认按每个权重 descriptor 自动选择激活格式；显式类型只匹配兼容算子 |
| `--activation_quant_backend auto\|npu\|torch` | W4A4 在 NPU 上默认调用原生 dynamic-quant 算子；Torch 公式路径仅用于诊断 |

---

## 4. L2: 子图反事实诊断

**入口**: `run_accuracy_check.py --l2`（注：`subgraph_locate.py` 是库模块，无 `__main__`，不能 `python -m` 直接跑）
**核心函数**: `diagnose_layer` / `diagnose_layers` (subgraph_locate.py)

### 核心思想

对 quant 模型正常 forward（含反量化），用 `ReplacementHook` 把某子图 output 替换为 ref 的对应 output，测量 **recovery ratio**（替换后误差下降比例）。recovery 最高的子图 = 主要误差源。

### 子图拆分（模型类型自动检测）

| model_type | auto-detect 条件 | 子图 |
|---|---|---|
| `dense` | 默认 (Qwen3, LLaMA) | `self_attn` \| `mlp` |
| `moe` | `layer.moe` / `block_sparse_moe` 存在 | `self_attn` \| `moe.gate` \| `moe.shared_expert(s)` \| `moe.experts` |
| `glm_moe_dsa` | `layer.mlp` 有 `gate` + `experts` (GLM-5) | `self_attn` \| `mlp.gate` \| `mlp.shared_experts` \| `mlp.experts` |
| `glm_mla` | `self_attn` 有 `q_a_proj`/`kv_a_proj_with_mqa` (dense MLP 层) | `self_attn` \| `mlp` |
| `qwen3_5_moe` / Qwen3.6 alias | `mlp.gate+experts`，attention 为 `self_attn` 或 `linear_attn` | attention \| `mlp.gate` \| `mlp.shared_expert` \| `mlp.experts` |
| `kimi_k3` | AttnRes / KDA / Stable LatentMoE 结构标记 | KDA/MLA \| `block_sparse_moe.*` \| AttnRes projections |

核心流程不再维护按模型名复制的 Adapter 类。`model_structure.py` 只解析主流程真正使用的能力：文本容器、decoder layers、特殊模块、MoE 容器和跨层状态。模型名 alias 仅保留在 CLI/子图展示层。

Kimi K3 官方 checkpoint 的每个 MoE 层包含 896 个独立 expert module。L1 `grouped_dual` 会在 meta 阶段移除 expert module 权重并从 safetensors 流式读取；普通 `dual` 会 fail-fast。KDA 在 Ascend 上默认使用 `torch` recurrence（普通 torch-npu elementwise/matmul），避免依赖 FLA chunk Triton 内核；可用 `--kimi_kda_backend` 覆盖。ref/quant 设备组互不重叠时，两侧 layer forward 由两个 worker 并发执行；设备重叠、DEBUG 模式或 `ACC_DUAL_FORWARD_SERIAL=1` 时回退为串行。L2 当前仍要求完整目标层，因此对这类 MoE 层 fail-fast，直到流式 expert replay 支持完成；packed 内部实现不受该结构检查限制。

加 `--mla_fine` 对 MLA attention 细粒度拆分：
`q_a_proj` / `q_b_proj` / `kv_a_proj_with_mqa` / `kv_b_proj` / `o_proj` / `indexer` (wq_b / wk / weights_proj / k_norm)

---

## 5. V2 指标体系

L2 用五类指标从不同角度刻画子图误差，适用于不同子图类型：

| 指标 | 语义 | 适用子图 | 解读 |
|------|------|---------|------|
| **Recovery Ratio** | `(base_l2 - patched_l2) / base_l2`，替换后误差恢复比例 | 大子图 (self_attn 整体, mlp.*, mlp.gate, mlp.experts, o_proj) | 越高 = 该子图是主要误差源 |
| **RotBErr** | rotation-aligned boundary error：ref 子图输出旋转到 quant 空间后与 quant 子图输出的 rel_l2 | attention 内部子图 (q_a/q_b/kv_a/kv_b) | 越高 = 该子图自身误差大 |
| **SelfRotErr** | controlled-input self quantization error：喂 ref 输入（旋转后）给 quant 单 op，测其自身量化误差 | attention 内部串联子链 + indexer 内部 | 越高 = 该 op 量化误差大 |
| **Flip Rate** | top-k 翻转率 | indexer (DSA 稀疏注意力选哪些 token) / mlp.gate (MoE 路由选哪些 expert) | 越高 = 路由决策发散 |
| **Input Recovery** | 把 ref hidden (rotated) 喂给 quant layer 的恢复比例 | 整层 | 区分"输入累积误差"vs"本层误差" |

### 为什么 attention 内部子图不能用 Recovery

attention 有 softmax 非线性：如果 Q 仍有量化误差但 K/V 被 patch 为 ref，Q-K 不匹配导致 attention 分布错位，比全部保持量化误差更差 → **负 recovery**。因此 q_a/q_b/kv_a/kv_b 用 RotBErr + SelfRotErr 代替，只有 o_proj（attention 之后）的 patch 是可靠正 recovery。

### patch 策略矩阵

| 子图类型 | patch? | 用什么指标 | 原因 |
|---|---|---|---|
| 大子图 (self_attn 整体, mlp.*) | ✅ | Recovery | 子图内部 pair 抵消，output 在原空间 |
| attention 内部 (q_a/q_b/kv_a/kv_b) | ❌ skip | RotBErr + SelfRotErr | Q-K mismatch, softmax 非线性 |
| indexer 内部 (wq_b/wk/weights_proj/k_norm) | ❌ skip | SelfRotErr | fp8_index 是 NPU 自定义算子，部分 patch 无效 |
| FLOAT op (未量化) | ❌ skip | 标记为 FLOAT | downstream bias，不参与 root cause 排名 |
| 旋转对齐失败 | ❌ skip | UNPATCHABLE | ref/quant 不在同一空间 |

FLOAT 检测：读 `quant_model_description.json`，识别未量化的 op。FLOAT op 的高 recovery 是 downstream bias（吸收所有上游误差），不参与 root cause 排名。

---

## 6. 旋转矩阵映射（GLM-5 DSA QuaRot）

QuaRot 模型权重融合了旋转矩阵 R，不同子图输出落在不同旋转空间。patch 前必须把 ref 子图 output 旋转到 quant 的对应空间。

`_MLA_SUBGRAPH_ROT_KEY` 映射（子图输出 → 旋转矩阵 key）：

| 子图 | 旋转 key | 空间 | 说明 |
|------|---------|------|------|
| `q_a_proj` | `rot_b_proj` | R3 (2048) | 权重 R1^T·W·R3 |
| `q_b_proj` | None | 恒等 | 权重 R3^T·W |
| `kv_a_proj_with_mqa` | `rot_kv_b_proj` | R4 (512) | split: [R4, I(64)] |
| `kv_b_proj` | `rot_uv` | R2 (256) | split per-head: [I(192), R2] |
| `o_proj` | `rot` | R1 (6144) | 权重 R2^T·W·R1 |
| `indexer.*` | None | - | 返回整数 topk_indices 或在恒等空间 |

非细粒度子图（self_attn 整体 / mlp 子图）：内部成对抵消 → 输出 R1 空间。

旋转矩阵文件：`rotate_matrix_w8a8.pt`，含 4 个矩阵：`rot`(R1), `rot_b_proj`(R3), `rot_kv_b_proj`(R4), `rot_uv`(R2)。

---

## 7. diagnose_layer 诊断流程

`diagnose_layer(handle, model_type, mla_fine, R, rot_mats, quant_desc, model_config)` 步骤：

1. **检测模型类型** + 生成子图列表 `get_subgraph_names()`
2. **Baseline**: ref_out vs quant_out (unrotated)，算 `base_l2`
3. **Input patch**: 把 ref_hidden (rotated) 喂给 quant layer，算 `input_recovery`（区分输入累积误差 vs 本层误差）
4. **捕获 ref 子图输出** `_capture_subgraph_outputs(ref_layer)`
5. **旋转对齐** `_rotate_subgraph_output()`: 把 ref 子图 output 旋转到 quant 内部空间，标记对齐失败的为 UNPATCHABLE
6. **(mla_fine) RotBErr + SelfRotErr**:
   - 捕获 quant 子图输出，逐子图算 RotBErr（旋转空间下的 rel_l2）
   - SelfRotErr：对 attention 内部串联子链，喂 ref 输入给 quant 单 op，测自身量化误差（支持 post-LN 链式追踪，如 wk → k_norm）
   - indexer / mlp.gate 的 top-k flip rate
7. **逐子图 patch** `_patch_subgraph()`: 对可 patch 子图挂 `ReplacementHook` 做反事实替换，算 Recovery
8. **Chain Delta**: attention 内部串联子链的增量分析（SelfRotErr 即 delta 等价物）
9. **root_suspect 排名**: 综合各指标定位主要误差源

输出 dict 含：`subgraphs`(recovery) / `subgraph_quant_types` / `subgraph_rotberr` / `subgraph_selfroterr` / `indexer_flip_rate` / `experts_routing_flip` / `input_recovery` / `baseline_l2` / `root_suspect`。

---

## 8. 资源管理

- **最少 2 张 NPU 卡**：ref 1 张，quant 1 张
- `LayerReplayHandle` 管理双卡 device + 单层 forward + cleanup
- 同一时刻只加载 1 层权重，诊断完 `unload_layers_to_meta` 卸载，显存占用小
- `ReplayProvider` 负责模型 skeleton 分发 + 逐层权重加载（indexed）+ L1 cache hidden_states 读取

---

## 9. 支持的模型

| 模型 | 路径 | L1 | L2 | 备注 |
|------|------|----|----|------|
| GLM-5.1 | HF | ✅ | ✅ | MLA + DSA + MoE, 78 层, 3D packed experts |
| Qwen3 | HF | ✅ | ✅ | dense |
| Qwen3VL | HF | ✅ | ✅ | dense |
| Qwen3MoE | HF | ✅ | ✅ | MoE |
| Qwen3.5 MoE | HF | ✅ | ✅ | MoE (Qwen35MoEAdapter) |

Custom 路径（DeepSeek V4/V3.2）在开源版已移除。

---

## 10. 命令示例

### L1 全层对比 + 缓存最差层

```bash
cd <your_workspace>/tools/accuracy_bench && \
ASCEND_RT_VISIBLE_DEVICES=0,1 python3 run_accuracy_check.py \
  --ref_model /path/to/glm5-fp16 \
  --quant_model /path/to/glm5-w8a8 \
  --l1 --ref_device npu:0 --quant_device npu:1 \
  --prompt "你好" --layers_per_shard 4 \
  --quant_method dequantize \
  --cache_top_k 3 \
  --rotation_matrix /path/to/rotate_matrix_w8a8.pt
```

### L2 子图诊断（从 L1 report 自动选候选层）

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
python3 run_accuracy_check.py --l2 \
    --ref_model /path/to/glm5-bf16 \
    --quant_model /path/to/glm5-int8 \
    --l1_report l1_reports/alignment_report.txt \
    --ref_device npu:0 --quant_device npu:1 \
    --quant_method dequantize \
    --rotation_matrix /path/to/rotate_matrix_w8a8.pt
```

### L2 手动指定层 + MLA 细粒度

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
python3 run_accuracy_check.py --l2 \
    --ref_model /path/to/glm5-bf16 \
    --quant_model /path/to/glm5-int8 \
    --layers 4 10 20 \
    --mla_fine \
    --ref_device npu:0 --quant_device npu:1 \
    --quant_method dequantize \
    --rotation_matrix /path/to/rotate_matrix_w8a8.pt
```

### L1 + L2 自动衔接

```bash
python3 run_accuracy_check.py --all \
  --ref_model ... --quant_model ... \
  --ref_device npu:0 --quant_device npu:1 \
  --cache_top_k 3 --mla_fine
```

`--all` 模式：先跑 L1，发现 first_bad_block 后自动对该层及前一层跑 L2。

---

## 11. 关键文件速查

| 文件 | 作用 |
|------|------|
| `run_accuracy_check.py` | 统一入口，L1/L2 自动衔接 |
| `accuracy_checker/subgraph_locate.py` | L2 反事实诊断主逻辑 |
| `accuracy_checker/replay_provider.py` | 双卡单层 replay + 权重加载 |
| `accuracy_checker/operator_patcher.py` | ReplacementHook (output 替换) |
| `accuracy_checker/v2_metrics.py` | rel_l2 / cos_sim / recovery_ratio |
| `accuracy_checker/layer1_block_compare.py` | L1 ShardedBlockComparator |
| `accuracy_checker/model_loader.py` | HF 模型加载 (3D expert, indexed) |
| `accuracy_checker/model_structure.py` | 模型容器、文本层、MoE 与跨层状态的统一结构探测 |
| `accuracy_checker/utils.py` | rotation / device 工具 |
| `accuracy_checker/cache.py` | L2 cache 目录管理 |
