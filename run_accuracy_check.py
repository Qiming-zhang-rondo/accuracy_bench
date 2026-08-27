#!/usr/bin/env python3
"""
量化精度对齐工具 - 统一入口 (acc_bench)

漏斗式诊断, 通过 ``--mode`` 选择阶段, 旧参数 (--l1/--l2/--all/--boundary) 仍可用:

    python run_accuracy_check.py --mode screening    # L0 模型完整性预检 (不依赖 NPU)
    python run_accuracy_check.py --mode boundary      # 定界: 框架 vs 权重/量化
    python run_accuracy_check.py --mode l1            # L1 逐 block 对比
    python run_accuracy_check.py --mode l2 --target-layers 8,9   # L2 子图诊断
    python run_accuracy_check.py --mode full          # L0→L1→Top-K→L2→logits→JSON→HTML
    python run_accuracy_check.py --mode report --result-json r.json   # 从 JSON 生成 HTML

支持 HF 模型 (GLM-5.1, Qwen3 系列)。
"""

import argparse
from datetime import datetime
import json
import os
import re
import sys

import torch

from accuracy_checker.utils import auto_device, parse_dtype
from accuracy_checker import AlignmentReport
import logging


logger = logging.getLogger(__name__)

VALID_MODES = ("screening", "boundary", "l1", "l2", "full", "report", "inference")
ACTIVATION_QUANT_TYPES = (
    "AUTO",
    "W8A8_MXFP8",
    "W4A8_MXFP",
    "W4A4_MXFP4",
    "W4A4_DYNAMIC",
    "W4A4_INT4_PER_GROUP",
)
ACTIVATION_QUANT_ALIASES = {
    "W4A4_LAOS": "W4A4_DYNAMIC",
    "INT4_PER_GROUP": "W4A4_INT4_PER_GROUP",
    "W4A4_PER_GROUP": "W4A4_INT4_PER_GROUP",
    "W4A4_INT4_PERGROUP": "W4A4_INT4_PER_GROUP",
}


def parse_activation_quant_type(value: str) -> str:
    """Normalize legacy activation-only aliases before argparse choices."""
    normalized = value.strip().upper()
    return ACTIVATION_QUANT_ALIASES.get(normalized, normalized)


def parse_positive_int(value: str) -> int:
    """Argparse type for strictly positive integer parameters."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description="量化精度对齐工具 (acc_bench)")

    # ---- 统一 mode 入口 ----
    parser.add_argument("--mode", type=str, default=None, choices=VALID_MODES,
                        help="统一模式: screening=L0预检 / boundary=定界 / "
                             "l1=逐block / l2=子图 / full=全流程 / "
                             "report=JSON→HTML / inference=独立推理HTML")
    parser.add_argument("--result_json", type=str, default=None,
                        help="[report] 从 JSON ReportData 重新生成 HTML 报告的输入路径")

    # Model paths
    parser.add_argument("--ref_model", type=str, default=None,
                        help="参考模型(FP16/BF16)路径")
    # quant_model 必须性按 mode 延迟到 dispatch 时校验 (report 模式可不需要)
    parser.add_argument("--quant_model", type=str, default=None,
                        help="量化模型路径")

    # 历史参数 (backward-compat): --l1/--l2/--all/--boundary 仍可用
    parser.add_argument("--l1", action="store_true", help="执行L1逐block对比")
    parser.add_argument("--l2", action="store_true", help="执行L2逐算子对比")
    parser.add_argument("--all", action="store_true", help="执行L1, L1有问题时自动跑L2")
    parser.add_argument("--boundary", action="store_true",
                        help="定界: 误差级别边界判定 (= --mode boundary 的简写)")

    # Common
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下你自己",
                        help="对比用的输入文本")
    parser.add_argument("--messages", type=str, default=None,
                        help='Chat messages JSON, 走 apply_chat_template')
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="[boundary] 完整 OpenAI/vLLM 请求 JSON 或对话列表文件")
    parser.add_argument("--request_json", type=str, default=None,
                        help="[boundary] 直接粘贴完整 OpenAI/vLLM 请求 JSON")
    parser.add_argument("--request_json_stdin", action="store_true",
                        help="[boundary] 从 stdin 读取完整请求 JSON，适合超长 prompt")
    parser.add_argument("--prompt_stdin", action="store_true",
                        help="从 stdin 读取原始 prompt，适合 boundary/L1/full 超长文本")
    parser.add_argument("--target_layers", type=str, default=None,
                        help="L2 只检查指定层 (逗号或空格分隔, 如 '8,9' 或 '8 9')")
    parser.add_argument("--target-layers", dest="target_layers_alt", type=str, default=None,
                        help="[alias] 与 --target_layers 等价")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 (cuda/cpu/npu:0)")
    parser.add_argument("--ref_device", type=str, default=None,
                        help="参考模型设备")
    parser.add_argument("--quant_device", type=str, default=None,
                        help="量化模型设备")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="数据类型")
    parser.add_argument("--clean_cache", action="store_true",
                        help="清除 L2 缓存目录后退出")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="L2 cache 目录")
    parser.add_argument("--debug", action="store_true",
                        help="输出模型结构、hidden norm 和 MoE expert 等诊断日志")

    # L1
    parser.add_argument("--per_device_memory", type=float, default=None,
                        help="每卡显存大小(GB)")
    parser.add_argument("--quant_method", type=str, default="dequantize",
                        choices=["dequantize", "fake_quant"],
                        help="量化模型加载方式 (dequantize=反量化到BF16比对, fake_quant=保留伪量化算子)")
    parser.add_argument("--layers_per_shard", type=int, default=8,
                        help="L1: 每个 shard 的层数")
    parser.add_argument(
        "--prefill_parallel", type=str, default="pp", choices=["pp", "tp"],
        help=("GLM 长 prefill 并行方式: pp=保留按层分片（默认）, "
              "tp=在每个 DSA indexer 的 key 维度跨 ref/quant 设备组并行"),
    )
    parser.add_argument(
        "--glm_attn_query_block", type=parse_positive_int, default=None,
        help="GLM 长 prefill attention query block（默认 64；也可用 ACC_GLM_DSA_ATTN_QUERY_BLOCK）",
    )
    parser.add_argument(
        "--glm_attn_selected_block", type=parse_positive_int, default=None,
        help="GLM 长 prefill selected-key block（默认 512；也可用 ACC_GLM_DSA_ATTN_SELECTED_BLOCK）",
    )
    parser.add_argument("--cache_top_k", type=int, default=0,
                        help="L1: 缓存 cos_sim 最低的 N 层到 L2 cache (0=不缓存)")
    parser.add_argument("--l1_target_layers", type=int, nargs="+", default=None,
                        help="L1: 只跑指定的层, 每层 input 自动保存到 L2 cache")
    parser.add_argument("--rotation_matrix", type=str, default=None,
                        help="旋转矩阵文件路径 (QuaRot global_rotation, 用于unrotate quant侧hidden states)")
    parser.add_argument("--mla_fine", action="store_true", default=True,
                        help="L2: 对 MLA attention 做细粒度拆分 (默认开启, 同时输出大子图+小子图)")
    parser.add_argument("--no_mla_fine", action="store_true",
                        help="L2: 关闭 MLA 细粒度拆分 (只输出粗粒度 self_attn)")
    parser.add_argument("--activation_quant", action="store_true",
                        help="L1: 启用激活伪量化 (类型由 --activation_quant_type 指定)")
    parser.add_argument("--activation_quant_type", type=parse_activation_quant_type,
                        default="AUTO",
                        choices=["AUTO", "W8A8_MXFP8", "W4A8_MXFP", "W4A4_MXFP4", "W4A4_DYNAMIC", "W4A4_INT4_PER_GROUP"],
                        help="L1: 激活伪量化类型 (AUTO=按 quant descriptor 逐算子选择; "
                             "W8A8_MXFP8/W4A8_MXFP=MXFP8 per-block; "
                             "W4A4_MXFP4=MXFP4 E2M1 per-block; "
                             "W4A4_DYNAMIC=INT4 per-token sym; "
                             "W4A4_INT4_PER_GROUP=INT4 hidden-axis per-group sym; legacy "
                             "W4A4_LAOS maps to W4A4_DYNAMIC)")
    parser.add_argument(
        "--activation_quant_backend", type=str, default="auto",
        choices=["auto", "npu", "torch"],
        help=("L1: INT4 激活量化后端 (auto=NPU 上使用原生 "
              "npu_dynamic_quant，CPU 上使用 Torch reference；torch 仅用于诊断)"),
    )
    parser.add_argument(
        "--activation_quant_group_size", type=parse_positive_int, default=128,
        help=("L1: W4A4_INT4_PER_GROUP 沿 hidden 维的 group size "
              "(默认 128；其他 activation 类型忽略)"),
    )
    parser.add_argument("--compare_mode", type=str, default="dual",
                        choices=["dual", "grouped_dual"],
                        help="L1: 对比模式 (dual=双卡分片, grouped_dual=MoE expert跨卡)")
    parser.add_argument("--ref_devices", type=str, default=None,
                        help="L1: ref 多卡设备 (如 'npu:0,npu:1')")
    parser.add_argument("--quant_devices", type=str, default=None,
                        help="L1: quant 多卡设备 (如 'npu:2,npu:3')")
    parser.add_argument("--expert_chunk_size", type=parse_positive_int, default=None,
                        help="L1 grouped_dual / boundary: expert 分块大小 (boundary 默认 8)")
    parser.add_argument(
        "--kimi_kda_backend",
        type=str,
        default="auto",
        choices=["auto", "torch", "chunk", "fused_recurrent"],
        help=(
            "L1: Kimi K3 KDA backend. auto 在 Ascend NPU 上使用无 Triton "
            "依赖的 torch recurrence，其他设备保留模型默认 chunk"
        ),
    )
    parser.add_argument("--output_dir", type=str, default=None,
                        help="报告输出目录")
    parser.add_argument("--model_type", type=str, default="auto",
                        choices=["auto", "dense", "moe",
                                 "glm_mla", "glm_moe_dsa",
                                 "deepseek_v4",
                                 "qwen3", "qwen3_moe", "qwen3_5_moe", "qwen3_vl",
                                 "qwen3_6", "qwen3_6_moe", "kimi_k3", "dspark"],
                        help="L2: 模型类型 (默认 auto 自动检测; 偶尔需手动覆盖; "
                             "qwen3=Qwen3/Qwen2, qwen3_moe=Qwen3 MoE, "
                             "qwen3_5_moe=Qwen3.5/3.6 official model_type, "
                             "deepseek_v4=DeepSeek-V4 official Transformers model, "
                             "qwen3_6=Qwen3.6 alias, kimi_k3=Kimi K3, "
                             "dspark=standalone DSpark draft)")
    parser.add_argument("--dspark_sample", type=str, default=None,
                        help="[DSpark L1] verifier hidden-state .pt sample; required fields: "
                             "input_ids, hidden_states/target_hidden_states, loss_mask; "
                             "Speculators format additionally needs verifier_last_hidden_states")
    parser.add_argument("--dspark_seed", type=int, default=0,
                        help="[DSpark L1] deterministic anchor/noise sampling seed")
    parser.add_argument("--dspark_max_anchors", type=int, default=8,
                        help="[DSpark L1] maximum verifier anchors per sample (default 8; "
                             "automatically capped by valid sample anchors/config)")

    # ---- full/boundary 模式专用 ----
    parser.add_argument("--devices", type=str, default=None,
                        help="[boundary] 逻辑设备列表 (如 'npu:0' / 'npu:0,1'); "
                             "缺省沿用 --device / 自动选择")
    parser.add_argument("--max_new_tokens", type=parse_positive_int, default=None,
                        help="[boundary/full] 生成 token 数；boundary 未传时读取请求 JSON 的 max_tokens")
    parser.add_argument("--thinking", type=str, default="chat",
                        choices=["chat", "none"],
                        help="[boundary] 思维链开关: chat=开, none=关")
    parser.add_argument("--framework_name", type=str, default=None,
                        help="[boundary] 部署框架名 (vllm/mindie/...)")
    parser.add_argument("--framework_bad_output", type=str, default=None,
                        help="[boundary] 部署框架实际生成的坏文本")
    parser.add_argument("--framework_bad_reproduced", choices=["true", "false"], default=None,
                        help="[boundary] 外部确认部署框架是否复现，不会调用该框架")
    parser.add_argument("--no_ref", action="store_true",
                        help="[boundary] Quant-only；不加载参考模型")
    parser.add_argument("--num_runs", type=parse_positive_int, default=1,
                        help="[boundary] 同一 prompt 总运行次数")
    parser.add_argument("--concurrency", type=parse_positive_int, default=1,
                        help="[boundary] 每个纯 Transformers generate batch 的样本数")
    parser.add_argument("--stop_on_first_badcase", action="store_true",
                        help="[boundary] 首批检测到 bad case 后提前停止")
    parser.add_argument("--repeat_4gram_max", type=float, default=None,
                        help="[boundary] 4-gram 重复比阈值，默认 0.5")
    parser.add_argument("--nonprintable_max", type=float, default=None,
                        help="[boundary] 非可打印字符比阈值，默认 0.3")
    parser.add_argument("--top_k", type=int, default=1,
                        help="[full] L1 后自动选取 cos_sim 最低的 Top-K 层进入 L2")
    parser.add_argument("--logits", action="store_true",
                        help="[full] 额外采集 ref/quant logits 对比")
    parser.add_argument(
        "--logits_max_positions", type=int, default=None,
        help=("[L1/full] logits 采集位置上限；默认短 prompt 采集全部、"
              "长 prompt 仅采集最后 32 个；传 0 表示不限制"),
    )
    parser.add_argument("--manifest", type=str, default="",
                        help="[full] bad case manifest JSON 路径, 用于 ground truth 对比")
    parser.add_argument("--quant_format", type=str, default="",
                        help="[full] 量化格式标注 (W8A8/W4A8_MXFP4 ...), 写入报告 overview")
    parser.add_argument("--model_name", type=str, default="",
                        help="[full/report] 模型名, 写入报告 overview")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# mode 解析 (backward-compat: 无 --mode 时从旧 flags 推断)
# ---------------------------------------------------------------------------

def _resolve_mode(args) -> str:
    if args.mode is not None:
        return args.mode
    if args.boundary:
        return "boundary"
    if args.all:
        return "full"
    if args.l2:
        return "l2"
    if args.l1:
        return "l1"
    return "l1"  # 默认与历史行为一致 (无 flag 时跑 L1)


def _parse_target_layers(args):
    """逗号或空格分隔的层号 -> List[int]。兼容旧的 '8 9' 空格形式。"""
    raw = args.target_layers or args.target_layers_alt
    if raw is None:
        return None
    toks = [t for t in raw.replace(",", " ").split() if t.strip()]
    out = []
    for t in toks:
        try:
            out.append(int(t))
        except ValueError:
            logger.warning(f"忽略无效层号: {t}")
    return out or None


def _parse_messages(args):
    """--messages JSON 字符串 -> List[Dict]; 不是 messages 则 None。"""
    if not args.messages:
        return None
    try:
        msgs = json.loads(args.messages)
        if isinstance(msgs, list):
            return msgs
        if isinstance(msgs, dict) and isinstance(msgs.get("messages"), list):
            return msgs["messages"]
    except Exception:  # noqa: BLE001
        pass
    return None


def _resolve_input(args, tokenizer=None):
    if args.messages:
        messages = json.loads(args.messages)
        if tokenizer is None:
            raise ValueError("--messages 需要 tokenizer")
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        return prompt_text, input_ids
    return args.prompt, None


def _cache_input_identity(args) -> str:
    """Return the canonical sample identity shared by L1 cache and L2 lookup."""
    raw_messages = getattr(args, "messages", None)
    if raw_messages:
        try:
            messages = json.loads(raw_messages)
            payload = json.dumps(
                messages,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            # Validation reports malformed JSON separately. Keep this fallback
            # deterministic so it does not mask the primary input error.
            payload = str(raw_messages)
        return f"messages:{payload}"
    return f"prompt:{getattr(args, 'prompt', None) or ''}"


def _clean_l2_cache():
    from accuracy_checker.cache import get_cache_dir
    import glob
    cache_dir = get_cache_dir()
    if os.path.isdir(cache_dir):
        files = glob.glob(os.path.join(cache_dir, "*.pt"))
        for f in files:
            os.remove(f)
        logger.info(f"L2 cache cleared: {len(files)} files removed from {cache_dir}")
    else:
        logger.info(f"L2 cache directory not found: {cache_dir}")


def _parse_dev_list(dev_str):
    """解析设备列表字符串, 自动补 npu: 前缀 (如 'npu:0,1,2' → ['npu:0','npu:1','npu:2'])"""
    if not dev_str:
        return None
    devs = []
    for d in dev_str.split(','):
        d = d.strip()
        if not d:
            continue
        if d.startswith('npu:') or d.startswith('cuda:'):
            devs.append(d)
        else:
            devs.append(f'npu:{d}')
    return devs


def _resolve_devices(args):
    device = auto_device() if args.device == "cuda" else args.device
    ref_devices = _parse_dev_list(getattr(args, 'ref_devices', None))
    quant_devices = _parse_dev_list(getattr(args, 'quant_devices', None))
    ref_device = args.ref_device or (ref_devices[0] if ref_devices else device)
    target_device = args.quant_device or (quant_devices[0] if quant_devices else device)
    dtype = parse_dtype(args.dtype)
    return ref_device, target_device, dtype


def _validate_standalone_dspark_args(args, mode: str, standalone_dspark: bool):
    """Fail early for options that the standalone DSpark runtime cannot honor."""
    if not standalone_dspark:
        return
    if mode not in ("l1", "screening", "report"):
        raise NotImplementedError(
            "standalone DSpark 仅支持 --mode l1/screening/report；"
            "draft 不能独立 generate，也不能进入普通 CausalLM boundary/full/L2。"
        )
    if mode != "l1":
        return
    if args.quant_method != "dequantize":
        raise ValueError(
            "standalone DSpark L1 仅支持 --quant_method dequantize；"
            "当前 draft API 不支持 fake_quant。"
        )
    if args.compare_mode != "dual":
        raise ValueError(
            "standalone DSpark L1 仅支持 --compare_mode dual；"
            "grouped_dual 只用于普通 MoE expert 分发。"
        )
    if args.ref_devices or args.quant_devices:
        raise ValueError(
            "standalone DSpark L1 使用 --ref_device/--quant_device；"
            "不要传 --ref_devices/--quant_devices 多卡列表。"
        )


# ===========================================================================
# L1 / L2 运行 (沿用历史实现)
# ===========================================================================

def run_hf_l1(args, ref_device, target_device, dtype):
    from accuracy_checker.dspark import is_dspark_checkpoint

    ref_is_dspark = is_dspark_checkpoint(args.ref_model)
    quant_is_dspark = is_dspark_checkpoint(args.quant_model)
    explicit_dspark = getattr(args, 'model_type', 'auto') == 'dspark'
    if ref_is_dspark or quant_is_dspark or explicit_dspark:
        if not (ref_is_dspark and quant_is_dspark):
            raise ValueError(
                "DSpark L1 requires both --ref_model and --quant_model to be "
                "standalone DSpark checkpoints"
            )
        if not args.dspark_sample:
            raise ValueError(
                "DSpark is a draft model and cannot run from prompt text alone; "
                "provide --dspark_sample <verifier_hidden_states.pt>"
            )
        if getattr(args, 'compare_mode', 'dual') != 'dual':
            raise ValueError(
                "DSpark draft uses its dedicated dual-device path; "
                "set --compare_mode dual"
            )
        from accuracy_checker.dspark import DSparkComparator
        comparator = DSparkComparator(
            ref_model_path=args.ref_model,
            quant_model_path=args.quant_model,
            sample_path=args.dspark_sample,
            ref_device=ref_device,
            quant_device=target_device,
            dtype=dtype,
            quant_method=args.quant_method,
            seed=args.dspark_seed,
            max_anchors=args.dspark_max_anchors,
            verbose=True,
        )
        return comparator.compare()

    # Kimi remote code imports FLA while the Python module is being loaded.
    # Install the portable import surface before AutoTokenizer/model dynamic
    # module resolution so --kimi_kda_backend=torch also works without FLA.
    from accuracy_checker.kimi_fla_shim import ensure_kimi_torch_import_path
    kimi_shim_active = ensure_kimi_torch_import_path(
        requested_backend=getattr(args, "kimi_kda_backend", "auto"),
        devices=(ref_device, target_device),
        model_type=getattr(args, "model_type", "auto"),
        model_paths=(args.ref_model, args.quant_model),
    )
    if kimi_shim_active:
        logger.info(
            "[Kimi K3] portable torch import path active; fla-core is not required"
        )

    from transformers import AutoTokenizer
    from accuracy_checker.glm_dsa_blockwise import install_glm_dsa_blockwise_indexer
    from accuracy_checker.deepseek_v4_blockwise import install_deepseek_v4_blockwise_runtime
    ref_tp_devices = _parse_dev_list(getattr(args, "ref_devices", None)) or [ref_device]
    quant_tp_devices = _parse_dev_list(getattr(args, "quant_devices", None)) or [target_device]
    glm_blockwise_ok = install_glm_dsa_blockwise_indexer(
        parallel_mode=getattr(args, "prefill_parallel", "pp"),
        device_groups=[ref_tp_devices, quant_tp_devices],
        attention_query_block=getattr(args, "glm_attn_query_block", None),
        attention_selected_block=getattr(args, "glm_attn_selected_block", None),
    )
    if not glm_blockwise_ok and getattr(args, "prefill_parallel", "pp") in {"pp", "tp"}:
        logger.warning(
            "  GLM DSA blockwise indexer 未安装；长 prompt 将使用 Transformers "
            "eager indexer，可能产生超大 score tensor"
        )
    install_deepseek_v4_blockwise_runtime()
    from accuracy_checker import ShardedBlockComparator

    logger.info("\n" + "=" * 70)
    logger.info("  L1: 逐Block对比 (HF Model)")
    logger.info("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(args.ref_model, trust_remote_code=True)

    comparator = ShardedBlockComparator(
        ref_model_path=args.ref_model,
        quant_model_path=args.quant_model,
        tokenizer=tokenizer,
        ref_device=ref_device,
        quant_device=target_device,
        dtype=dtype,
        use_fake_quant=(args.quant_method == "fake_quant"),
        per_device_memory_gb=args.per_device_memory or 64.0,
        verbose=True,
        cache_top_k=args.cache_top_k,
        quant_method=args.quant_method,
        rotation_matrix=args.rotation_matrix,
        l1_target_layers=args.l1_target_layers,
        activation_quant=args.activation_quant,
        activation_quant_type=args.activation_quant_type,
        activation_quant_backend=args.activation_quant_backend,
        activation_quant_group_size=args.activation_quant_group_size,
        compare_mode=getattr(args, 'compare_mode', 'dual'),
        ref_devices=_parse_dev_list(getattr(args, 'ref_devices', None)),
        quant_devices=_parse_dev_list(getattr(args, 'quant_devices', None)),
        expert_chunk_size=getattr(args, 'expert_chunk_size', None),
        kimi_kda_backend=getattr(args, 'kimi_kda_backend', 'auto'),
        logits_max_positions=getattr(args, 'logits_max_positions', None),
    )
    prompt_text, input_ids = _resolve_input(args, tokenizer)
    cache_prompt = _cache_input_identity(args)
    logger.info(f"  输入: {prompt_text[:100]}")
    if input_ids is not None:
        l1_report = comparator.compare_ids(
            input_ids,
            layers_per_shard=args.layers_per_shard,
            cache_prompt=cache_prompt,
        )
    else:
        l1_report = comparator.compare(
            prompt_text,
            layers_per_shard=args.layers_per_shard,
            cache_prompt=cache_prompt,
        )
    return l1_report


def run_hf_l2(args, ref_device, target_device, dtype, bad_layers):
    from accuracy_checker.dspark import is_dspark_checkpoint
    if (
        getattr(args, 'model_type', 'auto') == 'dspark'
        or is_dspark_checkpoint(args.ref_model)
        or is_dspark_checkpoint(args.quant_model)
    ):
        raise NotImplementedError(
            "Standalone DSpark currently supports dedicated L1 output alignment; "
            "normal CausalLM L2 replay is invalid because each draft layer also "
            "consumes verifier target_hidden_states"
        )
    from accuracy_checker.subgraph_locate import diagnose_layers, print_report

    logger.info("\n" + "=" * 70)
    logger.info(f"  L2: Sub-graph 反事实诊断 - 检查层: {bad_layers}")
    logger.info("=" * 70)

    prompt_text = _cache_input_identity(args)
    results = diagnose_layers(
        ref_model_path=args.ref_model,
        quant_model_path=args.quant_model,
        candidate_layers=bad_layers,
        prompt=prompt_text,
        quant_method=args.quant_method,
        ref_device=ref_device,
        quant_device=target_device,
        rotation_matrix=args.rotation_matrix,
        mla_fine=getattr(args, 'mla_fine', True) and not getattr(args, 'no_mla_fine', False),
        model_type=getattr(args, 'model_type', 'auto'),
    )
    print_report(results)
    return results


# ===========================================================================
# 辅助
# ===========================================================================

def _require(args, name, mode):
    val = getattr(args, name, None)
    if not val:
        logger.info(f"\n  [{mode}] 需要指定 --{name}")
        return None
    return val


def _topk_from_l1(l1_report, k):
    """从 L1 结果按 cos_sim 最低取 Top-K 层号。"""
    results = getattr(l1_report, "results", None) or []
    import re
    scored = []
    for r in results:
        name = getattr(r, "layer_name", "") or ""
        cs = (getattr(r, "metrics", {}) or {}).get("cos_sim")
        if cs is None:
            continue
        m = re.search(r"(\d+)", name)
        if m:
            scored.append((float(cs), int(m.group(1)), name))
    scored.sort(key=lambda x: x[0])
    return [idx for _, idx, _ in scored[:k]]


def _run_logits(args, ref_device, target_device, dtype):
    """采集 ref/quant logits 并对比。失败返回 None (不阻塞 full)。"""
    try:
        from transformers import AutoTokenizer
        from accuracy_checker import collect_logits, compare_logits
        from accuracy_checker.model_loader import load_model_for_comparison

        tokenizer = AutoTokenizer.from_pretrained(args.ref_model, trust_remote_code=True)
        ref_model = load_model_for_comparison(
            args.ref_model, device=ref_device, dtype=dtype, use_fake_quant=False)
        quant_model = load_model_for_comparison(
            args.quant_model, device=target_device, dtype=dtype,
            quant_method=args.quant_method, use_fake_quant=(args.quant_method == "fake_quant"))
        ref_model.eval(); quant_model.eval()
        ref_lc = collect_logits(ref_model, tokenizer, args.prompt,
                                device=str(ref_device), max_new_tokens=args.max_new_tokens)
        quant_lc = collect_logits(quant_model, tokenizer, args.prompt,
                                  device=str(target_device), max_new_tokens=args.max_new_tokens)
        comp = compare_logits(ref_lc, quant_lc, tokenizer)
        n = len(comp.token_positions)
        logger.info(f"  [full] logits 采集 {n} 位置")
        return comp
    except Exception as e:  # noqa: BLE001
        logger.info(f"  [full] logits 跳过 (加载/采集失败): {e}")
        return None


# ===========================================================================
# 各 mode 实现
# ===========================================================================

def _mode_screening(args):
    """L0 模型完整性预检 (不实例化整模型, 不依赖 NPU)。"""
    ref = _require(args, "ref_model", "screening")
    quant = _require(args, "quant_model", "screening")
    if not ref or not quant:
        return
    from accuracy_checker import run_l0_sanity
    logger.info("\n" + "=" * 70)
    logger.info("  L0: 模型完整性预检 (screening)")
    logger.info("=" * 70)
    result = run_l0_sanity(
        ref_model_path=ref,
        quant_model_path=quant,
        rotation_matrix=args.rotation_matrix,
        expected_dtype=args.dtype,
    )
    logger.info("\n" + str(result.summary))
    for c in result.checks:
        logger.info(f"  [{c.status:4s}] {c.name}: {c.detail}")
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "l0_sanity.json"), "w") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"  L0 结果已保存: {args.output_dir}/l0_sanity.json")


def _mode_boundary(args):
    """定界: 区分 WEIGHT / INFERENCE_FRAMEWORK / BOTH / INCONCLUSIVE / INVALID_RUN。"""
    quant = _require(args, "quant_model", "boundary")
    if not quant:
        return
    from accuracy_checker import run_boundary, boundary_result_to_dict
    from accuracy_checker.boundary_check import parse_request_json
    devices = args.devices or args.quant_device or args.ref_device or "npu:0"
    logger.info("\n" + "=" * 70)
    logger.info("  Boundary: 框架 vs 权重/量化 定界")
    logger.info("=" * 70)
    logger.info(f"  量化模型: {quant}")
    logger.info(f"  设备: {devices}")
    try:
        request_raw = args.request_json
        if args.request_json_stdin:
            request_raw = sys.stdin.read()
        elif args.prompt_stdin:
            args.prompt = sys.stdin.read()
        request_payload = parse_request_json(request_raw) if request_raw else None
    except ValueError as exc:
        logger.error(f"  Boundary 请求 JSON 无效: {exc}")
        return None
    messages = request_payload.get("messages") if request_payload else _parse_messages(args)
    bad_pattern = {}
    if args.repeat_4gram_max is not None:
        bad_pattern["repeat_4gram_max"] = args.repeat_4gram_max
    if args.nonprintable_max is not None:
        bad_pattern["nonprintable_max"] = args.nonprintable_max
    framework_reproduced = (
        args.framework_bad_reproduced == "true"
        if args.framework_bad_reproduced is not None else None
    )
    result = run_boundary(
        quant_model_path=quant,
        devices=devices,
        ref_model_path=args.ref_model,
        prompt=args.prompt if messages is None else None,
        messages=messages,
        prompt_file=args.prompt_file,
        request_payload=request_payload,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        thinking=args.thinking,
        framework_name=args.framework_name,
        framework_bad_output=args.framework_bad_output,
        framework_bad_reproduced=framework_reproduced,
        bad_pattern=bad_pattern or None,
        run_ref=not args.no_ref,
        verbose=True,
        num_runs=args.num_runs,
        concurrency=args.concurrency,
        stop_on_first_badcase=args.stop_on_first_badcase,
        expert_chunk_size=args.expert_chunk_size,
        prefill_parallel=getattr(args, "prefill_parallel", "pp"),
        glm_attn_query_block=getattr(args, "glm_attn_query_block", None),
        glm_attn_selected_block=getattr(args, "glm_attn_selected_block", None),
    )
    d = boundary_result_to_dict(result)
    logger.info("\n  定界结果: " + d["boundary_result"])
    logger.info(f"    framework={d.get('framework_name') or 'n/a'} "
                f"framework_reproduced={d.get('framework_badcase_reproduced')} "
                f"transformers_quant_reproduced={d.get('transformers_badcase_reproduced')} "
                f"ref_reproduced={d.get('ref_badcase_reproduced')}")
    quant_summary = d.get("evidence", {}).get("transformers_run", {}).get("quant", {})
    if quant_summary:
        logger.info(
            f"    quant runs={quant_summary.get('completed_runs')}/"
            f"{quant_summary.get('requested_runs')} bad={quant_summary.get('badcase_runs')} "
            f"rate={quant_summary.get('badcase_rate')}"
        )
    if d.get("limitations"):
        logger.info("    限制: " + "; ".join(d["limitations"])[:500])
    out_dir = _resolve_report_run_dir(args, "boundary")
    os.makedirs(out_dir, exist_ok=True)
    bpath = os.path.join(out_dir, "boundary_result.json")
    with open(bpath, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    runs_path = os.path.join(out_dir, "boundary_runs.jsonl")
    transformers_runs = d.get("evidence", {}).get("transformers_run", {})
    with open(runs_path, "w", encoding="utf-8") as f:
        for model_kind in ("quant", "ref"):
            for run in transformers_runs.get(model_kind, {}).get("runs", []):
                f.write(json.dumps(
                    {"model_kind": model_kind, **run}, ensure_ascii=False
                ) + "\n")
    logger.info(f"  Boundary 汇总: {bpath}")
    logger.info(f"  Boundary 逐次结果: {runs_path}")
    # 写入 AlignmentReport (供 full 模式复用)
    return d


def _mode_l1(args):
    if not args.ref_model:
        logger.info("\n  L1需要指定 --ref_model")
        return None
    if args.prompt_file and args.prompt_file.lower().endswith((".txt", ".text", ".prompt")):
        with open(args.prompt_file, encoding="utf-8") as prompt_handle:
            args.prompt = prompt_handle.read()
    ref_device, target_device, dtype = _resolve_devices(args)
    return run_hf_l1(args, ref_device, target_device, dtype)


def _report_run_name(args, mode):
    """Build a filesystem-safe, unique name for one archived report run."""
    model_path = args.quant_model or args.ref_model or "model"
    model_name = os.path.basename(os.path.normpath(model_path)) or "model"
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{safe_name or 'model'}_{mode}_{stamp}"


def _default_report_run_dir(args, mode):
    """Return the default archive directory for one report-producing run."""
    return os.path.join("reports", _report_run_name(args, mode))


def _resolve_report_run_dir(args, mode):
    """Resolve an output directory without overwriting a previous report.

    ``--output_dir reports`` is treated as the report archive root.  A more
    specific path is kept for its first run; if it already contains files, a
    timestamped sibling is used so history remains available from latest.html.
    """
    requested = getattr(args, "output_dir", None)
    if not requested:
        return _default_report_run_dir(args, mode)

    requested = os.path.normpath(requested)
    if os.path.basename(requested).lower() == "reports":
        return os.path.join(requested, _report_run_name(args, mode))

    if os.path.isdir(requested) and os.listdir(requested):
        parent = os.path.dirname(requested)
        base = os.path.basename(requested)
        return os.path.join(parent, f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    return requested


def _write_stage_report_artifacts(args, report, mode):
    """Persist text, ReportData JSON and product HTML for standalone L1/L2."""
    if report.l1 is None and not report.l2_reports:
        return None

    from accuracy_checker import (
        assemble_report,
        generate_index_html,
        generate_product_html_report,
    )

    out_dir = _resolve_report_run_dir(args, mode)
    os.makedirs(out_dir, exist_ok=True)

    summary_text = report.summary()
    text_path = os.path.join(out_dir, "alignment_report.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    if report.l1 is not None:
        with open(os.path.join(out_dir, "l1_report.txt"), "w", encoding="utf-8") as f:
            f.write(report.l1.summary())

    device_mode = (
        getattr(args, "ref_devices", None)
        or getattr(args, "ref_device", None)
        or getattr(args, "devices", None)
        or getattr(args, "device", "")
    )
    report_data = assemble_report(
        l1_report=report.l1,
        l2_results=report.l2_reports,
        model_name=args.model_name or os.path.basename(args.quant_model or "model"),
        ref_model_path=args.ref_model or "",
        quant_model_path=args.quant_model or "",
        quant_format=args.quant_format or "",
        device_mode=str(device_mode or ""),
        prompt=(getattr(args, "messages", None) or args.prompt or ""),
        input_mode="messages" if getattr(args, "messages", None) else "prompt",
        quant_method=getattr(args, "quant_method", "") or "",
        activation_quant_enabled=bool(getattr(args, "activation_quant", False)),
        activation_quant_type=getattr(args, "activation_quant_type", "") or "",
        activation_quant_backend=getattr(args, "activation_quant_backend", "") or "",
        activation_quant_group_size=getattr(args, "activation_quant_group_size", None),
    )
    json_path = os.path.join(out_dir, "report_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(report_data.to_json(indent=2))
    html_path = generate_product_html_report(
        report_data, output_path=os.path.join(out_dir, "product_report.html")
    )

    reports_root = os.path.dirname(os.path.abspath(out_dir))
    try:
        index_path = generate_index_html(reports_root)
        logger.info(f"  报告索引: {index_path}")
        logger.info("  当前/历史统一入口: latest.html")
    except Exception as exc:  # noqa: BLE001
        logger.info(f"  报告索引生成跳过: {exc}")

    logger.info(f"\n  文本报告: {text_path}")
    logger.info(f"  JSON 报告: {json_path}")
    logger.info(f"  HTML 报告: {html_path}")
    return html_path


def _mode_l2(args, l1_report=None):
    target_layers = _parse_target_layers(args)
    if target_layers is None and l1_report is not None and l1_report.first_bad_block:
        fbb = l1_report.first_bad_block
        if fbb.startswith("layer."):
            try:
                layer_idx = int(fbb.split(".")[1])
                target_layers = [layer_idx]
                if layer_idx > 0:
                    target_layers.insert(0, layer_idx - 1)
            except (ValueError, IndexError):
                pass

    # 从 cache 扫描可用层
    if target_layers is None:
        from accuracy_checker.cache import get_cache_dir, model_hash, prompt_hash
        import re
        _cache_dir = get_cache_dir()
        if args.ref_model and os.path.exists(_cache_dir):
            ref_mh = model_hash(args.ref_model)
            ph = prompt_hash(_cache_input_identity(args))
            cached_layers = set()
            for fname in os.listdir(_cache_dir):
                if f"{ref_mh}_{ph}" in fname and "_ref_" in fname:
                    m = re.search(r'_L(\d+)_', fname)
                    if m:
                        cached_layers.add(int(m.group(1)))
            if cached_layers:
                target_layers = sorted(cached_layers)
                logger.info(f"  [L2] 从 cache 自动发现 {len(target_layers)} 层: {target_layers}")

    if not target_layers:
        logger.info("\n  L2: 没有目标层 (请用 --target-layers 或先跑 L1 --cache-top-k)")
        return None
    if not args.ref_model:
        logger.info("\n  L2需要指定 --ref_model")
        return None
    ref_device, target_device, dtype = _resolve_devices(args)
    return run_hf_l2(args, ref_device, target_device, dtype, target_layers)


def _build_inference_compare_from_boundary(boundary_dict, quant_model_path, prompt=""):
    """从 boundary evidence 提取 ref/quant 生成文本, 构建 InferenceCompareData。

    Boundary 步骤已经跑了 ref+quant 模型 generate, 这里复用其输出做 token 级对比,
    无需额外模型加载。
    """
    if not boundary_dict:
        return None
    evidence = boundary_dict.get("evidence", {})
    tr = evidence.get("transformers_run", {})
    ref_info = tr.get("ref", {})
    quant_info = tr.get("quant", {})
    ref_text = ref_info.get("output_full", "")
    quant_text = quant_info.get("output_full", "")
    if not ref_text and not quant_text:
        return None

    # Tokenize for token-level comparison
    ref_tokens = []
    quant_tokens = []
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            quant_model_path, trust_remote_code=True, local_files_only=True)
        ref_tokens = tok.tokenize(ref_text) if ref_text else []
        quant_tokens = tok.tokenize(quant_text) if quant_text else []
    except Exception as e:
        logger.info(f"  [inference_compare] tokenizer 加载失败, 仅做文本级对比: {e}")

    from accuracy_checker.inference_compare import compare_inference
    comparison = compare_inference(
        ref_text,
        quant_text,
        ref_tokens,
        quant_tokens,
        prompt=prompt,
    )

    logger.info(f"  [inference_compare] ref {len(ref_tokens)} tokens, "
                f"quant {len(quant_tokens)} tokens, "
                f"match={(comparison.token_match_rate or 0.0)*100:.1f}%, "
                f"exact={comparison.exact_match}, "
                f"first_div={comparison.first_divergence_pos}")
    return comparison


def _full_run_l0(args):
    """[full] L0 预检, 返回 l0_result (失败返回 None)。"""
    if not (args.ref_model and args.quant_model):
        return None
    from accuracy_checker import run_l0_sanity
    logger.info("\n" + "-" * 50 + "\n  [full] 1/6 L0 预检\n" + "-" * 50)
    try:
        l0_result = run_l0_sanity(
            ref_model_path=args.ref_model, quant_model_path=args.quant_model,
            rotation_matrix=args.rotation_matrix, expected_dtype=args.dtype)
        logger.info("  " + l0_result.summary)
        if l0_result.overall_status == "INVALID_RUN":
            logger.info("  [full] L0 判定 INVALID_RUN, 仍继续 (结果会标注不可信)")
        return l0_result
    except Exception as e:  # noqa: BLE001
        logger.info(f"  [full] L0 失败: {e}")
        return None


def _full_run_l1(args, ref_device, target_device, dtype):
    """[full] L1 逐层对比, 返回 l1_report (失败/跳过返回 None)。"""
    logger.info("\n" + "-" * 50 + "\n  [full] 3/6 L1 逐层对比\n" + "-" * 50)
    try:
        if args.ref_model:
            l1_report = run_hf_l1(args, ref_device, target_device, dtype)
            logger.info(l1_report.summary())
            return l1_report
        logger.info("  [full] 跳过 L1 (无 --ref_model)")
    except Exception as e:  # noqa: BLE001
        logger.info(f"  [full] L1 失败: {e}")
    return None


def _full_run_l2(args, ref_device, target_device, dtype, target_layers):
    """[full] L2 子图诊断, 返回 l2_results (失败/跳过返回 None)。"""
    if not (target_layers and args.ref_model):
        return None
    logger.info("\n" + "-" * 50 + "\n  [full] 4/6 L2 子图诊断\n" + "-" * 50)
    try:
        return run_hf_l2(args, ref_device, target_device, dtype, target_layers)
    except Exception as e:  # noqa: BLE001
        logger.info(f"  [full] L2 失败: {e}")
        return None


def _full_resolve_logits(args, l1_report, ref_device, target_device, dtype):
    """[full] Logits: 优先复用 L1 forward 采集; 否则 standalone 采集。"""
    if l1_report is not None and getattr(l1_report, "logits_data", None) is not None:
        logits_comp = l1_report.logits_data
        n_pos = len(logits_comp.token_positions)
        logger.info(f"\n  [full] Logits 复用 L1 forward: {n_pos} positions (无额外模型加载)")
        return logits_comp
    if args.logits and args.ref_model and args.quant_model:
        logger.info("\n" + "-" * 50 + "\n  [full] 5/6 Logits 对比 (standalone)\n" + "-" * 50)
        return _run_logits(args, ref_device, target_device, dtype)
    return None


def _full_run_badcase_compare(args, l2_results):
    """[full] Bad case ground truth 对比 (--manifest 触发)。"""
    if not (args.manifest and l2_results):
        return None
    logger.info("\n" + "-" * 50 + "\n  [full] Bad Case ground truth 对比\n" + "-" * 50)
    try:
        from accuracy_checker import load_manifest, compare_with_ground_truth
        manifest = load_manifest(args.manifest)
        badcase_cmp = compare_with_ground_truth(l2_results, manifest)
        logger.info(f"  ground_truth: {badcase_cmp.ground_truth}")
        logger.info(f"  tool_located: {badcase_cmp.source_candidate}")
        logger.info(f"  whether_hit: {badcase_cmp.whether_hit_ground_truth}")
        logger.info(f"  hit_detail: {badcase_cmp.hit_detail}")
        return badcase_cmp
    except Exception as e:  # noqa: BLE001
        logger.info(f"  [full] Bad case 对比失败: {e}")
        return None


def _full_build_inference_compare(args, boundary_dict):
    """[full] 推理对比: 从 boundary evidence 提取 ref/quant 生成文本。"""
    if not boundary_dict:
        return None
    logger.info("\n  [full] 推理结果对比 (复用 Boundary 生成)")
    try:
        return _build_inference_compare_from_boundary(
            boundary_dict, args.quant_model, prompt=args.prompt or "")
    except Exception as e:  # noqa: BLE001
        logger.info(f"  [full] 推理对比构建失败: {e}")
        return None


def _full_assemble_and_write(args, l1_report, l2_results, boundary_dict,
                              logits_comp, inference_compare):
    """[full] 组装 ReportData 并写 JSON + HTML + index。"""
    from accuracy_checker import assemble_report, generate_product_html_report
    logger.info("\n" + "-" * 50 + "\n  [full] 6/6 报告组装 (JSON + HTML)\n" + "-" * 50)
    report_data = assemble_report(
        l1_report=l1_report,
        l2_results=l2_results,
        boundary_result=[boundary_dict] if boundary_dict else None,
        logits_comparison=logits_comp,
        inference_compare_data=inference_compare,
        model_name=args.model_name or os.path.basename(args.quant_model or "model"),
        ref_model_path=args.ref_model or "",
        quant_model_path=args.quant_model or "",
        quant_format=args.quant_format,
        device_mode=args.devices or args.device,
        prompt=(getattr(args, "messages", None) or args.prompt or ""),
        input_mode="messages" if getattr(args, "messages", None) else "prompt",
        quant_method=getattr(args, "quant_method", "") or "",
        activation_quant_enabled=bool(getattr(args, "activation_quant", False)),
        activation_quant_type=getattr(args, "activation_quant_type", "") or "",
        activation_quant_backend=getattr(args, "activation_quant_backend", "") or "",
        activation_quant_group_size=getattr(args, "activation_quant_group_size", None),
    )
    logger.info(f"  run_status = {report_data.run_status}")

    out_dir = _resolve_report_run_dir(args, "full")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "report_data.json")
    with open(json_path, "w") as f:
        f.write(report_data.to_json(indent=2))
    logger.info(f"  JSON 报告: {json_path}")

    html_path = generate_product_html_report(
        report_data, output_path=os.path.join(out_dir, "product_report.html"))
    logger.info(f"  HTML 报告: {html_path}")

    # Generate index.html with sidebar history
    from accuracy_checker import generate_index_html
    reports_base = os.path.dirname(os.path.abspath(out_dir))
    try:
        index_path = generate_index_html(reports_base)
        logger.info(f"  Index HTML: {index_path}")
        logger.info("  当前/历史统一入口: latest.html")
    except Exception as e:  # noqa: BLE001
        logger.info(f"  [full] Index HTML 生成失败: {e}")


def _mode_full(args):
    """L0 → L1 → Top-K → L2 → logits → JSON → HTML。Boundary 独立运行。"""
    ref_device, target_device, dtype = _resolve_devices(args)

    # L0
    _full_run_l0(args)

    # Boundary 的设备布局与 L1/L2 不同（例如 Quant-only 独占 16 卡），独立运行。
    logger.info("\n  [full] 跳过 Boundary (请用 --mode boundary 单独跑)")
    boundary_dict = None

    # L1
    l1_report = _full_run_l1(args, ref_device, target_device, dtype)

    # Top-K 选取
    target_layers = _parse_target_layers(args)
    if target_layers is None and l1_report is not None:
        target_layers = _topk_from_l1(l1_report, args.top_k)
        logger.info(f"\n  [full] L1 后自动选取 Top-{args.top_k} 层: {target_layers}")

    # L2
    l2_results = _full_run_l2(args, ref_device, target_device, dtype, target_layers)

    # Logits
    logits_comp = _full_resolve_logits(args, l1_report, ref_device, target_device, dtype)

    # Badcase ground truth 对比 (可选, --manifest 触发)
    _full_run_badcase_compare(args, l2_results)

    # 推理对比
    inference_compare = _full_build_inference_compare(args, boundary_dict)

    # 组装 ReportData
    _full_assemble_and_write(args, l1_report, l2_results, boundary_dict,
                              logits_comp, inference_compare)


def _mode_report(args):
    """从 JSON ReportData 重新渲染 HTML。"""
    json_path = args.result_json
    if not json_path:
        logger.info("\n  [report] 需要 --result-json <path>")
        return
    if not os.path.exists(json_path):
        logger.info(f"\n  [report] JSON 文件不存在: {json_path}")
        return
    from accuracy_checker import (
        ReportData,
        generate_index_html,
        generate_product_html_report,
    )
    with open(json_path) as f:
        data = json.load(f)
    report_data = ReportData.from_dict(data)
    out_dir = args.output_dir or os.path.dirname(json_path) or "reports"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "product_report.html")
    path = generate_product_html_report(report_data, output_path=out_path)
    logger.info(f"\n  [report] HTML 已生成: {path}")
    reports_base = os.path.dirname(os.path.abspath(out_dir))
    try:
        index_path = generate_index_html(reports_base)
        logger.info(f"  [report] 当前/历史统一入口: latest.html ({index_path})")
    except Exception as exc:  # noqa: BLE001
        logger.info(f"  [report] 报告索引生成跳过: {exc}")


def _mode_inference(args):
    """独立推理 HTML: 1 模型=单展示, 2 模型=并排对比+token diff。

    三条路径:
      A. --result_json (简单格式, 含 "models" 键): 直接用于 generate_inference_html
         (forward_pass_nothink.py --save_json 产出的格式)
      B. --result_json (ReportData 格式, 含 "overview" 键): 提取 inference_compare
      C. 直接加载模型 generate (仅适合单/双卡 HF 模型; 16 卡用路径 A)
    """
    from accuracy_checker.inference_html import generate_inference_html

    prompt = args.prompt
    models_data = []

    # 路径 A/B: 从 JSON 读
    if args.result_json:
        import json
        if not os.path.exists(args.result_json):
            logger.info(f"\n  [inference] JSON 不存在: {args.result_json}")
            return
        with open(args.result_json) as f:
            raw = json.load(f)

        if "models" in raw:
            # 路径 A: 简单格式 (forward_pass_nothink.py --save_json)
            models_data = raw["models"]
            prompt = raw.get("prompt", prompt)
            logger.info(f"  [inference] 从 JSON 加载 {len(models_data)} 个模型")
        elif "overview" in raw:
            # 路径 B: ReportData 格式
            from accuracy_checker import ReportData
            rd = ReportData.from_dict(raw)
            ic = rd.inference_compare
            if not ic or (not ic.ref_output and not ic.quant_output):
                logger.info("\n  [inference] JSON 中无推理对比数据")
                return
            if ic.ref_output and ic.quant_output:
                models_data = [
                    {"name": "ref", "output": ic.ref_output,
                     "token_strs": ic.ref_tokens or []},
                    {"name": "quant", "output": ic.quant_output,
                     "token_strs": ic.quant_tokens or []},
                ]
            else:
                out = ic.quant_output or ic.ref_output
                toks = ic.quant_tokens or ic.ref_tokens or []
                models_data = [{"name": "quant", "output": out, "token_strs": toks}]
        else:
            logger.info("\n  [inference] JSON 格式无法识别 (需要 'models' 或 'overview' 键)")
            return
    else:
        # 路径 B: 直接 generate
        if not args.quant_model:
            logger.info("\n  [inference] 需要 --quant_model 或 --result_json")
            return
        ref_device, target_device, dtype = _resolve_devices(args)
        from transformers import AutoTokenizer
        from accuracy_checker.model_loader import load_model_for_comparison

        def _run_one(model_path, device, name, quant_method="dequantize"):
            tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = load_model_for_comparison(
                model_path, device=device, dtype=dtype,
                quant_method=quant_method,
                use_fake_quant=(quant_method == "fake_quant"))
            model.eval()
            inputs = tok(prompt, return_tensors="pt").to(model.device if hasattr(model, "device") else device)
            import time as _t
            t0 = _t.time()
            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                         do_sample=False)
            elapsed = _t.time() - t0
            gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
            output_text = tok.decode(gen_ids, skip_special_tokens=True)
            token_strs = [tok.decode([tid]) for tid in gen_ids.tolist()]
            n = len(token_strs)
            prefill_t = None
            decode_t = None
            if n > 0:
                # 粗略: 总时间按 prefill≈首token比例估算
                prefill_t = elapsed * 0.3
                decode_t = [elapsed * 0.7 / n] * n
            logger.info(f"  [inference] {name}: {n} tokens, {elapsed:.1f}s")
            del model
            return {"name": name, "output": output_text, "token_strs": token_strs,
                    "num_tokens": n, "prefill_time": prefill_t,
                    "decode_times": decode_t}

        # 2 模型对比
        if args.ref_model:
            models_data.append(_run_one(args.ref_model, ref_device, "ref"))
            models_data.append(_run_one(args.quant_model, target_device, "quant",
                                        quant_method=args.quant_method))
        else:
            models_data.append(_run_one(args.quant_model, target_device,
                                         args.model_name or "quant",
                                         quant_method=args.quant_method))

    out_dir = args.output_dir or "reports"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "inference_report.html")
    path = generate_inference_html({"prompt": prompt, "models": models_data},
                                   output_path=out_path)
    logger.info(f"\n  [inference] HTML 已生成: {path}")
    logger.info(f"  [inference] 短链: latest_inference.html")


# ===========================================================================
# 入口
# ===========================================================================

def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(message)s",
    )

    if args.cache_dir:
        from accuracy_checker.cache import set_cache_dir
        set_cache_dir(args.cache_dir)

    if args.clean_cache:
        _clean_l2_cache()
        return

    mode = _resolve_mode(args)
    # Long L1/full prompts can be streamed through stdin just like Boundary.
    # Boundary consumes its stdin inside _mode_boundary because it may contain
    # either a request JSON or a raw prompt; other modes only need plain text.
    if getattr(args, "prompt_stdin", False) and mode != "boundary":
        args.prompt = sys.stdin.read()
    if args.max_new_tokens is None and mode != "boundary":
        args.max_new_tokens = 1024

    from accuracy_checker.dspark import is_dspark_checkpoint
    standalone_dspark = (
        getattr(args, "model_type", "auto") == "dspark"
        or is_dspark_checkpoint(getattr(args, "ref_model", None))
        or is_dspark_checkpoint(getattr(args, "quant_model", None))
    )
    _validate_standalone_dspark_args(args, mode, standalone_dspark)

    # report / inference 模式不需要 quant_model
    if mode not in ("report", "inference") and not args.quant_model:
        logger.info("\n  需要指定 --quant_model (report / inference 模式除外)")
        return

    if mode == "screening":
        _mode_screening(args); return
    if mode == "boundary":
        _mode_boundary(args); return
    if mode == "report":
        _mode_report(args); return
    if mode == "inference":
        _mode_inference(args); return

    report = AlignmentReport()

    if mode == "l1":
        l1_report = _mode_l1(args)
        if l1_report is not None:
            report.set_l1(l1_report)
            logger.info(l1_report.summary())
    elif mode == "l2":
        l2 = _mode_l2(args)
        if l2:
            for l2r in l2:
                report.add_l2(l2r)
    elif mode == "full":
        _mode_full(args)
        return

    summary_text = report.summary()
    logger.info(summary_text)
    _write_stage_report_artifacts(args, report, mode)


if __name__ == "__main__":
    main()
