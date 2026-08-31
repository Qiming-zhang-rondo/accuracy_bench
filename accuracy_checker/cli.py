"""
inference_check CLI 入口。

从 inference_check.py 拆出以满足 CI LinesPerFile 约束 (≤2000 行)。
"""
from __future__ import annotations

import argparse
import logging

from .inference_check import hf_inference_check, _run_boundary_cli


def main():
    parser = argparse.ArgumentParser(description="HF 推理检查 — 排除框架影响")
    parser.add_argument("--mode", default="inference",
                        choices=["inference", "boundary"],
                        help="inference=纯推理检查; boundary=框架vs权重定界")
    parser.add_argument("--boundary", action="store_true",
                        help="--mode boundary 的简写 (向后兼容)")
    parser.add_argument("--model_path", required=True, help="模型路径")
    parser.add_argument("--devices", default="npu:0",
                        help="逻辑设备列表，如 npu:0,1,2,3")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="最大生成 token；boundary 未传时读取 prompt_file 的 max_tokens")
    parser.add_argument("--skip_ppl", action="store_true")
    parser.add_argument("--thinking", default="chat",
                        choices=["chat", "none"],
                        help="thinking 模式: chat=开思维链, none=关闭")
    parser.add_argument("--prompt_file", default=None,
                        help="JSON 文件，vLLM 请求格式或对话列表")
    parser.add_argument("--request_json", default=None,
                        help="直接粘贴完整 OpenAI/vLLM 请求 JSON")
    parser.add_argument("--request_json_stdin", action="store_true",
                        help="从 stdin 读取完整请求 JSON，适合超长 prompt")
    parser.add_argument("--prompt_stdin", action="store_true",
                        help="从 stdin 读取原始 prompt，适合 boundary/L1/full 超长文本")
    parser.add_argument(
        "--chat_template_mode", choices=["auto", "always", "never"], default="auto",
        help="控制 tokenizer chat_template：auto（默认）、always、never",
    )
    parser.add_argument("--use_cpu_dequant", action="store_true",
                        help="回退到旧 CPU 全量反量化流程")
    parser.add_argument("--noquit", action="store_true",
                        help="推理完成后不退出，进入交互模式 (模型留在NPU上)")
    # ---- boundary mode args ----
    parser.add_argument("--prompt", default=None,
                        help="[boundary] 单轮 plain text prompt")
    parser.add_argument("--ref_model_path", default=None,
                        help="[boundary] 参考 BF16 模型路径; 给了会跑 ref 区分 quant 回归 vs base 本征")
    parser.add_argument("--framework_name", default=None,
                        help="[boundary] 部署框架名 (vllm/mindie/...)")
    parser.add_argument("--framework_bad_output", default=None,
                        help="[boundary] 部署框架实际生成的坏文本 (推荐提供)")
    parser.add_argument("--framework_bad_reproduced", default=None,
                        choices=["true", "false"],
                        help="[boundary] 调用方直接断言框架是否复现 (true/false)")
    parser.add_argument("--boundary_issue_mode", default="reproducible",
                        choices=["reproducible", "intermittent"],
                        help="[boundary] reproducible（默认）或 intermittent captured logits replay")
    parser.add_argument("--captured_logits_json", default=None,
                        help="[boundary/intermittent] vLLM 现场 captured logits JSON")
    parser.add_argument("--boundary_logits_cos_threshold", type=float, default=0.99)
    parser.add_argument("--boundary_logits_kl_threshold", type=float, default=0.05)
    parser.add_argument("--boundary_logits_margin_threshold", type=float, default=0.05)
    parser.add_argument("--repeat_4gram_max", type=float, default=None,
                        help="[boundary] 4-gram 重复比阈值, 超过判为 bad; 缺省 0.5")
    parser.add_argument("--nonprintable_max", type=float, default=None,
                        help="[boundary] 非可打印字符比阈值, 缺省 0.3")
    parser.add_argument("--no_ref", action="store_true",
                        help="[boundary] 跳过 ref 运行 (默认 run_ref=True)")
    parser.add_argument("--json_out", action="store_true",
                        help="[boundary] 以 JSON 打印结构化结果 (供 Agent D 解析)")
    parser.add_argument("--num_runs", type=int, default=1,
                        help="[boundary] 同一 prompt 总运行次数")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="[boundary] 每个 Transformers generate batch 的并发样本数")
    parser.add_argument("--stop_on_first_badcase", action="store_true",
                        help="[boundary] 首批检测到 bad case 后提前停止")
    parser.add_argument("--print_full_output", action="store_true",
                        help="[boundary/inference] 在终端打印每次生成的完整文本")
    parser.add_argument("--expert_chunk_size", type=int, default=None,
                        help="[boundary] resident expert 临时反量化分块，默认 8")
    parser.add_argument(
        "--prefill_parallel", choices=["pp", "tp"], default="pp",
        help="长 prefill: pp=按层分片（默认）, tp=GLM/DeepSeek-V4 query-parallel",
    )
    parser.add_argument(
        "--glm_attn_query_block", type=int, default=None,
        help="GLM 长 prefill attention query block（默认 64）",
    )
    parser.add_argument(
        "--glm_attn_selected_block", type=int, default=None,
        help="GLM 长 prefill selected-key block（默认 512）",
    )
    parser.add_argument(
        "--deepseek_v4_query_block", type=int, default=None,
        help="DeepSeek-V4 blockwise/TP query block（默认 64）",
    )
    parser.add_argument(
        "--deepseek_v4_key_block", type=int, default=None,
        help="DeepSeek-V4 blockwise/TP key block（默认 1024）",
    )
    args = parser.parse_args()

    if args.boundary and args.mode == "inference":
        args.mode = "boundary"

    if args.mode == "boundary":
        if args.request_json_stdin:
            import sys
            args.request_json = sys.stdin.read()
        elif args.prompt_stdin:
            import sys
            args.prompt = sys.stdin.read()
        _run_boundary_cli(args)
        return

    hf_inference_check(
        model_path=args.model_path,
        devices=args.devices,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens or 2048,
        prompt_file=args.prompt_file,
        skip_ppl=args.skip_ppl,
        thinking=args.thinking,
        use_cpu_dequant=args.use_cpu_dequant,
        noquit=args.noquit,
        prefill_parallel=args.prefill_parallel,
        glm_attn_query_block=args.glm_attn_query_block,
        glm_attn_selected_block=args.glm_attn_selected_block,
        deepseek_v4_query_block=args.deepseek_v4_query_block,
        deepseek_v4_key_block=args.deepseek_v4_key_block,
        chat_template_mode=args.chat_template_mode,
        print_full_output=args.print_full_output,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
