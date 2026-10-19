#!/usr/bin/env python3
"""Read-only PLE/hash/lookup diagnostics; does not run a complete L1 model.

Exit 0: sampled lookup checks aligned; 1: differences/incomplete checks;
2: invalid metadata, invalid input, or execution failure. See JSON for scope.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref_model", required=True)
    parser.add_argument("--quant_model", required=True)
    parser.add_argument("--layer", type=int, default=1, help="Zero-based decoder layer; default 1 (PLE config layer 2)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input_ids_json", help="Exact integer IDs: list, {input_ids:[...]}, or captured {prompt:[...]}")
    source.add_argument("--l1_cache", help="Exact L1 .pt cache file containing input_ids; loads safely with weights_only + mmap")
    source.add_argument("--prompt", help="Same text as L1; tokenized once using ref tokenizer")
    source.add_argument("--prompt_file", help="Raw UTF-8 text file, same semantics as L1 text input")
    parser.add_argument("--chat_template_mode", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--thinking", choices=("chat", "none"), default="chat")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cpu", help="cpu, npu:0, or cuda:0; only small hash tensors use this device")
    parser.add_argument("--max_positions", type=int, default=256,
                        help="Evenly spaced lookup positions, plus bounded EOS-neighbour probes; full prefix always hashed")
    parser.add_argument("--output", help="JSON output; default reports/qwen4_ple_diagnostic_TIMESTAMP.json")
    return parser


def load_input(args):
    import torch
    from accuracy_checker.qwen4_ple_diagnostics import validate_input_ids
    if args.input_ids_json:
        value = json.loads(Path(args.input_ids_json).read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("input_ids", value.get("prompt"))
        return validate_input_ids(value), dict(kind="recorded_input_ids", path=str(Path(args.input_ids_json).resolve()), retokenized=False)
    if args.l1_cache:
        value = torch.load(args.l1_cache, weights_only=True, map_location="cpu", mmap=True)
        if not isinstance(value, dict) or value.get("input_ids") is None:
            raise ValueError("L1 cache has no recorded input_ids; provide --input_ids_json or the exact L1 prompt")
        return validate_input_ids(value["input_ids"]), dict(kind="l1_cache", path=str(Path(args.l1_cache).resolve()), retokenized=False)
    from transformers import AutoTokenizer
    from accuracy_checker.input_resolver import resolve_model_input
    tokenizer = AutoTokenizer.from_pretrained(args.ref_model, trust_remote_code=True)
    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt
    resolved = resolve_model_input(tokenizer, prompt=prompt, chat_template_mode=args.chat_template_mode, thinking=args.thinking)
    return validate_input_ids(resolved["input_ids"]), dict(kind="prompt_file" if args.prompt_file else "prompt",
            tokenizer=args.ref_model, chat_template_mode=args.chat_template_mode, thinking=args.thinking,
            retokenized=True, note="Both sides use these same IDs; use recorded IDs/cache for exact reproduction of a past L1 run")


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = Path(args.output or f"reports/qwen4_ple_diagnostic_{datetime.now():%Y%m%d_%H%M%S_%f}.json")
    report = dict(status="ERROR", layer=args.layer, ref_model=args.ref_model, quant_model=args.quant_model)
    checkpoints = []
    try:
        import torch
        import transformers
        from accuracy_checker.qwen4_ple_diagnostics import inspect_checkpoint, diagnose_pair
        if args.max_positions < 1:
            raise ValueError("--max_positions must be positive")
        if not (args.device == "cpu" or args.device.startswith(("npu:", "cuda:"))):
            raise ValueError("--device must be cpu, npu:N or cuda:N")
        if args.device.startswith("npu:"):
            import torch_npu  # noqa: F401 — registers the backend
            torch.npu.set_device(args.device)
        ids, source = load_input(args)
        print(f"[PLE] layer={args.layer}, input={list(ids.shape)}, device={args.device}", flush=True)
        print("[PLE] Read-only: no full model / decoder / experts / activation-QDQ. Reading PLE metadata and sampled rows.", flush=True)
        for name, path in (("ref", args.ref_model), ("quant", args.quant_model)):
            print(f"[PLE] Inspect {name}: {path}", flush=True)
            checkpoints.append(inspect_checkpoint(path, args.layer, dtype=getattr(torch, args.dtype)))
        print("[PLE] Compare full-prefix hashes / installed HF CPU+device hash / sampled lookup and shard boundaries...", flush=True)
        report = diagnose_pair(*checkpoints, ids, max_positions=args.max_positions, device=args.device)
        report["input_source"] = source
        report["versions"] = dict(torch=torch.__version__, transformers=transformers.__version__)
        report["lookup_dtype"] = args.dtype
        print(f"[PLE] Status: {report['status']}")
        for name, value in report["checks"].items():
            check = value.get("comparison", value)
            if "equal" in check:
                print(f"  {'PASS' if check['equal'] else 'DIFF'} {name}: max_abs_diff={check.get('max_abs_diff')}, rel_l2={check.get('rel_l2')}")
            elif value.get("status") == "unavailable":
                print(f"  NOT CHECKED {name}: {value['error']}")
        for side, value in report["checkpoints"].items():
            for issue in value["issues"]:
                print(f"  {side} {issue['code']}: {issue['message']}")
        for item in report["findings"]:
            print(f"  {item['code']}: {item['message']}")
        print(report.get("next_step", "Invalid metadata: inspect checkpoint buffer dtype/content before replay."))
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["status"] = "ERROR"
        print(f"[PLE] ERROR: {report['error']}")
    finally:
        for checkpoint in checkpoints:
            checkpoint.close()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"[PLE] JSON: {output.resolve()}")
    return 0 if report["status"] == "SAMPLED_LOOKUP_ALIGNED" else (2 if report["status"] in {"ERROR", "INVALID_METADATA"} else 1)


if __name__ == "__main__":
    raise SystemExit(main())
