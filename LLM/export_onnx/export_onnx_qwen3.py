#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export a HuggingFace causal LM such as Qwen3 to a fixed-shape ONNX graph."""

import argparse
import gc
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM


class CausalLMLogitsWrapper(torch.nn.Module):
    """Keep the exported graph focused on prefill logits and avoid KV-cache outputs."""

    def __init__(self, model, logits_to_keep=0):
        super().__init__()
        self.model = model
        self.logits_to_keep = logits_to_keep

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=self.logits_to_keep,
            return_dict=True,
        )
        return outputs.logits


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Qwen/Qwen-like causal LM to a fixed-shape ONNX file."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/home/qsy/huggingface/model/Qwen3-0.6B",
        help="Local HuggingFace model directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/qsy/workspace/dataset/onnx/qwen3_0_6b_seq128.onnx",
        help="Output ONNX path.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Fixed batch size.")
    parser.add_argument("--seq-len", type=int, default=128, help="Fixed sequence length.")
    parser.add_argument(
        "--batch-sizes",
        help="Comma-separated batch-size grid, for example 1,2,4.",
    )
    parser.add_argument(
        "--seq-lens",
        help="Comma-separated sequence-length grid, for example 64,128,256.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Model weight dtype used during export.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device used during export. CPU is safer; CUDA is faster if memory is enough.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument(
        "--logits-to-keep",
        type=int,
        default=0,
        help=(
            "Only compute the last N token logits in the LM head. "
            "Use 1 for serving-style prefill; 0 preserves all sequence logits."
        ),
    )
    parser.add_argument(
        "--logits-modes",
        help="Comma-separated logits_to_keep grid, for example 0,1.",
    )
    parser.add_argument(
        "--output-root",
        help="Root directory for grid export. Enables grid naming and manifest output.",
    )
    parser.add_argument(
        "--model-tag",
        help="Filesystem/model-name tag used by grid export; defaults to model directory name.",
    )
    parser.add_argument(
        "--manifest",
        help="Manifest JSON path. Grid mode defaults to <output-root>/manifest.json.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to transformers.from_pretrained.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="eager",
        help="Attention implementation passed to from_pretrained when supported.",
    )
    parser.add_argument(
        "--dynamic-axes",
        action="store_true",
        help="Export dynamic batch/sequence axes. Disabled by default for fixed-shape TVM import.",
    )
    return parser.parse_args()


def get_torch_dtype(dtype):
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def parse_positive_int_list(raw, option_name, allow_zero=False):
    if raw is None:
        return None
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as err:
            raise ValueError(f"{option_name} contains a non-integer value: {item!r}") from err
        minimum = 0 if allow_zero else 1
        if value < minimum:
            raise ValueError(f"{option_name} values must be >= {minimum}: {value}")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError(f"{option_name} must contain at least one value")
    return values


def sanitize_tag(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = value.strip("._-")
    if not value:
        raise ValueError("--model-tag resolves to an empty filesystem name")
    return value


def logits_mode_name(logits_to_keep):
    if logits_to_keep == 0:
        return "full_logits"
    if logits_to_keep == 1:
        return "last_token"
    return f"logits_{logits_to_keep}"


def grid_output_path(output_root, model_tag, batch_size, seq_len, dtype, logits_to_keep):
    mode = logits_mode_name(logits_to_keep)
    shape_tag = f"{model_tag}-b{batch_size}-seq{seq_len}-{dtype}-{mode}"
    filename = f"{model_tag}_b{batch_size}_seq{seq_len}_{dtype}_{mode}.onnx"
    return os.path.join(output_root, shape_tag, filename)


def export_one(model, args, output_path, batch_size, seq_len, logits_to_keep):
    wrapper = CausalLMLogitsWrapper(model, logits_to_keep=logits_to_keep).eval().to(args.device)
    input_ids = torch.ones((batch_size, seq_len), dtype=torch.long, device=args.device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=args.device)

    dynamic_axes = None
    if args.dynamic_axes:
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"Exporting fixed shape: batch={batch_size}, seq_len={seq_len}")
    print(f"LM head logits_to_keep={logits_to_keep}")
    print(f"Writing ONNX to: {output_path}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
        )

    del wrapper, input_ids, attention_mask
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()


def write_manifest(path, args, model_tag, exports):
    manifest = {
        "format": "llm_compiler.onnx_export_manifest",
        "version": 1,
        "model_dir": os.path.abspath(args.model_dir),
        "model_tag": model_tag,
        "dtype": args.dtype,
        "device": args.device,
        "opset": args.opset,
        "dynamic_axes": bool(args.dynamic_axes),
        "exports": exports,
    }
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, path)
    print(f"Manifest: {path}")


def load_model(args):
    kwargs = {
        "torch_dtype": get_torch_dtype(args.dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, **kwargs)

    model.config.use_cache = False
    model.eval()
    model.to(args.device)
    return model


def main():
    args = parse_args()
    batch_sizes = parse_positive_int_list(args.batch_sizes, "--batch-sizes")
    seq_lens = parse_positive_int_list(args.seq_lens, "--seq-lens")
    logits_modes = parse_positive_int_list(
        args.logits_modes,
        "--logits-modes",
        allow_zero=True,
    )
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.seq_len < 1:
        raise ValueError("--seq-len must be positive")
    if args.logits_to_keep < 0:
        raise ValueError("--logits-to-keep must be non-negative")

    grid_mode = any(
        value is not None
        for value in (args.batch_sizes, args.seq_lens, args.logits_modes, args.output_root)
    )
    batch_sizes = batch_sizes or [args.batch_size]
    seq_lens = seq_lens or [args.seq_len]
    logits_modes = logits_modes or [args.logits_to_keep]
    if grid_mode and not args.output_root:
        raise ValueError("--output-root is required for grid export")
    if args.dynamic_axes and grid_mode:
        raise ValueError("Grid export is fixed-shape; do not combine it with --dynamic-axes")

    try:
        import onnx  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError as err:
        raise RuntimeError(
            "The onnx package is required for torch.onnx.export. "
            "Install it in an isolated export env if you do not want to change the TVM env."
        ) from err

    print(f"Loading model from: {args.model_dir}")
    print(f"Using device={args.device}, dtype={args.dtype}")
    model = load_model(args)
    raw_model_tag = args.model_tag or os.path.basename(os.path.normpath(args.model_dir))
    model_tag = sanitize_tag(raw_model_tag)
    exports = []
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            for logits_to_keep in logits_modes:
                if grid_mode:
                    output_path = grid_output_path(
                        os.path.abspath(args.output_root),
                        model_tag,
                        batch_size,
                        seq_len,
                        args.dtype,
                        logits_to_keep,
                    )
                else:
                    output_path = os.path.abspath(args.output)

                export_one(
                    model,
                    args,
                    output_path,
                    batch_size,
                    seq_len,
                    logits_to_keep,
                )
                exports.append(
                    {
                        "model_name": (
                            f"{model_tag}_b{batch_size}_seq{seq_len}_"
                            f"{args.dtype}_{logits_mode_name(logits_to_keep)}"
                        ),
                        "onnx_path": os.path.abspath(output_path),
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "dtype": args.dtype,
                        "logits_to_keep": logits_to_keep,
                        "input_shapes": {
                            "input_ids": [batch_size, seq_len],
                            "attention_mask": [batch_size, seq_len],
                        },
                        "input_dtypes": {
                            "input_ids": "int64",
                            "attention_mask": "int64",
                        },
                        "fix_mask_where_dtype": args.dtype in ("float16", "float32"),
                    }
                )

    manifest_path = args.manifest
    if grid_mode and manifest_path is None:
        manifest_path = os.path.join(os.path.abspath(args.output_root), "manifest.json")
    if manifest_path:
        write_manifest(manifest_path, args, model_tag, exports)

    print(f"Export finished: {len(exports)} graph(s).")


if __name__ == "__main__":
    main()
