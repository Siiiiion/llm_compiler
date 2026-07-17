#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export a HuggingFace causal LM such as Qwen3 to a fixed-shape ONNX graph."""

import argparse
import os

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
    if args.logits_to_keep < 0:
        raise ValueError("--logits-to-keep must be non-negative")

    try:
        import onnx  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError as err:
        raise RuntimeError(
            "The onnx package is required for torch.onnx.export. "
            "Install it in an isolated export env if you do not want to change the TVM env."
        ) from err

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading model from: {args.model_dir}")
    print(f"Using device={args.device}, dtype={args.dtype}")
    model = load_model(args)
    wrapper = CausalLMLogitsWrapper(model, logits_to_keep=args.logits_to_keep).eval().to(
        args.device
    )

    input_ids = torch.ones((args.batch_size, args.seq_len), dtype=torch.long, device=args.device)
    attention_mask = torch.ones(
        (args.batch_size, args.seq_len), dtype=torch.long, device=args.device
    )

    dynamic_axes = None
    if args.dynamic_axes:
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        }

    print(f"Exporting fixed shape: batch={args.batch_size}, seq_len={args.seq_len}")
    print(f"LM head logits_to_keep={args.logits_to_keep}")
    print(f"Writing ONNX to: {args.output}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask),
            args.output,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
        )

    print("Export finished.")


if __name__ == "__main__":
    main()
