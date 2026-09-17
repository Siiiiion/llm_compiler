#!/usr/bin/env python3
"""Benchmark eager and FlashAttention on the exact Q/K/V saved by task preparation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
import torch

from metrics import atomic_json, compare_arrays, timing_statistics


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark FlashAttention for one experiment.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--number", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--min-repeat-ms", type=float, default=500.0)
    parser.add_argument("--skip-eager", action="store_true")
    parser.add_argument("--skip-flash", action="store_true")
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def cuda_timing(function, warmup, number, repeat, min_repeat_ms):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    effective_number = max(number, 1)
    if min_repeat_ms > 0:
        while True:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(effective_number):
                function()
            end.record()
            end.synchronize()
            elapsed_ms = start.elapsed_time(end)
            if elapsed_ms >= min_repeat_ms:
                break
            scale = max(2, math.ceil(min_repeat_ms / max(elapsed_ms, 1e-6)))
            effective_number *= scale

    samples = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(effective_number):
            function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / effective_number)
    timing = timing_statistics(samples)
    timing["number_per_repeat"] = effective_number
    return timing


def eager_attention(q, k, v, scale, causal_mask):
    groups = q.shape[1] // k.shape[1]
    k_expanded = torch.repeat_interleave(k, groups, dim=1)
    v_expanded = torch.repeat_interleave(v, groups, dim=1)
    scores = torch.matmul(q.float(), k_expanded.float().transpose(-1, -2)) * scale
    probabilities = torch.softmax(scores.masked_fill(causal_mask, -torch.inf), dim=-1)
    return torch.matmul(probabilities.to(q.dtype), v_expanded).to(q.dtype)


def main():
    args = parse_args()
    if args.skip_eager and args.skip_flash:
        raise ValueError("At least one backend must be enabled")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the FlashAttention benchmark")
    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    with (experiment_dir / "metadata.json").open("r", encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)
    arrays = np.load(metadata["paths"]["inputs"])
    device = torch.device(args.device)
    q = torch.from_numpy(arrays["q"]).to(device)
    k = torch.from_numpy(arrays["k"]).to(device)
    v = torch.from_numpy(arrays["v"]).to(device)
    scale = float(metadata.get("softmax_scale", 1.0 / math.sqrt(q.shape[-1])))
    causal_mask = torch.ones(
        (q.shape[2], q.shape[2]), dtype=torch.bool, device=device
    ).triu(1)
    output_dir = experiment_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    report = {
        "format": "llm_compiler.attention_flash_benchmark",
        "version": 1,
        "experiment_dir": str(experiment_dir),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "benchmark_config": {
            "warmup": args.warmup,
            "number": args.number,
            "repeat": args.repeat,
            "min_repeat_ms": args.min_repeat_ms,
        },
        "backends": {},
    }

    eager_output = None
    with torch.inference_mode():
        if not args.skip_eager:
            eager_function = lambda: eager_attention(q, k, v, scale, causal_mask)
            allocated_before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            timing = cuda_timing(
                eager_function,
                args.warmup,
                args.number,
                args.repeat,
                args.min_repeat_ms,
            )
            eager_output = eager_function()
            output_path = output_dir / "torch_eager.npz"
            np.savez(output_path, output=eager_output.cpu().numpy())
            report["backends"]["torch_eager"] = {
                "timing": timing,
                "peak_memory_bytes": int(
                    torch.cuda.max_memory_allocated(device) - allocated_before
                ),
                "output_path": str(output_path),
            }
            print(f"torch_eager: median={timing['median_ms']:.6f} ms", flush=True)

        if not args.skip_flash:
            try:
                from flash_attn import flash_attn_func
            except ImportError as error:
                raise RuntimeError(
                    "flash-attn is not installed in this Python environment"
                ) from error
            if q.dtype not in {torch.float16, torch.bfloat16}:
                raise ValueError(f"FlashAttention requires FP16/BF16 input, got {q.dtype}")
            q_flash = q.permute(0, 2, 1, 3).contiguous()
            k_flash = k.permute(0, 2, 1, 3).contiguous()
            v_flash = v.permute(0, 2, 1, 3).contiguous()
            flash_function = lambda: flash_attn_func(
                q_flash,
                k_flash,
                v_flash,
                dropout_p=0.0,
                softmax_scale=scale,
                causal=True,
            )
            allocated_before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            timing = cuda_timing(
                flash_function,
                args.warmup,
                args.number,
                args.repeat,
                args.min_repeat_ms,
            )
            flash_output = flash_function().permute(0, 2, 1, 3).contiguous()
            output_path = output_dir / "flash_attention.npz"
            np.savez(output_path, output=flash_output.cpu().numpy())
            flash_report = {
                "timing": timing,
                "peak_memory_bytes": int(
                    torch.cuda.max_memory_allocated(device) - allocated_before
                ),
                "output_path": str(output_path),
                "flash_attn_version": importlib.metadata.version("flash-attn"),
            }
            if eager_output is not None:
                flash_report["correctness_vs_eager"] = compare_arrays(
                    eager_output.cpu().numpy(), flash_output.cpu().numpy()
                )
            report["backends"]["flash_attention"] = flash_report
            print(f"flash_attention: median={timing['median_ms']:.6f} ms", flush=True)

    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else experiment_dir / "flash_report.json"
    )
    atomic_json(report_path, report)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
