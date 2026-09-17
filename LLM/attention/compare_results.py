#!/usr/bin/env python3
"""Merge TVM and FlashAttention reports into one correctness/performance table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from metrics import atomic_json, compare_arrays


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare generated TVM schedules to FlashAttention."
    )
    parser.add_argument("--tvm-report", required=True)
    parser.add_argument("--flash-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pretrain-label", default="pretrain")
    parser.add_argument("--lapo-label", default="lapo")
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
    return parser.parse_args()


def load_json(path):
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_output(path):
    with np.load(path) as arrays:
        return arrays["output"]


def main():
    args = parse_args()
    tvm_report = load_json(args.tvm_report)
    flash_report = load_json(args.flash_report)
    flash_backends = flash_report.get("backends", {})
    if "flash_attention" not in flash_backends:
        raise ValueError("Flash report has no flash_attention result")
    reference_name = "torch_eager" if "torch_eager" in flash_backends else "flash_attention"
    reference = load_output(flash_backends[reference_name]["output_path"])
    flash_ms = flash_backends["flash_attention"]["timing"]["median_ms"]

    rows = []
    correctness = {}
    for label, variant in tvm_report.get("variants", {}).items():
        median_ms = variant["timing"]["median_ms"]
        rows.append(
            {
                "backend": label,
                "median_ms": median_ms,
                "p95_ms": variant["timing"].get("p95_ms"),
                "speedup_vs_flash": flash_ms / median_ms,
            }
        )
        correctness[label] = compare_arrays(
            reference,
            load_output(variant["output_path"]),
            rtol=args.rtol,
            atol=args.atol,
        )
    for label, backend in flash_backends.items():
        median_ms = backend["timing"]["median_ms"]
        rows.append(
            {
                "backend": label,
                "median_ms": median_ms,
                "p95_ms": backend["timing"].get("p95_ms"),
                "speedup_vs_flash": flash_ms / median_ms,
            }
        )
        correctness[label] = compare_arrays(
            reference,
            load_output(backend["output_path"]),
            rtol=args.rtol,
            atol=args.atol,
        )

    variants = tvm_report.get("variants", {})
    summary = {
        "format": "llm_compiler.attention_comparison",
        "version": 1,
        "reference_backend": reference_name,
        "rows": rows,
        "correctness": correctness,
    }
    if args.pretrain_label in variants and args.lapo_label in variants:
        pretrain_ms = variants[args.pretrain_label]["timing"]["median_ms"]
        lapo_ms = variants[args.lapo_label]["timing"]["median_ms"]
        summary["lapo_speedup_vs_pretrain"] = pretrain_ms / lapo_ms
        summary["lapo_speedup_vs_flash"] = flash_ms / lapo_ms
        summary["lapo_beats_pretrain"] = bool(lapo_ms < pretrain_ms)
        summary["lapo_beats_flash"] = bool(lapo_ms < flash_ms)

    atomic_json(args.output, summary)
    print("| Backend | Median ms | P95 ms | Speedup vs Flash | Correct |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        p95 = row["p95_ms"]
        p95_text = f"{p95:.6f}" if p95 is not None else "n/a"
        passed = correctness[row["backend"]]["passed"]
        print(
            f"| {row['backend']} | {row['median_ms']:.6f} | {p95_text} | "
            f"{row['speedup_vs_flash']:.4f}x | {passed} |"
        )
    if "lapo_speedup_vs_pretrain" in summary:
        print(f"LAPO vs Pretrain: {summary['lapo_speedup_vs_pretrain']:.4f}x")
        print(f"LAPO vs FlashAttention: {summary['lapo_speedup_vs_flash']:.4f}x")
    print(f"Report: {Path(args.output).expanduser().resolve()}")


if __name__ == "__main__":
    main()
