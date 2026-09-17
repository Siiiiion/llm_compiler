#!/usr/bin/env python3
"""Build and benchmark full Attention Relay graphs from measured schedule histories."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import tvm
from tvm import auto_scheduler


SCRIPT_DIR = Path(__file__).resolve().parent
LLM_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(LLM_DIR))

import benchmark_relay as relay_benchmark  # noqa: E402
from metrics import atomic_json  # noqa: E402


def labeled_path(raw):
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Variant label cannot be empty")
    return label, str(Path(path).expanduser().resolve())


def safe_label(label):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_.-")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark TVM Default and generated Attention schedules end to end."
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument(
        "--variant",
        action="append",
        type=labeled_path,
        default=[],
        help="Measured history as LABEL=PATH; repeat for pretrain and lapo.",
    )
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--number", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--min-repeat-ms", type=int, default=500)
    parser.add_argument("--report", default=None)
    parser.add_argument(
        "--reuse-libraries",
        action="store_true",
        help="Reuse existing .so files. By default every variant is rebuilt from its history.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.variant and not args.include_baseline:
        raise ValueError("Pass at least one --variant LABEL=PATH or --include-baseline")
    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    with (experiment_dir / "metadata.json").open("r", encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)
    relay_path = metadata["paths"]["relay_artifact"]
    inputs_path = metadata["paths"]["inputs"]
    mod, params, specs, artifact_kind = relay_benchmark.load_relay_artifact(relay_path)
    inputs = relay_benchmark.generate_inputs(specs, inputs_path, 0, 0, 1)
    target = tvm.target.Target(metadata.get("canonical_target", metadata["target"]))
    device = tvm.device(target.kind.name, args.device_id)
    if not device.exist:
        raise RuntimeError(f"TVM device does not exist: {target.kind.name}({args.device_id})")

    tasks, task_weights = auto_scheduler.extract_tasks(
        mod["main"],
        params,
        target,
        hardware_params=relay_benchmark.hardware_params_for_target(target),
    )
    report = {
        "format": "llm_compiler.attention_tvm_benchmark",
        "version": 1,
        "experiment_dir": str(experiment_dir),
        "target": str(target),
        "artifact_kind": artifact_kind,
        "inputs": relay_benchmark.input_report(inputs, inputs_path),
        "benchmark_config": {
            "warmup": args.warmup,
            "number": args.number,
            "repeat": args.repeat,
            "min_repeat_ms": args.min_repeat_ms,
        },
        "variants": {},
    }
    library_dir = experiment_dir / "libraries"
    output_dir = experiment_dir / "outputs"
    library_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    variants = list(args.variant)
    if args.include_baseline:
        variants.insert(0, ("tvm_default", None))
    for label, history_path in variants:
        normalized_label = safe_label(label)
        if not normalized_label:
            raise ValueError(f"Invalid variant label: {label!r}")
        library_path = library_dir / f"{normalized_label}.so"
        if not args.reuse_libraries and library_path.exists():
            library_path.unlink()

        filtered_history = None
        if history_path is None:
            mode = "baseline"
            history_for_build = []
            coverage = None
        else:
            if not os.path.isfile(history_path):
                raise FileNotFoundError(f"Measured history not found: {history_path}")
            mode = "optimized"
            coverage = relay_benchmark.history_coverage(
                tasks, task_weights, [history_path], target=target
            )
            if not coverage["complete"]:
                missing = [item["index"] for item in coverage["tasks"] if not item["covered"]]
                raise RuntimeError(
                    f"{label} history is incomplete; missing task indices {missing}: {history_path}"
                )
            filtered_history, build_record_count = relay_benchmark.filter_history_for_build(
                [history_path], [task.workload_key for task in tasks], target
            )
            coverage["build_record_count"] = build_record_count
            history_for_build = [filtered_history]

        try:
            library, compile_seconds, library_source = relay_benchmark.build_or_load(
                mode,
                mod,
                params,
                target,
                history_for_build,
                str(library_path),
            )
            outputs, timing = relay_benchmark.run_library(
                library,
                device,
                inputs,
                args.warmup,
                args.number,
                args.repeat,
                args.min_repeat_ms,
            )
            timing["p95_ms"] = float(np.percentile(timing["samples_ms"], 95))
            if len(outputs) != 1:
                raise RuntimeError(f"Expected one Attention output, got {len(outputs)}")
            output_path = output_dir / f"{normalized_label}.npz"
            np.savez(output_path, output=outputs[0])
            report["variants"][label] = {
                "history": history_path,
                "history_coverage": coverage,
                "compile_seconds": compile_seconds,
                "library_source": library_source,
                "library_path": str(library_path),
                "output_path": str(output_path),
                "output_shape": list(outputs[0].shape),
                "output_dtype": str(outputs[0].dtype),
                "timing": timing,
            }
            print(f"{label}: median={timing['median_ms']:.6f} ms", flush=True)
            del outputs, library
            gc.collect()
        finally:
            if filtered_history and os.path.exists(filtered_history):
                os.unlink(filtered_history)

    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else experiment_dir / "tvm_report.json"
    )
    atomic_json(report_path, report)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
