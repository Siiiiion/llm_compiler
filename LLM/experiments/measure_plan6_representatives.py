#!/usr/bin/env python3
"""Check three training workload families against NumPy and measure real CUDA latency."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from LLM.core.io import atomic_json, file_sha256
from LLM.core.schedule_record_utils import digest
from LLM.core.runtime import now


SELECTED_WORKLOADS = {
    "matmul_dense": '["00a059b856ac30ac172b6252254479a6",[1,512],[400,512],[1,400],[1,400]]',
    "pooling": '["8c53ca2904398da2889aa7508082d7bb",[1,7,7,512],[1,1,1,512]]',
    "batch_matmul": '["97d6699a083ccfb8655170ee1eb94e98",[8,64,64],[8,64,64],[8,64,64]]',
}


def reference(family, inputs):
    if family == "matmul_dense":
        return inputs[0] @ inputs[1].T + inputs[2]
    if family == "batch_matmul":
        return np.matmul(inputs[0], inputs[1])
    if family == "pooling":
        return inputs[0].mean(axis=(1, 2), keepdims=True)
    raise ValueError(f"Unsupported reference family: {family}")


def run(args):
    import tvm
    from tvm import auto_scheduler
    from tvm.auto_scheduler.measure import _local_build_worker, recover_measure_input
    from tvm.contrib import tar
    from LLM.audit.audit_tvm_backend import compiler_fingerprint, register_tasks

    directory = Path(args.run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    source = Path(args.source_run).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    if file_sha256(source / "samples.jsonl") != manifest["samples_sha256"]:
        raise ValueError("Frozen source samples changed")
    if file_sha256(source / "results.jsonl") != summary["results_sha256"]:
        raise ValueError("Frozen source results changed")
    audit = json.loads((Path(manifest["inputs"]["audit_dir"]) / "manifest.json").read_text())
    register_tasks(audit["registry"]["path"])
    measured = []
    started = time.monotonic()
    samples = [json.loads(line) for line in (source / "samples.jsonl").open()]
    chosen = [
        s
        for s in samples
        if s["split"] == "train" and s["workload_key"] in SELECTED_WORKLOADS.values()
    ]
    if len(chosen) != 9:
        raise ValueError("Expected three representative states for each training workload")
    run_manifest = {
        "created_at": now(),
        "source_manifest_id": manifest["manifest_id"],
        "samples": [s["sample_id"] for s in chosen],
        "device": args.device,
        "compiler": compiler_fingerprint(),
        "reference": "independent_numpy_float32",
        "seed": 42,
        "rtol": 1e-3,
        "atol": 1e-4,
        "number": 10,
        "repeat": 5,
        "min_repeat_ms": 100,
        "warmup": 1,
        "code_sha256": file_sha256(__file__),
        "holdout_used": False,
    }
    atomic_json(directory / "manifest.json", run_manifest)
    device = tvm.cuda(args.device)
    try:
        for sample in chosen:
            family = next(
                name for name, key in SELECTED_WORKLOADS.items() if key == sample["workload_key"]
            )
            atomic_json(
                directory / "status.json",
                {
                    "state": "RUNNING",
                    "stage": "BUILD_CHECK_MEASURE",
                    "completed": len(measured),
                    "sample_id": sample["sample_id"],
                    "updated_at": now(),
                },
            )
            record = sample["record"]
            inp, _ = auto_scheduler.measure_record.load_record_from_string(json.dumps(record))
            inp = recover_measure_input(inp, rebuild_state=True)
            if inp.task.layout_rewrite_option != 0:
                raise ValueError(
                    "This independent reference pilot requires unchanged input layouts"
                )
            filename, tensors, error_no, message, build_seconds = _local_build_worker(
                inp.serialize(), tar.tar, 0
            )
            if error_no:
                raise RuntimeError(str(message))
            module = tvm.runtime.load_module(filename)
            arrays, host_inputs = [], []
            rng = np.random.default_rng(42)
            for tensor in tensors:
                shape = tuple(int(v) for v in tensor.shape)
                if isinstance(tensor.op, tvm.te.PlaceholderOp):
                    values = rng.uniform(-0.5, 0.5, shape).astype(tensor.dtype)
                    host_inputs.append(values)
                    arrays.append(tvm.nd.array(values, device))
                else:
                    arrays.append(tvm.nd.empty(shape, tensor.dtype, device))
            if len(arrays) != len(host_inputs) + 1:
                raise ValueError("Pilot expects exactly one output tensor")
            expected = reference(family, host_inputs)
            module(*arrays)
            device.sync()
            actual = arrays[-1].numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
            timing = module.time_evaluator(
                module.entry_name, device, number=10, repeat=5, min_repeat_ms=100
            )(*arrays)
            costs = [float(value) for value in timing.results]
            record_dir = directory / "measure_records"
            record_dir.mkdir(exist_ok=True)
            output_file = record_dir / sample["source_file"]
            auto_scheduler.save_records(
                str(output_file),
                [inp],
                [auto_scheduler.MeasureResult(costs, 0, "", build_seconds, time.time())],
            )
            report = {
                "sample_id": sample["sample_id"],
                "family": family,
                "workload_key": sample["workload_key"],
                "source_input_hash": sample["source_input_hash"],
                "source_row": sample["row_index"],
                "build_seconds": float(build_seconds),
                "runtime_correctness_checked": True,
                "latency_measured": True,
                "max_abs_error": float(np.max(np.abs(actual - expected))),
                "mean_latency_seconds": float(np.mean(costs)),
                "repeat_costs_seconds": costs,
                "same_prompt_key": digest(sample["row"]["text"].split(" PPT", 1)[0]),
                "measured_record": str(output_file),
            }
            measured.append(report)
            atomic_json(directory / "results.json", measured)
            print(
                f"[{now()}] {family} row={sample['row_index']} correct; {np.mean(costs) * 1e6:.3f} us",
                flush=True,
            )
        atomic_json(
            directory / "status.json",
            {
                "state": "SUCCEEDED",
                "stage": "COMPLETE",
                "exit_code": 0,
                "finished_at": now(),
                "elapsed_seconds": time.monotonic() - started,
                "samples": len(measured),
                "runtime_correctness_checked": True,
                "latency_measured": True,
                "coverage": "nine_selected_training_candidates_only",
            },
        )
    except BaseException as error:
        atomic_json(
            directory / "status.json",
            {
                "state": "FAILED",
                "exit_code": 1,
                "error_message": str(error),
                "finished_at": now(),
                "completed": len(measured),
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--device", type=int, default=0)
    run(parser.parse_args())
