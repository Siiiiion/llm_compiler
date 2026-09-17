#!/usr/bin/env python3
"""Create a fixed-shape causal GQA Relay graph and shared schedule sketches."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import tvm
from tvm import auto_scheduler, relay


ARTIFACT_FORMAT = "llm_compiler.relay"
ARTIFACT_VERSION = 1


def hardware_params_for_target(target: tvm.target.Target):
    if target.kind.name == "llvm":
        return auto_scheduler.HardwareParams(target=target)
    if target.kind.name == "cuda":
        return auto_scheduler.HardwareParams(
            num_cores=-1,
            vector_unit_bytes=16,
            cache_line_bytes=64,
            max_shared_memory_per_block=int(target.attrs["max_shared_memory_per_block"]),
            max_threads_per_block=int(target.attrs["max_threads_per_block"]),
            max_local_memory_per_block=12345678,
            max_vthread_extent=8,
            warp_size=32,
        )
    raise ValueError(f"Unsupported target kind: {target.kind.name}")


def build_attention_module(batch, seq_len, num_heads, num_kv_heads, head_dim, dtype):
    if num_heads % num_kv_heads != 0:
        raise ValueError("--num-heads must be divisible by --num-kv-heads")
    if dtype not in {"float16", "float32"}:
        raise ValueError("TVM 0.12 attention preparation supports float16 or float32")

    q = relay.var("q", shape=(batch, num_heads, seq_len, head_dim), dtype=dtype)
    k = relay.var("k", shape=(batch, num_kv_heads, seq_len, head_dim), dtype=dtype)
    v = relay.var("v", shape=(batch, num_kv_heads, seq_len, head_dim), dtype=dtype)
    groups = num_heads // num_kv_heads
    k_expanded = relay.repeat(k, repeats=groups, axis=1)
    v_expanded = relay.repeat(v, repeats=groups, axis=1)
    flat_shape = (batch * num_heads, seq_len, head_dim)
    q_flat = relay.reshape(q, flat_shape)
    k_flat = relay.reshape(k_expanded, flat_shape)
    v_flat = relay.reshape(v_expanded, flat_shape)

    scores = relay.nn.batch_matmul(
        q_flat,
        k_flat,
        transpose_a=False,
        transpose_b=True,
        out_dtype="float32",
    )
    scores = relay.multiply(scores, relay.const(1.0 / math.sqrt(head_dim), "float32"))
    mask = np.triu(
        np.full((1, seq_len, seq_len), np.finfo(np.float32).min, dtype="float32"),
        k=1,
    )
    probabilities = relay.nn.softmax(relay.add(scores, relay.const(mask)), axis=-1)
    probabilities = relay.cast(probabilities, dtype)
    output = relay.nn.batch_matmul(
        probabilities,
        v_flat,
        transpose_a=False,
        transpose_b=False,
        out_dtype="float32",
    )
    output = relay.cast(
        relay.reshape(output, (batch, num_heads, seq_len, head_dim)), dtype
    )
    mod = tvm.IRModule.from_expr(relay.Function([q, k, v], output))
    return relay.transform.InferType()(mod), {}


def sample_sketches(task, count, max_rounds):
    policy = auto_scheduler.SketchPolicy(
        task,
        params={
            "evolutionary_search_num_iters": 1,
            "evolutionary_search_population": max(64, min(count, 2560)),
            "max_innermost_split_factor": 1024,
        },
        verbose=0,
    )
    states = []
    seen = set()
    for _ in range(max_rounds):
        for state in policy.sample_initial_population():
            key = str(state)
            if key in seen:
                continue
            seen.add(key)
            states.append(state)
            if len(states) >= count:
                return states
    return states


def atomic_pickle(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file_obj:
        pickle.dump(value, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file_obj:
        json.dump(value, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    os.replace(temporary, path)


def truncate_records_at_ppt(raw_path: Path, output_path: Path):
    """Match make_dataset.for_gen_eval_sketch's prompt construction exactly."""
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    record_count = 0
    with raw_path.open("r", encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            steps = record["i"][1][1]
            ppt_index = None
            for index, step in enumerate(steps):
                if step[0] == "SP":
                    step[4] = [1 for _ in step[4]]
                if step[0] == "PPT":
                    ppt_index = index
                    break
            if ppt_index is None:
                raise RuntimeError("Initial schedule state has no PPT boundary")
            record["i"][1][1] = steps[: ppt_index + 1]
            json.dump(record, destination, separators=(",", ":"))
            destination.write("\n")
            record_count += 1
    os.replace(temporary, output_path)
    return record_count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare one fixed-shape causal GQA experiment for TLM schedule generation."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="nvidia/geforce-rtx-4090")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--sketches-per-task", type=int, default=48)
    parser.add_argument("--max-sketch-rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    positive = {
        "batch": args.batch,
        "seq_len": args.seq_len,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "sketches_per_task": args.sketches_per_task,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"All shape/count arguments must be positive: {invalid}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    network_info_dir = output_dir / "network_info"
    to_measure_dir = output_dir / "to_measure_programs"
    relay_path = output_dir / "attention.relay.pkl"
    task_info_path = network_info_dir / "attention.task.pkl"
    all_tasks_path = network_info_dir / "all_tasks.pkl"
    sketch_path = output_dir / "attention_sketch.json"
    inputs_path = output_dir / "inputs.npz"
    metadata_path = output_dir / "metadata.json"
    generated_paths = [relay_path, task_info_path, all_tasks_path, sketch_path, inputs_path]
    existing = [str(path) for path in generated_paths if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Outputs already exist; pass --force to replace them: {existing}")
    network_info_dir.mkdir(parents=True, exist_ok=True)
    to_measure_dir.mkdir(parents=True, exist_ok=True)

    target = tvm.target.Target(args.target)
    hardware_params = hardware_params_for_target(target)
    mod, params = build_attention_module(
        args.batch,
        args.seq_len,
        args.num_heads,
        args.num_kv_heads,
        args.head_dim,
        args.dtype,
    )
    tasks, task_weights = auto_scheduler.extract_tasks(
        mod["main"], params, target, hardware_params=hardware_params
    )
    if not tasks:
        raise RuntimeError("AutoScheduler did not extract any tasks from the attention graph")

    artifact = {
        "format": ARTIFACT_FORMAT,
        "version": ARTIFACT_VERSION,
        "mod_json": tvm.ir.save_json(mod),
        "params_bytes": bytes(relay.save_param_dict(params)),
        "inputs": [
            {
                "name": "q",
                "shape": [args.batch, args.num_heads, args.seq_len, args.head_dim],
                "dtype": args.dtype,
            },
            {
                "name": "k",
                "shape": [args.batch, args.num_kv_heads, args.seq_len, args.head_dim],
                "dtype": args.dtype,
            },
            {
                "name": "v",
                "shape": [args.batch, args.num_kv_heads, args.seq_len, args.head_dim],
                "dtype": args.dtype,
            },
        ],
    }
    atomic_pickle(relay_path, artifact)
    atomic_pickle(task_info_path, (tasks, task_weights))
    atomic_pickle(all_tasks_path, tasks)

    measure_inputs = []
    measure_results = []
    sketch_counts = []
    for index, task in enumerate(tasks):
        states = sample_sketches(task, args.sketches_per_task, args.max_sketch_rounds)
        if not states:
            raise RuntimeError(f"No initial sketches could be generated for task {index}")
        sketch_counts.append(len(states))
        for state in states:
            measure_inputs.append(auto_scheduler.MeasureInput(task, state))
            measure_results.append(
                auto_scheduler.MeasureResult([0.0], 0, "", 0, time.time())
            )
    raw_sketch = sketch_path.with_suffix(".raw.json")
    if raw_sketch.exists():
        raw_sketch.unlink()
    auto_scheduler.save_records(str(raw_sketch), measure_inputs, measure_results)
    try:
        truncated_count = truncate_records_at_ppt(raw_sketch, sketch_path)
    finally:
        if raw_sketch.exists():
            raw_sketch.unlink()
    if truncated_count != len(measure_inputs):
        raise RuntimeError(
            f"Expected {len(measure_inputs)} truncated sketches, got {truncated_count}"
        )

    rng = np.random.default_rng(args.seed)
    np_dtype = np.dtype(args.dtype)
    q_array = rng.normal(
        0.0, 0.5, (args.batch, args.num_heads, args.seq_len, args.head_dim)
    ).astype(np_dtype)
    k_array = rng.normal(
        0.0, 0.5, (args.batch, args.num_kv_heads, args.seq_len, args.head_dim)
    ).astype(np_dtype)
    v_array = rng.normal(
        0.0, 0.5, (args.batch, args.num_kv_heads, args.seq_len, args.head_dim)
    ).astype(np_dtype)
    temporary_inputs = inputs_path.with_suffix(".tmp.npz")
    np.savez(temporary_inputs, q=q_array, k=k_array, v=v_array)
    os.replace(temporary_inputs, inputs_path)

    metadata = {
        "format": "llm_compiler.attention_experiment",
        "version": 1,
        "target": args.target,
        "canonical_target": str(target),
        "shape": {
            "batch": args.batch,
            "seq_len": args.seq_len,
            "num_heads": args.num_heads,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
        },
        "dtype": args.dtype,
        "causal": True,
        "dropout_p": 0.0,
        "softmax_scale": 1.0 / math.sqrt(args.head_dim),
        "task_count": len(tasks),
        "task_weights": [int(value) for value in task_weights],
        "sketch_counts": sketch_counts,
        "paths": {
            "relay_artifact": str(relay_path),
            "network_info_dir": str(network_info_dir),
            "to_measure_program_dir": str(to_measure_dir),
            "sketch": str(sketch_path),
            "inputs": str(inputs_path),
        },
        "tasks": [
            {
                "index": index,
                "workload_key": str(task.workload_key),
                "description": str(task.desc),
                "flop_count": float(task.compute_dag.flop_ct),
            }
            for index, task in enumerate(tasks)
        ],
    }
    atomic_json(metadata_path, metadata)
    print(f"Prepared attention experiment: {output_dir}")
    print(f"Extracted tasks: {len(tasks)}; sketches: {sum(sketch_counts)}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
