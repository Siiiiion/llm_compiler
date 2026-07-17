#!/usr/bin/env python3
"""Benchmark a fixed-shape Relay or ONNX model with and without tuning history."""

import argparse
import gc
import hashlib
import json
import os
import pickle
import statistics
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tvm
from tvm import auto_scheduler, relay
from tvm.contrib import graph_executor


RELAY_ARTIFACT_FORMAT = "llm_compiler.relay"
RELAY_ARTIFACT_VERSION = 1
MAX_VALID_LATENCY_SECONDS = 1e9


@dataclass(frozen=True)
class InputSpec:
    name: str
    shape: Tuple[int, ...]
    dtype: str


def _json_object(value: Optional[str], option: str) -> Dict:
    if value is None:
        return {}
    try:
        result = json.loads(value)
    except json.JSONDecodeError as err:
        raise ValueError(f"{option} must be a JSON object: {err}") from err
    if not isinstance(result, dict):
        raise ValueError(f"{option} must be a JSON object")
    return result


def normalize_input_specs(raw_inputs) -> List[InputSpec]:
    """Normalize legacy tuples and versioned input dictionaries."""
    if isinstance(raw_inputs, dict):
        raw_inputs = [
            {"name": name, **value} if isinstance(value, dict) else (name, value, "float32")
            for name, value in raw_inputs.items()
        ]
    elif (
        isinstance(raw_inputs, (tuple, list))
        and len(raw_inputs) == 3
        and isinstance(raw_inputs[0], str)
    ):
        raw_inputs = [raw_inputs]

    if not isinstance(raw_inputs, (tuple, list)) or not raw_inputs:
        raise ValueError("Relay artifact has no usable input metadata")

    specs = []
    for item in raw_inputs:
        if isinstance(item, dict):
            try:
                name, shape, dtype = item["name"], item["shape"], item["dtype"]
            except KeyError as err:
                raise ValueError(f"Invalid input metadata, missing {err.args[0]}: {item}") from err
        elif isinstance(item, (tuple, list)) and len(item) == 3:
            name, shape, dtype = item
        else:
            raise ValueError(f"Invalid input metadata: {item!r}")

        if not isinstance(name, str) or not name:
            raise ValueError(f"Invalid input name: {name!r}")
        try:
            normalized_shape = tuple(int(dim) for dim in shape)
        except (TypeError, ValueError) as err:
            raise ValueError(f"Invalid shape for input {name}: {shape!r}") from err
        if not normalized_shape or any(dim <= 0 for dim in normalized_shape):
            raise ValueError(f"Input {name} must have a fixed positive shape, got {shape!r}")
        try:
            normalized_dtype = np.dtype(dtype).name
        except TypeError as err:
            raise ValueError(f"Invalid dtype for input {name}: {dtype!r}") from err
        specs.append(InputSpec(name, normalized_shape, normalized_dtype))

    names = [item.name for item in specs]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate input names: {names}")
    return specs


def load_relay_artifact(path: str):
    """Load the versioned artifact or an old empty-parameter tuple artifact."""
    with open(path, "rb") as file:
        artifact = pickle.load(file)

    if isinstance(artifact, dict):
        if artifact.get("format") != RELAY_ARTIFACT_FORMAT:
            raise ValueError(f"Unknown Relay artifact format in {path}: {artifact.get('format')!r}")
        if artifact.get("version") != RELAY_ARTIFACT_VERSION:
            raise ValueError(
                f"Unsupported Relay artifact version in {path}: {artifact.get('version')}"
            )
        try:
            mod_json = artifact["mod_json"]
            params_bytes = artifact["params_bytes"]
            raw_inputs = artifact["inputs"]
        except KeyError as err:
            raise ValueError(f"Relay artifact {path} is missing {err.args[0]}") from err
        if not isinstance(params_bytes, (bytes, bytearray)):
            raise ValueError(f"Relay artifact {path} does not contain serialized parameter bytes")
        params = relay.load_param_dict(bytearray(params_bytes))
        artifact_kind = "versioned"
    elif isinstance(artifact, tuple) and len(artifact) == 3:
        mod_json, param_length, raw_inputs = artifact
        empty_param_length = len(relay.save_param_dict({}))
        if param_length not in (0, empty_param_length):
            raise ValueError(
                f"Legacy Relay artifact {path} only stored the parameter length "
                f"({param_length} bytes), not the parameter data. Re-dump the source model with "
                "the updated gen/dump_network_info.py --overwrite-relay before benchmarking."
            )
        params = {}
        artifact_kind = "legacy-empty-params"
    else:
        raise ValueError(f"Unsupported Relay artifact object in {path}: {type(artifact).__name__}")

    if not isinstance(mod_json, str):
        raise ValueError(f"Relay artifact {path} has a non-string mod_json")
    specs = normalize_input_specs(raw_inputs)
    mod = tvm.ir.load_json(mod_json)
    return mod, params, specs, artifact_kind


def load_onnx_model(path: str, shape_json: str, dtype_json: Optional[str], default_dtype: str):
    import onnx

    shapes = _json_object(shape_json, "--input-shapes")
    if not shapes:
        raise ValueError("--input-shapes must contain at least one fixed-shape input")
    dtypes = _json_object(dtype_json, "--input-dtypes")
    unknown_dtypes = sorted(set(dtypes) - set(shapes))
    if unknown_dtypes:
        raise ValueError(
            f"--input-dtypes contains names absent from --input-shapes: {unknown_dtypes}"
        )
    dtype_map = {name: dtypes.get(name, default_dtype) for name in shapes}
    specs = normalize_input_specs(
        [(name, shape, dtype_map[name]) for name, shape in shapes.items()]
    )
    shape_dict = {item.name: list(item.shape) for item in specs}
    dtype_dict = {item.name: item.dtype for item in specs}
    model = onnx.load(path, load_external_data=True)
    mod, params = relay.frontend.from_onnx(
        model,
        shape_dict,
        dtype=dtype_dict,
        freeze_params=False,
    )
    return mod, params, specs, "onnx"


def hardware_params_for_target(target: tvm.target.Target):
    if target.kind.name == "llvm":
        return auto_scheduler.HardwareParams(target=target)
    if target.kind.name == "cuda":
        required_attrs = {"max_shared_memory_per_block", "max_threads_per_block"}
        if not required_attrs.issubset(set(target.attrs.keys())):
            # A plain "cuda -arch=..." target does not carry device limits. TVM can
            # query them from the selected CUDA device when one is available.
            return auto_scheduler.HardwareParams(target=target)
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
    raise NotImplementedError(f"Unsupported target kind for task extraction: {target.kind.name}")


def _valid_latency(record: Dict) -> Optional[float]:
    result = record.get("r")
    if not isinstance(result, list) or len(result) < 2 or result[1] != 0:
        return None
    valid_costs = []
    for cost in result[0] or []:
        try:
            cost = float(cost)
        except (TypeError, ValueError):
            continue
        if 0 < cost < MAX_VALID_LATENCY_SECONDS:
            valid_costs.append(cost)
    return min(valid_costs) if valid_costs else None


def _canonical_target(target) -> str:
    return str(target if isinstance(target, tvm.target.Target) else tvm.target.Target(target))


def load_history_stats(paths: Sequence[str], target=None) -> Dict[str, Dict]:
    stats = {}
    expected_target = _canonical_target(target) if target is not None else None
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"History file does not exist: {path}")
        with open(path, "r", encoding="utf-8") as file:
            for raw in file:
                if not raw.strip():
                    continue
                try:
                    record = json.loads(raw)
                    workload_key = record["i"][0][0]
                    record_target = record["i"][0][1]
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue
                if expected_target is not None:
                    try:
                        if _canonical_target(record_target) != expected_target:
                            continue
                    except (ValueError, tvm.error.TVMError):
                        continue
                item = stats.setdefault(workload_key, {"total": 0, "valid": 0, "best": None})
                item["total"] += 1
                latency = _valid_latency(record)
                if latency is not None:
                    item["valid"] += 1
                    item["best"] = latency if item["best"] is None else min(item["best"], latency)
    return stats


def history_coverage(tasks, task_weights, history_paths: Sequence[str], target=None) -> Dict:
    stats = load_history_stats(history_paths, target=target)
    details = []
    for index, (task, weight) in enumerate(zip(tasks, task_weights)):
        item = stats.get(task.workload_key, {"total": 0, "valid": 0, "best": None})
        details.append(
            {
                "index": index,
                "weight": int(weight),
                "workload_key": task.workload_key,
                "records": item["total"],
                "valid_records": item["valid"],
                "best_latency_ms": None if item["best"] is None else item["best"] * 1000,
                "covered": item["valid"] > 0,
            }
        )
    covered = [item for item in details if item["covered"]]
    return {
        "task_count": len(details),
        "covered_task_count": len(covered),
        "task_weight": sum(item["weight"] for item in details),
        "covered_task_weight": sum(item["weight"] for item in covered),
        "complete": len(covered) == len(details),
        "tasks": details,
    }


def filter_history_for_build(
    paths: Sequence[str], workload_keys: Sequence[str], target: tvm.target.Target
) -> Tuple[str, int]:
    """Create a valid, model- and target-specific log for ApplyHistoryBest."""
    allowed_keys = set(workload_keys)
    expected_target = _canonical_target(target)
    descriptor, filtered_path = tempfile.mkstemp(prefix="relay-benchmark-history-", suffix=".json")
    kept = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            for path in paths:
                with open(path, "r", encoding="utf-8") as source:
                    for raw in source:
                        if not raw.strip():
                            continue
                        try:
                            record = json.loads(raw)
                            workload_key = record["i"][0][0]
                            record_target = _canonical_target(record["i"][0][1])
                        except (
                            json.JSONDecodeError,
                            KeyError,
                            IndexError,
                            TypeError,
                            ValueError,
                            tvm.error.TVMError,
                        ):
                            continue
                        if (
                            workload_key in allowed_keys
                            and record_target == expected_target
                            and _valid_latency(record) is not None
                        ):
                            output.write(raw if raw.endswith("\n") else f"{raw}\n")
                            kept += 1
    except Exception:
        if os.path.exists(filtered_path):
            os.unlink(filtered_path)
        raise
    return filtered_path, kept


def generate_inputs(
    specs: Sequence[InputSpec],
    npz_path: Optional[str],
    seed: int,
    integer_low: int,
    integer_high: int,
) -> Dict[str, np.ndarray]:
    if integer_high <= integer_low:
        raise ValueError("--integer-high must be greater than --integer-low")
    if npz_path:
        with np.load(npz_path, allow_pickle=False) as archive:
            missing = [item.name for item in specs if item.name not in archive]
            extra = sorted(set(archive.files) - {item.name for item in specs})
            if missing or extra:
                raise ValueError(
                    f"NPZ input names do not match model inputs; missing={missing}, extra={extra}"
                )
            result = {item.name: np.array(archive[item.name], copy=True) for item in specs}
    else:
        rng = np.random.default_rng(seed)
        result = {}
        for item in specs:
            dtype = np.dtype(item.dtype)
            if "mask" in item.name.lower():
                value = np.ones(item.shape, dtype=dtype)
            elif np.issubdtype(dtype, np.bool_):
                value = rng.integers(0, 2, size=item.shape).astype(dtype)
            elif np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                low = max(integer_low, int(info.min))
                high = min(integer_high, int(info.max) + 1)
                if high <= low:
                    raise ValueError(
                        f"Integer range [{integer_low}, {integer_high}) is invalid for {dtype}"
                    )
                value = rng.integers(low, high, size=item.shape, dtype=dtype)
            elif np.issubdtype(dtype, np.floating):
                value = rng.uniform(-1.0, 1.0, size=item.shape).astype(dtype)
            else:
                raise ValueError(f"Cannot generate input {item.name} with dtype {dtype}")
            result[item.name] = value

    for item in specs:
        value = result[item.name]
        if tuple(value.shape) != item.shape:
            raise ValueError(
                f"Input {item.name} shape mismatch: expected {item.shape}, got {value.shape}"
            )
        if value.dtype != np.dtype(item.dtype):
            raise ValueError(
                f"Input {item.name} dtype mismatch: expected {item.dtype}, got {value.dtype}"
            )
    return result


def input_report(inputs: Dict[str, np.ndarray], source: str) -> List[Dict]:
    result = []
    for name, value in inputs.items():
        contiguous = np.ascontiguousarray(value)
        result.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "source": source,
                "sha256": hashlib.sha256(contiguous.view(np.uint8)).hexdigest(),
            }
        )
    return result


def build_or_load(
    mode: str,
    mod,
    params,
    target,
    history_paths: Sequence[str],
    library_path: Optional[str],
):
    started = time.perf_counter()
    if library_path and os.path.isfile(library_path):
        return tvm.runtime.load_module(library_path), 0.0, "cache"

    if mode == "baseline":
        with tvm.transform.PassContext(opt_level=3):
            library = relay.build(mod, target=target, params=params)
    else:
        with auto_scheduler.ApplyHistoryBest(list(history_paths)):
            with tvm.transform.PassContext(
                opt_level=3,
                config={"relay.backend.use_auto_scheduler": True},
            ):
                library = relay.build(mod, target=target, params=params)

    compile_seconds = time.perf_counter() - started
    if library_path:
        parent = os.path.dirname(os.path.abspath(library_path))
        os.makedirs(parent, exist_ok=True)
        library.export_library(library_path)
    return library, compile_seconds, "built"


def timing_statistics(seconds: Iterable[float]) -> Dict:
    values_ms = [float(value) * 1000 for value in seconds]
    return {
        "samples_ms": values_ms,
        "mean_ms": statistics.fmean(values_ms),
        "median_ms": statistics.median(values_ms),
        "min_ms": min(values_ms),
        "max_ms": max(values_ms),
        "stdev_ms": statistics.pstdev(values_ms),
    }


def run_library(library, device, inputs, warmup, number, repeat, min_repeat_ms):
    module = graph_executor.GraphModule(library["default"](device))
    for name, value in inputs.items():
        module.set_input(name, tvm.nd.array(value, device=device))
    module.run()
    device.sync()
    outputs = [module.get_output(index).numpy() for index in range(module.get_num_outputs())]
    for _ in range(warmup):
        module.run()
    device.sync()
    benchmark = module.benchmark(
        device,
        func_name="run",
        number=number,
        repeat=repeat,
        min_repeat_ms=min_repeat_ms,
        end_to_end=False,
    )
    return outputs, timing_statistics(benchmark.results)


def compare_outputs(reference, candidate, rtol: float, atol: float) -> Dict:
    if len(reference) != len(candidate):
        return {"passed": False, "reason": "output_count", "outputs": []}
    details = []
    passed = True
    for index, (expected, actual) in enumerate(zip(reference, candidate)):
        shape_matches = expected.shape == actual.shape
        dtype_matches = expected.dtype == actual.dtype
        close = (
            shape_matches
            and dtype_matches
            and np.allclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True)
        )
        if shape_matches:
            expected64 = expected.astype("float64")
            actual64 = actual.astype("float64")
            difference = np.abs(expected64 - actual64)
            max_abs = float(np.nanmax(difference)) if difference.size else 0.0
            mean_abs = float(np.nanmean(difference)) if difference.size else 0.0
            rmse = (
                float(np.sqrt(np.nanmean(np.square(expected64 - actual64))))
                if difference.size
                else 0.0
            )
            abs_percentiles = {
                str(percentile): float(np.nanpercentile(difference, percentile))
                for percentile in (50, 90, 95, 99)
            }
            denominator = np.maximum(np.abs(expected64), max(atol, 1e-30))
            max_rel = float(np.nanmax(difference / denominator)) if difference.size else 0.0
            expected_flat = expected64.reshape(-1)
            actual_flat = actual64.reshape(-1)
            norm_product = np.linalg.norm(expected_flat) * np.linalg.norm(actual_flat)
            cosine_similarity = (
                float(np.dot(expected_flat, actual_flat) / norm_product)
                if norm_product
                else None
            )
            if expected.ndim and expected.shape[-1] > 1:
                expected_argmax = np.argmax(expected, axis=-1)
                actual_argmax = np.argmax(actual, axis=-1)
                argmax_total = int(expected_argmax.size)
                argmax_matches = int(np.count_nonzero(expected_argmax == actual_argmax))
                argmax_match_rate = argmax_matches / argmax_total if argmax_total else 1.0
            else:
                argmax_total = None
                argmax_matches = None
                argmax_match_rate = None
        else:
            max_abs = None
            mean_abs = None
            rmse = None
            abs_percentiles = None
            max_rel = None
            cosine_similarity = None
            argmax_total = None
            argmax_matches = None
            argmax_match_rate = None
        details.append(
            {
                "index": index,
                "expected_shape": list(expected.shape),
                "actual_shape": list(actual.shape),
                "expected_dtype": str(expected.dtype),
                "actual_dtype": str(actual.dtype),
                "shape_matches": shape_matches,
                "dtype_matches": dtype_matches,
                "max_abs_error": max_abs,
                "mean_abs_error": mean_abs,
                "rmse": rmse,
                "abs_error_percentiles": abs_percentiles,
                "max_rel_error": max_rel,
                "cosine_similarity": cosine_similarity,
                "argmax_total": argmax_total,
                "argmax_matches": argmax_matches,
                "argmax_match_rate": argmax_match_rate,
                "passed": bool(close),
            }
        )
        passed = passed and bool(close)
    return {"passed": passed, "rtol": rtol, "atol": atol, "outputs": details}


def write_report(path: str, report: Dict) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare whole-model TVM Default and AutoScheduler end-to-end latency."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--relay-pickle", help="Relay artifact produced by dump_network_info.py")
    source.add_argument("--onnx-model", help="Fixed-shape ONNX model, including external data")
    parser.add_argument("--input-shapes", help='ONNX shape JSON, e.g. {"input":[1,3,224,224]}')
    parser.add_argument("--input-dtypes", help='ONNX dtype JSON, e.g. {"input_ids":"int64"}')
    parser.add_argument("--input-dtype", default="float32", help="Default ONNX input dtype")
    parser.add_argument("--input-npz", help="NPZ containing one array for every model input")
    parser.add_argument(
        "--history", action="append", default=[], help="AutoScheduler log; repeatable"
    )
    parser.add_argument("--target", required=True, help="TVM target string")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--mode", choices=["both", "baseline", "optimized"], default="both")
    parser.add_argument("--allow-incomplete-history", action="store_true")
    parser.add_argument("--baseline-lib", help="Load existing library, or build and export here")
    parser.add_argument("--optimized-lib", help="Load existing library, or build and export here")
    parser.add_argument("--report", default="benchmark_report.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--integer-low", type=int, default=0)
    parser.add_argument("--integer-high", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--number", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--min-repeat-ms", type=int, default=500)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.onnx_model and not args.input_shapes:
        raise ValueError("--input-shapes is required with --onnx-model")
    if args.mode in ("both", "optimized") and not args.history:
        raise ValueError("At least one --history file is required for optimized benchmarking")
    if (
        args.mode == "both"
        and args.baseline_lib
        and args.optimized_lib
        and os.path.abspath(args.baseline_lib) == os.path.abspath(args.optimized_lib)
    ):
        raise ValueError("--baseline-lib and --optimized-lib must use different paths")
    for name in ("warmup", "number", "repeat"):
        if getattr(args, name) < (0 if name == "warmup" else 1):
            raise ValueError(f"--{name} has an invalid value: {getattr(args, name)}")

    target = tvm.target.Target(args.target)
    if args.relay_pickle:
        mod, params, specs, source_kind = load_relay_artifact(args.relay_pickle)
        model_path = args.relay_pickle
    else:
        mod, params, specs, source_kind = load_onnx_model(
            args.onnx_model, args.input_shapes, args.input_dtypes, args.input_dtype
        )
        model_path = args.onnx_model

    inputs = generate_inputs(
        specs, args.input_npz, args.seed, args.integer_low, args.integer_high
    )
    report = {
        "status": "running",
        "model": {"path": os.path.abspath(model_path), "source": source_kind},
        "target": str(target),
        "device_id": args.device_id,
        "mode": args.mode,
        "inputs": input_report(inputs, args.input_npz or f"generated:seed={args.seed}"),
        "benchmark_config": {
            "warmup": args.warmup,
            "number": args.number,
            "repeat": args.repeat,
            "min_repeat_ms": args.min_repeat_ms,
        },
    }

    if args.mode in ("both", "optimized"):
        print("Extracting AutoScheduler tasks from the exact Relay module...", flush=True)
        extract_started = time.perf_counter()
        tasks, task_weights = auto_scheduler.extract_tasks(
            mod["main"],
            params,
            target,
            hardware_params=hardware_params_for_target(target),
        )
        coverage = history_coverage(tasks, task_weights, args.history, target=target)
        coverage["extraction_seconds"] = time.perf_counter() - extract_started
        coverage["history_files"] = [os.path.abspath(path) for path in args.history]
        report["history_coverage"] = coverage
        print(
            f"History coverage: {coverage['covered_task_count']}/{coverage['task_count']} tasks, "
            f"weight {coverage['covered_task_weight']}/{coverage['task_weight']}",
            flush=True,
        )
        if not coverage["complete"] and not args.allow_incomplete_history:
            report["status"] = "refused_incomplete_history"
            write_report(args.report, report)
            missing = [item["index"] for item in coverage["tasks"] if not item["covered"]]
            raise RuntimeError(
                f"Optimized benchmark refused: missing valid history for task indices {missing}. "
                f"See {args.report}, complete measurement coverage, or explicitly pass "
                "--allow-incomplete-history to permit TVM fallback schedules."
            )

        filtered_history_path, build_record_count = filter_history_for_build(
            args.history, [task.workload_key for task in tasks], target
        )
        report["history_coverage"]["build_record_count"] = build_record_count
        optimized_history = [filtered_history_path]
    else:
        filtered_history_path = None
        optimized_history = []

    device = tvm.device(target.kind.name, args.device_id)
    if not device.exist:
        if filtered_history_path and os.path.exists(filtered_history_path):
            os.unlink(filtered_history_path)
        raise RuntimeError(f"TVM device does not exist: {target.kind.name}({args.device_id})")

    baseline_outputs = None
    modes = ["baseline", "optimized"] if args.mode == "both" else [args.mode]
    try:
        for mode in modes:
            library_path = args.baseline_lib if mode == "baseline" else args.optimized_lib
            print(f"Building/loading {mode} module...", flush=True)
            library, compile_seconds, library_source = build_or_load(
                mode, mod, params, target, optimized_history, library_path
            )
            outputs, timing = run_library(
                library,
                device,
                inputs,
                args.warmup,
                args.number,
                args.repeat,
                args.min_repeat_ms,
            )
            report[mode] = {
                "compile_seconds": compile_seconds,
                "library_source": library_source,
                "library_path": os.path.abspath(library_path) if library_path else None,
                "timing": timing,
                "outputs": [
                    {"index": index, "shape": list(value.shape), "dtype": str(value.dtype)}
                    for index, value in enumerate(outputs)
                ],
            }
            print(f"{mode}: median {timing['median_ms']:.6f} ms", flush=True)
            if mode == "baseline":
                baseline_outputs = outputs
            elif baseline_outputs is not None:
                report["correctness"] = compare_outputs(
                    baseline_outputs, outputs, args.rtol, args.atol
                )
            del outputs, library
            gc.collect()
    finally:
        if filtered_history_path and os.path.exists(filtered_history_path):
            os.unlink(filtered_history_path)

    if args.mode == "both":
        baseline_ms = report["baseline"]["timing"]["median_ms"]
        optimized_ms = report["optimized"]["timing"]["median_ms"]
        report["speedup"] = baseline_ms / optimized_ms
        report["status"] = "ok" if report["correctness"]["passed"] else "output_mismatch"
        print(f"Whole-model speedup: {report['speedup']:.4f}x", flush=True)
    else:
        report["status"] = "ok"
    write_report(args.report, report)
    print(f"Report: {os.path.abspath(args.report)}", flush=True)
    if report["status"] == "output_mismatch":
        raise RuntimeError("Baseline and optimized outputs do not match; see the JSON report")


if __name__ == "__main__":
    main()
