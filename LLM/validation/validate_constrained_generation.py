#!/usr/bin/env python3
"""Frozen, fixed-budget CUDA grammar A/B on validation sketches only."""

from LLM.core.layout import code_dependencies, source_path, module_command, project_pythonpath


import argparse
import os
import gc
import json
import random
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

from LLM.core.io import atomic_json, file_sha256
from LLM.core.schedule_record_utils import digest


def check_worker(args):
    """Run lowering, compilation and differential correctness under a hard timeout."""
    import numpy as np
    import tvm
    from tvm import auto_scheduler
    from tvm.auto_scheduler.measure import recover_measure_input, _local_build_worker
    from tvm.contrib import tar
    from LLM.audit.audit_tvm_backend import register_tasks
    from LLM.generation.constrained_generation import verify_gpu_resources

    register_tasks(args.registry)
    path = Path(args.worker_record)
    report = {"resource_valid": False, "built": False, "correct": False}
    started = time.monotonic()
    try:
        inp, _ = auto_scheduler.measure_record.load_record_from_string(path.read_text())
        inp = recover_measure_input(inp, rebuild_state=True)
        atomic_json(path.with_suffix(".progress.json"), {"stage": "verify_resources"})
        verify_gpu_resources(inp.task, inp.state)
        report["resource_valid"] = True
        with tempfile.TemporaryDirectory(prefix="grammar-build-") as temporary:
            tempfile.tempdir = temporary
            atomic_json(path.with_suffix(".progress.json"), {"stage": "build"})
            filename, tensors, error, message, seconds = _local_build_worker(
                inp.serialize(), tar.tar, 0
            )
            report.update(build_seconds=float(seconds), build_error=int(error))
            if error:
                raise ValueError(str(message)[-5000:])
            report["built"] = bool(filename and Path(filename).stat().st_size)
            if not report["built"]:
                raise ValueError("Build returned no exported module")
            if args.check_runtime:
                if inp.task.layout_rewrite_option != 0:
                    raise ValueError("Correctness reference requires unchanged input layouts")
                atomic_json(path.with_suffix(".progress.json"), {"stage": "correctness"})
                original = list(inp.task.compute_dag.tensors)
                if len(original) != len(tensors):
                    raise ValueError("Compiled and reference argument counts differ")
                total_bytes = sum(
                    int(np.prod([int(v) for v in t.shape])) * np.dtype(t.dtype).itemsize
                    for t in original
                )
                if total_bytes > 256 * 1024 * 1024:
                    raise ValueError("Correctness input exceeds 256 MiB pilot limit")
                inputs = [
                    i for i, t in enumerate(original) if isinstance(t.op, tvm.te.PlaceholderOp)
                ]
                outputs = [i for i in range(len(original)) if i not in inputs]
                cache = Path(args.reference_dir)
                cache.mkdir(exist_ok=True, parents=True)
                gpu = tvm.cuda(0)
                gpu_module = tvm.runtime.load_module(filename)
                errors = []
                for seed in (42, 43):
                    reference_file = cache / f"seed{seed}.npz"
                    rng = np.random.default_rng(seed)
                    values = {}
                    for i in inputs:
                        t = original[i]
                        shape = tuple(int(v) for v in t.shape)
                        dtype = np.dtype(t.dtype)
                        values[i] = (
                            rng.uniform(-0.5, 0.5, shape).astype(dtype)
                            if dtype.kind == "f"
                            else rng.integers(0, 3, shape).astype(dtype)
                        )
                    if not reference_file.exists():
                        schedule = tvm.te.create_schedule([original[i].op for i in outputs])
                        reference = tvm.build(schedule, original, target="llvm")
                        cpu_arrays = [
                            (
                                tvm.nd.array(values[i], tvm.cpu())
                                if i in inputs
                                else tvm.nd.empty(
                                    tuple(int(v) for v in t.shape), t.dtype, tvm.cpu()
                                )
                            )
                            for i, t in enumerate(original)
                        ]
                        reference(*cpu_arrays)
                        np.savez(reference_file, **{str(i): cpu_arrays[i].numpy() for i in outputs})
                    with np.load(reference_file) as reference:
                        arrays = [
                            (
                                tvm.nd.array(values[i], gpu)
                                if i in inputs
                                else tvm.nd.empty(tuple(int(v) for v in t.shape), t.dtype, gpu)
                            )
                            for i, t in enumerate(original)
                        ]
                        gpu_module(*arrays)
                        gpu.sync()
                        for i in outputs:
                            actual, expected = arrays[i].numpy(), reference[str(i)]
                            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
                            errors.append(float(np.max(np.abs(actual - expected))))
                report.update(
                    correct=True,
                    correctness_seeds=[42, 43],
                    max_abs_error=max(errors),
                    reference="unscheduled_same_TE_on_LLVM_not_independent_semantic_oracle",
                )
    except Exception:
        report["error"] = traceback.format_exc()[-7000:]
    report["seconds"] = time.monotonic() - started
    atomic_json(path.with_suffix(".check.json"), report)


def select_samples(source, count):
    prior = json.loads(
        Path("LLM/audit_runs/4090_plan6_generation_diagnosis_20261109_01/samples.json").read_text()
    )
    selected = {sample["workload_key"]: sample for sample in prior.values()}
    candidates = [json.loads(line) for line in (source / "samples.jsonl").open()]
    candidates = [s for s in candidates if s["split"] == "validation" and "median" in s["roles"]]
    random.Random(42).shuffle(candidates)
    groups = {}
    for sample in candidates:
        category = (sample["operator_family"], sample["length_bucket"])
        groups.setdefault(category, []).append(sample)
    while len(selected) < count and any(groups.values()):
        for group in sorted(groups):
            if groups[group] and len(selected) < count:
                sample = groups[group].pop()
                selected.setdefault(sample["workload_key"], sample)
    return list(selected.values())[:count]


def run_case(model, tokenizer, sample, mode, directory, args, registry, replay=False):
    from tvm import auto_scheduler
    from LLM.experiments.diagnose_model_generation import make_sketch
    from LLM.validation.eval_struct import _generate_decisions_for_states
    from LLM.generation.generation_protocol import isolated_generation_rng, resolve_max_new_tokens
    from LLM.data.make_dataset import input_to_tokens

    inp = make_sketch(sample)
    telemetry, callbacks = {}, []
    started = time.monotonic()
    result = {
        "sample_id": sample["sample_id"],
        "workload_key": sample["workload_key"],
        "family": sample["operator_family"],
        "length_bucket": sample["length_bucket"],
        "mode": mode,
        "replay": replay,
        "callbacks": callbacks,
        "checks": [],
        "parsed": 0,
    }

    def callback(task, states, requested):
        prompts = input_to_tokens(task, states, inp)
        traces = []
        if replay:
            if any(not sample["row"]["text"].startswith(p) for p in prompts):
                raise ValueError("Known suffix does not match actual state prompt")
            outputs = [sample["row"]["text"][len(p) :].split() for p in prompts]
        else:
            outputs = _generate_decisions_for_states(
                task,
                states,
                inp,
                tokenizer,
                model,
                "cuda:0",
                resolve_max_new_tokens(requested),
                False,
                0.95,
                50,
                0.7,
                1,
                False,
                telemetry,
                mode,
                traces,
            )
        callbacks.append(
            {"requested": int(requested), "prompts": prompts, "outputs": outputs, "traces": traces}
        )
        atomic_json(directory / "callbacks.json", callbacks)
        return outputs

    try:
        import torch

        torch.cuda.reset_peak_memory_stats()
        with isolated_generation_rng(42):
            states = auto_scheduler.SketchPolicy(inp.task, seed=42, verbose=0).gen_states(
                [inp.state], callback
            )
        result["generation_seconds"] = time.monotonic() - started
        result["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
        result["parsed"] = len(states)
        seen = set()
        for state in states:
            record = auto_scheduler.MeasureInput(inp.task, state).to_json()
            if record in seen:
                continue
            seen.add(record)
            record_path = directory / f"candidate{len(seen)}.json"
            atomic_json(record_path, {"i": json.loads(record), "r": [[0], 0, 0, 0], "v": "v0.6"})
            command = [
                *module_command("validate_constrained_generation.py"),
                "--worker-record",
                str(record_path),
                "--registry",
                registry,
                "--reference-dir",
                str(Path(args.run_dir).resolve() / "references" / digest(sample["workload_key"])),
            ]
            if args.check_runtime:
                command.append("--check-runtime")
            try:
                completed = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=180,
                )
                record_path.with_suffix(".log").write_text(completed.stdout)
                check_file = record_path.with_suffix(".check.json")
                if completed.returncode or not check_file.exists():
                    raise RuntimeError(completed.stdout[-5000:])
                result["checks"].append(json.loads(check_file.read_text()))
            except subprocess.TimeoutExpired:
                progress = record_path.with_suffix(".progress.json")
                result["checks"].append(
                    {
                        "built": False,
                        "correct": False,
                        "timeout": True,
                        "last_stage": (
                            json.loads(progress.read_text()) if progress.exists() else None
                        ),
                    }
                )
        result["unique"] = len(seen)
    except Exception:
        result["error"] = traceback.format_exc()
    result.update(telemetry=telemetry, seconds=time.monotonic() - started)
    return result


def run(args):
    import torch
    from LLM.audit.audit_tvm_backend import register_tasks, compiler_fingerprint
    from LLM.generation.gen_state import load_model, load_tokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for actual model and candidate validation")
    torch.set_num_threads(4)
    directory = Path(args.run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    source = Path(args.source_run).resolve()
    source_manifest = json.loads((source / "manifest.json").read_text())
    if file_sha256(source / "samples.jsonl") != source_manifest["samples_sha256"]:
        raise ValueError("Source sample identity changed")
    audit = json.loads((Path(source_manifest["inputs"]["audit_dir"]) / "manifest.json").read_text())
    registry = audit["registry"]["path"]
    register_tasks(registry)
    samples = select_samples(source, args.max_workloads)
    assert all(s["split"] == "validation" for s in samples)
    atomic_json(directory / "samples.json", samples)
    paths = {
        "base": args.base_model,
        "original": str(Path(args.training_run) / "original/model"),
        "filtered": str(Path(args.training_run) / "filtered/model"),
    }
    paths = {name: paths[name] for name in args.models}
    code = [Path(__file__).resolve()] + [
        source_path(n)
        for n in (
            "constrained_generation.py",
            "generation_protocol.py",
            "eval_struct.py",
            "gen_state.py",
            "diagnose_model_generation.py",
            "make_dataset.py",
            "tokenization_protocol.py",
            "schedule_record_utils.py",
            "audit_tvm_backend.py",
        )
    ]
    code = code_dependencies(code)
    frozen = directory / "frozen_code"
    frozen.mkdir()
    for path in code:
        shutil.copy2(path, frozen / path.name)
    hashes = {str(path.resolve()): file_sha256(path) for path in code}
    models = {
        str(p): file_sha256(p)
        for path in set(paths.values()) | {args.base_model}
        for p in Path(path).glob("*")
        if p.suffix in {".json", ".safetensors"}
    }
    manifest = {
        "created_at": time.time(),
        "code": hashes,
        "models": paths,
        "model_files": models,
        "samples_sha256": file_sha256(directory / "samples.json"),
        "source_manifest": source_manifest["manifest_id"],
        "compiler": compiler_fingerprint(),
        "seed": 42,
        "greedy": True,
        "fallback": False,
        "modes": args.modes,
        "min_tokens": 256,
        "scale": 4,
        "extra_cap": None,
        "batch_size": 1,
        "check_timeout": 180,
        "runtime_correctness": args.check_runtime,
        "holdout_used": False,
        "constraint_search_subset": "divisor_factors_standard_FSP_roots_only",
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv"], text=True
        ),
    }
    atomic_json(directory / "manifest.json", manifest)
    tokenizer = load_tokenizer(args.base_model, True, True)
    results = []
    try:
        for arm, checkpoint in paths.items():
            model = load_model(checkpoint, True, "cuda:0")
            model.eval()
            model.config.use_cache = True
            for mode in args.modes:
                for i, sample in enumerate(samples):
                    atomic_json(
                        directory / "status.json",
                        {
                            "state": "RUNNING",
                            "arm": arm,
                            "mode": mode,
                            "case": i,
                            "completed": len(results),
                        },
                    )
                    case_dir = directory / f"{arm}_{mode}_{i:03}"
                    case_dir.mkdir()
                    result = run_case(model, tokenizer, sample, mode, case_dir, args, registry)
                    result["arm"] = arm
                    results.append(result)
                    atomic_json(directory / "results.json", results)
                    print(
                        f"{arm}/{mode}/{i}: parsed={result['parsed']} built={sum(c.get('built',False) for c in result['checks'])} correct={sum(c.get('correct',False) for c in result['checks'])} tokens={result['telemetry'].get('generated_tokens',0)} error={result.get('error','')[-300:]}",
                        flush=True,
                    )
            del model
            gc.collect()
            torch.cuda.empty_cache()
        for path, sha in {**hashes, **models}.items():
            if file_sha256(path) != sha:
                raise ValueError(f"Frozen input changed: {path}")
        summaries = {}
        for arm in paths:
            for mode in args.modes:
                cases = [r for r in results if r["arm"] == arm and r["mode"] == mode]
                summaries[f"{arm}/{mode}"] = {
                    "workloads": len(cases),
                    "parsed": sum(r["parsed"] > 0 for r in cases),
                    "built": sum(any(c.get("built") for c in r["checks"]) for r in cases),
                    "correct": sum(any(c.get("correct") for c in r["checks"]) for r in cases),
                    "generated_tokens": sum(
                        r["telemetry"].get("generated_tokens", 0) for r in cases
                    ),
                    "generation_seconds": sum(r.get("generation_seconds", 0) for r in cases),
                    "errors": sum("error" in r for r in cases),
                }
        atomic_json(directory / "summary.json", summaries)
        errors = sum("error" in r for r in results)
        atomic_json(
            directory / "status.json",
            {
                "state": "FAILED" if errors else "SUCCEEDED",
                "cases": len(results),
                "errors": errors,
                "finished_at": time.time(),
            },
        )
        print(json.dumps(summaries, indent=2), flush=True)
        if errors:
            raise RuntimeError(f"{errors} infrastructure failures")
    except BaseException:
        atomic_json(
            directory / "status.json",
            {"state": "FAILED", "completed": len(results), "error": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    os.environ["PYTHONPATH"] = project_pythonpath(os.environ.get("PYTHONPATH", ""))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir")
    parser.add_argument(
        "--source-run", default="LLM/audit_runs/4090_plan5_roundtrip_full_20261102_01"
    )
    parser.add_argument(
        "--training-run", default="LLM/audit_runs/4090_plan6_controlled_20261109_02"
    )
    parser.add_argument(
        "--base-model",
        default="/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1-v2-ext-b2",
    )
    parser.add_argument("--max-workloads", type=int, default=4)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["base", "original", "filtered"],
        default=["base", "original", "filtered"],
    )
    parser.add_argument(
        "--modes", nargs="+", choices=["none", "state_v1"], default=["none", "state_v1"]
    )
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--worker-record")
    parser.add_argument("--registry")
    parser.add_argument("--reference-dir")
    arguments = parser.parse_args()
    if arguments.worker_record:
        check_worker(arguments)
    else:
        run(arguments)
