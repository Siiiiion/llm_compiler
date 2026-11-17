#!/usr/bin/env python3
"""Fixed-attempt, sampled validation of distinct CUDA schedule candidates."""

from LLM.core.layout import code_dependencies, source_path, module_command, project_pythonpath


from LLM.validation.candidate_metrics import aggregate, summarize_workload


import argparse
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path

from LLM.core.io import atomic_json, file_sha256
from LLM.core.schedule_record_utils import input_hash


def verify_inputs(manifest):
    for key in ("code", "model_files", "input_files"):
        for path, expected in manifest[key].items():
            if file_sha256(path) != expected:
                raise ValueError(f"Frozen input changed: {path}")


def read_worker_check(check_path, return_code, output):
    """Retain a reported runtime failure even if CUDA teardown aborts the worker."""
    if not check_path.exists():
        raise RuntimeError(f"Candidate worker exited {return_code}: {output[-4000:]}")
    check = json.loads(check_path.read_text())
    if return_code:
        # A signal or import failure without a complete negative report remains
        # an infrastructure failure. A teardown abort cannot erase a GPU error.
        if check.get("correct") is not False or not check.get("error"):
            raise RuntimeError(f"Candidate worker exited {return_code}: {output[-4000:]}")
        check["worker_return_code"] = return_code
        atomic_json(check_path, check)
    return check


def inspect_candidate(record, identity, directory, registry, reference_dir, timeout):
    path = directory / f"{identity}.json"
    atomic_json(path, record)
    command = [
        *module_command("validate_constrained_generation.py"),
        "--worker-record",
        str(path),
        "--registry",
        registry,
        "--reference-dir",
        str(reference_dir),
        "--check-runtime",
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout
        )
        path.with_suffix(".log").write_text(completed.stdout)
        check_path = path.with_suffix(".check.json")
        check = read_worker_check(check_path, completed.returncode, completed.stdout)
    except subprocess.TimeoutExpired as exc:
        path.with_suffix(".log").write_text(
            (exc.stdout or b"").decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        progress = path.with_suffix(".progress.json")
        check = {
            "timeout": True,
            "resource_valid": False,
            "built": False,
            "correct": False,
            "last_stage": json.loads(progress.read_text()) if progress.exists() else None,
        }
    return {
        "identity": identity,
        "record_path": str(path),
        "check": check,
        "check_wall_seconds": time.monotonic() - started,
    }


def run_attempt(model, tokenizer, sample, mode, seed, directory, cache, registry, settings):
    import torch
    from tvm import auto_scheduler
    from LLM.experiments.diagnose_model_generation import make_sketch, TracedModel
    from LLM.validation.eval_struct import _generate_decisions_for_states
    from LLM.generation.generation_protocol import isolated_generation_rng, resolve_max_new_tokens
    from LLM.data.make_dataset import input_to_tokens

    inp = make_sketch(sample)
    wrapped = TracedModel(model, tokenizer)
    telemetry, callbacks = {}, []
    row = {
        "mode": mode,
        "seed": seed,
        "callbacks": callbacks,
        "telemetry": telemetry,
        "candidates": [],
        "parsed": 0,
    }

    def callback(task, states, requested):
        traces = []
        prompts = input_to_tokens(task, states, inp)
        before = len(wrapped.calls)
        outputs = _generate_decisions_for_states(
            task,
            states,
            inp,
            tokenizer,
            wrapped,
            "cuda:0",
            resolve_max_new_tokens(requested),
            True,
            settings["top_p"],
            settings["top_k"],
            settings["temperature"],
            1,
            False,
            telemetry,
            mode,
            traces,
        )
        callbacks.append(
            {
                "requested": int(requested),
                "prompts": prompts,
                "outputs": outputs,
                "constraint_traces": traces,
                "model_calls": wrapped.calls[before:],
            }
        )
        atomic_json(directory / "callbacks.json", callbacks)
        return outputs

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with isolated_generation_rng(seed):
        states = auto_scheduler.SketchPolicy(inp.task, seed=seed, verbose=0).gen_states(
            [inp.state], callback
        )
    torch.cuda.synchronize()
    row.update(
        generation_seconds=time.monotonic() - started,
        parsed=len(states),
        peak_memory_bytes=torch.cuda.max_memory_allocated(),
    )
    for state in states:
        record = {
            "i": json.loads(auto_scheduler.MeasureInput(inp.task, state).to_json()),
            "r": [[0], 0, 0, 0],
            "v": "v0.6",
        }
        identity = input_hash(record)
        reused = identity in cache
        if not reused:
            cache[identity] = inspect_candidate(
                record,
                identity,
                directory.parent / "candidates",
                registry,
                directory.parent / "references",
                settings["check_timeout"],
            )
        row["candidates"].append({**cache[identity], "check_reused": reused})
    row["seconds"] = time.monotonic() - started
    atomic_json(directory / "attempt.json", row)
    return row


def run_shard(args):
    import torch
    from LLM.audit.audit_tvm_backend import register_tasks
    from LLM.generation.gen_state import load_model, load_tokenizer

    root = Path(args.run_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    directory = root / f"shard{args.shard:02}"
    directory.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        torch.set_num_threads(4)
        verify_inputs(manifest)
        register_tasks(manifest["registry"])
        tokenizer = load_tokenizer(manifest["model"], True, True)
        model = load_model(manifest["model"], True, "cuda:0")
        model.eval()
        model.config.use_cache = True
        workloads = []
        samples = json.loads((root / "samples.json").read_text())
        for index, sample in enumerate(samples):
            if index % len(manifest["gpus"]) != args.shard:
                continue
            case_dir = directory / f"workload{index:02}"
            case_dir.mkdir()
            attempts, cache = [], {}
            for attempt in range(manifest["attempts"]):
                # Alternate mode order; both modes always consume their full attempt budget.
                modes = manifest["modes"] if (index + attempt) % 2 == 0 else manifest["modes"][::-1]
                for mode in modes:
                    atomic_json(
                        directory / "status.json",
                        {
                            "state": "RUNNING",
                            "workload": index,
                            "attempt": attempt,
                            "mode": mode,
                            "completed_workloads": len(workloads),
                            "pid": os.getpid(),
                        },
                    )
                    attempt_dir = case_dir / f"{mode}_{attempt:02}"
                    attempt_dir.mkdir()
                    row = run_attempt(
                        model,
                        tokenizer,
                        sample,
                        mode,
                        manifest["seed"] + attempt,
                        attempt_dir,
                        cache,
                        manifest["registry"],
                        manifest,
                    )
                    attempts.append(row)
                    print(
                        f"workload={index} {mode}/{attempt} parsed={row['parsed']} "
                        f"correct={sum(c['check'].get('correct',False) for c in row['candidates'])} "
                        f"tokens={row['telemetry'].get('generated_tokens',0)}",
                        flush=True,
                    )
            result = {
                "index": index,
                "sample_id": sample["sample_id"],
                "workload_key": sample["workload_key"],
                "family": sample["operator_family"],
                "modes": {
                    mode: summarize_workload(attempts, mode, manifest["required"])
                    for mode in manifest["modes"]
                },
                "attempts": attempts,
                "distinct_checked": len(cache),
            }
            workloads.append(result)
            atomic_json(directory / "results.json", workloads)
        atomic_json(
            directory / "status.json",
            {
                "state": "SUCCEEDED",
                "workloads": len(workloads),
                "seconds": time.monotonic() - started,
            },
        )
    except BaseException:
        atomic_json(directory / "status.json", {"state": "FAILED", "error": traceback.format_exc()})
        raise


def prepare(args):
    from LLM.audit.audit_tvm_backend import compiler_fingerprint

    source = Path(args.source_run).resolve()
    source_manifest = json.loads((source / "manifest.json").read_text())
    samples = json.loads((source / "samples.json").read_text())
    assert file_sha256(source / "samples.json") == source_manifest["samples_sha256"]
    assert json.loads((source / "status.json").read_text())["state"] == "SUCCEEDED"
    assert samples and all(sample["split"] == "validation" for sample in samples)
    assert len({sample["workload_key"] for sample in samples}) == len(samples)
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    frozen = root / "frozen_code"
    frozen.mkdir()
    code_paths = [Path(path) for path in source_manifest["code"]]
    code_paths.extend(
        source_path(name)
        for name in ("validate_multicandidate_generation.py", "resource_feasibility.py")
    )
    code_paths = code_dependencies(code_paths)
    for path in code_paths:
        shutil.copy2(path, frozen / path.name)
    model = source_manifest["models"]["base"]
    model_files = {
        str(path): file_sha256(path)
        for path in Path(model).iterdir()
        if path.is_file() and path.suffix in {".json", ".safetensors", ".txt"}
    }
    atomic_json(root / "samples.json", samples)
    input_paths = [source / name for name in ("manifest.json", "samples.json", "status.json")]
    input_paths.append(Path(args.registry).resolve())
    manifest = {
        "created_at": time.time(),
        "model": model,
        "registry": str(Path(args.registry).resolve()),
        "source_run": str(source),
        "code": {str(p): file_sha256(p) for p in code_paths},
        "model_files": model_files,
        "input_files": {str(p): file_sha256(p) for p in input_paths},
        "samples_sha256": file_sha256(root / "samples.json"),
        "compiler": compiler_fingerprint(),
        "gpus": args.gpus,
        "attempts": args.attempts,
        "required": args.required,
        "seed": 42,
        "modes": args.modes,
        "sampling": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "min_tokens": 256,
        "token_scale": 4,
        "extra_token_cap": None,
        "early_stop_at_k": False,
        "fallback": False,
        "batch_size": 1,
        "check_timeout": 180,
        "holdout_used": False,
        "identity": "input_hash_includes_task_and_exact_transform_steps_no_factor_normalization",
        "correctness_reference": "same_TE_unscheduled_LLVM_not_independent_semantic_oracle",
        "latency_measured": False,
        "gpu_inventory": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv"], text=True
        ),
    }
    atomic_json(root / "manifest.json", manifest)
    return root, manifest


def run(args):
    root, manifest = prepare(args)
    processes, logs = [], []
    started = time.monotonic()
    try:
        for shard, gpu in enumerate(args.gpus):
            environment = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(gpu),
                OMP_NUM_THREADS="4",
                MKL_NUM_THREADS="4",
                TOKENIZERS_PARALLELISM="false",
            )
            log = (root / f"shard{shard:02}.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [
                        *module_command("validate_multicandidate_generation.py"),
                        "--run-dir",
                        str(root),
                        "--shard",
                        str(shard),
                    ],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes):
                raise RuntimeError("A shard failed; see its status and log")
            atomic_json(
                root / "status.json",
                {
                    "state": "RUNNING",
                    "pids": [p.pid for p in processes],
                    "return_codes": [p.poll() for p in processes],
                },
            )
            time.sleep(5)
        if any(process.returncode for process in processes):
            raise RuntimeError("A shard failed")
        verify_inputs(manifest)
        if file_sha256(root / "samples.json") != manifest["samples_sha256"]:
            raise ValueError("Sample identity changed")
        workloads = []
        for shard in range(len(args.gpus)):
            directory = root / f"shard{shard:02}"
            assert json.loads((directory / "status.json").read_text())["state"] == "SUCCEEDED"
            workloads.extend(json.loads((directory / "results.json").read_text()))
        workloads.sort(key=lambda row: row["index"])
        samples = json.loads((root / "samples.json").read_text())
        assert [row["sample_id"] for row in workloads] == [s["sample_id"] for s in samples]
        assert all(
            len(row["attempts"]) == args.attempts * len(manifest["modes"]) for row in workloads
        )
        atomic_json(root / "results.json", workloads)
        summary = aggregate(workloads, manifest["modes"])
        atomic_json(root / "summary.json", summary)
        atomic_json(
            root / "status.json",
            {
                "state": "SUCCEEDED",
                "identity_verified": True,
                "seconds": time.monotonic() - started,
            },
        )
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        atomic_json(root / "status.json", {"state": "FAILED", "error": traceback.format_exc()})
        raise
    finally:
        for log in logs:
            log.close()


if __name__ == "__main__":
    os.environ["PYTHONPATH"] = project_pythonpath(os.environ.get("PYTHONPATH", ""))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--source-run", default="LLM/audit_runs/4090_constraints_expanded_20261111_01"
    )
    parser.add_argument("--registry", default="/data3/qsy/dataset/network_info/4090/all_tasks.pkl")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--required", type=int, default=4)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("none", "state_v1", "state_v2"),
        default=["none", "state_v1"],
    )
    parser.add_argument("--shard", type=int)
    arguments = parser.parse_args()
    if arguments.shard is not None:
        run_shard(arguments)
    else:
        if arguments.required <= 0 or arguments.attempts < arguments.required:
            parser.error("attempts must be >= required > 0")
        if len(set(arguments.gpus)) != len(arguments.gpus):
            parser.error("GPU IDs must be distinct")
        if len(set(arguments.modes)) != len(arguments.modes):
            parser.error("Modes must be distinct")
        run(arguments)
