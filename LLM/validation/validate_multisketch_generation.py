#!/usr/bin/env python3
"""Validate distinct normalized PPT sketches with a fixed model budget per case."""

from LLM.core.layout import code_dependencies, source_path, module_command, project_pythonpath


import argparse
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path

from LLM.core.io import atomic_json, file_sha256
from LLM.validation.validate_multicandidate_generation import run_attempt, verify_inputs
from LLM.validation.candidate_metrics import summarize_workload


def mode_order(manifest, sample, attempt):
    """Keep each anchor's order identical to the preceding single-sketch run."""
    reverse = (sample["workload_index"] + sample["sketch_index"] + attempt) % 2
    return manifest["modes"][::-1] if reverse else manifest["modes"]


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
        results = []
        samples = json.loads((root / "samples.json").read_text())
        for index, sample in enumerate(samples):
            if sample["workload_index"] % len(manifest["gpus"]) != args.shard:
                continue
            case_dir = directory / f"case{index:02}"
            case_dir.mkdir()
            attempts, cache = [], {}
            for attempt in range(manifest["attempts"]):
                for mode in mode_order(manifest, sample, attempt):
                    atomic_json(
                        directory / "status.json",
                        {
                            "state": "RUNNING",
                            "case": index,
                            "workload": sample["workload_index"],
                            "sketch": sample["sketch_index"],
                            "attempt": attempt,
                            "mode": mode,
                            "completed_cases": len(results),
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
                        f"case={index} workload={sample['workload_index']} "
                        f"sketch={sample['sketch_index']} {mode}/{attempt} "
                        f"parsed={row['parsed']} "
                        f"correct={sum(bool(c['check'].get('correct')) for c in row['candidates'])}",
                        flush=True,
                    )
            result = {
                "index": index,
                **{
                    key: sample[key]
                    for key in (
                        "workload_index",
                        "sketch_index",
                        "sketch_id",
                        "sample_id",
                        "workload_key",
                    )
                },
                "family": sample["operator_family"],
                "modes": {
                    mode: summarize_workload(attempts, mode, manifest["required"])
                    for mode in manifest["modes"]
                },
                "attempts": attempts,
                "distinct_checked": len(cache),
            }
            results.append(result)
            atomic_json(directory / "results.json", results)
        atomic_json(
            directory / "status.json",
            {
                "state": "SUCCEEDED",
                "cases": len(results),
                "seconds": time.monotonic() - started,
            },
        )
    except BaseException:
        atomic_json(directory / "status.json", {"state": "FAILED", "error": traceback.format_exc()})
        raise


def prepare(args):
    from LLM.audit.audit_tvm_backend import compiler_fingerprint
    from LLM.validation.multisketch_selection import select_multisketch_suite

    # Check driver access before creating an experiment directory.
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv"], text=True
    )
    anchor = Path(args.anchor_run).resolve()
    previous = json.loads((anchor / "manifest.json").read_text())
    samples, selection = select_multisketch_suite(
        anchor, Path(args.source_run).resolve(), args.max_sketches
    )
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    frozen = root / "frozen_code"
    frozen.mkdir()
    paths = [Path(p) for p in previous["code"]]
    paths.extend(
        source_path(name).resolve()
        for name in (
            "validate_multisketch_generation.py",
            "multisketch_selection.py",
            "multisketch_metrics.py",
            "prepare_audited_clm.py",
            "common.py",
        )
    )
    paths = code_dependencies(paths)
    for path in paths:
        shutil.copy2(path, frozen / path.name)
    atomic_json(root / "samples.json", samples)
    atomic_json(root / "selection.json", selection)
    model = previous["model"]
    manifest = {
        "created_at": time.time(),
        "model": model,
        "registry": previous["registry"],
        "anchor_run": str(anchor),
        "source_run": str(Path(args.source_run).resolve()),
        "code": {str(p): file_sha256(p) for p in paths},
        "model_files": {
            str(p): file_sha256(p)
            for p in Path(model).iterdir()
            if p.is_file() and p.suffix in {".json", ".safetensors", ".txt"}
        },
        "input_files": selection["input_files"],
        "samples_sha256": file_sha256(root / "samples.json"),
        "selection_sha256": file_sha256(root / "selection.json"),
        "compiler": compiler_fingerprint(),
        "gpus": args.gpus,
        "attempts": args.attempts,
        "required": args.required,
        "max_sketches": args.max_sketches,
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
        "latency_measured": False,
        "budget_unit": "eight_model_attempts_per_workload_sketch_and_mode_not_per_pooled_workload",
        "check_reuse_scope": "exact_state_within_case_across_modes",
        "correctness_reference": "same_TE_unscheduled_LLVM_not_independent_semantic_oracle",
        "gpu_inventory": inventory,
    }
    verify_inputs(manifest)
    atomic_json(root / "manifest.json", manifest)
    return root, manifest


def run(args):
    from LLM.validation.multisketch_metrics import summarize_multisketch

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
                        *module_command("validate_multisketch_generation.py"),
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
        for filename, field in (
            ("samples.json", "samples_sha256"),
            ("selection.json", "selection_sha256"),
        ):
            if file_sha256(root / filename) != manifest[field]:
                raise ValueError(f"Frozen {filename} changed")
        results = []
        for shard in range(len(args.gpus)):
            directory = root / f"shard{shard:02}"
            assert json.loads((directory / "status.json").read_text())["state"] == "SUCCEEDED"
            results.extend(json.loads((directory / "results.json").read_text()))
        results.sort(key=lambda row: row["index"])
        samples = json.loads((root / "samples.json").read_text())
        assert [r["sample_id"] for r in results] == [s["sample_id"] for s in samples]
        assert all(len(r["attempts"]) == args.attempts * len(args.modes) for r in results)
        atomic_json(root / "results.json", results)
        summary = summarize_multisketch(results, args.modes, args.required)
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
        "--anchor-run", default="LLM/audit_runs/4090_resource_constraints_20261117_02"
    )
    parser.add_argument(
        "--source-run", default="LLM/audit_runs/4090_plan5_roundtrip_full_20261102_01"
    )
    parser.add_argument("--max-sketches", type=int, default=4)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--required", type=int, default=4)
    parser.add_argument(
        "--modes", nargs="+", choices=["state_v1", "state_v2"], default=["state_v1", "state_v2"]
    )
    parser.add_argument("--shard", type=int)
    arguments = parser.parse_args()
    if arguments.shard is not None:
        run_shard(arguments)
    else:
        if not 0 < arguments.required <= arguments.attempts or arguments.max_sketches < 1:
            parser.error("attempts >= required > 0 and max-sketches > 0 are required")
        if len(set(arguments.gpus)) != len(arguments.gpus):
            parser.error("GPU IDs must be distinct")
        if len(set(arguments.modes)) != len(arguments.modes):
            parser.error("Modes must be distinct")
        run(arguments)
