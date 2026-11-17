#!/usr/bin/env python3
"""Recheck unresolved roundtrip samples and sibling controls in a new, durable run."""

from LLM.core.layout import code_dependencies, source_path, project_pythonpath


import argparse
import fcntl
import json
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path

from LLM.validation.generation_roundtrip_worker import build_candidate, check_sample, initialize
from LLM.core.schedule_record_utils import append_journal, digest, read_journal
from LLM.core.io import atomic_json, file_sha256
from LLM.core.runtime import bounded_pool, now
from LLM.validation.validate_generation_roundtrip import verify_inputs


def load_source(directory):
    """Verify the original identities without rewriting or repairing its journal."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    identifier = manifest.pop("manifest_id")
    if digest(manifest) != identifier:
        raise ValueError("Source manifest identity differs")
    manifest["manifest_id"] = identifier
    if file_sha256(directory / "samples.jsonl") != manifest["samples_sha256"]:
        raise ValueError("Source samples changed")
    summary = json.loads((directory / "summary.json").read_text())
    if file_sha256(directory / "results.jsonl") != summary["results_sha256"]:
        raise ValueError("Source results changed")
    results = read_journal(directory / "results.jsonl")
    samples = read_journal(directory / "samples.jsonl")
    owners = {sample["sample_id"]: sample for sample in samples}
    seen = set()
    for result in results:
        value = dict(result)
        identity = value.pop("result_id")
        if (
            digest(value) != identity
            or result["manifest_id"] != identifier
            or result["sample_id"] not in owners
            or result["sample_id"] in seen
        ):
            raise ValueError("Invalid or duplicate source result")
        seen.add(result["sample_id"])
    if seen != set(owners):
        raise ValueError("Source run is incomplete")
    unresolved = {result["sample_id"] for result in results if result["status"] != "PASSED"}
    workloads = {owners[key]["workload_key"] for key in unresolved}
    selected = [sample for sample in samples if sample["workload_key"] in workloads]
    return manifest, selected, unresolved


def verify_compatible_inputs(source):
    """Only diagnostic code may differ; data, tokenizer and actual compiler must match."""
    from LLM.audit.audit_tvm_backend import compiler_fingerprint

    snapshot = dict(source["inputs"])
    snapshot["code"] = {
        name: value
        for name, value in snapshot["code"].items()
        if Path(name).name
        not in {
            "generation_roundtrip_worker.py",
            "validate_generation_roundtrip.py",
            "gen_state.py",
        }
    }
    verify_inputs(snapshot)
    audit = json.loads((Path(snapshot["audit_dir"]) / "manifest.json").read_text())
    actual = compiler_fingerprint()
    for name, expected in audit["compiler"].items():
        if name != "environment" and actual[name] != expected:
            raise ValueError(f"Compiler changed: {name}")
    return audit, actual


def run(args):
    # TVM's fresh Python workers do not inherit sys.path[0] from this script.
    os.environ["PYTHONPATH"] = project_pythonpath(os.environ.get("PYTHONPATH", ""))
    directory = Path(args.run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "manifest.json").exists():
            raise ValueError("Use a new diagnostic run directory")
        started = time.monotonic()
        atomic_json(directory / "status.json", {"state": "RUNNING", "stage": "PREFLIGHT"})
        try:
            source, samples, unresolved = load_source(args.source_run)
            if args.only_sample_id:
                chosen = set(args.only_sample_id)
                if not chosen.issubset({sample["sample_id"] for sample in samples}):
                    raise ValueError("Requested sample is not in the unresolved/control scope")
                samples = [sample for sample in samples if sample["sample_id"] in chosen]
                unresolved &= chosen
            audit, compiler = verify_compatible_inputs(source)
            code = {
                str(Path(__file__).resolve()): file_sha256(__file__),
                str(source_path("generation_roundtrip_worker.py").resolve()): file_sha256(
                    source_path("generation_roundtrip_worker.py")
                ),
            }
            for name in (
                "gen_state.py",
                "generation_protocol.py",
                "validate_generation_roundtrip.py",
            ):
                path = source_path(name).resolve()
                code[str(path)] = file_sha256(path)
            code = {str(path): file_sha256(path) for path in code_dependencies(code)}
            manifest = {
                "created_at": now(),
                "source_run": str(Path(args.source_run).resolve()),
                "source_manifest_id": source["manifest_id"],
                "source_summary_sha256": file_sha256(Path(args.source_run) / "summary.json"),
                "sample_ids": [sample["sample_id"] for sample in samples],
                "unresolved_sample_ids": sorted(unresolved),
                "sample_timeout": args.sample_timeout,
                "n_parallel": args.n_parallel,
                "separate_build": args.separate_build,
                "build_timeout": args.build_timeout,
                "compiler": compiler,
                "code": code,
                "partial_scope": True,
            }
            manifest["manifest_id"] = digest(manifest)
            atomic_json(directory / "manifest.json", manifest)
            (directory / "frozen_code").mkdir()
            for name in code:
                shutil.copy2(name, directory / "frozen_code" / Path(name).name)
            append_journal(directory / "samples.jsonl", samples)
            (directory / "tmp").mkdir()
            pending, reports = {}, []
            remaining = iter(samples)
            with bounded_pool(
                args.n_parallel,
                args.sample_timeout,
                initialize,
                (
                    audit["registry"]["path"],
                    source["inputs"]["tokenizer_path"],
                    str(directory / "tmp"),
                ),
            ) as pool, bounded_pool(
                args.n_parallel,
                args.build_timeout,
                initialize,
                (
                    audit["registry"]["path"],
                    source["inputs"]["tokenizer_path"],
                    str(directory / "tmp"),
                ),
            ) as build_pool:

                def submit_next():
                    sample = next(remaining, None)
                    if sample is None:
                        return
                    progress = directory / "stages" / f"{sample['sample_id']}.json"
                    future = pool.submit(check_sample, sample, str(progress), args.separate_build)
                    pending[future] = (sample, progress, None)

                for _ in range(args.n_parallel):
                    submit_next()
                while pending:
                    finished, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                    for future in finished:
                        sample, progress, reconstruction = pending.pop(future)
                        try:
                            report = future.result()
                            if report.get("serialized_candidate") is not None:
                                serialized = report.pop("serialized_candidate")
                                build_progress = (
                                    directory / "build_stages" / f"{sample['sample_id']}.json"
                                )
                                build_future = build_pool.submit(
                                    build_candidate,
                                    serialized,
                                    sample["sample_id"],
                                    str(build_progress),
                                )
                                pending[build_future] = (sample, build_progress, report)
                                submit_next()
                                continue
                            if reconstruction is not None:
                                build_report = report
                                report = reconstruction
                                report["reconstruction_stage_timings"] = report.pop("stage_timings")
                                report["build_stage_timings"] = build_report.pop(
                                    "stage_timings", []
                                )
                                report.update(build_report)
                        except Exception as error:
                            last = json.loads(progress.read_text()) if progress.exists() else {}
                            report = {
                                key: sample[key]
                                for key in (
                                    "sample_id",
                                    "row_index",
                                    "split",
                                    "workload_key",
                                    "length",
                                )
                            }
                            report.update(
                                status="FAILED",
                                stage=(
                                    "worker_timeout"
                                    if isinstance(error, TimeoutError)
                                    else "worker_environment"
                                ),
                                error_message=repr(error),
                                last_progress=last,
                            )
                            if reconstruction is not None:
                                report["reconstruction"] = reconstruction
                        report.pop("result_id", None)
                        report.update(
                            manifest_id=manifest["manifest_id"],
                            role=(
                                "unresolved"
                                if sample["sample_id"] in unresolved
                                else "sibling_control"
                            ),
                        )
                        report["result_id"] = digest(report)
                        append_journal(directory / "results.jsonl", [report])
                        reports.append(report)
                        print(
                            f"[{now()}] {len(reports)}/{len(samples)} row={sample['row_index']} "
                            f"{report['status']} stage={report['stage']}",
                            flush=True,
                        )
                        if reconstruction is None and not args.separate_build:
                            submit_next()
                        elif reconstruction is None and report["status"] != "PASSED":
                            submit_next()
                    atomic_json(
                        directory / "status.json",
                        {
                            "state": "RUNNING",
                            "stage": "DIAGNOSIS",
                            "updated_at": now(),
                            "completed": len(reports),
                            "samples": len(samples),
                            "elapsed_seconds": time.monotonic() - started,
                            "active": [sample["sample_id"] for sample, _, _ in pending.values()],
                        },
                    )
            summary = {
                "manifest_id": manifest["manifest_id"],
                "finished_at": now(),
                "samples": len(samples),
                "passed": sum(r["status"] == "PASSED" for r in reports),
                "failed": sum(r["status"] != "PASSED" for r in reports),
                "unresolved_rechecked": len(unresolved),
                "unresolved_now_passed": sum(
                    r["status"] == "PASSED" and r["role"] == "unresolved" for r in reports
                ),
                "results_sha256": file_sha256(directory / "results.jsonl"),
                "elapsed_seconds": time.monotonic() - started,
                "partial_scope": True,
                "model_generation_quality_evaluated": False,
                "runtime_correctness_checked": False,
                "latency_measured": False,
            }
            for path, expected in code.items():
                if file_sha256(path) != expected:
                    raise ValueError(f"Diagnostic code changed during execution: {path}")
            verify_compatible_inputs(source)
            atomic_json(directory / "summary.json", summary)
            atomic_json(
                directory / "status.json",
                {
                    "state": "SUCCEEDED" if not summary["failed"] else "FAILED",
                    "stage": "COMPLETE",
                    "exit_code": int(bool(summary["failed"])),
                    **summary,
                },
            )
            return int(bool(summary["failed"]))
        except BaseException as error:
            atomic_json(
                directory / "status.json",
                {
                    "state": "FAILED",
                    "error_message": str(error),
                    "updated_at": now(),
                    "exit_code": 1,
                },
            )
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--sample-timeout", type=int, default=120)
    parser.add_argument("--n-parallel", type=int, default=2)
    parser.add_argument("--separate-build", action="store_true")
    parser.add_argument("--build-timeout", type=int, default=60)
    parser.add_argument("--only-sample-id", action="append", default=[])
    arguments = parser.parse_args()
    if arguments.sample_timeout <= 0 or arguments.n_parallel <= 0 or arguments.build_timeout <= 0:
        parser.error("Timeout and concurrency must be positive")
    raise SystemExit(run(arguments))
