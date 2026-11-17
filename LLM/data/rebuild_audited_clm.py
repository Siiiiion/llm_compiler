#!/usr/bin/env python3
"""Export audited raw, encode complete CLM data, verify, and publish in a detached worker."""

from LLM.core.layout import code_dependencies, source_path, module_command, project_pythonpath


from LLM.core.runtime import now


import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from LLM.core.io import atomic_json, file_sha256


def snapshot_inputs(args):
    from LLM.audit.audit_tvm_backend import compiler_fingerprint
    from LLM.data.prepare_audited_clm import load_manifest

    audit = load_manifest(args.audit_dir)
    compiler = compiler_fingerprint()
    if compiler != audit["compiler"]:
        changed = [key for key in compiler if compiler[key] != audit["compiler"].get(key)]
        raise ValueError(f"Compiler environment differs from completed audit: {changed}")
    if file_sha256(audit["registry"]["path"]) != audit["registry"]["sha256"]:
        raise ValueError("Workload registry differs from completed audit")
    for path in code_dependencies(
        source_path(name)
        for name in [
            "schedule_record_utils.py",
            "make_dataset.py",
            "make_dataset_utils.py",
            "prepare_audited_clm.py",
        ]
    ):
        name = path.name
        if file_sha256(path) != audit["code"].get(name):
            raise ValueError(f"Audit formatter changed: {name}")
    code_names = [
        "rebuild_audited_clm.py",
        "verify_audited_clm.py",
        "prepare_audited_clm.py",
        "rebuild_clm_dataset.py",
        "schedule_record_utils.py",
        "make_dataset.py",
        "make_dataset_utils.py",
        "tokenization_protocol.py",
        "audit_tvm_backend.py",
        "common.py",
    ]
    tokenizer_dir = Path(args.tokenizer_path)
    tokenizer_files = sorted(
        path for path in tokenizer_dir.iterdir() if path.suffix in (".json", ".txt", ".model")
    )
    return {
        "created_at": now(),
        "audit_dir": str(Path(args.audit_dir).resolve()),
        "audit_manifest_id": audit["manifest_id"],
        "audit_manifest_sha256": file_sha256(Path(args.audit_dir) / "manifest.json"),
        "compiler_fingerprint": audit["compiler_fingerprint"],
        "tokenizer_path": str(tokenizer_dir.resolve()),
        "max_length": args.max_length,
        "overlength_policy": "error",
        "code": {
            str(path): file_sha256(path)
            for path in code_dependencies(source_path(name) for name in code_names)
        },
        "tokenizer": {str(path.resolve()): file_sha256(path) for path in tokenizer_files},
        "runtime_correctness_checked": False,
        "generation_parser_roundtrip_checked": False,
    }


def verify_snapshot(snapshot):
    for group in ("code", "tokenizer"):
        for path, expected in snapshot[group].items():
            if file_sha256(path) != expected:
                raise ValueError(f"Frozen {group} input changed: {path}")
    if (
        file_sha256(Path(snapshot["audit_dir"]) / "manifest.json")
        != snapshot["audit_manifest_sha256"]
    ):
        raise ValueError("Audit manifest changed during rebuild")


def reuse_completed_stages(source_dir, run_dir, snapshot):
    """Copy completed export/encoding stages from a failed verification run only."""
    source_dir = Path(source_dir).resolve()
    with (source_dir / "run.lock").open("a") as source_lock:
        fcntl.flock(source_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = json.loads((source_dir / "status.json").read_text())
        if (
            status["state"] != "FAILED"
            or status["stage"] != "VERIFYING"
            or not {"RAW_EXPORT", "ARROW_ENCODING"}.issubset(status["stage_seconds"])
        ):
            raise ValueError("Reuse requires completed raw/Arrow stages and failed verification")
        previous = json.loads((source_dir / "manifest.json").read_text())
        for key in (
            "audit_dir",
            "audit_manifest_id",
            "audit_manifest_sha256",
            "compiler_fingerprint",
            "tokenizer_path",
            "tokenizer",
            "max_length",
            "overlength_policy",
        ):
            if previous[key] != snapshot[key]:
                raise ValueError(f"Completed-stage inputs differ: {key}")
        if set(previous["code"]) != set(snapshot["code"]):
            raise ValueError("Completed-stage code inventory differs")
        changed_code = []
        for path, expected in previous["code"].items():
            name = Path(path).name
            if file_sha256(source_dir / "frozen_code" / name) != expected:
                raise ValueError(f"Previous frozen code changed: {name}")
            if snapshot["code"][path] != expected:
                if name not in {"rebuild_audited_clm.py", "verify_audited_clm.py"}:
                    raise ValueError(f"Export/encoding code differs: {name}")
                changed_code.append(name)
        for name in ("raw", "dataset.pending"):
            if not (source_dir / name).is_dir():
                raise ValueError(f"Missing completed-stage artifact: {name}")
        reuse = {
            "source_run_dir": str(source_dir),
            "source_manifest_sha256": file_sha256(source_dir / "manifest.json"),
            "source_status_sha256": file_sha256(source_dir / "status.json"),
            "source_stage_seconds": status["stage_seconds"],
            "changed_orchestration_or_verification_code": sorted(changed_code),
            "copied_at": now(),
            "verification_required_before_publish": True,
        }
        # Keep the failed run intact. Tokenization caches are not needed to load
        # the saved Arrow dataset and are intentionally omitted from the copy.
        shutil.copytree(source_dir / "raw", run_dir / "raw")
        shutil.copytree(
            source_dir / "dataset.pending",
            run_dir / "dataset.pending",
            ignore=shutil.ignore_patterns(".tokenize_cache"),
        )
        atomic_json(run_dir / "reuse.json", reuse)
        return reuse


def run(args):
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run_dir / "status.json").exists():
            raise ValueError("This rebuild was already started; choose a new run directory")
        atomic_json(run_dir / "worker.json", {"pid": os.getpid(), "started_at": now()})
        started, durations = time.monotonic(), {}
        stage = "PREFLIGHT"

        def update(state="RUNNING", **extra):
            atomic_json(
                run_dir / "status.json",
                {
                    "state": state,
                    "stage": stage,
                    "updated_at": now(),
                    "elapsed_seconds": time.monotonic() - started,
                    "stage_seconds": durations,
                    **extra,
                },
            )

        update()
        try:
            snapshot = snapshot_inputs(args)
            atomic_json(run_dir / "manifest.json", snapshot)
            (run_dir / "frozen_code").mkdir()
            for path in snapshot["code"]:
                shutil.copy2(path, run_dir / "frozen_code" / Path(path).name)
            raw, pending = run_dir / "raw", run_dir / "dataset.pending"
            commands = [
                (
                    "RAW_EXPORT",
                    [
                        *module_command("prepare_audited_clm.py"),
                        "--audit-dir",
                        str(Path(args.audit_dir).resolve()),
                        "--output-dir",
                        str(raw),
                    ],
                ),
                (
                    "ARROW_ENCODING",
                    [
                        *module_command("rebuild_clm_dataset.py"),
                        "--train-file",
                        str(raw / "train_raw.json"),
                        "--validation-file",
                        str(raw / "validation_raw.json"),
                        "--build-audit-manifest",
                        str(raw / "build_audit_manifest.json"),
                        "--tokenizer-path",
                        str(Path(args.tokenizer_path).resolve()),
                        "--output-dir",
                        str(pending),
                        "--max-length",
                        str(args.max_length),
                        "--overlength-policy",
                        "error",
                    ],
                ),
            ]
            if getattr(args, "reuse_run_dir", None):
                stage = "REUSE_COMPLETED_STAGES"
                update()
                before = time.monotonic()
                verify_snapshot(snapshot)
                reuse = reuse_completed_stages(args.reuse_run_dir, run_dir, snapshot)
                durations[stage] = time.monotonic() - before
                verify_snapshot(snapshot)
                commands = []
                print(f"[{now()}] Copied completed stages: {reuse}", flush=True)
            for stage, command in commands:
                verify_snapshot(snapshot)
                update()
                print(f"[{now()}] {stage}: starting", flush=True)
                before = time.monotonic()
                with subprocess.Popen(command, stdin=subprocess.DEVNULL) as child:
                    atomic_json(
                        run_dir / "child.json",
                        {"pid": child.pid, "stage": stage, "command": command, "started_at": now()},
                    )
                    try:
                        return_code = child.wait()
                    except BaseException:
                        child.terminate()
                        child.wait()
                        raise
                if return_code:
                    raise subprocess.CalledProcessError(return_code, command)
                durations[stage] = time.monotonic() - before
                verify_snapshot(snapshot)
                print(f"[{now()}] {stage}: finished in {durations[stage]:.1f}s", flush=True)
            stage = "VERIFYING"
            update()
            from LLM.data.verify_audited_clm import verify_dataset

            before = time.monotonic()
            verification = verify_dataset(raw, pending, args.tokenizer_path, args.max_length)
            durations[stage] = time.monotonic() - before
            verify_snapshot(snapshot)
            if (run_dir / "dataset").exists():
                raise ValueError("Published dataset already exists")
            pending.rename(run_dir / "dataset")
            stage = "COMPLETE"
            update(
                "SUCCEEDED",
                exit_code=0,
                dataset_dir=str(run_dir / "dataset"),
                raw_dir=str(raw),
                verification=verification,
            )
            print(f"[{now()}] SUCCEEDED: {run_dir / 'dataset'}", flush=True)
            return 0
        except BaseException as error:
            update("FAILED", exit_code=1, error=str(error))
            traceback.print_exc()
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--reuse-run-dir",
        help="Copy completed raw/Arrow from a failed verification run; verify again before publishing",
    )
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    if args.max_length <= 0:
        parser.error("--max-length must be positive")
    audit, run_dir = Path(args.audit_dir).resolve(), Path(args.run_dir).resolve()
    if audit == run_dir or audit in run_dir.parents or run_dir in audit.parents:
        parser.error("--run-dir must be separate from the completed audit")
    if args.reuse_run_dir:
        source_dir = Path(args.reuse_run_dir).resolve()
        if source_dir == run_dir or source_dir in run_dir.parents or run_dir in source_dir.parents:
            parser.error("--reuse-run-dir must be separate from the new run directory")
        if not (source_dir / "status.json").is_file():
            parser.error("--reuse-run-dir must be an existing rebuild run")
    if not Path(args.tokenizer_path).is_dir():
        parser.error("--tokenizer-path must be a local directory")
    if args.background:
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "launch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (run_dir / "status.json").exists():
                raise ValueError("Rebuild already started; inspect its status before submitting")
            env = dict(
                os.environ,
                PYTHONDONTWRITEBYTECODE="1",
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                RAYON_NUM_THREADS="8",
                TOKENIZERS_PARALLELISM="true",
            )
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                env[name] = "1"
            env["PYTHONPATH"] = project_pythonpath(env.get("PYTHONPATH", ""))
            command = [
                *module_command("rebuild_audited_clm.py"),
            ] + [arg for arg in sys.argv[1:] if arg != "--background"]
            with (run_dir / "rebuild.log").open("ab") as log:
                child = subprocess.Popen(
                    command,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            atomic_json(
                run_dir / "launch.json", {"pid": child.pid, "command": command, "started_at": now()}
            )
            for _ in range(100):
                worker = run_dir / "worker.json"
                if worker.exists() and json.loads(worker.read_text())["pid"] == child.pid:
                    break
                if child.poll() is not None:
                    raise RuntimeError(f"Rebuild worker exited; inspect {run_dir / 'rebuild.log'}")
                time.sleep(0.1)
            else:
                raise RuntimeError("Worker startup pending; inspect the run before resubmitting")
            print(f"Submitted PID {child.pid}; status: {run_dir / 'status.json'}", flush=True)
            return 0

    def stop(signum, _frame):
        raise InterruptedError(f"Received signal {signum}; rebuild in a new run directory")

    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, stop)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
