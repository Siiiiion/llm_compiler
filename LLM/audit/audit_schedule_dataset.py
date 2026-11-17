#!/usr/bin/env python3
"""Audit existing TVM schedules without modifying sources; resume with the same command.

Use --background for a detached session. Build results and source identities are
fsynced per batch. A restart reuses only results from the identical manifest.
"""

from LLM.core.layout import code_dependencies, source_path, module_command, project_pythonpath


from LLM.core.runtime import now


import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path

from LLM.core.schedule_record_utils import (
    AUDIT_SCHEMA,
    append_journal,
    digest,
    input_hash,
    read_journal,
    workload_key,
)
from LLM.core.io import atomic_json, file_sha256


def source_snapshot(path):
    path = Path(path).resolve()
    stat = path.stat()
    result = {"path": str(path), "bytes": stat.st_size, "sha256": file_sha256(path)}
    if path.stat().st_mtime_ns != stat.st_mtime_ns:
        raise ValueError(f"Source changed during hashing: {path}")
    return result


def scan_raw_splits(directory):
    """Read raw JSONL incrementally, retaining only the workload/source split membership."""
    sources, owners, snapshots = {}, {}, {}
    for split in ("train", "validation"):
        path = Path(directory) / f"{split}_raw.json"
        before = path.stat()
        hasher, count = hashlib.sha256(), 0
        with path.open("rb") as stream:
            for line in stream:
                hasher.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                key = workload_key(row["workload_key"])
                source = row["source_file"]
                if Path(source).name != source or not source.endswith(".json"):
                    raise ValueError(f"Invalid source_file in raw: {source}")
                membership = {"split": split, "workload_key": key}
                if source in sources and sources[source] != membership:
                    raise ValueError(f"Source split/workload conflict: {source}")
                if key in owners and owners[key] != split:
                    raise ValueError(f"Workload leaks across train/validation: {key}")
                sources[source], owners[key] = membership, split
                count += 1
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Raw split changed during scan: {path}")
        snapshots[split] = {
            "path": str(path.resolve()),
            "sha256": hasher.hexdigest(),
            "records": count,
        }
        print(f"[split] {split}: {count} raw rows; {len(sources)} source files so far", flush=True)
    return sources, snapshots


def discover_sources(args, hold_out_keys):
    directory = Path(args.source_dir).resolve()
    sources, raw_snapshots = {}, {}
    if args.split_from_raw:
        sources, raw_snapshots = scan_raw_splits(args.split_from_raw)
        leaks = [name for name, info in sources.items() if info["workload_key"] in hold_out_keys]
        if leaks:
            raise ValueError(f"Raw split contains hold-out workloads: {leaks[:3]}")
        paths = [directory / name for name in sorted(sources)]
    elif args.file_list:
        names = json.loads(Path(args.file_list).read_text())
        if not isinstance(names, list) or any(Path(name).name != name for name in names):
            raise ValueError("--file-list must contain a JSON array of source basenames")
        paths = [directory / name for name in sorted(set(names))]
    else:
        paths = sorted(directory.glob("*.json"))
    if args.max_files:
        if sources:
            # A small pilot must exercise both frozen splits, even with sorted file names.
            train = [path for path in paths if sources[path.name]["split"] == "train"]
            valid = [path for path in paths if sources[path.name]["split"] == "validation"]
            ordered = []
            for index in range(max(len(train), len(valid))):
                ordered.extend(part[index] for part in (train, valid) if index < len(part))
            paths = ordered[: args.max_files]
        else:
            paths = paths[: args.max_files]
    if not paths:
        raise ValueError("No source files selected")
    snapshots = []
    for path in paths:
        snapshot = source_snapshot(path)
        snapshot.update(sources.get(path.name, {"split": "audit_only", "workload_key": None}))
        snapshot["name"] = path.name
        snapshot["shard"] = digest(path.name)[:24]
        snapshots.append(snapshot)
    return snapshots, raw_snapshots


def make_manifest(args):
    from LLM.audit.audit_tvm_backend import compiler_fingerprint, register_tasks

    register_tasks(args.registry)
    hold_out_keys = set()
    if args.split_from_raw:
        import pickle
        import tvm
        from LLM.core.common import hold_out_task_files, register_data_path

        register_data_path(args.target, network_info_folder=str(Path(args.registry).parent))
        for path in hold_out_task_files(tvm.target.Target(args.target)).values():
            with open(path, "rb") as stream:
                tasks, _ = pickle.load(stream)
            hold_out_keys.update(workload_key(task.workload_key) for task in tasks)
    sources, raw_snapshots = discover_sources(args, hold_out_keys)
    code_paths = [
        Path(__file__),
        source_path("audit_tvm_backend.py"),
        source_path("audit_persistent_builder.py"),
        source_path("audit_reuse.py"),
        source_path("schedule_record_utils.py"),
        source_path("make_dataset.py"),
        source_path("make_dataset_utils.py"),
        source_path("prepare_audited_clm.py"),
    ]
    code_paths = code_dependencies(code_paths)
    manifest = {
        "schema": AUDIT_SCHEMA,
        "source_dir": str(Path(args.source_dir).resolve()),
        "registry": source_snapshot(args.registry),
        "compiler": compiler_fingerprint(),
        "code": {path.name: file_sha256(path) for path in code_paths},
        "hold_out_keys": sorted(hold_out_keys),
        "raw_splits": raw_snapshots,
        "sources": sources,
        "settings": {
            "expected_arch": args.expected_arch,
            "build_timeout": args.build_timeout,
            "n_parallel": args.n_parallel,
            "workload_parallel": args.workload_parallel,
            "build_engine": args.build_engine,
            "batch_size": args.batch_size,
            "max_retries": args.max_retries,
            "max_records_per_file": args.max_records_per_file,
            "partial_scope": bool(
                args.max_files or args.max_records_per_file or not args.split_from_raw
            ),
            "validation": "original_and_training_normalized_state_build_export",
            "runtime_correctness_checked": False,
        },
    }
    manifest["compiler_fingerprint"] = digest(manifest["compiler"])
    if args.reuse_audit_dir:
        from LLM.audit.audit_reuse import compatible_origin

        origin = compatible_origin(args.reuse_audit_dir, manifest)
        manifest["reused_audit"] = {
            "path": str(Path(args.reuse_audit_dir).resolve()),
            "manifest_id": origin["manifest_id"],
        }
    return manifest


def audit_source(source, manifest, run_dir, backend, args):
    path = Path(source["path"])
    shard = run_dir / "shards" / source["shard"]
    shard.mkdir(parents=True, exist_ok=True)
    journal, done = shard / "events.jsonl", shard / "done.json"
    output = run_dir / "records" / source["name"]
    if file_sha256(path) != source["sha256"]:
        raise ValueError(f"Source differs from frozen manifest: {path}")
    if done.exists():
        summary = json.loads(done.read_text())
        if (
            summary["manifest_id"] != manifest["manifest_id"]
            or file_sha256(journal) != summary["journal_sha256"]
            or file_sha256(output) != summary["output_sha256"]
        ):
            raise ValueError(f"Completed shard was changed: {shard}")
        return summary
    history = read_journal(journal, repair=True)
    cache = {}
    for row in history:
        if row["manifest_id"] != manifest["manifest_id"]:
            raise ValueError(f"Journal belongs to a different audit: {journal}")
        cache[row["input_hash"]] = row
    inputs, parse_errors = [], []
    with path.open() as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if args.max_records_per_file and len(inputs) >= args.max_records_per_file:
                break
            try:
                record = json.loads(line)
                key = input_hash(record)
                if (
                    source["workload_key"]
                    and workload_key(record["i"][0][0]) != source["workload_key"]
                ):
                    raise ValueError(
                        "Source workload differs from frozen train/validation membership"
                    )
                inputs.append((line_no, key, record, line.rstrip("\n")))
            except (ValueError, KeyError, TypeError, IndexError):
                key = hashlib.sha256(line.encode()).hexdigest()
                inputs.append((line_no, key, None, line.rstrip("\n")))
                if key not in cache:
                    parse_errors.append(
                        {
                            "input_hash": key,
                            "source_line": line_no,
                            "parse_status": "failed",
                            "build_status": "parse_error",
                            "error_no": None,
                            "error_message": traceback.format_exc()[-16000:],
                            "state_restore_status": "not_checked",
                            "apply_status": "not_checked",
                            "training_state_status": "not_checked",
                            "duration": 0,
                            "retry_count": 0,
                        }
                    )
    identity = {
        "source_file": source["path"],
        "source_sha256": source["sha256"],
        "manifest_id": manifest["manifest_id"],
        "compiler_fingerprint": manifest["compiler_fingerprint"],
    }

    def persist(rows):
        for row in rows:
            row.update(identity)
            row["audit_key"] = digest([row["input_hash"], manifest["compiler_fingerprint"]])
        append_journal(journal, rows)
        cache.update((row["input_hash"], row) for row in rows)

    persist(parse_errors)
    unique = {}
    for line_no, key, record, _ in inputs:
        if key not in cache and key not in unique:
            unique[key] = (line_no, record)
    pending = list(unique)
    for start in range(0, len(pending), args.batch_size):
        batch_started = time.monotonic()
        batch_keys = pending[start : start + args.batch_size]
        records = [unique[key][1] for key in batch_keys]
        streaming = getattr(backend, "build_engine", None) == "streaming"
        reports = (
            backend.audit_with_retries(records, args.build_timeout, args.max_retries)
            if streaming
            else backend.audit(records, args.build_timeout)
        )
        if len(reports) != len(records):
            raise RuntimeError("Auditor returned an incomplete batch")
        for report in reports:
            report.setdefault("retry_count", 0)
        for retry in range(1, (0 if streaming else args.max_retries) + 1):
            indices = [
                index
                for index, report in enumerate(reports)
                if report["build_status"] in ("timeout", "environment_error")
            ]
            if not indices:
                break
            retried = backend.audit([records[index] for index in indices], args.build_timeout * 2)
            for index, replacement in zip(indices, retried):
                previous = reports[index]
                replacement["duration"] += previous["duration"]
                replacement["prior_attempts"] = previous.get("prior_attempts", []) + [
                    {
                        "build_status": previous["build_status"],
                        "error_no": previous["error_no"],
                        "error_message": previous["error_message"],
                    }
                ]
                replacement["retry_count"] = retry
                reports[index] = replacement
        rows = []
        for key, record, report in zip(batch_keys, records, reports):
            report.update(
                input_hash=key,
                source_line=unique[key][0],
                parse_status="passed",
                workload_key=workload_key(record["i"][0][0]),
                target=record["i"][0][1],
                target_host=record["i"][0][3] if len(record["i"][0]) > 3 else None,
            )
            rows.append(report)
        persist(rows)
        batch_seconds = time.monotonic() - batch_started
        counts = Counter(row["build_status"] for row in cache.values())
        progress = {
            "updated_at": now(),
            "source_file": source["name"],
            "shard_unique_done": len(cache),
            "shard_unique_total": len(cache) + len(pending) - start - len(batch_keys),
            "build_status_counts": dict(counts),
            "batch_records": len(batch_keys),
            "batch_seconds": batch_seconds,
            "batch_records_per_second": len(batch_keys) / max(batch_seconds, 0.001),
        }
        atomic_json(shard / "progress.json", progress)
        atomic_json(run_dir / "progress.json", progress)
        print(
            f"[batch] {source['name'][:42]} checked={len(cache)} statuses={dict(counts)}",
            flush=True,
        )

    if file_sha256(path) != source["sha256"]:
        raise ValueError(f"Source changed during audit: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".pending")
    accepted = 0
    with temporary.open("w") as stream:
        for _, key, record, original_line in inputs:
            if record is not None and cache[key]["build_status"] == "passed":
                stream.write(original_line + "\n")
                accepted += 1
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    summary = {
        **identity,
        "split": source["split"],
        "records": len(inputs),
        "unique_inputs": len(cache),
        "reused_unique_inputs": sum("reused_from" in row for row in cache.values()),
        "passed_records": accepted,
        "build_status_counts": dict(Counter(cache[key]["build_status"] for _, key, _, _ in inputs)),
        "output_sha256": file_sha256(output),
        "journal_sha256": file_sha256(journal),
        "finished_at": now(),
    }
    atomic_json(done, summary)
    return summary


def run(args):
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(run_dir / "worker.json", {"pid": os.getpid(), "started_at": now()})
        atomic_json(run_dir / "status.json", {"state": "INITIALIZING", "updated_at": now()})
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(name, "1")
        temp_dir = run_dir / "build_tmp"
        temp_dir.mkdir(exist_ok=True)
        os.environ["TMPDIR"] = str(temp_dir)
        tempfile.tempdir = str(temp_dir)
        started = time.monotonic()
        try:
            from LLM.audit.audit_tvm_backend import TVMAuditor

            manifest = make_manifest(args)
            manifest["compiler_fingerprint"] = digest(manifest["compiler"])
            manifest["manifest_id"] = digest(manifest)
            manifest_path = run_dir / "manifest.json"
            if manifest_path.exists():
                if json.loads(manifest_path.read_text()) != manifest:
                    raise ValueError(
                        "Inputs, split, settings, code or toolchain changed; use a new run directory"
                    )
            else:
                atomic_json(manifest_path, manifest)
            if args.reuse_audit_dir:
                from LLM.audit.audit_reuse import reuse_results

                report = reuse_results(args.reuse_audit_dir, run_dir, manifest)
                print(
                    f"[reuse] {report['newly_imported_proofs']} durable proofs imported", flush=True
                )
            backends = [
                TVMAuditor(
                    args.registry,
                    args.build_timeout,
                    args.n_parallel,
                    temp_dir if args.workload_parallel == 1 else temp_dir / f"worker_{index}",
                    args.expected_arch,
                    build_engine=args.build_engine,
                )
                for index in range(args.workload_parallel)
            ]
            summaries, summary_lock = [], threading.Lock()
            pending, stop_requested = queue.Queue(), threading.Event()
            for source in manifest["sources"]:
                pending.put(source)
            atomic_json(run_dir / "status.json", {"state": "RUNNING", "updated_at": now()})

            def worker(backend):
                try:
                    while not stop_requested.is_set():
                        try:
                            source = pending.get_nowait()
                        except queue.Empty:
                            return
                        summary = audit_source(source, manifest, run_dir, backend, args)
                        with summary_lock:
                            summaries.append(summary)
                            totals = Counter()
                            for item in summaries:
                                totals.update(item["build_status_counts"])
                            atomic_json(
                                run_dir / "summary.json",
                                {
                                    "manifest_id": manifest["manifest_id"],
                                    "completed_files": len(summaries),
                                    "total_files": len(manifest["sources"]),
                                    "records": sum(item["records"] for item in summaries),
                                    "passed_records": sum(
                                        item["passed_records"] for item in summaries
                                    ),
                                    "reused_unique_inputs": sum(
                                        item["reused_unique_inputs"] for item in summaries
                                    ),
                                    "build_status_counts": dict(totals),
                                    "elapsed_seconds": time.monotonic() - started,
                                    "partial_scope": manifest["settings"]["partial_scope"],
                                    "updated_at": now(),
                                },
                            )
                            print(
                                f"[file] {len(summaries)}/{len(manifest['sources'])} "
                                f"passed={summary['passed_records']}/{summary['records']}",
                                flush=True,
                            )
                except BaseException:
                    stop_requested.set()
                    raise

            with ThreadPoolExecutor(max_workers=args.workload_parallel) as executor:
                futures = [executor.submit(worker, backend) for backend in backends]
                try:
                    for future in futures:
                        future.result()
                except BaseException:
                    stop_requested.set()
                    raise
            atomic_json(
                run_dir / "status.json", {"state": "SUCCEEDED", "exit_code": 0, "updated_at": now()}
            )
            return 0
        except BaseException as error:
            atomic_json(
                run_dir / "status.json",
                {
                    "state": "FAILED",
                    "exit_code": 1,
                    "error": str(error),
                    "updated_at": now(),
                },
            )
            raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--registry", required=True, help="all_tasks.pkl for this source/target")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--target", default="cuda -model=4090", help="Used only for hold-out registry lookup"
    )
    parser.add_argument("--expected-arch", default="sm_89")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--split-from-raw", help="Directory with original train_raw.json/validation_raw.json"
    )
    group.add_argument(
        "--file-list", help="JSON array of source basenames, for audit-only experiments"
    )
    group.add_argument(
        "--all-sources", action="store_true", help="Audit only; not a training split"
    )
    parser.add_argument("--build-timeout", type=int, default=30)
    parser.add_argument("--n-parallel", type=int, default=4)
    parser.add_argument("--workload-parallel", type=int, default=1)
    parser.add_argument(
        "--build-engine", choices=("local", "persistent", "streaming"), default="streaming"
    )
    parser.add_argument(
        "--reuse-audit-dir",
        help="Reuse durable proofs from a stopped compatible audit, preserving their origin",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--max-records-per-file", type=int)
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args(argv)
    for name in (
        "build_timeout",
        "n_parallel",
        "workload_parallel",
        "batch_size",
        "max_files",
        "max_records_per_file",
    ):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")
    if args.workload_parallel > 1 and args.build_engine == "local":
        parser.error("--workload-parallel > 1 requires persistent or streaming builders")
    source, output = Path(args.source_dir).resolve(), Path(args.run_dir).resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error("--run-dir must be separate from the original source directory")
    return args


def main():
    args = parse_args()
    if args.background:
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "launch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (run_dir / "run.lock").open("a") as worker_lock:
                fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            command = [
                *module_command("audit_schedule_dataset.py"),
            ] + [arg for arg in sys.argv[1:] if arg != "--background"]
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            # PopenPool workers must also be able to import the audit helper module.
            env["PYTHONPATH"] = project_pythonpath(env.get("PYTHONPATH", ""))
            with (run_dir / "audit.log").open("ab") as log:
                worker = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                    env=env,
                )
            atomic_json(
                run_dir / "launch.json",
                {"pid": worker.pid, "command": command, "started_at": now()},
            )
            # Keep the launch lock until the child owns run.lock, preventing double submission.
            for _ in range(100):
                identity_path = run_dir / "worker.json"
                if (
                    identity_path.exists()
                    and json.loads(identity_path.read_text())["pid"] == worker.pid
                ):
                    break
                if worker.poll() is not None:
                    raise RuntimeError(
                        f"Audit worker exited during launch; inspect {run_dir / 'audit.log'}"
                    )
                time.sleep(0.1)
            else:
                raise RuntimeError(
                    f"Worker startup is pending; inspect {run_dir / 'audit.log'} before resubmitting"
                )
            print(f"Submitted PID {worker.pid}; status: {run_dir / 'status.json'}", flush=True)
        return 0

    def stop(signum, _frame):
        raise InterruptedError(f"Received signal {signum}; completed batches can be resumed")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
