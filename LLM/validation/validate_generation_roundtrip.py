#!/usr/bin/env python3
"""Validate sampled audited schedules through production tokenization, parsing and builds.

Use --background for terminal-independent execution. Completed sample results are
fsynced and reused only with an identical frozen manifest. No model or GPU run is started.
"""

from LLM.core.layout import code_dependencies, source_path, module_command, project_pythonpath


from LLM.core.runtime import bounded_pool, now
from LLM.data.sampling import length_bucket, operator_family, representative_indices


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
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from LLM.core.schedule_record_utils import (
    append_journal,
    digest,
    input_hash,
    read_journal,
    workload_key,
)
from LLM.core.io import atomic_json, file_sha256


def input_snapshot(args):
    from LLM.data.rebuild_audited_clm import snapshot_inputs
    from LLM.data.rebuild_clm_dataset import verify_build_audit

    rebuild = Path(args.rebuild_dir).resolve()
    state = json.loads((rebuild / "status.json").read_text())
    if state["state"] != "SUCCEEDED" or state["exit_code"] != 0:
        raise ValueError("Raw/Arrow rebuilding must have succeeded")
    verification = json.loads((rebuild / "dataset/verification.json").read_text())
    if verification != state["verification"] or verification["status"] != "VERIFIED":
        raise ValueError("Published Arrow verification differs from successful rebuild")
    report_path = rebuild / "raw/build_audit_manifest.json"
    report = verify_build_audit(
        report_path, rebuild / "raw/train_raw.json", rebuild / "raw/validation_raw.json"
    )
    previous = json.loads((rebuild / "manifest.json").read_text())
    snapshot = snapshot_inputs(
        SimpleNamespace(
            audit_dir=report["audit_dir"],
            tokenizer_path=previous["tokenizer_path"],
            max_length=previous["max_length"],
        )
    )
    snapshot.pop("created_at")
    for key in ("audit_manifest_id", "compiler_fingerprint", "tokenizer", "max_length"):
        if snapshot[key] != previous[key]:
            raise ValueError(f"Published dataset inputs changed: {key}")
    for path in code_dependencies(
        source_path(name)
        for name in (
            "validate_generation_roundtrip.py",
            "generation_roundtrip_worker.py",
            "gen_state.py",
        )
    ):
        snapshot["code"][str(path)] = file_sha256(path)
    snapshot["rebuild_dir"] = str(rebuild)
    snapshot["raw_hashes"] = report["outputs"]
    snapshot["dataset_files"] = {
        str(path): file_sha256(path)
        for path in sorted((rebuild / "dataset").rglob("*"))
        if path.is_file()
    }
    return snapshot


def verify_inputs(snapshot):
    from LLM.data.rebuild_audited_clm import verify_snapshot

    verify_snapshot(snapshot)
    rebuild = Path(snapshot["rebuild_dir"])
    for split, expected in snapshot["raw_hashes"].items():
        if file_sha256(rebuild / f"raw/{split}_raw.json") != expected:
            raise ValueError(f"Raw input changed: {split}")
    for path, expected in snapshot["dataset_files"].items():
        if file_sha256(path) != expected:
            raise ValueError(f"Published dataset changed: {path}")


def select_samples(snapshot, run_dir, max_workloads=0):
    from datasets import load_from_disk
    from LLM.data.prepare_audited_clm import load_manifest, load_verified_source

    rebuild, audit_dir = Path(snapshot["rebuild_dir"]), Path(snapshot["audit_dir"])
    audit = load_manifest(audit_dir)
    datasets = load_from_disk(str(rebuild / "dataset"), keep_in_memory=False)
    sources = {source["name"]: source for source in audit["sources"]}
    owners, selected, metadata = {}, {}, {}
    for split in ("train", "validation"):
        lengths = datasets[split].data.column("length").to_pylist()
        groups = defaultdict(list)
        position, count = 0, 0
        with (rebuild / f"raw/{split}_raw.json").open("rb") as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                key = workload_key(row["workload_key"])
                source = sources[row["source_file"]]
                if source["split"] != split or source["workload_key"] != key:
                    raise ValueError("Raw source membership differs from frozen audit")
                if key in audit["hold_out_keys"] or owners.setdefault(key, split) != split:
                    raise ValueError("Hold-out leakage or train/validation overlap")
                if index >= len(lengths):
                    raise ValueError("Raw contains more samples than Arrow")
                groups[key].append((lengths[index], index, position))
                metadata.setdefault(
                    key,
                    {
                        "split": split,
                        "source_file": row["source_file"],
                        "operator_family": operator_family(row["text"], key),
                    },
                )
                position += len(line)
                count += 1
        if count != len(lengths):
            raise ValueError("Raw and Arrow counts differ")
        for key, entries in groups.items():
            selected[key] = representative_indices(entries)
        print(f"[{now()}] Selected {split}: {len(groups)} workloads", flush=True)
    if max_workloads:
        # A diagnostic pilot samples both splits, operator families and longest
        # sequences first. It is never reported as covering the full workload set.
        strata = defaultdict(list)
        for key, info in metadata.items():
            longest = max(v["length"] for v in selected[key].values())
            strata[(info["split"], info["operator_family"], length_bucket(longest))].append(key)
        order = []
        strata_order = sorted(
            strata,
            key=lambda value: (
                {"gt1024": 0, "513to1024": 1, "le512": 2}[value[2]],
                value[1],
            ),
        )
        while len(order) < max_workloads and any(strata.values()):
            for split in ("train", "validation"):
                choices = [key for key in strata_order if key[0] == split and strata[key]]
                if choices and len(order) < max_workloads:
                    order.append(strata[choices[0]].pop(0))
                    strata_order.append(strata_order.pop(strata_order.index(choices[0])))
        selected = {key: selected[key] for key in order}
    destination = run_dir / "samples.jsonl"
    coverage = Counter()
    sample_count = 0
    with destination.open("w") as output:
        for ordinal, (key, representatives) in enumerate(selected.items(), 1):
            info = metadata[key]
            source = sources[info["source_file"]]
            lines, _ = load_verified_source(audit_dir, audit, source)
            record_map = {input_hash(json.loads(line)): line for line in lines}
            with (rebuild / f"raw/{info['split']}_raw.json").open("rb") as raw:
                for index, item in representatives.items():
                    raw.seek(item["position"])
                    row = json.loads(raw.readline())
                    record = json.loads(record_map[row["source_input_hash"]])
                    arrow = datasets[info["split"]][index]
                    if row["build_audit_manifest_id"] != audit["manifest_id"]:
                        raise ValueError("Selected sample audit identity differs")
                    sample = {
                        **info,
                        "workload_key": key,
                        "row_index": index,
                        "source_input_hash": row["source_input_hash"],
                        "length": item["length"],
                        "length_bucket": length_bucket(item["length"]),
                        "roles": item["roles"],
                        "row": row,
                        "record": record,
                        "input_ids": arrow["input_ids"],
                        "ppt_end": arrow["ppt_end"],
                    }
                    sample["sample_id"] = digest(sample)
                    output.write(json.dumps(sample, allow_nan=False) + "\n")
                    coverage[(info["split"], info["operator_family"], sample["length_bucket"])] += 1
                    sample_count += 1
            if ordinal % 100 == 0:
                print(f"[{now()}] Verified sampling sources: {ordinal}/{len(selected)}", flush=True)
        output.flush()
        os.fsync(output.fileno())
    return {
        "policy": "shortest_median_longest_per_workload_by_complete_token_length",
        "samples": sample_count,
        "workloads": len(selected),
        "total_workloads": len(metadata),
        "partial_scope": bool(max_workloads),
        "strata": [
            dict(split=k[0], operator_family=k[1], length_bucket=k[2], samples=v)
            for k, v in sorted(coverage.items())
        ],
    }


def summarize(reports, selection):
    failures = Counter(report["stage"] for report in reports if report["status"] != "PASSED")
    strata = defaultdict(Counter)
    workloads = set()
    for report in reports:
        key = (report["split"], report["operator_family"], report["length_bucket"])
        strata[key][report["status"]] += 1
        workloads.add(report["workload_key"])
    passed = sum(report["status"] == "PASSED" for report in reports)
    complete = len(reports) == selection["samples"]
    return {
        "samples": selection["samples"],
        "completed": len(reports),
        "passed": passed,
        "failed": len(reports) - passed,
        "workloads": len(workloads),
        "selected_workloads": selection["workloads"],
        "partial_scope": selection["partial_scope"],
        "failure_stages": dict(failures),
        "all_selected_samples_passed": complete and not failures,
        "generation_parser_roundtrip_checked": complete and not failures,
        "full_workload_coverage": complete
        and not selection["partial_scope"]
        and len(workloads) == selection["total_workloads"],
        "all_dataset_records_roundtrip_checked": False,
        "model_generation_quality_evaluated": False,
        "runtime_correctness_checked": False,
        "latency_measured": False,
        "strata": [
            dict(split=k[0], operator_family=k[1], length_bucket=k[2], counts=dict(v))
            for k, v in sorted(strata.items())
        ],
    }


def run(args):
    from tvm.contrib.popen_pool import StatusKind
    from LLM.validation.generation_roundtrip_worker import check_sample, initialize

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started, stage, manifest, reports = time.monotonic(), "PREFLIGHT", None, []
        atomic_json(run_dir / "worker.json", {"pid": os.getpid(), "started_at": now()})

        def update(state="RUNNING", **extra):
            atomic_json(
                run_dir / "status.json",
                {
                    "state": state,
                    "stage": stage,
                    "pid": os.getpid(),
                    "updated_at": now(),
                    "elapsed_seconds": time.monotonic() - started,
                    "completed": len(reports),
                    **extra,
                },
            )

        update()
        try:
            snapshot = input_snapshot(args)
            settings = {
                "max_workloads": args.max_workloads,
                "n_parallel": args.n_parallel,
                "sample_timeout": args.sample_timeout,
                "max_retries": 1,
            }
            manifest_file = run_dir / "manifest.json"
            if manifest_file.exists():
                manifest = json.loads(manifest_file.read_text())
                identity = manifest.pop("manifest_id")
                if digest(manifest) != identity:
                    raise ValueError("Roundtrip manifest identity differs")
                manifest["manifest_id"] = identity
                if manifest["inputs"] != snapshot or manifest["settings"] != settings:
                    raise ValueError("Frozen inputs/settings changed; use a new run directory")
                if file_sha256(run_dir / "samples.jsonl") != manifest["samples_sha256"]:
                    raise ValueError("Frozen selected samples changed")
            else:
                stage = "SAMPLING"
                update()
                selection = select_samples(snapshot, run_dir, args.max_workloads)
                verify_inputs(snapshot)
                manifest = {
                    "inputs": snapshot,
                    "settings": settings,
                    "selection": selection,
                    "samples_sha256": file_sha256(run_dir / "samples.jsonl"),
                    "created_at": now(),
                }
                manifest["manifest_id"] = digest(manifest)
                atomic_json(manifest_file, manifest)
                (run_dir / "frozen_code").mkdir(exist_ok=True)
                for filename in snapshot["code"]:
                    shutil.copy2(filename, run_dir / "frozen_code" / Path(filename).name)
            samples = [json.loads(line) for line in (run_dir / "samples.jsonl").open()]
            owners = {sample["sample_id"]: sample for sample in samples}
            if len(owners) != len(samples):
                raise ValueError("Duplicate frozen sample identity")
            for result in read_journal(run_dir / "results.jsonl", repair=True):
                identifier = result.pop("result_id")
                if digest(result) != identifier or result["sample_id"] not in owners:
                    raise ValueError("Committed sample result changed")
                if result["manifest_id"] != manifest["manifest_id"]:
                    raise ValueError("Committed result belongs to another run identity")
                result["result_id"] = identifier
                reports.append(result)
            done = {report["sample_id"] for report in reports}
            if len(done) != len(reports):
                raise ValueError("Duplicate committed sample result")
            remaining = [sample for sample in samples if sample["sample_id"] not in done]
            stage = "ROUNDTRIP_AND_BUILD"
            update(summary=summarize(reports, manifest["selection"]))
            temp_dir = run_dir / "tmp"
            temp_dir.mkdir(exist_ok=True)
            audit = json.loads((Path(snapshot["audit_dir"]) / "manifest.json").read_text())
            with bounded_pool(
                args.n_parallel,
                args.sample_timeout,
                initialize,
                (audit["registry"]["path"], snapshot["tokenizer_path"], str(temp_dir)),
                maximum_process_uses=512,
            ) as pool:
                for start in range(0, len(remaining), args.n_parallel * 2):
                    batch = remaining[start : start + args.n_parallel * 2]
                    results = list(pool.map_with_error_catching(check_sample, batch))
                    committed = []
                    for sample, result in zip(batch, results):
                        attempts = []
                        for attempt in range(2):
                            if result.status == StatusKind.COMPLETE:
                                report = result.value
                            else:
                                report = {
                                    k: sample[k]
                                    for k in (
                                        "sample_id",
                                        "split",
                                        "row_index",
                                        "workload_key",
                                        "length",
                                        "length_bucket",
                                        "operator_family",
                                        "source_file",
                                        "roles",
                                    )
                                }
                                report.update(
                                    status="FAILED",
                                    stage=(
                                        "worker_timeout"
                                        if result.status == StatusKind.TIMEOUT
                                        else "worker_environment"
                                    ),
                                    error_message=str(result.value)[-16000:],
                                )
                            attempts.append(
                                {
                                    "status": report["status"],
                                    "stage": report["stage"],
                                    "error_message": report.get("error_message", ""),
                                }
                            )
                            if (
                                report["status"] == "PASSED"
                                or attempt
                                or report["stage"]
                                not in (
                                    "worker_timeout",
                                    "worker_environment",
                                    "build_export",
                                )
                            ):
                                break
                            pool._timeout = args.sample_timeout * 2
                            result = list(pool.map_with_error_catching(check_sample, [sample]))[0]
                            pool._timeout = args.sample_timeout
                        report.pop("result_id", None)
                        report.update(manifest_id=manifest["manifest_id"], attempts=attempts)
                        report["result_id"] = digest(report)
                        committed.append(report)
                    append_journal(run_dir / "results.jsonl", committed)
                    reports.extend(committed)
                    summary = summarize(reports, manifest["selection"])
                    atomic_json(run_dir / "summary.json", summary)
                    update(summary=summary)
                    print(
                        f"[{now()}] {len(reports)}/{len(samples)} samples; "
                        f"passed={summary['passed']} failed={summary['failed']}",
                        flush=True,
                    )
            stage = "VERIFYING_OUTPUTS"
            update()
            verify_inputs(snapshot)
            summary = summarize(reports, manifest["selection"])
            summary.update(
                manifest_id=manifest["manifest_id"],
                results_sha256=file_sha256(run_dir / "results.jsonl"),
                finished_at=now(),
            )
            atomic_json(run_dir / "summary.json", summary)
            succeeded = summary["all_selected_samples_passed"]
            stage = "COMPLETE"
            update(
                "SUCCEEDED" if succeeded else "FAILED",
                exit_code=0 if succeeded else 1,
                summary=summary,
            )
            return 0 if succeeded else 1
        except BaseException as error:
            update("FAILED", exit_code=1, error=str(error))
            traceback.print_exc()
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--n-parallel", type=int, default=8)
    parser.add_argument("--sample-timeout", type=int, default=60)
    parser.add_argument(
        "--max-workloads", type=int, default=0, help="Diagnostic partial scope only"
    )
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    if args.n_parallel <= 0 or args.sample_timeout <= 0 or args.max_workloads < 0:
        parser.error("Invalid concurrency/timeout/scope")
    source, destination = Path(args.rebuild_dir).resolve(), Path(args.run_dir).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        parser.error("Roundtrip output must be separate from the completed rebuild")
    if args.background:
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "launch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (destination / "run.lock").open("a") as worker_lock:
                fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                env = dict(
                    os.environ,
                    PYTHONDONTWRITEBYTECODE="1",
                    HF_HUB_OFFLINE="1",
                    TRANSFORMERS_OFFLINE="1",
                    RAYON_NUM_THREADS="1",
                    TOKENIZERS_PARALLELISM="false",
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                )
                env["PYTHONPATH"] = project_pythonpath(env.get("PYTHONPATH", ""))
                command = [
                    *module_command("validate_generation_roundtrip.py"),
                ] + [arg for arg in sys.argv[1:] if arg != "--background"]
                with (destination / "roundtrip.log").open("ab") as log:
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
                    destination / "launch.json",
                    {
                        "pid": child.pid,
                        "command": command,
                        "started_at": now(),
                    },
                )
            for _ in range(100):
                worker = destination / "worker.json"
                if worker.exists() and json.loads(worker.read_text())["pid"] == child.pid:
                    break
                if child.poll() is not None:
                    raise RuntimeError(f"Worker exited; inspect {destination / 'roundtrip.log'}")
                time.sleep(0.1)
            else:
                raise RuntimeError("Worker startup pending; inspect before resubmitting")
            print(f"Submitted PID {child.pid}; status: {destination / 'status.json'}", flush=True)
        return 0

    def stop(signum, _frame):
        raise InterruptedError(f"Received signal {signum}; committed samples can be resumed")

    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, stop)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
