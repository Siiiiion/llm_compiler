"""Migrate durable audit results across execution-only tuning, preserving their origin."""

import fcntl
import json
from pathlib import Path

from LLM.core.layout import code_dependencies, source_path

from LLM.core.schedule_record_utils import append_journal, digest, read_journal
from LLM.core.io import atomic_json, file_sha256


def compatible_origin(origin_dir, manifest):
    origin_dir = Path(origin_dir).resolve()
    origin = json.loads((origin_dir / "manifest.json").read_text())
    content = {key: value for key, value in origin.items() if key != "manifest_id"}
    if digest(content) != origin["manifest_id"]:
        raise ValueError("Origin manifest identity mismatch")
    for key in (
        "schema",
        "source_dir",
        "registry",
        "compiler",
        "compiler_fingerprint",
        "hold_out_keys",
        "raw_splits",
        "sources",
    ):
        if origin[key] != manifest[key]:
            raise ValueError(f"Cannot reuse results after audit input/environment change: {key}")
    for path in code_dependencies(
        source_path(name)
        for name in (
            "schedule_record_utils.py",
            "make_dataset.py",
            "make_dataset_utils.py",
            "prepare_audited_clm.py",
        )
    ):
        name = path.name
        if origin["code"].get(name) != manifest["code"].get(name):
            raise ValueError(f"Cannot reuse results after formatter change: {name}")
    # Only execution width, batching and equivalent builder implementation may change.
    for key in (
        "expected_arch",
        "build_timeout",
        "max_retries",
        "max_records_per_file",
        "partial_scope",
        "validation",
        "runtime_correctness_checked",
    ):
        if origin["settings"][key] != manifest["settings"][key]:
            raise ValueError(f"Cannot reuse results after audit validation change: {key}")
    return origin


def reuse_results(origin_dir, run_dir, manifest):
    """Import committed records only; require the previous worker to have stopped."""
    origin_dir, run_dir = Path(origin_dir).resolve(), Path(run_dir).resolve()
    if origin_dir == run_dir:
        raise ValueError("Result reuse requires a new run directory")
    with (origin_dir / "run.lock").open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Origin audit is still running; stop it before migration") from error
        origin = compatible_origin(origin_dir, manifest)
        reused = 0
        for source in manifest["sources"]:
            journal = origin_dir / "shards" / source["shard"] / "events.jsonl"
            if not journal.exists():
                continue
            if file_sha256(source["path"]) != source["sha256"]:
                raise ValueError("Source changed after origin audit")
            done = journal.with_name("done.json")
            if done.exists():
                from LLM.data.prepare_audited_clm import load_verified_source

                load_verified_source(origin_dir, origin, source)
            target = run_dir / "shards" / source["shard"] / "events.jsonl"
            existing = read_journal(target, repair=True) if target.exists() else []
            if any(row["manifest_id"] != manifest["manifest_id"] for row in existing):
                raise ValueError("Destination journal belongs to another audit")
            known = {row["input_hash"] for row in existing}
            rows = []
            for row in read_journal(journal):
                if (
                    row["manifest_id"] != origin["manifest_id"]
                    or row["compiler_fingerprint"] != manifest["compiler_fingerprint"]
                    or row["source_file"] != source["path"]
                    or row["source_sha256"] != source["sha256"]
                    or row["audit_key"]
                    != digest([row["input_hash"], manifest["compiler_fingerprint"]])
                    or (
                        row["build_status"] == "passed" and row["training_state_status"] != "passed"
                    )
                ):
                    raise ValueError("Origin contains incompatible build proof")
                if row["input_hash"] in known:
                    continue
                row["reused_from"] = {
                    "audit_dir": str(origin_dir),
                    "manifest_id": origin["manifest_id"],
                    "audit_key": row["audit_key"],
                }
                row["manifest_id"] = manifest["manifest_id"]
                rows.append(row)
            if rows:
                target.parent.mkdir(parents=True, exist_ok=True)
                append_journal(target, rows)
                reused += len(rows)
        result = {
            "origin_dir": str(origin_dir),
            "origin_manifest_id": origin["manifest_id"],
            "manifest_id": manifest["manifest_id"],
            "newly_imported_proofs": reused,
        }
        atomic_json(run_dir / "reuse_report.json", result)
        return result
