#!/usr/bin/env python3
"""Create new raw CLM files from completed audit shards, preserving workload splits."""

from LLM.core.layout import code_dependencies, source_path


import argparse
import json
import os
from collections import Counter
from pathlib import Path

from LLM.core.schedule_record_utils import (
    AUDIT_SCHEMA,
    digest,
    input_hash,
    read_journal,
    workload_key,
)
from LLM.core.io import atomic_json, file_sha256


def load_manifest(audit_dir, allow_partial=False):
    audit_dir = Path(audit_dir).resolve()
    manifest = json.loads((audit_dir / "manifest.json").read_text())
    if manifest["schema"] != AUDIT_SCHEMA:
        raise ValueError("Unsupported audit schema")
    identifier = manifest.pop("manifest_id")
    if (
        digest(manifest) != identifier
        or digest(manifest["compiler"]) != manifest["compiler_fingerprint"]
    ):
        raise ValueError("Audit manifest identity mismatch")
    manifest["manifest_id"] = identifier
    status = json.loads((audit_dir / "status.json").read_text())
    if status["state"] != "SUCCEEDED":
        raise ValueError("Audit is not complete")
    if manifest["settings"]["partial_scope"] and not allow_partial:
        raise ValueError("Partial/audit-only data cannot be published as a full formal dataset")
    if not manifest["raw_splits"]:
        raise ValueError("Audit has no frozen train/validation workload split")
    return manifest


def load_verified_source(audit_dir, manifest, source):
    """Verify both the record bytes and their per-input build proof before formatting."""
    audit_dir = Path(audit_dir)
    shard = audit_dir / "shards" / source["shard"]
    done = json.loads((shard / "done.json").read_text())
    output = audit_dir / "records" / source["name"]
    journal = shard / "events.jsonl"
    if (
        done["manifest_id"] != manifest["manifest_id"]
        or done["source_sha256"] != source["sha256"]
        or done["output_sha256"] != file_sha256(output)
        or done["journal_sha256"] != file_sha256(journal)
    ):
        raise ValueError(f"Audit shard or records changed: {source['name']}")
    proofs = {}
    for proof in read_journal(journal):
        if (
            proof["manifest_id"] != manifest["manifest_id"]
            or proof["compiler_fingerprint"] != manifest["compiler_fingerprint"]
            or proof["audit_key"] != digest([proof["input_hash"], manifest["compiler_fingerprint"]])
        ):
            raise ValueError("Build proof belongs to a different compiler/input")
        proofs[proof["input_hash"]] = proof
    lines = output.read_text().splitlines()
    for line in lines:
        record = json.loads(line)
        key = input_hash(record)
        proof = proofs.get(key)
        if (
            not proof
            or proof["build_status"] != "passed"
            or proof["training_state_status"] != "passed"
        ):
            raise ValueError(f"Record has no passing build proof: {source['name']}")
        wk = workload_key(record["i"][0][0])
        if wk in manifest["hold_out_keys"] or wk != source["workload_key"]:
            raise ValueError("Hold-out leakage or source workload mismatch")
    if len(lines) != done["passed_records"]:
        raise ValueError("Filtered record count differs from the audit shard")
    return lines, proofs


def prepare_raw(audit_dir, output_dir, min_suffix_tokens=16, allow_partial=False):
    from LLM.data.make_dataset import for_gen

    audit_dir, output_dir = Path(audit_dir).resolve(), Path(output_dir).resolve()
    manifest = load_manifest(audit_dir, allow_partial)
    if (
        output_dir == audit_dir
        or audit_dir in output_dir.parents
        or output_dir in audit_dir.parents
    ):
        raise ValueError("Raw output must be separate from audit artifacts")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Raw output must be new or empty; existing datasets are never overwritten")
    output_dir.mkdir(parents=True, exist_ok=True)
    # DAGs must come from the same registered workload definitions as the audit.
    from LLM.audit.audit_tvm_backend import compiler_fingerprint, register_tasks

    if file_sha256(manifest["registry"]["path"]) != manifest["registry"]["sha256"]:
        raise ValueError("Task registry changed after the audit")
    current_compiler = compiler_fingerprint()
    for name in ("libtvm_sha256", "tvm_python_sha256"):
        if current_compiler[name] != manifest["compiler"][name]:
            raise ValueError("Loaded TVM differs from the audit environment")
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
        if file_sha256(path) != manifest["code"].get(name):
            raise ValueError(f"Training formatter changed after audit: {name}")
    register_tasks(manifest["registry"]["path"])
    counts, stats, missing = Counter(), Counter(), []
    for split in ("train", "validation"):
        pending = output_dir / f"{split}_raw.pending"
        with pending.open("w") as stream:
            for source in manifest["sources"]:
                if source["split"] != split:
                    continue
                lines, proofs = load_verified_source(audit_dir, manifest, source)
                rows, file_stats = for_gen(
                    lines, source["name"], min_suffix_tokens, audit_proofs=proofs
                )
                stats.update(file_stats)
                if not rows:
                    missing.append({"source_file": source["name"], "split": split})
                for row in rows:
                    row["build_audit_manifest_id"] = manifest["manifest_id"]
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    counts[split] += 1
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, output_dir / f"{split}_raw.json")
    report = {
        "audit_dir": str(audit_dir),
        "manifest_id": manifest["manifest_id"],
        "compiler_fingerprint": manifest["compiler_fingerprint"],
        "partial_scope": manifest["settings"]["partial_scope"],
        "counts": dict(counts),
        "filter_stats": dict(stats),
        "workloads_without_samples": missing,
        "runtime_correctness_checked": False,
        "generation_parser_roundtrip_checked": False,
        "outputs": {
            split: file_sha256(output_dir / f"{split}_raw.json")
            for split in ("train", "validation")
        },
    }
    atomic_json(output_dir / "build_audit_manifest.json", report)
    if not counts["train"] or not counts["validation"]:
        raise ValueError("Audit filtering left an empty split; inspect build_audit_manifest.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-suffix-tokens", type=int, default=16)
    parser.add_argument("--allow-partial", action="store_true", help="Diagnostic use only")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_raw(
                args.audit_dir, args.output_dir, args.min_suffix_tokens, args.allow_partial
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
