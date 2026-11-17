"""Ensure multi-sketch selection measures structures and preserves audited validation provenance."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from LLM.validation.multisketch_selection import select_multisketch_suite, sketch_identity
from LLM.core.schedule_record_utils import (
    AUDIT_SCHEMA,
    atomic_json,
    digest,
    file_sha256,
    input_hash,
    workload_key,
)


def record(name="first", factor=2, structure=0):
    steps = [["SP", 0, 0, 16, [factor], 1]]
    if structure:
        steps.append(["AN", 0, 0, structure])
    steps.extend([["PPT"], ["SPC", 0, 0, 16, [factor], 1], ["TBS"], ["PRS", []]])
    return {
        "i": [[workload_key([name, [16]]), "cuda -arch=sm_89", [16]], [[], steps]],
        "r": [[0], 0],
        "v": "v0.6",
    }


def make_suite(tmp_path, source_split="validation", holdout=False):
    """Small, fully hashed audit with a real journal/proof chain and no TVM dependency."""
    audit, rebuild, roundtrip, anchor = [
        tmp_path / name for name in ("audit", "rebuild", "roundtrip", "anchor")
    ]
    for directory in (audit, rebuild / "raw", rebuild / "dataset", roundtrip, anchor):
        directory.mkdir(parents=True)
    registry = tmp_path / "registry.pkl"
    registry.write_bytes(b"unit-test-registry")
    source_records = [
        [
            record(factor=2),
            record(factor=4),
            record(structure=1),
            record(structure=2),
            record(structure=3),
            record(factor=8, structure=1),
        ],
        [record("second", factor=2), record("second", factor=4)],
    ]
    sources = []
    for index, records in enumerate(source_records):
        name = f"source{index}.json"
        path = tmp_path / name
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        sources.append(
            {
                "name": name,
                "path": str(path),
                "sha256": file_sha256(path),
                "shard": f"shard{index}",
                "split": source_split,
                "workload_key": workload_key(records[0]["i"][0][0]),
            }
        )
    compiler = {}
    audit_manifest = {
        "schema": AUDIT_SCHEMA,
        "compiler": compiler,
        "compiler_fingerprint": digest(compiler),
        "settings": {"partial_scope": False},
        "raw_splits": {"validation": {}},
        "sources": sources,
        "hold_out_keys": [sources[0]["workload_key"]] if holdout else [],
        "registry": {"path": str(registry), "sha256": file_sha256(registry)},
    }
    audit_manifest["manifest_id"] = digest(audit_manifest)
    atomic_json(audit / "manifest.json", audit_manifest)
    atomic_json(audit / "status.json", {"state": "SUCCEEDED"})
    raw_rows, anchors = [], []
    for index, (source, records) in enumerate(zip(sources, source_records)):
        shard = audit / "shards" / source["shard"]
        shard.mkdir(parents=True)
        output = audit / "records" / source["name"]
        output.parent.mkdir(exist_ok=True)
        output.write_text("".join(json.dumps(r) + "\n" for r in records))
        proofs = [
            {
                "manifest_id": audit_manifest["manifest_id"],
                "compiler_fingerprint": audit_manifest["compiler_fingerprint"],
                "input_hash": input_hash(r),
                "audit_key": digest([input_hash(r), audit_manifest["compiler_fingerprint"]]),
                "build_status": "passed",
                "training_state_status": "passed",
            }
            for r in records
        ]
        journal = shard / "events.jsonl"
        journal.write_text("".join(json.dumps(proof) + "\n" for proof in proofs))
        atomic_json(
            shard / "done.json",
            {
                "manifest_id": audit_manifest["manifest_id"],
                "source_sha256": source["sha256"],
                "output_sha256": file_sha256(output),
                "journal_sha256": file_sha256(journal),
                "passed_records": len(records),
            },
        )
        for ordinal, r in enumerate(records):
            row = {
                "text": "unit schedule PPT SPC TBS PRS",
                "workload_key": source["workload_key"],
                "source_file": source["name"],
                "source_input_hash": input_hash(r),
                "build_audit_manifest_id": audit_manifest["manifest_id"],
                "compiler_fingerprint": audit_manifest["compiler_fingerprint"],
                "build_audit_key": digest([input_hash(r), audit_manifest["compiler_fingerprint"]]),
            }
            raw_rows.append(row)
            if ordinal == 1:
                sample = {
                    "split": "validation",
                    "workload_key": source["workload_key"],
                    "operator_family": "elementwise_other",
                    "row_index": len(raw_rows) - 1,
                    "source_file": source["name"],
                    "source_input_hash": input_hash(r),
                    "row": copy.deepcopy(row),
                    "record": copy.deepcopy(r),
                    "roles": ["median"],
                    "input_ids": [1, 2, 3],
                    "ppt_end": 1,
                    "length": 3,
                }
                sample["sample_id"] = digest(sample)
                anchors.append(sample)
    raw = rebuild / "raw/validation_raw.json"
    raw.write_text("".join(json.dumps(row) + "\n" for row in raw_rows))
    report = {
        "audit_dir": str(audit),
        "manifest_id": audit_manifest["manifest_id"],
        "compiler_fingerprint": audit_manifest["compiler_fingerprint"],
        "partial_scope": False,
        "outputs": {"validation": file_sha256(raw)},
        "counts": {"validation": len(raw_rows)},
    }
    atomic_json(rebuild / "raw/build_audit_manifest.json", report)
    dataset_report = rebuild / "dataset/build_audit_manifest.json"
    atomic_json(dataset_report, report)
    verification_path = rebuild / "dataset/verification.json"
    verification = {"status": "VERIFIED"}
    atomic_json(verification_path, verification)
    atomic_json(
        rebuild / "status.json",
        {
            "state": "SUCCEEDED",
            "exit_code": 0,
            "verification": verification,
        },
    )
    source_manifest = {
        "inputs": {
            "rebuild_dir": str(rebuild),
            "audit_dir": str(audit),
            "audit_manifest_sha256": file_sha256(audit / "manifest.json"),
            "audit_manifest_id": audit_manifest["manifest_id"],
            "compiler_fingerprint": audit_manifest["compiler_fingerprint"],
            "raw_hashes": report["outputs"],
            "dataset_files": {
                str(dataset_report): file_sha256(dataset_report),
                str(verification_path): file_sha256(verification_path),
            },
        }
    }
    source_manifest["manifest_id"] = digest(source_manifest)
    atomic_json(roundtrip / "manifest.json", source_manifest)
    # Historical roundtrip failures are preserved. Publication and source build
    # proofs, rather than an unrelated former process status, establish selection.
    atomic_json(roundtrip / "status.json", {"state": "FAILED", "error": "old timeout"})
    atomic_json(anchor / "samples.json", anchors)
    atomic_json(
        anchor / "manifest.json",
        {
            "samples_sha256": file_sha256(anchor / "samples.json"),
            "registry": str(registry),
        },
    )
    atomic_json(anchor / "status.json", {"state": "SUCCEEDED", "identity_verified": True})
    return anchor, roundtrip, anchors


def test_sketch_identity_ignores_factors_suffix_and_result_without_mutation():
    original = record()
    before = copy.deepcopy(original)
    changed = record(factor=8)
    changed["i"][1][1][-1] = ["PRS", [0, 1]]
    changed["r"] = [[123], 7]
    assert sketch_identity(original) == sketch_identity(changed)
    assert original == before
    assert sketch_identity(original) != sketch_identity(record(structure=1))


@pytest.mark.parametrize("changed_field", ["target", "workload", "hardware"])
def test_sketch_identity_includes_exact_task(changed_field):
    original = record()
    changed = copy.deepcopy(original)
    if changed_field == "target":
        changed["i"][0][1] = "cuda -arch=sm_80"
    elif changed_field == "workload":
        changed["i"][0][0] = workload_key(["other", [16]])
    else:
        changed["i"][0][2] = [32]
    assert sketch_identity(original) != sketch_identity(changed)


def test_suite_keeps_anchors_and_earliest_distinct_representatives_with_explicit_coverage(tmp_path):
    anchor, roundtrip, originals = make_suite(tmp_path)
    cases, evidence = select_multisketch_suite(anchor, roundtrip)
    assert evidence["case_count"] == evidence["available_sketch_count"] == 5
    assert evidence["workload_count"] == 2
    assert evidence["multi_sketch_workload_count"] == 1
    assert evidence["single_sketch_workloads"] == [1]
    assert evidence["requested_sketch_shortfall"] == 3
    assert evidence["unselected_available_sketches"] == 0
    assert evidence["original_roundtrip_state"] == "FAILED"
    assert evidence["all_source_records_and_proofs_verified"]
    assert not evidence["holdout_used"]
    first = [case for case in cases if case["workload_index"] == 0]
    assert len(first) == 4
    assert [case["sketch_id"] for case in first[1:]] == sorted(
        case["sketch_id"] for case in first[1:]
    )
    assert {case["row_index"] for case in first[1:]} == {2, 3, 4}
    assert len({case["sketch_id"] for case in cases}) == 5
    for index, original in enumerate(originals):
        selected = next(case for case in cases if case["workload_index"] == index)
        assert selected["sketch_index"] == 0
        assert {key: selected[key] for key in original} == original
    for path, expected in evidence["input_files"].items():
        assert Path(path).is_absolute() and file_sha256(path) == expected
    assert any(path.endswith("events.jsonl") for path in evidence["input_files"])
    assert any(path.endswith("done.json") for path in evidence["input_files"])
    # Re-selecting is deterministic, including case IDs and source hash evidence.
    assert select_multisketch_suite(anchor, roundtrip) == (cases, evidence)


def test_suite_cap_reports_omitted_structures_without_factor_padding(tmp_path):
    anchor, roundtrip, _ = make_suite(tmp_path)
    cases, evidence = select_multisketch_suite(anchor, roundtrip, max_sketches=2)
    assert len(cases) == 3
    assert evidence["unselected_available_sketches"] == 2
    assert evidence["requested_sketch_shortfall"] == 1


@pytest.mark.parametrize("source_split,holdout", [("train", False), ("validation", True)])
def test_suite_rejects_nonvalidation_or_holdout_workloads(tmp_path, source_split, holdout):
    anchor, roundtrip, _ = make_suite(tmp_path, source_split=source_split, holdout=holdout)
    with pytest.raises(ValueError, match="validation|hold-out"):
        select_multisketch_suite(anchor, roundtrip)


def test_suite_rejects_raw_changed_after_frozen_publication(tmp_path):
    anchor, roundtrip, _ = make_suite(tmp_path)
    raw = tmp_path / "rebuild/raw/validation_raw.json"
    raw.write_text(raw.read_text() + "{}\n")
    with pytest.raises(ValueError, match="Frozen input changed"):
        select_multisketch_suite(anchor, roundtrip)


def test_suite_rejects_source_without_passing_build_proof(tmp_path):
    anchor, roundtrip, _ = make_suite(tmp_path)
    shard = tmp_path / "audit/shards/shard0"
    journal = shard / "events.jsonl"
    proofs = [json.loads(line) for line in journal.read_text().splitlines()]
    proofs[0]["build_status"] = "failed"
    journal.write_text("".join(json.dumps(proof) + "\n" for proof in proofs))
    done = json.loads((shard / "done.json").read_text())
    done["journal_sha256"] = file_sha256(journal)
    atomic_json(shard / "done.json", done)
    with pytest.raises(ValueError, match="no passing build proof"):
        select_multisketch_suite(anchor, roundtrip)


def test_suite_rejects_anchor_mislabeled_as_training(tmp_path):
    anchor, roundtrip, anchors = make_suite(tmp_path)
    anchors[0]["split"] = "train"
    atomic_json(anchor / "samples.json", anchors)
    manifest = json.loads((anchor / "manifest.json").read_text())
    manifest["samples_sha256"] = file_sha256(anchor / "samples.json")
    atomic_json(anchor / "manifest.json", manifest)
    with pytest.raises(ValueError, match="anchor must belong to validation"):
        select_multisketch_suite(anchor, roundtrip)
