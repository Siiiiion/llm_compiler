"""Select distinct PPT structures from verified validation records, without sampling outcomes."""

import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

from LLM.data.prepare_audited_clm import load_manifest, load_verified_source
from LLM.core.schedule_record_utils import digest, input_hash, normalize_record, workload_key
from LLM.core.io import file_sha256


def sketch_identity(record):
    """Identify the exact task and normalized PPT input consumed by make_sketch."""
    normalized, ppt = normalize_record(record)
    if ppt is None:
        raise ValueError("A selected record must contain a PPT marker")
    normalized["i"][1][1] = normalized["i"][1][1][: ppt + 1]
    return input_hash(normalized)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def select_multisketch_suite(anchor_run, roundtrip_run, max_sketches=4):
    """Return flattened validation cases and their frozen selection provenance.

    The previous median is always first. Other structures are ordered by their
    exact normalized PPT identity, using the earliest raw row of each structure.
    A workload with too few distinct structures is reported, never padded with
    factor variants. The old roundtrip status is provenance, not a success claim:
    this selection independently verifies the published raw and build proofs.
    """
    _require(
        isinstance(max_sketches, int) and not isinstance(max_sketches, bool) and max_sketches > 0,
        "max_sketches must be a positive integer",
    )
    anchor_run, roundtrip_run = Path(anchor_run).resolve(), Path(roundtrip_run).resolve()
    inputs = {}

    def freeze(path, expected=None):
        path = Path(path).resolve()
        actual = file_sha256(path)
        if expected is not None:
            _require(actual == expected, f"Frozen input changed: {path}")
        if str(path) in inputs:
            _require(inputs[str(path)] == actual, f"Input changed during selection: {path}")
        inputs[str(path)] = actual
        return path

    def read(path, expected=None):
        return json.loads(freeze(path, expected).read_text())

    previous = read(anchor_run / "manifest.json")
    anchor_status = read(anchor_run / "status.json")
    _require(
        anchor_status["state"] == "SUCCEEDED" and anchor_status.get("identity_verified") is True,
        "The anchor experiment must have succeeded with verified identity",
    )
    anchors = read(anchor_run / "samples.json", previous["samples_sha256"])
    _require(anchors, "The anchor experiment has no samples")
    keys = [workload_key(sample["workload_key"]) for sample in anchors]
    _require(len(set(keys)) == len(keys), "Anchor workloads must be distinct")
    for key, sample in zip(keys, anchors):
        _require(sample["split"] == "validation", "Every anchor must belong to validation")
        _require(
            workload_key(sample["record"]["i"][0][0]) == key
            and input_hash(sample["record"]) == sample["source_input_hash"],
            "Anchor record identity differs from its sample",
        )

    roundtrip = read(roundtrip_run / "manifest.json")
    identifier = roundtrip["manifest_id"]
    _require(
        digest({key: value for key, value in roundtrip.items() if key != "manifest_id"})
        == identifier,
        "Roundtrip manifest identity mismatch",
    )
    roundtrip_status = read(roundtrip_run / "status.json")
    snapshot = roundtrip["inputs"]
    rebuild, audit_dir = Path(snapshot["rebuild_dir"]), Path(snapshot["audit_dir"])
    rebuild_status = read(rebuild / "status.json")
    verification_path = rebuild / "dataset/verification.json"
    verification = read(verification_path, snapshot["dataset_files"][str(verification_path)])
    _require(
        rebuild_status["state"] == "SUCCEEDED"
        and rebuild_status["exit_code"] == 0
        and verification == rebuild_status["verification"]
        and verification["status"] == "VERIFIED",
        "Published raw/Arrow verification must have succeeded",
    )
    raw_report = read(rebuild / "raw/build_audit_manifest.json")
    dataset_report_path = rebuild / "dataset/build_audit_manifest.json"
    _require(
        read(dataset_report_path, snapshot["dataset_files"][str(dataset_report_path)])
        == raw_report,
        "Raw and published Arrow build provenance differ",
    )
    _require(
        Path(raw_report["audit_dir"]).resolve() == audit_dir.resolve()
        and raw_report["manifest_id"] == snapshot["audit_manifest_id"]
        and raw_report["compiler_fingerprint"] == snapshot["compiler_fingerprint"]
        and not raw_report.get("partial_scope", False),
        "Raw build provenance differs from the frozen audit",
    )
    freeze(audit_dir / "manifest.json", snapshot["audit_manifest_sha256"])
    freeze(audit_dir / "status.json")
    audit = load_manifest(audit_dir)
    _require(
        audit["manifest_id"] == snapshot["audit_manifest_id"]
        and audit["compiler_fingerprint"] == snapshot["compiler_fingerprint"],
        "Audit identity differs from the roundtrip source",
    )
    freeze(audit["registry"]["path"], audit["registry"]["sha256"])
    _require(
        file_sha256(previous["registry"]) == audit["registry"]["sha256"],
        "Anchor and audited workload registries differ",
    )
    freeze(previous["registry"], audit["registry"]["sha256"])
    target_keys, holdouts = set(keys), set(map(workload_key, audit["hold_out_keys"]))
    _require(not target_keys.intersection(holdouts), "Anchor workload belongs to final hold-out")
    sources = {source["name"]: source for source in audit["sources"]}
    selected_sources = [
        source for source in sources.values() if workload_key(source["workload_key"]) in target_keys
    ]
    _require(
        all(source["split"] == "validation" for source in selected_sources),
        "Selected source workload is not validation",
    )
    _require(
        {workload_key(source["workload_key"]) for source in selected_sources} == target_keys,
        "Some anchor workloads have no audited validation source",
    )
    records = {}
    for source in selected_sources:
        shard = audit_dir / "shards" / source["shard"]
        for path in (
            shard / "done.json",
            shard / "events.jsonl",
            audit_dir / "records" / source["name"],
        ):
            freeze(path)
        lines, _ = load_verified_source(audit_dir, audit, source)
        records[source["name"]] = {input_hash(record): record for record in map(json.loads, lines)}

    raw = rebuild / "raw/validation_raw.json"
    _require(
        snapshot["raw_hashes"]["validation"] == raw_report["outputs"]["validation"],
        "Validation raw identity differs between frozen manifests",
    )
    freeze(raw, snapshot["raw_hashes"]["validation"])
    by_workload = defaultdict(dict)
    raw_counts, seen_anchors = Counter(), set()
    anchor_by_row = {sample["row_index"]: (index, sample) for index, sample in enumerate(anchors)}
    count = 0
    with raw.open() as stream:
        for row_index, line in enumerate(stream):
            row = json.loads(line)
            key = workload_key(row["workload_key"])
            source = sources.get(row["source_file"])
            _require(
                source is not None
                and source["split"] == "validation"
                and workload_key(source["workload_key"]) == key
                and key not in holdouts,
                "Validation raw source membership or hold-out mismatch",
            )
            _require(
                row["build_audit_manifest_id"] == audit["manifest_id"]
                and row["compiler_fingerprint"] == audit["compiler_fingerprint"]
                and row["build_audit_key"]
                == digest([row["source_input_hash"], audit["compiler_fingerprint"]]),
                "Validation raw build proof identity mismatch",
            )
            count += 1
            if key not in target_keys:
                continue
            record = records.get(row["source_file"], {}).get(row["source_input_hash"])
            _require(record is not None, "Raw input has no verified source record")
            raw_counts[key] += 1
            identity = sketch_identity(record)
            by_workload[key].setdefault(
                identity, {"row_index": row_index, "row": row, "record": record}
            )
            if row_index in anchor_by_row:
                index, anchor = anchor_by_row[row_index]
                _require(
                    workload_key(anchor["workload_key"]) == key
                    and anchor["source_input_hash"] == row["source_input_hash"]
                    and anchor["source_file"] == row["source_file"]
                    and anchor["row"] == row
                    and sketch_identity(anchor["record"]) == identity,
                    "Anchor does not match its verified raw row",
                )
                seen_anchors.add(index)
    _require(count == raw_report["counts"]["validation"], "Validation raw row count mismatch")
    _require(seen_anchors == set(range(len(anchors))), "An anchor row was not found in validation")

    cases, coverage = [], []
    for index, (key, anchor) in enumerate(zip(keys, anchors)):
        anchor_identity = sketch_identity(anchor["record"])
        available = by_workload[key]
        _require(anchor_identity in available, "Anchor sketch has no verified representative")
        identities = [anchor_identity] + sorted(set(available) - {anchor_identity})[
            : max_sketches - 1
        ]
        for sketch_index, identity in enumerate(identities):
            if not sketch_index:
                sample = copy.deepcopy(anchor)
            else:
                representative = available[identity]
                sample = {
                    "split": "validation",
                    "workload_key": key,
                    "operator_family": anchor["operator_family"],
                    "row_index": representative["row_index"],
                    "source_file": representative["row"]["source_file"],
                    "source_input_hash": representative["row"]["source_input_hash"],
                    "roles": ["distinct_sketch_representative"],
                    "row": copy.deepcopy(representative["row"]),
                    "record": copy.deepcopy(representative["record"]),
                }
                sample["sample_id"] = digest(sample)
            sample.update(workload_index=index, sketch_index=sketch_index, sketch_id=identity)
            cases.append(sample)
        coverage.append(
            {
                "workload_index": index,
                "workload_key": key,
                "family": anchor["operator_family"],
                "raw_rows": raw_counts[key],
                "available_sketches": len(available),
                "selected_sketches": len(identities),
                "requested_sketch_shortfall": max(0, max_sketches - len(identities)),
                "unselected_available_sketches": len(available) - len(identities),
                "multi_sketch": len(identities) > 1,
                "anchor_sample_id": anchor["sample_id"],
                "sketch_ids": identities,
            }
        )
    for path, expected in inputs.items():
        _require(file_sha256(path) == expected, f"Input changed during selection: {path}")
    evidence = {
        "schema": 1,
        "policy": "original_median_anchor_then_sorted_distinct_PPT_hashes_earliest_verified_raw_row",
        "sketch_identity": "input_hash_of_normalized_task_and_state_truncated_through_first_PPT",
        "requested_sketches_per_workload": max_sketches,
        "workload_count": len(anchors),
        "case_count": len(cases),
        "available_sketch_count": sum(row["available_sketches"] for row in coverage),
        "multi_sketch_workload_count": sum(row["multi_sketch"] for row in coverage),
        "single_sketch_workloads": [
            row["workload_index"] for row in coverage if not row["multi_sketch"]
        ],
        "requested_sketch_shortfall": sum(row["requested_sketch_shortfall"] for row in coverage),
        "unselected_available_sketches": sum(
            row["unselected_available_sketches"] for row in coverage
        ),
        "per_workload": coverage,
        "validation_raw_rows": count,
        "audit_manifest_id": audit["manifest_id"],
        "roundtrip_manifest_id": identifier,
        "original_roundtrip_state": roundtrip_status["state"],
        "original_roundtrip_note": "Source provenance only; raw publication and source build proofs independently checked; original failures remain unchanged",
        "all_source_records_and_proofs_verified": True,
        "holdout_used": False,
        "selection_uses_model_outputs_or_latency": False,
        "input_files": inputs,
    }
    return cases, evidence
