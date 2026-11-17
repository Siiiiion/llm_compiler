"""Regression tests for schedule provenance, fail-closed filtering and resumable auditing."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))

from LLM.audit.audit_schedule_dataset import audit_source, scan_raw_splits
from LLM.data.prepare_audited_clm import load_manifest, load_verified_source
from LLM.core.schedule_record_utils import (
    AUDIT_SCHEMA,
    atomic_json,
    classify_build_error,
    digest,
    file_sha256,
    input_hash,
    measured_latency,
    normalize_record,
    read_journal,
    workload_key,
)


def record(factor=4, error=0, costs=None):
    return {
        "i": [
            [json.dumps(["workload", [16]]), "cuda -arch=sm_89", [1, 2], "llvm", 0, []],
            [
                [],
                [
                    ["SP", 0, 0, 16, [factor], 1],
                    ["PPT"],
                    ["SPC", 0, 0, 16, [factor], 1],
                    ["TBS"],
                    ["PRS", [0]],
                ],
            ],
        ],
        "r": [[0] if costs is None else costs, error, 0, 123],
        "v": "v0.6",
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["i"][0].__setitem__(1, "cuda -arch=sm_70"),
        lambda r: r["i"][0].__setitem__(3, "llvm -mcpu=avx2"),
        lambda r: r["i"][0].__setitem__(4, 1),
        lambda r: r["i"][0].__setitem__(5, ["constant_input"]),
        lambda r: r["i"][0][2].append(3),
        lambda r: r["i"][1][1][2][4].__setitem__(0, 8),
    ],
)
def test_input_identity_includes_full_target_state_and_task_configuration(mutate):
    original = record()
    modified = copy.deepcopy(original)
    mutate(modified)
    assert input_hash(original) != input_hash(modified)


def test_identity_ignores_measurements_and_workload_string_whitespace():
    original = record()
    modified = record(costs=[0.1])
    modified["i"][0][0] = ["workload", [16]]
    assert input_hash(original) == input_hash(modified)
    normalized, ppt = normalize_record(original)
    assert original["i"][1][1][0][4] == [4]
    assert normalized["i"][1][1][0][4] == [1]
    assert normalized["i"][1][1][2][4] == [4]
    assert ppt == 1


@pytest.mark.parametrize(
    "costs,error",
    [
        ([0], 0),
        ([1e10], 0),
        ([1e9], 0),
        ([float("nan")], 0),
        ([float("inf")], 0),
        ([0.001], 2),
        ([0.001], 4),
        ([0.001, 0], 0),
        ([0.001, -1], 0),
        ([True], 0),
        (["0.001"], 0),
        ([], 0),
    ],
)
def test_invalid_or_failed_measurement_never_becomes_performance_supervision(costs, error):
    assert measured_latency(record(costs=costs, error=error)) is None


def test_valid_measurement_mean():
    assert measured_latency(record(costs=[0.001, 0.003])) == pytest.approx(0.002)


def test_timeout_and_environment_failures_are_separate_from_build_rejections():
    assert classify_build_error(6, "") == "timeout"
    assert classify_build_error(2, "nvcc: No such file or directory") == "environment_error"
    assert classify_build_error(2, "out of memory") == "environment_error"
    assert classify_build_error(2, "LowerThreadAllreduce: Check failed") == "build_error"


def test_resume_repairs_only_torn_tail(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"ok": 1}\n{"unfinished":')
    assert read_journal(path, repair=True) == [{"ok": 1}]
    assert path.read_bytes() == b'{"ok": 1}\n'
    path.write_bytes(b'not-json\n{"ok": 1}\n')
    with pytest.raises(ValueError):
        read_journal(path, repair=True)


def setup_source(tmp_path, records):
    path = tmp_path / "source.json"
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    source = {
        "path": str(path),
        "name": path.name,
        "sha256": file_sha256(path),
        "workload_key": workload_key(records[0]["i"][0][0]),
        "shard": "shard",
        "split": "train",
    }
    manifest = {"manifest_id": "manifest", "compiler_fingerprint": "compiler", "hold_out_keys": []}
    args = SimpleNamespace(max_records_per_file=None, batch_size=1, build_timeout=2, max_retries=1)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return source, manifest, args, run_dir


def report(status="passed"):
    return {
        "build_status": status,
        "training_state_status": "passed" if status == "passed" else "not_checked",
        "error_no": 0 if status == "passed" else 6,
        "error_message": "",
        "duration": 0.1,
    }


def test_restart_reuses_committed_batch_and_deduplicates_builds(tmp_path):
    source, manifest, args, run_dir = setup_source(tmp_path, [record(), record(), record(8)])

    class Interrupted:
        calls = 0

        def audit(self, records, timeout):
            self.calls += 1
            if self.calls == 2:
                raise InterruptedError("simulated restart")
            return [report() for _ in records]

    with pytest.raises(InterruptedError):
        audit_source(source, manifest, run_dir, Interrupted(), args)

    class Resumed:
        seen = []

        def audit(self, records, timeout):
            self.seen.extend(input_hash(row) for row in records)
            return [report() for _ in records]

    backend = Resumed()
    summary = audit_source(source, manifest, run_dir, backend, args)
    assert backend.seen == [input_hash(record(8))]
    assert summary["passed_records"] == 3
    assert summary["unique_inputs"] == 2
    assert load_verified_source(run_dir, manifest, source)[0]
    assert audit_source(source, manifest, run_dir, None, args) == summary


def test_timeout_is_retried_and_previous_attempt_is_preserved(tmp_path):
    source, manifest, args, run_dir = setup_source(tmp_path, [record()])

    class Backend:
        budgets = []

        def audit(self, records, timeout):
            self.budgets.append(timeout)
            return [report("timeout" if len(self.budgets) == 1 else "passed")]

    backend = Backend()
    summary = audit_source(source, manifest, run_dir, backend, args)
    assert backend.budgets == [2, 4]
    assert summary["passed_records"] == 1
    proof = read_journal(run_dir / "shards/shard/events.jsonl")[0]
    assert proof["retry_count"] == 1 and proof["prior_attempts"][0]["build_status"] == "timeout"


def test_source_and_filtered_output_tampering_are_rejected(tmp_path):
    source, manifest, args, run_dir = setup_source(tmp_path, [record()])
    backend = SimpleNamespace(audit=lambda records, timeout: [report()])
    audit_source(source, manifest, run_dir, backend, args)
    (run_dir / "records/source.json").write_text(json.dumps(record(8)) + "\n")
    with pytest.raises(ValueError, match="changed"):
        load_verified_source(run_dir, manifest, source)
    Path(source["path"]).write_text("changed")
    with pytest.raises(ValueError, match="frozen manifest"):
        audit_source(source, manifest, run_dir, backend, args)


def test_existing_workload_split_is_preserved_and_leakage_is_rejected(tmp_path):
    row = {"workload_key": '["a"]', "source_file": "a.json"}
    (tmp_path / "train_raw.json").write_text(json.dumps(row) + "\n")
    validation = {"workload_key": '["b"]', "source_file": "b.json"}
    (tmp_path / "validation_raw.json").write_text(json.dumps(validation) + "\n")
    sources, _ = scan_raw_splits(tmp_path)
    assert sources["a.json"]["split"] == "train"
    assert sources["b.json"]["split"] == "validation"
    validation["workload_key"] = '["a"]'
    (tmp_path / "validation_raw.json").write_text(json.dumps(validation) + "\n")
    with pytest.raises(ValueError, match="leaks"):
        scan_raw_splits(tmp_path)


def test_partial_or_changed_audit_cannot_be_used_as_formal_dataset(tmp_path):
    manifest = {
        "schema": AUDIT_SCHEMA,
        "compiler": {},
        "compiler_fingerprint": digest({}),
        "settings": {"partial_scope": True},
        "raw_splits": {"train": {}},
    }
    manifest["manifest_id"] = digest(manifest)
    atomic_json(tmp_path / "manifest.json", manifest)
    atomic_json(tmp_path / "status.json", {"state": "SUCCEEDED"})
    with pytest.raises(ValueError, match="Partial"):
        load_manifest(tmp_path)
    manifest["compiler"] = {"changed": True}
    atomic_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="identity"):
        load_manifest(tmp_path, allow_partial=True)


def test_training_formatter_excludes_failed_or_unverified_states(monkeypatch):
    module = pytest.importorskip("LLM.data.make_dataset")
    task = SimpleNamespace(
        workload_key=json.dumps(["workload", [16]]),
        compute_dag=SimpleNamespace(print_min=lambda: "compute_dag"),
    )
    monkeypatch.setattr(module, "load_record_from_string", lambda line: (object(), None))
    monkeypatch.setattr(
        module.auto_scheduler.measure,
        "recover_measure_input",
        lambda inp: SimpleNamespace(task=task),
    )
    rows, stats = module.for_gen(
        [json.dumps(record()), json.dumps(record(8)), json.dumps(record(2, error=4))],
        "source.json",
        0,
        audit_proofs={
            input_hash(record()): {
                "build_status": "passed",
                "training_state_status": "passed",
                "audit_key": "proof",
                "compiler_fingerprint": "compiler",
            }
        },
    )
    assert len(rows) == 1
    assert stats["drop_unverified"] == stats["drop_recorded_error"] == 1
    assert rows[0]["build_audit_key"] == "proof"
    assert rows[0]["latency"] is None


def test_reencoding_rejects_raw_changed_since_audited_export(tmp_path):
    module = pytest.importorskip("LLM.data.rebuild_clm_dataset")
    audit = tmp_path / "audit"
    audit.mkdir()
    manifest = {
        "schema": AUDIT_SCHEMA,
        "compiler": {},
        "compiler_fingerprint": digest({}),
        "settings": {"partial_scope": False},
        "raw_splits": {"train": {}, "validation": {}},
    }
    manifest["manifest_id"] = digest(manifest)
    atomic_json(audit / "manifest.json", manifest)
    atomic_json(audit / "status.json", {"state": "SUCCEEDED"})
    train, validation = tmp_path / "train.json", tmp_path / "validation.json"
    train.write_text('{"text":"train"}\n')
    validation.write_text('{"text":"validation"}\n')
    report = {
        "audit_dir": str(audit),
        "manifest_id": manifest["manifest_id"],
        "compiler_fingerprint": manifest["compiler_fingerprint"],
        "outputs": {"train": file_sha256(train), "validation": file_sha256(validation)},
    }
    report_path = tmp_path / "build_audit_manifest.json"
    atomic_json(report_path, report)
    assert module.verify_build_audit(report_path, train, validation) == report
    train.write_text('{"text":"different state"}\n')
    with pytest.raises(ValueError, match="differs"):
        module.verify_build_audit(report_path, train, validation)


def migration_fixture(tmp_path):
    source, manifest, args, origin = setup_source(tmp_path, [record(), record(8)])
    manifest.update(
        schema=AUDIT_SCHEMA,
        compiler={},
        compiler_fingerprint=digest({}),
        source_dir=str(tmp_path),
        registry={},
        raw_splits={"train": {}},
        sources=[source],
        code={
            name: "unchanged"
            for name in (
                "schedule_record_utils.py",
                "make_dataset.py",
                "make_dataset_utils.py",
                "prepare_audited_clm.py",
            )
        },
        settings={
            "n_parallel": 8,
            "batch_size": 1,
            "expected_arch": "sm_89",
            "build_timeout": 2,
            "max_retries": 1,
            "max_records_per_file": None,
            "partial_scope": False,
            "validation": "original_and_training_normalized_state_build_export",
            "runtime_correctness_checked": False,
        },
    )
    manifest.pop("manifest_id")
    manifest["manifest_id"] = digest(manifest)
    atomic_json(origin / "manifest.json", manifest)
    (origin / "run.lock").touch()
    new = copy.deepcopy(manifest)
    new.pop("manifest_id")
    new["settings"].update(n_parallel=32, batch_size=256, build_engine="persistent")
    new["manifest_id"] = digest(new)
    destination = tmp_path / "accelerated"
    destination.mkdir()
    return source, manifest, new, args, origin, destination


def test_acceleration_reuses_committed_proofs_with_origin_and_preserves_resume(tmp_path):
    from LLM.audit.audit_reuse import reuse_results

    source, old, new, args, origin, destination = migration_fixture(tmp_path)

    class Backend:
        def audit(self, records, timeout):
            return [report() for _ in records]

    audit_source(source, old, origin, Backend(), args)
    result = reuse_results(origin, destination, new)
    assert result["newly_imported_proofs"] == 2
    summary = audit_source(source, new, destination, None, args)
    assert summary["passed_records"] == summary["reused_unique_inputs"] == 2
    _, proofs = load_verified_source(destination, new, source)
    assert all(row["reused_from"]["manifest_id"] == old["manifest_id"] for row in proofs.values())
    assert reuse_results(origin, destination, new)["newly_imported_proofs"] == 0


@pytest.mark.parametrize("field", ["compiler", "sources", "raw_splits", "code", "settings"])
def test_acceleration_cannot_reuse_changed_environment_sources_or_validation(tmp_path, field):
    from LLM.audit.audit_reuse import compatible_origin

    _, _, new, _, origin, _ = migration_fixture(tmp_path)
    if field == "sources":
        new[field][0]["sha256"] = "changed"
    elif field == "code":
        new[field]["make_dataset.py"] = "changed"
    elif field == "settings":
        new[field]["validation"] = "unchecked"
    else:
        new[field]["changed"] = True
    with pytest.raises(ValueError, match="Cannot reuse"):
        compatible_origin(origin, new)


def test_acceleration_rejects_running_origin(tmp_path):
    import fcntl
    from LLM.audit.audit_reuse import reuse_results

    _, _, new, _, origin, destination = migration_fixture(tmp_path)
    with (origin / "run.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="still running"):
            reuse_results(origin, destination, new)


def test_persistent_builder_keeps_timeouts_crashes_and_missing_exports_out_of_passes(tmp_path):
    from LLM.audit.audit_persistent_builder import PersistentLocalBuilder
    from tvm.contrib.popen_pool import StatusKind

    module = tmp_path / "module.tar"
    module.write_bytes(b"exported")
    outcomes = [
        SimpleNamespace(status=StatusKind.COMPLETE, value=(str(module), 0, "", 0.1)),
        SimpleNamespace(status=StatusKind.TIMEOUT, value=TimeoutError()),
        SimpleNamespace(status=StatusKind.EXCEPTION, value=RuntimeError("crash")),
        SimpleNamespace(status=StatusKind.COMPLETE, value=("", 0, "", 0.1)),
    ]

    class Pool:
        def map_with_error_catching(self, fn, inputs):
            assert len(inputs) == 4
            return iter(outcomes)

    builder = PersistentLocalBuilder.__new__(PersistentLocalBuilder)
    builder.temp_dir = None
    builder.pool = pool = Pool()
    inputs = [SimpleNamespace(serialize=lambda: "input") for _ in range(4)]
    for timeout in (2, 4):
        results = builder.build(inputs, timeout)
        assert builder.pool is pool and pool._timeout == timeout
        assert [classify_build_error(int(r.error_no), str(r.error_msg)) for r in results] == [
            "passed",
            "timeout",
            "environment_error",
            "environment_error",
        ]


def test_streaming_overlaps_retries_and_keeps_each_pool_timeout_fixed(tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from LLM.audit.audit_tvm_backend import TVMAuditor

    backend = TVMAuditor.__new__(TVMAuditor)
    backend.n_parallel, backend.temp_dir = 2, tmp_path
    backend.pool = primary = SimpleNamespace()
    backend.builder = SimpleNamespace(pool=SimpleNamespace())
    backend.retry_pool = retry = SimpleNamespace(_timeout=4)
    backend.retry_builder = SimpleNamespace(pool=SimpleNamespace(_timeout=4))
    backend.controller = ThreadPoolExecutor(max_workers=4)
    another_record_built = threading.Event()

    def attempt(self, item, timeout):
        if item == "slow" and self.pool is primary:
            assert timeout == self.pool._timeout == 2
            return report("timeout")
        if item == "slow":
            assert self.pool is retry and self.pool._timeout == timeout == 4
            assert another_record_built.wait(timeout=3)
        elif item == "fast":
            another_record_built.set()
        return report()

    monkeypatch.setattr(TVMAuditor, "_stream_attempt", attempt)
    try:
        results = backend.audit_with_retries(["slow", "fast"], 2, 1)
    finally:
        backend.controller.shutdown()
    assert all(row["build_status"] == "passed" for row in results)
    assert results[0]["retry_count"] == 1
    assert results[0]["prior_attempts"][0]["build_status"] == "timeout"
    assert results[1]["retry_count"] == 0
    assert primary._timeout == 2 and retry._timeout == 4


def test_workloads_run_concurrently_with_isolated_temporary_directories(tmp_path, monkeypatch):
    import tempfile
    import threading
    from LLM.audit import audit_schedule_dataset as module
    from LLM.audit import audit_tvm_backend

    sources = []
    for index in range(4):
        directory = tmp_path / str(index)
        directory.mkdir()
        source, _, args, _ = setup_source(directory, [record(index + 1)])
        source.update(name=f"source_{index}.json", shard=f"shard_{index}")
        sources.append(source)
    run_dir = tmp_path / "concurrent"
    args = SimpleNamespace(
        run_dir=str(run_dir),
        registry="registry",
        n_parallel=2,
        workload_parallel=2,
        build_timeout=2,
        build_engine="persistent",
        reuse_audit_dir=None,
        batch_size=1,
        max_records_per_file=None,
        max_retries=0,
    )
    manifest = {
        "sources": sources,
        "compiler": {},
        "settings": {"partial_scope": False},
        "hold_out_keys": [],
    }
    monkeypatch.setattr(module, "make_manifest", lambda args: copy.deepcopy(manifest))
    barrier, directories = threading.Barrier(2), []

    class Backend:
        def __init__(self, registry, timeout, parallel, temp_dir, arch, build_engine):
            directories.append(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)

        def audit(self, records, timeout):
            barrier.wait(timeout=5)
            return [report() for _ in records]

    args.expected_arch = "sm_89"
    monkeypatch.setattr(audit_tvm_backend, "TVMAuditor", Backend)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    assert module.run(args) == 0
    assert len(set(directories)) == 2
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["completed_files"] == summary["records"] == summary["passed_records"] == 4
    assert json.loads((run_dir / "status.json").read_text())["state"] == "SUCCEEDED"
