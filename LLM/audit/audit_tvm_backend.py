"""Actual TVM/toolchain identification and bounded schedule recovery/build checks."""

import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait

from LLM.core.schedule_record_utils import classify_build_error, digest, normalize_record
from LLM.core.io import file_sha256


def compiler_fingerprint():
    import tvm
    from tvm._ffi.base import _LIB

    package = Path(tvm.__file__).resolve().parent
    library = Path(_LIB._name).resolve()
    tools = {}
    for name in ("nvcc", "gcc", "g++", "llvm-config"):
        executable = shutil.which(name)
        if executable:
            result = subprocess.run(
                [executable, "--version"], capture_output=True, text=True, timeout=15, check=False
            )
            tools[name] = {
                "path": str(Path(executable).resolve()),
                "sha256": file_sha256(executable),
                "version": (result.stdout + result.stderr).strip(),
            }
    return {
        "python": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "tvm_python": str(package),
        "tvm_python_sha256": digest(
            {
                str(path.relative_to(package)): file_sha256(path)
                for path in sorted(package.rglob("*.py"))
            }
        ),
        "libtvm": str(library),
        "libtvm_sha256": file_sha256(library),
        "tvm_build_options": dict(tvm.support.libinfo()),
        "tools": tools,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "TVM_LIBRARY_PATH",
                "PYTHONPATH",
                "CUDA_PATH",
                "CUDA_HOME",
                "CC",
                "CXX",
                "PATH",
                "LD_LIBRARY_PATH",
            )
        },
    }


def register_tasks(registry_path):
    from tvm import auto_scheduler

    with open(registry_path, "rb") as stream:
        tasks = pickle.load(stream)
    for task in tasks:
        auto_scheduler.workload_registry.register_workload_tensors(
            task.workload_key, task.compute_dag.tensors
        )
    return tasks


def _prepare(serialized):
    """Run recovery and apply in a killable worker, including the training-normalized state."""
    from tvm import auto_scheduler
    from tvm.auto_scheduler.measure import MeasureInput, recover_measure_input

    started = time.monotonic()
    report = {
        "state_restore_status": "not_checked",
        "apply_status": "not_checked",
        "training_state_status": "not_checked",
        "build_status": "not_checked",
        "error_no": None,
        "error_message": "",
    }
    try:
        inp = MeasureInput.deserialize(serialized)
        inp = recover_measure_input(inp, rebuild_state=True)
        report["state_restore_status"] = "passed"
        inp.task.compute_dag.apply_steps_from_state(
            inp.state, layout_rewrite=inp.task.layout_rewrite_option
        )
        report["apply_status"] = "passed"

        # The training formatter replaces SP lengths by 1. Verify that actual state too.
        normalized, _ = normalize_record({"i": json.loads(inp.to_json())})
        normalized["i"][0][0] = json.dumps(normalized["i"][0][0])
        normalized.update(r=[[0], 0, 0, 0], v="v0.6")
        normalized_input, _ = auto_scheduler.measure_record.load_record_from_string(
            json.dumps(normalized)
        )
        normalized_input = recover_measure_input(normalized_input, rebuild_state=True)
        normalized_input.task.compute_dag.apply_steps_from_state(
            normalized_input.state, layout_rewrite=normalized_input.task.layout_rewrite_option
        )
        report["training_state_status"] = "applied"
        # Identical effective schedules need only one build, even if SP fields differ.
        equivalent = str(normalized_input.state) == str(inp.state)
        report["training_state_equivalent"] = equivalent
        report["prepared"] = [inp.serialize()]
        if not equivalent:
            report["prepared"].append(normalized_input.serialize())
    except Exception:
        report["error_no"] = 1
        report["error_message"] = traceback.format_exc()[-16000:]
        report["build_status"] = classify_build_error(1, report["error_message"])
        if report["apply_status"] == "passed":
            report["training_state_status"] = "failed"
        elif report["state_restore_status"] == "passed":
            report["apply_status"] = "failed"
        else:
            report["state_restore_status"] = "failed"
    report["duration"] = time.monotonic() - started
    return report


class TVMAuditor:
    """LocalBuilder builds and exports modules; this class never runs a GPU kernel."""

    def __init__(
        self, registry_path, timeout, n_parallel, temp_dir, expected_arch, build_engine="persistent"
    ):
        from tvm.auto_scheduler import measure
        from tvm.contrib.popen_pool import PopenPoolExecutor

        self.timeout = timeout
        self.n_parallel = n_parallel
        self.expected_arch = expected_arch
        self.temp_dir = Path(temp_dir).resolve()
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.tasks = register_tasks(registry_path)
        self.pool = PopenPoolExecutor(n_parallel, timeout)
        self.build_engine = build_engine
        if build_engine in ("persistent", "streaming"):
            from LLM.audit.audit_persistent_builder import PersistentLocalBuilder

            self.builder = PersistentLocalBuilder(n_parallel, timeout, self.temp_dir)
            if build_engine == "streaming":
                self.controller = ThreadPoolExecutor(max_workers=2 * n_parallel)
        elif build_engine == "local":
            self.builder = measure.LocalBuilder(timeout=timeout, n_parallel=n_parallel)
        else:
            raise ValueError(f"Unknown build engine: {build_engine}")

    def audit(self, records, timeout):
        if self.build_engine == "streaming":
            return self.audit_with_retries(records, timeout, 0)
        import tvm
        from tvm import auto_scheduler
        from tvm.auto_scheduler.measure import MeasureInput
        from tvm.contrib.popen_pool import StatusKind

        self.pool._timeout = timeout
        serialized, selected = [], []
        reports = [None] * len(records)
        for index, record in enumerate(records):
            try:
                target = tvm.target.Target(record["i"][0][1])
                if self.expected_arch and str(target.attrs.get("arch", "")) != self.expected_arch:
                    raise ValueError(f"Unexpected target architecture: {target}")
                inp, _ = auto_scheduler.measure_record.load_record_from_string(json.dumps(record))
                serialized.append(inp.serialize())
                selected.append(index)
            except Exception:
                reports[index] = {
                    "state_restore_status": "not_checked",
                    "apply_status": "not_checked",
                    "training_state_status": "not_checked",
                    "build_status": "input_error",
                    "error_no": None,
                    "error_message": traceback.format_exc()[-16000:],
                    "duration": 0.0,
                }

        build_inputs, owners = [], []
        for index, result in zip(selected, self.pool.map_with_error_catching(_prepare, serialized)):
            if result.status != StatusKind.COMPLETE:
                reports[index] = {
                    "state_restore_status": "unknown",
                    "apply_status": "not_checked",
                    "training_state_status": "not_checked",
                    "build_status": (
                        "timeout" if result.status == StatusKind.TIMEOUT else "environment_error"
                    ),
                    "error_no": 6 if result.status == StatusKind.TIMEOUT else 8,
                    "error_message": str(result.value)[-16000:],
                    "duration": float(timeout),
                }
                continue
            report = reports[index] = result.value
            for prepared in report.pop("prepared", []):
                build_inputs.append(MeasureInput.deserialize(prepared))
                owners.append(index)

        try:
            if not build_inputs:
                results = []
            elif self.build_engine == "persistent":
                results = self.builder.build(build_inputs, timeout)
            else:
                # Keep the original LocalBuilder available for comparison/control runs.
                self.builder = auto_scheduler.measure.LocalBuilder(
                    timeout=timeout, n_parallel=self.n_parallel
                )
                results = self.builder.build(build_inputs, verbose=0)
            if len(results) != len(build_inputs):
                raise RuntimeError("LocalBuilder returned an incomplete result batch")
            for owner, result in zip(owners, results):
                report = reports[owner]
                error_no = int(result.error_no)
                message = str(result.error_msg or "")[-16000:]
                status = classify_build_error(error_no, message)
                if error_no == 0 and (
                    not result.filename
                    or not Path(str(result.filename)).is_file()
                    or Path(str(result.filename)).stat().st_size == 0
                ):
                    status, error_no, message = "environment_error", 8, "Missing exported module"
                report["duration"] += float(result.time_cost)
                # Both original and training-normalized schedules must pass.
                if report["build_status"] in ("not_checked", "passed"):
                    report.update(build_status=status, error_no=error_no, error_message=message)
            for report in reports:
                if report["build_status"] == "passed":
                    report["training_state_status"] = "passed"
        finally:
            # TMPDIR is private to this run; all builders have returned at this point.
            for path in self.temp_dir.iterdir():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        return reports

    def _stream_attempt(self, record, timeout):
        """Keep each recovery/build bounded, scheduling other records while one waits."""
        import tvm
        from tvm import auto_scheduler

        try:
            target = tvm.target.Target(record["i"][0][1])
            if self.expected_arch and str(target.attrs.get("arch", "")) != self.expected_arch:
                raise ValueError(f"Unexpected target architecture: {target}")
            inp, _ = auto_scheduler.measure_record.load_record_from_string(json.dumps(record))
            serialized = inp.serialize()
        except Exception:
            return {
                "state_restore_status": "not_checked",
                "apply_status": "not_checked",
                "training_state_status": "not_checked",
                "build_status": "input_error",
                "error_no": None,
                "error_message": traceback.format_exc()[-16000:],
                "duration": 0.0,
            }

        try:
            report = self.pool.submit(_prepare, serialized).result()
        except Exception as error:
            return {
                "state_restore_status": "unknown",
                "apply_status": "not_checked",
                "training_state_status": "not_checked",
                "build_status": (
                    "timeout" if isinstance(error, TimeoutError) else "environment_error"
                ),
                "error_no": 6 if isinstance(error, TimeoutError) else 8,
                "error_message": str(error)[-16000:],
                "duration": float(timeout),
            }
        for prepared in report.pop("prepared", []):
            try:
                filename, error_no, message, duration = self.builder.submit(prepared).result()
                status = classify_build_error(error_no, message)
                if error_no == 0 and (
                    not filename
                    or not Path(filename).is_file()
                    or Path(filename).stat().st_size == 0
                ):
                    status, error_no, message = "environment_error", 8, "Missing exported module"
                report["duration"] += duration
            except Exception as error:
                status = "timeout" if isinstance(error, TimeoutError) else "environment_error"
                error_no = 6 if isinstance(error, TimeoutError) else 8
                message = str(error)[-16000:]
                report["duration"] += timeout
            if report["build_status"] in ("not_checked", "passed"):
                report.update(build_status=status, error_no=error_no, error_message=message)
            if status != "passed":
                # An original-state rejection already excludes this record.
                break
        if report["build_status"] == "passed":
            report["training_state_status"] = "passed"
        return report

    def audit_with_retries(self, records, timeout, max_retries):
        """Overlap recovery, builds and retries without changing their individual budgets."""
        # Retry budgets vary per submission. Keep separate persistent retry pools so
        # a doubled budget never races with an ordinary submission.
        from tvm.contrib.popen_pool import PopenPoolExecutor
        from LLM.audit.audit_persistent_builder import PersistentLocalBuilder

        self.pool._timeout = timeout
        self.builder.pool._timeout = timeout
        if max_retries and not hasattr(self, "retry_pool"):
            self.retry_pool = PopenPoolExecutor(max(1, self.n_parallel // 4), timeout * 2)
            self.retry_builder = PersistentLocalBuilder(
                max(1, self.n_parallel // 4), timeout * 2, self.temp_dir
            )

        def check(record):
            report = self._stream_attempt(record, timeout)
            report["retry_count"] = 0
            for retry in range(1, max_retries + 1):
                if report["build_status"] not in ("timeout", "environment_error"):
                    break
                previous = report
                # A shallow view selects the retry pools without touching shared state.
                import copy

                view = copy.copy(self)
                view.pool, view.builder = self.retry_pool, self.retry_builder
                report = view._stream_attempt(record, timeout * 2)
                report["duration"] += previous["duration"]
                report["prior_attempts"] = previous.get("prior_attempts", []) + [
                    {
                        "build_status": previous["build_status"],
                        "error_no": previous["error_no"],
                        "error_message": previous["error_message"],
                    }
                ]
                report["retry_count"] = retry
            return report

        futures = [self.controller.submit(check, record) for record in records]
        try:
            return [future.result() for future in futures]
        finally:
            wait(futures)
            # All submitted build operations have returned before cleaning this batch.
            for path in self.temp_dir.iterdir():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
