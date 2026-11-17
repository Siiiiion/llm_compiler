"""Reuse killable TVM workers while retaining LocalBuilder's build/export operation."""

from pathlib import Path


def _build_default(payload):
    import os
    import tempfile
    from tvm.auto_scheduler.measure import _local_build_worker
    from tvm.contrib import tar

    serialized, temp_dir = payload
    if temp_dir:
        os.environ["TMPDIR"] = temp_dir
        tempfile.tempdir = temp_dir
    # Call the same operation as the default LocalBuilder, including module export.
    filename, _, error_no, message, duration = _local_build_worker(serialized, tar.tar, 0)
    return filename, int(error_no), str(message or ""), float(duration)


class PersistentLocalBuilder:
    """LocalBuilder-compatible results without rebuilding the pool for every batch."""

    def __init__(self, n_parallel, timeout, temp_dir=None):
        from tvm.auto_scheduler.measure import AutotvmGlobalScope, reset_global_scope
        from tvm.contrib.popen_pool import PopenPoolExecutor

        self.temp_dir = str(Path(temp_dir).resolve()) if temp_dir else None
        self.pool = PopenPoolExecutor(
            n_parallel,
            timeout,
            reset_global_scope,
            (AutotvmGlobalScope.current,),
            maximum_process_uses=1024,
        )

    def submit(self, serialized):
        return self.pool.submit(_build_default, (serialized, self.temp_dir))

    def build(self, inputs, timeout):
        from tvm.auto_scheduler.measure import BuildResult, MeasureErrorNo
        from tvm.contrib.popen_pool import StatusKind

        self.pool._timeout = timeout
        results = []
        for result in self.pool.map_with_error_catching(
            _build_default, [(inp.serialize(), self.temp_dir) for inp in inputs]
        ):
            if result.status == StatusKind.COMPLETE:
                filename, error_no, message, duration = result.value
                if error_no == 0 and (
                    not filename
                    or not Path(filename).is_file()
                    or Path(filename).stat().st_size == 0
                ):
                    results.append(
                        BuildResult(None, [], 8, "Worker process missing exported module", duration)
                    )
                else:
                    results.append(BuildResult(filename, [], error_no, message, duration))
            elif result.status == StatusKind.TIMEOUT:
                results.append(BuildResult(None, [], MeasureErrorNo.BUILD_TIMEOUT, "", timeout))
            else:
                # A crashed worker is an environment failure, eligible for retry.
                results.append(
                    BuildResult(None, [], 8, f"Worker process failed: {result.value!r}", timeout)
                )
        return results
