"""Shared core.runtime functions; import without CLI side effects."""

from contextlib import contextmanager
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def bounded_pool(*args, **kwargs):
    from tvm.contrib.popen_pool import PopenPoolExecutor

    pool = PopenPoolExecutor(*args, **kwargs)
    try:
        yield pool
    finally:
        pool._threadpool.shutdown(wait=False, cancel_futures=True)
        with pool._lock:
            for worker in pool._worker_map.values():
                worker.kill()
            pool._worker_map.clear()
        pool._threadpool.shutdown(wait=True)
