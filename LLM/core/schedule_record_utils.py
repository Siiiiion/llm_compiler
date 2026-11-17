"""Record identities and durable files shared by schedule audit and dataset creation."""

from LLM.core.io import atomic_json as atomic_json, file_sha256 as file_sha256


import copy
import hashlib
import json
import math
import os
from pathlib import Path


AUDIT_SCHEMA = 1
FAILED_COST = 1e9


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def workload_key(value):
    return canonical_json(json.loads(value) if isinstance(value, str) else value)


def input_hash(record):
    """Include every task/state field and the record version, excluding measurement results."""
    measure_input = copy.deepcopy(record["i"])
    measure_input[0][0] = json.loads(workload_key(measure_input[0][0]))
    if len(measure_input) != 2 or not isinstance(measure_input[1][1], list):
        raise ValueError("Invalid measure input/state")
    return digest({"i": measure_input, "v": record.get("v")})


def normalize_record(record):
    """Return the exact normalization used for schedule training, without mutating input."""
    record = copy.deepcopy(record)
    record["i"][0][0] = json.loads(workload_key(record["i"][0][0]))
    ppt_idx = None
    for index, step in enumerate(record["i"][1][1]):
        if step[0] == "SP":
            step[4][:] = [1] * len(step[4])
        elif step[0] == "PPT" and ppt_idx is None:
            ppt_idx = index
    return record, ppt_idx


def measured_latency(record):
    """A failed, partial, non-finite or placeholder measurement is never performance data."""
    result = record.get("r", [])
    if len(result) < 2 or result[1] != 0 or not result[0]:
        return None
    costs = result[0]
    if not isinstance(costs, list):
        return None
    if any(
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(cost)
        or not 0 < cost < FAILED_COST
        for cost in costs
    ):
        return None
    return math.fsum(costs) / len(costs)


def read_journal(path, repair=False):
    """Only a torn final write is recoverable; corruption of a completed line is fatal."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r+b" if repair else "rb") as stream:
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line:
                return rows
            if not line.endswith(b"\n"):
                if not repair:
                    raise ValueError(f"Incomplete audit journal: {path}")
                stream.truncate(position)
                stream.flush()
                os.fsync(stream.fileno())
                return rows
            rows.append(json.loads(line))


def append_journal(path, rows):
    with Path(path).open("a") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def classify_build_error(error_no, message):
    if error_no == 0:
        return "passed"
    if error_no == 6:
        return "timeout"
    message = message.lower()
    environment_markers = (
        "out of memory",
        "cannot allocate memory",
        "no space left",
        "disk quota",
        "no such file or directory",
        "not found",
        "is not enabled",
        "cuda path",
        "cannot find",
        "worker process",
        "childprocesserror",
        "brokenpipe",
        "eoferror",
        "permission denied",
    )
    if any(marker in message for marker in environment_markers):
        return "environment_error"
    # This is a toolchain rejection, not a claim about semantic invalidity.
    return "state_error" if error_no == 1 else "build_error"
