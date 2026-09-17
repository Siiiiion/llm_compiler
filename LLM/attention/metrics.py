"""Small shared helpers for attention benchmark reports."""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import numpy as np


def timing_statistics(samples_ms):
    values = [float(value) for value in samples_ms]
    return {
        "samples_ms": values,
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "p95_ms": float(np.percentile(values, 95)),
        "stdev_ms": statistics.pstdev(values),
    }


def compare_arrays(reference, candidate, rtol=5e-2, atol=5e-2):
    shape_matches = reference.shape == candidate.shape
    if not shape_matches:
        return {
            "passed": False,
            "shape_matches": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    expected = reference.astype("float64", copy=False)
    actual = candidate.astype("float64", copy=False)
    delta = actual - expected
    expected_flat = expected.reshape(-1)
    actual_flat = actual.reshape(-1)
    denominator = np.linalg.norm(expected_flat) * np.linalg.norm(actual_flat)
    cosine = float(np.dot(expected_flat, actual_flat) / denominator) if denominator else 1.0
    return {
        "passed": bool(np.allclose(expected, actual, rtol=rtol, atol=atol)),
        "shape_matches": True,
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "max_abs_error": float(np.max(np.abs(delta))),
        "mean_abs_error": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "cosine_similarity": cosine,
        "rtol": rtol,
        "atol": atol,
    }


def atomic_json(path, value):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8") as file_obj:
        json.dump(value, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    os.replace(temporary, path)
