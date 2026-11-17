"""Shared data.sampling functions; import without CLI side effects."""

import json


def length_bucket(length):
    return "le512" if length <= 512 else "513to1024" if length <= 1024 else "gt1024"


def operator_family(text, key):
    prefix = text.split(json.loads(key)[0], 1)[0].lower()
    for name, markers in (
        ("batch_matmul", ("batch_matmul",)),
        ("matmul_dense", ("matmul", "dense")),
        ("convolution", ("conv",)),
        ("softmax", ("softmax",)),
        ("normalization", ("norm",)),
        ("pooling", ("pool",)),
        ("reduction", ("reduce", "sum", "max", "mean")),
        ("layout", ("transpose", "reshape", "layout", "concatenate")),
    ):
        if any(marker in prefix for marker in markers):
            return name
    return "elementwise_other"


def representative_indices(entries):
    """Choose shortest, median and longest complete sequences within one workload."""
    ordered = sorted(entries)
    selected = {}
    for role, offset in (("shortest", 0), ("median", len(ordered) // 2), ("longest", -1)):
        length, index, position = ordered[offset]
        item = selected.setdefault(index, {"length": length, "position": position, "roles": []})
        item["roles"].append(role)
    return selected
