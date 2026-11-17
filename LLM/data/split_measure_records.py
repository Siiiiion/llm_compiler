#!/usr/bin/env python3
"""Split a merged TVM measurement JSONL file into one file per workload."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import os
import shutil
import tempfile

from LLM.core.common import clean_name


def _canonical_workload_key(value) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def _target_kind(record: dict, override: str | None) -> str:
    if override:
        return override
    target = record["i"][0][1]
    if not isinstance(target, str) or not target.strip():
        raise ValueError("record does not contain a valid target string")
    return target.split()[0]


def _output_name(workload_key: str, target_kind: str) -> str:
    name = f"{clean_name((workload_key, target_kind))}.json"
    if len(os.fsencode(name)) <= 240:
        return name
    digest = hashlib.sha256(f"{workload_key}|{target_kind}".encode()).hexdigest()[:24]
    return f"workload-{digest}-{target_kind}.json"


def split_records(
    input_path: str,
    output_dir: str,
    target_kind: str | None,
    overwrite: bool,
    max_open_files: int,
) -> dict:
    input_path = os.path.abspath(input_path)
    output_dir = os.path.abspath(output_dir)
    parent = os.path.dirname(output_dir)
    os.makedirs(parent, exist_ok=True)

    if os.path.exists(output_dir) and os.listdir(output_dir) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Pass --overwrite to replace it."
        )

    tmp_dir = tempfile.mkdtemp(prefix=f".{os.path.basename(output_dir)}-", dir=parent)
    handles: OrderedDict[str, object] = OrderedDict()
    paths_by_workload: dict[tuple[str, str], str] = {}
    records = 0
    invalid = 0

    try:
        with open(input_path, "r") as source:
            for line_number, raw in enumerate(source, start=1):
                if not raw.strip():
                    continue
                try:
                    record = json.loads(raw)
                    workload_key = _canonical_workload_key(record["i"][0][0])
                    kind = _target_kind(record, target_kind)
                except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
                    invalid += 1
                    continue

                group = (workload_key, kind)
                path = paths_by_workload.get(group)
                if path is None:
                    path = os.path.join(tmp_dir, _output_name(workload_key, kind))
                    paths_by_workload[group] = path

                handle = handles.pop(path, None)
                if handle is None:
                    handle = open(path, "a")
                handles[path] = handle
                handle.write(raw if raw.endswith("\n") else raw + "\n")
                records += 1

                if len(handles) > max_open_files:
                    _, old_handle = handles.popitem(last=False)
                    old_handle.close()

        for handle in handles.values():
            handle.close()
        handles.clear()

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.replace(tmp_dir, output_dir)
    except Exception:
        for handle in handles.values():
            handle.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return {
        "input_path": input_path,
        "output_dir": output_dir,
        "records_written": records,
        "invalid_records": invalid,
        "workload_files": len(paths_by_workload),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True, help="Merged TVM measurement JSONL")
    parser.add_argument("--output-dir", required=True, help="Per-workload output directory")
    parser.add_argument("--target-kind", default=None, help="Override target kind, e.g. cuda")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace a non-empty output directory"
    )
    parser.add_argument("--max-open-files", type=int, default=64)
    args = parser.parse_args()
    if args.max_open_files <= 0:
        parser.error("--max-open-files must be positive")

    stats = split_records(
        args.input_path,
        args.output_dir,
        args.target_kind,
        args.overwrite,
        args.max_open_files,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
