#!/usr/bin/env python3
"""Find missing AutoScheduler history for failed Relay benchmark variants.

The script uses the ``*.task.pkl`` files created next to Relay artifacts, so it
does not need to extract Relay tasks again.  It reports whether each missing
workload already has an unmeasured candidate in one or more ``lapo_gen.json``
files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import tvm


TASK_SUFFIX = ".task.pkl"
SUCCESS_STATUSES = {"ok"}


def canonical_workload_key(value: Any) -> str:
    """Normalize both encoded and decoded TVM workload keys."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def canonical_target(value: Any) -> str:
    return str(value if isinstance(value, tvm.target.Target) else tvm.target.Target(value))


def valid_latency(record: dict, upper_bound: float) -> float | None:
    try:
        result = record["r"]
        costs = result[0]
        error_no = int(result[1])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if error_no != 0 or not isinstance(costs, list):
        return None
    valid = []
    for cost in costs:
        if isinstance(cost, (int, float)):
            value = float(cost)
            if math.isfinite(value) and 0.0 < value < upper_bound:
                valid.append(value)
    return min(valid) if valid else None


def resolve_record_files(raw_paths: Iterable[str], directory_pattern: str) -> list[Path]:
    files: set[Path] = set()
    for raw_path in raw_paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(item for item in path.rglob(directory_pattern) if item.is_file())
        else:
            raise FileNotFoundError(f"Record path does not exist: {path}")
    return sorted(files)


def read_record_index(
    paths: Iterable[Path], expected_target: str, upper_bound: float
) -> tuple[dict[str, dict], dict]:
    index: dict[str, dict] = {}
    stats = {
        "files": 0,
        "records": 0,
        "target_records": 0,
        "valid_records": 0,
        "invalid_records": 0,
        "wrong_target_records": 0,
        "malformed_records": 0,
    }
    resolved_paths = list(paths)
    stats["files"] = len(resolved_paths)
    for path in resolved_paths:
        with path.open("r", encoding="utf-8") as source:
            for raw in source:
                if not raw.strip():
                    continue
                stats["records"] += 1
                try:
                    record = json.loads(raw)
                    key = canonical_workload_key(record["i"][0][0])
                    record_target = canonical_target(record["i"][0][1])
                except (
                    json.JSONDecodeError,
                    KeyError,
                    IndexError,
                    TypeError,
                    ValueError,
                    tvm.error.TVMError,
                ):
                    stats["malformed_records"] += 1
                    continue
                if record_target != expected_target:
                    stats["wrong_target_records"] += 1
                    continue

                stats["target_records"] += 1
                item = index.setdefault(
                    key,
                    {
                        "records": 0,
                        "valid_records": 0,
                        "invalid_records": 0,
                        "best_latency_seconds": None,
                        "source_counts": Counter(),
                    },
                )
                item["records"] += 1
                item["source_counts"][str(path)] += 1
                latency = valid_latency(record, upper_bound)
                if latency is None:
                    item["invalid_records"] += 1
                    stats["invalid_records"] += 1
                else:
                    item["valid_records"] += 1
                    stats["valid_records"] += 1
                    previous = item["best_latency_seconds"]
                    item["best_latency_seconds"] = (
                        latency if previous is None else min(previous, latency)
                    )
    stats["workloads"] = len(index)
    stats["valid_workloads"] = sum(item["valid_records"] > 0 for item in index.values())
    return index, stats


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as source:
        summary = json.load(source)
    if not isinstance(summary.get("network_summaries"), list):
        raise ValueError(f"Benchmark summary has no network_summaries list: {path}")
    return summary


def select_variants(summary: dict, selection: str) -> tuple[dict[str, list[dict]], dict]:
    selected: dict[str, list[dict]] = {}
    reasons: dict[str, list[str]] = {}
    for network_summary in summary["network_summaries"]:
        network = network_summary["network"]
        variants = network_summary.get("variants", [])
        failed = [
            item
            for item in variants
            if item.get("status") not in SUCCESS_STATUSES or item.get("returncode", 0) != 0
        ]
        invalid = [
            item
            for item in variants
            if item.get("status") in SUCCESS_STATUSES
            and item.get("valid_comparison") is False
        ]
        items = list(failed)
        network_reasons = []
        if failed:
            network_reasons.append("failed_variant")
        if selection == "failed-or-invalid" and invalid:
            network_reasons.append("invalid_comparison")
            seen = {item.get("variant") for item in items}
            items.extend(item for item in invalid if item.get("variant") not in seen)
        if items:
            selected[network] = items
            reasons[network] = network_reasons
    return selected, reasons


def parse_task_filename(path: Path) -> tuple[str, str]:
    name = path.name
    stem = name[: -len(TASK_SUFFIX)] if name.endswith(TASK_SUFFIX) else path.stem
    match = re.match(r"^\(\(([^,]+),(.+)\),([^,]+)\)$", stem)
    if not match:
        raise ValueError(f"Cannot parse task registry filename: {path}")
    network_name, input_spec, target_kind = (value.strip() for value in match.groups())
    return f"({network_name},{input_spec})", target_kind


def discover_task_files(network_info_dirs: Iterable[str]) -> dict[str, Path]:
    by_variant: dict[str, Path] = {}
    for raw_dir in network_info_dirs:
        directory = Path(raw_dir).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"Network-info directory does not exist: {directory}")
        for path in sorted(directory.rglob(f"*{TASK_SUFFIX}")):
            variant, _ = parse_task_filename(path)
            previous = by_variant.get(variant)
            if previous is not None and previous != path:
                raise ValueError(
                    f"Duplicate task registry for variant {variant}: {previous} and {path}"
                )
            by_variant[variant] = path
    return by_variant


def load_selected_tasks(
    selected: dict[str, list[dict]], task_files: dict[str, Path]
) -> tuple[dict[str, dict[str, dict]], list[dict]]:
    network_workloads: dict[str, dict[str, dict]] = {}
    missing_task_files = []
    for network, variants in selected.items():
        workload_map = network_workloads.setdefault(network, {})
        for variant_result in variants:
            variant = variant_result.get("variant")
            task_path = task_files.get(variant)
            if task_path is None:
                missing_task_files.append({"network": network, "variant": variant})
                continue
            with task_path.open("rb") as source:
                tasks, task_weights = pickle.load(source)
            if len(tasks) != len(task_weights):
                raise ValueError(f"Task/weight length mismatch in {task_path}")
            for task_index, (task, raw_weight) in enumerate(zip(tasks, task_weights)):
                key = canonical_workload_key(task.workload_key)
                item = workload_map.setdefault(
                    key,
                    {
                        "occurrences": 0,
                        "task_weight": 0.0,
                        "variants": set(),
                        "task_locations": [],
                    },
                )
                item["occurrences"] += 1
                item["task_weight"] += float(raw_weight)
                item["variants"].add(variant)
                item["task_locations"].append(
                    {"variant": variant, "task_index": task_index, "task_weight": float(raw_weight)}
                )
    return network_workloads, missing_task_files


def source_counts(item: dict | None) -> list[dict]:
    if not item:
        return []
    return [
        {"path": path, "records": count}
        for path, count in sorted(item["source_counts"].items())
    ]


def build_network_reports(
    network_workloads: dict[str, dict[str, dict]],
    reasons: dict[str, list[str]],
    selected: dict[str, list[dict]],
    measured_index: dict[str, dict],
    candidate_index: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    summaries = []
    details = []
    for network in sorted(network_workloads):
        workload_map = network_workloads[network]
        missing_items = []
        for key, task_info in sorted(workload_map.items()):
            measured = measured_index.get(key)
            if measured and measured["valid_records"] > 0:
                continue
            candidate = candidate_index.get(key)
            candidate_available = bool(candidate and candidate["records"] > 0)
            measured_records = measured["records"] if measured else 0
            status = (
                "candidate_available_unmeasured"
                if candidate_available
                else (
                    "measured_only_invalid_no_shard_candidate"
                    if measured_records
                    else "no_candidate"
                )
            )
            detail = {
                "network": network,
                "workload_key": key,
                "task_occurrences": task_info["occurrences"],
                "task_weight": task_info["task_weight"],
                "variants": sorted(task_info["variants"]),
                "task_locations": task_info["task_locations"],
                "lapo_measured_records": measured_records,
                "lapo_measured_valid_records": measured["valid_records"] if measured else 0,
                "lapo_measured_invalid_records": measured["invalid_records"] if measured else 0,
                "lapo_measured_sources": source_counts(measured),
                "shard_candidate_records": candidate["records"] if candidate else 0,
                "shard_candidate_valid_latency_records": (
                    candidate["valid_records"] if candidate else 0
                ),
                "shard_candidate_sources": source_counts(candidate),
                "candidate_available": candidate_available,
                "status": status,
            }
            details.append(detail)
            missing_items.append(detail)

        unique_workloads = len(workload_map)
        missing_count = len(missing_items)
        summaries.append(
            {
                "network": network,
                "selection_reasons": reasons.get(network, []),
                "selected_variants": len(selected.get(network, [])),
                "task_occurrences": sum(item["occurrences"] for item in workload_map.values()),
                "unique_workloads": unique_workloads,
                "valid_history_workloads": unique_workloads - missing_count,
                "missing_workloads": missing_count,
                "missing_with_candidate": sum(
                    item["candidate_available"] for item in missing_items
                ),
                "missing_without_candidate": sum(
                    not item["candidate_available"] for item in missing_items
                ),
                "missing_task_weight": sum(item["task_weight"] for item in missing_items),
            }
        )
    return summaries, details


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(content)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_log(report: dict) -> str:
    lines = [
        "Missing AutoScheduler history audit",
        f"Target: {report['target']}",
        f"Selection: {report['selection']}",
        f"Analyzed networks: {len(report['network_summaries'])}",
        f"Measured files: {len(report['sources']['lapo_measured'])}",
        f"Candidate files: {len(report['sources']['candidates'])}",
        "",
        "Per-network summary:",
    ]
    for item in report["network_summaries"]:
        lines.append(
            "  {network}: missing={missing}/{total}, with_candidate={with_candidate}, "
            "without_candidate={without_candidate}, variants={variants}, reasons={reasons}".format(
                network=item["network"],
                missing=item["missing_workloads"],
                total=item["unique_workloads"],
                with_candidate=item["missing_with_candidate"],
                without_candidate=item["missing_without_candidate"],
                variants=item["selected_variants"],
                reasons=",".join(item["selection_reasons"]),
            )
        )
    if report["missing_task_files"]:
        lines.extend(["", "Missing task registry files:"])
        for item in report["missing_task_files"]:
            lines.append(f"  {item['network']}: {item['variant']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-summary", required=True)
    parser.add_argument(
        "--network-info-dir",
        action="append",
        required=True,
        help="Directory containing *.task.pkl files; repeatable",
    )
    parser.add_argument(
        "--lapo-measured",
        action="append",
        required=True,
        help="Measured JSONL file or directory; repeatable",
    )
    parser.add_argument(
        "--candidate-path",
        action="append",
        required=True,
        help="lapo_gen.json file or parent directory; repeatable",
    )
    parser.add_argument(
        "--candidate-name",
        default="lapo_gen.json",
        help="Filename selected recursively below candidate directories",
    )
    parser.add_argument(
        "--selection",
        choices=["failed", "failed-or-invalid"],
        default="failed-or-invalid",
        help="Also include status=ok variants rejected as invalid comparisons (default: yes)",
    )
    parser.add_argument("--target", help="Defaults to target stored in benchmark summary")
    parser.add_argument("--valid-cost-upper-bound", type=float, default=1e9)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.valid_cost_upper_bound <= 0:
        parser.error("--valid-cost-upper-bound must be positive")

    summary_path = Path(args.benchmark_summary).expanduser().resolve()
    benchmark_summary = load_summary(summary_path)
    target = args.target or benchmark_summary.get("target")
    if not target:
        parser.error("--target is required when benchmark summary has no target")
    expected_target = canonical_target(target)

    measured_files = resolve_record_files(args.lapo_measured, "*.json")
    candidate_files = resolve_record_files(args.candidate_path, args.candidate_name)
    if not measured_files:
        raise FileNotFoundError("No measured JSON files found")
    if not candidate_files:
        raise FileNotFoundError("No candidate files found")

    selected, reasons = select_variants(benchmark_summary, args.selection)
    task_files = discover_task_files(args.network_info_dir)
    network_workloads, missing_task_files = load_selected_tasks(selected, task_files)
    measured_index, measured_stats = read_record_index(
        measured_files, expected_target, args.valid_cost_upper_bound
    )
    candidate_index, candidate_stats = read_record_index(
        candidate_files, expected_target, args.valid_cost_upper_bound
    )
    network_summaries, missing_details = build_network_reports(
        network_workloads, reasons, selected, measured_index, candidate_index
    )

    report = {
        "target": expected_target,
        "selection": args.selection,
        "sources": {
            "benchmark_summary": str(summary_path),
            "network_info_dirs": [
                str(Path(path).expanduser().resolve()) for path in args.network_info_dir
            ],
            "lapo_measured": [str(path) for path in measured_files],
            "candidates": [str(path) for path in candidate_files],
        },
        "record_scan": {
            "lapo_measured": measured_stats,
            "candidates": candidate_stats,
        },
        "selected_networks": len(selected),
        "selected_variants": sum(len(items) for items in selected.values()),
        "missing_task_files": missing_task_files,
        "network_summaries": network_summaries,
        "missing_workloads": missing_details,
    }

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        output_dir / "missing_history_report.json",
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
    )
    network_columns = [
        "network",
        "selection_reasons",
        "selected_variants",
        "task_occurrences",
        "unique_workloads",
        "valid_history_workloads",
        "missing_workloads",
        "missing_with_candidate",
        "missing_without_candidate",
        "missing_task_weight",
    ]
    write_csv(
        output_dir / "network_missing_history.csv",
        network_columns,
        [
            {
                **item,
                "selection_reasons": "|".join(item["selection_reasons"]),
            }
            for item in network_summaries
        ],
    )
    detail_columns = [
        "network",
        "workload_key",
        "task_occurrences",
        "task_weight",
        "variants",
        "lapo_measured_records",
        "lapo_measured_valid_records",
        "lapo_measured_invalid_records",
        "shard_candidate_records",
        "shard_candidate_valid_latency_records",
        "candidate_available",
        "candidate_files",
        "status",
    ]
    detail_rows = []
    for item in missing_details:
        detail_rows.append(
            {
                "network": item["network"],
                "workload_key": item["workload_key"],
                "task_occurrences": item["task_occurrences"],
                "task_weight": item["task_weight"],
                "variants": "|".join(item["variants"]),
                "lapo_measured_records": item["lapo_measured_records"],
                "lapo_measured_valid_records": item["lapo_measured_valid_records"],
                "lapo_measured_invalid_records": item["lapo_measured_invalid_records"],
                "shard_candidate_records": item["shard_candidate_records"],
                "shard_candidate_valid_latency_records": item[
                    "shard_candidate_valid_latency_records"
                ],
                "candidate_available": item["candidate_available"],
                "candidate_files": "|".join(
                    source["path"] for source in item["shard_candidate_sources"]
                ),
                "status": item["status"],
            }
        )
    write_csv(output_dir / "missing_workload_details.csv", detail_columns, detail_rows)
    log = build_log(report)
    atomic_write_text(output_dir / "missing_history.log", log)
    print(log, end="")
    print(f"Reports: {output_dir}")
    if missing_task_files:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
