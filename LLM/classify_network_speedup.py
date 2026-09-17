#!/usr/bin/env python3
"""Classify measured TVM workloads by network and report schedule speedups.

The reported values are task-level proxy metrics. They compare the best valid
latency in a baseline measurement history with the best valid latency in a new
history. They are not end-to-end Relay model benchmark results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import tempfile
from typing import Any, Iterable


TASK_SUFFIX = ".task.pkl"


def canonical_workload_key(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def discover_json_files(raw_paths: Iterable[str]) -> list[Path]:
    files: set[Path] = set()
    for raw_path in raw_paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(item for item in path.rglob("*.json") if item.is_file())
        else:
            raise FileNotFoundError(f"Measurement path does not exist: {path}")
    return sorted(files)


def read_best_latencies(paths: Iterable[str], upper_bound: float) -> tuple[dict[str, float], dict]:
    files = discover_json_files(paths)
    best: dict[str, float] = {}
    stats = {
        "files": len(files),
        "records": 0,
        "valid_records": 0,
        "invalid_measurements": 0,
        "malformed_records": 0,
    }

    for path in files:
        with path.open("r", encoding="utf-8") as source:
            for raw in source:
                if not raw.strip():
                    continue
                stats["records"] += 1
                try:
                    record = json.loads(raw)
                    key = canonical_workload_key(record["i"][0][0])
                    costs = record["r"][0]
                    error_no = int(record["r"][1])
                except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
                    stats["malformed_records"] += 1
                    continue

                valid_costs = []
                if error_no == 0 and isinstance(costs, list):
                    for cost in costs:
                        if isinstance(cost, (int, float)):
                            value = float(cost)
                            if math.isfinite(value) and 0.0 < value < upper_bound:
                                valid_costs.append(value)
                if not valid_costs:
                    stats["invalid_measurements"] += 1
                    continue

                latency = sum(valid_costs) / len(valid_costs)
                stats["valid_records"] += 1
                if latency < best.get(key, math.inf):
                    best[key] = latency

    stats["valid_workloads"] = len(best)
    return best, stats


def network_family(network: str) -> str:
    family = network.strip(" '\"")
    if family.startswith("onnx_"):
        family = re.sub(
            r"_(?:b\d+_)?seq\d+"
            r"(?:_(?:float16|float32|float64|bfloat16))?"
            r"(?:_(?:full_logits|last_token))?$",
            "",
            family,
        )
    return family


def parse_task_filename(path: Path) -> tuple[str, str, str]:
    name = path.name
    stem = name[: -len(TASK_SUFFIX)] if name.endswith(TASK_SUFFIX) else path.stem
    match = re.match(r"^\(\(([^,]+),(.+)\),([^,]+)\)$", stem)
    if not match:
        return stem, "", ""
    return tuple(value.strip() for value in match.groups())


def discover_task_files(network_info_dirs: Iterable[str]) -> list[Path]:
    files: set[Path] = set()
    for raw_dir in network_info_dirs:
        directory = Path(raw_dir).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"Network-info directory does not exist: {directory}")
        files.update(item for item in directory.rglob(f"*{TASK_SUFFIX}") if item.is_file())
    return sorted(files, key=lambda path: path.name)


def classify_status(speedup: float, tolerance: float) -> str:
    if speedup > 1.0 + tolerance:
        return "improved"
    if speedup < 1.0 / (1.0 + tolerance):
        return "regressed"
    return "tied"


def load_registry_entries(
    network_info_dirs: Iterable[str],
    baseline: dict[str, float],
    candidate: dict[str, float],
    tie_tolerance: float,
) -> tuple[list[dict], dict]:
    entries: list[dict] = []
    registry_workloads: set[str] = set()
    task_files = discover_task_files(network_info_dirs)

    for task_file in task_files:
        network, input_spec, target_kind = parse_task_filename(task_file)
        family = network_family(network)
        with task_file.open("rb") as source:
            tasks, task_weights = pickle.load(source)
        if len(tasks) != len(task_weights):
            raise ValueError(f"Task/weight length mismatch in {task_file}")

        variant = task_file.name[: -len(TASK_SUFFIX)]
        for task_index, (task, raw_weight) in enumerate(zip(tasks, task_weights)):
            key = canonical_workload_key(task.workload_key)
            registry_workloads.add(key)
            baseline_latency = baseline.get(key)
            candidate_latency = candidate.get(key)
            comparable = baseline_latency is not None and candidate_latency is not None
            speedup = baseline_latency / candidate_latency if comparable else None
            entries.append(
                {
                    "network": family,
                    "variant": variant,
                    "network_name": network,
                    "input_spec": input_spec,
                    "target_kind": target_kind,
                    "task_file": str(task_file),
                    "task_index": task_index,
                    "task_weight": float(raw_weight),
                    "workload_key": key,
                    "baseline_available": baseline_latency is not None,
                    "lapo_available": candidate_latency is not None,
                    "baseline_latency_ms": baseline_latency * 1000.0 if baseline_latency is not None else None,
                    "lapo_latency_ms": candidate_latency * 1000.0 if candidate_latency is not None else None,
                    "speedup": speedup,
                    "status": classify_status(speedup, tie_tolerance) if speedup is not None else "not_comparable",
                }
            )

    registry_stats = {
        "task_files": len(task_files),
        "task_occurrences": len(entries),
        "unique_workloads": len(registry_workloads),
        "network_families": len({entry["network"] for entry in entries}),
        "registry_workload_keys": registry_workloads,
    }
    return entries, registry_stats


def geomean(values: list[float], weights: list[float] | None = None) -> float | None:
    if not values:
        return None
    if weights is None:
        return math.exp(sum(math.log(value) for value in values) / len(values))
    total_weight = sum(weights)
    if total_weight <= 0:
        return None
    return math.exp(
        sum(weight * math.log(value) for value, weight in zip(values, weights)) / total_weight
    )


def summarize_entries(entries: list[dict]) -> dict:
    comparable = [entry for entry in entries if entry["speedup"] is not None]
    speedups = [entry["speedup"] for entry in comparable]
    weights = [entry["task_weight"] for entry in comparable]
    sorted_speedups = sorted(speedups)
    midpoint = len(sorted_speedups) // 2
    if not sorted_speedups:
        median = None
    elif len(sorted_speedups) % 2:
        median = sorted_speedups[midpoint]
    else:
        median = (sorted_speedups[midpoint - 1] + sorted_speedups[midpoint]) / 2.0

    baseline_sum = sum(entry["baseline_latency_ms"] * entry["task_weight"] for entry in comparable)
    lapo_sum = sum(entry["lapo_latency_ms"] * entry["task_weight"] for entry in comparable)
    variants = {entry["variant"] for entry in entries}
    complete_variants = 0
    for variant in variants:
        variant_entries = [entry for entry in entries if entry["variant"] == variant]
        if variant_entries and all(entry["speedup"] is not None for entry in variant_entries):
            complete_variants += 1

    return {
        "variant_count": len(variants),
        "complete_variant_count": complete_variants,
        "task_occurrences": len(entries),
        "unique_workloads": len({entry["workload_key"] for entry in entries}),
        "baseline_covered": sum(entry["baseline_available"] for entry in entries),
        "lapo_covered": sum(entry["lapo_available"] for entry in entries),
        "comparable_workloads": len(comparable),
        "coverage_ratio": len(comparable) / len(entries) if entries else 0.0,
        "improved": sum(entry["status"] == "improved" for entry in comparable),
        "regressed": sum(entry["status"] == "regressed" for entry in comparable),
        "tied": sum(entry["status"] == "tied" for entry in comparable),
        "geomean_speedup": geomean(speedups),
        "weighted_geomean_speedup": geomean(speedups, weights),
        "median_speedup": median,
        "min_speedup": min(speedups) if speedups else None,
        "max_speedup": max(speedups) if speedups else None,
        "baseline_weighted_latency_sum_ms": baseline_sum if comparable else None,
        "lapo_weighted_latency_sum_ms": lapo_sum if comparable else None,
        "proxy_weighted_sum_speedup": baseline_sum / lapo_sum if lapo_sum > 0 else None,
    }


def grouped_summaries(entries: list[dict], group_key: str) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for entry in entries:
        groups.setdefault(entry[group_key], []).append(entry)
    summaries = []
    for name in sorted(groups):
        summary = {group_key: name}
        if group_key == "variant":
            first = groups[name][0]
            summary.update(
                network=first["network"],
                network_name=first["network_name"],
                input_spec=first["input_spec"],
                target_kind=first["target_kind"],
                task_file=first["task_file"],
            )
        summary.update(summarize_entries(groups[name]))
        summaries.append(summary)
    return summaries


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(content)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        atomic_write_text(path, "")
        return
    fieldnames = list(rows[0])
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as output:
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


def format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def build_log(report: dict) -> str:
    lines = [
        "Task-level proxy speedup report",
        "IMPORTANT: These are workload schedule metrics, not end-to-end Relay latency speedups.",
        "",
        f"Baseline: {', '.join(report['sources']['baseline_paths'])}",
        f"LAPO: {', '.join(report['sources']['lapo_paths'])}",
        f"Registry: {', '.join(report['sources']['network_info_dirs'])}",
        f"Registry variants: {report['registry']['task_files']}",
        f"Registry networks: {report['registry']['network_families']}",
        f"Registry unique workloads: {report['registry']['unique_workloads']}",
        f"LAPO valid workloads: {report['lapo_measurements']['valid_workloads']}",
        f"LAPO mapped workloads: {report['classification']['lapo_mapped_workloads']}",
        f"LAPO unmapped workloads: {report['classification']['lapo_unmapped_workloads']}",
        "",
        "Per-network metrics:",
    ]
    for summary in report["network_summaries"]:
        lines.append(
            "  {network}: variants={variant_count}, comparable={comparable_workloads}/{task_occurrences} "
            "({coverage_ratio:.2%}), geomean={geomean}, weighted_geomean={weighted}, "
            "proxy_sum={proxy}, improved/regressed/tied={improved}/{regressed}/{tied}".format(
                network=summary["network"],
                variant_count=summary["variant_count"],
                comparable_workloads=summary["comparable_workloads"],
                task_occurrences=summary["task_occurrences"],
                coverage_ratio=summary["coverage_ratio"],
                geomean=format_value(summary["geomean_speedup"]),
                weighted=format_value(summary["weighted_geomean_speedup"]),
                proxy=format_value(summary["proxy_weighted_sum_speedup"]),
                improved=summary["improved"],
                regressed=summary["regressed"],
                tied=summary["tied"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-path",
        action="append",
        required=True,
        help="Baseline measurement JSONL file or directory; repeatable",
    )
    parser.add_argument(
        "--lapo-path",
        action="append",
        required=True,
        help="New LAPO measurement JSONL file or directory; repeatable",
    )
    parser.add_argument(
        "--network-info-dir",
        action="append",
        required=True,
        help="Directory containing *.task.pkl registry files; repeatable",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--valid-cost-upper-bound", type=float, default=1e9)
    parser.add_argument(
        "--tie-tolerance",
        type=float,
        default=0.01,
        help="Relative tolerance around 1x used to classify ties (default: 1%%)",
    )
    args = parser.parse_args()
    if args.valid_cost_upper_bound <= 0:
        parser.error("--valid-cost-upper-bound must be positive")
    if args.tie_tolerance < 0:
        parser.error("--tie-tolerance must be non-negative")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline, baseline_stats = read_best_latencies(args.baseline_path, args.valid_cost_upper_bound)
    lapo, lapo_stats = read_best_latencies(args.lapo_path, args.valid_cost_upper_bound)
    entries, registry_stats = load_registry_entries(
        args.network_info_dir, baseline, lapo, args.tie_tolerance
    )
    registry_keys = registry_stats.pop("registry_workload_keys")
    network_summaries = grouped_summaries(entries, "network")
    variant_summaries = grouped_summaries(entries, "variant")

    report = {
        "metric_scope": "task_level_proxy",
        "metric_definition": "best valid baseline latency / best valid LAPO latency",
        "warning": "This report is not an end-to-end Relay model latency benchmark.",
        "tie_tolerance": args.tie_tolerance,
        "valid_cost_upper_bound_seconds": args.valid_cost_upper_bound,
        "sources": {
            "baseline_paths": [str(Path(path).expanduser().resolve()) for path in args.baseline_path],
            "lapo_paths": [str(Path(path).expanduser().resolve()) for path in args.lapo_path],
            "network_info_dirs": [
                str(Path(path).expanduser().resolve()) for path in args.network_info_dir
            ],
        },
        "baseline_measurements": baseline_stats,
        "lapo_measurements": lapo_stats,
        "registry": registry_stats,
        "classification": {
            "lapo_mapped_workloads": len(set(lapo) & registry_keys),
            "lapo_unmapped_workloads": len(set(lapo) - registry_keys),
            "baseline_mapped_workloads": len(set(baseline) & registry_keys),
            "baseline_unmapped_workloads": len(set(baseline) - registry_keys),
            "lapo_unmapped_workload_keys": sorted(set(lapo) - registry_keys),
        },
        "network_summaries": network_summaries,
        "variant_summaries": variant_summaries,
    }

    write_csv(output_dir / "network_speedup_summary.csv", network_summaries)
    write_csv(output_dir / "variant_speedup_summary.csv", variant_summaries)
    detail_columns = [
        "network",
        "variant",
        "network_name",
        "input_spec",
        "target_kind",
        "task_index",
        "task_weight",
        "workload_key",
        "baseline_available",
        "lapo_available",
        "baseline_latency_ms",
        "lapo_latency_ms",
        "speedup",
        "status",
        "task_file",
    ]
    write_csv(
        output_dir / "workload_speedup_details.csv",
        [{column: entry[column] for column in detail_columns} for entry in entries],
    )
    atomic_write_text(
        output_dir / "network_speedup_report.json",
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
    )
    log = build_log(report)
    atomic_write_text(output_dir / "classification.log", log)
    print(log, end="")
    print(f"Reports: {output_dir}")


if __name__ == "__main__":
    main()
