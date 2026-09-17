#!/usr/bin/env python3
"""Run benchmark_relay.py for many Relay networks and aggregate by network family."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network-info-dir",
        action="append",
        required=True,
        help="Directory containing *.relay.pkl artifacts; repeatable.",
    )
    parser.add_argument("--history", action="append", default=[], help="History JSONL; repeatable.")
    parser.add_argument(
        "--history-dir",
        action="append",
        default=[],
        help="Add every top-level *.json file in this directory as history; repeatable.",
    )
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--mode", choices=["both", "baseline", "optimized"], default="both")
    parser.add_argument("--network", action="append", default=[], help="Exact network family; repeatable.")
    parser.add_argument("--include", help="Regex applied to the Relay artifact basename.")
    parser.add_argument("--exclude", help="Regex applied to the Relay artifact basename.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--max-models", type=int)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--number", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--min-repeat-ms", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--allow-incomplete-history", action="store_true")
    parser.add_argument(
        "--cache-libraries",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist compiled .so files. Disabled by default because full suites are large.",
    )
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--benchmark-script",
        default=str(Path(__file__).with_name("benchmark_relay.py")),
    )
    parser.add_argument("--summary", help="Defaults to <output-dir>/summary.json")
    parser.add_argument("--log-file", help="Defaults to <output-dir>/batch_benchmark.log")
    args = parser.parse_args()

    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.end_index is not None and args.end_index < args.start_index:
        parser.error("--end-index must be >= --start-index")
    if args.max_models is not None and args.max_models <= 0:
        parser.error("--max-models must be positive")
    if args.mode in ("both", "optimized") and not (args.history or args.history_dir):
        parser.error("optimized benchmarking requires --history and/or --history-dir")
    return args


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, path)


def file_fingerprint(paths: Iterable[Path], extra: Iterable[str] = ()) -> str:
    digest = hashlib.sha256()
    for value in extra:
        digest.update(value.encode())
        digest.update(b"\0")
    for path in sorted(paths):
        stat = path.stat()
        digest.update(str(path.resolve()).encode())
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode())
    return digest.hexdigest()[:16]


def artifact_stem(path: Path) -> str:
    name = path.name
    return name[: -len(".relay.pkl")] if name.endswith(".relay.pkl") else path.stem


def network_family(path: Path) -> str:
    stem = artifact_stem(path)
    match = re.match(r"^\(([^,]+),", stem)
    family = match.group(1) if match else stem
    family = family.strip(" '\"")
    if family.startswith("onnx_"):
        family = re.sub(
            r"_(?:b\d+_)?seq\d+"
            r"(?:_(?:float16|float32|float64|bfloat16))?"
            r"(?:_(?:full_logits|last_token))?$",
            "",
            family,
        )
    return family


def slug(value: str, limit: int = 90) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    result = result or "network"
    digest = hashlib.sha256(value.encode()).hexdigest()[:10]
    return f"{result[:limit]}-{digest}"


def discover_artifacts(args: argparse.Namespace) -> list[Path]:
    files: set[Path] = set()
    for raw_dir in args.network_info_dir:
        directory = Path(raw_dir).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"Network-info directory does not exist: {directory}")
        files.update(directory.glob("*.relay.pkl"))

    include = re.compile(args.include) if args.include else None
    exclude = re.compile(args.exclude) if args.exclude else None
    requested = set(args.network)
    selected = []
    for path in sorted(files, key=lambda item: (network_family(item), item.name)):
        if requested and network_family(path) not in requested:
            continue
        if include and not include.search(path.name):
            continue
        if exclude and exclude.search(path.name):
            continue
        selected.append(path)

    selected = selected[args.start_index : args.end_index]
    if args.max_models is not None:
        selected = selected[: args.max_models]
    return selected


def discover_histories(args: argparse.Namespace) -> list[Path]:
    paths = {Path(path).expanduser().resolve() for path in args.history}
    for raw_dir in args.history_dir:
        directory = Path(raw_dir).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"History directory does not exist: {directory}")
        paths.update(directory.glob("*.json"))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"History files do not exist: {missing}")
    return sorted(paths)


def load_json(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as file:
            value = json.load(file)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def variant_result(path: Path, report_path: Path, wall_seconds: float, returncode: int) -> dict:
    report = load_json(report_path)
    result = {
        "network": network_family(path),
        "variant": artifact_stem(path),
        "relay_artifact": str(path),
        "report": str(report_path),
        "wall_seconds": wall_seconds,
        "returncode": returncode,
        "status": "missing_report" if report is None else report.get("status", "unknown"),
    }
    if report is None:
        return result

    result["source_kind"] = report.get("model", {}).get("source")
    coverage = report.get("history_coverage", {})
    result["coverage"] = {
        "complete": coverage.get("complete"),
        "covered_tasks": coverage.get("covered_task_count"),
        "total_tasks": coverage.get("task_count"),
        "covered_weight": coverage.get("covered_task_weight"),
        "total_weight": coverage.get("task_weight"),
        "records_used": coverage.get("build_record_count"),
        "extraction_seconds": coverage.get("extraction_seconds"),
    }
    for mode in ("baseline", "optimized"):
        item = report.get(mode)
        if item:
            result[mode] = {
                "compile_seconds": item.get("compile_seconds"),
                "library_source": item.get("library_source"),
                "library_path": item.get("library_path"),
                "timing": item.get("timing"),
            }
    correctness = report.get("correctness")
    if correctness:
        result["correctness"] = correctness
    result["speedup"] = report.get("speedup")
    result["valid_comparison"] = bool(
        result["status"] == "ok"
        and result.get("speedup") is not None
        and coverage.get("complete") is True
        and (correctness is None or correctness.get("passed") is True)
    )
    return result


def numeric(values: Iterable[object]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(value)]


def summary_stats(values: Iterable[object], geometric: bool = False) -> dict | None:
    items = numeric(values)
    if not items:
        return None
    result = {
        "count": len(items),
        "mean": statistics.fmean(items),
        "median": statistics.median(items),
        "min": min(items),
        "max": max(items),
        "stdev": statistics.pstdev(items),
    }
    if geometric and all(value > 0 for value in items):
        result["geometric_mean"] = math.exp(statistics.fmean(math.log(value) for value in items))
    return result


def aggregate_network(family: str, variants: list[dict]) -> dict:
    observed = [
        item
        for item in variants
        if item.get("status") == "ok"
        and item.get("speedup") is not None
        and item.get("correctness", {}).get("passed", True) is True
    ]
    valid = [item for item in variants if item.get("valid_comparison")]
    baseline_ms = [
        item.get("baseline", {}).get("timing", {}).get("median_ms") for item in variants
    ]
    optimized_ms = [
        item.get("optimized", {}).get("timing", {}).get("median_ms") for item in variants
    ]
    observed_speedups = [item.get("speedup") for item in observed]
    valid_speedups = [item.get("speedup") for item in valid]
    return {
        "network": family,
        "variant_count": len(variants),
        "completed_count": sum(item.get("status") in {"ok", "output_mismatch"} for item in variants),
        "observed_comparison_count": len(observed),
        "valid_comparison_count": len(valid),
        "failed_count": sum(item.get("returncode", 1) != 0 for item in variants),
        "output_mismatch_count": sum(item.get("status") == "output_mismatch" for item in variants),
        "incomplete_coverage_count": sum(
            item.get("coverage", {}).get("complete") is False for item in variants
        ),
        "wall_seconds": sum(float(item.get("wall_seconds", 0.0)) for item in variants),
        "baseline_compile_seconds": sum(
            numeric(item.get("baseline", {}).get("compile_seconds") for item in variants)
        ),
        "optimized_compile_seconds": sum(
            numeric(item.get("optimized", {}).get("compile_seconds") for item in variants)
        ),
        "baseline_median_ms": summary_stats(baseline_ms),
        "optimized_median_ms": summary_stats(optimized_ms),
        "observed_speedup": summary_stats(observed_speedups, geometric=True),
        "valid_speedup": summary_stats(valid_speedups, geometric=True),
        "variants": variants,
    }


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def stat_value(summary: dict | None, key: str) -> float | None:
    return summary.get(key) if summary else None


def write_network_outputs(output_dir: Path, network_summaries: list[dict]) -> None:
    summaries_dir = output_dir / "network_summaries"
    for item in network_summaries:
        atomic_json(summaries_dir / f"{slug(item['network'])}.json", item)

    network_rows = []
    variant_rows = []
    for item in network_summaries:
        network_rows.append(
            {
                "network": item["network"],
                "variant_count": item["variant_count"],
                "observed_comparison_count": item["observed_comparison_count"],
                "valid_comparison_count": item["valid_comparison_count"],
                "failed_count": item["failed_count"],
                "output_mismatch_count": item["output_mismatch_count"],
                "incomplete_coverage_count": item["incomplete_coverage_count"],
                "baseline_median_of_variants_ms": stat_value(
                    item["baseline_median_ms"], "median"
                ),
                "optimized_median_of_variants_ms": stat_value(
                    item["optimized_median_ms"], "median"
                ),
                "observed_geomean_speedup": stat_value(
                    item["observed_speedup"], "geometric_mean"
                ),
                "valid_geomean_speedup": stat_value(
                    item["valid_speedup"], "geometric_mean"
                ),
                "valid_min_speedup": stat_value(item["valid_speedup"], "min"),
                "valid_max_speedup": stat_value(item["valid_speedup"], "max"),
                "baseline_compile_seconds": item["baseline_compile_seconds"],
                "optimized_compile_seconds": item["optimized_compile_seconds"],
                "wall_seconds": item["wall_seconds"],
            }
        )
        for variant in item["variants"]:
            coverage = variant.get("coverage", {})
            correctness = variant.get("correctness", {})
            variant_rows.append(
                {
                    "network": item["network"],
                    "variant": variant["variant"],
                    "status": variant["status"],
                    "valid_comparison": variant.get("valid_comparison", False),
                    "baseline_median_ms": variant.get("baseline", {})
                    .get("timing", {})
                    .get("median_ms"),
                    "optimized_median_ms": variant.get("optimized", {})
                    .get("timing", {})
                    .get("median_ms"),
                    "speedup": variant.get("speedup"),
                    "coverage_complete": coverage.get("complete"),
                    "covered_tasks": coverage.get("covered_tasks"),
                    "total_tasks": coverage.get("total_tasks"),
                    "covered_weight": coverage.get("covered_weight"),
                    "total_weight": coverage.get("total_weight"),
                    "correctness_passed": correctness.get("passed"),
                    "baseline_compile_seconds": variant.get("baseline", {}).get(
                        "compile_seconds"
                    ),
                    "optimized_compile_seconds": variant.get("optimized", {}).get(
                        "compile_seconds"
                    ),
                    "wall_seconds": variant.get("wall_seconds"),
                    "report": variant.get("report"),
                }
            )

    write_csv(
        output_dir / "network_summary.csv",
        [
            "network",
            "variant_count",
            "observed_comparison_count",
            "valid_comparison_count",
            "failed_count",
            "output_mismatch_count",
            "incomplete_coverage_count",
            "baseline_median_of_variants_ms",
            "optimized_median_of_variants_ms",
            "observed_geomean_speedup",
            "valid_geomean_speedup",
            "valid_min_speedup",
            "valid_max_speedup",
            "baseline_compile_seconds",
            "optimized_compile_seconds",
            "wall_seconds",
        ],
        network_rows,
    )
    write_csv(
        output_dir / "variant_results.csv",
        [
            "network",
            "variant",
            "status",
            "valid_comparison",
            "baseline_median_ms",
            "optimized_median_ms",
            "speedup",
            "coverage_complete",
            "covered_tasks",
            "total_tasks",
            "covered_weight",
            "total_weight",
            "correctness_passed",
            "baseline_compile_seconds",
            "optimized_compile_seconds",
            "wall_seconds",
            "report",
        ],
        variant_rows,
    )


class RunLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a", encoding="utf-8", buffering=1)

    def write(self, message: str = "") -> None:
        print(message, flush=True)
        self.file.write(message + "\n")

    def close(self) -> None:
        self.file.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    reports_dir = output_dir / "reports"
    libraries_dir = output_dir / "libraries"
    summary_path = Path(args.summary).expanduser().resolve() if args.summary else output_dir / "summary.json"
    log_path = Path(args.log_file).expanduser().resolve() if args.log_file else output_dir / "batch_benchmark.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    libraries_dir.mkdir(parents=True, exist_ok=True)

    artifacts = discover_artifacts(args)
    histories = discover_histories(args)
    history_digest = file_fingerprint(histories, [args.target, args.mode]) if histories else "no-history"
    logger = RunLogger(log_path)
    started = time.perf_counter()
    started_at = now_iso()
    results: list[dict] = []

    try:
        logger.write("=" * 79)
        logger.write(f"Batch Relay benchmark started: {started_at}")
        logger.write(f"Target: {args.target}")
        logger.write(f"Artifacts: {len(artifacts)}")
        logger.write(f"Network families: {len({network_family(path) for path in artifacts})}")
        logger.write(f"History files: {len(histories)} (fingerprint={history_digest})")
        logger.write(f"Output: {output_dir}")

        if args.dry_run:
            grouped = defaultdict(list)
            for path in artifacts:
                grouped[network_family(path)].append(str(path))
            payload = {
                "status": "dry_run",
                "started_at": started_at,
                "target": args.target,
                "history_files": [str(path) for path in histories],
                "networks": dict(sorted(grouped.items())),
            }
            atomic_json(summary_path, payload)
            for family, paths in sorted(grouped.items()):
                logger.write(f"{family}: {len(paths)} variants")
            return

        for index, artifact in enumerate(artifacts, start=1):
            variant = artifact_stem(artifact)
            variant_slug = slug(variant)
            model_digest = file_fingerprint([artifact], [args.target])
            report_path = reports_dir / f"{variant_slug}.{history_digest}.json"
            baseline_lib = libraries_dir / f"{variant_slug}.{model_digest}.default.so"
            optimized_lib = libraries_dir / f"{variant_slug}.{model_digest}.{history_digest}.autoscheduler.so"

            existing = load_json(report_path) if args.resume else None
            if existing and existing.get("status") in {"ok", "output_mismatch"}:
                logger.write(f"[{index}/{len(artifacts)}] resume {variant}")
                results.append(variant_result(artifact, report_path, 0.0, 0))
                continue

            command = [
                args.python,
                args.benchmark_script,
                f"--relay-pickle={artifact}",
                f"--target={args.target}",
                f"--device-id={args.device_id}",
                f"--mode={args.mode}",
                f"--report={report_path}",
                f"--warmup={args.warmup}",
                f"--number={args.number}",
                f"--repeat={args.repeat}",
                f"--min-repeat-ms={args.min_repeat_ms}",
                f"--seed={args.seed}",
                f"--rtol={args.rtol}",
                f"--atol={args.atol}",
            ]
            if args.cache_libraries and args.mode in {"both", "baseline"}:
                command.append(f"--baseline-lib={baseline_lib}")
            if args.cache_libraries and args.mode in {"both", "optimized"}:
                command.append(f"--optimized-lib={optimized_lib}")
            if args.mode in {"both", "optimized"}:
                command.extend(f"--history={path}" for path in histories)
            if args.allow_incomplete_history:
                command.append("--allow-incomplete-history")

            logger.write("-" * 79)
            logger.write(
                f"[{index}/{len(artifacts)}] network={network_family(artifact)} variant={variant}"
            )
            item_started = time.perf_counter()
            process = subprocess.Popen(
                command,
                cwd=str(Path(args.benchmark_script).resolve().parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for raw in process.stdout:
                logger.write(f"[{variant}] {raw.rstrip()}")
            returncode = process.wait()
            wall_seconds = time.perf_counter() - item_started
            item = variant_result(artifact, report_path, wall_seconds, returncode)
            results.append(item)
            logger.write(
                f"[{index}/{len(artifacts)}] status={item['status']} "
                f"returncode={returncode} wall_seconds={wall_seconds:.2f} "
                f"speedup={item.get('speedup')}"
            )
            if returncode and not args.continue_on_error:
                break

        grouped_results: dict[str, list[dict]] = defaultdict(list)
        for item in results:
            grouped_results[item["network"]].append(item)
        network_summaries = [
            aggregate_network(family, variants)
            for family, variants in sorted(grouped_results.items())
        ]
        total_wall = time.perf_counter() - started
        if len(results) != len(artifacts):
            batch_status = "partial"
        elif any(
            item.get("returncode") != 0 or item.get("status") != "ok" for item in results
        ):
            batch_status = "completed_with_errors"
        elif args.mode == "both" and any(
            not item.get("valid_comparison") for item in results
        ):
            batch_status = "completed_with_invalid_comparisons"
        else:
            batch_status = "ok"
        payload = {
            "status": batch_status,
            "started_at": started_at,
            "finished_at": now_iso(),
            "wall_seconds": total_wall,
            "target": args.target,
            "device_id": args.device_id,
            "mode": args.mode,
            "cache_libraries": args.cache_libraries,
            "benchmark_config": {
                "warmup": args.warmup,
                "number": args.number,
                "repeat": args.repeat,
                "min_repeat_ms": args.min_repeat_ms,
                "seed": args.seed,
            },
            "history_fingerprint": history_digest,
            "history_files": [str(path) for path in histories],
            "artifact_count": len(artifacts),
            "processed_count": len(results),
            "network_count": len(network_summaries),
            "network_summaries": network_summaries,
        }
        atomic_json(summary_path, payload)
        write_network_outputs(output_dir, network_summaries)

        logger.write("=" * 79)
        logger.write("Per-network summary")
        for item in network_summaries:
            observed_speedup = item.get("observed_speedup") or {}
            valid_speedup = item.get("valid_speedup") or {}
            baseline = item.get("baseline_median_ms") or {}
            optimized = item.get("optimized_median_ms") or {}
            logger.write(
                f"{item['network']}: variants={item['variant_count']} "
                f"observed={item['observed_comparison_count']} "
                f"valid={item['valid_comparison_count']} failed={item['failed_count']} "
                f"baseline_variant_median_ms={baseline.get('median')} "
                f"optimized_variant_median_ms={optimized.get('median')} "
                f"observed_geomean_speedup={observed_speedup.get('geometric_mean')} "
                f"valid_geomean_speedup={valid_speedup.get('geometric_mean')} "
                f"wall_seconds={item['wall_seconds']:.2f}"
            )
            for variant in item["variants"]:
                baseline_ms = (
                    variant.get("baseline", {}).get("timing", {}).get("median_ms")
                )
                optimized_ms = (
                    variant.get("optimized", {}).get("timing", {}).get("median_ms")
                )
                logger.write(
                    f"  {variant['variant']}: status={variant['status']} "
                    f"baseline_ms={baseline_ms} optimized_ms={optimized_ms} "
                    f"speedup={variant.get('speedup')} wall_seconds={variant['wall_seconds']:.2f}"
                )
        logger.write(f"Summary JSON: {summary_path}")
        logger.write(f"Per-network JSON: {output_dir / 'network_summaries'}")
        logger.write(f"Network CSV: {output_dir / 'network_summary.csv'}")
        logger.write(f"Variant CSV: {output_dir / 'variant_results.csv'}")
        logger.write(f"Batch log: {log_path}")
        logger.write(f"Total wall seconds: {total_wall:.2f}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
