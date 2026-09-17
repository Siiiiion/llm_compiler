#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate unique TVM AutoScheduler states, optionally gated by LocalBuilder.

The legacy pipeline saved every state returned by ``sample_initial_population``
with a placeholder ``MeasureResult([0.0], 0, ...)``.  A concrete state is not
necessarily compilable, so build validation is enabled by default here.  The
saved records still have zero latency, but ``error_no == 0`` now means the
state passed ``LocalBuilder`` rather than being an unchecked placeholder.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import tvm
from tqdm import tqdm
from tvm import auto_scheduler

from common import get_to_measure_filename, load_and_register_tasks, register_data_path


def batches(values: list, batch_size: int) -> Iterable[list]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def atomic_json(path: str, payload: object) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}-", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2, sort_keys=True)
            file_obj.write("\n")
        os.replace(temporary, output)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def save_unmeasured_records(filename: str, task, states: list) -> None:
    """Atomically save build-valid but not yet benchmarked states."""
    output = Path(filename).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        if temporary.exists():
            temporary.unlink()
        measure_inputs = [auto_scheduler.MeasureInput(task, state) for state in states]
        now = time.time()
        measure_results = [
            auto_scheduler.MeasureResult([0.0], 0, "build_valid_unmeasured", 0, now)
            for _ in states
        ]
        auto_scheduler.save_records(str(temporary), measure_inputs, measure_results)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def make_policy(task, requested_size: int, evolutionary_search_population: int | None = None):
    population = evolutionary_search_population or min(max(requested_size, 1), 2560)
    return auto_scheduler.SketchPolicy(
        task,
        params={
            "evolutionary_search_num_iters": 1,
            "evolutionary_search_population": population,
            "max_innermost_split_factor": 1024,
        },
        verbose=0,
    )


def dump_program(
    task,
    size: int,
    output_dir: str | None = None,
    *,
    validate_build: bool = True,
    build_timeout: int = 30,
    build_n_parallel: int = 16,
    build_batch_size: int = 64,
    max_rounds: int = 20,
    max_stalled_rounds: int = 5,
    evolutionary_search_population: int | None = None,
    overwrite: bool = False,
    require_full_size: bool = False,
) -> dict:
    """Generate unique states and keep only states accepted by LocalBuilder."""
    filename = get_to_measure_filename(task, output_dir)
    stats = {
        "workload_key": str(task.workload_key),
        "target": str(task.target),
        "output": os.path.abspath(filename),
        "requested_states": size,
        "evolutionary_search_population": evolutionary_search_population
        or min(max(size, 1), 2560),
        "validate_build": validate_build,
        "status": None,
        "rounds": 0,
        "sampled_states": 0,
        "unique_candidates": 0,
        "duplicate_candidates": 0,
        "build_attempted": 0,
        "build_valid": 0,
        "accepted_without_build": 0,
        "build_error_counts": {},
        "stalled_rounds": 0,
        "elapsed_seconds": 0.0,
    }
    started = time.perf_counter()
    if os.path.exists(filename) and not overwrite:
        stats["status"] = "skipped_existing"
        stats["elapsed_seconds"] = time.perf_counter() - started
        return stats

    policy = make_policy(task, size, evolutionary_search_population)
    builder = (
        auto_scheduler.measure.LocalBuilder(
            timeout=build_timeout,
            n_parallel=build_n_parallel,
        )
        if validate_build
        else None
    )
    seen_states: set[str] = set()
    accepted_states = []
    build_error_counts: Counter[int] = Counter()
    stalled_rounds = 0

    while (
        len(accepted_states) < size
        and stats["rounds"] < max_rounds
        and stalled_rounds < max_stalled_rounds
    ):
        stats["rounds"] += 1
        sampled = policy.sample_initial_population()
        stats["sampled_states"] += len(sampled)
        new_states = []
        for state in sampled:
            state_key = str(state)
            if state_key in seen_states:
                stats["duplicate_candidates"] += 1
                continue
            seen_states.add(state_key)
            new_states.append(state)
        stats["unique_candidates"] = len(seen_states)

        accepted_before = len(accepted_states)
        if validate_build:
            for state_batch in batches(new_states, build_batch_size):
                measure_inputs = [
                    auto_scheduler.MeasureInput(task, state) for state in state_batch
                ]
                build_results = builder.build(measure_inputs) if measure_inputs else []
                stats["build_attempted"] += len(measure_inputs)
                for state, result in zip(state_batch, build_results):
                    error_no = int(result.error_no)
                    if error_no == 0:
                        accepted_states.append(state)
                    else:
                        build_error_counts[error_no] += 1
                    if len(accepted_states) >= size:
                        break
                if len(accepted_states) >= size:
                    break
        else:
            accepted_states.extend(new_states[: max(size - len(accepted_states), 0)])

        if len(accepted_states) == accepted_before:
            stalled_rounds += 1
        else:
            stalled_rounds = 0
        stats["stalled_rounds"] = stalled_rounds
        print(
            "[dump] workload={workload} round={round_no}/{max_rounds} "
            "new={new} accepted={accepted}/{requested} build_attempted={attempted} "
            "stalled={stalled}/{max_stalled}".format(
                workload=task.workload_key,
                round_no=stats["rounds"],
                max_rounds=max_rounds,
                new=len(new_states),
                accepted=len(accepted_states),
                requested=size,
                attempted=stats["build_attempted"],
                stalled=stalled_rounds,
                max_stalled=max_stalled_rounds,
            ),
            flush=True,
        )

    accepted_states = accepted_states[:size]
    if validate_build:
        stats["build_valid"] = len(accepted_states)
    else:
        stats["accepted_without_build"] = len(accepted_states)
    stats["build_error_counts"] = {
        str(error_no): count for error_no, count in sorted(build_error_counts.items())
    }
    stats["elapsed_seconds"] = time.perf_counter() - started

    if require_full_size and len(accepted_states) < size:
        stats["status"] = "shortfall_not_written"
        return stats
    if not accepted_states:
        stats["status"] = "no_valid_states"
        return stats

    save_unmeasured_records(filename, task, accepted_states)
    stats["status"] = "ok" if len(accepted_states) == size else "partial"
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--target", required=True)
    parser.add_argument("--hardware-name")
    parser.add_argument("--network-info-dir")
    parser.add_argument(
        "--output-dir",
        default="/home/qsy/workspace/dataset/to_measure_programs/4090",
    )
    parser.add_argument(
        "--validate-build",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter sampled states through LocalBuilder (default: enabled)",
    )
    parser.add_argument("--build-timeout", type=int, default=30)
    parser.add_argument("--build-n-parallel", type=int, default=16)
    parser.add_argument("--build-batch-size", type=int, default=64)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--max-stalled-rounds", type=int, default=5)
    parser.add_argument(
        "--evolutionary-search-population",
        type=int,
        help="Population sampled per initial-population round; defaults to --size capped at 2560",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--require-full-size",
        action="store_true",
        help="Do not write a task file unless exactly --size valid states are found",
    )
    parser.add_argument(
        "--stats-file",
        help="Defaults to <output-dir>/dump_programs_stats_<start>_<end>.json",
    )
    args = parser.parse_args()
    for option in (
        "size",
        "build_timeout",
        "build_n_parallel",
        "build_batch_size",
        "max_rounds",
        "max_stalled_rounds",
    ):
        if getattr(args, option) <= 0:
            parser.error(f"--{option.replace('_', '-')} must be positive")
    if (
        args.evolutionary_search_population is not None
        and args.evolutionary_search_population <= 0
    ):
        parser.error("--evolutionary-search-population must be positive")
    if args.start_idx < 0:
        parser.error("--start-idx must be non-negative")
    if args.end_idx is not None and args.end_idx < args.start_idx:
        parser.error("--end-idx must be >= --start-idx")
    return args


def main() -> None:
    args = parse_args()
    register_data_path(
        args.target,
        hardware_name=args.hardware_name,
        network_info_folder=args.network_info_dir,
        to_measure_program_folder=args.output_dir,
    )
    target = tvm.target.Target(args.target)
    tasks = load_and_register_tasks()
    end_idx = min(args.end_idx if args.end_idx is not None else len(tasks), len(tasks))
    selected_tasks = tasks[args.start_idx:end_idx]
    task_stats = []
    for task in tqdm(selected_tasks):
        task_stats.append(
            dump_program(
                task,
                size=args.size,
                output_dir=args.output_dir,
                validate_build=args.validate_build,
                build_timeout=args.build_timeout,
                build_n_parallel=args.build_n_parallel,
                build_batch_size=args.build_batch_size,
                max_rounds=args.max_rounds,
                max_stalled_rounds=args.max_stalled_rounds,
                evolutionary_search_population=args.evolutionary_search_population,
                overwrite=args.overwrite,
                require_full_size=args.require_full_size,
            )
        )
        gc.collect()

    stats_file = args.stats_file or os.path.join(
        args.output_dir,
        f"dump_programs_stats_{args.start_idx}_{end_idx}.json",
    )
    payload = {
        "target": str(target),
        "network_info_dir": args.network_info_dir,
        "output_dir": os.path.abspath(args.output_dir),
        "start_idx": args.start_idx,
        "end_idx": end_idx,
        "requested_states_per_task": args.size,
        "validate_build": args.validate_build,
        "build_timeout": args.build_timeout,
        "build_n_parallel": args.build_n_parallel,
        "build_batch_size": args.build_batch_size,
        "max_rounds": args.max_rounds,
        "max_stalled_rounds": args.max_stalled_rounds,
        "evolutionary_search_population": args.evolutionary_search_population
        or min(max(args.size, 1), 2560),
        "require_full_size": args.require_full_size,
        "tasks": task_stats,
        "summary": {
            "tasks": len(task_stats),
            "ok": sum(item["status"] == "ok" for item in task_stats),
            "partial": sum(item["status"] == "partial" for item in task_stats),
            "no_valid_states": sum(
                item["status"] == "no_valid_states" for item in task_stats
            ),
            "shortfall_not_written": sum(
                item["status"] == "shortfall_not_written" for item in task_stats
            ),
            "skipped_existing": sum(
                item["status"] == "skipped_existing" for item in task_stats
            ),
            "build_attempted": sum(item["build_attempted"] for item in task_stats),
            "build_valid": sum(item["build_valid"] for item in task_stats),
            "accepted_without_build": sum(
                item["accepted_without_build"] for item in task_stats
            ),
        },
    }
    atomic_json(stats_file, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"Stats: {os.path.abspath(stats_file)}")
    if args.require_full_size and payload["summary"]["shortfall_not_written"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
