from __future__ import annotations

from dataclasses import dataclass, field
import glob
import json
import math
import os
import pickle
from typing import Dict, List, Optional, Tuple

from transformers import HfArgumentParser
import tvm

from common import clean_name, hold_out_task_files, load_tasks_path, register_data_path


MODEL_PATH = "gen_data/task_scheduler"


@dataclass
class ScriptArguments:
    target: str = field(metadata={"help": "TVM target string"})
    network_info_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing all_tasks.pkl and *.task.pkl"},
    )
    to_measure_program_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing candidate measure-record JSON files"},
    )
    measure_record_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Override the target-derived measure_records directory"},
    )
    history_files: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated measured JSON files or glob patterns, ordered by round"
        },
    )
    history_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Add all *.json files in this directory as ordered history rounds"},
    )
    output_path: Optional[str] = field(
        default=None,
        metadata={"help": "Scheduler pickle output path"},
    )
    top_k: Optional[int] = field(
        default=None,
        metadata={"help": "Select this many tasks before applying the budget cap"},
    )
    top_ratio: float = field(
        default=0.25,
        metadata={"help": "Task fraction selected when top_k is not set"},
    )
    back_window_size: int = field(
        default=3,
        metadata={"help": "Recent measured rounds used for improvement scoring"},
    )
    measurement_budget: int = field(
        default=20000,
        metadata={"help": "Historical plus planned measurement cap; <=0 disables it"},
    )
    records_per_task: int = field(
        default=64,
        metadata={"help": "Planned measurements charged for each selected task"},
    )
    hold_out_only: bool = field(
        default=False,
        metadata={"help": "Use only the legacy built-in hold-out task files"},
    )


def _valid_latency(record: dict) -> Optional[float]:
    result = record.get("r")
    if not result or len(result) < 2 or result[1] != 0 or not result[0]:
        return None
    valid = []
    for cost in result[0]:
        try:
            cost = float(cost)
        except (TypeError, ValueError):
            continue
        if 0 < cost < 1e9:
            valid.append(cost)
    return sum(valid) / len(valid) if valid else None


def read_fine_tuning_json(json_path: str) -> Dict[str, Tuple[float, int]]:
    """Return the best valid latency and valid record count per workload."""
    best_and_count: Dict[str, Tuple[float, int]] = {}
    with open(json_path, "r") as file_obj:
        for raw in file_obj:
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
                workload_key = record["i"][0][0]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            latency = _valid_latency(record)
            if latency is None:
                continue
            old_best, old_count = best_and_count.get(workload_key, (float("inf"), 0))
            best_and_count[workload_key] = (min(old_best, latency), old_count + 1)
    return best_and_count


def _count_measurement_attempts(json_path: str) -> int:
    attempts = 0
    with open(json_path, "r") as file_obj:
        for raw in file_obj:
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
                _ = record["i"][0][0]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            attempts += 1
    return attempts


def _expand_history_files(history_files: Optional[str], history_dir: Optional[str]) -> List[str]:
    files: List[str] = []
    if history_files:
        for value in history_files.split(","):
            value = value.strip()
            if not value:
                continue
            matches = sorted(glob.glob(value))
            if not matches:
                raise FileNotFoundError(f"History file or pattern does not exist: {value}")
            files.extend(matches)
    if history_dir:
        if not os.path.isdir(history_dir):
            raise NotADirectoryError(f"History directory does not exist: {history_dir}")
        files.extend(sorted(glob.glob(os.path.join(history_dir, "*.json"))))

    unique_files = []
    seen = set()
    for path in files:
        path = os.path.abspath(path)
        if path not in seen:
            unique_files.append(path)
            seen.add(path)
    return unique_files


def _load_task_files(target: tvm.target.Target, hold_out_only: bool) -> List[str]:
    if hold_out_only:
        files = sorted(hold_out_task_files(target).values())
    else:
        files = sorted(load_tasks_path(target))
    files = [path for path in files if os.path.isfile(path)]
    if not files:
        raise FileNotFoundError(
            "No task pickle files found. Check --network_info_dir and the target kind."
        )
    return files


def _score_history(history: List[Optional[float]], weight: int, window_size: int):
    measured = [value for value in history if value is not None]
    if not measured:
        return float("inf"), "unmeasured"

    cumulative_best = []
    best = float("inf")
    for value in measured:
        best = min(best, value)
        cumulative_best.append(best)
    recent = cumulative_best[-max(window_size, 2) :]
    if len(recent) < 2 or recent[0] <= 0:
        return 0.0, "single_round"
    relative_improvement = max(0.0, (recent[0] - recent[-1]) / recent[0])
    return relative_improvement * weight, "history_improvement"


def _default_output_path() -> str:
    from common import HARDWARE_PLATFORM

    return os.path.join(MODEL_PATH, f"task_scheduler_{HARDWARE_PLATFORM}.pkl")


def _load_schedule_payload(schedule_file_path: Optional[str]):
    path = schedule_file_path or _default_output_path()
    with open(path, "rb") as file_obj:
        payload = pickle.load(file_obj)
    if isinstance(payload, set):
        return path, payload
    if not isinstance(payload, dict) or "selected_file_stems" not in payload:
        raise ValueError(f"Unsupported scheduler payload: {path}")
    return path, set(payload["selected_file_stems"])


def find_potential_files(files, schedule_file_path=None):
    """Filter candidate record files using a scheduler output pickle."""
    _, selected_stems = _load_schedule_payload(schedule_file_path)
    return [
        path
        for path in files
        if os.path.splitext(os.path.basename(path))[0] in selected_stems
    ]


def find_potential_files_len(schedule_file_path=None):
    _, selected_stems = _load_schedule_payload(schedule_file_path)
    return len(selected_stems)


def main():
    parser = HfArgumentParser(ScriptArguments)
    args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(args)

    register_data_path(
        args.target,
        network_info_folder=args.network_info_dir,
        to_measure_program_folder=args.to_measure_program_dir,
        measure_record_folder=args.measure_record_dir,
    )
    target = tvm.target.Target(args.target)
    task_files = _load_task_files(target, args.hold_out_only)
    history_files = _expand_history_files(args.history_files, args.history_dir)
    if not history_files:
        raise ValueError("At least one --history_files entry or --history_dir is required")

    history_by_workload: Dict[str, List[Optional[float]]] = {}
    measured_counts: Dict[str, int] = {}
    total_measurement_attempts = 0
    for round_idx, path in enumerate(history_files):
        round_stats = read_fine_tuning_json(path)
        total_measurement_attempts += _count_measurement_attempts(path)
        for workload_key in set(history_by_workload) | set(round_stats):
            history_by_workload.setdefault(workload_key, [None] * len(history_files))
        for workload_key, (latency, count) in round_stats.items():
            history_by_workload[workload_key][round_idx] = latency
            measured_counts[workload_key] = measured_counts.get(workload_key, 0) + count

    ranking_by_workload = {}
    for task_file in task_files:
        with open(task_file, "rb") as file_obj:
            tasks, task_weights = pickle.load(file_obj)
        for task, weight in zip(tasks, task_weights):
            workload_key = task.workload_key
            history = history_by_workload.get(workload_key, [None] * len(history_files))
            score, reason = _score_history(history, int(weight), args.back_window_size)
            item = {
                "workload_key": workload_key,
                "file_stem": clean_name((workload_key, target.kind)),
                "weight": int(weight),
                "score": score,
                "reason": reason,
                "history": history,
                "measured_records": measured_counts.get(workload_key, 0),
            }
            current = ranking_by_workload.get(workload_key)
            if current is None or item["weight"] > current["weight"]:
                ranking_by_workload[workload_key] = item

    ranking = list(ranking_by_workload.values())
    ranking.sort(
        key=lambda item: (
            item["reason"] == "unmeasured",
            item["score"],
            item["weight"],
            item["workload_key"],
        ),
        reverse=True,
    )

    if args.top_k is not None:
        requested_count = max(args.top_k, 0)
    else:
        if not 0 < args.top_ratio <= 1:
            raise ValueError("--top_ratio must be in the interval (0, 1]")
        requested_count = math.ceil(len(ranking) * args.top_ratio)

    total_valid_records = sum(measured_counts.values())
    selected_count = min(requested_count, len(ranking))
    if args.measurement_budget > 0:
        if args.records_per_task <= 0:
            raise ValueError("--records_per_task must be positive when budget capping is enabled")
        remaining_budget = max(args.measurement_budget - total_measurement_attempts, 0)
        selected_count = min(selected_count, remaining_budget // args.records_per_task)

    selected = ranking[:selected_count]
    output_path = os.path.abspath(args.output_path or _default_output_path())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "version": 2,
        "target": str(target),
        "network_info_dir": args.network_info_dir,
        "to_measure_program_dir": args.to_measure_program_dir,
        "history_files": history_files,
        "measurement_budget": args.measurement_budget,
        "records_per_task": args.records_per_task,
        "total_measurement_attempts": total_measurement_attempts,
        "total_valid_records": total_valid_records,
        "selected_workload_keys": {item["workload_key"] for item in selected},
        "selected_file_stems": {item["file_stem"] for item in selected},
        "ranking": ranking,
    }
    with open(output_path, "wb") as file_obj:
        pickle.dump(payload, file_obj)

    print(f"Task files: {len(task_files)}")
    print(f"History rounds: {len(history_files)}")
    print(f"Ranked workloads: {len(ranking)}")
    print(f"Historical measurement attempts: {total_measurement_attempts}")
    print(f"Valid historical records: {total_valid_records}")
    print(f"Selected workloads: {len(selected)}")
    for index, item in enumerate(selected, start=1):
        score = "inf" if math.isinf(item["score"]) else f"{item['score']:.6f}"
        print(
            f"  {index}. weight={item['weight']} score={score} "
            f"reason={item['reason']} key={item['workload_key']}"
        )
    print(f"Scheduler output: {output_path}")


if __name__ == "__main__":
    main()
