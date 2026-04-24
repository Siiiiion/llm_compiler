#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构造 DPO / LAPO 训练使用的偏好对数据集（创新点 2）。

数据来源：与 ``make_dataset.py`` 相同的 TVM 测量记录（``*.json`` lines）。
每个 workload 内：
    - 按 ``steps[:ppt_idx+1]``（含 PPT 步）划分 prompt group；
    - 同一个 group 内按 latency 排序；
    - chosen = latency 最低的 suffix；
    - rejected 从同组内 latency 更慢的样本中挑出（受 ``min_latency_gap_ratio`` 过滤）；
    - 记录 ``latency_gap``，供 LAPO（Latency-Weighted DPO）加权使用。

输出：
    - ``save_path/train_pairs.jsonl``
    - ``save_path/validation_pairs.jsonl``（可选）
    - ``save_path/build_stats.json``
"""

from __future__ import annotations

import copy
import glob
import json
import logging
import math
import os
import random
import shutil
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

from transformers import HfArgumentParser, set_seed

import tvm
from tvm import auto_scheduler
from tvm.auto_scheduler.measure_record import load_record_from_string

from common import (
    get_hold_out_five_files,
    load_and_register_tasks,
    register_data_path,
)
from make_dataset_utils import json_dfs_without_bracket


WEIGHT_STRATEGIES = ("uniform", "log_gap", "linear_gap", "clipped_linear")


@dataclass
class PreferenceArgs:
    """偏好对构造参数。"""

    target: str = field(metadata={"help": "目标硬件平台字符串，如 'cuda -model=4090'"})
    dataset_path: str = field(metadata={"help": "TVM 测量记录所在目录（包含 *.json）"})
    save_path: str = field(metadata={"help": "偏好对输出目录"})

    min_latency_gap_ratio: float = field(
        default=1.5,
        metadata={"help": "rejected/chosen latency 比率下界；低于此值则跳过该对"},
    )
    max_pairs_per_group: int = field(
        default=4,
        metadata={"help": "同一个 prompt group 最多抽几个 rejected 组成偏好对"},
    )
    max_pairs_per_workload: Optional[int] = field(
        default=512,
        metadata={"help": "同一个 measure_record 文件最多保留多少偏好对；None 表示不设上限"},
    )
    min_suffix_tokens: int = field(
        default=8,
        metadata={"help": "chosen/rejected 后缀按空格切分后的最小 token 数"},
    )
    max_prompt_tokens: int = field(
        default=512,
        metadata={"help": "prompt 文本按空格切分后的最大 token 数；超限丢弃"},
    )
    max_suffix_tokens: int = field(
        default=512,
        metadata={"help": "chosen/rejected 后缀按空格切分后的最大 token 数；超限截断"},
    )
    valid_percentage: int = field(
        default=5,
        metadata={"help": "按 workload 文件划分的验证集比例"},
    )
    split_seed: int = field(default=0, metadata={"help": "文件级划分随机种子"})
    sample_seed: int = field(default=42, metadata={"help": "rejected 抽样随机种子"})
    file_cnt: Optional[int] = field(
        default=None, metadata={"help": "仅采样 N 个 workload 文件（调试用）"},
    )
    weight_strategy: str = field(
        default="log_gap",
        metadata={
            "help": f"latency_weight 计算策略，支持 {WEIGHT_STRATEGIES}",
        },
    )
    weight_clip: float = field(
        default=5.0,
        metadata={"help": "clipped_linear 策略的权重上限"},
    )
    num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "并行处理文件的进程数；None 使用 os.cpu_count()"},
    )
    drop_hold_out_workloads: bool = field(
        default=True,
        metadata={"help": "从数据集中排除 hold-out workload（和 FOR_GEN 一致）"},
    )
    ppt_marker: str = field(
        default="PPT",
        metadata={"help": "用于定位 decision suffix 起点的 marker 字符串（仅作元信息记录）"},
    )


def _build_logger(save_path: str) -> logging.Logger:
    os.makedirs(os.path.join(save_path, "logs"), exist_ok=True)
    log_path = os.path.join(save_path, "logs", "build_preference_pairs.log")

    logger = logging.getLogger("build_preference_pairs")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _dfs_flatten(data) -> str:
    text_list: List[str] = []
    json_dfs_without_bracket(data, text_list)
    return " ".join(text_list)


def _compute_weight(strategy: str, lat_chosen: float, lat_rejected: float, clip: float) -> float:
    if lat_chosen <= 0 or lat_rejected <= 0 or not math.isfinite(lat_chosen) or not math.isfinite(lat_rejected):
        return 1.0
    ratio = lat_rejected / lat_chosen
    if strategy == "uniform":
        return 1.0
    if strategy == "log_gap":
        return max(1.0, math.log(max(ratio, 1.0 + 1e-6)) + 1.0)
    if strategy == "linear_gap":
        return max(1.0, ratio)
    if strategy == "clipped_linear":
        return max(1.0, min(clip, ratio))
    raise ValueError(f"Unknown weight_strategy={strategy}")


def _normalize_record(json_line: dict) -> Tuple[dict, Optional[int]]:
    workload_key = json_line["i"][0][0]
    if isinstance(workload_key, str):
        json_line["i"][0][0] = json.loads(workload_key)

    steps = json_line["i"][1][1]
    ppt_idx: Optional[int] = None
    for step_idx, step in enumerate(steps):
        if step[0] == "SP":
            sp_list = step[4]
            for i in range(len(sp_list)):
                sp_list[i] = 1
        elif step[0] == "PPT" and ppt_idx is None:
            ppt_idx = step_idx
    return json_line, ppt_idx


def _get_latency(json_line: dict) -> Optional[float]:
    latencies = json_line["r"][0]
    if not latencies:
        return None
    valid = [l for l in latencies if l and l > 0 and l < 1e9]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _process_file(args) -> Tuple[List[dict], Dict[str, int]]:
    (
        file,
        min_gap_ratio,
        max_pairs_per_group,
        max_pairs_per_workload,
        min_suffix_tokens,
        max_prompt_tokens,
        max_suffix_tokens,
        sample_seed,
        weight_strategy,
        weight_clip,
    ) = args

    stats = {
        "raw_records": 0,
        "drop_no_ppt": 0,
        "drop_no_latency": 0,
        "drop_short_suffix": 0,
        "drop_long_prompt": 0,
        "drop_long_suffix": 0,
        "groups": 0,
        "groups_with_pair": 0,
        "emitted_pairs": 0,
        "drop_small_gap": 0,
    }

    try:
        with open(file, "r") as f:
            lines = [line.strip() for line in f.read().splitlines() if line.strip()]
    except Exception:
        return [], stats

    if not lines:
        return [], stats

    try:
        input_obj, _ = load_record_from_string(lines[0])
        task = auto_scheduler.measure.recover_measure_input(input_obj).task
        compute_dag = task.compute_dag.print_min()
        workload_key = task.workload_key
    except Exception:
        return [], stats

    groups: Dict[str, List[dict]] = {}
    for line in lines:
        stats["raw_records"] += 1
        try:
            json_line = json.loads(line)
        except Exception:
            continue

        json_line, ppt_idx = _normalize_record(json_line)
        if ppt_idx is None:
            stats["drop_no_ppt"] += 1
            continue

        latency = _get_latency(json_line)
        if latency is None:
            stats["drop_no_latency"] += 1
            continue

        steps = json_line["i"][1][1]

        prefix_steps = steps[: ppt_idx + 1]
        suffix_steps = steps[ppt_idx + 1 :]

        suffix_text = _dfs_flatten(suffix_steps)
        suffix_tokens = suffix_text.split()
        if len(suffix_tokens) < min_suffix_tokens:
            stats["drop_short_suffix"] += 1
            continue
        if len(suffix_tokens) > max_suffix_tokens:
            stats["drop_long_suffix"] += 1
            continue

        prompt_input = copy.deepcopy(json_line["i"])
        prompt_input[1][1] = prefix_steps
        prompt_text = _dfs_flatten([compute_dag, prompt_input])
        if len(prompt_text.split()) > max_prompt_tokens:
            stats["drop_long_prompt"] += 1
            continue

        group_key = json.dumps(prefix_steps, sort_keys=True)
        groups.setdefault(group_key, []).append(
            {
                "latency": latency,
                "prompt_text": prompt_text,
                "suffix_text": suffix_text,
                "suffix_len": len(suffix_tokens),
            }
        )

    rng = random.Random(sample_seed)
    results: List[dict] = []
    for group_key, group in groups.items():
        stats["groups"] += 1
        if len(group) < 2:
            continue
        group_sorted = sorted(group, key=lambda x: x["latency"])
        chosen = group_sorted[0]
        candidates = []
        for cand in group_sorted[1:]:
            ratio = cand["latency"] / max(chosen["latency"], 1e-12)
            if ratio >= min_gap_ratio:
                candidates.append(cand)
            else:
                stats["drop_small_gap"] += 1
        if not candidates:
            continue

        stats["groups_with_pair"] += 1
        rng.shuffle(candidates)
        for rejected in candidates[:max_pairs_per_group]:
            gap = rejected["latency"] / max(chosen["latency"], 1e-12)
            weight = _compute_weight(weight_strategy, chosen["latency"], rejected["latency"], weight_clip)
            results.append(
                {
                    "source_file": os.path.basename(file),
                    "workload_key": workload_key,
                    "prompt": chosen["prompt_text"],
                    "chosen": chosen["suffix_text"],
                    "rejected": rejected["suffix_text"],
                    "latency_chosen": float(chosen["latency"]),
                    "latency_rejected": float(rejected["latency"]),
                    "latency_gap": float(gap),
                    "latency_weight": float(weight),
                    "chosen_suffix_tokens": chosen["suffix_len"],
                    "rejected_suffix_tokens": rejected["suffix_len"],
                }
            )
            stats["emitted_pairs"] += 1

    if max_pairs_per_workload is not None and len(results) > max_pairs_per_workload:
        rng.shuffle(results)
        results = results[:max_pairs_per_workload]

    return results, stats


def _split_files_for_validation(files: List[str], valid_percentage: int, seed: int) -> Tuple[List[str], List[str]]:
    files = list(files)
    if not files:
        return [], []
    if valid_percentage <= 0 or len(files) == 1:
        return sorted(files), []

    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    n_valid = max(1, round(len(shuffled) * valid_percentage / 100.0))
    n_valid = min(n_valid, len(shuffled) - 1)
    validation_files = sorted(shuffled[:n_valid])
    train_files = sorted(shuffled[n_valid:])
    return train_files, validation_files


def _merge_stats(stats_list: List[Dict[str, int]]) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for s in stats_list:
        for k, v in s.items():
            merged[k] = merged.get(k, 0) + int(v)
    return merged


def _run_group(
    files: List[str],
    args: PreferenceArgs,
    output_jsonl: str,
    logger: logging.Logger,
    group_name: str,
) -> Dict[str, int]:
    if not files:
        with open(output_jsonl, "w") as f:
            pass
        return {"pairs_written": 0}

    worker_args = [
        (
            f,
            args.min_latency_gap_ratio,
            args.max_pairs_per_group,
            args.max_pairs_per_workload,
            args.min_suffix_tokens,
            args.max_prompt_tokens,
            args.max_suffix_tokens,
            args.sample_seed + idx,
            args.weight_strategy,
            args.weight_clip,
        )
        for idx, f in enumerate(files)
    ]

    num_workers = args.num_workers if args.num_workers else max(1, os.cpu_count() or 1)
    num_workers = min(num_workers, len(worker_args))

    start = time.time()
    logger.info("%s: start processing %d files with %d workers", group_name, len(files), num_workers)

    if num_workers == 1:
        per_file_results = [_process_file(a) for a in worker_args]
    else:
        with Pool(num_workers) as pool:
            per_file_results = pool.map(_process_file, worker_args)

    stats_list = []
    total_pairs = 0
    with open(output_jsonl, "w") as f:
        for pairs, stats in per_file_results:
            stats_list.append(stats)
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False))
                f.write("\n")
                total_pairs += 1

    merged = _merge_stats(stats_list)
    merged["pairs_written"] = total_pairs
    merged["files_processed"] = len(files)
    merged["elapsed_seconds"] = round(time.time() - start, 2)
    logger.info("%s: wrote %d pairs to %s (stats=%s)", group_name, total_pairs, output_jsonl, merged)
    return merged


def main():
    parser = HfArgumentParser(PreferenceArgs)
    args: PreferenceArgs = parser.parse_args_into_dataclasses()[0]

    if args.weight_strategy not in WEIGHT_STRATEGIES:
        raise ValueError(f"Unknown weight_strategy={args.weight_strategy}")

    os.makedirs(args.save_path, exist_ok=True)
    logger = _build_logger(args.save_path)
    logger.info("Preference args: %s", args)

    set_seed(args.sample_seed)

    register_data_path(args.target)
    tvm_target = tvm.target.Target(args.target)
    logger.info("Target=%s", tvm_target)
    load_and_register_tasks()

    files = sorted(glob.glob(os.path.join(args.dataset_path, "*.json")))
    logger.info("Discovered %d measurement files", len(files))

    if args.drop_hold_out_workloads:
        hold_out = {os.path.basename(f) for f in get_hold_out_five_files(args.target)}
        files = [f for f in files if os.path.basename(f) not in hold_out]
        logger.info("After dropping hold-out workloads: %d files", len(files))

    if args.file_cnt and args.file_cnt < len(files):
        random.Random(args.split_seed).shuffle(files)
        files = files[: args.file_cnt]
        files.sort()
        logger.info("Subsampled to %d files (file_cnt=%d)", len(files), args.file_cnt)

    train_files, validation_files = _split_files_for_validation(
        files, args.valid_percentage, args.split_seed
    )
    logger.info("Split: train=%d validation=%d", len(train_files), len(validation_files))

    train_jsonl = os.path.join(args.save_path, "train_pairs.jsonl")
    train_stats = _run_group(train_files, args, train_jsonl, logger, "train")

    validation_stats: Dict[str, int] = {}
    if validation_files:
        validation_jsonl = os.path.join(args.save_path, "validation_pairs.jsonl")
        validation_stats = _run_group(
            validation_files, args, validation_jsonl, logger, "validation"
        )

    manifest = {
        "target": str(tvm_target),
        "dataset_path": args.dataset_path,
        "save_path": args.save_path,
        "ppt_marker": args.ppt_marker,
        "min_latency_gap_ratio": args.min_latency_gap_ratio,
        "max_pairs_per_group": args.max_pairs_per_group,
        "max_pairs_per_workload": args.max_pairs_per_workload,
        "min_suffix_tokens": args.min_suffix_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_suffix_tokens": args.max_suffix_tokens,
        "valid_percentage": args.valid_percentage,
        "weight_strategy": args.weight_strategy,
        "weight_clip": args.weight_clip,
        "train_file_count": len(train_files),
        "validation_file_count": len(validation_files),
        "train_stats": train_stats,
        "validation_stats": validation_stats,
    }
    manifest_path = os.path.join(args.save_path, "build_stats.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()
