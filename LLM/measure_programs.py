#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TVM 调度真实 latency 测量工具（v2）。

主要用途：
    1. 为创新点 2（DPO / LAPO）补齐 ``measure_records/4090/*.json`` 中 ``r==[[0]]`` 的行。
    2. 兼容旧的单文件 ``--to-measure-path / --measured-path`` 调用方式。

v2 改动（相对旧脚本）：
    - 移除硬编码 ``CUDA_VISIBLE_DEVICES="3"``；改用环境变量或 ``--cuda-visible`` 选项。
    - 修复 ``end_idx = min(args.end_idx, len(tasks))`` 的无效引用；移除
      ``--start-idx / --end-idx / --step-idx`` 三个事实上没生效的参数。
    - 新增 ``--input-dir`` 目录批量模式，对每个文件**就地**补测：
        * 已有有效 latency 的行保留不动；
        * 只把 ``r==[[0]]`` / 失败的行挑出来重测并追加回去；
        * 配合 ``--resume`` 可以安全地断点续测。
    - 新增 ``--drop-hold-out``（默认开），跳过 hold-out workload，避免污染评测集。
    - 新增 ``--max-records-per-file``，方便小规模试跑。

使用示例（创新点 2 偏好对前置测量）：
    CUDA_VISIBLE_DEVICES=0 python3 measure_programs.py \\
        --target "cuda -model=4090" \\
        --input-dir /data3/qsy/dataset/measure_records/4090 \\
        --max-records-per-file 128 \\
        --resume

旧模式（单文件）仍然可用：
    python3 measure_programs.py --target "cuda -model=4090" \\
        --to-measure-path 1122.json --measured-path 1122.measured.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time


# ---------------------------------------------------------------------------
# 在 import tvm 之前处理 --cuda-visible，保证进程级别生效
# ---------------------------------------------------------------------------
def _pre_parse_cuda_visible():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cuda-visible", default=None)
    known, _ = parser.parse_known_args()
    if known.cuda_visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = known.cuda_visible


_pre_parse_cuda_visible()


import tvm  # noqa: E402
from tvm import auto_scheduler  # noqa: E402
from tvm.auto_scheduler.measure_record import load_record_from_string  # noqa: E402
from tqdm import tqdm  # noqa: E402

from common import (  # noqa: E402
    register_data_path,
    load_and_register_tasks,
    get_hold_out_five_files,
)


# ---------------------------------------------------------------------------
# measurer 工厂
# ---------------------------------------------------------------------------


def make_measurer(
    run_timeout,
    repeat,
    number,
    enable_cpu_cache_flush,
    verbose,
    log_filename,
    min_repeat_ms,
):
    """组装 TVM LocalBuilder + LocalRunner + RecordToFile。"""
    builder = auto_scheduler.measure.LocalBuilder(timeout=30)
    runner = auto_scheduler.measure.LocalRunner(
        timeout=run_timeout,
        repeat=repeat,
        number=number,
        enable_cpu_cache_flush=enable_cpu_cache_flush,
        min_repeat_ms=min_repeat_ms,
    )
    # RecordToFile 以 append 模式打开文件，便于就地补测时续写
    measurer = auto_scheduler.measure.ProgramMeasurer(
        builder,
        runner,
        [auto_scheduler.RecordToFile(log_filename)],
        verbose=verbose,
    )
    return measurer


def _make_measurer_kwargs(task, target: tvm.target.Target):
    if target.kind.name == "llvm":
        kw = {
            "run_timeout": 5,
            "number": 1,
            "enable_cpu_cache_flush": True,
            "verbose": 1,
            "min_repeat_ms": 100,
        }
        flop_ct = task.compute_dag.flop_ct
        if flop_ct >= 2416443392.0:
            kw["repeat"] = 4
        elif flop_ct >= 834928640.0:
            kw["repeat"] = 6
        elif flop_ct <= 2097152.0:
            kw["repeat"] = 10
        else:
            kw["repeat"] = 8
        return kw
    if target.kind.name == "cuda":
        return {
            "run_timeout": 5,
            "number": 3,
            "enable_cpu_cache_flush": False,
            "verbose": 1,
            "repeat": 1,
            "min_repeat_ms": 300,
        }
    raise ValueError(f"Unsupported target kind: {target.kind.name}")


# ---------------------------------------------------------------------------
# 记录分拣 & 就地补测
# ---------------------------------------------------------------------------


def _latency_of(json_line: dict):
    """返回一条 record 的平均 latency；无效（0 / inf / 空）时返回 None。"""
    r = json_line.get("r")
    if not r:
        return None
    costs = r[0]
    if not costs:
        return None
    valid = [x for x in costs if x and x > 0 and x < 1e9]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _split_records(path: str):
    """读文件，拆成 (已测 lines, 待测 lines)。"""
    measured, unmeasured = [], []
    if not os.path.isfile(path):
        return measured, unmeasured
    with open(path, "r") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            try:
                j = json.loads(line)
            except Exception:
                # 解析失败的行直接丢弃，避免污染后续流程
                continue
            if _latency_of(j) is not None:
                measured.append(line)
            else:
                unmeasured.append(line)
    return measured, unmeasured


def remeasure_batch(
    inputs,
    target,
    target_host,
    batch_size,
    measurer_kwargs,
    measured_path,
):
    """把一组 MeasureInput 在 target 上跑一遍，结果追加到 measured_path。"""
    measurer_kwargs = dict(measurer_kwargs)
    measurer_kwargs["log_filename"] = measured_path
    measurer = make_measurer(**measurer_kwargs)

    task = auto_scheduler.measure.recover_measure_input(inputs[0]).task
    task = auto_scheduler.SearchTask(
        workload_key=task.workload_key,
        target=target,
        target_host=target_host,
        hardware_params=task.hardware_params,
        layout_rewrite_option=task.layout_rewrite_option,
    )
    empty_policy = auto_scheduler.search_policy.EmptyPolicy(task)

    for i in range(0, len(inputs), batch_size):
        end = min(len(inputs), i + batch_size)
        inp_batch = [
            auto_scheduler.MeasureInput(task, inp.state)
            for inp in inputs[i:end]
        ]
        print(f"    batch {i}-{end}/{len(inputs)}")
        measurer.measure(task, empty_policy, inp_batch)


def measure_file_inplace(
    path: str,
    target: tvm.target.Target,
    target_host,
    batch_size: int,
    resume: bool,
    max_records: int,
) -> dict:
    """对单个 record 文件就地补测。

    返回统计字典。语义：
        - 已有 latency 的行保留；
        - 先把"保留行"以 ``'w'`` 模式重写回文件（safe checkpoint）；
        - 再把"无 latency 行"按 workload 分组，逐组调用 measurer（append 模式写回）。
    """
    stats = {
        "path": path,
        "already_measured": 0,
        "to_measure": 0,
        "skipped_over_cap": 0,
        "workload_groups": 0,
        "elapsed_seconds": 0.0,
    }

    t0 = time.time()
    measured, unmeasured = _split_records(path)
    stats["already_measured"] = len(measured)
    stats["to_measure"] = len(unmeasured)

    if resume and not unmeasured:
        stats["elapsed_seconds"] = round(time.time() - t0, 2)
        return stats

    if max_records and len(unmeasured) > max_records:
        stats["skipped_over_cap"] = len(unmeasured) - max_records
        unmeasured = unmeasured[:max_records]

    # Step 1: 覆盖写入"已测" record，作为 checkpoint
    with open(path, "w") as f:
        for line in measured:
            f.write(line + "\n")

    if not unmeasured:
        stats["elapsed_seconds"] = round(time.time() - t0, 2)
        return stats

    # Step 2: 分组解析 MeasureInput
    groups = {}
    for line in unmeasured:
        try:
            inp, _ = load_record_from_string(line)
        except Exception:
            continue
        try:
            task = auto_scheduler.measure.recover_measure_input(inp).task
        except Exception:
            continue
        groups.setdefault(task.workload_key, (task, []))[1].append(inp)
    stats["workload_groups"] = len(groups)

    if not groups:
        stats["elapsed_seconds"] = round(time.time() - t0, 2)
        return stats

    # Step 3: 逐组测量，结果 append 回 path
    for wl_key, (task, inputs) in groups.items():
        measurer_kwargs = _make_measurer_kwargs(task, target)
        remeasure_batch(
            inputs=inputs,
            target=target,
            target_host=target_host,
            batch_size=batch_size,
            measurer_kwargs=measurer_kwargs,
            measured_path=path,
        )

    stats["elapsed_seconds"] = round(time.time() - t0, 2)
    return stats


# ---------------------------------------------------------------------------
# 兼容的旧模式
# ---------------------------------------------------------------------------


def legacy_single_file_mode(args, target):
    """旧 API：--to-measure-path 读入，--measured-path 写出。支持 --resume。"""
    to_measure_path = args.to_measure_path
    measured_path = args.measured_path
    if not to_measure_path or not measured_path:
        raise ValueError(
            "Legacy mode requires both --to-measure-path and --measured-path"
        )

    # 读入待测
    with open(to_measure_path, "r") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]

    # 已测过的 input 签名（用于 --resume）
    already_done = set()
    if args.resume and os.path.isfile(measured_path):
        with open(measured_path, "r") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if not line.strip():
                    continue
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                if _latency_of(j) is None:
                    continue
                already_done.add(json.dumps(j["i"]))
        print(f"[legacy]  skip {len(already_done)} already-measured records via --resume")
    else:
        # 与旧脚本一致：清空输出文件
        with open(measured_path, "w") as f:
            pass

    to_measure_lines = []
    for line in lines:
        try:
            j = json.loads(line)
        except Exception:
            continue
        if json.dumps(j["i"]) in already_done:
            continue
        to_measure_lines.append(line)

    if args.max_records_per_file:
        to_measure_lines = to_measure_lines[: args.max_records_per_file]

    # 分组 by workload
    groups = {}
    for line in to_measure_lines:
        try:
            inp, _ = load_record_from_string(line)
        except Exception:
            continue
        task = auto_scheduler.measure.recover_measure_input(inp).task
        groups.setdefault(task.workload_key, (task, []))[1].append(inp)

    print(f"[legacy]  to measure: {len(to_measure_lines)} records across {len(groups)} workloads")
    for wl_key, (task, inputs) in groups.items():
        measurer_kwargs = _make_measurer_kwargs(task, target)
        remeasure_batch(
            inputs=inputs,
            target=target,
            target_host=args.target_host,
            batch_size=args.batch_size,
            measurer_kwargs=measurer_kwargs,
            measured_path=measured_path,
        )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def _filter_files(files, args):
    files = sorted(files)
    if args.drop_hold_out:
        try:
            hold_out = {os.path.basename(p) for p in get_hold_out_five_files(tvm.target.Target(args.target))}
        except Exception as exc:
            print(f"[warn]  failed to resolve hold-out workloads: {exc}")
            hold_out = set()
        if hold_out:
            before = len(files)
            files = [f for f in files if os.path.basename(f) not in hold_out]
            print(f"[filter] dropped {before - len(files)} hold-out files (kept {len(files)})")

    if args.start_file_idx or args.end_file_idx is not None:
        end = args.end_file_idx if args.end_file_idx is not None else len(files)
        files = files[args.start_file_idx : end]
        print(f"[filter] sliced to files[{args.start_file_idx}:{end}] -> {len(files)} files")

    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=str, default="cuda -model=4090")
    parser.add_argument("--target-host", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cuda-visible", default=None,
                        help="便利选项，等价于在启动前 export CUDA_VISIBLE_DEVICES=X")

    # 目录批量模式
    parser.add_argument("--input-dir", type=str, default=None,
                        help="measure_records 目录；提供时进入批量就地补测模式")
    parser.add_argument("--resume", action="store_true", default=None,
                        help="遇到已全部测完的文件跳过；batch 模式默认开，legacy 模式默认关")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="关闭 --resume（legacy 模式下清空输出文件）")
    parser.add_argument("--drop-hold-out", action="store_true", default=True,
                        help="跳过 hold-out workload（默认开）")
    parser.add_argument("--no-drop-hold-out", dest="drop_hold_out", action="store_false")
    parser.add_argument("--max-records-per-file", type=int, default=None,
                        help="每个文件最多测多少条待测 record；None 表示不限")
    parser.add_argument("--start-file-idx", type=int, default=0)
    parser.add_argument("--end-file-idx", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None,
                        help="最多处理多少个文件；调试用")

    # 兼容旧接口
    parser.add_argument("--to-measure-path", type=str, default=None)
    parser.add_argument("--measured-path", type=str, default=None)

    args = parser.parse_args()

    # 决定模式
    batch_mode = args.input_dir is not None
    legacy_mode = args.to_measure_path is not None or args.measured_path is not None
    if batch_mode and legacy_mode:
        parser.error(
            "不能同时指定 --input-dir 和 --to-measure-path/--measured-path，请二选一"
        )
    if not batch_mode and not legacy_mode:
        parser.error(
            "必须指定 --input-dir（批量就地模式）或 --to-measure-path + --measured-path（旧模式）"
        )

    # --resume 默认值按模式设置
    if args.resume is None:
        args.resume = True if batch_mode else False

    # 注册数据路径，加载 workload registry
    register_data_path(args.target)
    target = tvm.target.Target(args.target)
    print(f"[init]  target={target}, resume={args.resume}, batch_mode={batch_mode}")
    print(f"[init]  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    load_and_register_tasks()

    if legacy_mode:
        legacy_single_file_mode(args, target)
        return

    # batch_mode
    files = glob.glob(os.path.join(args.input_dir, "*.json"))
    files = _filter_files(files, args)
    if args.max_files:
        files = files[: args.max_files]

    print(f"[init]  will process {len(files)} files under {args.input_dir}")
    summary = {
        "total_files": len(files),
        "fully_measured_skipped": 0,
        "files_updated": 0,
        "total_already_measured": 0,
        "total_newly_measured": 0,
        "total_seconds": 0.0,
    }

    t_start = time.time()
    for idx, path in enumerate(files):
        print(f"\n===== [{idx + 1}/{len(files)}] {os.path.basename(path)} =====")
        try:
            stats = measure_file_inplace(
                path=path,
                target=target,
                target_host=args.target_host,
                batch_size=args.batch_size,
                resume=args.resume,
                max_records=args.max_records_per_file or 0,
            )
        except Exception as exc:
            print(f"[error] failed to measure {path}: {exc}")
            continue

        if args.resume and stats["to_measure"] == 0:
            summary["fully_measured_skipped"] += 1
            print(f"[skip]  {stats['already_measured']} records already have latency")
        else:
            summary["files_updated"] += 1
            summary["total_newly_measured"] += stats["to_measure"]
            print(
                f"[done]  already_measured={stats['already_measured']} "
                f"to_measure={stats['to_measure']} "
                f"workload_groups={stats['workload_groups']} "
                f"elapsed={stats['elapsed_seconds']}s"
            )
        summary["total_already_measured"] += stats["already_measured"]

    summary["total_seconds"] = round(time.time() - t_start, 2)
    print("\n========== SUMMARY ==========")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
