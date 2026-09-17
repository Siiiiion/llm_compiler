#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用预训练后的 Qwen3-0.6B 生成 TVM auto_scheduler state。

功能与 gen/gen_state.py 对齐：
1. 加载模型与 tokenizer
2. 读取 sketch 记录
3. 多 GPU 并行生成 state
4. 保存并合并测量记录
"""

from dataclasses import dataclass, field
from multiprocessing import Process, Queue
from queue import Empty

import json
import math
import os
import random
import shutil
import tempfile
import time

import torch
import tqdm
import tvm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from tvm import auto_scheduler

# Reuse existing project helpers from gen/
import sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
GEN_DIR = os.path.join(os.path.dirname(CUR_DIR), "gen")
if GEN_DIR not in sys.path:
    sys.path.append(GEN_DIR)

from common import load_and_register_tasks, register_data_path  # noqa: E402
from make_dataset import input_to_tokens  # noqa: E402
from postprocess import check_measured  # noqa: E402


@dataclass
class ScriptArguments:
    sketch_path: str = field(metadata={"help": "初始 sketch 记录路径"})
    save_path: str = field(metadata={"help": "生成记录保存路径"})
    keep_cnt: int = field(metadata={"help": "每个 workload 保留的状态数量"})
    target: str = field(metadata={"help": "目标硬件，如 'cuda -model=4090'"})

    model_name_or_path: str = field(
        default="/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1",
        metadata={"help": "预训练后 Qwen3-0.6B 的模型路径（可覆盖）"},
    )

    allow_repeat: bool = field(
        default=True,
        metadata={"help": "是否允许使用历史已测状态；单次运行内始终去重"},
    )
    is_build: bool = field(default=False, metadata={"help": "是否做实际构建测试"})
    build_timeout: int = field(
        default=30,
        metadata={"help": "单个候选的TVM构建超时秒数"},
    )
    build_n_parallel: int = field(
        default=16,
        metadata={"help": "每个worker并行构建数；多GPU分片时不要使用CPU总核数"},
    )
    max_retries: int = field(
        default=20,
        metadata={"help": "每个 workload 为达到 keep_cnt 最多生成多少轮"},
    )
    require_full_keep_cnt: bool = field(
        default=False,
        metadata={"help": "任一 workload 不足 keep_cnt 时失败且不覆盖旧输出"},
    )
    stats_path: str = field(
        default=None,
        metadata={"help": "生成统计 JSON；默认使用 <save_path>.stats.json"},
    )
    resume_path: str = field(
        default=None,
        metadata={"help": "已有分片记录；已达到keep_cnt的workload将跳过并保留"},
    )
    max_empty_rounds: int = field(
        default=5,
        metadata={"help": "连续没有新增候选时提前结束该workload"},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "是否信任远程模型代码"},
    )
    fix_mistral_regex: bool = field(
        default=True,
        metadata={"help": "加载 tokenizer 时尝试修复已知 mistral regex 问题"},
    )
    do_sample: bool = field(
        default=True,
        metadata={"help": "是否启用采样。对新结构化 checkpoint 默认建议开启"},
    )
    generation_seed: int = field(
        default=0,
        metadata={"help": "模型采样和候选截取的随机种子"},
    )
    disable_eos_stop: bool = field(
        default=False,
        metadata={
            "help": (
                "是否禁用 eos 提前终止。"
                "对新 checkpoint 默认建议保留 eos 停止"
            )
        },
    )
    use_model: bool = field(
        default=True,
        metadata={
            "help": "是否使用语言模型生成状态；关闭后直接复用 sketch 状态"
        },
    )
    fallback_to_sketch_when_invalid: bool = field(
        default=True,
        metadata={"help": "当模型生成状态全部无效时，是否回退到 sketch 状态"},
    )
    trim_last_input_token: bool = field(
        default=False,
        metadata={
            "help": (
                "是否裁掉 prompt 最后一个 token。"
                "Qwen/BPE 通常应关闭以避免边界错位"
            )
        },
    )
    min_gen_tokens: int = field(
        default=256,
        metadata={
            "help": "当 policy 给出的 max_new_tokens 过短时，至少生成这么多 token"
        },
    )
    gen_token_scale: float = field(
        default=4.0,
        metadata={
            "help": "将 policy 的 max_new_tokens 缩放后与 min_gen_tokens 取最大值"
        },
    )
    generation_batch_size: int = field(
        default=32,
        metadata={"help": "生成阶段的 batch size。更长后缀推荐使用更小 batch"},
    )
    adaptive_generation: bool = field(
        default=True,
        metadata={"help": "根据尚缺候选数和历史通过率动态缩小每轮生成prompt数"},
    )
    initial_acceptance_estimate: float = field(
        default=0.33,
        metadata={"help": "自适应首轮预估的prompt到可构建候选通过率"},
    )
    min_generation_prompts: int = field(
        default=8,
        metadata={"help": "自适应模式每轮最少使用的prompt数"},
    )
    sample_top_k: int = field(
        default=0,
        metadata={"help": "采样时使用的 top-k。0 表示关闭 top-k 截断"},
    )
    sample_top_p: float = field(
        default=1.0,
        metadata={"help": "采样时使用的 top-p"},
    )
    sample_temperature: float = field(
        default=0.6,
        metadata={"help": "采样时使用的 temperature"},
    )
    network_info_dir: str = field(
        default=None,
        metadata={"help": "覆盖 target 推导出的 network_info 目录"},
    )
    to_measure_program_dir: str = field(
        default=None,
        metadata={"help": "覆盖 target 推导出的 to_measure_programs 目录"},
    )
    measure_record_dir: str = field(
        default=None,
        metadata={"help": "覆盖 target 推导出的 measure_records 目录"},
    )


def load_tokenizer(tokenizer_path, trust_remote_code, fix_mistral_regex):
    kwargs = {"trust_remote_code": trust_remote_code}
    if fix_mistral_regex:
        kwargs["fix_mistral_regex"] = True

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **kwargs)
    except TypeError:
        # Older/newer tokenizer classes may not accept this argument.
        kwargs.pop("fix_mistral_regex", None)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **kwargs)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name_or_path, trust_remote_code, device):
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
    }
    try:
        model_kwargs["dtype"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs).to(device)
    except TypeError:
        model_kwargs.pop("dtype", None)
        model_kwargs["torch_dtype"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs).to(device)
    return model


def decode_decision_tokens(tokenizer, token_ids):
    decoded_text = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    # TVM parser expects plain whitespace-separated tokens such as SPC/CI/CR and numbers.
    decoded_text = decoded_text.strip()
    if not decoded_text:
        return []
    return decoded_text.split()


def _resolve_max_new_tokens(requested_max_new_tokens, min_gen_tokens, gen_token_scale):
    requested_max_new_tokens = max(int(requested_max_new_tokens), 1)
    scaled_budget = max(1, math.ceil(requested_max_new_tokens * gen_token_scale))
    return max(requested_max_new_tokens, scaled_budget, min_gen_tokens)


def gen_func(
    task,
    states,
    input_obj,
    tokenizer,
    model,
    device,
    gen_kwargs,
    trim_last_input_token,
    generation_batch_size,
):
    if len(states) == 0:
        return []

    tokens = input_to_tokens(task, states, input_obj)
    if len(tokens) == 0:
        return []
    tokenizer.padding_side = "left"
    batch = tokenizer(tokens, padding=True, max_length=None)

    input_ids_all = batch["input_ids"]
    attention_mask_all = batch["attention_mask"]
    batch_size = max(int(generation_batch_size), 1)

    response_list = []
    with torch.no_grad():
        for start in range(0, len(input_ids_all), batch_size):
            input_ids = input_ids_all[start : start + batch_size]
            attention_mask = attention_mask_all[start : start + batch_size]

            input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)
            if trim_last_input_token and input_ids.shape[-1] > 1:
                input_ids = input_ids[:, :-1]
                attention_mask = attention_mask[:, :-1]

            local_gen_kwargs = dict(gen_kwargs)
            available_budget = tokenizer.model_max_length - input_ids.shape[-1]
            if available_budget <= 0:
                response_list.extend([[] for _ in range(input_ids.shape[0])])
                continue
            local_gen_kwargs["max_new_tokens"] = min(
                local_gen_kwargs["max_new_tokens"], available_budget
            )

            response = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **local_gen_kwargs,
            )
            response = response[:, input_ids.shape[-1] :]
            response_list.extend(response.tolist())

    return [decode_decision_tokens(tokenizer, item) for item in response_list]


def worker(
    result_queue,
    save_path_i,
    sketch_dic_list_i,
    gen_kwargs,
    tokenizer_path,
    model_name_or_path,
    device,
    generation_seed,
    allow_repeat,
    keep_cnt,
    is_build,
    build_timeout,
    build_n_parallel,
    trust_remote_code,
    fix_mistral_regex,
    use_model,
    fallback_to_sketch_when_invalid,
    trim_last_input_token,
    min_gen_tokens,
    gen_token_scale,
    generation_batch_size,
    adaptive_generation,
    initial_acceptance_estimate,
    min_generation_prompts,
    max_empty_rounds,
    max_retries,
):
    try:
        random.seed(generation_seed)
        torch.manual_seed(generation_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(generation_seed)
        tokenizer = None
        model = None
        if use_model:
            tokenizer = load_tokenizer(tokenizer_path, trust_remote_code, fix_mistral_regex)
            model = load_model(model_name_or_path, trust_remote_code, device)
            if not gen_kwargs.get("do_sample", False):
                # Sampling-only values cause warnings in greedy mode.
                if hasattr(model, "generation_config") and model.generation_config is not None:
                    model.generation_config.temperature = None
                    model.generation_config.top_p = None
                    model.generation_config.top_k = None
            model.eval()

        builder = auto_scheduler.measure.LocalBuilder(
            timeout=build_timeout,
            n_parallel=build_n_parallel,
        )
        with open(save_path_i, "w"):
            pass

        worker_stats = []

        for workload_key, inputs in tqdm.tqdm(sketch_dic_list_i):
            stats = {
                "workload_key": workload_key,
                "device": device,
                "input_sketches": len(inputs),
                "generation_rounds": 0,
                "generation_prompts": 0,
                "generation_seconds": 0.0,
                "generation_token_budget": 0,
                "generation_token_budget_max": 0,
                "parsed_states": 0,
                "fallback_rounds": 0,
                "fallback_states": 0,
                "duplicates": 0,
                "history_skipped": 0,
                "build_attempted": 0,
                "build_valid": 0,
                "build_seconds": 0.0,
                "accepted_without_build": 0,
                "build_error_counts": {},
                "unique_accepted": 0,
                "kept": 0,
                "shortfall": 0,
                "empty_rounds": 0,
            }

            def gen_func_inner(task, states, max_new_tokens):
                resolved_max_new_tokens = _resolve_max_new_tokens(
                    max_new_tokens,
                    min_gen_tokens=min_gen_tokens,
                    gen_token_scale=gen_token_scale,
                )
                stats["generation_token_budget"] += resolved_max_new_tokens * len(states)
                stats["generation_token_budget_max"] = max(
                    stats["generation_token_budget_max"], resolved_max_new_tokens
                )
                gen_kwargs["max_new_tokens"] = resolved_max_new_tokens
                return gen_func(
                    task,
                    states,
                    inputs[0],
                    tokenizer,
                    model,
                    device,
                    gen_kwargs,
                    trim_last_input_token,
                    generation_batch_size,
                )

            policy = auto_scheduler.SketchPolicy(inputs[0].task)
            measure_inputs = []
            measure_results = []
            candidate_set = set()
            input_states = [inp.state for inp in inputs]
            input_cursor = 0
            workload_started = time.perf_counter()
            no_progress_rounds = 0

            retry_i = 0
            while retry_i < max_retries:
                stats["generation_rounds"] += 1
                remaining = max(keep_cnt - len(measure_inputs), 1)
                if adaptive_generation:
                    if stats["generation_prompts"]:
                        acceptance_rate = len(measure_inputs) / stats["generation_prompts"]
                        acceptance_rate = min(max(acceptance_rate, 0.05), 1.0)
                    else:
                        acceptance_rate = initial_acceptance_estimate
                    prompt_count = math.ceil(remaining / acceptance_rate)
                    prompt_count = min(
                        len(input_states),
                        max(min_generation_prompts, prompt_count),
                    )
                    round_states = [
                        input_states[(input_cursor + offset) % len(input_states)]
                        for offset in range(prompt_count)
                    ]
                    input_cursor = (input_cursor + prompt_count) % len(input_states)
                else:
                    round_states = input_states

                stats["generation_prompts"] += len(round_states)
                generation_started = time.perf_counter()
                if use_model:
                    all_state_list = policy.gen_states(
                        round_states, gen_func_inner
                    )
                    stats["parsed_states"] += len(all_state_list)
                else:
                    all_state_list = round_states
                    stats["parsed_states"] += len(all_state_list)
                stats["generation_seconds"] += time.perf_counter() - generation_started

                if len(all_state_list) == 0 and fallback_to_sketch_when_invalid:
                    all_state_list = round_states
                    stats["fallback_rounds"] += 1
                    stats["fallback_states"] += len(all_state_list)

                measure_inputs_tmp = []
                for state in all_state_list:
                    inp = auto_scheduler.MeasureInput(inputs[0].task, state)
                    i_str = inp.to_json()
                    if i_str in candidate_set:
                        stats["duplicates"] += 1
                        continue
                    candidate_set.add(i_str)
                    if not allow_repeat and check_measured(i_str):
                        stats["history_skipped"] += 1
                        continue
                    measure_inputs_tmp.append(inp)

                default_build_result = auto_scheduler.measure.BuildResult(None, [], 0, None, 0)
                build_started = time.perf_counter()
                if is_build:
                    stats["build_attempted"] += len(measure_inputs_tmp)
                    build_results = builder.build(measure_inputs_tmp) if measure_inputs_tmp else []
                else:
                    build_results = [default_build_result for _ in measure_inputs_tmp]
                stats["build_seconds"] += time.perf_counter() - build_started

                for res, inp in zip(build_results, measure_inputs_tmp):
                    if res.error_no == 0:
                        measure_inputs.append(inp)
                        measure_results.append(
                            auto_scheduler.MeasureResult([0.0], 0, "", 0, time.time())
                        )
                        if is_build:
                            stats["build_valid"] += 1
                        else:
                            stats["accepted_without_build"] += 1
                    elif is_build:
                        error_no = str(res.error_no)
                        stats["build_error_counts"][error_no] = (
                            stats["build_error_counts"].get(error_no, 0) + 1
                        )

                if len(measure_inputs) == stats["unique_accepted"]:
                    no_progress_rounds += 1
                else:
                    no_progress_rounds = 0
                stats["unique_accepted"] = len(measure_inputs)
                stats["empty_rounds"] = no_progress_rounds

                retry_i += 1
                if len(measure_inputs) >= keep_cnt:
                    break
                if no_progress_rounds >= max_empty_rounds:
                    print(
                        f"[EARLY_STOP] device={device} no new candidates for "
                        f"{max_empty_rounds} rounds workload={workload_key}",
                        flush=True,
                    )
                    break
                if not use_model:
                    break

            stats["unique_accepted"] = len(measure_inputs)
            if len(measure_inputs) > keep_cnt:
                measure_inputs, measure_results = zip(
                    *random.sample(list(zip(measure_inputs, measure_results)), keep_cnt)
                )
                measure_inputs = list(measure_inputs)
                measure_results = list(measure_results)

            stats["kept"] = len(measure_inputs)
            stats["shortfall"] = max(keep_cnt - len(measure_inputs), 0)
            stats["elapsed_seconds"] = time.perf_counter() - workload_started
            worker_stats.append(stats)
            status = "OK" if stats["shortfall"] == 0 else "SHORTFALL"
            print(
                f"[{status}] device={device} kept={stats['kept']}/{keep_cnt} "
                f"rounds={stats['generation_rounds']} prompts={stats['generation_prompts']} "
                f"parsed={stats['parsed_states']} build_valid={stats['build_valid']} "
                f"token_budget={stats['generation_token_budget']} "
                f"max_new_tokens={stats['generation_token_budget_max']} "
                f"gen_s={stats['generation_seconds']:.2f} "
                f"build_s={stats['build_seconds']:.2f} "
                f"elapsed_s={stats['elapsed_seconds']:.2f} duplicates={stats['duplicates']} "
                f"workload={workload_key}",
                flush=True,
            )
            if measure_inputs:
                auto_scheduler.save_records(save_path_i, measure_inputs, measure_results)

        result_queue.put({"ok": True, "device": device, "stats": worker_stats})
    except Exception as exc:
        result_queue.put({"ok": False, "device": device, "error": repr(exc)})


def _atomic_write_json(path, payload):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w") as file_obj:
            json.dump(payload, file_obj, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _atomic_merge_parts(part_paths, save_path):
    save_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tmp_path = f"{save_path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w") as output:
            for part_path in part_paths:
                with open(part_path, "r") as part:
                    shutil.copyfileobj(part, output)
        os.replace(tmp_path, save_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _completed_workloads(record_path, keep_cnt):
    """Return workload keys that already have enough records in a partial shard."""
    inputs, _ = auto_scheduler.RecordReader(record_path).read_lines()
    counts = {}
    for inp in inputs:
        key = inp.task.workload_key
        counts[key] = counts.get(key, 0) + 1
    return {key for key, count in counts.items() if count >= keep_cnt}


def _make_stats_payload(script_args, workload_stats, worker_errors):
    total_fields = (
        "input_sketches",
        "generation_rounds",
        "generation_prompts",
        "generation_seconds",
        "generation_token_budget",
        "parsed_states",
        "fallback_rounds",
        "fallback_states",
        "duplicates",
        "history_skipped",
        "build_attempted",
        "build_valid",
        "build_seconds",
        "accepted_without_build",
        "unique_accepted",
        "kept",
        "shortfall",
        "elapsed_seconds",
        "empty_rounds",
    )
    totals = {field: sum(item[field] for item in workload_stats) for field in total_fields}
    totals["generation_token_budget_max"] = max(
        (item["generation_token_budget_max"] for item in workload_stats),
        default=0,
    )
    build_error_counts = {}
    for item in workload_stats:
        for error_no, count in item["build_error_counts"].items():
            build_error_counts[error_no] = build_error_counts.get(error_no, 0) + count
    totals["build_error_counts"] = build_error_counts
    return {
        "target": str(script_args.target),
        "sketch_path": os.path.abspath(script_args.sketch_path),
        "save_path": os.path.abspath(script_args.save_path),
        "keep_cnt": script_args.keep_cnt,
        "max_retries": script_args.max_retries,
        "is_build": script_args.is_build,
        "build_timeout": script_args.build_timeout,
        "build_n_parallel": script_args.build_n_parallel,
        "adaptive_generation": script_args.adaptive_generation,
        "initial_acceptance_estimate": script_args.initial_acceptance_estimate,
        "min_generation_prompts": script_args.min_generation_prompts,
        "generation_batch_size": script_args.generation_batch_size,
        "generation_seed": script_args.generation_seed,
        "min_gen_tokens": script_args.min_gen_tokens,
        "gen_token_scale": script_args.gen_token_scale,
        "allow_historical_repeat": script_args.allow_repeat,
        "require_full_keep_cnt": script_args.require_full_keep_cnt,
        "resume_path": os.path.abspath(script_args.resume_path)
        if script_args.resume_path
        else None,
        "max_empty_rounds": script_args.max_empty_rounds,
        "resumed_workload_count": getattr(script_args, "resumed_workload_count", 0),
        "workload_count": len(workload_stats),
        "totals": totals,
        "worker_errors": worker_errors,
        "workloads": sorted(workload_stats, key=lambda item: item["workload_key"]),
    }


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    if script_args.keep_cnt <= 0:
        raise ValueError("--keep_cnt must be positive")
    if script_args.max_retries <= 0:
        raise ValueError("--max_retries must be positive")
    if script_args.build_timeout <= 0:
        raise ValueError("--build_timeout must be positive")
    if script_args.build_n_parallel <= 0:
        raise ValueError("--build_n_parallel must be positive")
    if not 0 < script_args.initial_acceptance_estimate <= 1:
        raise ValueError("--initial_acceptance_estimate must be in (0, 1]")
    if script_args.min_generation_prompts <= 0:
        raise ValueError("--min_generation_prompts must be positive")
    if script_args.max_empty_rounds <= 0:
        raise ValueError("--max_empty_rounds must be positive")

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    print("Load all tasks...")
    register_data_path(
        script_args.target,
        network_info_folder=script_args.network_info_dir,
        to_measure_program_folder=script_args.to_measure_program_dir,
        measure_record_folder=script_args.measure_record_dir,
    )
    script_args.target = tvm.target.Target(script_args.target)
    _ = load_and_register_tasks()

    tokenizer = None
    if script_args.use_model:
        tokenizer = load_tokenizer(
            script_args.model_name_or_path,
            script_args.trust_remote_code,
            script_args.fix_mistral_regex,
        )

    eos_token_id = None
    pad_token_id = 0
    if tokenizer is not None:
        eos_token_id = (
            tokenizer.eos_token_id
            if tokenizer.eos_token_id is not None
            else tokenizer.sep_token_id
        )
        pad_token_id = tokenizer.pad_token_id
    gen_kwargs = {
        "min_length": -1,
        "num_return_sequences": 1,
        "do_sample": script_args.do_sample,
        "pad_token_id": pad_token_id,
    }
    if script_args.do_sample:
        gen_kwargs["top_k"] = script_args.sample_top_k
        gen_kwargs["top_p"] = script_args.sample_top_p
        gen_kwargs["temperature"] = script_args.sample_temperature
    if not script_args.disable_eos_stop and eos_token_id is not None:
        gen_kwargs["eos_token_id"] = eos_token_id

    inputs, _ = auto_scheduler.RecordReader(script_args.sketch_path).read_lines()
    if not inputs:
        raise ValueError(f"No sketch records found: {script_args.sketch_path}")

    sketch_dic = {}
    inp_dic = {}
    for inp in tqdm.tqdm(inputs):
        workload_key = inp.task.workload_key
        inp_str = inp.to_json()
        if inp_str in inp_dic:
            inp = auto_scheduler.measure.recover_measure_input(inp_dic[inp_str])
        else:
            inp = auto_scheduler.measure.recover_measure_input(inp, rebuild_state=True)
            inp_dic[inp_str] = inp
        sketch_dic.setdefault(workload_key, []).append(inp)

    sketch_dic_list = list(sketch_dic.items())
    resumed_workload_count = 0
    if script_args.resume_path:
        if not os.path.isfile(script_args.resume_path):
            raise FileNotFoundError(f"Resume records not found: {script_args.resume_path}")
        completed = _completed_workloads(script_args.resume_path, script_args.keep_cnt)
        resumed_workload_count = sum(1 for key, _ in sketch_dic_list if key in completed)
        sketch_dic_list = [
            item for item in sketch_dic_list if item[0] not in completed
        ]
        script_args.resumed_workload_count = resumed_workload_count
        print(
            f"Resume records: {resumed_workload_count} complete workloads; "
            f"remaining={len(sketch_dic_list)}",
            flush=True,
        )

    if not sketch_dic_list:
        stats_path = script_args.stats_path or f"{script_args.save_path}.stats.json"
        _atomic_write_json(
            stats_path,
            _make_stats_payload(script_args, [], []),
        )
        _atomic_merge_parts([script_args.resume_path], script_args.save_path)
        print(f"Generation stats: {os.path.abspath(stats_path)}")
        print(f"Generated records: {os.path.abspath(script_args.save_path)}")
        return

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 0:
        raise RuntimeError("No GPU found. This script requires CUDA devices.")

    num_workers = min(num_gpus, len(sketch_dic_list))
    per_len = math.ceil(len(sketch_dic_list) / num_workers)
    tmp_folder = tempfile.mkdtemp(prefix=".gen_state-", dir=CUR_DIR)
    processes = []
    part_paths = [script_args.resume_path] if script_args.resume_path else []
    result_queue = Queue()

    try:
        for worker_i in range(num_workers):
            part_path = os.path.join(tmp_folder, f"{worker_i}_part")
            workload_slice = sketch_dic_list[
                worker_i * per_len : (worker_i + 1) * per_len
            ]
            if not workload_slice:
                continue
            device = f"cuda:{worker_i}"
            process = Process(
                target=worker,
                args=(
                    result_queue,
                    part_path,
                    workload_slice,
                    gen_kwargs,
                    script_args.model_name_or_path,
                    script_args.model_name_or_path,
                    device,
                    script_args.generation_seed + worker_i,
                    script_args.allow_repeat,
                    script_args.keep_cnt,
                    script_args.is_build,
                    script_args.build_timeout,
                    script_args.build_n_parallel,
                    script_args.trust_remote_code,
                    script_args.fix_mistral_regex,
                    script_args.use_model,
                    script_args.fallback_to_sketch_when_invalid,
                    script_args.trim_last_input_token,
                    script_args.min_gen_tokens,
                    script_args.gen_token_scale,
                    script_args.generation_batch_size,
                    script_args.adaptive_generation,
                    script_args.initial_acceptance_estimate,
                    script_args.min_generation_prompts,
                    script_args.max_empty_rounds,
                    script_args.max_retries,
                ),
            )
            process.start()
            processes.append(process)
            part_paths.append(part_path)

        worker_results = []
        pending_results = len(processes)
        while pending_results:
            try:
                worker_results.append(result_queue.get(timeout=5))
                pending_results -= 1
            except Empty:
                if all(not process.is_alive() for process in processes):
                    break

        for process in processes:
            process.join()

        while len(worker_results) < len(processes):
            worker_results.append(
                {"ok": False, "device": "unknown", "error": "worker returned no result"}
            )

        worker_errors = [result for result in worker_results if not result["ok"]]
        for process in processes:
            if process.exitcode != 0:
                worker_errors.append(
                    {
                        "ok": False,
                        "device": "unknown",
                        "error": f"worker exited with code {process.exitcode}",
                    }
                )
        workload_stats = []
        for result in worker_results:
            if result["ok"]:
                workload_stats.extend(result["stats"])

        stats_path = script_args.stats_path or f"{script_args.save_path}.stats.json"
        stats_payload = _make_stats_payload(script_args, workload_stats, worker_errors)
        _atomic_write_json(stats_path, stats_payload)
        print(f"Generation stats: {os.path.abspath(stats_path)}")

        if worker_errors:
            raise RuntimeError(f"Generation workers failed: {worker_errors}")

        shortfalls = [item for item in workload_stats if item["shortfall"] > 0]
        if shortfalls:
            print("WARNING: some workloads did not reach keep_cnt:", flush=True)
            for item in shortfalls:
                print(
                    f"  kept={item['kept']}/{script_args.keep_cnt} "
                    f"shortfall={item['shortfall']} workload={item['workload_key']}",
                    flush=True,
                )
            if script_args.require_full_keep_cnt:
                raise RuntimeError(
                    f"{len(shortfalls)} workload(s) did not reach keep_cnt; old output preserved"
                )

        _atomic_merge_parts(part_paths, script_args.save_path)
        print(f"Generated records: {os.path.abspath(script_args.save_path)}")
    finally:
        shutil.rmtree(tmp_folder, ignore_errors=True)


if __name__ == "__main__":
    main()
