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

import math
import os
import random
import shutil
import subprocess
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
    model_name_or_path: str = field(
        default="/home/qsy/workspace/gen_data/qwen3_0_6b_pretrain",
        metadata={"help": "预训练后 Qwen3-0.6B 的模型路径（可覆盖）"},
    )
    sketch_path: str = field(metadata={"help": "初始 sketch 记录路径"})
    save_path: str = field(metadata={"help": "生成记录保存路径"})
    keep_cnt: int = field(metadata={"help": "每个 workload 保留的状态数量"})
    target: str = field(metadata={"help": "目标硬件，如 'cuda -model=4090'"})

    allow_repeat: bool = field(default=True, metadata={"help": "是否允许重复状态"})
    is_build: bool = field(default=False, metadata={"help": "是否做实际构建测试"})
    trust_remote_code: bool = field(default=True, metadata={"help": "是否信任远程模型代码"})


def gen_func(task, states, input_obj, tokenizer, model, device, gen_kwargs):
    if len(states) == 0:
        return []

    tokens = input_to_tokens(task, states, input_obj)
    tokenizer.padding_side = "left"
    batch = tokenizer(tokens, padding=True, max_length=None)

    input_ids_all = batch["input_ids"]
    attention_mask_all = batch["attention_mask"]
    batch_size = 128

    response_list = []
    with torch.no_grad():
        for start in range(0, len(input_ids_all), batch_size):
            input_ids = input_ids_all[start : start + batch_size]
            attention_mask = attention_mask_all[start : start + batch_size]

            input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)[:, :-1]
            attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)[:, :-1]

            gen_kwargs["max_new_tokens"] = min(
                gen_kwargs["max_new_tokens"], tokenizer.model_max_length - input_ids.shape[-1]
            )

            response = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
            response = response[:, input_ids.shape[-1] :]
            response_list.extend(response.tolist())

    return [tokenizer.batch_decode(item) for item in response_list]


def worker(
    err_queue,
    save_path_i,
    sketch_dic_list_i,
    gen_kwargs,
    tokenizer_path,
    model_name_or_path,
    device,
    allow_repeat,
    keep_cnt,
    is_build,
    trust_remote_code,
):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype="auto",
        ).to(device)
        model.eval()

        builder = auto_scheduler.measure.LocalBuilder(timeout=30)
        if os.path.exists(save_path_i):
            os.remove(save_path_i)

        for _, inputs in tqdm.tqdm(sketch_dic_list_i):
            def gen_func_inner(task, states, max_new_tokens):
                max_new_tokens = max(max_new_tokens, 1)
                gen_kwargs["max_new_tokens"] = max_new_tokens
                return gen_func(task, states, inputs[0], tokenizer, model, device, gen_kwargs)

            policy = auto_scheduler.SketchPolicy(inputs[0].task)
            measure_inputs = []
            measure_results = []
            input_set = set()

            retry_i = 0
            while retry_i < 5:
                all_state_list = policy.gen_states([inp.state for inp in inputs], gen_func_inner)

                measure_inputs_tmp = []
                for state in all_state_list:
                    inp = auto_scheduler.MeasureInput(inputs[0].task, state)
                    i_str = inp.to_json()
                    if i_str in input_set:
                        continue
                    if allow_repeat is False and check_measured(i_str):
                        continue

                    input_set.add(i_str)
                    measure_inputs_tmp.append(inp)

                default_build_result = auto_scheduler.measure.BuildResult(None, [], 0, None, 0)
                if is_build:
                    build_results = builder.build(measure_inputs_tmp)
                else:
                    build_results = [default_build_result for _ in measure_inputs_tmp]

                for res, inp in zip(build_results, measure_inputs_tmp):
                    if res.error_no == 0:
                        measure_inputs.append(inp)
                        measure_results.append(auto_scheduler.MeasureResult([0.0], 0, "", 0, time.time()))

                retry_i += 1
                if len(measure_inputs) >= keep_cnt:
                    break

            if len(measure_inputs) > keep_cnt:
                measure_inputs, measure_results = zip(
                    *random.sample(list(zip(measure_inputs, measure_results)), keep_cnt)
                )

            auto_scheduler.save_records(save_path_i, measure_inputs, measure_results)
    except Exception as exc:
        err_queue.put(exc)


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    print("Load all tasks...")
    register_data_path(script_args.target)
    script_args.target = tvm.target.Target(script_args.target)
    _ = load_and_register_tasks()

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=script_args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    gen_kwargs = {
        "min_length": -1,
        "top_k": 0,
        "top_p": 1,
        "num_return_sequences": 1,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": eos_token_id,
    }

    inputs, _ = auto_scheduler.RecordReader(script_args.sketch_path).read_lines()

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

        if workload_key not in sketch_dic:
            sketch_dic[workload_key] = []
        sketch_dic[workload_key].append(inp)

    sketch_dic_list = list(sketch_dic.items())
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 0:
        raise RuntimeError("No GPU found. This script requires CUDA devices.")

    per_len = math.ceil(len(sketch_dic_list) / num_gpus)
    processes = []
    tmp_folder = ".gen_state"

    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)

    err_queue = Queue()

    for gpu_i in range(num_gpus):
        save_path_i = f"{tmp_folder}/{gpu_i}_part"
        sketch_dic_list_i = sketch_dic_list[gpu_i * per_len : (gpu_i + 1) * per_len]
        device = f"cuda:{gpu_i}"

        p = Process(
            target=worker,
            args=(
                err_queue,
                save_path_i,
                sketch_dic_list_i,
                gen_kwargs,
                script_args.model_name_or_path,
                script_args.model_name_or_path,
                device,
                script_args.allow_repeat,
                script_args.keep_cnt,
                script_args.is_build,
                script_args.trust_remote_code,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    if not err_queue.empty():
        raise Exception(f"An exception occurred in child process: {err_queue.get()}")

    subprocess.run(f"cat {tmp_folder}/*_part > {script_args.save_path}", shell=True, check=True)
    shutil.rmtree(tmp_folder)


if __name__ == "__main__":
    main()
