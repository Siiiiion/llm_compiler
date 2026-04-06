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
    sketch_path: str = field(metadata={"help": "初始 sketch 记录路径"})
    save_path: str = field(metadata={"help": "生成记录保存路径"})
    keep_cnt: int = field(metadata={"help": "每个 workload 保留的状态数量"})
    target: str = field(metadata={"help": "目标硬件，如 'cuda -model=4090'"})

    model_name_or_path: str = field(
        default="/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1",
        metadata={"help": "预训练后 Qwen3-0.6B 的模型路径（可覆盖）"},
    )

    allow_repeat: bool = field(default=True, metadata={"help": "是否允许重复状态"})
    is_build: bool = field(default=False, metadata={"help": "是否做实际构建测试"})
    trust_remote_code: bool = field(default=True, metadata={"help": "是否信任远程模型代码"})
    fix_mistral_regex: bool = field(
        default=True,
        metadata={"help": "加载 tokenizer 时尝试修复已知 mistral regex 问题"},
    )
    do_sample: bool = field(
        default=True,
        metadata={"help": "是否启用采样。对新结构化 checkpoint 默认建议开启"},
    )
    disable_eos_stop: bool = field(
        default=False,
        metadata={"help": "是否禁用 eos 提前终止。对新 checkpoint 默认建议保留 eos 停止"},
    )
    use_model: bool = field(
        default=True,
        metadata={"help": "是否使用语言模型生成状态；关闭后直接复用 sketch 状态"},
    )
    fallback_to_sketch_when_invalid: bool = field(
        default=True,
        metadata={"help": "当模型生成状态全部无效时，是否回退到 sketch 状态"},
    )
    trim_last_input_token: bool = field(
        default=False,
        metadata={"help": "是否裁掉 prompt 最后一个 token。Qwen/BPE 通常应关闭以避免边界错位"},
    )
    min_gen_tokens: int = field(
        default=256,
        metadata={"help": "当 policy 给出的 max_new_tokens 过短时，至少生成这么多 token"},
    )
    gen_token_scale: float = field(
        default=4.0,
        metadata={"help": "对 policy 给出的 max_new_tokens 乘以该系数后再与 min_gen_tokens 取最大值"},
    )
    generation_batch_size: int = field(
        default=32,
        metadata={"help": "生成阶段的 batch size。更长后缀推荐使用更小 batch"},
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
            local_gen_kwargs["max_new_tokens"] = min(local_gen_kwargs["max_new_tokens"], available_budget)

            response = model.generate(input_ids=input_ids, attention_mask=attention_mask, **local_gen_kwargs)
            response = response[:, input_ids.shape[-1] :]
            response_list.extend(response.tolist())

    return [decode_decision_tokens(tokenizer, item) for item in response_list]


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
    fix_mistral_regex,
    use_model,
    fallback_to_sketch_when_invalid,
    trim_last_input_token,
    min_gen_tokens,
    gen_token_scale,
    generation_batch_size,
):
    try:
        tokenizer = load_tokenizer(tokenizer_path, trust_remote_code, fix_mistral_regex)
        model = load_model(model_name_or_path, trust_remote_code, device)
        if not gen_kwargs.get("do_sample", False):
            # In greedy mode these sampling-only flags are ignored; clear them to avoid noisy warnings.
            if hasattr(model, "generation_config") and model.generation_config is not None:
                model.generation_config.temperature = None
                model.generation_config.top_p = None
                model.generation_config.top_k = None
        model.eval()

        builder = auto_scheduler.measure.LocalBuilder(timeout=30)
        if os.path.exists(save_path_i):
            os.remove(save_path_i)

        for _, inputs in tqdm.tqdm(sketch_dic_list_i):
            def gen_func_inner(task, states, max_new_tokens):
                resolved_max_new_tokens = _resolve_max_new_tokens(
                    max_new_tokens,
                    min_gen_tokens=min_gen_tokens,
                    gen_token_scale=gen_token_scale,
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
            input_set = set()

            retry_i = 0
            while retry_i < 5:
                if use_model:
                    all_state_list = policy.gen_states([inp.state for inp in inputs], gen_func_inner)
                else:
                    all_state_list = [inp.state for inp in inputs]

                if len(all_state_list) == 0 and fallback_to_sketch_when_invalid:
                    all_state_list = [inp.state for inp in inputs]

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
                if not use_model:
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

    tokenizer = load_tokenizer(
        script_args.model_name_or_path,
        script_args.trust_remote_code,
        script_args.fix_mistral_regex,
    )

    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    gen_kwargs = {
        "min_length": -1,
        "num_return_sequences": 1,
        "do_sample": script_args.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if script_args.do_sample:
        gen_kwargs["top_k"] = script_args.sample_top_k
        gen_kwargs["top_p"] = script_args.sample_top_p
        gen_kwargs["temperature"] = script_args.sample_temperature
    if not script_args.disable_eos_stop and eos_token_id is not None:
        gen_kwargs["eos_token_id"] = eos_token_id

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
                script_args.fix_mistral_regex,
                script_args.use_model,
                script_args.fallback_to_sketch_when_invalid,
                script_args.trim_last_input_token,
                script_args.min_gen_tokens,
                script_args.gen_token_scale,
                script_args.generation_batch_size,
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
