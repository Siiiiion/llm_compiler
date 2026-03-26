#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TVM 自动调度器状态生成工具

该脚本使用预训练的语言模型生成TVM自动调度器的程序状态，用于优化深度学习算子在特定硬件上的性能。
主要功能包括：
1. 加载预训练的语言模型和tokenizer
2. 读取初始的sketch程序
3. 利用多GPU并行生成新的程序状态
4. 构建并保存测量记录文件

使用示例：
python gen_state.py --model_name_or_path gpt2 --sketch_path sketch.json --save_path output.json --keep_cnt 10 --target "cuda -model=4090"
"""

# 导入必要的库
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from tvm import auto_scheduler
from common import register_data_path, load_and_register_tasks
import tvm
from make_dataset import input_to_tokens
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import tqdm
import time
import os
import random
import json
from postprocess import check_measured
import math
from multiprocessing import Process, Queue
import subprocess
import shutil


@dataclass
class ScriptArguments:
    """命令行参数配置类"""
    model_name_or_path: str = field(metadata={"help": "预训练语言模型的名称或路径"})
    sketch_path: str = field(metadata={"help": "初始sketch文件路径"})
    save_path: str = field(metadata={"help": "生成结果保存路径"})
    keep_cnt: int = field(metadata={"help": "每个任务保留的状态数量"})
    target: str = field(metadata={"help": "目标硬件平台"})

    # device: str = field(default="cuda:0", metadata={"help": ""})
    allow_repeat: bool = field(default=True, metadata={"help": "是否允许重复的状态"})
    is_build: bool = field(default=False, metadata={"help": "是否进行实际构建测试"})


def gen_func(task, states, input, tokenizer, model, device, gen_kwargs):
    """
    使用语言模型生成新的程序状态
    
    参数:
        task: TVM任务对象
        states: 当前状态列表
        input: 输入数据
        tokenizer: 分词器
        model: 语言模型
        device: 运行设备
        gen_kwargs: 生成参数
    
    返回:
        list: 生成的结果列表
    """
    if len(states) == 0:
        return []
    # 将任务、状态和输入转换为token序列
    tokens = input_to_tokens(task, states, input)
    tokenizer.padding_side = "left"
    try:
        batch = tokenizer(tokens, padding=True, max_length=None)
    except Exception as e:
        print(e)
        print(task, states, input, tokenizer, model, device, gen_kwargs)
        raise Exception()
    input_ids_all = batch["input_ids"]
    attention_mask_all = batch["attention_mask"]
    batch_size = 128  # 批处理大小

    response_list = []
    with torch.no_grad():
        # 分批处理输入，避免内存不足
        for start in range(0, len(input_ids_all), batch_size):
            input_ids = input_ids_all[start : start + batch_size]
            attention_mask = attention_mask_all[start : start + batch_size]

            # 转换为tensor并移至指定设备，移除最后一个token
            input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)[:, :-1]
            attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(device)[:, :-1]
            # 确保生成的token数量不超过模型最大长度限制
            gen_kwargs['max_new_tokens'] = min(gen_kwargs['max_new_tokens'], tokenizer.model_max_length - input_ids.shape[-1])

            # 使用语言模型生成新的token序列
            response = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
            response = response[:, input_ids.shape[-1]:]  # 只取新生成的部分
            response_list.extend(response.tolist())
    # 将生成的token序列解码为文本
    return [tokenizer.batch_decode(item) for item in response_list]


def worker(err_queue, save_path_i, sketch_dic_list_i, gen_kwargs, tokenizer, model_name_or_path, device, allow_repeat, keep_cnt, is_build):
    """
    工作进程函数，负责在单个GPU上生成和处理程序状态
    
    参数:
        err_queue: 错误队列，用于传递子进程中的异常
        save_path_i: 当前进程的结果保存路径
        sketch_dic_list_i: 当前进程处理的任务列表
        gen_kwargs: 生成参数
        tokenizer: 分词器
        model_name_or_path: 模型名称或路径
        device: 运行设备
        allow_repeat: 是否允许重复状态
        keep_cnt: 保留的状态数量
        is_build: 是否进行实际构建
    """
    try:
        # 加载预训练模型并移至指定设备
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
        model.eval()
        # 创建TVM本地构建器
        builder = auto_scheduler.measure.LocalBuilder(timeout=30)
        # 如果保存路径已存在，则删除
        if os.path.exists(save_path_i):
            os.remove(save_path_i)
        
        # 遍历所有工作负载任务
        for workload_key, inputs in tqdm.tqdm(sketch_dic_list_i):
            # 内部函数，用于生成新状态
            def gen_func_inner(task, states, max_new_tokens):
                max_new_tokens = max(max_new_tokens, 1)  # 确保至少生成1个token
                gen_kwargs["max_new_tokens"] = max_new_tokens
                return gen_func(task, states, inputs[0], tokenizer, model, device, gen_kwargs)

            # 创建TVM自动调度器的Sketch策略
            policy = auto_scheduler.SketchPolicy(inputs[0].task)
            measure_inputs = []  # 存储测量输入
            measure_results = []  # 存储测量结果
            input_set = set()  # 用于去重
            
            retry_i = 0  # 重试计数器
            # 最多尝试5次生成足够的状态
            while retry_i < 5:
                # 使用Sketch策略生成新状态
                all_state_list = policy.gen_states([inp.state for inp in inputs], gen_func_inner)

                measure_inputs_tmp = []
                # 处理生成的每个状态
                for state in all_state_list:
                    # 创建测量输入对象
                    inp = auto_scheduler.MeasureInput(inputs[0].task, state)
                    i_str = inp.to_json()
                    # 跳过重复的输入
                    if i_str in input_set:
                        continue
                    # 如果不允许重复且该输入已经被测量过，则跳过
                    if allow_repeat is False and check_measured(i_str):
                        continue
                        
                    input_set.add(i_str)
                    measure_inputs_tmp.append(inp)

                # 默认构建结果
                default_build_result = auto_scheduler.measure.BuildResult(None, [], 0, None, 0)
                # 如果需要实际构建，则执行构建过程
                if is_build:
                    build_results = builder.build(measure_inputs_tmp)
                else:
                    build_results = [default_build_result for x in measure_inputs_tmp]
                
                # 处理构建结果，只保留构建成功的
                for res, inp in zip(build_results, measure_inputs_tmp):
                    if res.error_no == 0:
                        measure_inputs.append(inp)
                        # 创建测量结果对象，使用默认时间0.0
                        measure_results.append(auto_scheduler.MeasureResult([0.0], 0, "", 0, time.time()))

                retry_i += 1
                # 如果已经收集了足够的状态，则停止重试
                if len(measure_inputs) >= keep_cnt:
                    break
            
            # 如果收集的状态超过需要保留的数量，则随机采样
            if len(measure_inputs) > keep_cnt:
                measure_inputs, measure_results = zip(
                    *random.sample(list(zip(measure_inputs, measure_results)), keep_cnt)
                )
            # 保存测量记录
            auto_scheduler.save_records(save_path_i, measure_inputs, measure_results)
    except Exception as e:
        # 将异常放入队列，供主进程处理
        err_queue.put(e)


def main():
    """主函数，负责解析参数、准备数据并启动工作进程"""
    # 创建参数解析器并解析命令行参数
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    # 设置tokenizer并行处理
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    # 加载任务注册表
    print("Load all tasks...")
    register_data_path(script_args.target)  # 注册数据路径
    script_args.target = tvm.target.Target(script_args.target)  # 解析目标硬件
    tasks = load_and_register_tasks()  # 加载并注册所有任务

    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    # 设置生成参数
    gen_kwargs = {
        "min_length": -1,  # 不限制最小长度
        "top_k": 0,  # 不使用top-k采样
        "top_p": 1,  # 使用top-p采样，这里设为1表示不进行采样
        "num_return_sequences": 1,  # 每个输入只生成一个序列
        "do_sample": True,  # 启用采样
        "pad_token_id": tokenizer.pad_token_id,  # padding token ID
        "eos_token_id": tokenizer.sep_token_id  # 结束token ID
    }

    # 读取sketch文件
    inputs, _ = auto_scheduler.RecordReader(script_args.sketch_path).read_lines()
    sketch_dic = {}  # 按工作负载分组的sketch字典
    inp_dic = {}  # 用于去重的输入字典
    # 遍历所有输入，按工作负载分组
    for inp in tqdm.tqdm(inputs):
        workload_key = inp.task.workload_key
        inp_str = inp.to_json()
        # 检查是否已经处理过相同的输入
        if inp_str in inp_dic:
            inp = auto_scheduler.measure.recover_measure_input(inp_dic[inp_str])
        else:
            # 恢复测量输入并重建状态
            inp = auto_scheduler.measure.recover_measure_input(inp, rebuild_state=True)
            inp_dic[inp_str] = inp
        # 按工作负载键分组
        if workload_key not in sketch_dic:
            sketch_dic[workload_key] = []
        sketch_dic[workload_key].append(inp)

    # 准备并行处理
    sketch_dic_list = list(sketch_dic.items())
    num_gpus = torch.cuda.device_count()  # 获取可用GPU数量
    per_len = math.ceil(len(sketch_dic_list) / num_gpus)  # 每个GPU处理的任务数量
    processes = []  # 存储所有进程
    tmp_folder = '.gen_state'  # 临时文件夹，用于存储各进程的结果
    
    # 创建临时文件夹
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)  # 如果已存在则删除
    os.makedirs(tmp_folder)
    
    err_queue = Queue()  # 用于收集子进程中的异常
    
    # 为每个GPU创建一个工作进程
    for gpu_i in range(num_gpus):
        save_path_i = f'{tmp_folder}/{gpu_i}_part'  # 当前进程的结果保存路径
        # 分配当前GPU处理的任务
        sketch_dic_list_i = sketch_dic_list[gpu_i*per_len : (gpu_i+1)*per_len]
        device = f'cuda:{gpu_i}'  # 当前GPU设备
        # 创建并启动进程
        p = Process(target=worker, args=(err_queue, save_path_i, sketch_dic_list_i, gen_kwargs, tokenizer, script_args.model_name_or_path, device, script_args.allow_repeat, script_args.keep_cnt, script_args.is_build))
        p.start()
        processes.append(p)
    
    # 等待所有进程完成
    for p in processes:
        p.join()

    # 检查是否有异常发生
    if not err_queue.empty():
        raise Exception(f"An exception occurred in the child process: {err_queue.get()}")

    # 合并所有进程的结果
    subprocess.run(f"cat {tmp_folder}/*_part > {script_args.save_path}", shell=True)
    # 删除临时文件夹
    shutil.rmtree(tmp_folder)
    


if __name__ == "__main__":
    # 程序入口
    main()