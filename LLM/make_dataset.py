#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TVM 自动调度器数据集生成工具

该脚本用于处理 TVM 自动调度器的测量记录，并生成训练和评估数据集，
支持多种数据处理模式，包括生成令牌器、生成训练数据、提取最佳调度等。
"""
from dataclasses import dataclass, field
from transformers import HfArgumentParser, set_seed
import glob
import os
import random
import logging
import time
from multiprocessing import Pool
import json
import copy
from tvm import auto_scheduler
from tvm.auto_scheduler.measure_record import load_record_from_string
from common import register_data_path, load_and_register_tasks, get_hold_out_five_files, get_bert_files
import tvm
from functools import partial
import subprocess
import shutil
from tokenizer import train_tokenizer, test_model_max_length
from make_dataset_utils import json_to_token, make_dataset, make_dataset_test
import re
from enum import Enum
from tvm.auto_scheduler.measure import MeasureInput
import numpy as np
import math


# 数据集生成类型常量定义
FOR_GEN_TOKENIZER = "for_gen_tokenizer"  # 用于生成令牌器的数据集
FOR_LATENCY = "for_latency"  # 用于延迟预测的数据集
FOR_GEN = "for_gen"  # 用于生成调度的数据集
FOR_GEN_BEST = "for_gen_best"  # 只包含最佳调度的数据集
FOR_GEN_EVAL_SKETCH = "for_gen_eval_sketch"  # 用于评估调度草图的数据集
FOR_GEN_EVAL_SKETCH_ONLY_BERT = "for_gen_eval_sketch_only_bert"  # 仅包含BERT模型调度草图的评估数据集
FOR_GEN_EVALTUNING_SKETCH = "for_gen_evaltuning_sketch"  # 用于评估和调优调度草图的数据集
FOR_GEN_TRAIN_SKETCH = "for_gen_train_sketch"  # 用于训练调度草图的数据集
FOR_GEN_BEST_ALL = "for_gen_best_all"  # 包含所有最佳调度的数据集


def _build_logger(for_type, save_path, tokenizer_path):
    base_dir = save_path or tokenizer_path or os.getcwd()
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"make_dataset_{for_type}.log")

    logger = logging.getLogger(f"make_dataset.{for_type}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger, log_file


@dataclass
class ScriptArguments:
    """脚本命令行参数定义"""
    for_type: str = field(metadata={"help": "数据集生成类型", "choices": [FOR_GEN_TOKENIZER, FOR_GEN, FOR_GEN_BEST, FOR_GEN_EVAL_SKETCH, FOR_GEN_TRAIN_SKETCH, FOR_GEN_BEST_ALL, FOR_GEN_EVALTUNING_SKETCH, FOR_LATENCY, FOR_GEN_EVAL_SKETCH_ONLY_BERT]})
    target: str = field(metadata={"help": "目标硬件平台"})
    dataset_path: str = field(metadata={"help": "数据集路径"})
    tokenizer_path: str = field(metadata={"help": "令牌器路径"})

    save_path: str = field(default=None, metadata={"help": "保存路径"})
    file_cnt: int = field(default=None, metadata={"help": "采样文件数量"})
    keep_cnt: int = field(default=None, metadata={"help": "保留的样本数量"})
    test_file_idx: int = field(default=None, metadata={"help": "测试文件索引"})
    schedule_file_path: str = field(default=None, metadata={"help": "调度文件路径"})
    max_length: int = field(default=None, metadata={"help": "tokenize 时的最大长度，默认自动推断"})


def for_clm_or_mlm(for_type):
    """
    根据数据集类型判断是使用因果语言模型(CLM)还是掩码语言模型(MLM)
    
    参数:
        for_type: 数据集类型
    
    返回:
        str: "clm" 或 "mlm"
    """
    if for_type == FOR_GEN_TOKENIZER or for_type == FOR_GEN or \
       for_type == FOR_GEN_BEST or for_type == FOR_GEN_EVAL_SKETCH or \
       for_type == FOR_GEN_TRAIN_SKETCH or for_type == FOR_GEN_BEST_ALL or \
       for_type == FOR_GEN_EVALTUNING_SKETCH or for_type == FOR_GEN_EVAL_SKETCH_ONLY_BERT:
        return "clm"
    elif for_type == FOR_LATENCY:
        return "mlm"
    else:
        assert(False)


def for_gen(lines):
    """
    处理一般生成类型的数据
    
    参数:
        lines: 包含测量记录的行列表
    
    返回:
        list: 处理后的数据列表，每个元素包含延迟和文本信息
    """
    # 从第一条记录恢复任务信息和计算图
    input, _ = load_record_from_string(lines[0])
    compute_dag = auto_scheduler.measure.recover_measure_input(input).task.compute_dag.print_min()

    data_list = []
    latency_min = 1e10  # 初始化最小延迟为很大的值
    for line in lines:
        json_line = json.loads(line)
        # 解析工作负载键
        workload_key = json_line["i"][0][0]
        json_line["i"][0][0] = json.loads(workload_key)

        # 处理SP(StatePoint)步骤，将其参数设置为1
        steps = json_line["i"][1][1]
        for step_idx, step in enumerate(steps):
            if step[0] == "SP":
                sp_list = step[4]
                for i in range(len(sp_list)):
                    sp_list[i] = 1

        # 计算延迟值
        latencies = json_line["r"][0]
        if latencies == [0]:
            latency = 1e10  # 无效延迟设为很大的值
        else:
            latency = sum(latencies) / len(latencies)  # 计算平均延迟
        latency_min = min(latency_min, latency)  # 更新最小延迟

        # 构建数据项
        data = {}
        data["latency"] = latency
        data["text"] = [compute_dag, json_line["i"]]
        data_list.append(data)

    # 计算每个数据项的标签值，标签为最小延迟与当前延迟的比值
    for data in data_list:
        data["labels"] = latency_min / data["latency"]

    return data_list


def for_gen_best(lines):
    """
    提取每个PPT(PostOrderRewrite)子序列的最佳调度
    
    参数:
        lines: 包含测量记录的行列表
    
    返回:
        list: 每个PPT子序列的最佳调度数据列表
    """
    # 从第一条记录恢复任务信息
    input, _ = load_record_from_string(lines[0])
    task = auto_scheduler.measure.recover_measure_input(input).task
    compute_dag = task.compute_dag.print_min()
    workload_key = task.workload_key

    ppt_str_min = {}  # 存储每个PPT字符串的最小延迟和对应数据
    latency_min = 1e10  # 初始化全局最小延迟
    random.shuffle(lines)  # 随机打乱行顺序
    
    for line in lines:
        json_line = json.loads(line)
        # 解析工作负载键
        workload_key = json_line["i"][0][0]
        json_line["i"][0][0] = json.loads(workload_key)

        ppt_str = None
        steps = json_line["i"][1][1]
        # 处理SP步骤并提取PPT字符串
        for step_idx, step in enumerate(steps):
            if step[0] == "SP":
                sp_list = step[4]
                for i in range(len(sp_list)):
                    sp_list[i] = 1
            if step[0] == "PPT":
                # 提取到PPT步骤为止的所有步骤作为键
                ppt_str = str(json_line["i"][1][1][:step_idx+1])

        # 计算延迟
        latencies = json_line["r"][0]
        if latencies == [0]:
            latency = 1e10  # 无效延迟
        else:
            latency = sum(latencies) / len(latencies)

        if latency >= 1e10:
            continue  # 跳过无效延迟

        latency_min = min(latency_min, latency)  # 更新全局最小延迟

        assert(ppt_str is not None)
        # 保存每个PPT子序列的最小延迟版本
        if ppt_str not in ppt_str_min or ppt_str_min[ppt_str][0] > latency:
            data = {}
            data["text"] = [compute_dag, json_line["i"]]
            data["latency"] = latency
            data["line"] = line
            ppt_str_min[ppt_str] = (latency, data)

    # 构建结果列表
    data_list = [it[1] for it in ppt_str_min.values()]
    # if latency_min >= 1e10:
    #     return []
    # if (len(data_list) > 2):
    #     print(len(data_list))
    data_list_new = []
    for data in data_list:
        labels = latency_min / data["latency"]
    #     if labels < 1.0:
    #         continue
        data["labels"] = labels
        data_list_new.append(data)

    # 按标签值排序，并根据硬件平台选择结果数量
    data_list_new.sort(key=lambda x: x["labels"], reverse=True)
    from common import HARDWARE_PLATFORM
    if HARDWARE_PLATFORM == 'i7':
        data_list_new = data_list_new[:1]  # i7平台只保留最好的一个
    elif HARDWARE_PLATFORM == 'v100':
        # data_list_new = data_list_new[:2]
        pass
    elif HARDWARE_PLATFORM == '4090':
        pass
    else:
        assert(False)

    return data_list_new


def softmax(x, temperature=1.0):
    """
    带温度参数的softmax函数
    
    参数:
        x: 输入数组
        temperature: 温度参数，控制概率分布的平滑程度
    
    返回:
        numpy.ndarray: 归一化后的概率分布
    """
    e_x = np.exp((x - np.max(x))/temperature)  # 减去最大值以防止数值溢出
    return e_x / e_x.sum(axis=0)  # 归一化


def for_gen_eval_sketch(lines, keep_cnt, for_type):
    """
    为评估或训练调度草图准备数据
    
    参数:
        lines: 包含测量记录的行列表
        keep_cnt: 要保留的样本数量
        for_type: 数据集类型
    
    返回:
        list: 处理后的数据集
    """
    # 从第一条记录恢复任务信息
    input, _ = load_record_from_string(lines[0])
    task = auto_scheduler.measure.recover_measure_input(input).task
    compute_dag = task.compute_dag.print_min()
    workload_key = task.workload_key

    json_line_dict = {}
    for line in lines:
        json_line = json.loads(line)
        steps = json_line["i"][1][1]
        ppt_idx = None
        # 找到第一个PPT步骤的索引
        for step_idx, step in enumerate(steps):
            if step[0] == "SP":
                # 处理SP步骤
                sp_list = step[4]
                for i in range(len(sp_list)):
                    sp_list[i] = 1
                continue
            if step[0] == "PPT":
                ppt_idx = step_idx
                break
        assert(ppt_idx is not None)
        # 只保留到PPT步骤的部分
        json_line["i"][1][1] = steps[:ppt_idx+1]

        # if for_type == FOR_GEN_TRAIN_SKETCH or for_type == FOR_GEN_EVALTUNING_SKETCH:
        costs = json_line["r"][0]
        if costs == [0]:
            latency = 1e10
        else:
            latency = sum(costs) / len(costs)
        # elif for_type == FOR_GEN_EVAL_SKETCH:
        #     latency = 1
        # else:
        #     assert(False)

        # 使用处理后的输入作为键，保存最小延迟
        json_line_str = str(json_line["i"])
        if json_line_str not in json_line_dict:
            json_line_dict[json_line_str] = [latency, json_line]
        else:
            json_line_dict[json_line_str][0] = min(json_line_dict[json_line_str][0], latency)

    # 提取延迟列表和对应的JSON行
    latency_list, json_line_list = zip(*json_line_dict.values())
    latency_min = min(latency_list)
    # 计算相对性能指标
    latency_list = [latency_min / it for it in latency_list]

    # 使用softmax计算采样概率，温度参数设为0.3以增加差异
    probs = softmax(np.array(latency_list), temperature=0.3)
    # 根据概率分布采样指定数量的样本
    indices = np.random.choice(np.arange(len(latency_list)), size=keep_cnt, replace=True, p=probs)

    # 构建结果列表
    data_list = []
    for select_i in indices:
        json_line = json_line_list[select_i]
        # json_line["labels"] = latency_list[select_i]
        # json_line["latency"] = latency_min / json_line["labels"]
        data_list.append(json_line)

    return data_list


def input_to_tokens(task, states, input):
    """
    将输入和状态转换为标记序列
    
    参数:
        task: TVM任务
        states: 状态列表
        input: 测量输入
    
    返回:
        list: 标记序列列表
    """
    compute_dag = task.compute_dag.print_min()
    json_line_i = json.loads(input.to_json())
    workload_key = json_line_i[0][0]
    json_line_i[0][0] = json.loads(workload_key)

    data_list = []
    for state in states:
        inp = MeasureInput(task, state)
        steps = json.loads(inp.to_json())[1][1]
        # 处理SP步骤
        for step_idx, step in enumerate(steps):
            if step[0] == "SP":
                sp_list = step[4]
                for i in range(len(sp_list)):
                    sp_list[i] = 1
        json_line_i[1][1] = steps
        data = {}
        data["text"] = [compute_dag, copy.deepcopy(json_line_i)]
        data_list.append(data)

    # 转换为标记序列
    return [item["text"] for item in json_to_token(data_list)]


def process_file(args, tmp_folder, for_type, keep_cnt):
    """
    处理单个文件
    
    参数:
        args: 包含文件索引和文件路径的元组
        tmp_folder: 临时文件夹路径
        for_type: 处理类型
        keep_cnt: 保留样本数量
    """
    file_i, file = args
    print(file_i, end="    \r", flush=True)  # 显示进度
    with open(file, "r") as f:
        lines = f.read().strip().split("\n")
    
    # 根据处理类型选择不同的处理函数
    if for_type == FOR_GEN_TOKENIZER or for_type == FOR_GEN or for_type == FOR_LATENCY:
        data_list = for_gen(lines)
        data_list = json_to_token(data_list)
    elif for_type == FOR_GEN_BEST or for_type == FOR_GEN_BEST_ALL:
        data_list = for_gen_best(lines)
        data_list = json_to_token(data_list)
    elif for_type == FOR_GEN_EVAL_SKETCH or for_type == FOR_GEN_TRAIN_SKETCH or for_type == FOR_GEN_EVALTUNING_SKETCH or for_type == FOR_GEN_EVAL_SKETCH_ONLY_BERT:
        data_list = for_gen_eval_sketch(lines, keep_cnt, for_type)
    else:
        assert(False)

    # 将处理结果写入临时文件
    with open(f"{tmp_folder}/{file_i}_part", "w") as f:
        for data in data_list:
            json.dump(data, f)
            f.write("\n")


def token_files_and_merge(for_type, files, save_path, keep_cnt=None):
    """
    并行处理多个文件并合并结果
    
    参数:
        for_type: 处理类型
        files: 文件列表
        save_path: 保存路径
        keep_cnt: 保留样本数量
    
    返回:
        str: 合并后的文件名
    """
    os.makedirs(save_path, exist_ok=True)  # 创建保存目录
    filename = f"{save_path}/0_merge.json"  # 合并后的文件名
    tmp_folder = f"{save_path}/0_tmp"  # 临时文件夹
    
    # 清理旧的临时文件夹
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)  # 创建新的临时文件夹
    
    # 使用多进程并行处理文件
    with Pool(os.cpu_count()) as pool:
        pool.map(partial(process_file, tmp_folder=tmp_folder, for_type=for_type, keep_cnt=keep_cnt), enumerate(files))
    print()
    
    # 合并所有临时文件
    subprocess.run(f"cat {tmp_folder}/*_part > {filename}", shell=True)
    shutil.rmtree(tmp_folder)  # 清理临时文件夹
    
    return filename


def main():
    """主函数，解析参数并执行相应的数据集生成任务"""
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    logger, log_file = _build_logger(
        script_args.for_type,
        script_args.save_path,
        script_args.tokenizer_path,
    )
    total_start_time = time.time()
    logger.info("Start run for_type=%s", script_args.for_type)
    logger.info("Args: %s", script_args)
    logger.info("Log file: %s", log_file)

    # Load task registry
    print("Load all tasks...")
    register_data_path(script_args.target)
    script_args.target = tvm.target.Target(script_args.target)
    tasks = load_and_register_tasks()
    logger.info("Task registry loaded for target=%s", script_args.target)

    # 根据不同的处理类型执行相应的处理流程
    branch_start_time = time.time()
    try:
        if script_args.for_type == FOR_GEN_TOKENIZER:
            # 生成tokenizer数据集
            files = glob.glob(os.path.join(script_args.dataset_path, "*.json"))
            files.sort()
            print("Dataset file cnt:", len(files))
            logger.info("FOR_GEN_TOKENIZER file count before sample=%d", len(files))
            if script_args.file_cnt:
                set_seed(0)
                files = random.sample(files, script_args.file_cnt)  # 采样指定数量的文件
                print("Sampled file cnt:", len(files))
                logger.info("FOR_GEN_TOKENIZER sampled file count=%d", len(files))
            # 处理文件并训练tokenizer
            filename = token_files_and_merge(script_args.for_type, files, script_args.tokenizer_path)
            logger.info("FOR_GEN_TOKENIZER merged file=%s", filename)
            train_tokenizer([filename], script_args.tokenizer_path, test_length=True)
            logger.info("FOR_GEN_TOKENIZER train_tokenizer finished")

        elif script_args.for_type == FOR_GEN:
            # 生成普通训练数据集
            files = glob.glob(os.path.join(script_args.dataset_path, "*.json"))
            files.sort()
            print("Dataset file cnt:", len(files))
            logger.info("FOR_GEN file count before hold-out=%d", len(files))
            # 排除预留的测试文件
            hold_out_files = get_hold_out_five_files(script_args.target)
            for out in hold_out_files:
                for file in files:
                    if os.path.basename(out) == os.path.basename(file):
                        files.remove(file)
            print("After hold out, file cnt:", len(files))
            logger.info("FOR_GEN file count after hold-out=%d", len(files))
            if script_args.file_cnt:
                set_seed(0)
                files = random.sample(files, script_args.file_cnt)
                print("Sampled file cnt:", len(files))
                logger.info("FOR_GEN sampled file count=%d", len(files))
            # 处理文件并创建数据集
            filename = token_files_and_merge(script_args.for_type, files, script_args.save_path)
            logger.info("FOR_GEN merged file=%s", filename)
            make_dataset(
                filename,
                script_args.save_path,
                script_args.tokenizer_path,
                for_clm_or_mlm(script_args.for_type),
                max_length=script_args.max_length,
            )
            logger.info("FOR_GEN make_dataset finished, output=%s", script_args.save_path)

        elif script_args.for_type == FOR_LATENCY:
            # 生成延迟预测数据集
            files = glob.glob(os.path.join(script_args.dataset_path, "*.json"))
            files.sort()
            print("Dataset file cnt:", len(files))
            logger.info("FOR_LATENCY file count before hold-out=%d", len(files))
            # 排除预留的测试文件
            hold_out_files = get_hold_out_five_files(script_args.target)
            for out in hold_out_files:
                for file in files:
                    if os.path.basename(out) == os.path.basename(file):
                        files.remove(file)
            print("After hold out, file cnt:", len(files))
            logger.info("FOR_LATENCY file count after hold-out=%d", len(files))
            if script_args.file_cnt:
                set_seed(0)
                files = random.sample(files, script_args.file_cnt)
                print("Sampled file cnt:", len(files))
                logger.info("FOR_LATENCY sampled file count=%d", len(files))
            # 处理文件并创建数据集
            filename = token_files_and_merge(script_args.for_type, files, script_args.save_path)
            logger.info("FOR_LATENCY merged file=%s", filename)
            make_dataset(
                filename,
                script_args.save_path,
                script_args.tokenizer_path,
                for_clm_or_mlm(script_args.for_type),
                max_length=script_args.max_length,
            )
            logger.info("FOR_LATENCY make_dataset finished, output=%s", script_args.save_path)

        elif script_args.for_type == FOR_GEN_BEST:
            # 生成最佳调度数据集
            files = glob.glob(os.path.join(script_args.dataset_path, "*.json"))
            files.sort()
            print("Dataset file cnt:", len(files))
            logger.info("FOR_GEN_BEST file count before hold-out=%d", len(files))
            # 排除预留的测试文件
            hold_out_files = get_hold_out_five_files(script_args.target)
            for out in hold_out_files:
                for file in files:
                    if os.path.basename(out) == os.path.basename(file):
                        files.remove(file)
            print("After hold out, file cnt:", len(files))
            logger.info("FOR_GEN_BEST file count after hold-out=%d", len(files))
            if script_args.file_cnt:
                set_seed(0)
                files = random.sample(files, script_args.file_cnt)
                print("Sampled file cnt:", len(files))
                logger.info("FOR_GEN_BEST sampled file count=%d", len(files))
            # 处理文件并创建数据集，valid_percentage=0表示不创建验证集
            filename = token_files_and_merge(script_args.for_type, files, script_args.save_path)
            logger.info("FOR_GEN_BEST merged file=%s", filename)
            make_dataset(
                filename,
                script_args.save_path,
                script_args.tokenizer_path,
                for_clm_or_mlm(script_args.for_type),
                valid_percentage=0,
                max_length=script_args.max_length,
            )
            logger.info("FOR_GEN_BEST make_dataset finished, output=%s", script_args.save_path)

        elif script_args.for_type == FOR_GEN_BEST_ALL:
            # 生成包含所有任务的最佳调度数据集
            files = glob.glob(os.path.join(script_args.dataset_path, "*.json"))
            files.sort()
            print("Dataset file cnt:", len(files))
            logger.info("FOR_GEN_BEST_ALL file count=%d", len(files))
            # 不排除任何文件
            filename = token_files_and_merge(script_args.for_type, files, script_args.save_path)
            logger.info("FOR_GEN_BEST_ALL merged file=%s", filename)
            make_dataset(
                filename,
                script_args.save_path,
                script_args.tokenizer_path,
                for_clm_or_mlm(script_args.for_type),
                valid_percentage=0,
                max_length=script_args.max_length,
            )
            logger.info("FOR_GEN_BEST_ALL make_dataset finished, output=%s", script_args.save_path)

        elif script_args.for_type == FOR_GEN_EVAL_SKETCH or script_args.for_type == FOR_GEN_EVALTUNING_SKETCH or script_args.for_type == FOR_GEN_EVAL_SKETCH_ONLY_BERT:
            # 生成用于评估调度草图的数据集
            files = glob.glob(os.path.join(script_args.dataset_path, "*.json"))
            files.sort()
            print("Dataset file cnt:", len(files))
            logger.info("%s file count before hold-out=%d", script_args.for_type, len(files))
            # 根据不同类型选择不同的预留文件
            if script_args.for_type == FOR_GEN_EVAL_SKETCH_ONLY_BERT:
                hold_out_files = get_bert_files(script_args.target)
            else:
                hold_out_files = get_hold_out_five_files(script_args.target)
            hold_out_set = set()
            for file in hold_out_files:
                hold_out_set.add(os.path.basename(file))
            # 只保留预留的文件
            files_new = []
            for file in files:
                if os.path.basename(file) in hold_out_set:
                    files_new.append(file)
            files = files_new
            print("After hold out, file cnt:", len(files))
            logger.info("%s file count after hold-out=%d", script_args.for_type, len(files))
            # 可选：根据调度文件筛选潜在文件
            if script_args.schedule_file_path:
                from task_sheduler import find_potential_files

                files = find_potential_files(files)
                print("Find potential file cnt:", len(files))
                logger.info("%s potential file count=%d", script_args.for_type, len(files))
            # 处理文件
            filename = token_files_and_merge(script_args.for_type, files, script_args.save_path, keep_cnt=script_args.keep_cnt)
            logger.info("%s merged file=%s", script_args.for_type, filename)

        elif script_args.for_type == FOR_GEN_TRAIN_SKETCH:
            # 生成用于训练调度草图的数据集
            files = glob.glob(os.path.join(script_args.dataset_path, "*.json"))
            files.sort()
            print("Dataset file cnt:", len(files))
            logger.info("FOR_GEN_TRAIN_SKETCH file count before hold-out=%d", len(files))
            # 排除预留的测试文件
            hold_out_files = get_hold_out_five_files(script_args.target)
            hold_out_set = set()
            for file in hold_out_files:
                hold_out_set.add(os.path.basename(file))
            files_new = []
            for file in files:
                if os.path.basename(file) not in hold_out_set:
                    files_new.append(file)
            files = files_new
            print("After hold out, file cnt:", len(files))
            logger.info("FOR_GEN_TRAIN_SKETCH file count after hold-out=%d", len(files))
            # if 'to_measure_programs' in files[0]:
            files_new = []
            for file_i, file in enumerate(files):
                if file_i % 4 == script_args.test_file_idx % 4:
                    files_new.append(file)
            files = files_new
            print(f"test_file_idx: {script_args.test_file_idx}, len files: {len(files)}")
            logger.info(
                "FOR_GEN_TRAIN_SKETCH test_file_idx=%s, selected file count=%d",
                script_args.test_file_idx,
                len(files),
            )
            # else:
            #     from task_sheduler import find_potential_files
            #     files = find_potential_files(files)
            #     print("Find potential file cnt:", len(files))
            filename = token_files_and_merge(script_args.for_type, files, script_args.save_path, keep_cnt=script_args.keep_cnt)
            logger.info("FOR_GEN_TRAIN_SKETCH merged file=%s", filename)

        else:
            assert(False)
    except Exception:
        logger.exception("Run failed for for_type=%s", script_args.for_type)
        raise
    finally:
        logger.info("for_type=%s elapsed_seconds=%.2f", script_args.for_type, time.time() - branch_start_time)
        logger.info("total elapsed_seconds=%.2f", time.time() - total_start_time)


if __name__ == "__main__":
    main()