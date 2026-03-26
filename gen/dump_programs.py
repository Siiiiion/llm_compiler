#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TVM自动调度器程序转储工具

该脚本用于为TVM自动调度器的所有任务生成并保存程序状态。
它使用TVM的SketchPolicy生成指定数量的唯一状态，并将其保存为测量记录文件，
供后续的自动调度和模型训练使用。

主要功能：
- 为每个任务生成指定数量的唯一程序状态
- 将生成的状态保存为标准格式的测量记录文件
- 支持指定任务的索引范围和目标硬件平台
"""
import argparse
import pickle
import gc  # 垃圾回收模块，用于释放内存
import glob
import time
import os
from tqdm import tqdm  # 进度条显示
from tvm import auto_scheduler  # TVM自动调度器模块
from common import register_data_path, load_and_register_tasks, get_to_measure_filename
import tvm


def dump_program(task, size, max_retry_iter=5):
    """为指定任务生成并保存程序状态

    参数:
        task: TVM自动调度器任务对象
        size: 要生成的唯一状态数量
        max_retry_iter: 最大重试迭代次数，当连续没有新状态产生时
    
    返回值:
        无，但会将生成的状态保存到文件
    """
    # 获取要保存的文件名
    filename = get_to_measure_filename(task)
    # 如果文件已存在，则直接返回
    if os.path.exists(filename):
        return

    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # 创建SketchPolicy对象，用于生成程序状态
    policy = auto_scheduler.SketchPolicy(task,
            params={'evolutionary_search_num_iters': 1,
                    'evolutionary_search_population': min(size, 2560),
                    'max_innermost_split_factor': 1024}, verbose=0)

    # 生成唯一状态
    all_state_str_set = set()  # 用于存储已生成状态的字符串表示，用于去重
    all_state_list = []  # 用于存储唯一的状态对象

    retry_ct = 0  # 重试计数器，记录连续没有新状态的次数
    # niter = 0

    # 当状态数量未达到要求且重试次数未超过最大值时继续生成
    while len(all_state_list) < size and retry_ct < max_retry_iter:
        # 采样初始种群
        states = policy.sample_initial_population()

        ct_before = len(all_state_list)  # 记录当前状态数量

        # states = policy.evolutionary_search(states, len(states))
        # 处理每个状态，确保唯一性
        for s in states:
            str_s = str(s)  # 转换为字符串表示用于去重
            if str_s not in all_state_str_set:
                all_state_str_set.add(str_s)
                all_state_list.append(s)

        # 如果已达到所需数量，跳出循环
        if len(all_state_list) >= size:
            break

        ct_after = len(all_state_list)  # 记录处理后的状态数量

        # 如果没有新状态添加，增加重试计数
        if ct_before == ct_after:
            retry_ct += 1
        else:
            retry_ct = 0  # 有新状态添加，重置重试计数

    # 打印生成的状态数量
    print(len(all_state_list))
    # niter += 1
    # all_state_list = all_state_list[:size]

    # 创建测量输入和结果
    measure_inputs = []
    measure_results = []
    for state in all_state_list:
        # 为每个状态创建测量输入
        measure_inputs.append(auto_scheduler.MeasureInput(task, state))
        # 创建对应的测量结果（这里使用默认值）
        measure_results.append(auto_scheduler.MeasureResult([0.0], 0, "", 0, time.time()))

    # 将记录保存到文件
    auto_scheduler.save_records(filename, measure_inputs, measure_results)


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='为TVM自动调度器任务生成并保存程序状态')
    parser.add_argument("--start-idx", type=int, help='要处理的第一个任务的索引')
    parser.add_argument("--end-idx", type=int, help='要处理的最后一个任务的索引+1')
    parser.add_argument("--size", type=int, default=1000, help='为每个任务生成的状态数量，默认1000')
    parser.add_argument("--target", type=str, required=True, help='目标硬件平台，例如"cuda"或"llvm"')
    args = parser.parse_args()

    # 注册数据路径
    register_data_path(args.target)
    # 将目标字符串转换为TVM目标对象
    args.target = tvm.target.Target(args.target)
    # 加载并注册所有任务
    tasks = load_and_register_tasks()

    # 确定任务处理范围
    start_idx = args.start_idx or 0  # 如果未指定，从0开始
    end_idx = args.end_idx or len(tasks)  # 如果未指定，处理所有任务

    # 为所有任务生成并保存程序状态
    for task in tqdm(tasks[start_idx:end_idx]):
        dump_program(task, size=args.size)
        gc.collect()  # 回收内存，避免内存泄漏

