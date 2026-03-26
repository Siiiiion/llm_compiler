#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TVM自动调度器网络信息转储工具

此脚本用于为TVM框架转储网络的中继IR（中间表示）和任务信息。
主要功能：
1. 导出网络的Relay IR及其参数
2. 提取并保存自动调度器任务信息
3. 收集所有任务并创建索引表

使用方法：
python dump_network_info.py --target <target_platform>
"""
import gc
import glob
import os
import pickle
import argparse
from tqdm import tqdm  # 进度条显示
import tvm
from tvm import relay
from tvm import auto_scheduler
from common import get_relay_ir_filename, get_task_info_filename, register_data_path
from tvm.meta_schedule.testing.dataset_collect_models import build_network_keys
from tvm.meta_schedule.testing.relay_workload import get_network


def dump_network(network_key, target, hardware_params):
    """转储网络的中继IR和任务信息

    参数:
        network_key: 网络标识符，包含网络名称和参数
        target: TVM目标平台
        hardware_params: 硬件参数配置
    """
    name, args = network_key  # 解包网络名称和参数
    network_task_key = (network_key,) + (target,)  # 构建网络任务键

    # 获取保存文件路径
    relay_ir_filename = get_relay_ir_filename(target, network_key)
    task_info_filename = get_task_info_filename(network_key, target)

    # 如果任务信息文件已存在，则跳过
    if os.path.exists(task_info_filename):
        return

    # 获取网络模型、参数和输入
    # relay.frontend.from_pytorch
    mod, params, inputs = get_network(*network_key)

    # 导出网络中继IR
    if not os.path.exists(relay_ir_filename):
        print(f"Dump relay ir for {network_key}...")
        mod_json = tvm.ir.save_json(mod)  # 将模块保存为JSON格式
        params_bytes = relay.save_param_dict(params)  # 保存参数字典
        # 保存模块JSON、参数字节长度和输入信息
        pickle.dump((mod_json, len(params_bytes), inputs),
                    open(relay_ir_filename, "wb"))

    # 导出任务信息
    if not os.path.exists(task_info_filename):
        print(f"Dump task info for {network_task_key}...")
        # 提取自动调度器任务
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, tvm.target.Target(target), hardware_params=hardware_params)
        # 保存任务和任务权重
        pickle.dump((tasks, task_weights), open(task_info_filename, "wb"))


def get_all_tasks():
    """收集所有任务信息并去重

    返回:
        list: 去重后的任务列表
    """
    all_task_keys = set()  # 用于存储唯一任务键
    all_tasks = []  # 存储所有唯一任务
    duplication = 0  # 记录重复任务数量

    # 获取所有任务信息文件
    filenames = glob.glob(f"{NETWORK_INFO_FOLDER}/*.task.pkl")
    filenames.sort()

    # 遍历所有文件
    for filename in tqdm(filenames):
        tasks, task_weights = pickle.load(open(filename, "rb"))
        for t in tasks:
            # 使用工作负载键和目标类型构建任务键
            task_key = (t.workload_key, str(t.target.kind))

            # 检查任务是否已存在
            if task_key not in all_task_keys:
                all_task_keys.add(task_key)
                all_tasks.append(t)
            else:
                duplication += 1

    return all_tasks


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="目标硬件平台，如'llvm'或'cuda'"
    )
    args = parser.parse_args()
    
    # 注册数据路径并解析目标平台
    register_data_path(args.target)
    args.target = tvm.target.Target(args.target)

    # 导入网络信息文件夹路径并确保其存在
    from common import NETWORK_INFO_FOLDER
    assert(NETWORK_INFO_FOLDER is not None)
    os.makedirs(NETWORK_INFO_FOLDER, exist_ok=True)

    # 构建网络键列表
    network_keys = build_network_keys()

    # 根据目标平台设置硬件参数
    if args.target.kind.name == "llvm":
        # LLVM目标平台使用默认硬件参数
        hardware_params = auto_scheduler.HardwareParams(target=args.target)
    elif args.target.kind.name == "cuda":
        # CUDA目标平台需要手动设置硬件参数
        hardware_params = auto_scheduler.HardwareParams(
            num_cores=-1,  # 核心数（-1表示自动检测）
            vector_unit_bytes=16,  # 向量单元字节数
            cache_line_bytes=64,  # 缓存行字节数
            max_shared_memory_per_block=int(args.target.attrs["max_shared_memory_per_block"]),  # 每块最大共享内存
            max_threads_per_block=int(args.target.attrs["max_threads_per_block"]),  # 每块最大线程数
            # 注意：max_local_memory_per_block在AutoScheduler中未使用，但API要求提供
            max_local_memory_per_block=12345678,
            max_vthread_extent=8,  # 最大虚拟线程范围
            warp_size=32,  # CUDA warp大小
        )
    else:
        # 不支持的目标平台
        raise NotImplementedError(f"Unsupported target {args.target}")
    
    # 为所有网络转储信息
    for key in tqdm(network_keys):
        dump_network(key, args.target, hardware_params)
        gc.collect()  # 回收内存

    # 生成包含所有任务的索引表
    tasks = get_all_tasks()
    # 按目标类型、计算图浮点操作数和工作负载键排序
    tasks.sort(key=lambda x: (str(x.target.kind), x.compute_dag.flop_ct, x.workload_key))
    # 保存排序后的任务列表
    pickle.dump(tasks, open(f"{NETWORK_INFO_FOLDER}/all_tasks.pkl", "wb"))
