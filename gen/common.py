#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TVM自动调度器通用工具函数模块

此模块提供了TVM自动调度器相关的通用工具函数，主要用于：
1. 管理数据路径和文件命名
2. 注册和加载任务信息
3. 获取特定网络模型的任务文件

这些工具函数被其他TVM自动调度器相关脚本（如dump_network_info.py、make_dataset.py等）共享使用。
"""
import pickle
from tvm import auto_scheduler
import re
import glob


# 数据路径全局变量
NETWORK_INFO_FOLDER = None  # 网络信息文件夹路径
TO_MEASURE_PROGRAM_FOLDER = None  # 待测量程序文件夹路径
MEASURE_RECORD_FOLDER = None  # 测量记录文件夹路径
HARDWARE_PLATFORM = None  # 硬件平台类型

class tlm_tile:
    def __init__(self):
        pass


def clean_name(x):
    """清理字符串，移除空格和引号

    参数:
        x: 输入字符串
    返回:
        str: 清理后的字符串
    """
    x = str(x)
    x = x.replace(" ", "")  # 移除空格
    x = x.replace("\"", "")  # 移除双引号
    x = x.replace("'", "")  # 移除单引号
    return x


def register_data_path(target_str):
    """根据目标平台字符串注册数据路径

    参数:
        target_str: 目标平台字符串，如"llvm"、"cuda -model=v100"
    """
    assert(isinstance(target_str, str))
    # 支持的硬件平台模型列表
    model_list = ['i7', 'v100', 'a100', '2080', 'None', '4090']
    # 从目标字符串中识别硬件平台
    for model in model_list:
        if model in target_str:
            break
    assert(model != 'None')  # 确保识别到了有效的硬件平台

    print(f'register data path: {model}')
    # 设置全局数据路径变量
    global NETWORK_INFO_FOLDER, TO_MEASURE_PROGRAM_FOLDER, MEASURE_RECORD_FOLDER, HARDWARE_PLATFORM
    NETWORK_INFO_FOLDER = f"/data3/qsy/dataset/network_info/{model}"  # 网络信息文件夹
    TO_MEASURE_PROGRAM_FOLDER = f"/data3/qsy/dataset/to_measure_programs/{model}"  # 待测量程序文件夹
    MEASURE_RECORD_FOLDER = f"/data3/qsy/dataset/measure_records/{model}"  # 测量记录文件夹
    HARDWARE_PLATFORM = model  # 硬件平台类型


def get_relay_ir_filename(target, network_key):
    """获取Relay IR文件的保存路径

    参数:
        target: TVM目标平台
        network_key: 网络标识符
    返回:
        str: Relay IR文件路径
    """
    assert(NETWORK_INFO_FOLDER is not None)
    # 生成清理后的文件名并拼接路径
    return f"{NETWORK_INFO_FOLDER}/{clean_name(network_key)}.relay.pkl"


def get_task_info_filename(network_key, target):
    """获取任务信息文件的保存路径

    参数:
        network_key: 网络标识符
        target: TVM目标平台
    返回:
        str: 任务信息文件路径
    """
    assert(NETWORK_INFO_FOLDER is not None)
    # 构建包含网络和目标平台的任务键
    network_task_key = (network_key,) + (str(target.kind),)
    # 生成清理后的文件名并拼接路径
    return f"{NETWORK_INFO_FOLDER}/{clean_name(network_task_key)}.task.pkl"


def load_tasks_path(target):
    """加载特定目标平台的所有任务文件路径

    参数:
        target: TVM目标平台
    返回:
        list: 任务文件路径列表
    """
    assert(NETWORK_INFO_FOLDER is not None)
    # 使用glob匹配所有包含目标平台类型的任务文件
    files = glob.glob(f"{NETWORK_INFO_FOLDER}/*{target.kind}*.pkl")
    return files


def load_and_register_tasks():
    """加载所有任务并注册到自动调度器工作负载注册表

    返回:
        list: 已注册的任务列表
    """
    assert(NETWORK_INFO_FOLDER is not None)
    # 加载所有任务信息
    tasks = pickle.load(open(f"{NETWORK_INFO_FOLDER}/all_tasks.pkl", "rb"))

    # 将每个任务的工作负载张量注册到工作负载注册表
    for task in tasks:
        auto_scheduler.workload_registry.register_workload_tensors(
            task.workload_key, task.compute_dag.tensors)

    return tasks


def get_to_measure_filename(task):
    """获取待测量程序文件的保存路径

    参数:
        task: TVM自动调度器任务
    返回:
        str: 待测量程序文件路径
    """
    assert(TO_MEASURE_PROGRAM_FOLDER is not None)
    # 构建任务键
    task_key = (task.workload_key, str(task.target.kind))
    # 生成清理后的文件名并拼接路径
    return f"{TO_MEASURE_PROGRAM_FOLDER}/{clean_name(task_key)}.json"


def get_measure_record_filename(task, target=None):
    """获取测量记录文件的保存路径

    参数:
        task: TVM自动调度器任务
        target: 可选的目标平台，默认为任务的目标平台
    返回:
        str: 测量记录文件路径
    """
    assert(MEASURE_RECORD_FOLDER is not None)
    # 如果未提供目标平台，则使用任务的目标平台
    target = target or task.target
    # 构建任务键
    task_key = (task.workload_key, str(target.kind))
    # 生成清理后的文件名并拼接路径
    return f"{MEASURE_RECORD_FOLDER}/{clean_name(task_key)}.json"


def hold_out_task_files(target, only_bert=False):
    """获取留出测试的任务文件路径

    参数:
        target: TVM目标平台
        only_bert: 是否只返回BERT模型的文件，默认为False
    返回:
        dict: 模型名称到任务文件路径的映射
    """
    if only_bert:
        # 只返回BERT基础模型的任务文件
        files = {
            "bert_base": get_task_info_filename(('bert_base', [1,128]), target)
        }
    else:
        # 返回多个常见网络模型的任务文件
        files = {
            "resnet_50": get_task_info_filename(('resnet_50', [1,3,224,224]), target),  # ResNet-50模型
            "mobilenet_v2": get_task_info_filename(('mobilenet_v2', [1,3,224,224]), target),  # MobileNetV2模型
            "resnext_50": get_task_info_filename(('resnext_50', [1,3,224,224]), target),  # ResNeXt-50模型
            "bert_base": get_task_info_filename(('bert_base', [1,128]), target),  # BERT基础模型
            # "gpt2": get_task_info_filename(('gpt2', [1,128]), target),  # GPT-2模型（注释掉）
            # "llama": get_task_info_filename(('llama', [4,256]), target),  # LLaMA模型（注释掉）
            "bert_tiny": get_task_info_filename(('bert_tiny', [1,128]), target),  # BERT小型模型
            
            "densenet_121": get_task_info_filename(('densenet_121', [8,3,256,256]), target),  # DenseNet-121模型
            "bert_large": get_task_info_filename(('bert_large', [4,256]), target),  # BERT大型模型
            "wide_resnet_50": get_task_info_filename(('wide_resnet_50', [8,3,256,256]), target),  # Wide ResNet-50模型
            "resnet3d_18": get_task_info_filename(('resnet3d_18', [4,3,144,144,16]), target),  # 3D ResNet-18模型
            "dcgan": get_task_info_filename(('dcgan', [8,3,64,64]), target)  # DCGAN模型
        }
    return files


def yield_hold_out_five_files(target, only_bert=False):
    """生成留出测试的五个文件的信息

    参数:
        target: TVM目标平台
        only_bert: 是否只返回BERT模型的文件，默认为False
    返回:
        generator: 生成(工作负载名称, 任务, 测量记录文件路径, 任务权重)的元组
    """
    files = hold_out_task_files(target, only_bert=only_bert)

    # 遍历每个工作负载的任务文件
    for workload, file in files.items():
        # 加载任务和任务权重
        tasks_part, task_weights = pickle.load(open(file, "rb"))
        # 遍历每个任务和对应的权重
        for task, weight in zip(tasks_part, task_weights):
            # 生成工作负载名称、任务、测量记录文件路径和权重的元组
            yield workload, task, get_measure_record_filename(task, target), weight


def get_hold_out_five_files(target):
    """获取所有留出测试的测量记录文件路径

    参数:
        target: TVM目标平台
    返回:
        list: 排序后的测量记录文件路径列表
    """
    # 获取所有测量记录文件路径，并去重
    files = list(set([it[2] for it in list(yield_hold_out_five_files(target))]))
    # 排序文件路径
    files.sort()
    return files


def get_bert_files(target):
    """获取BERT模型的测量记录文件路径

    参数:
        target: TVM目标平台
    返回:
        list: 排序后的BERT模型测量记录文件路径列表
    """
    # 获取BERT模型的测量记录文件路径，并去重
    files = list(set([it[2] for it in list(yield_hold_out_five_files(target, True))]))
    # 排序文件路径
    files.sort()
    return files