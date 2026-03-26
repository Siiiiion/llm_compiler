#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TVM 自动调度辅助函数集合。

该模块负责维护数据目录、构造文件名、加载任务文件，
并为自动调度流程提供统一的文件路径入口。
"""
import pickle
from tvm import auto_scheduler
import re
import glob


NETWORK_INFO_FOLDER = None
TO_MEASURE_PROGRAM_FOLDER = None
MEASURE_RECORD_FOLDER = None
HARDWARE_PLATFORM = None

class tlm_tile:
    def __init__(self):
        pass


def clean_name(x):
    """将输入转换为文件名友好的字符串。

    该函数会去掉空格、单引号和双引号，常用于生成稳定的文件名。

    参数:
        x: 任意可转为字符串的对象。
    返回:
        清理后的字符串。
    """
    x = str(x)
    x = x.replace(" ", "")
    x = x.replace("\"", "")
    x = x.replace("'", "")
    return x


def register_data_path(target_str):
    """根据目标字符串解析硬件型号并初始化数据目录。

    参数:
        target_str: 目标平台描述字符串，例如 ``llvm`` 或 ``cuda -model=v100``。
    """
    assert(isinstance(target_str, str))
    model_list = ['i7', 'v100', 'a100', '2080', 'None', '4090']
    for model in model_list:
        if model in target_str:
            break
    assert(model != 'None')

    print(f'register data path: {model}')
    global NETWORK_INFO_FOLDER, TO_MEASURE_PROGRAM_FOLDER, MEASURE_RECORD_FOLDER, HARDWARE_PLATFORM
    NETWORK_INFO_FOLDER = f"/data3/qsy/dataset/network_info/{model}"
    TO_MEASURE_PROGRAM_FOLDER = f"/data3/qsy/dataset/to_measure_programs/{model}"
    MEASURE_RECORD_FOLDER = f"/data3/qsy/dataset/measure_records/{model}"
    HARDWARE_PLATFORM = model


def get_relay_ir_filename(target, network_key):
    """返回指定网络对应的 Relay IR 存储路径。

    参数:
        target: TVM 目标对象（保留该参数以保持接口一致）。
        network_key: 网络标识信息。
    返回:
        Relay IR 文件完整路径。
    """
    assert(NETWORK_INFO_FOLDER is not None)
    return f"{NETWORK_INFO_FOLDER}/{clean_name(network_key)}.relay.pkl"


def get_task_info_filename(network_key, target):
    """生成某个网络在指定目标下的任务信息文件路径。

    参数:
        network_key: 网络标识信息。
        target: TVM 目标对象。
    返回:
        任务信息文件完整路径。
    """
    assert(NETWORK_INFO_FOLDER is not None)
    network_task_key = (network_key,) + (str(target.kind),)
    return f"{NETWORK_INFO_FOLDER}/{clean_name(network_task_key)}.task.pkl"


def load_tasks_path(target):
    """列出当前网络信息目录下与目标平台匹配的任务文件。

    参数:
        target: TVM 目标对象。
    返回:
        匹配到的任务文件路径列表。
    """
    assert(NETWORK_INFO_FOLDER is not None)
    files = glob.glob(f"{NETWORK_INFO_FOLDER}/*{target.kind}*.pkl")
    return files


def load_and_register_tasks():
    """加载 all_tasks 文件并注册其中的 workload 张量。

    返回:
        完成注册后的任务列表。
    """
    assert(NETWORK_INFO_FOLDER is not None)
    tasks = pickle.load(open(f"{NETWORK_INFO_FOLDER}/all_tasks.pkl", "rb"))

    for task in tasks:
        auto_scheduler.workload_registry.register_workload_tensors(
            task.workload_key, task.compute_dag.tensors)

    return tasks


def get_to_measure_filename(task):
    """根据任务信息生成待测程序 JSON 文件路径。

    参数:
        task: TVM 自动调度任务对象。
    返回:
        待测程序文件完整路径。
    """
    assert(TO_MEASURE_PROGRAM_FOLDER is not None)
    task_key = (task.workload_key, str(task.target.kind))
    return f"{TO_MEASURE_PROGRAM_FOLDER}/{clean_name(task_key)}.json"


def get_measure_record_filename(task, target=None):
    """根据任务与目标平台生成测量记录 JSON 路径。

    参数:
        task: TVM 自动调度任务对象。
        target: 可选目标对象；未传入时使用 ``task.target``。
    返回:
        测量记录文件完整路径。
    """
    assert(MEASURE_RECORD_FOLDER is not None)
    target = target or task.target
    task_key = (task.workload_key, str(target.kind))
    return f"{MEASURE_RECORD_FOLDER}/{clean_name(task_key)}.json"


def hold_out_task_files(target, only_bert=False):
    """返回留出评测使用的任务文件映射。

    参数:
        target: TVM 目标对象。
        only_bert: 为 ``True`` 时仅返回 ``bert_base``。
    返回:
        字典，键为模型名，值为对应任务文件路径。
    """
    if only_bert:
        files = {
            "bert_base": get_task_info_filename(('bert_base', [1,128]), target)
        }
    else:
        files = {
            "resnet_50": get_task_info_filename(('resnet_50', [1,3,224,224]), target),
            "mobilenet_v2": get_task_info_filename(('mobilenet_v2', [1,3,224,224]), target),
            "resnext_50": get_task_info_filename(('resnext_50', [1,3,224,224]), target),
            "bert_base": get_task_info_filename(('bert_base', [1,128]), target),
            "bert_tiny": get_task_info_filename(('bert_tiny', [1,128]), target),
            
            "densenet_121": get_task_info_filename(('densenet_121', [8,3,256,256]), target),
            "bert_large": get_task_info_filename(('bert_large', [4,256]), target),
            "wide_resnet_50": get_task_info_filename(('wide_resnet_50', [8,3,256,256]), target),
            "resnet3d_18": get_task_info_filename(('resnet3d_18', [4,3,144,144,16]), target),
            "dcgan": get_task_info_filename(('dcgan', [8,3,64,64]), target)
        }
    return files


def yield_hold_out_five_files(target, only_bert=False):
    """遍历留出任务并逐条产出任务信息。

    参数:
        target: TVM 目标对象。
        only_bert: 为 ``True`` 时仅处理 BERT 任务。
    返回:
        生成器，元素为 ``(workload, task, record_file, weight)``。
    """
    files = hold_out_task_files(target, only_bert=only_bert)

    for workload, file in files.items():
        tasks_part, task_weights = pickle.load(open(file, "rb"))
        for task, weight in zip(tasks_part, task_weights):
            yield workload, task, get_measure_record_filename(task, target), weight


def get_hold_out_five_files(target):
    """收集并返回留出任务对应的测量记录文件列表。

    参数:
        target: TVM 目标对象。
    返回:
        排序后的去重文件路径列表。
    """
    files = list(set([it[2] for it in list(yield_hold_out_five_files(target))]))
    files.sort()
    return files


def get_bert_files(target):
    """收集并返回 BERT 留出任务的测量记录文件列表。

    参数:
        target: TVM 目标对象。
    返回:
        排序后的去重文件路径列表。
    """
    files = list(set([it[2] for it in list(yield_hold_out_five_files(target, True))]))
    files.sort()
    return files