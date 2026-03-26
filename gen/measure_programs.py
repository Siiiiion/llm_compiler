"""程序性能测量工具

功能：测量各种硬件平台上程序的实际运行性能

使用示例：
python3 measure_programs.py --target "llvm -mcpu=core-avx2 -model=e5-2666"  # 在Intel E5-2666 CPU上测量
python3 measure_programs.py --target "llvm -mcpu=core-avx2 -model=e5-2673"  # 在Intel E5-2673 CPU上测量
python3 measure_programs.py --target "llvm -mcpu=core-avx2 -model=epyc-7452"  # 在AMD EPYC 7452 CPU上测量
python3 measure_programs.py --target "llvm -mcpu=core-avx2 -model=epyc-7r32"  # 在AMD EPYC 7R32 CPU上测量
python3 measure_programs.py --target "llvm -mcpu=core-avx2 -model=i7-8750h"  # 在Intel i7-8750H CPU上测量
python3 measure_programs.py --target "llvm -mcpu=skylake-avx512 -model=platinum-8124m"  # 在Intel Platinum 8124M CPU上测量
python3 measure_programs.py --target "llvm -mcpu=skylake-avx512 -model=platinum-8272l"  # 在Intel Platinum 8272L CPU上测量
python3 measure_programs.py --target "llvm -mtriple=aarch64-linux-gnu -mattr=+neon,+v8.2a,+dotprod -model=graviton2"  # 在AWS Graviton2 ARM CPU上测量
python3 measure_programs.py --target "llvm -mtriple=aarch64-linux-gnu -mattr=+neon -model=a72" --other-args "--rpc-device-key rasp4b-64 --rpc-host kraken --rpc-port 9191 --rpc-n-parallel 4"  # 在树莓派4B上远程测量
"""

# 导入必要的库
import argparse  # 解析命令行参数
import glob  # 文件路径匹配
import os  # 操作系统接口
import pickle  # 数据序列化
import time  # 时间操作

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 设置可见的CUDA设备

from tqdm import tqdm  # 进度条显示

import tvm  # TVM深度学习编译器
from tvm import auto_scheduler  # TVM自动调度器

# 导入自定义工具函数
from common import (register_data_path, load_and_register_tasks,
    get_measure_record_filename, get_to_measure_filename)
import json  # JSON数据处理
from tvm.auto_scheduler.measure_record import load_record_from_string  # 从字符串加载测量记录

def make_measurer(run_timeout, repeat, number, enable_cpu_cache_flush,
                  verbose, log_filename, min_repeat_ms):
    """
    创建程序性能测量器
    
    参数:
        run_timeout: 运行超时时间(秒)
        repeat: 重复测量次数
        number: 每次测量中运行的次数
        enable_cpu_cache_flush: 是否启用CPU缓存刷新
        verbose: 详细日志级别
        log_filename: 测量结果日志文件路径
        min_repeat_ms: 最小重复测量时间(毫秒)
    
    返回:
        ProgramMeasurer: 配置好的程序测量器实例
    """
    # 创建本地构建器，负责编译生成可执行代码，超时时间30秒
    builder = auto_scheduler.measure.LocalBuilder(timeout=30)
    
    # 创建本地运行器，负责在目标硬件上执行代码并测量性能
    runner = auto_scheduler.measure.LocalRunner(
        timeout=run_timeout,  # 运行超时时间
        repeat=repeat,        # 重复测量次数
        number=number,        # 每次测量运行多少次
        enable_cpu_cache_flush=enable_cpu_cache_flush,  # 是否刷新CPU缓存
        min_repeat_ms=min_repeat_ms  # 最小重复测量时间
    )
    
    # 创建程序测量器，连接构建器和运行器，并设置结果记录器
    measurer = auto_scheduler.measure.ProgramMeasurer(
	builder,
	runner,
        [auto_scheduler.RecordToFile(log_filename)],  # 将结果记录到文件
	verbose=verbose,  # 设置日志详细程度
    )
    return measurer


def remeasure_file(task_idx, inputs, target, target_host, batch_size, measurer_kwargs, measured_path):
    """
    重新测量一组输入程序在目标硬件上的性能
    
    参数:
        task_idx: 任务索引
        inputs: 要测量的输入程序列表
        target: 目标硬件平台
        target_host: 目标主机平台
        batch_size: 每批测量的程序数量
        measurer_kwargs: 测量器配置参数
        measured_path: 测量结果保存路径
    """
    # 设置日志文件名并创建测量器
    measurer_kwargs['log_filename'] = measured_path
    measurer = make_measurer(**measurer_kwargs)
    

    # 从第一个输入恢复任务信息，并使用指定的目标平台重新创建任务
    task = auto_scheduler.measure.recover_measure_input(inputs[0]).task
    task = auto_scheduler.SearchTask(
        workload_key=task.workload_key,  # 工作负载标识
        target=target,  # 目标硬件平台
        target_host=target_host,  # 目标主机平台
        hardware_params=task.hardware_params,  # 硬件参数
        layout_rewrite_option=task.layout_rewrite_option,  # 布局重写选项
    )
    
    # 创建空策略，因为我们只是测量已有程序而非搜索新程序
    empty_policy = auto_scheduler.search_policy.EmptyPolicy(task)

    # 批量执行测量
    for i in range(0, len(inputs), batch_size):
        print(f"===== 任务: {task_idx}\t 程序数: {i}/{len(inputs)} =====")
        inp_batch = []
        # 准备一批测量输入
        for inp in inputs[i:min(len(inputs), i + batch_size)]:
            inp_batch.append(auto_scheduler.MeasureInput(task, inp.state))
        # 执行批量测量
        res_batch = measurer.measure(task, empty_policy, inp_batch)

        timeout_ct = 0
        for res in res_batch:
            if res.error_no == auto_scheduler.measure.MeasureErrorNo.BUILD_TIMEOUT:
                timeout_ct += 1

def main(args):
    """
    主函数，处理命令行参数并执行程序性能测量
    
    参数:
        args: 命令行参数对象
    """
    # 读取已测量过的程序记录，避免重复测量
    measured_set = set()
    # if os.path.exists(args.measured_path):
    #     with open(args.measured_path, 'r') as f:
    #         lines = f.read().strip().split('\n')
    #     for line in lines:
    #         inp_str = json.dumps(json.loads(line)['i'])  # 获取输入部分并标准化
    #         measured_set.add(inp_str)

    # 读取需要测量的程序，并过滤掉已测量过的程序
    to_measure_list = []
    with open(args.to_measure_path, 'r') as f:
        lines = f.read().strip().split('\n')
    with open(args.measured_path, 'w') as f:  # 清空测量结果文件
        pass
    for line in lines:
        if line:  # 跳过空行
            inp_str = json.dumps(json.loads(line)['i'])
            if inp_str in measured_set:  # 跳过已测量过的程序
                continue
            to_measure_list.append(line)
    
    # inputs, _ = auto_scheduler.RecordReader(args.to_measure_path).read_lines()
    input_dict = {}
    for inp_str in to_measure_list:
        inp, _ = load_record_from_string(inp_str)  # 解析测量记录
        task = auto_scheduler.measure.recover_measure_input(inp).task  # 恢复任务信息
        if task.workload_key not in input_dict:
            input_dict[task.workload_key] = (task, [])  # 创建新任务组
        input_dict[task.workload_key][1].append(inp)  # 添加到对应任务组

    end_idx = min(args.end_idx, len(tasks))  # 确定处理的任务范围

    print("len input_dict:", len(input_dict))
    # if os.path.exists(args.measured_path):
    #     os.remove(args.measured_path)

    # 遍历所有任务组进行测量
    for task_i, (workload_key, (task, records)) in enumerate(input_dict.items()):
        target = tvm.target.Target(args.target)  # 创建目标硬件平台
        
        # 根据目标硬件类型设置不同的测量参数
        if target.kind.name == 'llvm':  # CPU平台
            # 设置CPU平台的测量参数
            measurer_kwargs = {
                "run_timeout": 5,  # 运行超时时间5秒
                "number": 1,  # 每次测量运行1次
                "enable_cpu_cache_flush": True,  # 启用CPU缓存刷新以获得更准确的测量
                "verbose": 1,  # 日志详细级别
                "min_repeat_ms": 100  # 最小重复测量时间100毫秒
            }
            # 根据计算量调整重复测量次数
            if task.compute_dag.flop_ct >= 2416443392.0:  # 计算密集型任务
                measurer_kwargs['repeat'] = 4  # 重复4次
            elif task.compute_dag.flop_ct >= 834928640.0:  # 中等计算量任务
                measurer_kwargs['repeat'] = 6  # 重复6次
            elif task.compute_dag.flop_ct <= 2097152.0:  # 轻量级任务
                measurer_kwargs['repeat'] = 10  # 重复10次以提高准确性
            else:  # 一般任务
                measurer_kwargs['repeat'] = 8  # 重复8次
        elif target.kind.name == 'cuda':  # GPU平台
            # 设置GPU平台的测量参数
            measurer_kwargs = {
                "run_timeout": 5,  # 运行超时时间5秒
                "number": 3,  # 每次测量运行3次
                "enable_cpu_cache_flush": False,  # GPU不需要CPU缓存刷新
                "verbose": 1,  # 日志详细级别
                "repeat": 1,  # 重复1次
                "min_repeat_ms": 300  # 最小重复测量时间300毫秒
            }
        else:
            assert(False)  # 不支持的目标平台

        print(target)

        # 执行测量
        remeasure_file(task_i, records, target, args.target_host, args.batch_size, measurer_kwargs, args.measured_path)


if __name__ == "__main__":
    """程序入口点，解析命令行参数并启动测量过程"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 目标硬件平台，默认为Intel i7 CPU
    parser.add_argument("--target", type=str, default="nvidia/geforce-rtx-4090", help="目标硬件平台描述")
    # 目标主机平台，可选
    parser.add_argument("--target-host", type=str, help="目标主机平台描述")
    # 每批测量的程序数量，默认为128
    parser.add_argument("--batch-size", type=int, default=128, help="每批测量的程序数量")
    # 开始任务索引，默认为0
    parser.add_argument("--start-idx", type=int, default=0, help="开始处理的任务索引")
    # 结束任务索引，默认为一个大数
    parser.add_argument("--end-idx", type=int, default=1000000, help="结束处理的任务索引")
    # 任务处理步长，默认为1
    parser.add_argument("--step-idx", type=int, default=1, help="任务处理步长")
    # 需要测量的程序文件路径，必需参数
    parser.add_argument("--to-measure-path", type=str, required=True, help="包含待测量程序的文件路径")
    # 测量结果保存路径，必需参数
    parser.add_argument("--measured-path", type=str, required=True, help="测量结果保存的文件路径")
    # 解析命令行参数
    args = parser.parse_args()

    # 注册数据路径，根据目标平台设置数据目录
    register_data_path(args.target)
    print("加载所有任务...")
    # 加载并注册所有任务
    tasks = load_and_register_tasks()

    # 执行主函数
    main(args)



# python measure_programs.py --target="llvm -mcpu=core-avx2 -model=i7" --to-measure-path=1122.json --measured-path=tmp.json

# python measure_programs.py --target="nvidia/nvidia-v100" --to-measure-path=1122.json --measured-path=tmp.json
