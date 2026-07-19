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
import json
from tqdm import tqdm  # 进度条显示
import tvm
from tvm import relay
from tvm import auto_scheduler
from common import get_relay_ir_filename, get_task_info_filename, register_data_path
from tvm.meta_schedule.testing.dataset_collect_models import build_network_keys
from tvm.meta_schedule.testing.relay_workload import get_network


RELAY_ARTIFACT_FORMAT = "llm_compiler.relay"
RELAY_ARTIFACT_VERSION = 1


def parse_shape_dict(shape_str):
    """解析形如 input:[1,3,224,224] input2:[1,128] 的输入shape字符串。"""
    if not shape_str:
        return None

    try:
        raw_shape_dict = json.loads(shape_str)
        return {name: [int(dim) for dim in shape] for name, shape in raw_shape_dict.items()}
    except json.JSONDecodeError:
        pass

    from tvm.driver.tvmc.shape_parser import parse_shape_string

    return parse_shape_string(shape_str)


def get_inputs_from_shape_dict(shape_dict, dtype):
    """生成和内置workload兼容的inputs描述。"""
    return tuple(
        (name, list(shape), dtype[name] if isinstance(dtype, dict) else dtype)
        for name, shape in shape_dict.items()
    )


def fix_mask_where_constants(onnx_model):
    """Align FP64 attention-mask sentinels with FP16/FP32 Where branches."""
    import numpy as np
    from onnx import TensorProto, numpy_helper

    producers = {output: node for node in onnx_model.graph.node for output in node.output}
    patched = []

    def constant_tensor(node):
        if node is None or node.op_type != "Constant":
            return None
        for attr in node.attribute:
            if attr.name == "value":
                return attr.t
        return None

    for node in onnx_model.graph.node:
        if node.op_type != "Where" or len(node.input) != 3:
            continue
        branch_tensors = [constant_tensor(producers.get(name)) for name in node.input[1:]]
        branch_dtypes = [
            tensor.data_type if tensor is not None else None for tensor in branch_tensors
        ]
        supported_targets = {
            TensorProto.FLOAT16: np.float16,
            TensorProto.FLOAT: np.float32,
        }
        target_dtypes = [dtype for dtype in branch_dtypes if dtype in supported_targets]
        if len(target_dtypes) != 1 or branch_dtypes.count(TensorProto.DOUBLE) != 1:
            continue

        double_index = branch_dtypes.index(TensorProto.DOUBLE)
        target_index = 1 - double_index
        target_dtype = target_dtypes[0]
        numpy_dtype = supported_targets[target_dtype]
        double_tensor = branch_tensors[double_index]
        target_tensor = branch_tensors[target_index]
        double_value = numpy_helper.to_array(double_tensor)
        target_value = numpy_helper.to_array(target_tensor)
        if double_value.shape != () or target_value.shape != ():
            continue
        if float(target_value) != 0.0:
            continue
        if float(double_value) != float(np.finfo(numpy_dtype).min):
            continue

        double_tensor.CopyFrom(
            numpy_helper.from_array(np.asarray(double_value, dtype=numpy_dtype))
        )
        patched.append(
            {
                "where": node.name,
                "constant": producers[node.input[double_index + 1]].name,
                "dtype": TensorProto.DataType.Name(target_dtype),
            }
        )

    return patched


def load_onnx_network(
    model_path,
    input_shapes,
    input_dtype,
    input_dtypes=None,
    fix_mask_where_dtype=False,
):
    """从本地ONNX模型加载Relay网络。"""
    import onnx

    shape_dict = parse_shape_dict(input_shapes)
    if not shape_dict:
        raise ValueError("--input-shapes is required when dumping a local ONNX model")

    dtype_dict = {name: input_dtype for name in shape_dict}
    if input_dtypes:
        unknown = sorted(set(input_dtypes) - set(shape_dict))
        if unknown:
            raise ValueError(f"--input-dtypes contains unknown inputs: {unknown}")
        dtype_dict.update(input_dtypes)
    onnx_model = onnx.load(model_path, load_external_data=True)
    if fix_mask_where_dtype:
        patched = fix_mask_where_constants(onnx_model)
        print(f"Patched mask Where constants: {patched}")
    mod, params = relay.frontend.from_onnx(
        onnx_model, shape_dict, dtype=dtype_dict, freeze_params=False
    )
    return mod, params, get_inputs_from_shape_dict(shape_dict, dtype_dict)


def normalize_artifact_inputs(inputs):
    """将内置和多输入网络的描述统一为可扩展的字典列表。"""
    if not isinstance(inputs, (tuple, list)) or not inputs:
        raise ValueError(f"Invalid network input metadata: {inputs!r}")
    if len(inputs) == 3 and isinstance(inputs[0], str):
        inputs = [inputs]
    return [
        {"name": name, "shape": [int(dim) for dim in shape], "dtype": str(dtype)}
        for name, shape, dtype in inputs
    ]


def dump_relay_network(
    network_key,
    target,
    hardware_params,
    mod,
    params,
    inputs,
    overwrite_relay=False,
    overwrite_task_info=False,
    skip_relay_artifact=False,
):
    """转储已加载的Relay网络和任务信息。"""
    network_task_key = (network_key,) + (target,)  # 构建网络任务键
    relay_ir_filename = get_relay_ir_filename(target, network_key)
    task_info_filename = get_task_info_filename(network_key, target)

    if not skip_relay_artifact and (overwrite_relay or not os.path.exists(relay_ir_filename)):
        print(f"Dump relay ir for {network_key}...")
        mod_json = tvm.ir.save_json(mod)
        params_bytes = relay.save_param_dict(params)
        artifact = {
            "format": RELAY_ARTIFACT_FORMAT,
            "version": RELAY_ARTIFACT_VERSION,
            "mod_json": mod_json,
            "params_bytes": bytes(params_bytes),
            "inputs": normalize_artifact_inputs(inputs),
        }
        temporary = f"{relay_ir_filename}.tmp"
        with open(temporary, "wb") as file:
            pickle.dump(artifact, file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, relay_ir_filename)

    if overwrite_task_info or not os.path.exists(task_info_filename):
        print(f"Dump task info for {network_task_key}...")
        tasks, task_weights = auto_scheduler.extract_tasks(
            mod["main"], params, tvm.target.Target(target), hardware_params=hardware_params
        )
        temporary = f"{task_info_filename}.tmp"
        with open(temporary, "wb") as file:
            pickle.dump((tasks, task_weights), file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, task_info_filename)


def dump_network(
    network_key,
    target,
    hardware_params,
    overwrite_relay=False,
    overwrite_task_info=False,
    skip_relay_artifact=False,
):
    """转储网络的中继IR和任务信息

    参数:
        network_key: 网络标识符，包含网络名称和参数
        target: TVM目标平台
        hardware_params: 硬件参数配置
    """
    task_info_filename = get_task_info_filename(network_key, target)

    # 如果任务信息文件已存在，则跳过
    if os.path.exists(task_info_filename) and not overwrite_relay and not overwrite_task_info:
        return

    # 获取网络模型、参数和输入
    # relay.frontend.from_pytorch
    mod, params, inputs = get_network(*network_key)
    dump_relay_network(
        network_key,
        target,
        hardware_params,
        mod,
        params,
        inputs,
        overwrite_relay=overwrite_relay,
        overwrite_task_info=overwrite_task_info,
        skip_relay_artifact=skip_relay_artifact,
    )


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


def get_hardware_params(target):
    """根据目标平台设置硬件参数。"""
    if target.kind.name == "llvm":
        return auto_scheduler.HardwareParams(target=target)
    if target.kind.name == "cuda":
        return auto_scheduler.HardwareParams(
            num_cores=-1,
            vector_unit_bytes=16,
            cache_line_bytes=64,
            max_shared_memory_per_block=int(target.attrs["max_shared_memory_per_block"]),
            max_threads_per_block=int(target.attrs["max_threads_per_block"]),
            max_local_memory_per_block=12345678,
            max_vthread_extent=8,
            warp_size=32,
        )
    raise NotImplementedError(f"Unsupported target {target}")


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="目标硬件平台，如'llvm'或'cuda'"
    )
    parser.add_argument(
        "--hardware-name",
        type=str,
        help="数据目录使用的硬件名称。target不包含i7/v100/a100/2080/4090时必须指定。",
    )
    parser.add_argument(
        "--network-info-dir",
        type=str,
        help="网络信息输出目录，默认使用/data3/qsy/dataset/network_info/{hardware-name}",
    )
    parser.add_argument(
        "--workload",
        action="append",
        help="额外或指定内置workload，格式为name:dim,dim,...，可重复传入。",
    )
    parser.add_argument(
        "--only-workloads",
        action="store_true",
        help="只dump --workload 指定的内置workload，不dump默认数据集。",
    )
    parser.add_argument("--onnx-model", type=str, help="本地ONNX模型路径")
    parser.add_argument(
        "--onnx-manifest",
        type=str,
        help=(
            "export_onnx_qwen3.py 网格导出的 manifest.json；"
            "批量导入其中全部固定shape ONNX"
        ),
    )
    parser.add_argument("--model-name", type=str, help="本地模型名称，默认取ONNX文件名")
    parser.add_argument(
        "--input-shapes",
        type=str,
        help='ONNX输入shape，例如 "input:[1,3,224,224]" 或 JSON {"input":[1,3,224,224]}',
    )
    parser.add_argument("--input-dtype", type=str, default="float32", help="本地模型输入dtype")
    parser.add_argument(
        "--input-dtypes",
        type=str,
        help='每个ONNX输入的dtype JSON，例如 {"input_ids":"int64"}',
    )
    parser.add_argument(
        "--fix-mask-where-dtype",
        "--fix-fp16-mask-where",
        dest="fix_mask_where_dtype",
        action="store_true",
        help=(
            "将Where中误导出为FP64的attention-mask最小值常量转回另一分支的"
            "FP16/FP32 dtype；--fix-fp16-mask-where 是兼容别名"
        ),
    )
    parser.add_argument(
        "--overwrite-relay",
        action="store_true",
        help="重新生成Relay工件；用于迁移缺失参数数据的旧格式",
    )
    parser.add_argument(
        "--overwrite-task-info",
        action="store_true",
        help="重新提取对应网络的AutoScheduler任务",
    )
    parser.add_argument(
        "--skip-relay-artifact",
        action="store_true",
        help="只提取task.pkl/all_tasks.pkl，不保存包含完整参数的Relay artifact",
    )
    args = parser.parse_args()
    if args.onnx_model and args.onnx_manifest:
        parser.error("--onnx-model and --onnx-manifest are mutually exclusive")
    
    # 注册数据路径并解析目标平台
    register_data_path(
        args.target,
        hardware_name=args.hardware_name,
        network_info_folder=args.network_info_dir,
    )
    args.target = tvm.target.Target(args.target)

    # 导入网络信息文件夹路径并确保其存在
    from common import NETWORK_INFO_FOLDER
    assert(NETWORK_INFO_FOLDER is not None)
    os.makedirs(NETWORK_INFO_FOLDER, exist_ok=True)

    # 构建网络键列表
    network_keys = [] if args.only_workloads else build_network_keys()
    for workload in args.workload or []:
        name, shape_str = workload.split(":", 1)
        shape = [int(dim) for dim in shape_str.split(",") if dim]
        network_keys.append((name, shape))

    hardware_params = get_hardware_params(args.target)
    
    # 为所有网络转储信息
    for key in tqdm(network_keys):
        dump_network(
            key,
            args.target,
            hardware_params,
            overwrite_relay=args.overwrite_relay,
            overwrite_task_info=args.overwrite_task_info,
            skip_relay_artifact=args.skip_relay_artifact,
        )
        gc.collect()  # 回收内存

    onnx_entries = []
    if args.onnx_model:
        input_dtypes = json.loads(args.input_dtypes) if args.input_dtypes else None
        if input_dtypes is not None and not isinstance(input_dtypes, dict):
            raise ValueError("--input-dtypes must be a JSON object")
        onnx_entries.append(
            {
                "onnx_path": args.onnx_model,
                "model_name": (
                    args.model_name or os.path.splitext(os.path.basename(args.onnx_model))[0]
                ),
                "input_shapes": args.input_shapes,
                "input_dtypes": input_dtypes,
                "fix_mask_where_dtype": args.fix_mask_where_dtype,
            }
        )

    if args.onnx_manifest:
        manifest_path = os.path.abspath(args.onnx_manifest)
        with open(manifest_path, "r", encoding="utf-8") as file:
            manifest = json.load(file)
        if manifest.get("format") != "llm_compiler.onnx_export_manifest":
            raise ValueError(f"Unsupported ONNX manifest format: {manifest.get('format')!r}")
        if manifest.get("version") != 1:
            raise ValueError(f"Unsupported ONNX manifest version: {manifest.get('version')!r}")
        if manifest.get("dynamic_axes"):
            raise ValueError(
                "Dynamic-axis ONNX manifests are not supported by this fixed-shape flow"
            )
        exports = manifest.get("exports")
        if not isinstance(exports, list) or not exports:
            raise ValueError("ONNX manifest must contain a non-empty exports list")
        manifest_dir = os.path.dirname(manifest_path)
        for index, entry in enumerate(exports):
            if not isinstance(entry, dict):
                raise ValueError(f"Manifest export {index} must be an object")
            missing = [
                key
                for key in ("onnx_path", "model_name", "input_shapes", "input_dtypes")
                if key not in entry
            ]
            if missing:
                raise ValueError(f"Manifest export {index} is missing fields: {missing}")
            onnx_path = entry["onnx_path"]
            if not os.path.isabs(onnx_path):
                onnx_path = os.path.join(manifest_dir, onnx_path)
            onnx_entries.append(
                {
                    "onnx_path": os.path.abspath(onnx_path),
                    "model_name": str(entry["model_name"]),
                    "input_shapes": entry["input_shapes"],
                    "input_dtypes": entry["input_dtypes"],
                    "fix_mask_where_dtype": bool(
                        entry.get(
                            "fix_mask_where_dtype",
                            entry.get("fix_fp16_mask_where", True),
                        )
                    ),
                }
            )

    for entry in tqdm(onnx_entries, desc="ONNX networks"):
        if not os.path.isfile(entry["onnx_path"]):
            raise FileNotFoundError(f"ONNX model not found: {entry['onnx_path']}")
        input_shapes = entry["input_shapes"]
        if isinstance(input_shapes, dict):
            input_shapes = json.dumps(input_shapes)
        model_shape_key = parse_shape_dict(input_shapes)
        if not model_shape_key:
            raise ValueError(f"Missing input shapes for ONNX model: {entry['onnx_path']}")
        network_key = (f"onnx_{entry['model_name']}", sorted(model_shape_key.items()))
        task_info_filename = get_task_info_filename(network_key, args.target)
        relay_ir_filename = get_relay_ir_filename(args.target, network_key)
        task_is_current = os.path.exists(task_info_filename) and not args.overwrite_task_info
        relay_is_current = args.skip_relay_artifact or (
            os.path.exists(relay_ir_filename) and not args.overwrite_relay
        )
        if task_is_current and relay_is_current:
            print(f"Skip existing ONNX task: {network_key}")
            continue
        input_dtypes = entry["input_dtypes"]
        if input_dtypes is not None and not isinstance(input_dtypes, dict):
            raise ValueError(f"input_dtypes must be an object for {entry['onnx_path']}")
        mod, params, inputs = load_onnx_network(
            entry["onnx_path"],
            input_shapes,
            args.input_dtype,
            input_dtypes,
            fix_mask_where_dtype=entry["fix_mask_where_dtype"],
        )
        dump_relay_network(
            network_key,
            args.target,
            hardware_params,
            mod,
            params,
            inputs,
            overwrite_relay=args.overwrite_relay,
            overwrite_task_info=args.overwrite_task_info,
            skip_relay_artifact=args.skip_relay_artifact,
        )
        gc.collect()

    # 生成包含所有任务的索引表
    tasks = get_all_tasks()
    # 按目标类型、计算图浮点操作数和工作负载键排序
    tasks.sort(key=lambda x: (str(x.target.kind), x.compute_dag.flop_ct, x.workload_key))
    # 保存排序后的任务列表
    pickle.dump(tasks, open(f"{NETWORK_INFO_FOLDER}/all_tasks.pkl", "wb"))
