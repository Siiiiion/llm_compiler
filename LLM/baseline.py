# 可以自己预设 workloads 

# python baseline.py --workload=resnet_50 \
# --input-shape=\[1,128\] \
# --target=nvidia/geforce-rtx-4090 \
# --backend=graph




import argparse
import json
import os
from distutils.util import strtobool

import tvm
from tvm import auto_scheduler
from tvm import meta_schedule as ms
from tvm import relay
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.meta_schedule.testing.tune_utils import create_timer, generate_input_data
from tvm.meta_schedule.utils import cpu_count
from tvm.support import describe
import numpy as np


def _default_output_log_path(parsed):
    shape = "x".join(str(dim) for dim in parsed.input_shape)
    filename = f"{parsed.workload}_{shape}_{parsed.target.kind.name}_{parsed.backend}.json"
    return os.path.join("logs", "auto_scheduler", filename)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--input-shape",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=0,
        help="AutoScheduler measure trials. Use 0 to compile/evaluate from an existing log.",
    )
    # parser.add_argument(
    #     "--rpc-host",
    #     type=str,
    #     required=True,
    # )
    # parser.add_argument(
    #     "--rpc-port",
    #     type=int,
    #     required=True,
    # )
    # parser.add_argument(
    #     "--rpc-key",
    #     type=str,
    #     required=True,
    # )
    # parser.add_argument(
    #     "--work-dir",
    #     type=str,
    #     required=True,
    # )
    parser.add_argument(
        "--output-log",
        type=str,
        default=None,
        help="Path for AutoScheduler tuning records.",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--number",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--min-repeat-ms",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--adaptive-training",
        type=lambda x: bool(strtobool(x)),
        help="example: True / False",
        default=True,
    )
    # parser.add_argument(
    #     "--cpu-flush",
    #     type=lambda x: bool(strtobool(x)),
    #     help="example: True / False",
    #     required=True,
    # )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["graph", "vm"],
        help="example: graph / vm",
        required=True,
    )
    parsed = parser.parse_args()
    if parsed.num_trials < 0:
        parser.error("--num-trials must be >= 0")
    parsed.target = tvm.target.Target(parsed.target)
    parsed.input_shape = json.loads(parsed.input_shape)
    if parsed.output_log is None:
        parsed.output_log = _default_output_log_path(parsed)
    # parsed.rpc_config = ms.runner.RPCConfig(
    #     tracker_host=parsed.rpc_host,
    #     tracker_port=parsed.rpc_port,
    #     tracker_key=parsed.rpc_key,
    #     session_timeout_sec=600,
    # )
    return parsed


ARGS = _parse_args()


def main():
    log_file = os.path.abspath(ARGS.output_log)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if ARGS.num_trials == 0 and not os.path.exists(log_file):
        raise FileNotFoundError(
            f"AutoScheduler log not found: {log_file}. "
            "Set --num-trials > 0 to tune first, or pass --output-log to an existing log."
        )
    # runner = auto_scheduler.RPCRunner(
    #     key=ARGS.rpc_key,
    #     host=ARGS.rpc_host,
    #     port=ARGS.rpc_port,
    #     n_parallel=cpu_count(logical=True),
    #     number=ARGS.number,
    #     repeat=ARGS.repeat,
    #     min_repeat_ms=ARGS.min_repeat_ms,
    #     enable_cpu_cache_flush=ARGS.cpu_flush,
    #     timeout=ARGS.rpc_config.session_timeout_sec,
    # )

    if ARGS.target.kind.name == "llvm":
        enable_cpu_cache_flush = True
        hardware_params = auto_scheduler.HardwareParams(
            # num_cores=int(ARGS.target.attrs["num-cores"]),
            target=ARGS.target,
        )
    elif ARGS.target.kind.name == "cuda":
        enable_cpu_cache_flush = False
        hardware_params = auto_scheduler.HardwareParams(
            num_cores=-1,
            vector_unit_bytes=16,
            cache_line_bytes=64,
            max_shared_memory_per_block=int(ARGS.target.attrs["max_shared_memory_per_block"]),
            max_threads_per_block=int(ARGS.target.attrs["max_threads_per_block"]),
            # The value `max_local_memory_per_block` is not used in AutoScheduler,
            # but is required by the API.
            max_local_memory_per_block=12345678,
            max_vthread_extent=8,
            warp_size=32,
        )
    else:
        raise NotImplementedError(f"Unsupported target {ARGS.target}")
    runner = auto_scheduler.LocalRunner(
        number=ARGS.number,
        repeat=ARGS.repeat,
        min_repeat_ms=ARGS.min_repeat_ms,
        enable_cpu_cache_flush=enable_cpu_cache_flush,
        timeout=5,
    )

    # describe()
    print(f"Workload: {ARGS.workload}")
    print(f"AutoScheduler log: {log_file}")
    mod, params, (input_name, input_shape, input_dtype) = get_network(
        ARGS.workload,
        ARGS.input_shape,
        layout=ARGS.layout,
        cache_dir=ARGS.cache_dir,
    )
    input_info = [
        {
            "name": input_name,
            "shape": input_shape,
            "dtype": input_dtype,
        },
    ]
    input_data = {
        item["name"]: generate_input_data(item["shape"], item["dtype"]) for item in input_info
    }
    # for item in input_info:
    #     print(f"  input_name : {item['name']}")
    #     print(f"  input_shape: {item['shape']}")
    #     print(f"  input_dtype: {item['dtype']}")

    with ms.Profiler() as profiler:
        with ms.Profiler.timeit("TaskExtraction"):
            tasks, task_weights = auto_scheduler.extract_tasks(
                mod["main"],
                params,
                target=ARGS.target,
                hardware_params=hardware_params,
            )
            for idx, (task, task_weight) in enumerate(zip(tasks, task_weights)):
                print(
                    f"==== Task {idx}: {task.desc} "
                    f"(weight {task_weight} key: {task.workload_key}) ====="
                )
                print(task.compute_dag)

        with ms.Profiler.timeit("Tuning"):
            if ARGS.num_trials > 0:
                tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
                tuner.tune(
                    auto_scheduler.TuningOptions(
                        num_measure_trials=ARGS.num_trials,
                        runner=runner,
                        measure_callbacks=[
                            auto_scheduler.RecordToFile(log_file),
                        ],
                    ),
                    adaptive_training=ARGS.adaptive_training,
                )

        relay_build = {"graph": relay.build, "vm": relay.vm.compile}[ARGS.backend]
        with ms.Profiler.timeit("PostTuningCompilation"):
            with auto_scheduler.ApplyHistoryBest(log_file):
                with tvm.transform.PassContext(
                    opt_level=3,
                    config={"relay.backend.use_auto_scheduler": True},
                ):
                    lib = relay_build(
                        mod,
                        target=ARGS.target,
                        params=params,
                    )
    print("Tuning Time:")
    print(profiler.table())

    from tvm.contrib import graph_executor
    dev = tvm.device(str(ARGS.target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(input_dtype))
    module.set_input(input_name, data_tvm)
    print("Evaluate inference time cost...")
    print(module.benchmark(dev, repeat=10, min_repeat_ms=500))


if __name__ == "__main__":
    main()
