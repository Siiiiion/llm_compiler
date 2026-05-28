# 指定 Workload 的生成与测试方法

本文记录如何复用原来的 `only_bert` 测试流程，对任意 hold-out workload 单独生成、测量并用于端到端评估。

## 1. 支持的 Workload

当前 `common.py` 中的 hold-out workload 包括：


| workload         | input shape        |
| ---------------- | ------------------ |
| `resnet_50`      | `[1,3,224,224]`    |
| `mobilenet_v2`   | `[1,3,224,224]`    |
| `resnext_50`     | `[1,3,224,224]`    |
| `bert_base`      | `[1,128]`          |
| `bert_tiny`      | `[1,128]`          |
| `densenet_121`   | `[8,3,256,256]`    |
| `bert_large`     | `[4,256]`          |
| `wide_resnet_50` | `[8,3,256,256]`    |
| `resnet3d_18`    | `[4,3,144,144,16]` |
| `dcgan`          | `[8,3,64,64]`      |

新增参数：

```shell
--eval_workloads=<workload_name>
```

也支持一次指定多个，用逗号分隔：

```shell
--eval_workloads=resnet_50,mobilenet_v2
```

如果不传 `--eval_workloads`，`for_gen_eval_sketch` 仍会使用全部 hold-out workload。
旧的 `for_gen_eval_sketch_only_bert` 仍然保留，默认等价于只选择 `bert_base`。

## 2. 回归原 only_bert 流程

生成 `bert_base` 的 prompt：

```shell
python make_dataset.py \
  --for_type=for_gen_eval_sketch_only_bert \
  --target=nvidia/nvidia-v100 \
  --dataset_path=dataset/to_measure_programs/v100 \
  --tokenizer_path=gen_data/gen_tokenizer_v100 \
  --save_path=gen_data/v100_gen_eval_only_bert \
  --keep_cnt=64
```

也可以使用新的通用写法：

```shell
python make_dataset.py \
  --for_type=for_gen_eval_sketch \
  --target=nvidia/nvidia-v100 \
  --dataset_path=dataset/to_measure_programs/v100 \
  --tokenizer_path=gen_data/gen_tokenizer_v100 \
  --save_path=gen_data/v100_gen_eval_only_bert \
  --keep_cnt=64 \
  --eval_workloads=bert_base
```

两种方式选出的 workload 应该一致。

## 3. 单独测试其他 Workload

下面以 `resnet_50` 为例。

### 3.1 生成 prompt

```shell
python make_dataset.py \
  --for_type=for_gen_eval_sketch \
  --target=nvidia/nvidia-v100 \
  --dataset_path=dataset/to_measure_programs/v100 \
  --tokenizer_path=gen_data/gen_tokenizer_v100 \
  --save_path=gen_data/v100_gen_eval_only_resnet50 \
  --keep_cnt=64 \
  --eval_workloads=resnet_50
```

### 3.2 用模型生成 tensor programs

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python gen_state.py \
  --model_name_or_path=/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1 \
  --sketch_path=/home/qsy/workspace/gen_data/4090_gen_eval_only_resnet50/0_merge.json \
  --save_path=/home/qsy/workspace/gen_data/4090_gen_eval_only_resnet50/gen_eval.json \
  --allow_repeat=True \
  --target=nvidia/geforce-rtx-4090 \
  --keep_cnt=32 \
  --is_build=True
```

`--is_build=True` 会在生成阶段先用 TVM LocalBuilder 过滤掉无法构建的 state，避免后续
`tune_relay.py` 因某些 workload 没有有效历史记录而回落到无 thread binding 的占位 schedule。

### 3.3 测量生成程序的 latency

测量时应独占 GPU，避免 latency 抖动。

```shell
CUDA_VISIBLE_DEVICES=3 python measure_programs.py \
  --batch-size=64 \
  --target=nvidia/nvidia-v100 \
  --to-measure-path=gen_data/v100_gen_eval_only_resnet50/gen_eval.json \
  --measured-path=measured_only_resnet50.json
```

### 3.4 用最佳记录编译并测端到端 latency

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 TLM_LOG_FILE=measured_only_resnet50.json python tune_relay.py \
  --workload=resnet_50 \
  --input-shape=\[1,3,224,224\] \
  --target=nvidia/nvidia-v100 \
  --backend=graph
```

## 4. 4090 示例

如果目标是本机 4090，命令中的路径和 target 可按当前数据目录调整。例如：

```shell
python make_dataset.py \
  --for_type=for_gen_eval_sketch \
  --target=nvidia/geforce-rtx-4090 \
  --dataset_path=/home/qsy/workspace/dataset/to_measure_programs/4090 \
  --tokenizer_path=/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1 \
  --save_path=/home/qsy/workspace/gen_data/4090_gen_eval_only_resnet50 \
  --keep_cnt=64 \
  --eval_workloads=resnet_50
```

后续生成、测量和 `tune_relay.py` 只需要把路径、target、模型 checkpoint 改成对应的 4090 版本。

## 5. 常用命令模板

将 `<workload>` 和 `<shape>` 替换成上表中的值：

```shell
python make_dataset.py \
  --for_type=for_gen_eval_sketch \
  --target=<target> \
  --dataset_path=<to_measure_programs_dir> \
  --tokenizer_path=<tokenizer_dir> \
  --save_path=<save_dir> \
  --keep_cnt=64 \
  --eval_workloads=<workload>
```

```shell
CUDA_VISIBLE_DEVICES=<gpu_id> python gen_state.py \
  --model_name_or_path=<clm_checkpoint> \
  --sketch_path=<save_dir>/0_merge.json \
  --save_path=<save_dir>/gen_eval.json \
  --allow_repeat=True \
  --target=<target> \
  --keep_cnt=32 \
  --is_build=True
```

```shell
CUDA_VISIBLE_DEVICES=<gpu_id> python measure_programs.py \
  --batch-size=64 \
  --target=<target> \
  --to-measure-path=<save_dir>/gen_eval.json \
  --measured-path=measured_only_<workload>.json
```

```shell
CUDA_VISIBLE_DEVICES=<gpu_id> TLM_LOG_FILE=measured_only_<workload>.json python tune_relay.py \
  --workload=<workload> \
  --input-shape=<shape> \
  --target=<target> \
  --backend=graph
```

## 6. 验证点

1. `make_dataset.py` 输出的 `0_merge.json` 应只包含指定 workload 对应的记录。
2. `gen_state.py` 输出的 `gen_eval.json` 不应为空。
3. `measure_programs.py` 输出的 `measured_only_<workload>.json` 中每个 extracted task 都应有有效 latency，`r[0]` 不应全部为 `0` 或 `1e+10`。
4. `tune_relay.py` 应能读取 `TLM_LOG_FILE`，通过 history coverage 检查，完成 Relay 编译并输出端到端 latency。
