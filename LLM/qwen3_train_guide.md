# Qwen3-0.6B 继续预训练指南（基于 `4090_gen_qwen` 新数据集）

## 1. 这份指南解决什么问题

这份文档面向当前已经重建完成的新数据集：

- 数据集路径：`/home/qsy/workspace/gen_data/4090_gen_qwen`
- 基础模型路径：`/home/qsy/huggingface/model/Qwen3-0.6B`
- 训练主脚本：`train_qwen3_clm.py`
- 启动脚本：`run_train_qwen3_clm.py`

目标是让 Qwen3-0.6B 学会在 `PPT` 之后补全**结构合法的 TVM 调度后缀**，并且先通过 smoke test 验证训练链路可用，再开展正式训练。

## 2. 当前数据集状态

当前数据集已经按新的规则重建完成，关键特征如下：

- `max_length=1024`
- `min_suffix_tokens=16`
- 训练集与验证集按**源 workload 文件**切分，而不是按合并后的前 `5%` 切分
- `train_file_count=2046`
- `validation_file_count=108`
- `train` 样本数：`2156881`
- `validation` 样本数：`111627`

抽样验证结果：

- 在 `train` 随机抽样 `5000` 条：`missing_ppt=0`，`short_suffix_lt16=0`
- 在 `validation` 随机抽样 `5000` 条：`missing_ppt=0`，`short_suffix_lt16=0`

当前这份数据集适合做：

- 学习 `PPT` 之后的合法结构补全
- 训练模型先输出更少的 `invalid` 组合

当前这份数据集**不适合直接做性能偏好训练**，原因是：

- 原始 `to_measure_programs/4090` 里的记录基本没有可用 latency
- `for_gen_stats.json` 里 `records_without_latency` 等于样本总数

也就是说，当前训练重点是：

- 先学会生成**合法**的调度后缀

而不是：

- 学会区分“更快”和“更慢”的调度

## 3. 必要路径

默认使用下面这些路径：

- 代码目录：`/home/qsy/workspace/complier/llm_compiler/LLM`
- Conda Python：`/home/qsy/anaconda3/envs/tlm/bin/python`
- Torchrun：`/home/qsy/anaconda3/envs/tlm/bin/torchrun`
- 基础模型：`/home/qsy/huggingface/model/Qwen3-0.6B`
- 数据集：`/home/qsy/workspace/gen_data/4090_gen_qwen`
- Stage1 输出：`/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1`
- Stage2 输出：`/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage2`

## 4. 训练脚本现在的关键行为

### 4.1 `train_qwen3_clm.py`

这个脚本已经做了以下适配：

- 默认数据集路径改为 `4090_gen_qwen`
- 默认 `min_suffix_tokens=16`
- 默认 `fix_mistral_regex=True`
- 支持 `sample_seed`
- 支持 `subsample_before_mask=True`
- 当你传 `max_train_samples` / `max_eval_samples` 时，会**先抽样，再做 `PPT` masking**

这一点非常重要，因为它让 smoke test 不会先扫描完整个 `215` 万训练样本。

### 4.2 `run_train_qwen3_clm.py`

这个脚本现在支持：

- `TRAIN_STAGE=stage1|stage2`
- `SMOKE_TEST=1`
- `CUDA_VISIBLE_DEVICES`
- `NPROC_PER_NODE`
- `MODEL_NAME_OR_PATH`
- `TOKENIZER_NAME`
- `DATASET_NAME`
- `OUTPUT_DIR`
- `LEARNING_RATE`
- `NUM_TRAIN_EPOCHS`
- `PER_DEVICE_TRAIN_BATCH_SIZE`
- `PER_DEVICE_EVAL_BATCH_SIZE`
- `GRADIENT_ACCUMULATION_STEPS`
- `MAX_STEPS`
- `MAX_TRAIN_SAMPLES`
- `MAX_EVAL_SAMPLES`
- `MIN_SUFFIX_TOKENS`
- `LOG_FILE`
- `SESSION_NAME`
- `DELETE_LOG_IF_EXISTS`

并且：

- `SMOKE_TEST=1` 时会自动切到单卡
- 默认使用 `tmux` 后台启动
- 日志通过 `tee` 写入 `LOG_FILE`

## 5. 训练前检查

先确认环境和数据集都正常：

```bash
cd /home/qsy/workspace/complier/llm_compiler/LLM

/home/qsy/anaconda3/envs/tlm/bin/python - <<'PY'
import transformers, accelerate, datasets, numpy, pyarrow
print('transformers', transformers.__version__)
print('accelerate', accelerate.__version__)
print('datasets', datasets.__version__)
print('numpy', numpy.__version__)
print('pyarrow', pyarrow.__version__)
PY
```

```bash
ls -lh /home/qsy/workspace/gen_data/4090_gen_qwen
ls -lh /home/qsy/workspace/gen_data/4090_gen_qwen/train | head
ls -lh /home/qsy/workspace/gen_data/4090_gen_qwen/validation | head
```

```bash
/home/qsy/anaconda3/envs/tlm/bin/python - <<'PY'
from datasets import load_from_disk
ds = load_from_disk('/home/qsy/workspace/gen_data/4090_gen_qwen')
print('train', len(ds['train']))
print('validation', len(ds['validation']))
PY
```

你应该能看到：

- 顶层有 `dataset_dict.json`、`train/`、`validation/`、`for_gen_stats.json`
- `train/` 和 `validation/` 里有 `data-*.arrow`、`dataset_info.json`、`state.json`

## 6. 最小可行 smoke test

建议先跑 smoke test，再跑正式训练。

### 6.1 直接调用 `train_qwen3_clm.py`

这条命令已经在当前机器上验证可以完整走通：

- 数据加载
- `PPT` masking
- train
- eval
- checkpoint 保存

```bash
cd /home/qsy/workspace/complier/llm_compiler/LLM

CUDA_VISIBLE_DEVICES=0 /home/qsy/anaconda3/envs/tlm/bin/python train_qwen3_clm.py \
  --do_train \
  --do_eval \
  --model_name_or_path /home/qsy/huggingface/model/Qwen3-0.6B \
  --tokenizer_name /home/qsy/huggingface/model/Qwen3-0.6B \
  --dataset_name /home/qsy/workspace/gen_data/4090_gen_qwen \
  --output_dir /home/qsy/huggingface/model/Qwen3-0.6B-smoke-4090-gen-fixed-v2 \
  --overwrite_output_dir \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --logging_steps 5 \
  --num_train_epochs 1 \
  --max_steps 10 \
  --max_train_samples 1024 \
  --max_eval_samples 128 \
  --sample_seed 42 \
  --subsample_before_mask True \
  --remove_unused_columns False \
  --learning_rate 1e-5 \
  --warmup_steps 2 \
  --eval_strategy steps \
  --eval_steps 5 \
  --save_strategy steps \
  --save_steps 10 \
  --save_total_limit 1 \
  --dataloader_num_workers 0 \
  --bf16 True \
  --gradient_checkpointing True \
  --mask_prefix_before_ppt True \
  --drop_samples_without_ppt True \
  --min_suffix_tokens 16 \
  --report_to none
```

这条 smoke 的预期特征：

- 训练前会先打印 `Subsampled split=train before_mask=... after_subsample=1024`
- `PPT mask split=train before=1024 after=1024 dropped=0`
- `PPT mask split=validation before=128 after=128 dropped=0`
- 最终会保存到 `Qwen3-0.6B-smoke-4090-gen-fixed-v2`

### 6.2 通过 `run_train_qwen3_clm.py` 跑 smoke

如果你想验证后台启动脚本本身，也可以直接跑：

```bash
cd /home/qsy/workspace/complier/llm_compiler/LLM

SMOKE_TEST=1 \
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
OUTPUT_DIR=/home/qsy/huggingface/model/Qwen3-0.6B-smoke-helper-test \
LOG_FILE=run_train_smoke.log \
SESSION_NAME=run_train_qwen3_smoke \
DELETE_LOG_IF_EXISTS=1 \
/home/qsy/anaconda3/envs/tlm/bin/python run_train_qwen3_clm.py
```

查看日志：

```bash
tail -f /home/qsy/workspace/complier/llm_compiler/LLM/run_train_smoke.log
```

这条 helper smoke 在当前机器上也已经验证通过。

## 7. 推荐正式训练流程

推荐顺序：

1. 先跑 direct smoke 或 helper smoke
2. 再跑正式 `stage1`
3. 如需继续低学习率稳态训练，再跑 `stage2`
4. 训练结束后再用 `gen_state.py` 做生成有效性验证

## 8. Stage1 正式训练

Stage1 的目标是：

- 基于新数据集学习 `PPT` 之后的**结构合法后缀**

推荐命令：

```bash
cd /home/qsy/workspace/complier/llm_compiler/LLM

TRAIN_STAGE=stage1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
MODEL_NAME_OR_PATH=/home/qsy/huggingface/model/Qwen3-0.6B \
TOKENIZER_NAME=/home/qsy/huggingface/model/Qwen3-0.6B \
DATASET_NAME=/home/qsy/workspace/gen_data/4090_gen_qwen \
OUTPUT_DIR=/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1 \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
PER_DEVICE_EVAL_BATCH_SIZE=2 \
GRADIENT_ACCUMULATION_STEPS=4 \
LEARNING_RATE=1e-5 \
NUM_TRAIN_EPOCHS=1 \
MIN_SUFFIX_TOKENS=16 \
LOGGING_STEPS=100 \
EVAL_STEPS=2000 \
SAVE_STEPS=2000 \
SAVE_TOTAL_LIMIT=3 \
LOG_FILE=run_train_stage1.log \
SESSION_NAME=qwen3_stage1_4090 \
DELETE_LOG_IF_EXISTS=1 \
/home/qsy/anaconda3/envs/tlm/bin/python run_train_qwen3_clm.py
```

查看日志：

```bash
tail -f /home/qsy/workspace/complier/llm_compiler/LLM/run_train_stage1.log
```

Stage1 默认会：

- 开启 `mask_prefix_before_ppt=True`
- 开启 `drop_samples_without_ppt=True`
- 使用 `min_suffix_tokens=16`
- 使用 `load_best_model_at_end=True`
- 以 `eval_accuracy` 作为最佳模型指标

## 9. Stage2 继续训练

Stage2 的目标是：

- 在 Stage1 基础上做更低学习率的继续训练
- 进一步稳定结构补全能力

注意：

- 当前数据集没有有效 latency
- 所以 Stage2 仍然只是结构对齐的继续训练
- 它不是“性能偏好精调”

推荐命令：

```bash
cd /home/qsy/workspace/complier/llm_compiler/LLM

TRAIN_STAGE=stage2 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
MODEL_NAME_OR_PATH=/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1 \
TOKENIZER_NAME=/home/qsy/huggingface/model/Qwen3-0.6B \
DATASET_NAME=/home/qsy/workspace/gen_data/4090_gen_qwen \
OUTPUT_DIR=/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage2 \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
PER_DEVICE_EVAL_BATCH_SIZE=2 \
GRADIENT_ACCUMULATION_STEPS=4 \
LEARNING_RATE=5e-6 \
NUM_TRAIN_EPOCHS=1 \
MIN_SUFFIX_TOKENS=16 \
LOGGING_STEPS=100 \
EVAL_STEPS=2000 \
SAVE_STEPS=2000 \
SAVE_TOTAL_LIMIT=3 \
LOG_FILE=run_train_stage2.log \
SESSION_NAME=qwen3_stage2_4090 \
DELETE_LOG_IF_EXISTS=1 \
/home/qsy/anaconda3/envs/tlm/bin/python run_train_qwen3_clm.py
```

如果 `MODEL_NAME_OR_PATH` 指向的 Stage1 目录不存在，启动脚本会按下面顺序回退：

1. `Qwen3-0.6B-4090-struct-stage1`
2. `Qwen3-0.6B-fintuned`
3. `Qwen3-0.6B`

但正式训练时建议始终显式传入你真正想用的 checkpoint 路径。

## 10. 常用参数解释

下面这些参数最重要：

| 参数 | 作用 | 推荐值 |
| --- | --- | --- |
| `--dataset_name` | tokenized 数据集路径 | `/home/qsy/workspace/gen_data/4090_gen_qwen` |
| `--mask_prefix_before_ppt` | 只对 `PPT` 后缀算 loss | `True` |
| `--drop_samples_without_ppt` | 丢弃找不到 `PPT` 的样本 | `True` |
| `--min_suffix_tokens` | 要保留的最小后缀监督长度 | `16` |
| `--subsample_before_mask` | 先抽样，再做 masking，便于 smoke | `True` |
| `--max_train_samples` | smoke 时限制训练样本数 | `1024` 或 `2048` |
| `--max_eval_samples` | smoke 时限制验证样本数 | `128` 或 `256` |
| `--per_device_train_batch_size` | 单卡训练 batch | smoke 用 `1`，正式训练用 `2` 起步 |
| `--gradient_accumulation_steps` | 梯度累积 | `4` |
| `--learning_rate` | 学习率 | Stage1 用 `1e-5`，Stage2 用 `5e-6` |
| `--bf16` | 使用 bfloat16 | `True` |
| `--gradient_checkpointing` | 降低显存占用 | `True` |
| `--eval_steps` | 评估间隔 | smoke `5`/`10`，正式训练 `2000` |
| `--save_steps` | checkpoint 保存间隔 | smoke `10`/`20`，正式训练 `2000` |

## 11. 启动脚本环境变量解释

下面这些环境变量是 `run_train_qwen3_clm.py` 最常用的：

| 环境变量 | 作用 | 备注 |
| --- | --- | --- |
| `TRAIN_STAGE` | 选择 `stage1` 或 `stage2` | 默认 `stage1` |
| `SMOKE_TEST` | 启用小规模 smoke | `1` 时自动单卡 |
| `CUDA_VISIBLE_DEVICES` | 指定可见 GPU | 例如 `0,1,2,3` |
| `NPROC_PER_NODE` | `torchrun` 进程数 | 必须小于等于可见 GPU 数 |
| `MODEL_NAME_OR_PATH` | 模型初始化路径 | 默认按 stage 决定 |
| `TOKENIZER_NAME` | tokenizer 路径 | 默认基础模型路径 |
| `DATASET_NAME` | 数据集路径 | 默认 `4090_gen_qwen` |
| `OUTPUT_DIR` | 模型输出目录 | 建议显式传 |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | 单卡训练 batch | 正式训练推荐从 `2` 开始 |
| `GRADIENT_ACCUMULATION_STEPS` | 梯度累积 | 默认 `4` |
| `MAX_STEPS` | 限制训练步数 | smoke 常用 |
| `MAX_TRAIN_SAMPLES` | 限制训练样本数 | smoke 常用 |
| `MAX_EVAL_SAMPLES` | 限制验证样本数 | smoke 常用 |
| `LOG_FILE` | 日志文件路径 | 默认在当前目录 |
| `SESSION_NAME` | tmux session 名称 | 建议显式传，避免冲突 |
| `DELETE_LOG_IF_EXISTS` | 若日志已存在是否先删掉 | 设为 `1` 更省心 |

## 12. 训练产物会写到哪里

训练完成后你通常会看到：

- 输出目录下有 `checkpoint-*`
- 输出目录下有 `config.json`
- 输出目录下有 `model.safetensors`
- 输出目录下有 `tokenizer_config.json`
- 输出目录下有 `trainer_state.json`
- 输出目录下有 `train_results.json`
- 如果启用了 eval，还会有 `eval_results.json`

另外，首次在新数据集上训练时，数据目录下会出现：

- `train/cache-*.arrow`
- `validation/cache-*.arrow`

这是 `PPT` mask 和抽样索引的缓存，属于正常现象。

## 13. 常见问题

### 13.1 训练一开始很慢

如果你看到一开始在做：

- `Caching indices mapping`
- `Applying PPT suffix mask`

这是正常的，因为脚本在构建缓存。第二次跑同样配置通常会更快。

### 13.2 `invalid device ordinal`

说明：

- `NPROC_PER_NODE` 大于 `CUDA_VISIBLE_DEVICES` 中实际可见的 GPU 数

修复方式：

- 例如只用 `0,1,2` 三张卡时，`NPROC_PER_NODE=3`

### 13.3 显存不够

优先按这个顺序调小：

1. `PER_DEVICE_TRAIN_BATCH_SIZE`
2. `PER_DEVICE_EVAL_BATCH_SIZE`
3. 提高 `GRADIENT_ACCUMULATION_STEPS`
4. 减少 `NPROC_PER_NODE`

### 13.4 smoke 还是太慢

确保你用了：

- `--subsample_before_mask True`
- `--max_train_samples`
- `--max_eval_samples`

或者直接用：

- `SMOKE_TEST=1`

### 13.5 Stage2 没有真正学到“更优调度”

这是当前数据集本身决定的：

- 现在的数据几乎没有有效 latency
- 因此当前训练只能先学“合法结构”

如果要做性能导向训练，需要后续接入带真实测量结果的数据集。

## 14. 训练后下一步

训练完成后，建议不要只看 `eval_accuracy`，还要做一轮真实生成验证：

1. 用新 checkpoint 跑 `gen_state.py`
2. 统计生成结果是否能被 TVM parser 接受
3. 统计 build 成功率
4. 统计 fallback 到 sketch 的比例

只有这一步也变好，才说明继续预训练真的对 `gen_state.py` 有帮助。

## 15. 附：如果需要重建数据集

如果以后要重新生成训练数据集，当前正确命令是：

```bash
cd /home/qsy/workspace/complier/llm_compiler/LLM

/home/qsy/anaconda3/envs/tlm/bin/python make_dataset.py \
  --for_type for_gen \
  --target nvidia/geforce-rtx-4090 \
  --dataset_path /home/qsy/workspace/dataset/to_measure_programs/4090 \
  --tokenizer_path /home/qsy/huggingface/model/Qwen3-0.6B \
  --save_path /home/qsy/workspace/gen_data/4090_gen_qwen \
  --max_length 1024 \
  --valid_percentage 5 \
  --min_suffix_tokens 16
```

重建完成后可以检查：

```bash
cat /home/qsy/workspace/gen_data/4090_gen_qwen/for_gen_stats.json
```
