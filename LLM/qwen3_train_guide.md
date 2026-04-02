# Qwen3-0.6B 训练说明

## 适用脚本

- 训练主脚本: `train_qwen3_clm.py`
- 启动脚本: `run_train_qwen3_clm.py`

训练数据默认使用:

- `/data/qsy/workspace/gen_data/4090_gen_qwen`

模型输入输出路径:

- base model: `/data/qsy/huggingface/model/Qwen3-0.6B`
- finetuned output: `/data/qsy/huggingface/model/Qwen3-0.6B-fintuned`

## 一键启动（推荐）

在 LLM 目录执行:

```bash
cd /data/qsy/workspace/complier/llm_compiler/LLM
/data/qsy/anaconda3/envs/tlm/bin/python run_train_qwen3_clm.py
```

脚本会使用 tmux 后台运行并把日志写入当前目录的:

- `run_train_qwen3_clm.py.log`

说明：启动脚本已使用 `torchrun --nproc_per_node=4`（DDP）。
这是为了规避单进程多卡（DataParallel）在本环境下的段错误问题。

查看训练日志:

```bash
tail -f /data/qsy/workspace/complier/llm_compiler/LLM/run_train_qwen3_clm.py.log
```

## 直接命令行启动（可自定义参数）

```bash
cd /data/qsy/workspace/complier/llm_compiler/LLM

CUDA_VISIBLE_DEVICES=0,1,2,3 /data/qsy/anaconda3/envs/tlm/bin/torchrun --nproc_per_node=4 --master_port=29531 train_qwen3_clm.py \
  --do_train \
  --model_name_or_path=/data/qsy/huggingface/model/Qwen3-0.6B \
  --tokenizer_name=/data/qsy/huggingface/model/Qwen3-0.6B \
  --dataset_name=/data/qsy/workspace/gen_data/4090_gen_qwen \
  --output_dir=/data/qsy/huggingface/model/Qwen3-0.6B-fintuned \
  --per_device_train_batch_size=5 \
  --num_train_epochs=3 \
  --learning_rate=5e-5 \
  --logging_steps=100 \
  --save_steps=4000 \
  --save_total_limit=3 \
  --remove_unused_columns=False \
  --dataloader_num_workers=4 \
  --bf16=True \
  --gradient_checkpointing=True \
  --ddp_find_unused_parameters=False \
  --report_to=none \
  --overwrite_output_dir=True
```

## 参数说明

- `--dataset_name`: tokenized 数据集路径，必须包含 `train`，可选 `validation`。
- `--per_device_train_batch_size`: 单卡 batch size。
- `--num_train_epochs`: 总训练轮数。
- `--learning_rate`: 学习率。
- `--save_steps`: 每多少步保存一次 checkpoint。
- `--save_total_limit`: 最多保留的 checkpoint 数量，防止占满磁盘。
- `--bf16=True`: 开启 bfloat16（4090 可用，通常更稳）。
- `--gradient_checkpointing=True`: 降低显存占用，训练速度会略降。
- `--ddp_find_unused_parameters=False`: DDP 下建议关闭 unused 参数检查，提升稳定性与性能。
- `--overwrite_output_dir=True`: 覆盖输出目录。

说明：脚本已兼容 `--evaluation_strategy`（旧参数名）与 `--eval_strategy`（新参数名），两者都可用。

## 已验证环境组合（2026-03）

在 `tlm` 环境中，本仓库已验证可跑通小样本训练+评估+checkpoint 保存的组合如下：

- `transformers==4.57.6`
- `accelerate==0.28.0`
- `numpy==1.26.4`
- `pyarrow==16.1.0`
- `datasets==2.10.0`
- `fsspec==2023.6.0`

可用以下命令快速检查：

```bash
/data/qsy/anaconda3/envs/tlm/bin/python - <<'PY'
import transformers, accelerate, numpy, pyarrow, datasets, fsspec
print('transformers', transformers.__version__)
print('accelerate', accelerate.__version__)
print('numpy', numpy.__version__)
print('pyarrow', pyarrow.__version__)
print('datasets', datasets.__version__)
print('fsspec', fsspec.__version__)
PY
```

## 续训方式

如果不想覆盖输出目录，去掉 `--overwrite_output_dir=True`，并加上:

```bash
--resume_from_checkpoint=/data/qsy/huggingface/model/Qwen3-0.6B-fintuned/checkpoint-xxxx
```

或保留已有输出目录并让脚本自动从最后 checkpoint 恢复。

## 训练前检查

建议先确认数据集结构:

```bash
ls -la /data/qsy/workspace/gen_data/4090_gen_qwen
ls -la /data/qsy/workspace/gen_data/4090_gen_qwen/train | head
ls -la /data/qsy/workspace/gen_data/4090_gen_qwen/validation | head
```

应包含:

- 顶层: `dataset_dict.json`、`train/`、`validation/`
- split 子目录: `data-*.arrow`、`dataset_info.json`、`state.json`

## 本次关键修复记录

