# 4090_gen_qwen 数据集改造说明

## 目标

为 Qwen3-0.6B 训练生成一份和模型 tokenizer 对齐的数据集：

- 原数据集: `/data/qsy/workspace/gen_data/4090_gen`
- 新数据集: `/data/qsy/workspace/gen_data/4090_gen_qwen`

核心原因：`train_qwen3_clm.py` 使用 `load_from_disk()` 直接读取已 tokenized 的 `input_ids`，不会在训练时重新分词。如果继续使用历史 `gen_tokenizer_4090` 构建的 `4090_gen`，会和 Qwen tokenizer 的词表不一致。

---

## 本次报错分析

你在执行 `make_dataset.py --for_type=for_gen` 时遇到：

`NotImplementedError: Loading a dataset cached in a LocalFileSystem is not supported.`

这是依赖版本组合导致的兼容性问题，不是参数写错。

当前环境里可以确认：

- `datasets==2.10.0`
- `fsspec==2024.5.0`

在 `datasets 2.10.0` 中，`is_remote_filesystem()` 对本地文件系统的判断逻辑较老；而新版本 `fsspec` 的 `LocalFileSystem.protocol` 是 `("file", "local")`（元组），会被旧逻辑误判成“远程文件系统”，从而触发上述异常。

你随后遇到的第二个报错是：

`ValueError: Tokenizer class Qwen2Tokenizer does not exist or is not currently imported.`

这同样是环境兼容问题。当前环境是 `transformers==4.35.0`，而 Qwen3 模型目录中：

- `tokenizer_config.json` 声明 `tokenizer_class: Qwen2Tokenizer`
- `config.json` 声明 `model_type: qwen3`

`transformers 4.35.0` 对 Qwen3/Qwen2Tokenizer 支持不完整，因此在 `AutoTokenizer.from_pretrained()` 阶段直接报错。

你随后又遇到的第三个报错是：

`KeyError: 'token_type_ids'`

原因是 `LLM/make_dataset_utils.py` 里原先直接 `del output["token_type_ids"]`，但 Qwen tokenizer 默认不返回该字段。

另外，Qwen3 tokenizer 的 `model_max_length=131072`，原逻辑使用 `padding="max_length"` 会把样本 pad 到超长长度，显著拖慢甚至导致内存风险。

---

## 先修环境再生成数据集（必须）

推荐方案（风险最低）：保留 `datasets==2.10.0`，仅降级 `fsspec`。

```bash
conda activate tlm
python -m pip install "fsspec==2023.6.0"
```

然后升级 transformers 到支持 Qwen3 的版本（必须）：

```bash
conda activate tlm
python -m pip install -U "transformers>=4.51,<4.58"
```

可选方案（升级栈）：升级 datasets 到较新版本并保持 fsspec 新版本。

```bash
conda activate tlm
python -m pip install -U "datasets>=2.18,<3"
```

说明：当前项目依赖较旧，优先建议“只降 fsspec”，对现有代码影响最小。

说明补充：为了支持 Qwen3，需要升级 transformers；这一步是必须的。保持 `datasets` 不动、仅修 `fsspec + transformers`，对现有流程改动最小。

修复后建议快速确认：

```bash
python - <<'PY'
import datasets, fsspec, transformers
print('datasets =', datasets.__version__)
print('fsspec =', fsspec.__version__)
print('transformers =', transformers.__version__)
PY
```

再做一次 tokenizer 可用性验证：

```bash
python - <<'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/data/qsy/huggingface/model/Qwen3-0.6B')
print(type(tok))
PY
```

---

## 需要修改的代码

本仓库已完成以下代码修复：

- `LLM/make_dataset_utils.py`
  - 将 `del output["token_type_ids"]` 改为 `output.pop("token_type_ids", None)`，兼容不返回该键的 tokenizer。
  - 增加 `_resolve_max_length()`：当 tokenizer 的 `model_max_length` 过大（>4096）时，默认回退到 `544`，保持与历史数据规模接近。
  - `make_dataset()` / `make_dataset_test()` 新增可选 `max_length` 参数。
- `LLM/make_dataset.py`
  - `ScriptArguments` 新增 `--max_length` 可选参数。
  - 调用 `make_dataset()` 时透传 `max_length`。

## 1) 修改训练启动脚本中的数据集路径

文件: `LLM/run_train_qwen3_clm.py`

将:

```python
--dataset_name=/data/qsy/workspace/gen_data/4090_gen \
```

改为:

```python
--dataset_name=/data/qsy/workspace/gen_data/4090_gen_qwen \
```

## 2) (建议) 修改训练脚本默认数据集路径

文件: `LLM/train_qwen3_clm.py`

将 `DataTrainingArguments.dataset_name` 默认值改为:

```python
default="/data/qsy/workspace/gen_data/4090_gen_qwen"
```

说明：第 2 条是“防误用”改动，避免后续忘记在命令行传 `--dataset_name` 时回落到旧路径。

---

## 如何生成 4090_gen_qwen

在 `LLM` 目录执行：

```bash
cd /data/qsy/workspace/complier/llm_compiler/LLM

python make_dataset.py \
  --for_type=for_gen \
  --target="nvidia/geforce-rtx-4090" \
  --dataset_path=/data/qsy/workspace/dataset/to_measure_programs/4090 \
  --tokenizer_path=/data/qsy/huggingface/model/Qwen3-0.6B \
  --save_path=/data/qsy/workspace/gen_data/4090_gen_qwen
```

如果希望显式指定序列长度，可增加：

```bash
  --max_length=544
```

要点：

- `--for_type=for_gen` 沿用原数据构造流程。
- 关键在 `--tokenizer_path` 必须改为 Qwen3-0.6B。
- `--save_path` 使用新目录，避免覆盖旧 `4090_gen`。

---

## 生成后校验

## 1) 校验数据集目录结构

```bash
ls -la /data/qsy/workspace/gen_data/4090_gen_qwen
```

应至少包含 datasets 的落盘文件（如 `dataset_info.json`、`state.json`、`train/`、`validation/` 等）。

## 2) 快速校验可读取性

```bash
python - <<'PY'
from datasets import load_from_disk
ds = load_from_disk('/data/qsy/workspace/gen_data/4090_gen_qwen')
print(ds)
print(ds['train'].column_names)
print(ds['train'][0].keys())
PY
```

应看到 `input_ids`、`attention_mask`、`labels`。

如果仍报 `LocalFileSystem` 相关错误，优先再次检查 `datasets/fsspec` 版本是否已生效，并确认命令在 `tlm` 环境执行。

如果仍报 `Qwen2Tokenizer` 相关错误，优先检查 `transformers` 版本是否已升级并实际生效。

## 3) 小样本试训（可选）

先用少量样本确认无报错：

```bash
cd /data/qsy/workspace/complier/llm_compiler/LLM

python train_qwen3_clm.py \
  --do_train \
  --model_name_or_path=/data/qsy/huggingface/model/Qwen3-0.6B \
  --tokenizer_name=/data/qsy/huggingface/model/Qwen3-0.6B \
  --dataset_name=/data/qsy/workspace/gen_data/4090_gen_qwen \
  --output_dir=/tmp/qwen3-0.6b-4090-gen-qwen-smoke \
  --max_train_samples=64 \
  --per_device_train_batch_size=2 \
  --num_train_epochs=1 \
  --overwrite_output_dir=True
```

---

## 兼容性说明

- `LLM/make_dataset.py` 的逻辑本身无需新增 `for_type`，直接替换 `--tokenizer_path` 即可生成 Qwen 对齐版数据集。
- `LLM/gen_state.py` 推理时会用 `model_name_or_path` 加载同一路径的 tokenizer；因此训练数据和推理 tokenizer 必须一致，避免 token-id 语义偏移。

---

## 训练阶段新增问题与修复（2026-03）

在数据集生成完成后，训练阶段又出现了几类依赖/API 兼容问题，已完成修复：

1. `ImportError: cannot import name 'is_torch_tpu_available' from transformers`

- 原因：新版 `transformers` 中导出位置变化。
- 修复：在 `train_qwen3_clm.py` 改为从 `transformers.utils` 导入，并在失败时回退为 `False`。

2. `ValueError: ... arguments are not used ... ['--evaluation_strategy=steps']`

- 原因：当前 `TrainingArguments` 使用 `--eval_strategy`。
- 修复：在脚本里增加参数别名兼容，把 `--evaluation_strategy` 自动映射到 `--eval_strategy`。

3. `AttributeError: module 'evaluate' has no attribute 'load'`

- 原因：项目目录下存在 `LLM/evaluate.py`，遮蔽了第三方 `evaluate` 包。
- 修复：训练脚本移除 `evaluate` 依赖，改为 numpy 直接计算 token-level accuracy。

4. `TypeError: Accelerator.unwrap_model() got an unexpected keyword argument 'keep_torch_compile'`

- 原因：`transformers 4.57.x` 与 `accelerate 0.28.0` 的接口差异。
- 修复：脚本中为 `accelerate.Accelerator.unwrap_model` 增加兼容包装。

5. 多卡训练在 `0%` 步骤直接 `Segmentation fault`

- 现象：`n_gpu=4` 且 Trainer 自动走单进程多卡（DataParallel）时，训练刚开始就段错误。
- 复现结论：
  - 多卡 DataParallel + BF16 + gradient_checkpointing：稳定段错误。
  - 单卡同参数可正常训练。
  - 切换到 `torchrun` DDP（4 进程）后可稳定启动并完成 smoke run。
- 修复：训练入口改为 DDP 启动（`torchrun --nproc_per_node=4`），并加 `--ddp_find_unused_parameters=False`。

经过上述修复，已完成 smoke run（8 train / 8 eval）并成功：

- 启动训练
- 执行评估
- 保存 checkpoint
- 保存最终模型
