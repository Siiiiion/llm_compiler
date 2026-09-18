# llm_compiler

`llm_compiler` 基于 Apache TVM，使用大语言模型生成张量程序的调度决策，再通过编译检查、
真实硬件测量和偏好训练改进调度。当前主要入口在 `LLM/`，支持 Qwen3 继续预训练、
SFT、DPO/LAPO，以及固定形状网络的 Relay 端到端评测。

这里的“训练模型”和“被编译模型”是两个角色：Qwen3 可以作为**调度生成器**；
ResNet、BERT 或另一个 Qwen3 ONNX 是**被编译的网络**。训练调度生成器不会自动改变
被编译网络的权重。

## 目录与阅读顺序

| 目录 / 文件 | 用途 |
| --- | --- |
| `src/`、`include/`、`python/tvm/` | TVM 编译器、运行时和 Python 接口 |
| `gen/dump_network_info.py` | 导出 Relay、提取 workload、建立任务注册表 |
| `gen/dump_programs.py` | 采样初始调度，并通过 LocalBuilder 检查 |
| `LLM/make_dataset.py`、`LLM/extend_tokenizer.py` | 制作训练数据、prompt，扩充调度词表 |
| `LLM/train/` | Qwen3 继续预训练、SFT、DPO/LAPO |
| `LLM/gen_state.py`、`LLM/measure_programs.py` | 生成调度、测量 GPU 延迟 |
| `LLM/benchmark_relay.py`、`LLM/benchmark_relay_batch.py` | 单网络 / 按网络汇总的端到端评测 |
| `LLM/export_onnx/`、`LLM/attention/` | 固定形状 ONNX 导出、独立 attention 实验 |
| `tests/`、`docs/`、`gallery/`、`plan/` | 测试、文档、教程、设计记录 |

从零开始：第 1–4 节准备环境和训练语料，第 5 节训练，第 6–8 节评测一个网络。
已有**经过调度训练的 checkpoint**：完成第 1–3 节后，在第 6 节设置该 checkpoint，
可跳过第 4–5 节。第 9 节进行偏好训练，第 10 节处理 Qwen3 ONNX，第 11–12 节扩展评测和迭代。

```text
网络 → Relay / workload 注册表 → 初始调度
                                  ├→ 训练数据 → Qwen3 调度生成器
                                  └→ sketch prompt
                                         ↓
生成决策 → 解析 / 去重 / 编译检查 → 真实硬件测量
                                         ├→ ApplyHistoryBest → 网络评测
                                         └→ 偏好对 → DPO/LAPO → 下一轮生成
```

## 1. 执行约定与系统准备

下面按 Linux、Bash、NVIDIA RTX 4090、CUDA Toolkit 12.1、Python 3.10 编写。
命令中的 GPU `0` 是示例；运行前用 `nvidia-smi` 确认设备和占用情况。训练可多卡，
测量使用一张独占 GPU。完整网络集提取、状态采样和训练都是长任务，先执行标注的 smoke 检查。

所有路径通过 `LLMC_` 变量指定，不依赖开发者的个人目录。请在**同一个 Bash 会话**
依次执行；新终端需要重新激活环境并设置这些变量。`\` 表示命令续行，其后不要加空格。
每条命令完成并通过对应检查后再继续；发生错误时先停止后续步骤，避免使用上次遗留产物。
下文假设已安装 Conda，可先用 `conda --version` 确认。

### 1.1 安装系统工具

Ubuntu / Debian 示例：

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git llvm-14-dev libclang-14-dev \
  libtinfo-dev zlib1g-dev libedit-dev libxml2-dev \
  tmux jq nano
```

另行安装与驱动兼容的 CUDA Toolkit；PyTorch 的 CUDA wheel 不包含 TVM 编译所需的完整
`nvcc` 工具链。确认：

```bash
nvidia-smi
/usr/local/cuda-12.1/bin/nvcc --version
/usr/bin/llvm-config-14 --version
cmake --version
```

如果 CUDA / LLVM 安装在其他位置，修改后面的 `LLMC_CUDA` / `LLMC_LLVM_CONFIG`。
CMake 至少需要 3.18。旧环境记录 `version.log` 仅供追溯，不作为 Qwen3 安装清单。

长任务可先进入持久会话，再执行后续步骤：

```bash
tmux new-session -s llm_compiler
```

### 1.2 设置项目和产物目录

先在终端进入当前仓库根目录，再执行：

```bash
test -f CMakeLists.txt
test -d LLM
export LLMC_REPO="$(pwd -P)"
export LLMC_WORK="$(dirname "$LLMC_REPO")/llm_compiler_artifacts/run01"
export LLMC_TARGET="nvidia/geforce-rtx-4090"
export LLMC_CUDA="/usr/local/cuda-12.1"
export LLMC_LLVM_CONFIG="/usr/bin/llvm-config-14"
export LLMC_TRAIN_GPUS="0"
export LLMC_NPROC=1
export LLMC_MEASURE_GPU="0"

export LLMC_NET="$LLMC_WORK/network_info/full"
export LLMC_STATES="$LLMC_WORK/to_measure/full"
export LLMC_BASE="$LLMC_WORK/models/Qwen3-0.6B"
export LLMC_EXT="$LLMC_WORK/models/Qwen3-0.6B-tvm-ext"
export LLMC_DATASET="$LLMC_WORK/datasets/stage1"
export LLMC_STAGE1="$LLMC_WORK/models/stage1"
export LLMC_PREFS="$LLMC_WORK/datasets/preferences"
export LLMC_LAPO="$LLMC_WORK/models/lapo"

mkdir -p "$LLMC_WORK"/{models,datasets,network_info,to_measure,measure_records,prompts,runs,reports,logs,onnx}
export PATH="$LLMC_CUDA/bin:$PATH"
export LD_LIBRARY_PATH="$LLMC_CUDA/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$LLMC_REPO/python:$LLMC_REPO/LLM"
export TVM_LIBRARY_PATH="$LLMC_REPO/build"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
unset TVM_USE_RUNTIME_LIB
set -o pipefail
```

`run01` 是本轮实验目录；新实验改为 `run02` 等新目录。后面的数据制作会重建指定输出目录，
不要把 `--save_path` 指向原始数据、注册表或模型目录。`pipefail` 让带 `tee` 的命令在主程序失败时仍返回失败状态。

硬件变化时必须同时修改 target、数据和输出目录。当前 `LLM/common.py` 的路径路由识别
`i7 / v100 / a100 / 2080 / 4090`；不能仅把 target 改成不带型号的 `cuda`。
其他硬件的接入需要先检查该路由和 `src/target/tag.cc`。

## 2. Python 环境与 TVM 构建

### 2.1 创建 Qwen3 工作环境

下面提供 Python 3.10 / CUDA 12.1 的安装示例，不是仓库已经锁定并全链验证的环境文件。
安装后必须完成第 2.4 节的检查。

```bash
conda create -n llm_compiler python=3.10 -y
conda activate llm_compiler

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install \
  "numpy==1.26.4" "scipy>=1.11,<1.14" \
  "transformers==4.51.3" "accelerate==1.6.0" \
  "datasets==3.6.0" "pyarrow==19.0.1" "evaluate==0.4.1" \
  "onnx==1.16.2" "onnxruntime==1.20.1" \
  sentencepiece safetensors huggingface-hub \
  attrs decorator cloudpickle psutil tornado tqdm pandas \
  "xgboost<3" pytest
```

根目录 `requirements.txt` 固定了 `transformers==4.31.0`，不支持这里的 Qwen3 主线；
不要在上述环境里再用它覆盖新版依赖。`numpy<2` 用于兼容这份 TVM 的旧接口。

### 2.2 初始化子模块和构建配置

```bash
cd "$LLMC_REPO"
git submodule update --init --recursive
mkdir -p build
if [ ! -f build/config.cmake ]; then
  cp cmake/config.cmake build/config.cmake
fi
nano build/config.cmake
```

在编辑器中修改对应配置项。以下是 **CMake 配置内容，不是终端命令**：

```cmake
set(USE_CUDA /usr/local/cuda-12.1)
set(USE_LLVM "/usr/bin/llvm-config-14 --link-static")
set(HIDE_PRIVATE_SYMBOLS ON)
set(USE_GRAPH_EXECUTOR ON)
```

CUDA / LLVM 路径必须与第 1 节一致。隐藏私有符号并静态链接 LLVM，可以减少 TVM 与
PyTorch 同进程加载时的 LLVM 符号冲突。已有 `build/config.cmake` 应编辑现有设置，避免重复定义。

### 2.3 编译与安装本地 Python 接口

```bash
cd "$LLMC_REPO"
cmake -S . -B build
cmake --build build -j4
python -m pip install -e ./python
```

`-j4` 表示最多四个并行编译任务，内存紧张时改成 `-j2`。修改 C++ 后重新构建；
仅修改 Python 脚本通常不需要重新编译。需要 Cython 扩展时再执行 `make cython3`。

必须使用本仓库编译出的 TVM，通用的 pip TVM 包不包含这里的调度扩展。

### 2.4 检查解释器、动态库、GPU 和 Qwen3 支持

```bash
cd "$LLMC_REPO"
python - <<'PY'
import os
import sys
from pathlib import Path
import torch
import transformers
import tvm
from tvm._ffi.base import _LIB

repo = Path(os.environ["LLMC_REPO"]).resolve()
print("Python:", sys.executable)
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("Transformers:", transformers.__version__)
print("TVM Python:", tvm.__file__)
print("TVM library:", _LIB._name)
assert (repo / "python") in Path(tvm.__file__).resolve().parents
assert Path(_LIB._name).resolve().parent == (repo / "build").resolve()
assert "runtime" not in Path(_LIB._name).name, "需要完整 libtvm，不是仅运行时库"
from tvm import auto_scheduler

assert hasattr(transformers, "Qwen3ForCausalLM")
assert hasattr(auto_scheduler.SketchPolicy, "gen_states")
assert tvm.runtime.enabled("llvm")
assert tvm.runtime.enabled("cuda")
assert torch.cuda.is_available()
assert tvm.cuda(0).exist
print("GPU:", torch.cuda.get_device_name(0))
print("Environment check passed")
PY

python -m pip check
python -m pytest tests/python/unittest/test_target_codegen_llvm.py -q
```

检查失败时先修复环境。尤其要核对打印出的 `TVM Python` 和 `TVM library` 是否都属于当前仓库，
不能把另一个 checkout 的 Python 文件与本地动态库混用。

### 2.5 内置网络提取使用独立环境

内置 BERT 提取代码基于较早的 Transformers 接口；新版默认 SDPA 可能产生这份 Relay
frontend 不支持的算子。因此第 4、6、11 节的**内置网络提取**使用独立环境，
Qwen3 的训练、生成、ONNX 导出继续使用 `llm_compiler`。

```bash
conda create -n llm_compiler_collect python=3.10 -y
conda run -n llm_compiler_collect python -m pip install \
  torch==2.2.0 torchvision==0.17.0 \
  --index-url https://download.pytorch.org/whl/cu121
conda run -n llm_compiler_collect python -m pip install \
  "numpy==1.26.4" "transformers==4.31.0" "scipy>=1.11,<1.14" \
  attrs decorator cloudpickle psutil tornado tqdm "xgboost<3"
conda run -n llm_compiler_collect python -m pip install -e "$LLMC_REPO/python"
conda run -n llm_compiler_collect python -c \
  'import tvm, torch, transformers; print(tvm.__file__); print(torch.__version__, transformers.__version__)'
```

此环境复用同一份 TVM 构建，不需要再编译一次。后文的 `conda run --no-capture-output`
会实时显示提取日志；它不会切换当前 shell 的训练环境。

## 3. 准备基础模型与调度 tokenizer

### 3.1 下载或指定 Qwen3-0.6B

有本地模型时，将 `LLMC_BASE` 改为其绝对路径。需要下载时执行：

```bash
python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    local_dir=os.environ["LLMC_BASE"],
)
PY
```

输出目录应包含 `config.json`、权重和 tokenizer 文件。模型下载需要访问 Hugging Face；
已有完整目录时无需重复下载。

### 3.2 扩充调度 token

```bash
cd "$LLMC_REPO/LLM"
python extend_tokenizer.py \
  --base_model "$LLMC_BASE" \
  --output_dir "$LLMC_EXT" \
  --promote_marker PPT \
  --pad_to_multiple_of 8
```

该命令输出新的模型和 tokenizer，并写入 `extend_tokenizer_manifest.json`。
`PPT` 是 prompt 与待生成调度后缀的分界。扩词表后必须重新制作数据集；
训练、生成必须使用与模型权重匹配的 tokenizer。

检查：

```bash
python - <<'PY'
import os
from transformers import AutoConfig, AutoTokenizer

path = os.environ["LLMC_EXT"]
tok = AutoTokenizer.from_pretrained(path)
cfg = AutoConfig.from_pretrained(path)
ids = tok("PPT", add_special_tokens=False)["input_ids"]
print("tokenizer size:", len(tok), "embedding rows:", cfg.vocab_size, "PPT:", ids)
assert len(ids) == 1
assert cfg.vocab_size >= len(tok)
PY
```

原始 Qwen3 只具备通用语言能力；扩词表本身也不等于已经学会调度，下一步需要继续预训练。

## 4. 提取 workload 并制作 Stage1 数据

### 4.1 先检查少量网络和初始调度

```bash
cd "$LLMC_REPO"
conda run --no-capture-output -n llm_compiler_collect \
  python gen/dump_network_info.py \
  --target "$LLMC_TARGET" \
  --network-info-dir "$LLMC_WORK/network_info/smoke" \
  --only-workloads \
  --workload resnet_50:1,3,224,224

python gen/dump_programs.py \
  --target "$LLMC_TARGET" \
  --network-info-dir "$LLMC_WORK/network_info/smoke" \
  --output-dir "$LLMC_WORK/to_measure/smoke" \
  --start-idx 0 --end-idx 2 \
  --size 16 \
  --build-n-parallel 4 \
  --build-batch-size 16 \
  --stats-file "$LLMC_WORK/reports/dump_smoke.json"

jq '.summary' "$LLMC_WORK/reports/dump_smoke.json"
```

`--only-workloads` 限制为指定网络；`--start-idx 0 --end-idx 2` 只处理前两个 workload，
右端不包含。这里只验证提取和编译流程，不代表完整网络已覆盖。

注册表包含 `*.task.pkl` 和 `all_tasks.pkl`；未使用 `--skip-relay-artifact` 时还会保存含完整参数
的 `*.relay.pkl`。初始调度按 workload 分别写成 `*.json`，其中延迟 `[0.0]` 代表**尚未测量**。

### 4.2 提取完整内置网络注册表

```bash
cd "$LLMC_REPO"
conda run --no-capture-output -n llm_compiler_collect \
  python gen/dump_network_info.py \
  --target "$LLMC_TARGET" \
  --network-info-dir "$LLMC_NET" \
  --skip-relay-artifact \
  2>&1 | tee "$LLMC_WORK/logs/dump_networks.log"
```

默认网络和形状组合由 `python/tvm/meta_schedule/testing/dataset_collect_models.py` 定义。
`--skip-relay-artifact` 省略包含模型权重的庞大 Relay 文件，仍保存训练需要的任务注册表。
后续对指定网络评测时再补出其 Relay 文件。

当前 `make_dataset.py --for_type for_gen` 会读取 `LLM/common.py` 中全部内置 hold-out 映射，
且该模式下 `--skip_hold_out` 不生效。因此 Stage1 不能直接使用第 4.1 节的小注册表，
也不能直接换成仅含 Qwen3 的注册表。

### 4.3 为完整注册表生成训练调度

```bash
cd "$LLMC_REPO"
python gen/dump_programs.py \
  --target "$LLMC_TARGET" \
  --network-info-dir "$LLMC_NET" \
  --output-dir "$LLMC_STATES" \
  --size 1000 \
  --validate-build \
  --build-timeout 30 \
  --build-n-parallel 4 \
  --build-batch-size 32 \
  --max-rounds 20 \
  --stats-file "$LLMC_WORK/reports/dump_full.json" \
  2>&1 | tee "$LLMC_WORK/logs/dump_programs.log"

jq '.summary' "$LLMC_WORK/reports/dump_full.json"
```

`--size 1000` 是每个 workload 的目标候选数。`partial` 表示数量不足，
`no_valid_states` 表示没有找到可编译候选。默认跳过已存在的 workload 文件；
如需增加候选数，使用新输出目录，或在明确重建范围后使用 `--overwrite`。

统计文件必须放在 `reports/` 等独立目录。数据制作会扫描输入目录下全部 `*.json`，
不能把统计 JSON 混入 TVM 调度记录。

### 4.4 Tokenize 并划分训练 / 验证集

```bash
cd "$LLMC_REPO/LLM"
python make_dataset.py \
  --for_type for_gen \
  --target "$LLMC_TARGET" \
  --network_info_dir "$LLMC_NET" \
  --to_measure_program_dir "$LLMC_STATES" \
  --measure_record_dir "$LLMC_WORK/measure_records/full" \
  --dataset_path "$LLMC_STATES" \
  --tokenizer_path "$LLMC_EXT" \
  --save_path "$LLMC_DATASET" \
  --max_length 1024 \
  --valid_percentage 5 \
  --min_suffix_tokens 16 \
  --split_seed 0 \
  --ppt_marker PPT
```

此步排除 hold-out workload，按源 workload 文件划分训练 / 验证数据，并预计算
`length`、`ppt_end`。`max_length` 是 tokenize 后的长度上限，
`min_suffix_tokens` 用于过滤过短的决策后缀。输出目录会被重建，保留其中的 `logs/`。

检查数据是否非空、列是否齐全：

```bash
python - <<'PY'
import os
from datasets import load_from_disk

ds = load_from_disk(os.environ["LLMC_DATASET"])
print(ds)
for split, rows in ds.items():
    print(split, len(rows), rows.column_names)
    assert len(rows) > 0
    assert {"input_ids", "attention_mask", "labels", "length", "ppt_end"} <= set(rows.column_names)
assert "train" in ds and "validation" in ds
PY

jq '.' "$LLMC_DATASET/for_gen_stats.json"
```

输出包括 Hugging Face DatasetDict、`train_raw.json`、`validation_raw.json`、
`0_merge.json` 和 `for_gen_stats.json`。原始未测调度可用于此阶段的结构学习，
不能直接作为延迟偏好数据。

## 5. 训练 Qwen3 调度生成器

### 5.1 单卡 smoke：只训练 10 步

```bash
cd "$LLMC_REPO/LLM"
CUDA_VISIBLE_DEVICES="${LLMC_TRAIN_GPUS%%,*}" \
python train/train_qwen3_clm.py \
  --model_name_or_path "$LLMC_EXT" \
  --tokenizer_name "$LLMC_EXT" \
  --dataset_name "$LLMC_DATASET" \
  --output_dir "$LLMC_WORK/models/stage1_smoke" \
  --do_train --do_eval \
  --max_train_samples 128 --max_eval_samples 32 \
  --max_steps 10 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --bf16 True --torch_dtype bfloat16 \
  --mask_prefix_before_ppt True --mask_in_collator False \
  --min_suffix_tokens 16 \
  --eval_strategy steps --eval_steps 5 \
  --save_strategy no --logging_steps 1 \
  --report_to none \
  2>&1 | tee "$LLMC_WORK/logs/stage1_smoke.log"
```

`${LLMC_TRAIN_GPUS%%,*}` 从 GPU 列表中取第一张卡。检查日志中训练、验证均完成，
loss 为有限值。smoke checkpoint 只用于确认流程，不作为正式性能结果。

### 5.2 正式训练

以下默认一张卡。四卡时设置 `LLMC_TRAIN_GPUS="0,1,2,3"`、`LLMC_NPROC=4`；
进程数必须等于可见 GPU 数。

```bash
export LLMC_NPROC=1
cd "$LLMC_REPO/LLM"
CUDA_VISIBLE_DEVICES="$LLMC_TRAIN_GPUS" \
python -m torch.distributed.run --standalone --nproc_per_node="$LLMC_NPROC" \
  train/train_qwen3_clm.py \
  --model_name_or_path "$LLMC_EXT" \
  --tokenizer_name "$LLMC_EXT" \
  --dataset_name "$LLMC_DATASET" \
  --output_dir "$LLMC_STAGE1" \
  --do_train --do_eval \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.03 --weight_decay 0.01 \
  --lr_scheduler_type cosine \
  --bf16 True --torch_dtype bfloat16 \
  --mask_prefix_before_ppt True --mask_in_collator False \
  --min_suffix_tokens 16 \
  --group_by_length True --length_column_name length \
  --dataloader_num_workers 4 \
  --eval_strategy steps --eval_steps 500 \
  --save_strategy steps --save_steps 500 --save_total_limit 2 \
  --logging_steps 20 \
  --ddp_find_unused_parameters False \
  --report_to none \
  2>&1 | tee "$LLMC_WORK/logs/stage1.log"
```

有效 batch size 为 `每卡 batch × 梯度累积 × GPU 数`。显存不足时先降低每卡 batch；
还可加 `--gradient_checkpointing True`，代价是额外计算。

输出 `LLMC_STAGE1` 包含最终权重、tokenizer 和训练指标，周期性 checkpoint 位于
`checkpoint-*` 子目录。恢复训练时在同一命令中加
`--resume_from_checkpoint "$LLMC_STAGE1/checkpoint-实际步数"`，保留原数据和训练设置。

这里使用 `--mask_in_collator False`，在预处理时过滤 tokenize / 截断后没有有效 `PPT`
或决策后缀太短的样本。确认整个数据集的 `ppt_end` 和后缀长度有效后，可改为 `True`，
在 batch 内应用 mask，减少大规模 Arrow 缓存开销。两种方式都只监督 `PPT` 后的决策后缀。

## 6. 为指定网络生成可编译调度

本节以 `resnet_50`、输入 `[1,3,224,224]` 为例；它属于内置 hold-out 网络。

### 6.1 选择 checkpoint，准备评测注册表和初始状态

如果使用已有 checkpoint，将第一条路径改为其绝对路径，目录必须同时有匹配的 tokenizer。

```bash
export LLMC_GENERATOR="$LLMC_STAGE1"
export LLMC_EVAL_NET="$LLMC_WORK/network_info/resnet50"
export LLMC_EVAL_STATES="$LLMC_WORK/to_measure/resnet50"
export LLMC_EVAL_RUN="$LLMC_WORK/runs/resnet50_stage1"
mkdir -p "$LLMC_EVAL_RUN"

cd "$LLMC_REPO"
conda run --no-capture-output -n llm_compiler_collect \
  python gen/dump_network_info.py \
  --target "$LLMC_TARGET" \
  --network-info-dir "$LLMC_EVAL_NET" \
  --only-workloads \
  --workload resnet_50:1,3,224,224

python gen/dump_programs.py \
  --target "$LLMC_TARGET" \
  --network-info-dir "$LLMC_EVAL_NET" \
  --output-dir "$LLMC_EVAL_STATES" \
  --size 256 \
  --build-n-parallel 4 \
  --stats-file "$LLMC_WORK/reports/dump_resnet50.json"
```

这里不加 `--skip-relay-artifact`，因为第 8 节需要完整网络和参数。

### 6.2 制作 sketch prompt

```bash
cd "$LLMC_REPO/LLM"
python make_dataset.py \
  --for_type for_gen_eval_sketch \
  --target "$LLMC_TARGET" \
  --network_info_dir "$LLMC_EVAL_NET" \
  --to_measure_program_dir "$LLMC_EVAL_STATES" \
  --measure_record_dir "$LLMC_WORK/measure_records/resnet50" \
  --dataset_path "$LLMC_EVAL_STATES" \
  --tokenizer_path "$LLMC_GENERATOR" \
  --save_path "$LLMC_EVAL_RUN/prompts" \
  --eval_workloads resnet_50 \
  --keep_cnt 48

wc -l "$LLMC_EVAL_RUN/prompts/0_merge.json"
```

此处 `keep_cnt` 是每个 workload 采样的 **prompt 数**，允许重复抽样，
不是不同 sketch 的数量保证。可选网络名由 `LLM/common.py` 的 `hold_out_task_files` 定义；
例如 `bert_base` 对应其固定映射形状，不会自动覆盖所有 BERT 变体。

### 6.3 生成并检查调度

```bash
CUDA_VISIBLE_DEVICES="$LLMC_TRAIN_GPUS" \
python gen_state.py \
  --target "$LLMC_TARGET" \
  --network_info_dir "$LLMC_EVAL_NET" \
  --model_name_or_path "$LLMC_GENERATOR" \
  --sketch_path "$LLMC_EVAL_RUN/prompts/0_merge.json" \
  --save_path "$LLMC_EVAL_RUN/candidates.json" \
  --stats_path "$LLMC_EVAL_RUN/generation_stats.json" \
  --keep_cnt 16 \
  --allow_repeat True \
  --is_build True \
  --build_n_parallel 4 --build_timeout 30 \
  --max_retries 20 --require_full_keep_cnt True \
  --do_sample True --disable_eos_stop False \
  --min_gen_tokens 256 --gen_token_scale 4.0 \
  --generation_batch_size 8 \
  --sample_top_k 0 --sample_top_p 1.0 --sample_temperature 0.6 \
  2>&1 | tee "$LLMC_WORK/logs/generate_resnet50.log"

jq '.workloads[] | {
  workload_key, parsed_states, build_attempted, build_valid,
  duplicates, kept, shortfall, build_error_counts
}' "$LLMC_EVAL_RUN/generation_stats.json"
wc -l "$LLMC_EVAL_RUN/candidates.json"
```

这里 `keep_cnt=16` 指每个输入 workload 最终保留的**唯一候选数**。
`is_build=True` 会实际编译候选。`allow_repeat=True` 允许与历史轮次重复，
当前运行内仍去重。`require_full_keep_cnt=True` 在数量不足时返回失败，保留原输出，
不会用不足的结果覆盖它。

成功时全部 `shortfall` 应为 `0`。若不足，先看编译错误和生成统计，
再增加 `max_retries`、调整采样或改进模型；增加重试不保证一定补齐。
完整候选覆盖仍不等于完整**有效测量**覆盖。

## 7. 测量候选与检查有效记录

### 7.1 在独占 GPU 上测量

确保同一张 GPU 上的训练 / 生成已经结束：

```bash
nvidia-smi
cd "$LLMC_REPO/LLM"
CUDA_VISIBLE_DEVICES="$LLMC_MEASURE_GPU" \
python measure_programs.py \
  --target "$LLMC_TARGET" \
  --network-info-dir "$LLMC_EVAL_NET" \
  --to-measure-path "$LLMC_EVAL_RUN/candidates.json" \
  --measured-path "$LLMC_EVAL_RUN/measured.json" \
  --batch-size 64 \
  --run-timeout 60 \
  --print-error-details \
  --resume \
  2>&1 | tee "$LLMC_WORK/logs/measure_resnet50.log"
```

`--resume` 根据输入签名跳过已写入的记录，适合中断后续测。
`--no-resume` 会清空该输出重新测量；要重新测失败记录或候选已经换了一批，
优先使用新输出文件。不要把不同模型、target 或实验条件的记录无意混在一起。

`--input-dir` 是另一种**就地补测**模式，会改写输入目录中的文件；
这里使用输入 / 输出分开的接口，保留原始候选。

### 7.2 检查成功率

```bash
jq -s '{
  total: length,
  valid: ([.[] | select(
    .r[1] == 0 and any(.r[0][]; . > 0 and . < 1000000000)
  )] | length),
  errors: (group_by(.r[1]) | map({error_no: .[0].r[1], count: length}))
}' "$LLMC_EVAL_RUN/measured.json"
```

TVM record 的 `r[0]` 是秒单位的测量 cost，`r[1]` 是错误码。
有效记录要求错误码为 `0` 且存在有限、正的有效 cost；`0` 是未测量，
`1e10` 等失败占位值不能用于训练偏好或统计加速比。
总有效条数很高也可能遗漏某个 workload，第 8 节会检查精确的任务覆盖。

## 8. 单网络 Relay 端到端评测

```bash
export LLMC_RELAY="$LLMC_EVAL_NET/(resnet_50,[1,3,224,224]).relay.pkl"
test -f "$LLMC_RELAY"

cd "$LLMC_REPO/LLM"
CUDA_VISIBLE_DEVICES="$LLMC_MEASURE_GPU" \
python benchmark_relay.py \
  --relay-pickle "$LLMC_RELAY" \
  --history "$LLMC_EVAL_RUN/measured.json" \
  --target "$LLMC_TARGET" \
  --mode both \
  --warmup 5 --number 1 --repeat 10 --min-repeat-ms 500 \
  --report "$LLMC_EVAL_RUN/benchmark.json" \
  2>&1 | tee "$LLMC_WORK/logs/benchmark_resnet50.log"

jq '{
  status, history_coverage, correctness, speedup,
  baseline_ms: .baseline.timing.median_ms,
  optimized_ms: .optimized.timing.median_ms
}' "$LLMC_EVAL_RUN/benchmark.json"
```

两种模式都使用同一个网络、参数和输入：

- `baseline`：Relay `opt_level=3` 默认优化。
- `optimized`：Relay + AutoScheduler `ApplyHistoryBest`，使用有效历史中的最佳调度。
- `speedup = baseline median_ms / optimized median_ms`；大于 `1` 表示更快。
- 编译时间单独记录，不计入推理延迟；输出逐项与 baseline 比较。

正式比较应满足 `status=ok`、`correctness.passed=true`、
`history_coverage.complete=true`。这是 TVM 两种编译路径的一致性检查，
不等于已经与原始 PyTorch 模型验证等价。

缺少任务历史时，脚本写入 `refused_incomplete_history` 报告并拒绝优化评测。
根据报告补测缺失 workload。`--allow-incomplete-history` 仅用于排错：
允许 fallback 不保证可编译，其结果不属于完整调度覆盖的正式加速比。

默认示例每次重新编译。如指定 `--baseline-lib` / `--optimized-lib`，已有 `.so` 会被直接加载；
更换网络、target、TVM 构建或历史后必须选择新的缓存路径。
微秒级算子的计时应增加 `number` 和 `min-repeat-ms`，并关注 GPU 功耗状态。

## 9. 从真实测量构建偏好对，训练 DPO / LAPO

这一节使用**非 hold-out 的训练 workload**重新生成并测量调度，不把第 8 节的测试网络
记录直接当成训练集。仅有 Stage1 数据里的零延迟候选不足以构造偏好对。

如果前面使用了已有 checkpoint 的快捷路径，本节仍需完成第 4 节的注册表和初始调度准备，
并将 `LLMC_STAGE1` 设置为该 checkpoint 的绝对路径。`LLMC_NPROC` 必须与训练 GPU 数一致。

### 9.1 为训练 workload 生成下一批候选

```bash
export LLMC_TRAIN_RUN="$LLMC_WORK/runs/train_round01"
mkdir -p "$LLMC_TRAIN_RUN"
cd "$LLMC_REPO/LLM"

python make_dataset.py \
  --for_type for_gen_train_sketch \
  --target "$LLMC_TARGET" \
  --network_info_dir "$LLMC_NET" \
  --to_measure_program_dir "$LLMC_STATES" \
  --measure_record_dir "$LLMC_WORK/measure_records/full" \
  --dataset_path "$LLMC_STATES" \
  --tokenizer_path "$LLMC_STAGE1" \
  --save_path "$LLMC_TRAIN_RUN/prompts" \
  --keep_cnt 48 \
  --test_file_idx 0

CUDA_VISIBLE_DEVICES="$LLMC_TRAIN_GPUS" \
python gen_state.py \
  --target "$LLMC_TARGET" \
  --network_info_dir "$LLMC_NET" \
  --model_name_or_path "$LLMC_STAGE1" \
  --sketch_path "$LLMC_TRAIN_RUN/prompts/0_merge.json" \
  --save_path "$LLMC_TRAIN_RUN/candidates.json" \
  --keep_cnt 16 --allow_repeat True \
  --is_build True --build_n_parallel 4 \
  --require_full_keep_cnt True --max_retries 20 \
  --generation_batch_size 8 \
  --do_sample True --sample_temperature 0.6 \
  2>&1 | tee "$LLMC_WORK/logs/generate_train_round01.log"

CUDA_VISIBLE_DEVICES="$LLMC_MEASURE_GPU" \
python measure_programs.py \
  --target "$LLMC_TARGET" \
  --network-info-dir "$LLMC_NET" \
  --to-measure-path "$LLMC_TRAIN_RUN/candidates.json" \
  --measured-path "$LLMC_TRAIN_RUN/measured.json" \
  --batch-size 64 --run-timeout 60 --print-error-details --resume \
  2>&1 | tee "$LLMC_WORK/logs/measure_train_round01.log"
```

`test_file_idx` 按四组选择 workload，`0/1/2/3` 分别选择一组。
不传此参数会处理全部非 hold-out workload。多组实验使用不同的 `LLMC_TRAIN_RUN`，
不要依次覆盖同一个输出目录。

### 9.2 按 workload 拆分测量文件

```bash
python split_measure_records.py \
  --input-path "$LLMC_TRAIN_RUN/measured.json" \
  --output-dir "$LLMC_TRAIN_RUN/measured_by_workload"
```

偏好构造按每个文件的第一条记录恢复计算图，因此输入必须是**每个 workload 一个文件**。
不能把包含多个 workload 的 merged JSON 当成一个 workload 文件。
拆分器不会过滤所有测量错误，后续偏好构造会筛选有效延迟。
输出目录非空时默认拒绝；`--overwrite` 会替换整个目标目录。

### 9.3 构建偏好对，并显式绑定注册表

当前 `train/build_preference_pairs.py` 尚未提供 `--network_info_dir`，
直接运行会使用 `common.py` 中的机器默认路径。下面通过模块入口绑定本轮目录，
其余仍走原脚本的参数解析和实现；不需要修改源文件。

```bash
cd "$LLMC_REPO/LLM"
python - \
  --target "$LLMC_TARGET" \
  --dataset_path "$LLMC_TRAIN_RUN/measured_by_workload" \
  --save_path "$LLMC_PREFS" \
  --min_latency_gap_ratio 1.5 \
  --max_pairs_per_group 4 \
  --valid_percentage 5 \
  --weight_strategy log_gap \
  --drop_hold_out_workloads True \
  --num_workers 4 <<'PY'
import os
from functools import partial
from train import build_preference_pairs as pairs

pairs.register_data_path = partial(
    pairs.register_data_path,
    network_info_folder=os.environ["LLMC_NET"],
    to_measure_program_folder=os.environ["LLMC_STATES"],
    measure_record_folder=os.path.join(os.environ["LLMC_WORK"], "measure_records/full"),
)
pairs.main()
PY

wc -l "$LLMC_PREFS/train_pairs.jsonl"
jq '.' "$LLMC_PREFS/build_stats.json"
python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["LLMC_PREFS"])
assert (root / "train_pairs.jsonl").stat().st_size > 0, "训练偏好对为空"
validation = root / "validation_pairs.jsonl"
assert not validation.exists() or validation.stat().st_size > 0, "验证偏好文件存在但为空"
PY
```

偏好对共享同一个 workload 和 `PPT` 前缀，最低延迟后缀为 chosen，较慢后缀为 rejected；
`min_latency_gap_ratio=1.5` 要求 rejected / chosen 延迟比至少为 1.5。
输出是 `train_pairs.jsonl`、可选的 `validation_pairs.jsonl` 和 `build_stats.json`。

若验证文件存在但为空，应选择新的 `LLMC_PREFS` 输出目录，并以 `--valid_percentage 0`
重建后关闭训练内验证。训练入口会尝试读取已有验证文件，单纯去掉 `--do_eval` 无法绕过
空文件；复用旧目录还可能留下过期的验证集。

如果训练对数为零，检查是否有同前缀的多个有效测量、延迟差距是否足够、workload 是否全被
hold-out 过滤。增加每个 workload 的生成候选数并测量；不要把失败 cost 当作 rejected 延迟。

### 9.4 先做 LAPO smoke，再正式训练

以下使用直接训练入口，避免包装脚本的默认 GPU、解释器和 tmux 路径。

```bash
cd "$LLMC_REPO/LLM"
CUDA_VISIBLE_DEVICES="${LLMC_TRAIN_GPUS%%,*}" \
python train/train_qwen3_dpo.py \
  --model_name_or_path "$LLMC_STAGE1" \
  --ref_model_name_or_path "$LLMC_STAGE1" \
  --tokenizer_name "$LLMC_STAGE1" \
  --preference_dataset "$LLMC_PREFS" \
  --output_dir "$LLMC_WORK/models/lapo_smoke" \
  --training_mode lapo --beta 0.1 \
  --max_prompt_length 512 --max_length 1024 \
  --do_train --max_steps 5 --max_train_samples 32 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 1 \
  --learning_rate 1e-6 \
  --bf16 True --torch_dtype bfloat16 \
  --eval_strategy no --save_strategy no --logging_steps 1 \
  --remove_unused_columns False --report_to none \
  2>&1 | tee "$LLMC_WORK/logs/lapo_smoke.log"
```

确认训练对非空且 loss 有限后执行正式训练：

```bash
CUDA_VISIBLE_DEVICES="$LLMC_TRAIN_GPUS" \
python -m torch.distributed.run --standalone --nproc_per_node="$LLMC_NPROC" \
  train/train_qwen3_dpo.py \
  --model_name_or_path "$LLMC_STAGE1" \
  --ref_model_name_or_path "$LLMC_STAGE1" \
  --tokenizer_name "$LLMC_STAGE1" \
  --preference_dataset "$LLMC_PREFS" \
  --output_dir "$LLMC_LAPO" \
  --training_mode lapo --beta 0.1 \
  --lapo_weight_clip 10 --lapo_normalize True \
  --max_prompt_length 512 --max_length 1024 \
  --do_train --num_train_epochs 1 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
  --learning_rate 1e-6 --warmup_ratio 0.03 \
  --bf16 True --torch_dtype bfloat16 \
  --eval_strategy no --save_strategy steps --save_steps 100 \
  --save_total_limit 2 --logging_steps 10 \
  --remove_unused_columns False \
  --ddp_find_unused_parameters False --report_to none \
  2>&1 | tee "$LLMC_WORK/logs/lapo.log"
```

默认关闭训练内验证，使少量 workload 没有 validation 文件时也可执行。
如确实存在非空验证集，可增加 `--do_eval` 并改为 `--eval_strategy steps --eval_steps 100`。

LAPO 的权重在每张卡的当前 batch 内归一化，所以这里每卡 batch 至少为 2。
如果 batch 为 1 且 `lapo_normalize=True`，权重会变成 1，退化为普通 DPO；增加梯度累积
不能解决这一点。显存不足时可开启 `--gradient_checkpointing True`；必须用 batch 1 时，
显式设置 `--lapo_normalize False` 并记录这一实验差异。

同一偏好数据上：`--training_mode dpo` 使用标准 DPO，`sft` 仅监督 chosen 后缀，
`lapo` 使用延迟权重。消融实验应使用相同 Stage1 初始模型、数据、预算和不同输出目录。

比较训练效果时，将第 6 节 `LLMC_GENERATOR` 改为 `LLMC_LAPO`，
`LLMC_EVAL_RUN` 改为新的 `resnet50_lapo` 目录，重新执行 prompt、生成、测量、评测。
核对 `dpo_manifest.json` 中的模型来源；低训练 loss 不保证端到端更快。

## 10. 编译与评测固定形状 Qwen3 ONNX

本节使用第 5 节的调度生成器，处理一个新的被编译网络。它是固定形状 prefill，
不包含 KV cache 解码循环，结果不能直接称为完整聊天服务的 tokens/s。

### 10.1 从原始 Qwen3 权重导出 ONNX

```bash
export LLMC_GENERATOR="$LLMC_STAGE1"
export LLMC_QWEN_NET="$LLMC_WORK/network_info/qwen3_seq128_fp32"
export LLMC_QWEN_STATES="$LLMC_WORK/to_measure/qwen3_seq128_fp32"
export LLMC_QWEN_RUN="$LLMC_WORK/runs/qwen3_stage1"
export LLMC_QWEN_ONNX="$LLMC_WORK/onnx/qwen3_seq128_fp32/model.onnx"
mkdir -p "$LLMC_QWEN_RUN"

cd "$LLMC_REPO"
CUDA_VISIBLE_DEVICES="$LLMC_MEASURE_GPU" \
python LLM/export_onnx/export_onnx_qwen3.py \
  --model-dir "$LLMC_BASE" \
  --output "$LLMC_QWEN_ONNX" \
  --batch-size 1 --seq-len 128 \
  --dtype float32 --device cuda --opset 17 \
  --logits-to-keep 0 \
  --attn-implementation eager \
  --no-trust-remote-code

python - <<'PY'
import os
import onnx

path = os.environ["LLMC_QWEN_ONNX"]
onnx.checker.check_model(path)
model = onnx.load(path, load_external_data=False)
for item in model.graph.input:
    print(item.name, [dim.dim_value for dim in item.type.tensor_type.shape.dim])
PY
```

导出使用 `LLMC_BASE` 原始模型，而不是学过调度 token 的 checkpoint。
`logits-to-keep=0` 保留所有位置 logits；`1` 仅计算最后位置。
不同 dtype、形状和 logits 模式使用独立 artifact，比较时必须相同。
ONNX 可能另存外部权重文件，移动时需保留整个目录。

### 10.2 建立 Qwen3 专用注册表和初始调度

```bash
python gen/dump_network_info.py \
  --target "$LLMC_TARGET" \
  --network-info-dir "$LLMC_QWEN_NET" \
  --only-workloads \
  --onnx-model "$LLMC_QWEN_ONNX" \
  --model-name qwen3_0_6b_seq128_fp32 \
  --input-shapes '{"input_ids":[1,128],"attention_mask":[1,128]}' \
  --input-dtypes '{"input_ids":"int64","attention_mask":"int64"}' \
  --fix-mask-where-dtype

python gen/dump_programs.py \
  --target "$LLMC_TARGET" \
  --network-info-dir "$LLMC_QWEN_NET" \
  --output-dir "$LLMC_QWEN_STATES" \
  --size 256 --build-n-parallel 4 \
  --stats-file "$LLMC_WORK/reports/dump_qwen3.json"
```

`--only-workloads` 避免先遍历全部内置网络。`--fix-mask-where-dtype` 修正特定 ONNX
attention mask 常量的 FP64 / FP32 / FP16 类型不一致，后续评测复用此处生成的 Relay。
不要混用通用 4090 注册表和 Qwen3 专用记录；任务数量以当前提取结果为准。

### 10.3 遍历专用注册表中的全部 workload

```bash
cd "$LLMC_REPO/LLM"
python make_dataset.py \
  --for_type for_gen_train_sketch \
  --target "$LLMC_TARGET" \
  --network_info_dir "$LLMC_QWEN_NET" \
  --to_measure_program_dir "$LLMC_QWEN_STATES" \
  --measure_record_dir "$LLMC_WORK/measure_records/qwen3" \
  --dataset_path "$LLMC_QWEN_STATES" \
  --tokenizer_path "$LLMC_GENERATOR" \
  --save_path "$LLMC_QWEN_RUN/prompts" \
  --keep_cnt 48 --skip_hold_out True

CUDA_VISIBLE_DEVICES="$LLMC_TRAIN_GPUS" \
python gen_state.py \
  --target "$LLMC_TARGET" \
  --network_info_dir "$LLMC_QWEN_NET" \
  --model_name_or_path "$LLMC_GENERATOR" \
  --sketch_path "$LLMC_QWEN_RUN/prompts/0_merge.json" \
  --save_path "$LLMC_QWEN_RUN/candidates.json" \
  --keep_cnt 16 --allow_repeat True \
  --is_build True --build_n_parallel 4 \
  --require_full_keep_cnt True --max_retries 20 \
  --generation_batch_size 8 --do_sample True \
  2>&1 | tee "$LLMC_WORK/logs/generate_qwen3.log"

CUDA_VISIBLE_DEVICES="$LLMC_MEASURE_GPU" \
python measure_programs.py \
  --target "$LLMC_TARGET" \
  --network-info-dir "$LLMC_QWEN_NET" \
  --to-measure-path "$LLMC_QWEN_RUN/candidates.json" \
  --measured-path "$LLMC_QWEN_RUN/measured.json" \
  --batch-size 64 --run-timeout 60 --print-error-details --resume \
  2>&1 | tee "$LLMC_WORK/logs/measure_qwen3.log"
```

这里使用 `for_gen_train_sketch` 仅为了读取自定义注册表的 prompt，
并不表示一定要用这些 workload 训练。不要传 `--test_file_idx` 或调度筛选文件，
否则只生成部分任务，无法用于首次完整网络覆盖。
自定义 ONNX 名字不属于 `--eval_workloads` 的内置名字映射。

### 10.4 使用合法输入进行整图评测

```bash
export LLMC_QWEN_RELAY="$LLMC_QWEN_NET/(onnx_qwen3_0_6b_seq128_fp32,[(attention_mask,[1,128]),(input_ids,[1,128])]).relay.pkl"
test -f "$LLMC_QWEN_RELAY"

python - <<'PY'
import os
import numpy as np

rng = np.random.default_rng(0)
np.savez(
    os.path.join(os.environ["LLMC_QWEN_RUN"], "inputs.npz"),
    input_ids=rng.integers(0, 100, size=(1, 128), dtype=np.int64),
    attention_mask=np.ones((1, 128), dtype=np.int64),
)
PY

CUDA_VISIBLE_DEVICES="$LLMC_MEASURE_GPU" \
python benchmark_relay.py \
  --relay-pickle "$LLMC_QWEN_RELAY" \
  --input-npz "$LLMC_QWEN_RUN/inputs.npz" \
  --history "$LLMC_QWEN_RUN/measured.json" \
  --target "$LLMC_TARGET" \
  --warmup 5 --number 1 --repeat 10 --min-repeat-ms 500 \
  --report "$LLMC_QWEN_RUN/benchmark.json" \
  2>&1 | tee "$LLMC_WORK/logs/benchmark_qwen3.log"

jq '{status, correctness, speedup, coverage: .history_coverage.complete}' \
  "$LLMC_QWEN_RUN/benchmark.json"
```

输入是形状 / 类型合法的合成数据，attention mask 全为 1；它不是任务准确率数据集。
FP32 全 logits 和长序列会占用较多显存、主存，必要时先缩短序列并完整重建该形状的注册表与调度。

## 11. 多网络评测与结果汇总

第 4 节的训练注册表省略了 Relay 参数。若要评测全部内置网络，先补齐工件；
这一步会为许多形状分别保存完整参数，应预留充足磁盘和主存。

```bash
cd "$LLMC_REPO"
conda run --no-capture-output -n llm_compiler_collect \
  python gen/dump_network_info.py \
  --target "$LLMC_TARGET" \
  --network-info-dir "$LLMC_NET" \
  --overwrite-relay \
  2>&1 | tee "$LLMC_WORK/logs/redump_relay.log"
```

已有旧版 `*.relay.pkl` 如果只保存参数字节长度，需要从原始模型重导出。
`--overwrite-relay` 替换对应 Relay 工件，任务配置变化时还需要
`--overwrite-task-info`，并重新生成、测量调度。

先列出单个网络将执行的变体，再实际运行：

```bash
cd "$LLMC_REPO/LLM"
python benchmark_relay_batch.py \
  --network-info-dir "$LLMC_NET" \
  --history "$LLMC_EVAL_RUN/measured.json" \
  --target "$LLMC_TARGET" \
  --network resnet_50 \
  --max-models 1 \
  --output-dir "$LLMC_WORK/reports/batch_smoke" \
  --dry-run

CUDA_VISIBLE_DEVICES="$LLMC_MEASURE_GPU" \
python benchmark_relay_batch.py \
  --network-info-dir "$LLMC_NET" \
  --history "$LLMC_EVAL_RUN/measured.json" \
  --target "$LLMC_TARGET" \
  --network resnet_50 \
  --include '1,3,224,224' \
  --output-dir "$LLMC_WORK/reports/batch_resnet50" \
  --warmup 5 --number 1 --repeat 10 --min-repeat-ms 500 \
  --resume
```

`--history` 可以重复传入多个**同一 target**的测量文件；`--history-dir` 可读取按 workload
拆分的目录。扩展到全网络时，先生成和测量相应全部 workload，再移除 `--network` /
`--include` 限制。单个 ResNet 的历史不能作为“已完成全网络调优”的证据。

输出：

| 产物 | 查看内容 |
| --- | --- |
| `summary.json` | 整批状态、耗时、处理数量、历史文件 |
| `network_summary.csv` | 按网络家族汇总的结果 |
| `variant_results.csv` | 每个网络 / 输入形状的独立状态和延迟 |
| `network_summaries/` | 各网络的详细 JSON |
| `reports/`、`batch_benchmark.log` | 子任务报告及完整编译 / 运行日志 |

优先报告每个网络的 `valid_geomean_speedup`、有效变体数和失败数。
只在完整覆盖且正确性通过的变体上汇总正式加速比；`missing_report` 要回看子进程日志，
不能当作正常延迟。

`classify_network_speedup.py` 还可比较两份已测调度数据集。下面的自检使用第 9.2 节的拆分结果：

```bash
python classify_network_speedup.py \
  --baseline-path "$LLMC_TRAIN_RUN/measured_by_workload" \
  --lapo-path "$LLMC_TRAIN_RUN/measured_by_workload" \
  --network-info-dir "$LLMC_NET" \
  --output-dir "$LLMC_WORK/reports/classification_selfcheck"
```

上面是同一数据对自身的检查，匹配有效 workload 的比值应为 1。
正式使用时，把 `--lapo-path` 换成新一轮 LAPO 的测量目录。
这里的“baseline”是提供的测量数据集；几何均值和任务加权耗时属于任务级指标，
不等于第 8 节 Relay 默认编译基线或端到端模型延迟。

## 12. 根据历史选择下一轮 workload

```bash
export LLMC_NEXT_RUN="$LLMC_WORK/runs/train_round02"
mkdir -p "$LLMC_NEXT_RUN"
cd "$LLMC_REPO/LLM"

python task_scheduler.py \
  --target "$LLMC_TARGET" \
  --network_info_dir "$LLMC_NET" \
  --to_measure_program_dir "$LLMC_STATES" \
  --history_files "$LLMC_TRAIN_RUN/measured.json" \
  --output_path "$LLMC_NEXT_RUN/scheduler.pkl" \
  --top_k 8 \
  --records_per_task 16 \
  --measurement_budget 20000

python make_dataset.py \
  --for_type for_gen_train_sketch \
  --target "$LLMC_TARGET" \
  --network_info_dir "$LLMC_NET" \
  --to_measure_program_dir "$LLMC_STATES" \
  --measure_record_dir "$LLMC_WORK/measure_records/full" \
  --dataset_path "$LLMC_STATES" \
  --tokenizer_path "$LLMC_LAPO" \
  --save_path "$LLMC_NEXT_RUN/prompts" \
  --schedule_file_path "$LLMC_NEXT_RUN/scheduler.pkl" \
  --keep_cnt 48
```

多轮历史用逗号分隔，按时间从旧到新排列。`measurement_budget` 计入历史尝试数和计划测量数；
预算不足可能没有任务被选中。下一步用第 9.1 节的生成 / 测量命令，
将输入 prompt 和输出改为 `LLMC_NEXT_RUN`，模型改为 `LLMC_LAPO`。

训练 prompt 会继续过滤 hold-out；若调度器选择的文件经过过滤后为空，需要检查选择结果。
可用下面的命令核对所选文件中有多少可进入训练：

```bash
cd "$LLMC_REPO/LLM"
python - <<'PY'
import os
import pickle
from pathlib import Path
import tvm
from common import register_data_path, get_hold_out_five_files

register_data_path(os.environ["LLMC_TARGET"], network_info_folder=os.environ["LLMC_NET"])
target = tvm.target.Target(os.environ["LLMC_TARGET"])
held_out = {Path(path).stem for path in get_hold_out_five_files(target)}
states = {path.stem for path in Path(os.environ["LLMC_STATES"]).glob("*.json")}
with open(Path(os.environ["LLMC_NEXT_RUN"]) / "scheduler.pkl", "rb") as stream:
    selection = pickle.load(stream)["selected_file_stems"]
eligible = (selection & states) - held_out
print("selected:", len(selection), "eligible training workloads:", len(eligible))
assert eligible, "需要调整选择范围或测量预算后重跑调度器"
PY
```

若因为前 8 个任务都被过滤而失败，可将调度命令的 `--top_k 8` 改为 `--top_k 64` 后重新执行，
检查通过再重做 prompt；如果剩余测量预算为零，则需先确定并设置新的预算。
对同一个目标网络继续用其测量数据训练属于该目标的进一步调优，不能再视为未见网络泛化评测。

## 13. 长任务监控、停止与复现记录

在 tmux 中运行时，按 `Ctrl-b` 再按 `d` 可离开会话，任务继续执行：

```bash
tmux ls
tmux attach-session -t llm_compiler
```

在任务终端按 `Ctrl-c` 停止；不要直接删除正在写入的目录。
日志和 GPU 状态可以在另一个终端查看，路径变量需先设置：

```bash
tail -f "$LLMC_WORK/logs/stage1.log"
nvidia-smi
```

生成中断后检查统计与正式候选文件；严格生成失败时旧输出可能仍在，不能将它误认成本次成功结果。
测量可按第 7 节续测，训练按 checkpoint 恢复。

每轮完成后保存实际软件版本和代码提交：

```bash
cd "$LLMC_REPO"
python -m pip freeze > "$LLMC_WORK/reports/python_packages.txt"
git rev-parse HEAD > "$LLMC_WORK/reports/git_commit.txt"
git diff --stat > "$LLMC_WORK/reports/worktree_diff_stat.txt"
nvidia-smi > "$LLMC_WORK/reports/gpu.txt"
```

另行记录模型 / tokenizer 来源、target、输入 shape / dtype、采样种子、计时参数和 GPU
是否独占。Stage1、DPO、LAPO 使用独立运行目录和匹配的训练基线。

## 14. 常见问题

| 现象 | 检查与处理 |
| --- | --- |
| `KeyError: qwen3` / 不支持 Qwen3 | 检查是否误用了旧 Transformers 或 `llm_compiler_collect` 环境 |
| TVM 缺少 `gen_states` / FFI 符号 | 执行第 2.4 节，核对当前仓库 Python 与完整 `libtvm`，必要时重新编译 |
| NumPy / SciPy ABI 错误 | 检查实际导入版本、禁用用户级包，使用兼容的 `numpy<2` 组合 |
| 内置 BERT 出现 SDPA 算子导入错误 | 使用第 2.5 节的网络提取环境；Qwen ONNX 显式指定 eager attention |
| 找不到 `all_tasks.pkl` 或 hold-out `.task.pkl` | 检查路径覆盖与 `for_gen` 对完整内置注册表的要求 |
| JSON 缺少 `i` / `r` 或计算图错误 | 状态目录只放 TVM records；偏好数据先按 workload 拆分 |
| 生成 `shortfall` | 查看解析 / 编译错误、checkpoint 与 tokenizer，调整候选数、重试或模型 |
| `error_no=4` | 开启 `--print-error-details`；区分超时、CUDA launch、分配和正确性错误 |
| `TimeoutError` | 大型输入可能在准备阶段超时；单条记录诊断后适当增加 `--run-timeout` |
| `Memory verification failed` / `ptxas` 错误 | 检查线程绑定、资源约束和 TVM/codegen；超时参数无法解决编译合法性 |
| `refused_incomplete_history` | 按报告补齐有效任务历史；只生成四分之一 workload 不够整图评测 |
| 旧 Relay artifact 缺参数 | 从原模型用 `--overwrite-relay` 重导出，不能从参数长度恢复权重 |
| 训练 loss 下降但模型更慢 | 重新测量相同网络，检查正确性、完整覆盖和匹配的 Stage1 checkpoint |

## 15. 开发与进一步阅读

修改后先运行与变更相关的窄范围检查：

```bash
cd "$LLMC_REPO"
python -m pytest tests/python/unittest/test_target_codegen_llvm.py -q
make pylint
make cpplint
make mypy
```

Python 使用 Black、100 字符行宽；C++ 使用仓库的 `.clang-format`。
C++ 测试需先 `make cpptest`，再执行 `./build/cpptest`。

- [贡献指南](AGENTS.md)
- [Qwen3 训练实现说明](LLM/train/qwen3_train_guide.md)
- [指定 workload 的评测说明](LLM/eval_workload_test_guide.md)
- [ONNX 导出与形状网格](LLM/export_onnx/README.md)
- [Attention 独立比较](LLM/attention/README.md)
- [TVM 源码构建文档](docs/install/from_source.rst)

补充文档中可能保留历史路径与环境示例；执行时以当前脚本接口及本文显式设置的路径为准。
项目包含 Apache TVM 及第三方组件，版权声明见 [NOTICE](NOTICE)，第三方许可见 [licenses/](licenses/)。
