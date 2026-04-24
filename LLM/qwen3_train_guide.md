# Qwen3-0.6B 继续预训练指南（v2 — 对齐最新实现）

> 本版指南配合仓库当前代码：动态填充、`ppt_end` 预计算、`DataCollatorForSeq2Seq`、
> 新的 `run_train_qwen3_clm.py` 默认值、以及新增的 `eval_struct.py`。
> 如果你之前跑过 v1 流程，**数据集需要重建一次**才能享受 O(1) PPT 掩码带来的加速。

---

## 0. v2 变更速览（相对旧版）

| 模块 | 旧实现 | 新实现 |
| --- | --- | --- |
| tokenize | `padding="max_length"` 静态填充到 `max_length=1024` | 只 truncation，训练/评估时由 `DataCollatorForSeq2Seq` 动态填充 |
| PPT 掩码 | 每个 step 在 Python 里对 `input_ids` 做子序列搜索 | 数据集阶段一次性写入 `ppt_end` 列，训练时 O(1) 直接用 |
| 数据集列 | `input_ids / attention_mask / labels` | 额外写 `ppt_end`、`length` 两列 |
| 训练默认 | `per_device_bs=2, grad_accum=4, gradient_checkpointing=True` | `per_device_bs=8, grad_accum=1, gradient_checkpointing=False` |
| DataLoader | `num_workers=4, persistent=False` | `num_workers=8, persistent=True`（可关） |
| 最佳模型指标 | 只看 `eval_accuracy` | 可选 `eval_struct_parse_valid_rate` / `eval_struct_build_valid_rate` |
| 评估 | 无结构性验证 | `eval_struct.py` 独立脚本 + `StructuralEvalCallback` |
| tokenizer | 原生 Qwen3 BPE（`"32"` 切两段、`"SPC"` 切两段） | `extend_tokenizer.py` 扩 TVM 关键字 + 常用整数，`PPT` 升级为 special token，warm-start 子词均值 |

预期收益（4090 × 4、Qwen3-0.6B、bf16）：

- 不扩词表：单 epoch `≈25h → 5–8h`
- 扩词表后：序列长度再减 `30–40%`，同 batch size 吞吐再涨 `1.4–1.8×`，且首步 loss 明显更低

---

## 1. 关键路径

| 用途 | 路径 |
| --- | --- |
| 代码目录 | `/home/qsy/workspace/complier/llm_compiler/LLM` |
| Conda Python | `/home/qsy/anaconda3/envs/tlm/bin/python` |
| Torchrun | `/home/qsy/anaconda3/envs/tlm/bin/torchrun` |
| 基础模型（原始） | `/home/qsy/huggingface/model/Qwen3-0.6B` |
| 扩词表模型（§3 产物） | `/home/qsy/huggingface/model/Qwen3-0.6B-tvm-ext` |
| 原始 measure records | `/home/qsy/workspace/dataset/to_measure_programs/4090` |
| 新数据集（v2） | `/home/qsy/workspace/gen_data/4090_gen_qwen_v2` |
| Stage1 输出 | `/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1-v2` |
| Stage2 输出 | `/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage2-v2` |

下文默认流程是：§3 扩词表 → §4 构建数据集 → §5 smoke → §6/§7 Stage1/2 → §8 结构性验证 → §9 生成。
如果你暂时不想做扩词表，直接跳过 §3，把所有 `Qwen3-0.6B-tvm-ext` 当作 `Qwen3-0.6B` 使用即可。
如果你只想复用旧 `4090_gen_qwen` 数据集，见 §10 迁移说明。
**如果要跑创新点 2 的 DPO / LAPO 偏好训练**（不改动 Stage1/Stage2），见文末 §15。

---

## 2. 环境检查

```bash
cd /home/qsy/workspace/complier/llm_compiler/LLM

/home/qsy/anaconda3/envs/tlm/bin/python - <<'PY'
import transformers, accelerate, datasets, numpy, pyarrow, torch
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("datasets", datasets.__version__)
print("numpy", numpy.__version__)
print("pyarrow", pyarrow.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
```

最低验证过的版本：`transformers>=4.46`、`datasets>=2.19`、`torch>=2.3+cu121`。

---

## 3. 扩词表（`extend_tokenizer.py`，可选但推荐）

### 3.1 为什么要扩

原生 Qwen3 BPE 面向自然语言，对 TVM 调度 token 切分很低效。常见情况：

- `"32"` → `["3", "2"]`
- `"SPC"` → `["S", "PC"]`
- `"CHR"` / `"CHW"` → 3 个 sub-token
- `"1024"` → 2 个 sub-token

对 200 万条平均长度 700–900 的样本来说，这意味着输入长度被放大 `1.5–2×`。
扩词表把调度里最高频的 TVM step 关键字与常用整数一次性加进词表，并用子词均值
warm-start 新 embedding，得到三重收益：

1. **吞吐**：同 batch size 下 step/s 提升约 `1.4–1.8×`。
2. **质量**：首步 loss 不再从随机 embedding 起步，结构对齐更快收敛。
3. **PPT 掩码稳定**：`--promote_marker PPT` 把 `PPT` 升级为 single special token，
   `_find_ppt_end` 退化为 `ids.index(PPT_ID)`，不再受 BPE 合并歧义影响。

### 3.2 默认新增 token 集合

脚本按三组默认值（均可用 `--no_*` 关掉或 `--extra_tokens_file` 追加）：

| 组 | 内容 | 说明 |
| --- | --- | --- |
| step keyword | `SP FU RE CA CI CR CHR CHW RF AN PA FSP FFSP SA PPT MEM SPC ROOT` | 对应 `tvm/auto_scheduler/transform_step.cc` 的 step / marker |
| integer const | `0..9 / 10 / 12 / 14 / 16 / 20 / 24 / 28 / 32 / 40 / 48 / 56 / 64 / 72 / 80 / 96 / 112 / 128 / ... / 4096 / -1` | 调度里常见 tile 尺寸、block/thread 数、循环上限 |
| boolean | `True False` | 许多 annotation 的取值 |

脚本会自动跳过已经是单 token 的字符串（`add_tokens` 否则会是无效操作或 id 冲突）。
所有真正新加的 token 都会在 `extend_tokenizer_manifest.json` 里列出。

### 3.3 命令

```bash
cd /home/qsy/workspace/complier/llm_compiler/LLM

/home/qsy/anaconda3/envs/tlm/bin/python extend_tokenizer.py \
  --base_model /home/qsy/huggingface/model/Qwen3-0.6B \
  --output_dir /home/qsy/huggingface/model/Qwen3-0.6B-tvm-ext \
  --promote_marker PPT \
  --pad_to_multiple_of 8
```

可选参数：

| 参数 | 作用 | 默认 |
| --- | --- | --- |
| `--base_model` | 原始 Qwen3 模型 / tokenizer 目录 | 必填 |
| `--output_dir` | 扩词表后的输出目录 | 必填 |
| `--promote_marker` | 把给定字符串加为 **special token**（推荐 `PPT`） | 空 |
| `--extra_tokens_file` | 追加每行一个 token 的自定义列表文件 | 空 |
| `--no_steps` / `--no_ints` / `--no_bool` | 关掉对应组 | 全开 |
| `--pad_to_multiple_of` | resize 后把 embedding 行数 pad 到该倍数（TensorCore 友好） | `8` |
| `--dry_run` | 只打印将要新增的 token，不写磁盘 | 关 |

首次使用建议先 `--dry_run` 检查实际会新增多少 token，再去掉它正式保存。

### 3.4 产物验证

```bash
ls -lh /home/qsy/huggingface/model/Qwen3-0.6B-tvm-ext

cat /home/qsy/huggingface/model/Qwen3-0.6B-tvm-ext/extend_tokenizer_manifest.json

/home/qsy/anaconda3/envs/tlm/bin/python - <<'PY'
from transformers import AutoTokenizer
old = AutoTokenizer.from_pretrained("/home/qsy/huggingface/model/Qwen3-0.6B", trust_remote_code=True)
new = AutoTokenizer.from_pretrained("/home/qsy/huggingface/model/Qwen3-0.6B-tvm-ext", trust_remote_code=True)
for tok in ["PPT", "SP", "CHR", "CHW", "32", "1024", "True"]:
    print(f"  {tok:>5}  old={old(tok, add_special_tokens=False)['input_ids']}  "
          f"new={new(tok, add_special_tokens=False)['input_ids']}")
print("old_vocab_size =", len(old))
print("new_vocab_size =", len(new))
PY
```

预期输出（数字视具体 BPE 而定，但长度规律一致）：

```
  PPT   old=[47, 50, 51]       new=[151646]
  SP    old=[9851]              new=[9851]          # 已是单 token，未改
  CHR   old=[34, 17781]         new=[151660]
  1024  old=[16, 15, 17, 19]    new=[151693]
  True  old=[2081]               new=[2081]         # 已单 token
...
```

### 3.5 非常重要：扩完必须重建数据集

扩词表后 tokenizer 的 vocab 发生变化，**旧 `4090_gen_qwen*` 数据集里的 `input_ids` 已经
不可用**（旧 id 里不会出现新的 PPT/SP 单 token）。继续用旧数据集训练会导致：

- `ppt_end` 对应的是旧 id 序列中的位置，但 tokenizer 重跑会给出不同序列
- 新 embedding 行（warm-start 过的）永远拿不到梯度

所以扩完必须按 §4 流程重跑一次 `make_dataset.py`，**把 `--tokenizer_path` 指向
`Qwen3-0.6B-tvm-ext`**；产物建议落到新的目录，例如 `4090_gen_qwen_v2_ext`，方便 A/B。

---

## 4. 数据集构建（`make_dataset.py`）

### 4.1 新版产物列清单

每个样本落盘后会包含：

| 列名 | 类型 | 说明 |
| --- | --- | --- |
| `input_ids` | `List[int]` | 原始 token ids，**不再静态 pad** |
| `attention_mask` | `List[int]` | 动态长度，与 `input_ids` 对齐 |
| `labels` | `List[int]` | = `input_ids.copy()`；训练时 PPT 前会被改写为 `-100` |
| `length` | `int` | `len(input_ids)`，`group_by_length=True` 时可加速 batching |
| `ppt_end` | `int` | `PPT` marker 的**最后一个 token 下标**（包含）；找不到为 `-1` |

`ppt_end` 是通过 `tokenizer("PPT", add_special_tokens=False)` 及其带空格的几种变体作为
候选子序列，在每条 `input_ids` 上做一次 KMP 风格搜索得到；训练时直接 `labels[:ppt_end+1] = -100`。

### 4.2 命令：构建 `for_gen`（默认流程）

> **tokenizer 选择**：
> - 做了 §3 扩词表：`--tokenizer_path /home/qsy/huggingface/model/Qwen3-0.6B-tvm-ext`，`--save_path` 建议改成 `4090_gen_qwen_v2_ext`
> - 没做 §3：保留 `--tokenizer_path /home/qsy/huggingface/model/Qwen3-0.6B`
>
> 一旦扩了词表，**旧数据集不可与新模型混用**，务必用对应 tokenizer 重新生成。

```bash
cd /home/qsy/workspace/complier/llm_compiler/LLM

/home/qsy/anaconda3/envs/tlm/bin/python make_dataset.py \
    --for_type for_gen \
    --target nvidia/geforce-rtx-4090 \
    --dataset_path /home/qsy/workspace/dataset/to_measure_programs/4090 \
    --tokenizer_path /home/qsy/huggingface/model/Qwen3-0.6B-tvm-ext \
    --save_path /home/qsy/workspace/gen_data/4090_gen_qwen_v2_ext \
    --max_length 1024 \
    --valid_percentage 5 \
    --min_suffix_tokens 16 \
    --split_seed 0 \
    --ppt_marker PPT
```

完整参数说明：

| CLI 参数 | 作用 | 建议值 |
| --- | --- | --- |
| `--for_type` | 产物类型（`for_gen` / `for_gen_best` / `for_latency` …） | `for_gen` |
| `--target` | 硬件键，会写进 register_data_path | `nvidia/geforce-rtx-4090` |
| `--dataset_path` | 原始 `to_measure_programs/*` 目录 | 同上表 |
| `--tokenizer_path` | 用来 tokenize 的模型目录 | 基础模型路径 |
| `--save_path` | 输出 HuggingFace 数据集目录 | `4090_gen_qwen_v2` |
| `--max_length` | 序列截断上限 | `1024` |
| `--valid_percentage` | 按文件切分的验证集比例 | `5` |
| `--min_suffix_tokens` | `PPT` 之后至少保留多少 token 才保留样本 | `16` |
| `--split_seed` | 按文件切分的随机种子 | `0`（可复现） |
| `--ppt_marker` | 用来定位 decision suffix 的标记 | `PPT`（与训练数据生成保持一致） |

> `--for_type` 可选的其他值在 `make_dataset.py::for_clm_or_mlm` 里可以查到；
> 本指南默认只走 `for_gen`，因为它是目前唯一配套的 CLM 训练流。

### 4.3 产物检查

```bash
ls -lh /home/qsy/workspace/gen_data/4090_gen_qwen_v2_ext
cat /home/qsy/workspace/gen_data/4090_gen_qwen_v2_ext/for_gen_stats.json | head -40
```

`for_gen_stats.json` 里会多出 `ppt_marker` 字段，用于记录这次数据集使用的标记。

再跑一次 sanity check，确认 `ppt_end` 列真的被写入，而且训练可见：

```bash
/home/qsy/anaconda3/envs/tlm/bin/python - <<'PY'
from datasets import load_from_disk
# 扩词表的版本路径：/home/qsy/workspace/gen_data/4090_gen_qwen_v2_ext
# 未扩词表的版本路径：/home/qsy/workspace/gen_data/4090_gen_qwen_v2
ds = load_from_disk("/home/qsy/workspace/gen_data/4090_gen_qwen_v2_ext")
for split in ("train", "validation"):
    d = ds[split]
    print(split, "n=", len(d), "cols=", d.column_names)
    sample = d[0]
    for k, v in sample.items():
        if isinstance(v, list):
            print(f"  {k}: len={len(v)} head={v[:8]}")
        else:
            print(f"  {k}: {v}")
    import random
    idx = random.Random(0).sample(range(len(d)), k=min(2000, len(d)))
    missing = sum(1 for i in idx if d[i]["ppt_end"] < 0)
    print(split, "missing_ppt_in_sample=", missing, "/", len(idx))
PY
```

正常输出中每个 split 的 `cols` 都应包含 `['attention_mask', 'input_ids', 'labels', 'length', 'ppt_end']`，
并且 `missing_ppt_in_sample` 接近 0。

---

## 5. 训练前 smoke test

可靠的两种姿势，任选其一。

### 5.1 直调 `train_qwen3_clm.py`（推荐）

```bash
cd /home/qsy/workspace/complier/llm_compiler/LLM

CUDA_VISIBLE_DEVICES=0 /home/qsy/anaconda3/envs/tlm/bin/python train_qwen3_clm.py \
  --do_train \
  --do_eval \
  --model_name_or_path /home/qsy/huggingface/model/Qwen3-0.6B \
  --tokenizer_name /home/qsy/huggingface/model/Qwen3-0.6B \
  --dataset_name /home/qsy/workspace/gen_data/4090_gen_qwen_v2 \
  --output_dir /home/qsy/huggingface/model/Qwen3-0.6B-smoke-v2 \
  --overwrite_output_dir \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --logging_steps 5 \
  --num_train_epochs 1 \
  --max_steps 10 \
  --max_train_samples 1024 \
  --max_eval_samples 128 \
  --sample_seed 42 \
  --subsample_before_mask True \
  --remove_unused_columns True \
  --learning_rate 1e-5 \
  --warmup_steps 2 \
  --eval_strategy steps \
  --eval_steps 5 \
  --save_strategy steps \
  --save_steps 10 \
  --save_total_limit 1 \
  --dataloader_num_workers 2 \
  --dataloader_persistent_workers True \
  --bf16 True \
  --gradient_checkpointing False \
  --mask_prefix_before_ppt True \
  --drop_samples_without_ppt True \
  --min_suffix_tokens 16 \
  --report_to none
```

预期日志里能看到：

- `PPT mask split=train using precomputed 'ppt_end' column (fast path)`
- `PPT mask split=train before=1024 after=1024 dropped=0`

只要没看到 `falling back to runtime subseq search`，说明新快路径生效。

### 5.2 通过 `run_train_qwen3_clm.py` 的 smoke 模式

```bash
cd /home/qsy/workspace/complier/llm_compiler/LLM

SMOKE_TEST=1 \
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
DATASET_NAME=/home/qsy/workspace/gen_data/4090_gen_qwen_v2 \
OUTPUT_DIR=/home/qsy/huggingface/model/Qwen3-0.6B-smoke-v2-helper \
LOG_FILE=run_train_smoke_v2.log \
SESSION_NAME=qwen3_smoke_v2 \
DELETE_LOG_IF_EXISTS=1 \
/home/qsy/anaconda3/envs/tlm/bin/python run_train_qwen3_clm.py
```

`SMOKE_TEST=1` 会自动把 batch/step 调小，并单卡运行。

---

## 6. Stage1 正式训练

Stage1 目标：**学会在 PPT 之后补全结构合法的 TVM 调度后缀**。

### 6.1 推荐命令（4 × 4090 bf16）

> **扩词表 / 不扩词表对应的三处路径**：
> - `MODEL_NAME_OR_PATH` / `TOKENIZER_NAME`：扩了→`Qwen3-0.6B-tvm-ext`；没扩→`Qwen3-0.6B`
> - `DATASET_NAME`：扩了→`4090_gen_qwen_v2_ext`；没扩→`4090_gen_qwen_v2`
> - `OUTPUT_DIR`：建议带上 `-ext` 后缀以便区分，例如 `-struct-stage1-v2-ext`
>
> 下面以扩词表版本为例；不扩就把 `-tvm-ext` / `_ext` 后缀去掉。

```bash
cd /home/qsy/workspace/complier/llm_compiler/LLM

TRAIN_STAGE=stage1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
MODEL_NAME_OR_PATH=/home/qsy/huggingface/model/Qwen3-0.6B-tvm-ext \
TOKENIZER_NAME=/home/qsy/huggingface/model/Qwen3-0.6B-tvm-ext \
DATASET_NAME=/home/qsy/workspace/gen_data/4090_gen_qwen_v2_ext \
OUTPUT_DIR=/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1-v2-ext \
PER_DEVICE_TRAIN_BATCH_SIZE=8 \
PER_DEVICE_EVAL_BATCH_SIZE=8 \
GRADIENT_ACCUMULATION_STEPS=1 \
LEARNING_RATE=1e-5 \
NUM_TRAIN_EPOCHS=1 \
MIN_SUFFIX_TOKENS=16 \
LOGGING_STEPS=100 \
EVAL_STEPS=2000 \
SAVE_STEPS=2000 \
SAVE_TOTAL_LIMIT=3 \
DATALOADER_NUM_WORKERS=8 \
DATALOADER_PERSISTENT_WORKERS=1 \
GRADIENT_CHECKPOINTING=0 \
GROUP_BY_LENGTH=0 \
REMOVE_UNUSED_COLUMNS=1 \
METRIC_FOR_BEST_MODEL=eval_accuracy \
GREATER_IS_BETTER=1 \
LOAD_BEST_MODEL_AT_END=1 \
LOG_FILE=run_train_stage1_v2_ext.log \
SESSION_NAME=qwen3_stage1_v2_ext \
DELETE_LOG_IF_EXISTS=1 \
/home/qsy/anaconda3/envs/tlm/bin/python run_train_qwen3_clm.py
```

观察日志：

```bash
tail -f /home/qsy/workspace/complier/llm_compiler/LLM/run_train_stage1_v2_ext.log
```

### 6.2 显存不够怎么办（按这个顺序调）

1. `PER_DEVICE_TRAIN_BATCH_SIZE=4` + `GRADIENT_ACCUMULATION_STEPS=2`（等效总 batch 不变）。
2. 还不够：再加 `GRADIENT_CHECKPOINTING=1`（会牺牲约 25% 训练吞吐，但显存省一半以上）。
3. 还不够：`PER_DEVICE_TRAIN_BATCH_SIZE=2 / GRADIENT_ACCUMULATION_STEPS=4 / GRADIENT_CHECKPOINTING=1`。
4. 还不够：减 `NPROC_PER_NODE`，或把 `--max_length` 在数据集阶段降到 `768`。

### 6.3 打开结构性评估（可选，强烈推荐）

当前 dataset 几乎没有真实 latency，纯 `eval_accuracy` 容易误导。建议把"生成的 suffix 能否被
TVM 接受"作为最佳模型指标：

```bash
# 额外叠加这些环境变量到上面的 Stage1 命令即可
STRUCT_EVAL_SKETCH_PATH=/home/qsy/workspace/dataset/sketch/4090/sketch.json \
STRUCT_EVAL_TARGET="cuda -model=4090" \
STRUCT_EVAL_MAX_WORKLOADS=32 \
STRUCT_EVAL_MAX_STATES=2 \
STRUCT_EVAL_MAX_NEW_TOKENS=512 \
STRUCT_EVAL_BATCH_SIZE=8 \
STRUCT_EVAL_DO_BUILD=0 \
STRUCT_EVAL_DO_SAMPLE=0 \
STRUCT_EVAL_SEED=42 \
METRIC_FOR_BEST_MODEL=eval_struct_parse_valid_rate \
GREATER_IS_BETTER=1 \
...（保持其它 Stage1 参数不变）
```

每次 `EVAL_STEPS` 触发 eval 时，会在日志里看到：

```
StructuralEvalCallback metrics: {
  'parse_valid_rate': 0.72, 'fallback_rate': 0.06, 'distinct_rate': 0.94,
  'avg_parse_per_sketch': 1.84, 'num_workloads': 32, 'num_generated': 64,
  'samples_per_sec': 0.45, 'elapsed_sec': 142.8
}
```

metrics 字典里会新增以下 key（最佳模型自动按 `metric_for_best_model` 选取）：

| key | 含义 |
| --- | --- |
| `eval_struct_parse_valid_rate` | 生成 suffix 被 `SketchPolicy.gen_states` 成功解析的比例 |
| `eval_struct_fallback_rate` | 全部解析失败、只能回退 sketch 的 workload 比例 |
| `eval_struct_distinct_rate` | 生成 state 去重后比例（越高越好，避免坍缩） |
| `eval_struct_avg_parse_per_sketch` | 平均每个 workload 成功 parse 的 state 数 |
| `eval_struct_samples_per_sec` / `eval_struct_elapsed_sec` | 本轮评估耗时 |
| `eval_struct_build_valid_rate` | 仅当 `STRUCT_EVAL_DO_BUILD=1`，用 `LocalBuilder` 真实编译成功比例 |
| `eval_struct_num_built` | 同上，进入 builder 的样本数 |

> 注意：Callback **仅 rank 0 执行**；首次 `on_evaluate` 要装载 TVM task registry，
> 会额外花 30–60s。建议把 `EVAL_STEPS` 调到 4000–5000，`STRUCT_EVAL_MAX_WORKLOADS` 控制在 16–32。

---

## 7. Stage2 继续训练

Stage2 目标：在 Stage1 基础上用更小学习率继续训练，稳固结构对齐（**并非**偏好学习，
因为当前数据集仍没有真实 latency）。

```bash
cd /home/qsy/workspace/complier/llm_compiler/LLM

TRAIN_STAGE=stage2 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
MODEL_NAME_OR_PATH=/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1-v2-ext \
TOKENIZER_NAME=/home/qsy/huggingface/model/Qwen3-0.6B-tvm-ext \
DATASET_NAME=/home/qsy/workspace/gen_data/4090_gen_qwen_v2_ext \
OUTPUT_DIR=/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage2-v2-ext \
PER_DEVICE_TRAIN_BATCH_SIZE=8 \
PER_DEVICE_EVAL_BATCH_SIZE=8 \
GRADIENT_ACCUMULATION_STEPS=1 \
LEARNING_RATE=5e-6 \
NUM_TRAIN_EPOCHS=1 \
MIN_SUFFIX_TOKENS=16 \
LOGGING_STEPS=100 \
EVAL_STEPS=2000 \
SAVE_STEPS=2000 \
SAVE_TOTAL_LIMIT=3 \
DATALOADER_NUM_WORKERS=8 \
DATALOADER_PERSISTENT_WORKERS=1 \
GRADIENT_CHECKPOINTING=0 \
LOG_FILE=run_train_stage2_v2_ext.log \
SESSION_NAME=qwen3_stage2_v2_ext \
DELETE_LOG_IF_EXISTS=1 \
/home/qsy/anaconda3/envs/tlm/bin/python run_train_qwen3_clm.py
```

`MODEL_NAME_OR_PATH` 指向不存在时，`run_train_qwen3_clm.py` 会按以下顺序回退：
`Qwen3-0.6B-4090-struct-stage1-v2-ext → Qwen3-0.6B-4090-struct-stage1-v2 → Qwen3-0.6B-4090-struct-stage1 → Qwen3-0.6B-fintuned → Qwen3-0.6B`。
生产训练建议始终显式传入，并确保 base / dataset / tokenizer 的"扩词表状态"三者一致。

---

## 8. 结构性评估脚本 `eval_struct.py`

Stage1 结束后、Stage2 之前，或每个 checkpoint 出来后，都建议单独跑一次。
（`--tokenizer_name` 可以省略；省略时 `eval_struct.py` 会从 checkpoint 自己加载，
只要你训练时用的是扩词表后的 `Qwen3-0.6B-tvm-ext`，checkpoint 就自带扩好的 tokenizer。）

```bash
cd /home/qsy/workspace/complier/llm_compiler/LLM

/home/qsy/anaconda3/envs/tlm/bin/python eval_struct.py \
  --model_name_or_path /home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1-v2-ext \
  --sketch_path /home/qsy/workspace/dataset/sketch/4090/sketch.json \
  --target "cuda -model=4090" \
  --max_workloads 64 \
  --max_states_per_workload 2 \
  --max_new_tokens 512 \
  --batch_size 8 \
  --do_build True \
  --do_sample False \
  --seed 42 \
  --output_json ./eval_struct_stage1_v2_ext.json
```

CLI 参数表：

| 参数 | 作用 | 备注 |
| --- | --- | --- |
| `--model_name_or_path` | 要评估的 checkpoint | 训练输出目录或其中某个 `checkpoint-*` |
| `--tokenizer_name` | 可选，默认与 `model_name_or_path` 同 | checkpoint 自带 tokenizer 时可不填 |
| `--sketch_path` | sketch 记录（与 `gen_state.py` 一致） | 通常是 `sketch.json` 或等价文件 |
| `--target` | 目标硬件 | 如 `"cuda -model=4090"` |
| `--max_workloads` | 采样多少个 workload | 64 常够用，越多越稳 |
| `--max_states_per_workload` | 每个 workload 让模型补全几个 state | `2` |
| `--max_new_tokens` | 单条生成上限（会被 token 长度 scale） | `512` |
| `--batch_size` | 生成 batch size | `8` |
| `--do_build` | 是否在 `LocalBuilder` 上构建 | 推荐 `True`，多花几分钟 |
| `--do_sample` | 是否采样生成；否则贪心 | `False` 可复现 |
| `--seed` | workload 采样种子 | `42` |
| `--output_json` | 报告写入位置 | 最终 metrics 落盘点 |

生成的 JSON 字段与前述 callback metrics 相同。

---

## 9. 下游生成 `gen_state.py`

验证模型真正"能产出调度"的最终步骤：

```bash
cd /home/qsy/workspace/complier/llm_compiler/LLM

CUDA_VISIBLE_DEVICES=0,1,2,3 /home/qsy/anaconda3/envs/tlm/bin/python gen_state.py \
  --model_name_or_path /home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1-v2-ext \
  --sketch_path /home/qsy/workspace/dataset/sketch/4090/sketch.json \
  --save_path /home/qsy/workspace/gen_data/4090_gen_state_stage1_v2_ext.json \
  --keep_cnt 64 \
  --target "cuda -model=4090" \
  --is_build True \
  --allow_repeat False \
  --do_sample True \
  --sample_top_p 1.0 \
  --sample_top_k 0 \
  --sample_temperature 0.6 \
  --generation_batch_size 32 \
  --min_gen_tokens 256 \
  --gen_token_scale 4.0 \
  --trim_last_input_token False \
  --fallback_to_sketch_when_invalid True
```

> 如果做最终对比基准，建议同时改 `--do_sample False` 再跑一遍做贪心基线。

主要参数意义：

| 参数 | 作用 |
| --- | --- |
| `--sketch_path` | 输入 sketch（定义 workload 集合） |
| `--save_path` | 生成 state 的落盘路径 |
| `--keep_cnt` | 每个 workload 保留的 state 数 |
| `--is_build` | 是否真的用 LocalBuilder 做构建 |
| `--allow_repeat` | 是否允许输出重复 state |
| `--do_sample` / `--sample_top_p` / `--sample_top_k` / `--sample_temperature` | 采样超参 |
| `--generation_batch_size` | 生成 batch size（后缀越长建议越小） |
| `--min_gen_tokens` / `--gen_token_scale` | 当 policy 给出的 `max_new_tokens` 太短时的兜底放大 |
| `--trim_last_input_token` | 旧版 SentencePiece 常用，BPE/Qwen 默认 False |
| `--fallback_to_sketch_when_invalid` | 全部无效时退回 sketch state |

---

## 10. 从旧数据集迁移

如果你还想先复用旧的 `/home/qsy/workspace/gen_data/4090_gen_qwen`（**没有 `ppt_end` / `length` 列**）：

- 训练脚本会自动 fallback 到旧逻辑（日志里会打印
  `PPT mask split=... falling back to runtime subseq search`），结果正确但**无加速**。
- 你仍然可以开 `DataCollatorForSeq2Seq` 动态填充，但旧数据集本身是预填充好的，动态 pad 等同于
  no-op，不会带来吞吐提升。
- 因此**强烈建议直接按 §4 重建一次**，收益最大。
- 如果同时做了 §3 扩词表，更是**必须**重建（旧数据集的 `input_ids` 基于旧 tokenizer，
  继续喂给扩了词表的模型会立即 `size mismatch` 或 embedding 错位）。

重建成本参考：2.15M 样本在单机 16 core 上约 `20–30min`。

---

## 11. `run_train_qwen3_clm.py` 全部环境变量（v2）

### 核心

| 环境变量 | 作用 | 默认 |
| --- | --- | --- |
| `TRAIN_STAGE` | `stage1` / `stage2` | `stage1` |
| `SMOKE_TEST` | smoke 模式 | `0` |
| `CUDA_VISIBLE_DEVICES` | 可见 GPU | `0,1,2,3` |
| `NPROC_PER_NODE` | `torchrun` 进程数 | 可见 GPU 数 |
| `MASTER_PORT` | DDP 端口 | `29531` |
| `TORCHRUN_BIN` | torchrun 二进制路径 | `~/anaconda3/envs/tlm/bin/torchrun` |

### 路径

| 环境变量 | 作用 | 默认 |
| --- | --- | --- |
| `MODEL_NAME_OR_PATH` | 初始化模型（扩词表后改成 `Qwen3-0.6B-tvm-ext`） | 按 stage 自动推断 |
| `TOKENIZER_NAME` | tokenizer（应与 `MODEL_NAME_OR_PATH` 的词表一致） | 同上 |
| `DATASET_NAME` | tokenized 数据集（扩词表用 `*_ext` 版本） | `4090_gen_qwen` |
| `OUTPUT_DIR` | 输出 checkpoint 目录 | 推荐显式传 |
| `LOG_FILE` | 日志文件 | `run_train_qwen3_clm_py.log` |
| `SESSION_NAME` | tmux session | 推荐显式传 |
| `DELETE_LOG_IF_EXISTS` | 启动前删旧日志 | `0` |

### 批大小 / 吞吐

| 环境变量 | 作用 | 默认（v2）| smoke 默认 |
| --- | --- | --- | --- |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | 单卡训练 batch | `8` | `2` |
| `PER_DEVICE_EVAL_BATCH_SIZE` | 单卡评估 batch | `8` | `2` |
| `GRADIENT_ACCUMULATION_STEPS` | 梯度累积 | `1` | `1` |
| `DATALOADER_NUM_WORKERS` | dataloader 线程数 | `8` | `0` |
| `DATALOADER_PERSISTENT_WORKERS` | 持久 workers | `1` | `0` |
| `GRADIENT_CHECKPOINTING` | 激活重算 | `0` | `0` |
| `GROUP_BY_LENGTH` | 按长度分 batch（需要 `length` 列） | `0` | `0` |
| `REMOVE_UNUSED_COLUMNS` | 让 Trainer 剔除无关列 | `1` | `1` |

### 调度 / 日志

| 环境变量 | 作用 | 默认 | smoke 默认 |
| --- | --- | --- | --- |
| `LEARNING_RATE` | 初始 LR | Stage1 `1e-5` / Stage2 `5e-6` | 同 |
| `NUM_TRAIN_EPOCHS` | 训练 epoch 数 | `1` | `1` |
| `MAX_STEPS` | 硬截断步数 | 空 | `20` |
| `MAX_TRAIN_SAMPLES` | 限制训练样本 | 空 | `2048` |
| `MAX_EVAL_SAMPLES` | 限制验证样本 | 空 | `256` |
| `MIN_SUFFIX_TOKENS` | 最小后缀 token 数 | `16` | `16` |
| `SAMPLE_SEED` | 抽样种子 | `42` | `42` |
| `LOGGING_STEPS` | log 频率 | `100` | `5` |
| `EVAL_STEPS` | eval 频率 | `2000` | `10` |
| `SAVE_STEPS` | 保存频率 | `2000` | `20` |
| `SAVE_TOTAL_LIMIT` | checkpoint 保留数 | `3` | `1` |
| `WARMUP_RATIO` / `WARMUP_STEPS` | warmup | `0.03` / — | — |
| `LOAD_BEST_MODEL_AT_END` | 训练结束加载最佳 | `1` | `0` |
| `METRIC_FOR_BEST_MODEL` | 最佳模型指标 | `eval_accuracy` | 同 |
| `GREATER_IS_BETTER` | 指标越大越好 | `1` | `1` |
| `OVERWRITE_OUTPUT_DIR` | 覆盖输出目录 | `0` | `1` |

### 结构性评估（可选）

| 环境变量 | 作用 | 默认 |
| --- | --- | --- |
| `STRUCT_EVAL_SKETCH_PATH` | 开启 callback 的开关（设了即启用） | 空 |
| `STRUCT_EVAL_TARGET` | TVM target 字符串 | 空 |
| `STRUCT_EVAL_MAX_WORKLOADS` | 每轮 eval 采样多少 workload | `32` |
| `STRUCT_EVAL_MAX_STATES` | 每个 workload 生成几个 state | `2` |
| `STRUCT_EVAL_MAX_NEW_TOKENS` | 单条生成上限 | `512` |
| `STRUCT_EVAL_BATCH_SIZE` | 生成 batch size | `8` |
| `STRUCT_EVAL_DO_BUILD` | 是否跑 LocalBuilder | `0` |
| `STRUCT_EVAL_DO_SAMPLE` | 是否采样（否则贪心） | `0` |
| `STRUCT_EVAL_SEED` | workload 采样种子 | `42` |

---

## 12. 常用 `train_qwen3_clm.py` CLI 参数（直调时用）

| 参数 | 作用 | 推荐值 |
| --- | --- | --- |
| `--dataset_name` | tokenized 数据集目录 | `4090_gen_qwen_v2` |
| `--mask_prefix_before_ppt` | 只对 PPT 后缀算 loss | `True` |
| `--drop_samples_without_ppt` | 丢弃找不到 PPT 的样本 | `True` |
| `--min_suffix_tokens` | 最小后缀监督长度 | `16` |
| `--subsample_before_mask` | `先抽样再 mask`，smoke 必开 | `True` |
| `--max_train_samples` / `--max_eval_samples` | 限样本数 | smoke 用 |
| `--per_device_train_batch_size` | 单卡训练 batch | 正式训练 `8` |
| `--gradient_accumulation_steps` | 梯度累积 | `1` |
| `--learning_rate` | LR | `1e-5 → 5e-6` |
| `--bf16` | bfloat16 | `True` |
| `--gradient_checkpointing` | 激活重算 | `False`（显存紧张时 `True`） |
| `--dataloader_num_workers` / `--dataloader_persistent_workers` | 提升吞吐 | `8 / True` |
| `--group_by_length` | 按长度分 batch（需 `length` 列） | 可选 `True` |
| `--eval_steps` / `--save_steps` | eval / save 间隔 | 正式训练 `2000` |
| `--load_best_model_at_end` | 训练结束加载最佳 | `True` |
| `--metric_for_best_model` | 最佳指标 | `eval_accuracy` 或 `eval_struct_parse_valid_rate` |

---

## 13. 训练产物

输出目录下会有：

- `checkpoint-*/`：中间 checkpoint（受 `SAVE_TOTAL_LIMIT` 控制）
- `config.json` / `model.safetensors` / `generation_config.json`
- `tokenizer_config.json` / `tokenizer.json` / `special_tokens_map.json`
- `trainer_state.json`：训练曲线、最佳指标、步数
- `train_results.json` / `eval_results.json`：训练 / 验证汇总指标
- `all_results.json`：二者合并

数据目录下还会多出 `train/cache-*.arrow`、`validation/cache-*.arrow`，是 `datasets.map()` 的缓存，属正常现象。

---

## 14. FAQ

### 14.1 `falling back to runtime subseq search` 是什么意思？

旧 dataset 没有 `ppt_end` 列，训练脚本按兼容路径在 Python 里做子序列搜索。结果正确，
只是每个样本都要扫一遍 `input_ids`，Stage1 会因此慢几个小时。按 §4 重建即可变成 fast path。

### 14.2 能不能继续用 `max_length=1024` 的静态填充？

理论上可以（把训练脚本里 `DataCollatorForSeq2Seq` 换回 `default_data_collator`），但：

- 大部分样本长度在 `500–900`，静态 pad 到 `1024` 浪费 10–25% 的算力；
- `ppt_end` 预计算依赖动态长度时更省事；
- 所以**不推荐**回滚。

### 14.3 `group_by_length=True` 要不要开？

- 开之后 batch 内长度差异缩小，吞吐一般再涨 `5–10%`。
- 但会牺牲一些随机性，早期 loss 曲线会更"锯齿"。
- Stage1 建议**先关**，Stage2 稳态期可以试 `GROUP_BY_LENGTH=1`。

### 14.4 `StructuralEvalCallback` 导致训练变慢太多？

把 `EVAL_STEPS` 调大（例如 5000–8000），把 `STRUCT_EVAL_MAX_WORKLOADS` 降到 16，
`STRUCT_EVAL_DO_BUILD=0`。实在不想在训练里跑，就只用 §8 的独立脚本。

### 14.5 多卡训练时 `eval_struct_*` 指标全是 0？

Callback 默认只在 rank 0 计算，其它 rank metrics 里没这个 key。如果你用
`METRIC_FOR_BEST_MODEL=eval_struct_parse_valid_rate`，请在启动命令里确认开启了 Callback 的
`STRUCT_EVAL_*` 环境变量，否则 rank 0 也会找不到指标导致保存失败。

### 14.6 `invalid device ordinal`

`NPROC_PER_NODE` 大于 `CUDA_VISIBLE_DEVICES` 中实际可见 GPU 数。修正后重跑。

### 14.7 扩词表后 checkpoint 加载报 `size mismatch`

这说明你用扩后的数据集 / checkpoint，但 `--model_name_or_path` 还指向原始 `Qwen3-0.6B`。
扩词表会改变 `vocab_size` 与 embedding 行数，必须把 base model 统一切到
`Qwen3-0.6B-tvm-ext`，继续训练也要从 ext checkpoint 出发。

### 14.8 扩词表后新 token embedding 永远不更新

检查 `extend_tokenizer_manifest.json` 里 `warm_started_rows` 是否等于你期望的新增数；
再确保训练脚本传入的 `--tokenizer_name` 也用了 ext 路径（否则 tokenize 阶段还会把
`SP/CHR/PPT` 切成旧 sub-token，新 id 自然永远不被激活）。

### 14.9 Stage2 还是没学到"更优调度"

这是**数据集**决定的，当前数据几乎无有效 latency。继续改进方向：

1. 补测量（让 `run_measure` 跑一批带真实时延的记录）。
2. 或换成 DPO / 偏好学习（见 §15 与 `plan/02_thesis_innovations.md`）。

---

## 15. 创新点 2：DPO / LAPO 偏好训练（独立流水线，不改动 Stage1/Stage2）

本章介绍 `plan/02_thesis_innovations.md` 中的创新点 2。整条流水线**完全与 §6/§7 的 Stage1/Stage2 解耦**：
不修改已有数据集，不改动 CLM 训练脚本，只新增三个脚本——

| 新增文件 | 作用 |
| --- | --- |
| `build_preference_pairs.py` | 把 TVM 测量记录拼成 `(prompt, chosen, rejected, latency_gap, latency_weight)` 的 JSONL |
| `train_qwen3_dpo.py` | 纯 HF Trainer 自实现 DPO / LAPO 损失；通过 `--training_mode` 切换 `sft / dpo / lapo` |
| `run_train_qwen3_dpo.py` | torchrun 包装；用 `USE_DPO` / `TRAINING_MODE` / `LAPO` 环境变量一键切换 |

训练阶段拓扑：

```
Stage1 (CLM) ─┐
              ├──> build_preference_pairs.py ──> Stage3 (DPO / LAPO / SFT-基线)
Stage2 (可选)─┘
```

即：**在 Stage1（或 Stage1 + Stage2）的 checkpoint 之上新增一个独立的偏好训练阶段**，
旧的两个阶段的代码、数据、产物全都原样保留。

### 15.0 前置步骤：为 `measure_records/4090/*.json` 补齐真实 latency

创新点 2 的每一个偏好对都需要 chosen / rejected 的 **真实硬件 latency**。如果直接看
`/data3/qsy/dataset/measure_records/4090/*.json`，里面绝大多数 record 的 `r` 字段仍然是
`[[0], 0, 0, 0, 0]`（TVM 在调度搜索时占位，尚未跑 LocalRunner）。跳过这一步，`build_preference_pairs.py`
会因为无法排序 chosen / rejected 而产出空数据集。

`measure_programs.py` (v2) 新增了目录批量 + 就地补测模式，**不会覆盖已有的有效 latency 行**：

```bash
cd /home/qsy/workspace/complier/llm_compiler/LLM

# 在 GPU 0 上批量补测 4090 目录下所有文件，已经测过的行保留不动，
# 只把 r==[[0]] 的那些行挑出来重跑 LocalRunner，结果追加回原文件。
CUDA_VISIBLE_DEVICES=0 /home/qsy/anaconda3/envs/tlm/bin/python measure_programs.py \
  --target "cuda -model=4090" \
  --input-dir /data3/qsy/dataset/measure_records/4090 \
  --batch-size 128 \
  --resume \
  --drop-hold-out
```

关键行为：

- **就地写回原文件**：内部流程是「先把文件重写成只含已测 record」作为 checkpoint，再调用 TVM
  `RecordToFile`（默认 append 模式）把补测结果追加进同一个文件。中途 Ctrl-C 不会丢失已测数据。
- `--resume`：若某个文件里所有 record 都已有 latency，则整文件跳过（批量模式默认开）。
- `--drop-hold-out`：自动排除 `common.get_hold_out_five_files()` 返回的 hold-out workload，避免偏好对
  训练集污染最终评测集（默认开，显式关闭请用 `--no-drop-hold-out`）。
- `--max-records-per-file N`：调试期限制每文件最多只测 N 行（默认不限）。
- `--start-file-idx / --end-file-idx / --max-files`：按目录排序切片，方便多机多卡并行；例如两张卡各测一半：

  ```bash
  # GPU 0：前一半文件
  CUDA_VISIBLE_DEVICES=0 python3 measure_programs.py --target "cuda -model=4090" \
      --input-dir /data3/qsy/dataset/measure_records/4090 --end-file-idx 100
  # GPU 1：后一半文件
  CUDA_VISIBLE_DEVICES=1 python3 measure_programs.py --target "cuda -model=4090" \
      --input-dir /data3/qsy/dataset/measure_records/4090 --start-file-idx 100
  ```

- `--cuda-visible X`：等价于启动前 `export CUDA_VISIBLE_DEVICES=X`，脚本内部会在 `import tvm` 之前生效。

运行结束后控制台会打印 SUMMARY：

```
========== SUMMARY ==========
  total_files: 237
  fully_measured_skipped: 15
  files_updated: 222
  total_already_measured: 18394
  total_newly_measured: 126801
  total_seconds: 17842.5
```

对旧用法的兼容：仍然支持 `--to-measure-path X --measured-path Y` 单文件模式；加 `--resume` 可在
已有输出文件上续测，不加则保持旧行为（启动时清空 `--measured-path`）。

> v2 修复：移除旧脚本硬编码的 `CUDA_VISIBLE_DEVICES="3"`；删除名存实亡的 `--start-idx / --end-idx /
> --step-idx` 参数；修复 resume 能力缺失。

### 15.1 第一步：构造偏好对数据集

**硬前提**：必须先跑完 §15.0，让 `/data3/qsy/dataset/measure_records/4090/*.json` 中每条记录的
`r[0]` 至少有一个 `>0` 的元素。`build_preference_pairs.py` 会直接读取这个目录。

```bash
cd /home/qsy/workspace/complier/llm_compiler/LLM

/home/qsy/anaconda3/envs/tlm/bin/python build_preference_pairs.py \
  --target "cuda -model=4090" \
  --dataset_path /data3/qsy/dataset/measure_records/4090 \
  --save_path /home/qsy/workspace/gen_data/4090_prefs \
  --min_latency_gap_ratio 1.5 \
  --max_pairs_per_group 4 \
  --max_pairs_per_workload 512 \
  --min_suffix_tokens 8 \
  --max_prompt_tokens 512 \
  --max_suffix_tokens 512 \
  --valid_percentage 5 \
  --split_seed 0 \
  --sample_seed 42 \
  --weight_strategy log_gap \
  --weight_clip 5.0 \
  --drop_hold_out_workloads True \
  --ppt_marker PPT
```

产物：

- `4090_prefs/train_pairs.jsonl`
- `4090_prefs/validation_pairs.jsonl`（`valid_percentage=0` 时没有）
- `4090_prefs/build_stats.json`：过滤 / 采样统计
- `4090_prefs/logs/build_preference_pairs.log`

每条记录字段：

| 字段 | 含义 |
| --- | --- |
| `prompt` | compute DAG + 直到 PPT 为止的 steps 序列化（与 §4 的 text 编码规则完全一致） |
| `chosen` | 同组内 latency 最低的 suffix 序列 |
| `rejected` | 同组内 latency ≥ `min_latency_gap_ratio × chosen` 的较慢 suffix |
| `latency_chosen` / `latency_rejected` | 原始 ms 值 |
| `latency_gap` | `latency_rejected / latency_chosen` |
| `latency_weight` | 由 `--weight_strategy` 计算的 LAPO 权重，≥ 1 |

`weight_strategy` 可选：

| 值 | 公式 | 适用 |
| --- | --- | --- |
| `uniform` | 恒为 1 | 等价于 vanilla DPO |
| `log_gap` | `max(1, log(gap) + 1)` | 推荐首选，抑制极端值 |
| `linear_gap` | `max(1, gap)` | 强调大差距样本 |
| `clipped_linear` | `min(weight_clip, max(1, gap))` | 需要手工截断时使用 |

### 15.2 第二步：训练——一键切换 SFT / DPO / LAPO

最关键的可控参数：**`TRAINING_MODE`**（也可用更直观的 `USE_DPO` / `LAPO`），优先级 `TRAINING_MODE > LAPO > USE_DPO`。

**(a) 标准 DPO 训练（默认）**

```bash
cd /home/qsy/workspace/complier/llm_compiler/LLM

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
PREFERENCE_DATASET=/home/qsy/workspace/gen_data/4090_prefs \
POLICY_MODEL_PATH=/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1-v2-ext \
REF_MODEL_PATH=/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1-v2-ext \
TOKENIZER_NAME=/home/qsy/huggingface/model/Qwen3-0.6B-tvm-ext \
OUTPUT_DIR=/home/qsy/huggingface/model/Qwen3-0.6B-4090-dpo-v2-ext \
TRAINING_MODE=dpo \
BETA=0.1 \
LEARNING_RATE=5e-7 \
NUM_TRAIN_EPOCHS=1 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
GRADIENT_ACCUMULATION_STEPS=2 \
MAX_PROMPT_LENGTH=512 \
MAX_LENGTH=1024 \
LOG_FILE=run_train_dpo_v2_ext.log \
SESSION_NAME=qwen3_dpo_v2_ext \
DELETE_LOG_IF_EXISTS=1 \
/home/qsy/anaconda3/envs/tlm/bin/python run_train_qwen3_dpo.py
```

**(b) 关闭 DPO，只做 SFT 基线（用于消融对比）**

```bash
# 其它环境变量不变，只改一行：
USE_DPO=false \
OUTPUT_DIR=/home/qsy/huggingface/model/Qwen3-0.6B-4090-sft-baseline-v2-ext \
SESSION_NAME=qwen3_sft_baseline_v2_ext \
/home/qsy/anaconda3/envs/tlm/bin/python run_train_qwen3_dpo.py
```

此时脚本会：不加载 ref_model、不做偏好损失，只把 `chosen` suffix 当作监督目标做标准 CLM，相当于一个"在
高质量子集上的 Stage2"基线。论文消融里的 "DPO vs SFT" 两栏就靠这一组命令跑出来。

**(c) LAPO (Latency-Weighted DPO)**

```bash
# 等价于 TRAINING_MODE=lapo
LAPO=true \
BETA=0.1 \
LAPO_WEIGHT_CLIP=10.0 \
LAPO_NORMALIZE=1 \
OUTPUT_DIR=/home/qsy/huggingface/model/Qwen3-0.6B-4090-lapo-v2-ext \
SESSION_NAME=qwen3_lapo_v2_ext \
/home/qsy/anaconda3/envs/tlm/bin/python run_train_qwen3_dpo.py
```

若 `build_preference_pairs.py` 用 `--weight_strategy log_gap`（默认），LAPO 将按
`log(latency_gap)` 对每对样本加权。`LAPO_NORMALIZE=1` 会把一个 batch 内的平均权重归一化成 1，
保持与 DPO 相近的梯度尺度。

### 15.3 `run_train_qwen3_dpo.py` 全部环境变量

#### 模式切换（核心）

| 变量 | 作用 | 默认 |
| --- | --- | --- |
| `TRAINING_MODE` | 显式指定 `sft` / `dpo` / `lapo`；优先级最高 | 空 |
| `LAPO` | 便捷开关；为 true 时覆盖 `USE_DPO` | `false` |
| `USE_DPO` | DPO 总开关；为 false 则退化为 SFT | `true` |
| `BETA` | DPO 温度系数 β | `0.1` |
| `LOSS_TYPE` | `sigmoid` 或 `hinge` | `sigmoid` |
| `LABEL_SMOOTHING` | IPO 风格的 label smoothing | `0.0` |
| `LAPO_WEIGHT_CLIP` | LAPO 权重上限 | `10.0` |
| `LAPO_NORMALIZE` | batch 内权重归一化 | `1` |

#### 路径 & 数据

| 变量 | 作用 | 默认 |
| --- | --- | --- |
| `PREFERENCE_DATASET` | §15.1 产出目录（含 `train_pairs.jsonl`） | `~/workspace/gen_data/4090_prefs` |
| `POLICY_MODEL_PATH` / `MODEL_NAME_OR_PATH` | policy 起点（通常 Stage1 / Stage2 产物） | 自动 fallback 到 `struct-stage2 → struct-stage1 → tvm-ext → base` |
| `REF_MODEL_PATH` | 参考模型；不设则复用 policy | 同 policy |
| `TOKENIZER_NAME` | tokenizer；建议与 policy 一致 | policy |
| `OUTPUT_DIR` | 输出目录；不设则按 `training_mode` 自动命名 | `Qwen3-0.6B-4090-<mode>` |
| `MAX_PROMPT_LENGTH` / `MAX_LENGTH` | 截断长度 | `512 / 1024` |

#### 分布式 / 训练超参

| 变量 | 默认 |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` |
| `NPROC_PER_NODE` | 可见 GPU 数 |
| `MASTER_PORT` | `29541`（与 CLM 的 `29531` 错开） |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | `4`（smoke `1`） |
| `GRADIENT_ACCUMULATION_STEPS` | `2` |
| `LEARNING_RATE` | `5e-7`（DPO 通常比 CLM 小一个量级） |
| `NUM_TRAIN_EPOCHS` | `1` |
| `LOGGING_STEPS / EVAL_STEPS / SAVE_STEPS` | `50 / 500 / 1000` |
| `METRIC_FOR_BEST_MODEL` | `eval_loss` |
| `GREATER_IS_BETTER` | `0`（loss 越小越好） |
| `SMOKE_TEST` | `0`；为 `1` 时自动缩小 batch / step |

其它 `WARMUP_*`、`MAX_STEPS`、`MAX_TRAIN_SAMPLES`、`GRADIENT_CHECKPOINTING`、
`DATALOADER_NUM_WORKERS` 等语义与 §11 的 CLM 版本完全一致，不再赘述。

### 15.4 日志里关注这些字段

训练时每 `LOGGING_STEPS` 步会看到：

```
{'loss': 0.482, 'loss/dpo': 0.482,
 'reward/chosen_mean': 0.0134,  'reward/rejected_mean': -0.0921,
 'reward/margin_mean': 0.1055,  'reward/accuracy': 0.78, ... }
```

| 指标 | 含义 |
| --- | --- |
| `reward/accuracy` | 一个 batch 内 "chosen 的隐式 reward > rejected 的隐式 reward" 的比例；健康训练应从 `0.5` 上升到 `0.8+` |
| `reward/margin_mean` | 平均 margin，逐步变大说明模型在学会"分辨好坏" |
| `loss/sft` | 仅 `TRAINING_MODE=sft` 时出现，是普通 CLM 交叉熵 |

### 15.5 训练后的下游评估

DPO / LAPO checkpoint 的评估 **完全复用 §8 `eval_struct.py` 与 §9 `gen_state.py`**，
把 `--model_name_or_path` 换成 DPO 产物即可。例如：

```bash
/home/qsy/anaconda3/envs/tlm/bin/python eval_struct.py \
  --model_name_or_path /home/qsy/huggingface/model/Qwen3-0.6B-4090-dpo-v2-ext \
  --sketch_path /home/qsy/workspace/dataset/sketch/4090/sketch.json \
  --target "cuda -model=4090" \
  --do_build True \
  --output_json ./eval_struct_dpo.json
```

论文里推荐报告的对比三组：

1. Stage1 checkpoint（baseline，只学合法性）
2. Stage1 + SFT（`USE_DPO=false`，仅在 chosen 上继续训练）
3. Stage1 + DPO 或 Stage1 + LAPO

以 `build_valid_rate`、真实 latency speedup、`reward/accuracy` 三条指标画表。

### 15.6 FAQ

**Q1：ref_model 和 policy 是同一份是否冲突？**
不会。训练前 ref_model 被 `.eval()` 并 `requires_grad_(False)`；在 DDP 下也不会被 wrap。
不过它确实会额外占用 0.6B × 2 字节 ≈ 1.2 GB 显存。4090 完全装得下。

**Q2：我没做 §3 扩词表，这条流水线能跑吗？**
可以。只要把 `TOKENIZER_NAME` 和 `POLICY_MODEL_PATH` 换成原生 `Qwen3-0.6B` / `struct-stage1` 即可；
`build_preference_pairs.py` 与 tokenizer 解耦（它只按空格切分文本，不调用 tokenizer）。

**Q3：偏好对太少怎么办？**
调低 `--min_latency_gap_ratio`（比如 1.2）、调高 `--max_pairs_per_group`；或继续补 LocalRunner 测量。
`build_stats.json` 里 `drop_small_gap` / `groups` 数字可以直接指示哪个卡点最大。

**Q4：训练 loss 不降？**
先确认 `reward/accuracy` 是否 > 0.5。若 < 0.5，说明偏好对构造错了（chosen/rejected 方向搞反）——
检查 `latency_chosen < latency_rejected` 是否成立。其次降低 `BETA`（0.05 / 0.02），或
把 `LEARNING_RATE` 降到 `1e-7`。

**Q5：要不要在 DPO 前先做 Stage2？**
经验上：数据足够时（> 50k 偏好对），跳过 Stage2 直接 Stage1 → DPO 效果更好；
数据少时 Stage1 → Stage2 → DPO 更稳。两条路径都不需要改动 §6/§7 的任何代码。
