# LLM 子项目优化计划（Qwen3-0.6B × TVM auto_scheduler）

> 对应目录：`qusiyuan/projects/llm_compiler/LLM/`
> 生成日期：2026-04-24
> 基础：Stage1 已跑完，`train_runtime ≈ 25h 34min`（4 GPU），`eval_accuracy=0.978`，`eval_loss=0.0532`，`perplexity=1.0547`。

本文档用于后续复现与演进，按「项目现状 → 分层优化建议 → 落地路线图 → 风险提示」的顺序组织。

---

## 1. 项目现状梳理

### 1.1 流水线结构

```text
TVM auto_scheduler 测量记录 (JSON)
        ↓ make_dataset.py  把 MeasureInput 的 steps 序列化成 token
        ↓ 保留 PPT 之前为 prompt，PPT 之后为 decision suffix
HuggingFace datasets (tokenized, 1024 pad)
        ↓ train_qwen3_clm.py  仅对 PPT 之后的 suffix 算 loss
Qwen3-0.6B-4090-struct-stageX
        ↓ gen_state.py  在 SketchPolicy.gen_states 里替换采样策略
TVM measure records（带回 LocalBuilder 验证）
```

### 1.2 核心文件职责

| 文件 | 作用 |
| --- | --- |
| `make_dataset.py` + `make_dataset_utils.py` | 从 `to_measure_programs/4090` 读 MeasureInput，做 SP 归一化 / 去重 / PPT 切分，tokenize 后 `save_to_disk` |
| `train_qwen3_clm.py` | CLM 训练主脚本，带 **PPT suffix-only loss mask** |
| `run_train_qwen3_clm.py` | 包装 `torchrun` 的 DDP 启动器（tmux + tee） |
| `gen_state.py` | 多 GPU 推理：把 LLM 当 `SketchPolicy.gen_states` 的后端 |
| `common.py` | 硬件目录 / 留出任务列表 |
| `task_sheduler.py` | 按「改进空间 × weight」挑值得继续 tune 的任务 |
| `tokenizer.py` | 遗留 WordLevel tokenizer（Qwen3 链路上已不再使用） |
| `evaluate.py` | 实质只是 matplotlib 画图脚本，曾遮蔽第三方 `evaluate` 包 |

### 1.3 当前关键配置摘要

- 数据集：`train=2,156,881`、`validation=111,627`；`max_length=1024`；`min_suffix_tokens=16`
- Stage1：`per_device_bs=2`，`grad_accum=4`，`lr=1e-5`，`bf16=True`，`gradient_checkpointing=True`
- Stage1 指标：`train_loss=0.0739`、`eval_accuracy=0.978`、`perplexity=1.0547`
- 数据特征：**几乎所有 record 缺失有效 latency**（`records_without_latency ≈ 样本总数`），当前训练只能学「合法结构」

### 1.4 最大盲点

`eval_accuracy` 只衡量 next-token 预测，**不衡量「生成的 state 能否被 TVM parser 接受 / 能否 build 成功」**。这是后续所有优化最该补齐的监控维度。

---

## 2. 分层优化建议

### 2.1 数据管线（`make_dataset.py` / `make_dataset_utils.py`）

#### 【P0】`padding="max_length"` 换成动态填充

位置：`make_dataset_utils.py::make_dataset` 第 119–155 行。

- 实际 suffix 远短于 1024，全量 pad 浪费 2–4× FLOPs。
- 改法：tokenize 阶段只 `truncation`，训练阶段用 `DataCollatorForSeq2Seq(pad_to_multiple_of=8)`。

```python
output = tokenizer(examples["text"], truncation=True, max_length=effective_max_length)
output["labels"] = [ids.copy() for ids in output["input_ids"]]
output.pop("token_type_ids", None)
return output
```

`train_qwen3_clm.py`：

```python
from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
```

配合 `--group_by_length=True --length_column_name=length`（需要在 tokenize 时写入 `length` 列），进一步降低 pad 浪费。

#### 【P0】`PPT mask` 在数据构建阶段预计算，训练阶段零扫描

位置：`train_qwen3_clm.py::_apply_ppt_suffix_mask` 第 187–237 行。

- 当前每次训练启动都要对 215 万样本做 Python 子串搜索。
- 改法：在 `make_dataset` 里一次性计算 `ppt_end` 列写进 arrow。训练时直接 `labels[: ppt_end + 1] = -100`。

```python
def tokenize_function(examples):
    out = tokenizer(examples["text"], truncation=True, max_length=effective_max_length)
    ends = []
    L = len(PPT_MARKER_IDS)
    for ids in out["input_ids"]:
        end = -1
        for i in range(len(ids) - L + 1):
            if ids[i:i+L] == PPT_MARKER_IDS:
                end = i + L - 1
                break
        ends.append(end)
    out["ppt_end"] = ends
    out["labels"] = [ids.copy() for ids in out["input_ids"]]
    return out
```

#### 【P1】重新审视 SP tile factor 归一化策略

位置：`make_dataset.py::_normalize_measure_record` 第 128–132 行、`for_gen_basic` 第 217–220 行。

- 当前把所有 SP 的 tile factor 都写 1，等价于让模型看不到 split 大小，导致 `best_by_input` 去重后同一 input 只保留一个 suffix。
- 保守修法：保留 normalize，但确保 `compute_dag.print_min()` 产生的 prompt 里覆盖关键 shape。
- 激进修法：新增 `normalize_sp` 开关，允许保留真实 tile factors，dedup key 只基于 `steps[:ppt_idx]`。

#### 【P2】`for_gen_best::random.shuffle(lines)` 未传 seed，可复现性差

位置：`make_dataset.py` 第 360 行。改成 `random.Random(split_seed).shuffle(lines)`。

#### 【P2】`best_by_input` 在无 latency 时几乎全部丢弃，suffix 多样性不足

位置：`make_dataset.py::for_gen` 第 293–305 行。

- 当前 `existing.latency=None and latency=None` 时 `should_replace=False`，同 input 下只保留第一条。
- 在无 latency 场景，应允许每个 input 最多保留 `k=2~3` 条 suffix（random sample），维持训练集 suffix 多样性。

#### 【P2】多进程池大小 + TVM 反序列化开销

位置：`make_dataset.py::token_files_and_merge` 第 623 行 `Pool(os.cpu_count())`。

- 每个进程都要 import TVM 并 `recover_measure_input(...).task.compute_dag.print_min()`。
- 改法：池大小 `min(os.cpu_count() // 2, 32)`；worker 里缓存 `workload_key -> compute_dag_str`。

---

### 2.2 训练管线（`train_qwen3_clm.py` / `run_train_qwen3_clm.py`）

#### 【P0】0.6B 模型显存严重冗余，吞吐可翻倍

位置：`run_train_qwen3_clm.py` 第 148–150 行。

- 当前 `per_device_bs=2 + grad_accum=4 + gradient_checkpointing=True`，`samples/s=23.4`（4 卡未跑满）。
- 对 0.6B bf16，单卡 24G 可直接 `bs=8~16` 且无需 gradient checkpointing。
- 动作：默认 `PER_DEVICE_TRAIN_BATCH_SIZE=8`、`GRADIENT_ACCUMULATION_STEPS=1`、新增 `GRADIENT_CHECKPOINTING=0` 可关。
- 追加：`--torch_compile=True` + `--optim adamw_torch_fused`。
- **预期：Stage1 从 25h 压到 5–7h。**

#### 【P0】加入结构性评估指标，替代纯 token accuracy

位置：`train_qwen3_clm.py::compute_metrics` 第 404–413 行 + 新增 `TrainerCallback`。

做法：

1. 从 `eval_dataset` 抽 K=64/256 条 prompt（固定 seed，取到 PPT 位置）。
2. 调 `model.generate` 生成 suffix。
3. 拼回完整 input，用 `MeasureInput.from_json + recover_measure_input` 判断是否 parse 通过。
4. 用 `LocalBuilder(timeout=15)` 测 build 成功率。
5. 记录 metrics：`eval_parse_valid_rate`、`eval_build_valid_rate`、`eval_distinct_rate`、`eval_fallback_rate`。

`metric_for_best_model` 换成 `eval_parse_valid_rate` 或 `eval_build_valid_rate`。

#### 【P1】引入 LoRA，把训练成本降一个数量级

位置：`run_train_qwen3_clm.py` + `train_qwen3_clm.py` 新增 `--use_lora`。

- 配置建议：`r=16, alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"]`，dropout=0.05。
- 目标：训练吞吐再翻倍，切换实验成本近乎免费（checkpoint 几十 MB）。

#### 【P1】EarlyStopping 默认生效 + 新指标联动

位置：`train_qwen3_clm.py` 第 431–433 行。

- 默认传 `EARLY_STOPPING_PATIENCE=3`，必须配合新的结构性指标才有意义（token accuracy 为指标时容易误杀）。

#### 【P1】Stage2 默认策略需要重新设计

- 观察：Stage1 token accuracy 已 0.978、perplexity=1.05，再以同数据跑 Stage2 意义不大。
- 短期：`Stage2 MAX_STEPS=5000` 或 `num_train_epochs=0.2~0.3`。
- 长期：等数据侧补齐真实 latency 后，Stage2 切成性能偏好训练（见 `02_thesis_innovations.md` 创新点 2）。

#### 【P2】`run_train_qwen3_clm.py` 的 tmux 包装脆弱

- 建议改成写一个临时 shell 文件，tmux 只执行它；或直接 `subprocess.Popen + nohup + PID 文件`。

#### 【P2】DataLoader 参数

- `dataloader_num_workers=8 or 16`，`dataloader_persistent_workers=True`，`dataloader_prefetch_factor=4`。

---

### 2.3 推理 / 生成管线（`gen_state.py`）

#### 【P0】迁移到 vLLM（或 TensorRT-LLM）

位置：`gen_state.py::gen_func` 第 171–204 行。

- 当前：`batch_size=32`，同 batch pad 到最大 prompt 长度，不同 workload 间填充浪费大；每个 workload 一次 forward，无 continuous batching；`top_k=0 / top_p=1.0` 实际只用 temperature。
- 三档迁移：

**短期（当天见效）：**
- 改采样参数：`top_p=0.95, top_k=50, temperature=0.7, num_return_sequences=keep_cnt`。
- 确保推理时 `model.config.use_cache=True`（当前训练结束时这个值被锁为 False 的风险，见 §2.5）。

**中期（1–2 天）：**
- 整体切 vLLM：`LLM(model=..., dtype='bfloat16', gpu_memory_utilization=0.85, tensor_parallel_size=num_gpus)` + `SamplingParams`，一次把 workload 队列喂进去，让 vLLM 做 continuous batching。Qwen3 原生支持。

**中长期：**
- 接入 **constrained decoding**（`outlines` / `lm-format-enforcer` / 自写 `LogitsProcessor`）把 TVM 合法 token / grammar 作为约束。
- 对「只学合法结构」目标是直接收益，`invalid rate` 可以压到接近 0。

#### 【P0】`is_build=False` 默认值需要重新评估

位置：`gen_state.py` 第 289–304 行。

- 当前默认 `is_build=False`，所有生成的 state 都被视为 build 成功，`keep_cnt` 立即达标，`retry_i<5` 形同虚设。
- 且 `measure_results=[MeasureResult([0.0]...)]` 全是 0 latency，下游无法区分好坏。
- 如果目标是评估生成有效性，**必须默认 `is_build=True`**。

#### 【P1】`worker()` 内资源复用

- `LocalBuilder(timeout=30)` 放到循环外只 new 一次。
- 在 `is_build=False` 分支下不要创建 LocalBuilder。

#### 【P2】`input_to_tokens` 的 import 路径脆弱

位置：`gen_state.py` 第 38 行。

- 当前靠 `sys.path.append(GEN_DIR)` + `from make_dataset import input_to_tokens`，是巧合匹配 `LLM/make_dataset.py`。
- 把 `input_to_tokens` 搬到 `make_dataset_utils.py` 或新 `serialize.py`，作为单一真源。

---

### 2.4 模型 & 词表层（被低估的收益来源）

#### 【P0】Qwen3 BPE 切 TVM token 效率差，扩词表

位置：新增预处理脚本 `extend_tokenizer.py`。

- `32` 会切成 `['3','2']`，`SPC` 会切成 `['S','PC']`，序列长度被放大。
- 把 TVM 高频 token 一次性加进词表并 warm-start 新 embedding：

```python
EXTRA = ["SPC","CI","CR","CA","FU","PA","RF","PPT","MEM","SP"]
EXTRA += [str(i) for i in [1,2,4,8,16,24,32,48,64,96,128,192,256,384,512,1024]]
EXTRA += ["True","False"]
tok.add_tokens(EXTRA, special_tokens=False)
model.resize_token_embeddings(len(tok))
# 用子词均值作为新 embedding 初始化
for new_id, tok_str in enumerate(EXTRA, start=old_vocab_size):
    sub_ids = old_tok(tok_str, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        model.get_input_embeddings().weight[new_id] = model.get_input_embeddings().weight[sub_ids].mean(dim=0)
```

预期：平均序列长度缩短 1.5–2×，训练速度再翻一倍，首步 loss 明显更低。

#### 【P1】PPT 改成 special token

- `<PPT>` 加为 special token，`make_dataset` 手动插入 id。
- `train_qwen3_clm.py::_find_subseq_end` 退化成 `ids.index(PPT_ID)`，O(N)。
- 与 §2.1【P0】的 `ppt_end` 列合并实施。

#### 【P2】双头多任务（未来）

- 等数据侧有真实 latency 后，`AutoModelForCausalLM` 改双头：CLM head + 回归 head（对 PPT 位置 pooled state 预测 latency）。
- 这是通往性能偏好训练的前置步骤。

---

### 2.5 评估与监控体系

新增脚本 `eval_struct.py`（独立，可被 `TrainerCallback.on_evaluate` 调用）：

```text
1. decode -> whitespace tokens
2. 尝试还原 MeasureInput (parse_valid)
3. LocalBuilder 构建 (build_valid, timeout=15s)
4. 若已有 RPC runner，则测真实 latency
5. 写 JSON 到 output_dir/eval_struct.json
```

推荐 KPI（以 Stage1 `eval_accuracy=0.978` 为基线）：

| 指标 | 期望值 |
| --- | --- |
| parse_valid_rate | ≥ 0.95 |
| build_valid_rate | ≥ 0.70 |
| fallback_to_sketch_rate | ≤ 0.15 |
| distinct_suffix_rate | ≥ 0.60 |

达不到即不值得进 Stage2。

---

### 2.6 仓库工程质量

| 问题 | 位置 | 修法 |
| --- | --- | --- |
| `evaluate.py` 实际是画图脚本，且历史上遮蔽过第三方 `evaluate` 包 | `LLM/evaluate.py` | 重命名 `plot_inference_latency.py`；从 git 移除 `bar_chart*.png`、`combined_bar_chart.png` |
| `tokenizer.py` / `FOR_GEN_TOKENIZER` 在 Qwen3 链路上已无用 | `make_dataset.py` 第 684–687 行 | 移除该分支与 `tokenizer.py`，或加 deprecation warning |
| `run_train_stage1.log` 29MB 不应入库 | 根目录 | 加 `LLM/.gitignore`：`*.log`、`__pycache__/`、`.gen_state/`、`bar_chart*.png` |
| `accelerate.Accelerator.unwrap_model` monkey-patch | `train_qwen3_clm.py` 第 19–25 行 | 可接受，但 `requirements.txt` 要绑死版本 |
| 缺 `requirements.txt` | 根目录 | 补一份：`transformers>=4.51,<4.58`、`accelerate>=0.28`、`datasets>=2.10`、`fsspec>=2023.6`、`peft`、`vllm` |
| `_prepare_dataset_output_dir` 会强删整个 save_path | `make_dataset.py` 第 171–180 行 | 加 `--force` 才允许删；或把旧版本 rename 成 `<save_path>.bak-<timestamp>` |
| `gen_state.py::subprocess.run("cat ... > save_path", shell=True)` | 第 417 行 | 换成 Python `shutil.copyfileobj` 合并 |
| `task_sheduler.py` 里 `from utils import get_finetuning_files` 路径隐式 | 第 52 行 | 显式 `from gen.utils import ...` 并处理 sys.path |
| `make_dataset.py::process_file` 的 `print(... end="\r")` 与 logger 冲突 | 第 571 行 | 改成 `tqdm` |

---

## 3. 落地路线图（按 ROI 排序）

| 阶段 | 预计工作量 | 关键动作 | 预期收益 |
| --- | --- | --- | --- |
| **Day 1** | 0.5 天 | 关 `gradient_checkpointing` + `per_device_bs=8` + 动态 pad + `DataCollatorForSeq2Seq` | 训练时间 25h → 6–8h |
| **Day 2** | 1 天 | make_dataset 阶段写 `ppt_end` 列 + `<PPT>` special token + 扩词表 | 序列长度 / 启动时间再减 30–40% |
| **Day 3** | 1 天 | 写 `eval_struct.py` + `TrainerCallback` 注入 `parse_valid_rate / build_valid_rate`，并作为 best_model 准则 | 能真实判断模型是否变好 |
| **Week 1** | 2 天 | `gen_state.py` 迁移到 vLLM + 开启 constrained decoding + `is_build=True` 默认 | 推理吞吐 3–5×，invalid 率显著下降 |
| **Week 2** | 2 天 | 新增 LoRA 训练路径，做一次 A/B；据此决定 Stage2 是否值得跑 | 训练再快 2×，实验迭代成本降低 |
| **Week 2–3** | 3 天 | 修 `SP tile factor` 信息丢失（引入 `normalize_sp` 开关 + shape 注入 prompt），重建数据 | 决策质量本质提升，build_rate 可见改善 |
| **长期** | — | 数据侧接真实 latency（4090 RPC runner）→ 引入偏好学习（DPO / reward model） | 从「学合法」升级到「学更快」 |

---

## 4. 风险 & 容易踩的坑

1. **`train_qwen3_clm.py` 第 335–337 行** 在 `gradient_checkpointing=True` 时把 `model.config.use_cache=False`；这个值会被写进 checkpoint，`gen_state.py` 加载后如果不手动恢复，`generate` 性能会大降。建议：
   ```python
   if training_args.gradient_checkpointing:
       trainer.model.config.use_cache = True
   trainer.save_model()
   ```

2. **`gen_state.py` 里 `tokenizer.model_max_length`** 在 Qwen3 默认是 131072。`available_budget = tokenizer.model_max_length - input_ids.shape[-1]` 值可能巨大，一旦 policy 给出大 `max_new_tokens`，会真的生成上万 token。应追加 clamp：
   ```python
   available_budget = min(available_budget, 2048)
   ```

3. **`make_dataset.py::_prepare_dataset_output_dir`** 会强删整个 save_path（只留 logs/）。误传目录会丢数据，建议默认 dry-run 或 rename 备份。

4. **Qwen3 tokenizer 的 `fix_mistral_regex` 参数**：现在靠 `TypeError` 捕获兜底，启动日志会有噪声。建议探测一次后持久化到 `data_args.fix_mistral_regex`。

5. **数据集里 `records_without_latency ≈ 总样本数`**：在目前数据不变的前提下，任何"性能偏好"训练都不可能生效，必须先补齐真实 latency 测量。

---

## 5. 复现优先动作清单（TL;DR）

直接可以执行的最小落地顺序：

1. `make_dataset_utils.py` 去掉 `padding="max_length"`，改动态 pad。
2. `make_dataset` 阶段写入 `ppt_end` 列。
3. `train_qwen3_clm.py` 换 `DataCollatorForSeq2Seq`，简化 `_apply_ppt_suffix_mask` 为向量化版本。
4. `run_train_qwen3_clm.py` 默认 `per_device_bs=8, grad_accum=1, gradient_checkpointing=0`。
5. 新增 `eval_struct.py` + Callback，把 `parse_valid_rate / build_valid_rate` 作为 `metric_for_best_model`。
6. `extend_tokenizer.py` 一次性扩 TVM token，warm-start 新 embedding。
7. `gen_state.py` 切 vLLM + grammar-constrained decoding，`is_build=True` 作为默认。

做完 1–5，Stage1 训练时间预计从 25.5h 压到 5–7h；做完 6–7，生成有效率与推理吞吐有数量级变化，是后续做毕业论文的地基。
