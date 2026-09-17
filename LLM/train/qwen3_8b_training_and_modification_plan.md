# Qwen3-8B TVM 调度继续预训练与 LAPO 改造计划

> 状态：实施计划，尚不表示相关 QLoRA、reference log-prob 预计算和闭环校验代码已经实现。
>
> 目标模型：`/home/qsy/huggingface/model/Qwen3-8B`
>
> 目标目录：`/home/qsy/workspace/complier/llm_compiler/LLM`
>
> 目标硬件：优先按 `4 × RTX 4090 24 GiB` 设计；实际开跑前必须重新执行 GPU preflight。

## 1. 目标与完成定义

本计划中的“预训练”指基于 TVM schedule 序列的领域继续预训练（continued pretraining / CLM），
不是从随机权重开始训练通用语言模型。训练只监督 `PPT` 后的调度决策 suffix，随后用目标 GPU
上的真实 latency 构造偏好对，执行 LAPO（Latency-Aware Preference Optimization）。

项目只有同时满足以下条件，才算完成 Qwen3-8B 闭环：

1. Qwen3-8B 能以 QLoRA 方式完成 Stage1 CLM smoke、正式训练和断点恢复。
2. 训练产物能被重新加载、生成 TVM schedule，并通过 parse/build 校验。
3. measure records 中的 latency 来自目标硬件真实测量，而不是 `r == [[0]]` 占位值。
4. preference dataset 通过方向、泄漏、长度、权重和覆盖率校验。
5. reference log-prob 可离线预计算，并与在线 reference 前向结果一致。
6. LAPO 只加载一份量化 base/policy 模型，不在每张卡复制第二份 8B reference model。
7. LAPO smoke 的初始 loss、梯度、保存、恢复、adapter 合并均通过校验。
8. Stage1、chosen-only SFT、DPO/LAPO 使用相同起点、数据划分和测量协议完成可比评估。
9. 新模型生成的 schedule 经“生成 → 编译 → 实测 → 构造新偏好 → 再训练”至少完成一轮。
10. 每阶段均产生 manifest、日志、统计 JSON 和 `_SUCCESS` 标记；失败不得伪装成成功产物。

## 2. 当前基线快照（2026-08-19）

以下内容来自当前工作区检查，正式实施时由 preflight 脚本重新采集，不能永久视为不变事实。

### 2.1 模型

- Qwen3-8B 路径：`/home/qsy/huggingface/model/Qwen3-8B`
- safetensors 索引记录的权重总量：`16,381,470,720 bytes`，约 `15.26 GiB`
- 架构：36 层，hidden size 4096，32 个 attention head，8 个 KV head
- dtype：BF16
- vocab size：151936（config）；tokenizer 长度为 151669
- Qwen3-8B 与 Qwen3-0.6B 的原始 `tokenizer.json` 和 `tokenizer_config.json` 哈希一致

### 2.2 CLM 数据

- 现有扩词表数据：`/home/qsy/workspace/gen_data/4090_gen_qwen_v2_ext`
- train：2,169,636 条
- validation：111,611 条
- 列：`input_ids`、`attention_mask`、`labels`、`length`、`ppt_end`
- 占用约 70 GiB
- `for_gen_stats.json` 显示 train/validation 均无 `drop_no_ppt`

### 2.3 LAPO 数据

- 当前目录：`/home/qsy/workspace/dataset/lapo_dataset_v1`
- train pairs：884
- validation pairs：39
- 原始记录 8,238 条中，有 5,803 条因无有效 latency 被丢弃
- 当前偏好数量不足以支撑可信的 8B LAPO 正式结论，只适合功能 smoke 或早期实验

### 2.4 环境与实现限制

- 当前 `tlm` 环境：PyTorch 2.2.0+cu121、Transformers 4.57.6、Accelerate 0.28.0
- 当前环境未安装：`peft`、`bitsandbytes`、`deepspeed`、`trl`
- `run_train_qwen3_clm.py` 当前是 torchrun + DDP，没有参数/优化器分片
- `train_qwen3_dpo.py` 当前会同时加载 policy 和完整 reference model
- 当前沙箱内无法访问 NVIDIA driver；正式 GPU 能力必须在宿主机重新验证

## 3. 技术路线选择

### 3.1 主路线：4-bit QLoRA

4 × 4090 上采用 QLoRA，而不是沿用当前全参数 DDP。原因如下：

- 8B BF16 权重单份约 15.26 GiB。
- 仅 BF16 权重、BF16 梯度和两份 FP32 Adam 状态的理论下限已约 91 GiB/GPU，尚未计入激活、
  master weights、CUDA workspace 和 logits。
- DDP 在每张 GPU 上复制完整训练状态，增加 GPU 数不能解决单卡状态占用。
- 当前 LAPO 还会额外复制一份约 15.26 GiB 的 reference 权重。

QLoRA 初始配置：

| 项目 | 初始值 |
| --- | --- |
| quantization | 4-bit NF4 |
| compute dtype | BF16 |
| double quant | 开启 |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| target modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| train batch/GPU | 1 |
| eval batch/GPU | 1 |
| gradient accumulation | 16 |
| max length | 1024 |
| gradient checkpointing | 开启 |
| use cache | 关闭 |
| optimizer | `paged_adamw_8bit`，不可用时退回 `adamw_torch` 并重新评估显存 |

### 3.2 扩词表策略

优先生成 `/home/qsy/huggingface/model/Qwen3-8B-tvm-ext`，并要求新增 token 的 ID 与现有
`Qwen3-0.6B-tvm-ext` 完全相同。ID 一致时可复用现有 70 GiB 数据集。

QLoRA 冻结 base weights 后，新增 embedding 不能默认获得梯度。实现时采用以下顺序：

1. 优先使用 PEFT 的 `trainable_token_indices`，只训练新增 token 对应的 embedding 行。
2. 若目标 PEFT 版本不支持该能力，不允许静默将整个 embedding/lm_head 加入 `modules_to_save`。
3. fallback A：冻结 warm-start 后的新增 embedding，并在 manifest 中明确记录。
4. fallback B：使用原生 tokenizer 重建非扩词表数据集，再执行 QLoRA。

不允许在未记录的情况下混用原始 tokenizer、扩词表模型和扩词表数据集。

### 3.3 LAPO reference 路线

采用“离线预计算 reference log-prob”，而不是在线加载第二份 8B reference model：

1. 固定 Stage1 checkpoint 作为 policy 起点和 reference lineage。
2. 对每个 chosen/rejected suffix 计算 sequence log-prob。
3. 将 `ref_chosen_logps`、`ref_rejected_logps`、token 数和 reference fingerprint 写入新数据集。
4. LAPO 训练只加载一份 4-bit base + policy adapter。
5. loss 使用在线 policy log-prob 和离线 reference log-prob。

这一设计同时降低显存、缩短训练时间，并固定 reference，便于复现。

## 4. 计划修改的代码

### 4.1 新增 `train/qwen3_peft_utils.py`

集中实现并单元测试：

- `BitsAndBytesConfig` 构造与环境能力检查
- Qwen3 LoRA target module 校验
- `prepare_model_for_kbit_training`
- LoRA adapter 创建、加载、恢复和 trainable parameter 统计
- 新增 token ID 提取与 `trainable_token_indices` 能力探测
- tokenizer/model/dataset fingerprint
- adapter manifest 和 base lineage 校验
- 禁止 `device_map="auto"` 与 torchrun/DDP 混用

验收条件：

- 量化关闭时不要求 bitsandbytes，可在 CPU 测试。
- 量化开启但依赖或 CUDA 不可用时，启动前给出可操作错误。
- trainable 参数只包含 LoRA 和显式允许的 token embedding 行。

### 4.2 修改 `train/train_qwen3_clm.py`

新增参数：

- `--use_peft`
- `--load_in_4bit`
- `--bnb_4bit_quant_type`
- `--bnb_4bit_compute_dtype`
- `--bnb_4bit_use_double_quant`
- `--lora_r`
- `--lora_alpha`
- `--lora_dropout`
- `--lora_target_modules`
- `--train_new_token_embeddings`
- `--adapter_name_or_path`
- `--expected_tokenizer_fingerprint`

行为修改：

- 在量化模型加载后调用统一 PEFT helper。
- gradient checkpointing 时显式执行 input gradient enable 和 `use_cache=False`。
- 启动后打印总参数、可训练参数和比例。
- 保存 adapter、tokenizer、`training_manifest.json`、metrics 和 trainer state。
- 恢复训练时校验 base model、tokenizer、LoRA config 和 dataset fingerprint。
- 不允许把普通 checkpoint 当 adapter checkpoint 恢复，反之亦然。

### 4.3 修改 `train/run_train_qwen3_clm.py`

新增环境变量映射：

- `USE_PEFT=1`
- `LOAD_IN_4BIT=1`
- `LORA_R=32`
- `LORA_ALPHA=64`
- `LORA_DROPOUT=0.05`
- `LORA_TARGET_MODULES=...`
- `TRAIN_NEW_TOKEN_EMBEDDINGS=1`
- `OPTIM=paged_adamw_8bit`
- `RESUME_FROM_CHECKPOINT=...`
- `PREFLIGHT_ONLY=1`

同时修正：

- 训练脚本使用绝对路径，并固定 `cwd=script_dir`。
- 输出启动命令的 JSON 版本，避免只靠不可机器解析的 shell log。
- tmux session 已存在时明确退出，不覆盖另一训练任务。
- 只有 preflight 通过后才创建训练 session。
- 正式训练默认不设置 `OVERWRITE_OUTPUT_DIR=1`。

### 4.4 新增 `train/precompute_reference_logps.py`

输入：

- base model
- reference adapter（Stage1）
- preference dataset
- tokenizer
- max prompt/max total length

输出目录包含：

- `train_pairs_with_ref.jsonl`
- `validation_pairs_with_ref.jsonl`
- `reference_manifest.json`
- `reference_stats.json`
- `_SUCCESS`

每条样本新增：

- `pair_id`
- `ref_chosen_logps`
- `ref_rejected_logps`
- `chosen_label_tokens`
- `rejected_label_tokens`
- `reference_fingerprint`

只有全部样本可解析、数值有限、条数一致并完成抽样在线复核后，才写 `_SUCCESS`。

### 4.5 修改 `train/build_preference_pairs.py`

新增：

- 稳定 `pair_id`，由 workload、prompt、chosen、rejected 内容哈希生成
- `--require_all_measured` 和 `--max_drop_no_latency_ratio`
- chosen/rejected 严格方向校验
- 重复 pair 检测
- train/validation workload 交集检测
- latency 分布和 workload 覆盖统计
- 原始输入文件 checksum 列表

正式训练 gate：

- `latency_chosen < latency_rejected` 必须 100% 成立。
- train/validation workload 交集必须为空。
- latency 必须为有限正数。
- 不得包含 hold-out workload。
- 默认 `drop_no_latency_ratio <= 5%`；超出时只允许 smoke，不允许标记 formal-ready。

### 4.6 修改 `train/train_qwen3_dpo.py`

新增参数：

- 与 CLM 相同的 QLoRA 参数
- `--reference_mode={online,precomputed}`
- `--require_reference_fingerprint`
- `--allow_online_reference_for_smoke`

核心修改：

- 正式 8B LAPO 默认强制 `reference_mode=precomputed`。
- collator 携带离线 reference log-prob。
- DPO/LAPO loss 使用预计算 reference log-ratio。
- 保留 online 模式，仅用于小模型或一致性测试。
- policy 初始状态等于 reference 时，首批 sigmoid DPO loss 应接近 `log(2)`。
- 梯度只能进入 policy adapter，不得进入量化 base 或 reference 字段。
- 保存 `dpo_manifest.json` 时记录 base、Stage1 adapter、reference dataset、preference dataset 的完整 lineage。

### 4.7 修改 `train/run_train_qwen3_dpo.py`

新增环境变量：

- `REFERENCE_MODE=precomputed`
- `REFERENCE_DATASET=...`
- 全部 QLoRA 环境变量
- `PREFLIGHT_ONLY=1`
- `RESUME_FROM_CHECKPOINT=...`

启动前必须检查：

- preference/reference `_SUCCESS` 存在。
- pair 数量和 `pair_id` 一一对应。
- Stage1 adapter fingerprint 与 reference fingerprint 一致。
- tokenizer 与 CLM 数据/偏好数据一致。
- formal 模式的 preference 数量达到设定阈值。

### 4.8 新增 `train/merge_qwen3_adapter.py`

职责：

- 加载 BF16 base model 和指定 adapter。
- 校验 adapter lineage。
- `merge_and_unload()`。
- 保存完整模型、tokenizer 和 `merge_manifest.json`。
- 重新加载合并模型，对固定输入比较 adapter 模式与 merged 模式 logits。
- 通过后写 `_SUCCESS`。

由于合并需要约 16 GiB 以上 CPU 内存和额外磁盘空间，preflight 必须检查资源。

### 4.9 修改 `LLM/eval_struct.py` 与 `LLM/gen_state.py`

两种方式至少支持一种，推荐两者都支持：

- `--adapter_name_or_path`：运行时加载 PEFT adapter。
- 直接加载 `merge_qwen3_adapter.py` 产生的完整 merged model。

评估输出必须记录：

- base/adapter/merged fingerprint
- tokenizer fingerprint
- workload registry fingerprint
- target string
- seed、生成参数和样本数
- parse/build/measure 失败分类

### 4.10 新增 `train/validate_qwen3_8b_pipeline.py`

统一 preflight 和阶段 gate，建议子命令：

```text
env
model
tokenizer
clm-dataset
clm-checkpoint
preference-dataset
reference-dataset
lapo-checkpoint
merged-model
formal-result
```

每个子命令输出 JSON；通过返回 0，失败返回非 0。禁止仅打印 warning 后继续正式训练。

### 4.11 新增测试

建议文件：

- `tests/python/llm/test_qwen3_peft_utils.py`
- `tests/python/llm/test_qwen3_reference_logps.py`
- `tests/python/llm/test_qwen3_lapo_loss.py`
- `tests/python/llm/test_qwen3_pipeline_validation.py`

测试使用 tiny 随机 Qwen3 config 和微型 JSONL，不下载网络模型，不要求 GPU；GPU 集成测试单独标记。

## 5. 环境计划

不要直接污染现有已验证的 `tlm` 环境。克隆环境：

```bash
conda create -n tlm-qwen8b --clone tlm
conda activate tlm-qwen8b
```

安装前先记录环境：

```bash
python -m pip freeze > train/artifacts/qwen3_8b/pre_install_freeze.txt
```

计划安装：

```bash
python -m pip install peft bitsandbytes
```

版本不能只写“latest”；第一次 smoke 通过后，将实际版本冻结到：

```text
LLM/train/requirements-qwen3-8b-lock.txt
```

环境 gate：

- `torch.cuda.is_available()` 为真。
- 4 张卡名称和显存符合预期。
- BF16 可用。
- bitsandbytes 能完成 4-bit tiny model load/forward/backward。
- PEFT 支持计划使用的 LoRA API。
- TVM 能识别 `cuda -model=4090`。

## 6. 数据和 tokenizer 阶段

### 6.1 生成 Qwen3-8B 扩词表模型

```bash
cd /home/qsy/workspace/complier/llm_compiler/LLM

/home/qsy/anaconda3/envs/tlm-qwen8b/bin/python extend_tokenizer.py \
  --base_model /home/qsy/huggingface/model/Qwen3-8B \
  --output_dir /home/qsy/huggingface/model/Qwen3-8B-tvm-ext \
  --promote_marker PPT \
  --pad_to_multiple_of 8
```

注意：`extend_tokenizer.py` 当前会在 CPU 内存加载完整 8B 模型并保存完整副本。执行前检查至少
40 GiB 可用内存和 20 GiB 可用磁盘。

### 6.2 tokenizer parity gate

要求：

- Qwen3-8B-tvm-ext 与 Qwen3-0.6B-tvm-ext 的 token → ID 映射完全一致。
- `PPT` 为单 token。
- model embedding rows 覆盖 tokenizer 全部 ID。
- `extend_tokenizer_manifest.json` 的新增 token 顺序一致。

通过后才允许复用：

```text
/home/qsy/workspace/gen_data/4090_gen_qwen_v2_ext
```

若 parity 失败，使用 Qwen3-8B tokenizer 重建到新目录，禁止覆盖旧数据：

```bash
/home/qsy/anaconda3/envs/tlm-qwen8b/bin/python make_dataset.py \
  --for_type for_gen \
  --target nvidia/geforce-rtx-4090 \
  --dataset_path /home/qsy/workspace/dataset/to_measure_programs/4090 \
  --tokenizer_path /home/qsy/huggingface/model/Qwen3-8B-tvm-ext \
  --save_path /home/qsy/workspace/gen_data/4090_gen_qwen3_8b_ext_v1 \
  --max_length 1024 \
  --valid_percentage 5 \
  --min_suffix_tokens 16 \
  --split_seed 0 \
  --ppt_marker PPT
```

### 6.3 CLM dataset gate

抽样和全量元数据检查：

- 必需列齐全。
- `len(input_ids) == len(attention_mask) == len(labels)`。
- `0 <= ppt_end < len(input_ids)`。
- `ppt_end` 确实对应 `PPT` marker。
- suffix supervised token 数不少于 16。
- 最大 token ID 小于 embedding rows。
- train/validation 源 workload 不交叉。
- dataset fingerprint 写入 manifest。

## 7. Stage1 QLoRA 继续预训练

### 7.1 单卡 10-step smoke

建议参数：

```bash
TRAIN_STAGE=stage1 \
USE_PEFT=1 \
LOAD_IN_4BIT=1 \
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
MODEL_NAME_OR_PATH=/home/qsy/huggingface/model/Qwen3-8B-tvm-ext \
TOKENIZER_NAME=/home/qsy/huggingface/model/Qwen3-8B-tvm-ext \
DATASET_NAME=/home/qsy/workspace/gen_data/4090_gen_qwen_v2_ext \
OUTPUT_DIR=/home/qsy/huggingface/model/Qwen3-8B-4090-stage1-qlora-smoke \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
PER_DEVICE_EVAL_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=2 \
GRADIENT_CHECKPOINTING=1 \
MAX_STEPS=10 \
MAX_TRAIN_SAMPLES=64 \
MAX_EVAL_SAMPLES=16 \
LEARNING_RATE=1e-4 \
OVERWRITE_OUTPUT_DIR=1 \
SESSION_NAME=qwen3_8b_stage1_smoke \
LOG_FILE=train/qwen3_8b_stage1_smoke.log \
/home/qsy/anaconda3/envs/tlm-qwen8b/bin/python train/run_train_qwen3_clm.py
```

通过条件：

- 峰值显存不超过单卡安全阈值，默认保留至少 1.5 GiB 余量。
- 10 步 loss 全部有限，无 NaN/Inf。
- trainable parameter 集合符合 LoRA 配置。
- adapter、trainer state、tokenizer、manifest 均可重新加载。
- 固定样本 reload 前后 logits 一致。

### 7.2 四卡 100-step smoke

使用 `NPROC_PER_NODE=4`、batch/GPU=1、gradient accumulation=16。验证：

- DDP 无 unused parameter/hang。
- global batch 计算正确。
- rank 0 独占保存；其他 rank 不写冲突文件。
- 100 步后可从 checkpoint 恢复并再训练 5 步。
- 恢复后的 global step、scheduler 和 optimizer 状态连续。

### 7.3 正式 Stage1

初始建议：

| 参数 | 值 |
| --- | --- |
| batch/GPU | 1 |
| gradient accumulation | 16 |
| effective batch | 64 |
| learning rate | `1e-4` |
| warmup ratio | 0.03 |
| max length | 1024 |
| max steps | 先 2,000，再按 gate 扩展到 5,000/完整 epoch |
| eval steps | 500 |
| save steps | 500 |
| save total limit | 3 |

不要未经中期评估直接跑完整一轮。完整一轮约为：

```text
2,169,636 / 64 ≈ 33,900 optimizer steps
```

每 500 步执行：

- validation loss/accuracy
- 固定 workload 的 parse-valid smoke
- adapter 可加载检查
- checkpoint manifest 完整性检查

## 8. Stage2 决策

Stage2 不是 LAPO，仅是更小学习率的 CLM 稳固阶段。是否执行由偏好对数量决定：

- formal train pairs `< 10,000`：建议 Stage1 → Stage2 → LAPO。
- formal train pairs `>= 10,000`：并行比较 Stage1 → LAPO 与 Stage1 → Stage2 → LAPO。
- 不允许仅因为文档旧经验就固定使用 50k 门槛；最终以验证曲线决定。

Stage2 必须从 Stage1 的明确 adapter checkpoint 开始，并继承同一 base/tokenizer/dataset lineage。

## 9. 真实 latency 与偏好数据

### 9.1 补测

```bash
cd /home/qsy/workspace/complier/llm_compiler/LLM

CUDA_VISIBLE_DEVICES=0 \
/home/qsy/anaconda3/envs/tlm-qwen8b/bin/python measure_programs.py \
  --target "cuda -model=4090" \
  --input-dir /home/qsy/workspace/dataset/measure_records/4090 \
  --batch-size 128 \
  --resume
```

要求：

- 使用空闲、独占或已记录负载状态的 4090。
- 记录 driver、CUDA、TVM commit、GPU clock/power 状态。
- 长任务运行在 tmux，日志可观察，可安全停止并 `--resume`。
- 不同 GPU 并行分片时，记录每个文件由哪张卡测量。

### 9.2 构造 formal preference dataset

```bash
/home/qsy/anaconda3/envs/tlm-qwen8b/bin/python train/build_preference_pairs.py \
  --target "cuda -model=4090" \
  --dataset_path /home/qsy/workspace/dataset/measure_records/4090 \
  --save_path /home/qsy/workspace/dataset/lapo_dataset_qwen3_8b_v2 \
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

formal-ready gate：

- 无效 latency 比例不超过 5%。
- 至少 10,000 train pairs；不足时明确标为 pilot，而不是 formal。
- 至少 20 个 validation workload，并保证 workload-level split。
- chosen/rejected 方向正确率 100%。
- latency gap 的 P50/P90/max 可解释，没有异常值主导全部权重。
- `log_gap` 权重经过 clip/normalize 后有效样本量不过度坍缩。

## 10. reference log-prob 预计算

执行顺序：

1. preference dataset gate。
2. Stage1 checkpoint gate。
3. 固定 tokenizer、max prompt length、max length、EOS 策略。
4. 预计算 train/validation reference log-prob。
5. 抽取至少 64 对做 online/offline 一致性复核。
6. 全量检查有限值、pair_id 对齐和条数。
7. 写 `_SUCCESS`。

一致性通过条件：

- offline 与 online 使用完全相同的 tokenization 和 label mask。
- sequence log-prob 差异的相对误差小于 `1e-4`；若 BF16 累积使绝对值变大，报告
  token-average log-prob 误差，不允许简单放宽而不解释。
- chosen/rejected 的 label token 数与训练 collator 结果一致。
- reference fingerprint 与 Stage1 adapter fingerprint 一致。

## 11. LAPO QLoRA

### 11.1 损失正确性 gate

在 tiny model 上覆盖：

- policy == reference 时 sigmoid DPO 每样本 loss 为 `log(2)`，约 0.693147。
- uniform weight 的 LAPO 等价于 DPO。
- batch 内归一化后，LAPO weight 均值约为 1。
- clip 生效且不会产生负权重。
- chosen/rejected 交换后 reward margin 符号翻转。
- precomputed 和 online reference 两种模式 loss 一致。

### 11.2 单卡 smoke

初始建议：

```text
training mode          lapo
reference mode         precomputed
batch/GPU              1
gradient accumulation  4
max steps              10
max train samples      64
max eval samples       16
beta                   0.1
learning rate          1e-5
weight clip            5.0
weight normalize       true
```

通过条件：

- 进程只加载一份 4-bit base/policy model。
- 第一批 loss 接近 0.693147，允许 adapter 初始化和数值精度带来小误差。
- 只有 adapter/token embedding 获得梯度。
- reward accuracy、margin、chosen/rejected log-prob 均被记录。
- 保存和重载后可继续训练。

### 11.3 正式实验矩阵

所有实验使用同一个 Stage1 起点：

| 实验 | 数据 | loss | 目的 |
| --- | --- | --- | --- |
| Stage1 | CLM 数据 | suffix CLM | 结构基线 |
| chosen-only SFT | preference chosen | CLM | 排除高质量子集继续训练收益 |
| DPO | preference pairs | uniform DPO | 排除 latency weight 收益 |
| LAPO | preference pairs | weighted DPO | 目标方法 |

固定：

- base model
- Stage1 adapter checkpoint
- tokenizer
- workload split
- train pair IDs
- seed 集合
- generation budget
- measurement protocol

至少运行 3 个 seed；若资源不足只能运行 1 个 seed，报告中必须明确是 preliminary result。

## 12. adapter 合并与下游评估

每个候选 adapter 先做 reload gate，再合并：

```bash
python train/merge_qwen3_adapter.py \
  --base_model /home/qsy/huggingface/model/Qwen3-8B-tvm-ext \
  --adapter_path /home/qsy/huggingface/model/Qwen3-8B-4090-lapo-qlora-v1 \
  --output_dir /home/qsy/huggingface/model/Qwen3-8B-4090-lapo-merged-v1
```

合并 gate：

- adapter 模式与 merged 模式固定样本 logits/生成一致。
- merged model 可由 `AutoModelForCausalLM.from_pretrained` 独立加载。
- tokenizer 和 embedding rows 一致。
- 输出目录包含 manifest 和 `_SUCCESS`。

结构评估：

```bash
/home/qsy/anaconda3/envs/tlm-qwen8b/bin/python eval_struct.py \
  --model_name_or_path /home/qsy/huggingface/model/Qwen3-8B-4090-lapo-merged-v1 \
  --sketch_path /home/qsy/workspace/dataset/sketch/4090/sketch.json \
  --target "cuda -model=4090" \
  --do_build True \
  --output_json train/artifacts/qwen3_8b/eval_struct_lapo.json
```

正式评估至少报告：

- parse valid rate
- build valid rate
- fallback rate
- distinct rate
- compile failure count
- VerifyMemory failure count
- CUDA launch failure count
- valid measured workload count
- latency geomean
- 相对 matched Stage1、SFT、DPO 的 speedup

## 13. 生成—测量—再训练闭环

一轮真正的 LAPO 闭环如下：

```text
Stage1/LAPO adapter
        ↓
gen_state.py 生成候选 schedules
        ↓
parse + build gate
        ↓
measure_programs.py 在目标 4090 实测
        ↓
split_measure_records.py 按 workload 拆分
        ↓
build_preference_pairs.py 构造新偏好
        ↓
precompute_reference_logps.py 固定 reference
        ↓
train_qwen3_dpo.py 进行下一轮 LAPO
```

闭环数据规则：

- 生成记录和实测记录分目录、版本化，不覆盖上一轮。
- 合并训练集前按 `pair_id` 去重。
- hold-out workload 永远不回流训练。
- 新旧 preference pair 保留 source round 字段。
- 每轮固定一个 reference checkpoint；不能训练中途改变 reference。
- 每轮比较新增数据带来的覆盖率、有效 pair 数和性能增量。

建议目录：

```text
/home/qsy/workspace/gen_data/qwen3_8b_lapo_round_0/
/home/qsy/workspace/gen_data/qwen3_8b_lapo_round_1/
/home/qsy/workspace/gen_data/qwen3_8b_lapo_round_2/
```

只有 round N 的生成结果被真实测量并成功形成 round N+1 偏好对，才算完成一次闭环；仅生成 JSON
或仅完成训练不算闭环。

## 14. 分层代码校验闭环

### Gate A：静态校验

```bash
python -m compileall \
  LLM/train/qwen3_peft_utils.py \
  LLM/train/train_qwen3_clm.py \
  LLM/train/train_qwen3_dpo.py \
  LLM/train/precompute_reference_logps.py \
  LLM/train/merge_qwen3_adapter.py \
  LLM/train/validate_qwen3_8b_pipeline.py

python -m black --check \
  LLM/train/qwen3_peft_utils.py \
  LLM/train/train_qwen3_clm.py \
  LLM/train/train_qwen3_dpo.py \
  LLM/train/precompute_reference_logps.py \
  LLM/train/merge_qwen3_adapter.py \
  LLM/train/validate_qwen3_8b_pipeline.py
```

### Gate B：CPU 单元测试

```bash
python -m pytest -q \
  tests/python/llm/test_qwen3_peft_utils.py \
  tests/python/llm/test_qwen3_reference_logps.py \
  tests/python/llm/test_qwen3_lapo_loss.py \
  tests/python/llm/test_qwen3_pipeline_validation.py
```

### Gate C：数据契约

```bash
python LLM/train/validate_qwen3_8b_pipeline.py tokenizer --config ...
python LLM/train/validate_qwen3_8b_pipeline.py clm-dataset --config ...
python LLM/train/validate_qwen3_8b_pipeline.py preference-dataset --config ...
python LLM/train/validate_qwen3_8b_pipeline.py reference-dataset --config ...
```

### Gate D：单卡 GPU smoke

- QLoRA load/forward/backward
- CLM 10 steps
- reference online/offline 一致性
- LAPO 10 steps
- adapter save/reload
- merge/reload

### Gate E：多卡与恢复

- 4 卡 100-step CLM
- checkpoint 恢复 5 步
- 4 卡 100-step LAPO
- LAPO checkpoint 恢复 5 步
- 无 rank 写入冲突和进程 hang

### Gate F：端到端最小闭环

使用少量非 hold-out workloads：

1. Stage1 adapter 生成候选。
2. parse/build。
3. 真实测量。
4. 构造偏好。
5. 预计算 reference。
6. LAPO 训练。
7. 新 adapter 再生成并测量。
8. 产出前后对比 JSON。

### Gate G：正式实验

只有 A–F 全部通过才允许执行。正式结果必须由 `formal-result` validator 检查 lineage、覆盖率、
失败分类和比较协议，成功后写 `_SUCCESS`。

## 15. 失败分类与处置

| 类型 | 典型症状 | 处置 |
| --- | --- | --- |
| 环境 | import/CUDA/bitsandbytes 失败 | 停在 Gate A/C，不启动 tmux |
| tokenizer | ID 不一致、embedding 越界 | 重建数据或修复扩词表，不允许继续 |
| 数据 | 无 PPT、pair 方向错、split 泄漏 | 数据 gate 非 0 退出 |
| 显存 | load/backward OOM | 降长度/rank，检查是否误加载 online reference |
| 数值 | NaN/Inf、首批 loss 异常 | 检查 mask、reference log-prob、BF16/FP32 累积 |
| 分布式 | hang、重复写 checkpoint | 缩小到 2 卡复现，检查 sampler/barrier/rank guard |
| 恢复 | global step 或 scheduler 不连续 | checkpoint 判失败，不进入正式训练 |
| 生成 | parse fallback 高 | 分离模型质量与解码参数问题 |
| 编译 | TVM build/VerifyMemory 失败 | 单独计数，不当作 latency 超时 |
| 运行 | CUDA launch 失败 | 单独分类，不纳入有效 speedup geomean |
| 测量 | GPU 非独占、频率漂移 | 重测并记录 steady-state，不采用 best-of-one |

## 16. Artifact 与 lineage 规范

所有阶段 manifest 至少包含：

- UTC/local timestamp
- git commit 和 dirty files
- Python、PyTorch、CUDA、Transformers、PEFT、bitsandbytes、TVM 版本
- GPU 型号、数量、显存、driver
- base model path/fingerprint
- tokenizer path/fingerprint
- dataset path/fingerprint/stats
- parent checkpoint/adapter fingerprint
- 完整参数
- seed
- 输出文件 checksum
- gate 结果

正式产物禁止复用含义模糊的目录名。建议：

```text
Qwen3-8B-4090-stage1-qlora-r32-v1
Qwen3-8B-4090-stage2-qlora-r32-v1
Qwen3-8B-4090-sft-qlora-r32-v1
Qwen3-8B-4090-dpo-qlora-r32-v1
Qwen3-8B-4090-lapo-qlora-r32-v1
Qwen3-8B-4090-lapo-merged-v1
```

## 17. 实施顺序与退出条件

1. 新建隔离环境并通过 env gate。
2. 实现 PEFT helper 和 CPU 单元测试。
3. 实现 CLM QLoRA，完成单卡/四卡/恢复 smoke。
4. 完成 Stage1 的 2,000-step pilot。
5. 同时补齐 measure records 的真实 latency。
6. 增强 preference builder 并形成 formal-ready 数据。
7. 实现 reference log-prob 预计算和一致性测试。
8. 实现 precomputed-reference LAPO 和 loss 单元测试。
9. 完成 LAPO 单卡/四卡/恢复 smoke。
10. 实现 adapter merge 与 eval/gen adapter 加载。
11. 跑端到端最小闭环。
12. Gate A–F 全通过后，启动正式矩阵。

任一阶段失败时，只修复当前 gate 对应问题并重跑；不得跳过失败 gate，也不得用后续结果反向宣称
前置数据或代码已正确。

## 18. 最终交付清单

- [ ] `qwen3_peft_utils.py`
- [ ] QLoRA 版 `train_qwen3_clm.py` 与 wrapper
- [ ] 增强版 preference builder
- [ ] `precompute_reference_logps.py`
- [ ] precomputed-reference QLoRA LAPO
- [ ] `merge_qwen3_adapter.py`
- [ ] adapter-aware `eval_struct.py` / `gen_state.py`
- [ ] `validate_qwen3_8b_pipeline.py`
- [ ] CPU 单元测试
- [ ] 单卡 GPU smoke 报告
- [ ] 四卡/恢复 smoke 报告
- [ ] 端到端最小闭环报告
- [ ] Stage1/SFT/DPO/LAPO formal manifests
- [ ] workload-level 真实 latency 对比报告
- [ ] 至少一轮生成—测量—再训练闭环产物

在上述清单全部完成前，只能称为“Qwen3-8B 适配进行中”，不能称为完整训练闭环已经验证。
