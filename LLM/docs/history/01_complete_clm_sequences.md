# 第一个问题：修复 CLM 调度样本截断

修改日期：2026-10-08。

本文记录训练序列截断问题的修复；旧总计划已合并到[当前路线图](../ROADMAP.md)。

## 1. 已实施的代码修改

- [make_dataset_utils.py](../../data/make_dataset_utils.py)：CLM 使用完整编码，不再执行 token 截断。对超长样本默认报错；显式选择 drop 时整条丢弃，不能把半条调度放入训练数据。
- [make_dataset.py](../../data/make_dataset.py)：FOR_GEN 默认最大长度从 1024 改为 2048；增加 --overlength_policy，并将完整编码统计纳入 for_gen_stats.json。其他 CLM 训练数据分支也使用同一完整序列检查。
- [rebuild_clm_dataset.py](../../data/rebuild_clm_dataset.py)：新增原始 JSON 重建入口，不依赖 TVM task registry；复用原始 train／validation 文件，保留数据划分，只写新目录。
- [train_qwen3_clm.py](../../train/train_qwen3_clm.py)：加载模型权重前检查序列完整性，防止继续使用已截断或缺少长度来源的 Arrow。训练前移除新增的审计列，保持动态 padding 和 PPT 后缀 mask 兼容。
- [run_train_qwen3_clm.py](../../train/run_train_qwen3_clm.py)：增加 ALLOW_LEGACY_DATASET 显式兼容开关，默认关闭，仅供复现旧训练。

数据加载和编码缓存采用磁盘映射，避免完整长序列数据全部常驻内存。构建目录中的 .tokenize_cache 是可重用的中间缓存，正式数据保存后仍会占用额外磁盘空间。

## 2. 新增数据字段和统计

每条 CLM 样本保存：

| 字段 | 含义 |
| --- | --- |
| original_length | 未截断编码的完整 token 数 |
| length | 保存的 input_ids 长度，必须与 original_length 相同 |
| was_truncated | 必须为 false |
| sequence_complete | 表示保留了完整输入文本的 token，必须为 true |
| ppt_end | `PPT` marker 的最后一个 token；不包含尾随空格，详见 [第二个问题的实施记录](02_schedule_token_boundary.md) |

sequence_complete **不代表 TVM 编译有效或性能优秀**，只证明构建过程没有截掉输入文本。重建时应使用原始完整调度文本，不能先从已截断 Arrow 解码成文本再重建。

tokenization_stats.json 包含：

- 每个 split 的输入、保存、超长和丢弃数量。
- 完整序列、prompt、suffix 的长度直方图，以及 min／max／p50／p95／p99。
- 按 workload 汇总的样本量、超长数量和最大长度。
- 超出上限而没有任何可保留样本的 workload 清单。
- 数据来源、tokenizer 路径、长度上限和处理策略。

选择 error 时，只要存在超长样本，构建就失败并保留审计报告，不写最终 dataset_dict.json。选择 drop 时也拒绝输出空的 split。

## 3. 重建现有数据

从项目根目录执行以下命令。它会创建新的数据目录，不覆盖旧基线。

```bash
/home/qsy/anaconda3/envs/tlm/bin/python LLM/__main__.py rebuild-dataset \
  --train-file /home/qsy/workspace/gen_data/4090_gen_qwen_v2_ext/train_raw.json \
  --validation-file /home/qsy/workspace/gen_data/4090_gen_qwen_v2_ext/validation_raw.json \
  --tokenizer-path /home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1-v2-ext-b2 \
  --output-dir /home/qsy/workspace/gen_data/4090_gen_qwen_v3_len2048 \
  --max-length 2048 \
  --overlength-policy error
```

不要输入旧 Arrow 目录。该入口重新编码的是已有原始文本，不重新划分 workload，也不重新生成 TVM 调度。

若报告发现仍有样本超过 2048，先检查 tokenization_stats.json：可以使用更长上限和另一新目录重建，或明确决定整条丢弃。复杂调度不能仅靠删去末尾来满足上限。

旧 1024-token Arrow 已丢失的尾部无法通过修改 Trainer 参数恢复。新训练需要将 DATASET_NAME 指向重建目录。旧数据仅在显式设置 ALLOW_LEGACY_DATASET=1 时允许复现；已明确标记 was_truncated=true 或 sequence_complete=false 的数据仍会被拒绝。

## 4. 后续继续训练配置

完整长序列会增加单卡显存和计算开销。建议先对新数据做少量样本 smoke，记录峰值显存后再确定 batch。以下是启动参数示例，不代表本轮已运行正式训练：

```bash
cd /home/qsy/workspace/complier/llm_compiler/LLM/train

TRAIN_STAGE=stage2 \
MODEL_NAME_OR_PATH=/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1-v2-ext-b2 \
TOKENIZER_NAME=/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1-v2-ext-b2 \
DATASET_NAME=/home/qsy/workspace/gen_data/4090_gen_qwen_v3_len2048 \
OUTPUT_DIR=/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage2-v3-len2048-smoke \
SMOKE_TEST=1 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
PER_DEVICE_EVAL_BATCH_SIZE=1 \
GRADIENT_CHECKPOINTING=1 \
/home/qsy/anaconda3/envs/tlm/bin/python train/run_train_qwen3_clm.py
```

此 smoke 会使用 launcher 的小样本／短步数设置，不能当作正式收敛实验。正式训练和固定预算的生成／编译／测量对照，需要独立确定配置和输出目录。

## 5. 验证范围

回归测试入口：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/qsy/anaconda3/envs/tlm/bin/python -m pytest \
  tests/python/unittest/test_llm_schedule_dataset.py -q
```

测试覆盖长序列尾部保留、严格上限报错、整条丢弃、长度边界、原始 split 保留、旧数据显式兼容和训练完整性检查。

2026-10-08 验证结果：

- 上述回归测试 17 项全部通过（其中 2 项覆盖第二个问题的 token 边界协议）。
- 使用真实扩词表 tokenizer 重建 2 条训练样本和 2 条验证样本。训练长度为 239／1151，验证长度为 327／1118；重载后 input_ids 和 labels 均与完整原始文本编码逐项相同，保留了尾部 TBS／PRS 决策。相同输入在 1024 上限下被明确拒绝。
- 用微型随机初始化 Qwen3、1094-token 长样本，在 CPU 上分别执行 batch-time PPT mask 和 legacy PPT mask 两条路径；两条路径均完成 1 个训练 step、验证和模型保存。此 smoke 只验证数据与 Trainer 的兼容性，不代表正式模型训练结果。
- 现有 4090_gen_qwen_v2_ext Arrow 因缺少序列完整性来源字段被默认拒绝，错误信息明确指向原始 JSON 重建入口。
- 代码语法、改动的空白检查和文档相对链接检查通过。

真实数据重建验证产物位于 /tmp/llm_complete_sequences_real-dad3rgl2；CPU 训练 smoke 产物位于 /tmp/llm_complete_sequences_train-znj3erpy。这些临时目录均包含 verification.json，CPU 训练目录另有两条路径的日志。

尚未重建全量约 228 万条数据，也未启动正式 GPU 继续训练。本轮代码修复不代表已恢复模型对失败 workload 的生成能力，该结论仍需完整数据重建和继续训练对照。

## 6. 本轮范围

本轮首先修复“训练构建过程截断完整调度”的直接问题。没有同时修改 PPT 空格边界、LAPO 权重、生成策略或 GPU 调度语义。

阶段训练样本拆分属于后续增强：如果完整调度仍然超过合理上下文长度，应按 TVM 的真实中间状态构建阶段样本。当前实现通过明确报错或整条丢弃阻止不完整监督，尚未实现阶段拆分。
