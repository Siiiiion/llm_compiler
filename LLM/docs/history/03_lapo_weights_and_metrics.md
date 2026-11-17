# 第三个问题：修复 LAPO 权重与偏好评估统计

修改日期：2026-10-21。

本文对应主计划第 6.2、7.2 节的两个 P0 代码问题。完整序列和 token 边界已在前两步修复，本次让延迟权重在小 batch 下生效，并修正训练、验证指标的聚合方式。

## 1. 已确认的问题

原 LAPO 在每卡当前 microbatch 内执行 `w / mean(w)`。历史配置为每卡 batch=1，因此不论原始延迟权重是多少，归一化后都变成 1。梯度累积发生在单次损失计算之后，无法恢复已消失的权重差异。

原 Trainer 通过 `_last_metrics` 覆盖保存最后一个 batch 的 reward accuracy、reward margin 等指标，再把这些值附加到后续日志。由此既不能代表完整验证集，也容易混入训练和验证的日志。简单平均各 batch 的均值同样不正确，因为最后一个 batch 的样本数可能较少。

## 2. 固定训练集归一化

[train_qwen3_dpo.py](../../train/train_qwen3_dpo.py) 改为：

```text
D_train = 完成 token 编码、整对过滤和 max_train_samples 抽样后的训练集
c_i = min(latency_weight_i, lapo_weight_clip)
normalization_divisor = mean(c_i for i in D_train)
w_i = c_i / normalization_divisor
loss = mean(w_i * dpo_loss_i)
```

- 固定分母只来自实际保留的训练集，不使用验证集，也不随 microbatch、GPU 划分或梯度累积步数改变。
- `lapo_weight_clip=0` 表示关闭上限裁剪；负数、非有限权重或裁剪参数会报错。开启归一化时，训练权重均值必须大于 0。
- `lapo_normalize=False` 保留裁剪，但不除以均值。这个开关仍由原启动器的 `LAPO_NORMALIZE` 控制。
- 验证阶段沿用相同的训练分母。验证集权重均值无需等于 1。
- 显式声明自定义损失不消费 Trainer 的 token-count kwargs，由 Trainer 对梯度累积进行缩放；等大小 microbatch 的累积更新与完整 batch 对齐。

训练开始前，仅由主进程写入 `dpo_manifest.json`，记录归一化协议 `fixed_train_clipped_mean_v1`、实际分母、训练样本数、数据指纹、裁剪统计、tokenizer 路径和边界协议。只做评估时，同样从指定偏好目录的训练文件计算分母；需要提供与训练一致的数据、过滤和抽样参数。

## 3. 按样本汇总指标

- 训练阶段累计日志区间内每条样本的指标总和与数量，在日志输出时跨进程求和，再计算均值。
- 验证阶段先通过 `Accelerator.gather_for_metrics` 收集逐样本指标，由该接口移除分布式最后一批补齐的重复样本，再累加总和与数量。
- 训练、验证使用独立累加器；每轮验证重新初始化，异常退出也清理验证状态。指标分别采用 `train_`、`eval_` 前缀；自定义验证前缀同样生效。
- `eval_loss` 也使用逐样本损失汇总，避免普通 Trainer 重复 batch 平均损失时无法剔除单条补齐样本贡献的问题。
- 只保存各指标的总和与样本数，不缓存完整词表 logits 或全部验证样本。

主要指标：

| 字段 | 含义 |
| --- | --- |
| `eval_loss`、`eval_loss/dpo` | 按样本平均的实际偏好目标；LAPO 中包含延迟权重 |
| `eval_loss/dpo_unweighted` | LAPO 的未加权 DPO 损失，便于判断基础排序目标 |
| `eval_reward/accuracy` | chosen reward 严格大于 rejected reward 的样本比例；相等时记为 0 |
| `eval_reward/margin_mean` | chosen 与 rejected reward 差值的样本均值，可通过原参数关闭 |
| `eval_lapo/weight_mean` | 使用固定训练分母后的验证权重均值 |
| `eval_preference_samples` | 去除分布式补齐样本后的实际评估数量 |

DPO/LAPO 损失不是逐 token 的语言模型交叉熵，因此不再将 `exp(eval_loss)` 标记为 perplexity；SFT 模式保留原有 perplexity 输出。偏好准确率仍不能替代调度生成成功率、编译通过率或实测延迟。

## 4. 现有偏好数据复核

使用实际 Qwen 扩词表 tokenizer，对 `/home/qsy/workspace/dataset/lapo_dataset_v1/train_pairs.jsonl` 重新编码；参数为 `max_prompt_length=512`、`max_length=1024`、追加 EOS、权重裁剪上限 10，未启用训练子集抽样。

| 项目 | 结果 |
| --- | ---: |
| 原始训练 pair 数 | 884 |
| 完整编码过滤后训练 pair 数 | 851 |
| 固定裁剪后训练权重均值 | 2.8058504890325144 |
| 裁剪后最小权重 | 1.4066678586915022 |
| 裁剪后最大权重 | 7.668022259913096 |
| 本配置被权重裁剪的 pair 数 | 0 |

改变长度限制、抽样数量或输入文件后，应由新一轮实际保留的训练数据重新计算，不能把这里的均值硬编码到训练参数中。

## 5. 验证结果

新增 [test_llm_lapo_training.py](../../../tests/python/unittest/test_llm_lapo_training.py) 回归测试，覆盖：

- batch=1 时，相同内容、不同权重的样本产生 1:3 的损失与梯度贡献。
- 先裁剪再计算固定均值、关闭归一化、拒绝非法权重和零分母。
- 验证 batch size 为 1、2、3 时，3 条样本的准确率一致为 2/3，且损失与逐条计算结果相同；重复验证和训练日志互不污染。
- 单个完整 batch 与两个等大小 microbatch 的梯度累积更新一致。
- 双进程 CPU/Gloo：5 条验证样本、每卡 batch=2，补齐后实际计数仍为 5，准确率为 4/5，损失与单条计算均值一致；训练日志的权重均值跨进程聚合为 1。

本次两个回归文件合计 34 项测试全部通过（31.59 秒）。运行方式：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/qsy/anaconda3/envs/tlm/bin/python -m pytest \
  tests/python/unittest/test_llm_schedule_dataset.py \
  tests/python/unittest/test_llm_lapo_training.py -q
```

双进程测试需要本机 socket 通信；本次在允许本机进程通信的执行环境完成验证。

另使用随机初始化的极小 Qwen3，在 CPU 上通过实际命令行入口完成 JSONL 加载、固定分母计算、两步 LAPO 训练、过程中验证、最终验证和模型保存。样例权重 `[1, 3, 100]` 在上限 10 下得到分母 `14/3`；验证样本数为 3，manifest 与模型文件均已写出。临时产物位于 `/tmp/llm_lapo_qwen_smoke-gwtbhg_p`，可随系统清理消失。

## 6. 后续使用

原启动器无需增加参数；`TRAINING_MODE=lapo`、`LAPO_NORMALIZE=true` 即使用固定训练分母。建议使用新的输出目录和原始 Stage1/Stage2 checkpoint 进行对照训练，避免将旧 LAPO 优化器状态与新目标混合；旧模型文件本身不会被该代码自动修正。

本次没有启动全量 GPU 重训，也没有得出调度性能提升结论。下一步仍需扩大实测偏好数据覆盖，并在修复后的训练数据和模型上验证生成、TVM state 恢复、编译及相同搜索预算下的真实延迟。
