# LLM 辅助 TVM 自动调度的实验计划

## 1. 实验目标

本文实验旨在系统评估大语言模型辅助张量程序自动调度的有效性。围绕 TVM AutoScheduler 与 Qwen3-0.6B 的结合，实验需要回答以下问题：

1. 基于 TVM 原生测量记录的结构化序列表示，是否能够被大语言模型有效学习。
2. 以 `PPT` 为边界的调度草图补全任务，是否能够生成 TVM 可解析、可构建的调度状态。
3. 与传统 Ansor/AutoScheduler 搜索相比，LLM 生成候选 schedule 是否能降低搜索成本，并在真实硬件上获得有竞争力的推理性能。
4. 基于真实 latency 的偏好优化是否能进一步提升生成 schedule 的性能倾向。
5. 所提出方法是否能够在不同类型 workload 之间保持泛化能力。

因此，实验设计分为三条主线：第一条验证结构化调度表示与模型训练有效性；第二条验证生成 schedule 的结构合法性与端到端性能；第三条验证真实硬件反馈驱动的偏好优化效果。

## 2. 当前实验基础

仓库中已经具备完整的实验流水线：

| 环节 | 脚本 | 作用 |
| --- | --- | --- |
| 数据构建 | `LLM/make_dataset.py` | 将 TVM measure records 转换为 Qwen3 训练数据，支持 `PPT` 切分与 hold-out workload |
| 领域词表扩展 | `LLM/extend_tokenizer.py` | 将 TVM step token 与常用整数加入 tokenizer，并 warm-start embedding |
| 结构监督训练 | `LLM/train_qwen3_clm.py` | 对 `PPT` 后缀进行 causal LM 训练 |
| 训练启动封装 | `LLM/run_train_qwen3_clm.py` | 启动 Stage1/Stage2 训练 |
| 结构合法性评估 | `LLM/eval_struct.py` | 评估 parse/build/fallback/distinct 等结构指标 |
| schedule 生成 | `LLM/gen_state.py` | 将模型接入 `SketchPolicy.gen_states` 生成 TVM State |
| latency 测量 | `LLM/measure_programs.py` | 使用 LocalRunner 补测候选 schedule 的真实 latency |
| 端到端编译评估 | `LLM/tune_relay.py`, `LLM/baseline.py` | 在 Relay 模型上应用 history best 并测量推理延迟 |
| 偏好对构建 | `LLM/build_preference_pairs.py` | 从真实 latency records 构造 `(prompt, chosen, rejected)` 偏好对 |
| DPO/LAPO 训练 | `LLM/train_qwen3_dpo.py`, `LLM/run_train_qwen3_dpo.py` | 进行 SFT、DPO 或 Latency-Aware DPO 训练 |

截至当前梳理，`qwen_4090_gen` 训练集约包含 2,156,881 条样本，验证集约包含 111,627 条样本。Stage1 模型已经完成一轮训练，验证集 token accuracy 约为 0.978，perplexity 约为 1.055。需要注意的是，当前大规模训练数据中的 `r` 字段大多仍为占位 latency，因此 Stage1 主要学习结构合法性，而不是性能偏好。

现有 hold-out 评估目录已经覆盖多个 workload：

| Workload | `0_merge.json` 样本数 | `gen_eval.json` 样本数 | 说明 |
| --- | ---: | ---: | --- |
| `bert_base` | 576 | 278 | Transformer 类 workload |
| `bert_large` | 576 | 288 | 更大规模 Transformer |
| `resnet_50` | 1728 | 864 | CNN 基准网络 |
| `mobilenet_v2` | 2048 | 1024 | 轻量 CNN |
| `mobilenet_v3` | 3328 | 1664 | 轻量 CNN |
| `resnext_50` | 1728 | 864 | 分组卷积网络 |
| `vgg_16` | 1152 | 576 | 经典 CNN |
| `densenet_121` | 4608 | 2304 | 密集连接 CNN |
| `wide_resnet_50` | 1728 | 864 | 宽残差网络 |

这些文件为结构合法性评估、生成消融和真实 latency 测量提供了直接的数据基础。

## 3. 实验环境与统一设置

所有主要实验建议统一在 RTX 4090 平台上完成，目标字符串使用 `nvidia/geforce-rtx-4090` 或等价 CUDA target。为保证结果可复现，所有实验应记录以下信息：

| 项目 | 建议记录内容 |
| --- | --- |
| 硬件 | GPU 型号、显存容量、CUDA 驱动版本 |
| 编译器 | TVM commit 或仓库版本 |
| 模型 | Qwen3-0.6B 原始模型、Stage1/Stage2/DPO checkpoint 路径 |
| tokenizer | 是否使用 TVM 扩词表，扩展 token 数量 |
| 数据集 | 训练集路径、验证集路径、hold-out workload 列表 |
| 生成参数 | `keep_cnt`, `temperature`, `top_p`, `top_k`, `min_gen_tokens`, `gen_token_scale` |
| 测量参数 | `repeat`, `number`, `min_repeat_ms`, `batch_size`, GPU 独占情况 |

所有 latency 测量应尽量在独占 GPU 环境下执行，避免后台进程造成测量抖动。每个 workload 的最终结果建议报告均值、中位数以及 geometric mean speedup。

## 4. 数据划分与评估 Workload

训练数据使用 `make_dataset.py` 中的 hold-out 排除机制，避免评估 workload 泄漏到训练集中。评估 workload 按模型结构划分如下：

| 类别 | Workload |
| --- | --- |
| Transformer | `bert_base`, `bert_large` |
| 常规 CNN | `resnet_50`, `vgg_16` |
| 轻量 CNN | `mobilenet_v2`, `mobilenet_v3` |
| 复杂连接 CNN | `resnext_50`, `densenet_121`, `wide_resnet_50` |

论文主表建议至少覆盖 `bert_base`、`bert_large`、`resnet_50`、`mobilenet_v2`、`densenet_121` 五类代表性 workload。若时间允许，应报告全部已生成的 hold-out workload。

## 5. 实验一：结构化调度表示与训练有效性

### 5.1 实验目的

该实验验证 TVM 原生 measure record 序列是否适合作为大语言模型的训练输入，并评估 `PPT` 后缀监督、动态 padding 和领域 tokenizer 对训练质量与效率的影响。

### 5.2 对照组

| 组别 | 描述 |
| --- | --- |
| Full-CLM | 不使用 `PPT` mask，整条调度序列均参与 loss |
| PPT-Suffix CLM | 使用 `PPT` 之前 mask，只监督调度后缀 |
| PPT + Dynamic Padding | 当前新版训练方式，训练时动态 padding |
| PPT + Dynamic Padding + Extended Tokenizer | 使用 `extend_tokenizer.py` 后重建数据集并训练 |

### 5.3 评价指标

| 指标 | 含义 |
| --- | --- |
| `eval_loss` | 验证集语言建模损失 |
| `eval_accuracy` | 后缀 token 预测准确率 |
| `perplexity` | 语言建模困惑度 |
| 平均 token 长度 | tokenizer 对 TVM DSL 的编码效率 |
| `train_samples_per_second` | 训练吞吐 |
| 总训练时长 | 训练成本 |
| `parse_valid_rate` | 生成后缀能否被 TVM 解析 |

### 5.4 论文中预期结论

若 PPT 后缀监督在结构合法性指标上优于 Full-CLM，则说明调度草图补全比普通序列复现更贴合自动调度任务。若扩词表组显著降低平均 token 长度并提升训练吞吐，则可证明领域 tokenizer 对编译器 DSL 建模具有必要性。

## 6. 实验二：结构合法性与生成质量评估

### 6.1 实验目的

该实验评估模型生成的 schedule suffix 是否能被 TVM `SketchPolicy.gen_states` 成功解析，并进一步通过 LocalBuilder 构建。

### 6.2 对照组

| 方法 | 描述 |
| --- | --- |
| Base Qwen3-0.6B | 未经过 TVM 数据训练的基础模型 |
| Stage1 | 经过 `PPT` 后缀监督训练的模型 |
| Stage2 | 在 Stage1 基础上继续训练的模型 |
| Stage1 + Extended Tokenizer | 使用 TVM 扩词表训练的模型 |
| Stage1 + 不同采样策略 | 改变 temperature/top-p/keep_cnt 的生成设置 |

### 6.3 指标

| 指标 | 说明 |
| --- | --- |
| `parse_valid_rate` | 生成后缀被 `SketchPolicy.gen_states` 成功解析的比例 |
| `build_valid_rate` | LocalBuilder 构建成功的比例 |
| `fallback_rate` | 全部生成失败而回退到原 sketch 的 workload 比例 |
| `distinct_rate` | 去重后候选 state 数量占比 |
| `avg_parse_per_sketch` | 每个 sketch 平均可解析候选数 |
| `samples_per_sec` | 结构评估吞吐 |

### 6.4 执行建议

先运行不带 build 的快速评估，再对代表性 workload 运行 `--do_build=True`：

```bash
cd /home/qsy/workspace/complier/llm_compiler/LLM

/home/qsy/anaconda3/envs/tlm/bin/python eval_struct.py \
  --model_name_or_path /home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1 \
  --sketch_path /home/qsy/workspace/gen_data/4090_gen_eval_only_bert/0_merge.json \
  --target nvidia/geforce-rtx-4090 \
  --max_workloads 64 \
  --max_states_per_workload 2 \
  --max_new_tokens 512 \
  --output_json /home/qsy/workspace/gen_data/4090_gen_eval_only_bert/eval_struct_stage1.json
```

### 6.5 论文中预期结论

该实验用于证明模型生成的并非普通文本，而是能够重新进入 TVM 编译流程的结构化调度状态。相比 token accuracy，`parse_valid_rate` 和 `build_valid_rate` 更能反映模型在编译器任务中的真实有效性。

## 7. 实验三：采样策略与候选规模消融

### 7.1 实验目的

该实验研究生成候选数量和采样参数对 schedule 合法性、多样性和最佳性能的影响。

### 7.2 消融变量

| 变量 | 建议取值 |
| --- | --- |
| `keep_cnt` | 8, 16, 32, 64 |
| `temperature` | 0.2, 0.6, 1.0 |
| `top_p` | 0.8, 0.95, 1.0 |
| `top_k` | 0, 50 |
| `min_gen_tokens` | 128, 256, 512 |
| `gen_token_scale` | 2.0, 4.0 |
| `allow_repeat` | True, False |

### 7.3 指标

| 指标 | 说明 |
| --- | --- |
| 有效候选数量 | parse/build 成功的 schedule 数 |
| 去重候选数量 | 衡量多样性 |
| invalid/fallback 次数 | 生成稳定性 |
| best-of-K latency | 候选集合中真实测量最优 latency |
| 生成时间 | 生成成本 |

### 7.4 论文中预期结论

该实验用于解释主实验中参数选择的合理性。例如，若 `keep_cnt=32` 在生成成本和 best-of-K latency 之间达到较好平衡，则可将其作为后续端到端性能实验的默认设置。

## 8. 实验四：真实硬件 latency 与端到端推理性能

### 8.1 实验目的

该实验验证 LLM 生成的 schedule 在真实 RTX 4090 硬件上的性能收益，是论文最核心的性能实验。

### 8.2 对照组

| 方法 | 描述 |
| --- | --- |
| Relay Default | 不使用 AutoScheduler history best |
| Ansor-64 | Ansor 搜索 64 trials |
| Ansor-256 | Ansor 搜索 256 trials |
| Ansor-1024 | Ansor 搜索 1024 trials |
| LLM-only Top-K | 仅测量 LLM 生成的 K 个候选 |
| LLM Best-of-K | 从 LLM 生成候选中选择真实 latency 最小者 |
| LLM + Ansor Same Budget | LLM 生成加少量 Ansor 搜索，总时间与 Ansor 对齐 |

### 8.3 指标

| 指标 | 说明 |
| --- | --- |
| kernel best latency | `measure_programs.py` 输出的最优算子 latency |
| end-to-end inference latency | `tune_relay.py` 输出的模型推理延迟 |
| speedup over Relay | 相对默认 Relay 编译的加速比 |
| speedup over Ansor | 相对 Ansor baseline 的加速比 |
| tuning/search time | 搜索或生成候选所需时间 |
| trials-to-target | 达到目标 latency 所需测量次数 |

### 8.4 执行流程

1. 使用 `gen_state.py` 为每个 hold-out workload 生成候选 schedule。
2. 使用 `measure_programs.py` 对 `gen_eval.json` 进行真实 latency 测量，输出 `measured.json`。
3. 使用 `tune_relay.py` 读取 `measured.json`，编译 Relay 模型并测量端到端延迟。
4. 使用 `baseline.py` 跑 Ansor 不同 trials 的对照实验。
5. 汇总每个 workload 的 latency 与 speedup，并计算 geometric mean。

示例：

```bash
CUDA_VISIBLE_DEVICES=0 /home/qsy/anaconda3/envs/tlm/bin/python measure_programs.py \
  --target nvidia/geforce-rtx-4090 \
  --to-measure-path /home/qsy/workspace/gen_data/4090_gen_eval_only_bert_large/gen_eval.json \
  --measured-path /home/qsy/workspace/gen_data/4090_gen_eval_only_bert_large/measured.json \
  --batch-size 64 \
  --resume
```

### 8.5 论文中预期结论

若 LLM Best-of-K 能在更少候选或更短搜索时间下达到接近 Ansor-1024 的性能，或者在部分 workload 上超过 Ansor，则可证明 LLM 作为调度候选生成器具有实际价值。

## 9. 实验五：跨 workload 泛化能力

### 9.1 实验目的

该实验验证模型是否学习到通用调度规律，而不是记忆单一模型结构的 schedule 模板。

### 9.2 实验设置

将评估 workload 按结构类别分组，并分别报告结构指标和性能指标：

| 类别 | 指标 |
| --- | --- |
| Transformer | `bert_base`, `bert_large` 的 parse/build/latency |
| 常规 CNN | `resnet_50`, `vgg_16` 的 parse/build/latency |
| 轻量 CNN | `mobilenet_v2`, `mobilenet_v3` 的 parse/build/latency |
| 复杂连接 CNN | `resnext_50`, `densenet_121`, `wide_resnet_50` 的 parse/build/latency |

### 9.3 指标

| 指标 | 说明 |
| --- | --- |
| 类别内平均 `parse_valid_rate` | 是否能跨结构生成合法 schedule |
| 类别内平均 `build_valid_rate` | 是否能构建 |
| 类别内 geomean speedup | 是否有性能收益 |
| workload 间方差 | 方法稳定性 |

### 9.4 论文中预期结论

若模型在 Transformer 与 CNN workload 上均保持较高结构合法性，则说明该方法学习到的是 TVM schedule DSL 与硬件映射之间的通用关系，而非单一 workload 的模板。

## 10. 实验六：Latency-Aware Preference Optimization

### 10.1 实验目的

Stage1 主要学习结构合法性，缺少真实性能信号。本实验通过真实 latency 构造偏好对，验证 DPO/LAPO 是否能够提升生成 schedule 的性能倾向。

### 10.2 前置条件

需要使用 `measure_programs.py` 补齐 `measure_records/4090` 中更多真实 latency。当前已有有效 latency 的记录数量有限，可以先进行小规模 proof-of-concept；若作为论文主实验，建议每个训练 workload 至少补测几十到上百条候选记录。

### 10.3 对照组

| 方法 | 描述 |
| --- | --- |
| Stage1 | 只学习结构合法性 |
| Stage1 + SFT-chosen | 只在低 latency chosen suffix 上继续监督 |
| Stage1 + DPO | 标准 Direct Preference Optimization |
| Stage1 + LAPO | 使用 `latency_weight` 加权的 DPO |
| LAPO weight ablation | 比较 `uniform`, `log_gap`, `linear_gap`, `clipped_linear` |

### 10.4 指标

| 指标 | 说明 |
| --- | --- |
| `reward/accuracy` | chosen 隐式 reward 是否高于 rejected |
| `reward/margin_mean` | 偏好区分度 |
| `parse_valid_rate` | 偏好训练后结构合法性是否保持 |
| `build_valid_rate` | 构建合法性是否保持 |
| best-of-K latency | 真实硬件上候选集合最优 latency |
| geomean speedup | across workloads 的整体性能收益 |

### 10.5 执行流程

1. 补测 latency：

```bash
CUDA_VISIBLE_DEVICES=0 /home/qsy/anaconda3/envs/tlm/bin/python measure_programs.py \
  --target "cuda -model=4090" \
  --input-dir /home/qsy/workspace/dataset/measure_records/4090 \
  --batch-size 128 \
  --resume \
  --drop-hold-out
```

2. 构造偏好对：

```bash
/home/qsy/anaconda3/envs/tlm/bin/python build_preference_pairs.py \
  --target "cuda -model=4090" \
  --dataset_path /home/qsy/workspace/dataset/measure_records/4090 \
  --save_path /home/qsy/workspace/gen_data/4090_prefs \
  --min_latency_gap_ratio 1.5 \
  --max_pairs_per_group 4 \
  --max_pairs_per_workload 512 \
  --weight_strategy log_gap \
  --drop_hold_out_workloads True
```

3. 训练 DPO/LAPO：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
PREFERENCE_DATASET=/home/qsy/workspace/gen_data/4090_prefs \
POLICY_MODEL_PATH=/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1 \
REF_MODEL_PATH=/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1 \
TRAINING_MODE=lapo \
/home/qsy/anaconda3/envs/tlm/bin/python run_train_qwen3_dpo.py
```

### 10.6 论文中预期结论

若 LAPO 在不降低结构合法性的前提下降低 best-of-K latency，则说明真实硬件反馈能够通过偏好学习有效注入调度生成模型。该实验可作为论文性能优化主创新点。

## 11. 实验七：LLM 与 Ansor 的混合自动调优

### 11.1 实验目的

该实验验证 LLM 生成候选能否作为 Ansor 的 warm-start，从而减少传统搜索的冷启动成本。

### 11.2 对照组

| 方法 | 描述 |
| --- | --- |
| Ansor 原始搜索 | 使用默认初始种群与搜索策略 |
| LLM-only | 只测量 LLM 生成候选 |
| LLM Warm-start Ansor | 将 LLM 生成候选作为初始 history 或候选池 |
| LLM + Ansor Same Budget | LLM 生成加 Ansor 搜索，总时间与 Ansor 对齐 |

### 11.3 指标

| 指标 | 说明 |
| --- | --- |
| fixed-budget best latency | 固定 10min/30min/1h 下的最优 latency |
| trials-to-target | 达到 Ansor 最优 90% 所需测量次数 |
| convergence curve | 随 trials/time 增加的 best latency |
| final end-to-end latency | 最终 Relay 推理性能 |

### 11.4 论文中预期结论

若 LLM warm-start 能在早期快速达到较优 latency，则可说明 LLM 适合作为传统 evolutionary search 的候选先验，而不是必须完全替代 Ansor。

## 12. 实验结果组织方式

论文中建议按如下表格组织结果。

### 12.1 训练效率表

| 方法 | 平均 token 长度 | train samples/s | train runtime | eval loss | eval accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full-CLM | | | | | |
| PPT-Suffix | | | | | |
| PPT + Dynamic Padding | | | | | |
| PPT + Extended Tokenizer | | | | | |

### 12.2 结构合法性表

| 方法 | parse valid | build valid | fallback | distinct | avg parse/sketch |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base Qwen3 | | | | | |
| Stage1 | | | | | |
| Stage2 | | | | | |
| DPO/LAPO | | | | | |

### 12.3 端到端性能表

| Workload | Relay default | Ansor-256 | Ansor-1024 | LLM Best-of-K | Speedup vs Relay | Speedup vs Ansor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bert_base` | | | | | | |
| `bert_large` | | | | | | |
| `resnet_50` | | | | | | |
| `mobilenet_v2` | | | | | | |
| `densenet_121` | | | | | | |
| Geomean | | | | | | |

### 12.4 偏好优化表

| 方法 | reward acc | parse valid | build valid | best latency | geomean speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage1 | | | | | |
| SFT-chosen | | | | | |
| DPO | | | | | |
| LAPO | | | | | |

## 13. 执行优先级

建议按以下顺序执行：

1. **优先级 P0：结构合法性实验**。先跑 `eval_struct.py`，快速获得 parse/build/fallback/distinct 指标。
2. **优先级 P0：生成策略消融**。对 `keep_cnt`、`temperature`、`top_p` 做小规模消融，确定主实验默认参数。
3. **优先级 P0：真实 latency 与端到端性能实验**。补测 `gen_eval.json`，使用 `tune_relay.py` 输出最终推理延迟。
4. **优先级 P1：训练效率与 tokenizer 消融**。若时间允许，完成扩词表重建数据与训练，对比训练成本和结构合法性。
5. **优先级 P1：DPO/LAPO 偏好优化**。在补齐更多真实 latency 后执行，作为论文性能增强实验。
6. **优先级 P2：LLM + Ansor 混合调优**。若论文时间充足，将其作为系统扩展实验。

## 14. 建议的论文实验叙事

实验章节可按如下逻辑展开：

首先，通过训练效率与结构合法性实验说明 TVM 原生调度状态序列能够被大语言模型有效学习，且 `PPT` 后缀监督使模型聚焦于真正的调度决策。其次，通过多 workload 的 parse/build 结果证明模型生成结果能够重新进入 TVM 编译流程，具备可回放、可构建的系统属性。然后，通过真实 latency 和端到端推理实验验证 LLM 生成候选在较低搜索成本下能够获得有竞争力的性能。最后，通过 DPO/LAPO 实验证明真实硬件反馈可以进一步转化为偏好学习信号，使模型从“生成合法 schedule”升级为“生成高性能 schedule”。

该实验设计能够支撑硕士论文或学术论文中的方法有效性、系统可用性和性能收益三类核心论点。
