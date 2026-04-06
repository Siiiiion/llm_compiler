# Qwen3-0.6B 面向 TVM 调度继续预训练计划

## 1. 结论

当前模型持续生成 `invalid` 组合，主要问题更像是**训练数据集与训练目标不对齐**，而不只是模型能力不足。

更准确地说：

- 原始测量记录不一定坏。
- 但当前继续预训练使用的 `workspace/gen_data/4090_gen_qwen` 这份数据集，作为“让模型学会生成合法 TVM 调度后缀”的训练数据，存在明显结构性问题。
- 因此不建议直接在当前数据集和当前 checkpoint 上继续盲目续训。

## 2. 已确认的问题

### 2.1 训练集存在明显截断，很多样本根本看不到 `PPT`

对当前 tokenized 数据集做抽样检查后，得到：

- `train` 总样本数：`2155083`
- `validation` 总样本数：`113425`
- 在 `train` 中随机抽样 `5000` 条：
  - `17.68%` 的样本找不到 `PPT`
  - 这些缺失 `PPT` 的样本长度全部都是 `544`
- 在 `validation` 中随机抽样 `5000` 条：
  - `0%` 的样本缺失 `PPT`

这几乎可以直接说明：

- `max_length=544` 会把不少训练样本截断在 `PPT` 之前。
- 模型在大量训练样本上甚至没有看到“prefix 到 `PPT` 再到 suffix”的完整监督链条。

### 2.2 当前验证集分布和训练集分布不一致

`make_dataset_utils.py` 里的切分方式是：

- `validation = train[:5%]`
- `train = train[5%:]`

这里没有 shuffle，也不是按 workload/file 分组切分，而是直接取 `0_merge.json` 前 `5%`。

这会带来两个问题：

- `validation` 很可能只覆盖更短、更容易的样本。
- `train` 中出现的大量 `PPT` 截断问题，在 `validation` 中几乎看不到。

因此当前的：

- `eval_accuracy = 0.9754`
- `eval_loss = 0.0554`

不能代表模型真的学会了生成**合法调度组合**，更可能只是说明它在一个更容易的验证子集上把 token 续写得很像。

### 2.3 `for_gen` 数据集混入了失败测量和低质量样本

当前 `make_dataset.py` 里的 `for_gen` 会：

- 保留所有记录
- 当 `latencies == [0]` 时，把该条记录视为失败样本，但仍然写入训练文本
- 计算出的 `labels = latency_min / latency` 最终并没有被 CLM 训练真正使用

也就是说，最终训练目标其实是：

- 让模型模仿“所有 suffix”

而不是：

- 优先学习“合法且高质量的 suffix”

这会导致模型学到很多：

- 低质量组合
- 失败组合对应的文本模式
- 仅 token 层面像样、但 parser/build 不通过的组合

### 2.4 当前监督目标过弱，只是在做 token 续写

当前训练主逻辑是：

- 仅对 `PPT` 之后的 token 计算 loss

这一步方向是对的，但还不够。

抽样结果显示：

- `PPT` 后缀监督 token 平均长度：`52.96`
- 中位数：`32`
- `p90`：`139`

说明模型真正被监督的“决策区”并不长，而且监督信号仅仅是：

- “下一个 token 是否和训练文本一致”

而不是：

- “整个后缀是否满足 TVM grammar”
- “能否被 `gen_state.py` 成功转成合法 state”
- “能否 build 成功”

### 2.5 现有第二阶段训练本身也在丢样本

第二阶段训练日志显示：

- 原始 `train` 样本：`2155083`
- 实际加载后的 `train` 样本：`1713828`

也就是大约有 `20%` 的样本在 `PPT` mask/filter 后被丢弃。

这进一步说明：

- 当前 tokenized 数据集并不是天然适配“只监督 `PPT` 后缀”的训练目标。

### 2.6 tokenizer 不是当前主因

检查了：

- 基础模型 tokenizer
- finetuned tokenizer
- `fix_mistral_regex=True` 与否

在代表性调度字符串上的编码结果一致。

因此当前问题的主因不是 tokenizer 不匹配，而是：

- 数据截断
- 数据切分偏差
- 样本质量混杂
- 评估指标失真

## 3. 对问题的判断

结论是：

- **是，数据集本身出现了问题。**

但这里的“数据集问题”主要不是说原始测量记录完全不可用，而是说：

- 当前这份用于继续预训练的 `4090_gen_qwen` 数据集，在构造方式上不适合“学习合法调度后缀生成”。

因此后续工作重点不应是继续堆 epoch，而应先：

1. 重建训练数据
2. 重做切分
3. 增加更贴近 `gen_state.py` 的验证方式

## 4. 推荐的数据重建方案

### 4.1 不再直接使用当前 `4090_gen_qwen` 作为主训练集

建议重新从原始测量记录生成一个新的训练数据集，目标是：

- 每条样本都包含完整 `PPT`
- `PPT` 之后是完整、未截断的 decision suffix
- 尽量只保留合法、有效、可学习的调度后缀

### 4.2 新数据集至少拆成两类

建议准备两个主数据集：

#### A. `valid_all_suffix`

用途：

- 学习“合法 suffix 的整体分布”

构造原则：

- 仅保留包含 `PPT` 的完整样本
- 丢弃 `latencies == [0]` 的失败测量
- 丢弃被截断到 `PPT` 之前的样本
- 丢弃 `PPT` 后缀过短的样本

#### B. `best_per_ppt_suffix`

用途：

- 学习“同一 sketch/prefix 下更优的决策选择”

构造原则：

- 对同一个 `PPT` 前缀，只保留最优或前若干优的 suffix
- 可以直接借鉴 `for_gen_best`
- 作为第二阶段或第三阶段精调集使用

### 4.3 数据切分必须按 workload 或源文件切

不要再使用：

- `train[:5%]`

建议改为：

- 先按 workload/file 分组
- 再做随机划分
- validation/test 必须和 train 在 workload 级别上隔离

这样才能避免：

- 同类样本泄漏
- 验证集过于简单
- 指标高但实际生成无效

### 4.4 在 tokenize 前保留结构化字段

不要一开始就只留下纯文本。

建议在中间 JSONL 中保留这些字段：

- `source_file`
- `workload_key`
- `compute_dag`
- `prefix_text`
- `suffix_text`
- `full_text`
- `latency`
- `relative_score`
- `is_best`
- `suffix_token_len`
- `full_token_len`

这样后面可以方便地做：

- 数据筛选
- 重采样
- 统计分析
- 质量回溯

### 4.5 重新测定 `max_length`

当前 `544` 明显偏小。

建议做法：

1. 先在重建后的中间数据上统计 `full_token_len` 分布
2. 选一个能让“缺失 `PPT` 比例 < 1%”的长度

优先候选：

- `768`
- `1024`

如果显存允许，优先 `1024`。

## 5. 推荐的继续预训练路线

## 5.1 起点模型建议

不建议直接从当前 `Qwen3-0.6B-fintuned-2` 继续盲续。

推荐优先级：

1. 从基础模型 `Qwen3-0.6B` 重新开始一轮干净训练
2. 如果时间成本太高，再考虑从当前模型低学习率续训

原因：

- 当前 finetuned 模型已经吸收了有问题的数据分布
- 如果数据不先修正，只会继续强化错误模式

### 5.2 三阶段训练建议

#### 阶段 A：合法 suffix 分布对齐

数据：

- `valid_all_suffix`

目标：

- 先学会生成“完整且合法的后缀形态”

训练方式：

- 保留现在的 `PPT` 前缀 masking 思路
- 但输入必须保证包含完整 `PPT`
- 只在高质量样本上训练

#### 阶段 B：偏向优质决策

数据：

- `best_per_ppt_suffix`

目标：

- 在已经会生成合法结构的基础上，进一步偏向更优调度组合

训练方式：

- 继续只监督 `PPT` 之后
- 对更优 suffix 提高采样权重

#### 阶段 C：面向 `gen_state.py` 的闭环精调

数据来源：

- 用阶段 B 模型在 held-out sketch 上生成结果
- 收集：
  - parser 失败样本
  - build 失败样本
  - 合法但低质量样本

目标：

- 让模型更适应真实推理接口，而不是只适应离线文本拟合

这一步可以理解为：

- “把训练目标从纯 CLM 拉近到真实 state 生成”

## 6. 推荐的训练配置方向

以下是方向性建议，暂不改代码：

- `max_length`：优先试 `768` 或 `1024`
- `mask_prefix_before_ppt=True`：保留
- `drop_samples_without_ppt=True`：保留
- `min_suffix_tokens`：可以从 `8` 提高到 `16` 做对比实验
- 训练数据必须先 shuffle，再 split
- split 必须按 workload/file，而不是按合并后顺序

如果从基础模型重训：

- 阶段 A 学习率可从 `1e-5 ~ 2e-5` 起试
- 阶段 B 再降到 `5e-6 ~ 1e-5`

如果从当前 finetuned 模型续训：

- 建议直接用更低学习率，如 `5e-6`

避免再次把已有错误模式放大。

## 7. 评估方式必须改

当前只看：

- `eval_accuracy`
- `eval_loss`

是不够的。

后续评估至少要补三类指标：

### 7.1 数据集静态指标

- `missing_ppt_ratio`
- `suffix_len` 分布
- `latency==0` 占比
- 每个 workload 的样本数分布

### 7.2 离线生成有效性指标

在 held-out sketch 上运行 `LLM/gen_state.py`，统计：

- 生成后被 TVM parser 接受的比例
- 唯一有效 state 数量
- fallback 到 sketch 的比例
- build 成功率

### 7.3 搜索实用性指标

最终要看的不是 token 准确率，而是：

- 每个 workload 能否生成足够多的有效候选
- 这些候选是否能进入后续测量流程
- 是否能找到比 baseline 更好的 schedule

## 8. 推荐执行顺序

1. 重新生成中间 JSONL，不要只保留纯文本。
2. 做一次数据审计，确认：
   - `missing_ppt_ratio < 1%`
   - validation 与 train 的长度分布接近
   - 不再混入失败测量主样本
3. 生成新的 tokenized 数据集。
4. 从基础模型做一个小规模 pilot 训练。
5. 用 `LLM/gen_state.py` 在 held-out sketch 上验证 parser/build 有效率。
6. 如果有效率明显提升，再跑完整训练。
7. 最后再考虑是否需要进一步调整推理侧解码策略。

## 9. 现阶段最重要的判断

当前最优先要修的不是：

- 模型大小
- tokenizer
- 多训练几轮

而是：

- **训练数据集的构造**

如果不先修这个问题，继续在当前数据上训练，很可能只会继续得到：

- `eval_accuracy` 看起来不错
- 但 `gen_state.py` 里依然大量生成 `invalid` 组合

## 10. 下一步建议

下一步应优先做两件事：

1. 重建一个“完整 `PPT` + 合法 suffix + 按 workload/file 正确切分”的新数据集
2. 在新数据集上做一次小规模 pilot，而不是直接在旧数据集上长时间续训

在这两步完成之前，不建议把当前 `Qwen3-0.6B-fintuned-2` 当成可靠基线继续叠加训练。
