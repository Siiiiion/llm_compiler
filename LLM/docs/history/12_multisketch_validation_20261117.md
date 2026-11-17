# 多 sketch 固定预算验收

执行日期：2026-11-17。承接[资源可行性约束复验](11_resource_feasibility_20261117.md)。

## 覆盖问题与样本设计

上一轮 12 个 validation workload 各只测试一个 median sketch。本轮先核对正式验证数据中实际存在多少种不同结构，再扩展模型生成验收。不能把同一结构的不同 SP 因子记录当作新的 sketch。

sketch 身份为完整 task（含 target、hardware）与输入 state：使用训练/生成相同的 `normalize_record` 将 SP 因子归一为 1，然后将变换步骤截断至首个 PPT，再计算 `input_hash`。PPT 后的已知后缀、历史延迟和原始分块因子不参与 sketch 身份；最终候选仍按保留真实因子的完整 state 身份去重。

原 shortest/median/longest 共 36 条代表记录只包含 15 种 sketch。扫描完整、摘要一致的正式 validation raw，原 12 个 workload 实际只有 19 种：workload 3/10 各 4 种、workload 7 有 2 种，其余 9 个各 1 种。本轮选择全部 19 种，原 12 个 median anchor 排在各 workload 的 sketch 0，新结构按 sketch hash 排序，取最早 verified raw row 作为代表。选择只依赖数据结构与身份，不依赖模型输出或延迟。

这意味着本轮只有 **3 个 workload 具备多 sketch 覆盖**；其余 9 个明确保留为单 sketch 对照，不能称作已通过多 sketch 验收。已观察数据中的结构覆盖与所有可能合法 sketch 的覆盖也不同。原 validation 归属、最终 hold-out 排除、源记录和构建凭证均须核验，数据不进入训练。

## 验收设计与统计

新增 [validate_multisketch_generation.py](../../validation/validate_multisketch_generation.py)，复用上一轮实际生成、候选身份、独立资源/编译/GPU 输出检查。新增选择与统计模块：[multisketch_selection.py](../../validation/multisketch_selection.py)、[multisketch_metrics.py](../../validation/multisketch_metrics.py)。

相同基线 checkpoint、`state_v1/state_v2`、每个 workload/sketch/mode 固定 8 次模型尝试，seed 42..49，K=4，达到 K 后仍跑满。19 个 case 共 304 次尝试，两模式均 temperature=0.7、top_p=0.95、top_k=50；阶段 minimum=256、scale=4、无额外 cap，batch=1、无 fallback。原 anchor 保持上轮尝试顺序，新 sketch 交替两模式执行顺序。

预算单位是 **每 workload/sketch**，不是合并后的每 workload。拥有 4 个 sketch 的 workload 每模式用 32 次尝试，不能将合并候选数与上轮的 8 次候选池作等预算比较。`state_v2` 编译器预览是额外成本，单列数量、拒绝原因、建池与缓存耗时；生成阶段已经包含这些时间，不重复相加。

统计同时给出：

- 每个 case 的不同资源通过、可构建、输出正确候选数与 K 短缺；零候选 case 计入宏平均。
- 每个真实 workload 的精确候选去重，以及所有 sketch 均达 K、至少一个 sketch 达 K、合并池达 K 三种不同条件。合并池达标不能掩盖单个 sketch 失败。
- 仅以实际多 sketch 的 3 个 workload 作为多 sketch 覆盖分母；单 sketch workload 单列。

四个 GPU shard 按原 workload 分配，每个 case 内精确相同 state 可跨模式复用检查，跨 case 不复用候选检查。源代码、模型、样本、选择证据、原始输入和实际 TVM/工具链身份写入 manifest 并冻结。每候选硬超时 180 秒，GPU 输出参考仍为相同 TE 的未调度 LLVM、输入 seed 42/43，不是独立算子语义参考；本轮不测 kernel 延迟。

## 正式结果与独立核验

正式运行：[4090_multisketch_20261117_01](../../audit_runs/4090_multisketch_20261117_01)。304 次模型尝试全部完成，四个 shard 和根运行均为 `SUCCEEDED`，结束时全部冻结输入身份核验通过，墙钟耗时 814.10 秒。启动前四张 RTX 4090 均为 0% 利用率、11 MiB 显存占用；运行中进程抽查只有本实验的四个 GPU 进程。每个 case 仍跑满每模式 8 次，没有达到 K 后提前退出。

| 指标 | 状态约束 state_v1 | 资源约束 state_v2 |
| --- | ---: | ---: |
| 模型尝试 / 返回最终 state | 152 / 152 | 152 / 152 |
| 按 case 去重的不同 state | 147 | 144 |
| 重复 state / 返回 state 比例 | 5 / 3.29% | 8 / 5.26% |
| 不同资源通过 / 可构建 / 输出正确候选 | 112 / 112 / 112 | 144 / 144 / 144 |
| 最终候选资源拒绝 | 35 | 0 |
| K=4 达标 case | 16/19（84.21%） | 19/19（100%） |
| 每 case 不同正确候选均值 | 5.89 | 7.58 |
| 累计 case K 短缺 | 4 | 0 |
| 全部已选 sketch 均达 K 的 workload | 9/12 | 12/12 |
| 至少一个 sketch 达 K / 合并池达 K 的 workload | 9/12 / 9/12 | 12/12 / 12/12 |
| 仅实际多 sketch workload：全部 sketch 均达 K | 3/3 | 3/3 |
| 生成 token / 阶段预算累计 | 13,483 / 102,592 | 13,476 / 102,592 |
| 生成阶段累计耗时（秒） | 401.07 | 434.00 |

两模式均无解析空结果、检查超时、资源通过后的编译失败或 GPU 输出错误。`state_v2` 全部 144 个不同最终 state 均通过检查。原来的短缺仍发生在 `state_v1` 的单 sketch workload 0/8/11，不是新增结构的失败。按 case 比较，`state_v2` 在 8 个 case 上增加正确候选、11 个持平，无数量退化，但重复比例有所增加。

跨模式、各 case 内精确去重后实际检查 208 个 state，其中 173 个正确、35 个资源拒绝；两个模式共享 83 个精确 state 的检查结果。不同 sketch 合并后没有额外的精确候选重叠，但合并计数仍按完整身份去重，不能据此推断 kernel 或性能多样性。

### 新增结构与原 anchor 分开解释

| workload / 类别 | 已选 sketch 数 | 各 sketch 的 state_v1 / state_v2 不同正确候选数 |
| --- | ---: | --- |
| 3 / softmax | 4 | 8/8、8/8、8/8、8/8 |
| 7 / elementwise_other | 2 | 8/8、4/4 |
| 10 / softmax | 4 | 8/8、8/8、8/8、8/8 |

新增的 7 个真实结构，在两模式中均产生合计 52 个不同正确候选、全部达到 K；因此本轮证明 `state_v2` 兼容这批新增结构，没有证明其在新增结构上优于 `state_v1`。整体改善来自原 anchor 中已知的资源问题。

workload 7 的第二个 sketch，两模式均只有 4 个不同正确候选，另有 4 次重复，到第 6 次尝试才达到 K。`state_v2` 的该 case PPT pool 只有 1 个通过 completion，后续阶段仍可产生多个最终 state；PPT completion 数量不能当作最终候选数量。其余新 sketch 均在第 4 次尝试达到 K。候选供给达到本轮目标，但不应外推为更大的 K 或全部合法结构均已覆盖。

[原 anchor 复现报告](../../audit_runs/4090_multisketch_20261117_01/anchor_comparison_report.json) 确认 12 个原始 median anchor、两模式共 192 次尝试的提示词、原始生成 token、候选身份和资源/构建/输出门禁全部与上一轮精确一致。checkpoint、tokenizer、实际编译器二进制及 TVM Python 身份相同，PATH 的包装器/临时目录差异单独保留。原 anchor 不能作为新增多 sketch 证据重复计数。

### 资源预览成本

[资源 trace 核验](../../audit_runs/4090_multisketch_20261117_01/resource_trace_report.json) 确认 152 次 `state_v2` 尝试的 PPT/TBS/PRS 阶段全部完成，无空域或缺失失败 trace；最终 record 的 1,376 次因子选择与模型 trace 一致。15 个 case 创建资源 completion pool，另外 4 个没有此类 pool。120 次受限 PPT 选择均属于实际检查通过的域，128 次受限向量因子均为 1，dtype 字节上限和保守对齐策略保留。

共 15 次建池、105 次缓存命中、478 次不同完整赋值的编译器预览：305 个接受，173 个因 shared memory 超限拒绝。原 8 个 pool 共 429 次预览；新增 7 个结构额外检查 49 个赋值，全部通过，接受数分别为 9/9/9、1、7/7/7。较小 pool 反映当前有限提议过程实际找到的域，不宣称完整合法空间。预览拒绝不计作模型尝试或最终候选失败。

建池累计 37.66 秒，缓存查询累计约 0.00040 秒，均已计入 `state_v2` 的 434.00 秒生成阶段时间，不能再次相加。`state_v1` 为 401.07 秒；两模式不是等总计算预算。保守 scalar 向量策略尚未接受独立延迟评估，本轮没有 kernel 性能或整网加速结论。

### 产物核验与回归

以下四项均通过：

- [产物与来源核验](../../audit_runs/4090_multisketch_20261117_01/report.json)：49 个冻结来源、完整 validation raw/source build proof、确定性选择、源码/模型/编译器身份、304 次尝试的顺序/seed、候选身份及检查复用。汇总重放复用经过聚焦回归的生产统计函数。
- [独立计数核验](../../audit_runs/4090_multisketch_20261117_01/independent_count_report.json)：另一份仅使用标准库的实现，不导入生产统计或选择模块，直接从原始尝试重算候选去重、各类失败、case K、workload all/any/pooled K 及实际多 sketch 的分母，与正式 summary 一致。
- [资源 trace 核验](../../audit_runs/4090_multisketch_20261117_01/resource_trace_report.json)：逐项复核实际 pool、最终模型因子和对齐策略，独立保留编译器预览数量及耗时。
- [原 anchor 复现](../../audit_runs/4090_multisketch_20261117_01/anchor_comparison_report.json)：两模式的全部 192 次原样本生成结果与上一轮一致。

启动前 60 项聚焦回归通过（4.80 秒），覆盖资源可行性、约束生成、多候选计数、sketch 选择和多 sketch 指标。真实数据重复选样一致、19 个样本均可恢复成生成输入；5 个新增模块/测试的 Black 100 列、AST 和 `git diff --check` 通过。审查及测试记录保存在运行目录。

复现时使用新的运行目录；模型、来源和实际环境以 manifest 为准：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/qsy/anaconda3/envs/tlm/bin/python -u LLM/__main__.py validate-sketches \
  --run-dir LLM/audit_runs/4090_multisketch_NEW \
  --gpus 0 1 2 3 --attempts 8 --required 4 --max-sketches 4 \
  --modes state_v1 state_v2

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  /home/qsy/anaconda3/envs/tlm/bin/python -m pytest -q -p no:cacheprovider \
  tests/python/unittest/test_llm_resource_feasibility.py \
  tests/python/unittest/test_llm_constrained_generation.py \
  tests/python/unittest/test_llm_multicandidate_generation.py \
  tests/python/unittest/test_llm_multisketch_metrics.py \
  tests/python/unittest/test_llm_multisketch_selection.py
```

四份核验脚本均保存在运行目录，根状态成功后执行；此命令核验现有产物，不重新运行 GPU 实验：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/qsy/anaconda3/envs/tlm/bin/python \
  LLM/audit_runs/4090_multisketch_20261117_01/collect_multisketch_report.py
PYTHONDONTWRITEBYTECODE=1 /home/qsy/anaconda3/envs/tlm/bin/python \
  LLM/audit_runs/4090_multisketch_20261117_01/audit_multisketch_resource.py
PYTHONDONTWRITEBYTECODE=1 /home/qsy/anaconda3/envs/tlm/bin/python \
  LLM/audit_runs/4090_multisketch_20261117_01/compare_multisketch_anchors.py
PYTHONDONTWRITEBYTECODE=1 /home/qsy/anaconda3/envs/tlm/bin/python \
  LLM/audit_runs/4090_multisketch_20261117_01/independent_count_audit.py
```

## 后续顺序

1. 本轮多 sketch 验收已完成：覆盖该 12 个 workload 在正式验证数据中全部 19 种实际结构，真正具有多个结构的任务仅有 3 个，其余 9 个仍为单 sketch 对照。扩大结构覆盖时需另取可追溯的合法 sketch，不能重复旧因子记录或声称已穷尽合法域。
2. 下一步在训练侧候选上补独立算子语义参考和真实 GPU 延迟：先固定可验证的 workload/形状/布局、候选与基线、正确性输入、计时配置和独占 GPU，再判断候选是否更快及 scalar 向量策略的代价。本轮 validation 候选不直接混入训练。
3. 取得足量通过独立正确性、带实际延迟的训练侧候选后，再决定偏好对构建、等监督 token/多 seed 的训练对照和整网评估；继续排除最终 hold-out。
