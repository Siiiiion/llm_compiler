# 资源可行性约束与固定预算复验

执行日期：2026-11-17。承接[固定预算多候选验证](10_multicandidate_validation_20261111.md)。

## 解决的问题与实现

上一轮 `state_v1` 在固定 8 次采样、K=4 下只有 9/12 个 workload 达标，35 个不同 state 被资源门禁拒绝。共享内存与向量宽度是可复现的主要拒绝原因。本轮新增可选 `state_v2`，将资源检查前移到因子选择阶段，同时保留最终候选的独立资源、编译和 GPU 输出检查。

[resource_feasibility.py](../../generation/resource_feasibility.py) 根据完整 PPT 因子重建私有 state，补全标准 tiled-root 的 block/vthread/thread 绑定，实际执行 `tvm.lower` 和 `VerifyGPUCode`。使用原 task 的硬件约束，记录降低后真实 shared allocation，不改变输入 state。不能支持的几何结构和 lowering 错误明确拒绝，不估算为通过。

[constrained_generation.py](../../generation/constrained_generation.py) 为每个精确 task/sketch 用独立 `Random(0)` 提议最多 128 次完整因子赋值，按 `factor**-0.5` 加权，收集最多 32 个通过检查的不同 completion；重复提议不重复降低。解码前缀必须存在于这些已检查 completion 中，最终因子仍由模型通过 token 选择，不将编译器提议当作模型候选或 fallback。该域是有限的保守子集，不是全部合法调度空间；搜索为空会明确失败。缓存身份包含序列化 task/state（含 target、hardware、原始 sketch），LRU 上限为 32。

TBS 中识别 `SP` 后的 vectorize annotation，按元素 dtype 计算字节宽度上限。然而合法向量宽度不能证明基址和索引对齐：正式策略暂将这些向量化分块因子限制为 1，trace 同时保存字节上限和 `factor_1_until_base_and_index_alignment_proven` 策略。模型在此合法域内生成，不在输出后静默修改 state。此策略可能损失性能，须在以后证明对齐、测量延迟后再扩大域；本轮不作性能提升结论。

## 失败记录与修复证据

旧运行 [4090_resource_constraints_20261111_01](../../audit_runs/4090_resource_constraints_20261111_01) 保留 `FAILED`。其中 workload 8 的候选 `2b3793db…` 已通过资源和编译，却在 GPU 同步时出现 `CUDA_ERROR_MISALIGNED_ADDRESS`；PTX 的 shared buffer 仅保证 4 字节对齐，随后 `st.shared.v4.f32` 要求 16 字节。运行错误写入检查报告后，CUDA 析构又导致 worker 以 -6 退出，原执行器因此中断整个运行。

修复包括上述因子 1 对齐策略，以及执行器对完整负面检查报告的保留：若 worker 已保存 `correct=false` 且含实际错误的报告，析构异常作为 `worker_return_code` 记录，候选仍计运行失败；缺失/不完整报告、或报告成功但进程异常退出，仍使运行按基础设施失败终止。不会把进程崩溃转换为成功。

旧失败 state 的显式 scalar-vector 诊断通过资源、构建和 GPU 输入 seed 42/43，最大绝对误差约 `1.77e-6`，见[诊断检查](../../audit_runs/4090_resource_constraints_20261111_01/alignment_diagnosis/2b3793dbcf1c80e2363287b73bf028c942d5753da48df4e461953207b10b7d4e.scalar_vector_diagnostic.check.json)。这是人为修改的定位实验，不计入正式模型候选或 K。

历史共享内存失败记录重放共 35 条，全部被新 PPT preview 拒绝：31 条由真实 `VerifyGPUCode` 拒绝、4 条自由生成的非整除因子提前拒绝；包含全部 26 条 `state_v1` 共享内存失败，源 state 均未变。见[重放证据](../../audit_runs/4090_resource_constraints_20261111_01/ppt_preview_evidence/report.json)。这验证已知失败的拒绝能力，不代替新模型生成验收。

2026-11-17 的 `_01` 目录仅完成准备文件复制，沙箱内 `nvidia-smi` 退出码为 9，未生成 manifest 或候选。正式复验在驱动可访问的环境使用新的 `_02` 目录，从零运行，没有复用上述失败运行的尝试或候选。

## 正式设计与预算边界

运行目录：[4090_resource_constraints_20261117_02](../../audit_runs/4090_resource_constraints_20261117_02)。相同 12 个 validation workload、每个相同 median sketch、原基线 checkpoint，比较 `state_v1/state_v2`。每 workload 每模式恰好 8 次模型尝试，seed=42..49，K=4，达标后仍跑满，共 192 次。temperature=0.7、top_p=0.95、top_k=50；每阶段 minimum=256、scale=4，无额外 cap，batch=1，无 fallback。按 workload 和尝试编号交替模式顺序。

启动前四张 RTX 4090 均为 0% 利用率、11 MiB 占用，每卡一个 shard，CPU/磁盘环境共享。manifest 冻结源码、模型、输入、实际 TVM Python/动态库和工具链身份。不同候选按精确 task/state 去重，同一 workload 的精确相同 state 可跨模式复用检查结果，但各模式 K 分开统计。检查子进程硬超时 180 秒；参考仍是同一 TE 的未调度 LLVM、输入 seed 42/43，尚不是独立算子语义参考。

固定的是模型尝试数与阶段 token 上限。`state_v2` 另外付出有限 completion pool 的编译器预览成本，需独立报告检查数量、接受/拒绝、缓存命中和耗时。两模式不是等总 token、等总算力或等墙钟预算，生成时间也不是 kernel 延迟。精确 state 多样性不代表 kernel 或性能多样性。本轮数据只作 validation，不使用最终 hold-out，不进入 SFT/DPO/LAPO。

## 正式结果

192 次尝试全部完成，四个 shard 与根运行均为 `SUCCEEDED`，最终输入身份复查通过，墙钟耗时 545.42 秒。独立[产物核验报告](../../audit_runs/4090_resource_constraints_20261117_02/report.json) 检查冻结源码、模型/输入摘要、实际 TVM Python、动态库与工具链、全部尝试的 seed/顺序、候选身份、检查复用和原始产物，重新计算的汇总与正式 summary 一致。

| 指标 | 状态约束 state_v1 | 资源约束 state_v2 |
| --- | ---: | ---: |
| 模型尝试 / 返回最终 state | 96 / 96 | 96 / 96 |
| 不同 state | 95 | 92 |
| 重复 state / 占返回 state 比例 | 1 / 1.04% | 4 / 4.17% |
| 不同资源通过 / 可构建 / 输出正确候选 | 60 / 60 / 60 | 92 / 92 / 92 |
| 资源拒绝 | 35 | 0 |
| K=4 达标 workload | 9/12（75%） | 12/12（100%） |
| 每 workload 不同正确候选均值 | 5.00 | 7.67 |
| 累计 K 短缺 | 4 | 0 |
| 生成 token / 阶段预算累计 | 10,835 / 65,728 | 10,828 / 65,728 |
| 生成阶段累计耗时（秒） | 310.77 | 338.33 |

所有 `state_v2` 最终不同 state 均通过资源、构建和两组 GPU 输入输出检查；两模式均无检查超时、资源通过后的编译失败或输出错误。跨模式精确去重后实际检查 156 个不同 state，其中 121 个正确、35 个资源拒绝，两个模式共享 31 个相同 state 的检查结果。候选数量在 8 个 workload 上增加、4 个持平，没有数量退化，但重复率有所增加；这不等价于性能多样性改善。

| 编号 | 类别 | state_v1 / state_v2 不同正确候选 | state_v2 重复次数 | state_v2 达到 K 所需尝试 |
| --- | --- | ---: | ---: | ---: |
| 0 | batch_matmul | 3 / 8 | 0 | 4 |
| 1 | convolution | 4 / 8 | 0 | 4 |
| 2 | matmul_dense | 4 / 8 | 0 | 4 |
| 3 | softmax | 8 / 8 | 0 | 4 |
| 4 | batch_matmul | 4 / 7 | 1 | 4 |
| 5 | convolution | 5 / 8 | 0 | 4 |
| 6 | convolution | 4 / 8 | 0 | 4 |
| 7 | elementwise_other | 8 / 8 | 0 | 4 |
| 8 | matmul_dense | 2 / 7 | 1 | 5 |
| 9 | pooling | 7 / 7 | 1 | 4 |
| 10 | softmax | 8 / 8 | 0 | 4 |
| 11 | batch_matmul | 3 / 7 | 1 | 4 |

原短缺 workload 0/8/11 分别从 3/2/3 个增至 8/7/7 个正确候选，全部补足 K。workload 8 需要第 5 次尝试达到 K，其余 11 个 workload 在第 4 次达到；仍完整执行全部 8 次尝试。

[基线复现核验](../../audit_runs/4090_resource_constraints_20261117_02/baseline_state_v1_comparison.json) 确认本轮 `state_v1` 的 96 次模型 token 序列、候选身份及检查结果与 11 月 11 日全部精确一致；checkpoint/tokenizer、输入、实际编译器二进制及 TVM Python 身份一致。环境仅 PATH 的 Codex 包装器/临时目录前缀变化，实际解析到的工具及其摘要一致，报告保留该差异，不宣称字面环境完全相同。

## 编译器预览成本与覆盖核验

[资源 trace 核验](../../audit_runs/4090_resource_constraints_20261117_02/resource_trace_report.json) 确认 96 次 `state_v2` 模型尝试均完成 PPT/TBS/PRS 阶段，最终 record 的 1,272 次因子选择与模型 trace 一致。8 个需要资源 completion pool 的 workload 共 64 次 PPT 选择全部属于实际检查通过的域；另外 4 个 workload 无此类 PPT pool。128 次受限向量因子均为 1，且保留对应 dtype 字节上限凭证。

共创建 8 个 pool，缓存命中 56 次，实际编译器预览 429 个不同赋值，其中 256 个通过、173 个因共享内存超限拒绝，每个 pool 收集 32 个通过 completion。所有预览严格沿用每 block 49,152 字节 shared memory、16 字节 vector width 等 task 限额。预览中的拒绝不是额外模型尝试，也没有被计为最终候选失败。

pool 创建累计耗时 36.05 秒（包含 Python 提议和域构造，并未逐 probe 单独计时），缓存查询累计约 0.00019 秒。这些时间已包含在 `state_v2` 的生成累计 338.33 秒中，不能再次相加。`state_v1` 生成累计为 310.77 秒，不能把通过率提高表述为生成加速或等算力收益。标量向量策略也未接受 kernel 延迟验收。

## 回归与复现

本轮 39 项聚焦回归通过（5.56 秒），覆盖私有 state 的完整赋值/不可变性、lowering 与异常拒绝、completion 前缀可行性、容量/空域、dtype 和对齐策略、非向量线程分块不被误限、生产/评估路由一致、预算终止、重复候选与跨模式计数，以及 GPU 析构崩溃后的完整负面报告保留。Black 100 列 API 格式检查和 `git diff --check` 通过。

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/qsy/anaconda3/envs/tlm/bin/python -u LLM/__main__.py validate-candidates \
  --run-dir LLM/audit_runs/4090_resource_constraints_20261117_02 \
  --gpus 0 1 2 3 --attempts 8 --required 4 --modes state_v1 state_v2

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  /home/qsy/anaconda3/envs/tlm/bin/python -m pytest -q -p no:cacheprovider \
  tests/python/unittest/test_llm_resource_feasibility.py \
  tests/python/unittest/test_llm_constrained_generation.py \
  tests/python/unittest/test_llm_multicandidate_generation.py
```

三项独立核验命令：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/qsy/anaconda3/envs/tlm/bin/python \
  LLM/audit_runs/4090_resource_constraints_20261117_02/collect_report.py
PYTHONDONTWRITEBYTECODE=1 /home/qsy/anaconda3/envs/tlm/bin/python \
  LLM/audit_runs/4090_resource_constraints_20261117_02/resource_trace_audit.py
PYTHONDONTWRITEBYTECODE=1 /home/qsy/anaconda3/envs/tlm/bin/python \
  LLM/audit_runs/4090_resource_constraints_20261117_02/baseline_state_v1_comparison.py
```

重跑使用新目录；正式执行时须可访问 GPU 驱动。核验脚本与最终凭证保存在运行目录。

## 后续顺序

1. 本轮单一 median sketch 验收已完成。后续[多 sketch 固定预算验收](12_multisketch_validation_20261117.md)也已完成：304 次尝试覆盖原 12 个 workload 的全部 19 种已观察结构，`state_v2` 的 19/19 个 case 均达到 K=4；真正有多 sketch 的任务只有 3 个，两模式在新增结构上均达标。来源/产物、资源 trace、原 anchor 复现和独立计数核验通过，原 192 次 anchor 生成结果全部复现。
2. 候选供给稳定后，在训练侧候选上补充独立算子语义参考和真实 GPU 延迟，检查 scalar 向量策略的性能代价。
3. 再决定是否进行等监督 token、多 seed 的训练对照；当前没有 kernel 延迟、整网加速或多 sketch 泛化结论，不据本轮直接扩大训练。
