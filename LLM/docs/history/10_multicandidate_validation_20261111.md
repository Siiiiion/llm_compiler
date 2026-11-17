# 固定预算多候选验证

执行日期：2026-11-11。承接 [状态约束生成](09_state_constrained_generation_20261111.md)。

2026-11-17 后续完成：本页发现的资源拒绝已通过可选 `state_v2` 的 PPT lowering 预览和保守向量对齐约束处理。相同 12 个 median sketch、每模式 8 次尝试的正式复验中，不同正确候选从 60 增至 92、K 达标从 9/12 增至 12/12，原基线的全部 96 次 token/候选/检查结果精确复现。额外编译器成本、多样性边界、上次失败与完整核验见[资源可行性验收](11_resource_feasibility_20261117.md)。下文保留本页原始实验，不将新结果回填为旧运行结果。

## 本轮问题与设计

单次 greedy 生成通过，仍不能说明生产需要多个候选时能提供足够不同的有效 state。本轮只验证候选数量、多样性、资源与输出正确性，不训练模型或比较 kernel 延迟。

新增 [validate_multicandidate_generation.py](../../validation/validate_multicandidate_generation.py)，复用前一轮相同的 12 个 validation workload 和同一 median sketch。每个 workload、每种模式固定采样 8 次，K=4，达到 K 后继续跑满预算，共 192 次尝试。使用原始基线 checkpoint，`none/state_v1` 的 temperature=0.7、top_p=0.95、top_k=50；每次 seed=42..49。按 workload 和尝试编号交替两种模式的执行顺序。

两种模式阶段 token 上限均为 minimum=256、scale=4，无额外 cap；每次独立重建相同 sketch，关闭 fallback，batch=1。固定的是候选尝试数与每阶段上限；自由模式和约束模式可能到达不同阶段、实际生成不同 token，因此不是固定总 token 或固定墙钟预算比较。有限 token 掩码改变合法空间内的采样分布，使用相同 seed 不表示两种模式输出应逐 token 对应。

唯一候选按完整 task/state 的 `input_hash` 去重，保留真实分块因子；不使用会归一化因子的 `normalize_record` 作为最终候选身份。重复尝试计入尝试数、生成 token 和时间。身份是调度 state 身份，尚不保证不同 state 对应不同 CUDA kernel 或性能。

每个不同 state 在独立子进程中执行资源检查、编译导出和 GPU 输出检查。相同 workload 中精确重复 state 的检查结果可以在两种模式间复用，但每种模式的唯一候选数分别计算。每候选检查硬超时 180 秒。输出参考仍为相同 TE 的未调度 LLVM 结果，两组输入 seed=42/43；这不是独立的算子语义参考。

四张空闲 RTX 4090 各运行一个 shard，每张卡依次完成分配的 workload。并行运行会共享 CPU、磁盘和功耗环境，因此生成时间是本次运行成本，不作为单卡性能基准。沿用实际 `/data3/qsy/complier/tlm` TVM 动态库，manifest 保存完整身份。

## 产物与统计

运行目录：[4090_multicandidate_20261111_01](../../audit_runs/4090_multicandidate_20261111_01)。

每次尝试保存原始 prompt/continuation IDs、解码文本、阶段预算与约束 trace；每个不同 state 保存 record、检查日志和检查结果。根目录保存冻结代码、样本、模型摘要、manifest；各 shard 保存逐 workload 结果与状态，结束后统一核验身份和案例数量。

报告包括解析尝试、唯一 state、重复率、资源通过、唯一可构建/输出正确候选、K 达标率、短缺、达到 K 所需尝试数，以及各阶段失败原因。候选检查超时单列，不能当成已证明的非法 state。运行 `SUCCEEDED` 只表示流程和身份检查完成，K 是否达标以 summary 为准。

## 最终结果

192 次尝试全部完成，四个分片均为 `SUCCEEDED`，结束身份复查通过，墙钟耗时 818.06 秒。独立汇总脚本重新核对源码与冻结副本、模型与输入摘要、TVM Python 源码、动态库与工具链摘要、全部 seed、尝试顺序、候选身份及检查产物，原始 attempts 重算与 summary 完全一致。凭证见 [report.json](../../audit_runs/4090_multicandidate_20261111_01/report.json)，相邻 `collect_report.py` 保存核验方法。

| 指标 | 自由生成 none | 状态约束 state_v1 |
| --- | ---: | ---: |
| 尝试次数 | 96 | 96 |
| 返回最终 state 的尝试 | 37 | 96 |
| 不同 state | 37 | 95 |
| 重复 state / 占返回 state 比例 | 0 / 0% | 1 / 1.04% |
| 不同资源通过 / 可编译 / 输出正确候选 | 23 / 23 / 23 | 60 / 60 / 60 |
| K=4 达标 workload | 2/12（16.7%） | 9/12（75.0%） |
| 每 workload 不同正确候选均值 | 1.92 | 5.00 |
| 按 workload 累计的 K 短缺 | 32 | 4 |
| 未返回最终 state 的尝试 | 59 | 0 |
| 不同 state 的资源拒绝 | 14 | 35 |
| 生成 token | 54,720 | 10,835 |
| 生成阶段累计耗时（秒） | 1,538.13 | 387.56 |

两种模式之间有 2 个相同 state，实际跨模式去重后检查 130 个不同 state，其中 81 个通过资源、编译和两组 GPU 输入输出检查，49 个在资源门禁拒绝。没有检查超时、资源通过后的编译失败或输出不一致。`no_final_state` 包含 C++ 后端裁剪，不能将自由模式的 59 次全部称为文本解析失败。

约束模式在 10 个 workload 上增加正确候选数、2 个持平，没有候选数量退化。逐 workload 结果如下，编号与运行中的 `index` 一致；短横线表示在 8 次预算内未达到 K。

| 编号 | 类别 | 自由 / 约束不同正确候选 | 约束资源拒绝 | 约束达到 K 所需尝试 |
| --- | --- | ---: | ---: | ---: |
| 0 | batch_matmul | 1 / 3 | 5 | — |
| 1 | convolution | 0 / 4 | 4 | 7 |
| 2 | matmul_dense | 2 / 4 | 4 | 7 |
| 3 | softmax | 0 / 8 | 0 | 4 |
| 4 | batch_matmul | 2 / 4 | 4 | 6 |
| 5 | convolution | 0 / 5 | 3 | 7 |
| 6 | convolution | 0 / 4 | 4 | 7 |
| 7 | elementwise_other | 8 / 8 | 0 | 4 |
| 8 | matmul_dense | 1 / 2 | 6 | — |
| 9 | pooling | 7 / 7 | 0 | 4 |
| 10 | softmax | 1 / 8 | 0 | 4 |
| 11 | batch_matmul | 1 / 3 | 5 | — |

唯一一次精确重复发生在 pooling，仍达到 K。未达标的编号 0、8、11 分别只得到 3、2、3 个正确候选，累计短缺 4 个；这三者没有精确重复，短缺直接对应资源拒绝。本轮主要瓶颈不是精确 state 重复。

生成 token 减少约 80.2%，生成阶段累计时间减少约 74.8%。这些数值包含强制固定字段和阶段提前停止的效果，且来自四卡并发环境；两种模式实际到达的阶段与 token 预算总量不同，不能当作等总 token、等墙钟预算下的 kernel 性能比较。

## 资源拒绝重放与下一步

原检查只保留错误尾部，长 TIR 可能挤掉开头的具体约束信息。新增运行内脚本 `recheck_resource_failures.py`，在隐藏 GPU 的 CPU 进程中逐条恢复这 49 个 state，仅重放 lowering 和 `VerifyGPUCode`，不重新编译或测量。49 个资源拒绝全部复现，保存完整错误文本及 SHA256，耗时 29.28 秒，详见 [资源重放凭证](../../audit_runs/4090_multicandidate_20261111_01/resource_recheck/report.json)。

以下类别按不同 state 互斥划分：

| 资源拒绝类型 | 自由生成 | 状态约束 |
| --- | ---: | ---: |
| 仅共享内存超限 | 3 | 10 |
| 仅向量宽度超限 | 5 | 9 |
| 两项同时超限 | 6 | 16 |
| 合计 | 14 | 35 |

实际 task 门禁为共享内存每 block 49,152 字节、向量宽度 16 字节；本表沿用冻结 task 和实际后端的参数，不能据 GPU 产品规格放宽限额。约束模式共有 26 个 state 触及共享内存、25 个触及向量宽度，二者重叠 16 个，不能直接相加为拒绝总数。本轮未发现 thread/vthread 超限，但不能据此保证其完整覆盖。

后续按以下顺序推进：

1. 先根据这些可复现 state 追踪向量化步骤与分块因子的依赖，把向量字节数可行性和共享内存可行性尽量提前到候选构造/解码阶段。保留最终 lowering 门禁，不通过放宽硬件限额或事后静默改写调度来提高通过率。
2. 用相同 8 次尝试、K=4 设计复验，重点检查编号 0、8、11 的短缺与既有 workload 是否退化，再扩展到同 workload 的多个 validation sketch。本轮仍只是每 workload 一个 median sketch，尚未完成多 sketch 验收。
3. 候选供给稳定后，在训练侧候选上补充独立算子参考和真实延迟比较，再决定等监督 token、多 seed 的训练对照。当前仍没有 kernel 延迟、整网加速或性能多样性结论，不据此扩大训练。

## 回归

统计测试覆盖重复候选消耗预算但不增加 K、模式之间不混合计数、资源/编译/输出检查分层、达到 K 后继续统计全部尝试、宏平均包含零候选 workload，以及候选身份保留分块因子并排除测量结果。与既有约束生成回归共 16 项通过，耗时 4.25 秒。新增 Python 文件按 Black 100 列格式化，`git diff --check` 通过。

## 复现

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/qsy/anaconda3/envs/tlm/bin/python -u LLM/__main__.py validate-candidates \
  --run-dir LLM/audit_runs/4090_multicandidate_20261111_01 \
 --gpus 0 1 2 3 --attempts 8 --required 4
```

汇总及资源拒绝重放命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  LLM/audit_runs/4090_multicandidate_20261111_01/collect_report.py

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/qsy/anaconda3/envs/tlm/bin/python -u \
  LLM/audit_runs/4090_multicandidate_20261111_01/recheck_resource_failures.py
```

重放脚本要求输出目录尚不存在；已有重放结果及完整错误保留在 `resource_recheck/`，再次诊断须先为脚本选择新的输出目录。

重跑必须改用新目录。本轮数据始终只用于 validation；不进入 SFT、DPO/LAPO，不使用最终 hold-out。
