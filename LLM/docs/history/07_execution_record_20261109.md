# 计划 5、6 与总体优化方案执行记录

执行日期：2026-11-09。本文时间使用北京时间；JSON 产物保留 UTC 时间戳。

后续定位：同日继续完成 37 个生成诊断/干预案例，确认 matmul 的线程 extent=16<32、长卷积的 TBS 固定字段和 PRS 格式错误，以及 52 次生成均无 EOS 的终止行为；详见 [生成失败定位记录](08_generation_failure_diagnosis_20261109.md)。这些是独立后续结果，不改写本轮原始模型成绩。

## 1. 本轮结果与完成范围

本轮完成了超时定位与补验、共享生成协议和评估修复、两组实际短训练，以及小规模独立正确性检查和 GPU 延迟测量。**验证链路取得了可复核的结果，但短训练尚未证明模型质量提升，也没有整网加速结论。**

| 工作 | 执行前状态 | 本轮结果 | 验收范围 |
| --- | --- | --- | --- |
| 计划 5：源记录审计、正式重建 | 已完成，但文档进度过期 | 重新核验冻结 raw/Arrow/tokenizer 摘要，全部匹配 | 591,126 条训练/验证记录，无截断 |
| 计划 5：真实解析与构建回放 | 原任务 6,501 通过、6 超时，已于 11 月 2 日结束 | 新冻结运行 18/18 通过，覆盖原 6 条超时和 12 条同 workload 对照 | 关联两轮证据后，6,507 条代表样本、2,169 个 workload 均有通过凭证 |
| 计划 6：生成与结构评估一致性 | 预算、上下文与回调行为不同 | 共用生成实现；新增有效候选指标、RNG 隔离与跨进程指标/异常同步 | 单元回归、实际 GPU 自由生成试点；CPU Gloo 验证同步 |
| 计划 6：受控训练 | 尚无新对照 | 基线 3/4、原来源 1/4、过滤来源 2/4 个 workload 有可构建候选 | 各 256 条、32 步、4 个验证 workload 的试点 |
| 总体方案：真实性能监督 | 正式重建集没有有效延迟 | 9 条历史候选独立正确性检查通过并完成测量，生成 2 对偏好样本 | 3 个训练 workload；不是试点模型的新生成候选 |
| 回归验证 | 已有数据门禁和 LAPO 回归 | 101 项测试通过 | 98 项普通回归、2 项双进程 Gloo、1 项回调恢复回归 |

关联验收报告：[report.json](../../audit_runs/4090_plan56_execution_20261109_01/report.json)。逐样本凭证索引：[combined_representatives.jsonl](../../audit_runs/4090_plan56_execution_20261109_01/combined_representatives.jsonl)。原目录、数据和旧任务状态未改写。

## 2. 起点与历史任务核验

实际 checkout 为 `/data3/qsy/complier/llm_compiler`，与用户给出的 `/home/qsy/workspace/complier/llm_compiler` 指向同一目录。

| 任务 | 完成时间 | 已核实结果 |
| --- | --- | --- |
| 全量源记录审计 `4090_plan5_full_fast_20261022_01` | 10 月 24 日 23:42:29 | 2,281,247 条、2,169 workloads；591,126 通过，1,670,170 构建失败，19,951 超时 |
| 正式重建 `4090_plan5_rebuild_20261102_02` | 11 月 2 日 15:36:39 | train 556,948 条 / 2,061 workloads；validation 34,178 条 / 108 workloads；最大长度 1,189 / 1,177，max_length=2048，无截断 |
| 原代表性回放 `4090_plan5_roundtrip_full_20261102_01` | 11 月 2 日 19:50:38 | 6,507 条全部处理；6,501 通过、6 超时；47 条进入重试、41 条恢复；最终 `FAILED` |
| 本轮正式补验 `4090_plan5_diagnosis_20261109_05` | 11 月 9 日 16:33:16 | 18/18 通过；原 6 条 unresolved 全部通过 |
| 本轮训练试点 `4090_plan6_controlled_20261109_02` | 11 月 9 日 16:34:14 | 两组训练和三组自由生成评估完成，`SUCCEEDED` |

执行前最近一次完整结束的任务是 11 月 2 日的代表性回放，距本轮约 7 天；它正常结束并发布了失败汇总，不能继续描述为“仍在运行”。原脚本复用已有失败 journal，直接用相同命令续跑不会重新检查这 6 条，因此本轮使用独立目录。

实际 TVM Python 为 `/data3/qsy/complier/tlm/python/tvm`，动态库为 `/data3/qsy/complier/tlm/build/libtvm.so`，目标为 RTX 4090 / `sm_89`。本次没有将其他 TVM checkout 的构建误称为当前仓库新编译的库。编译器、CUDA/LLVM、动态库与 tokenizer 身份见各 manifest。

## 3. 超时诊断与正式补验

### 3.1 执行过程与失败尝试

新增 [diagnose_generation_roundtrip.py](../../experiments/diagnose_generation_roundtrip.py) 和 worker 阶段记录，在新目录中只读关联原输入及结果。

| 新运行后缀 | 过程与结果 | 是否作为正式补验凭证 |
| --- | --- | --- |
| `_01` | 子进程模块路径缺失，18 条在实际样本检查前失败；修复 worker `PYTHONPATH` | 否，环境初始化失败 |
| `_02` | 原 120 秒整样本预算，14/18 通过；12 条对照全部通过；4 条超时均已完成 state、文本与 TIR 等价性比较，停在构建导出 | 仅定位阶段 |
| `_03` | 重建 120 秒、构建 60 秒，原 6 条中 5 条通过；运行期间脚本被编辑，结束身份检查拒绝发布正式汇总 | 否，保留诊断记录 |
| `_04` | 单查 row 69103，重建/构建各 120 秒，通过；端到端 186.41 秒 | 单样本诊断 |
| `_05` | 全程冻结代码，重建/构建各 120 秒，两个池各 2 个 worker，18/18 通过 | 是，正式补验 |

`_05` 端到端 382.99 秒，约 6.4 分钟。逐样本阶段在操作前原子写盘，包含最后阶段和累计耗时；超时回收后仍可定位阶段。使用持续收集与补位、独立可回收的重建/构建池，保留所有比较和真实模块导出检查。

### 3.2 原 6 条超时的结果

下表行号是 train raw 的零起始索引。重建耗时按阶段终点取值，包含阶段记录开销；build 为实际构建器报告的编译导出耗时。

| row | token 长度 | 重建与等价性检查（秒） | 编译导出（秒） | 结果 |
| ---: | ---: | ---: | ---: | --- |
| 33093 | 801 | 88.97 | 55.77 | PASSED |
| 69103 | 791 | 109.63 | 66.14 | PASSED |
| 126355 | 1,160 | 88.43 | 48.53 | PASSED |
| 182730 | 515 | 93.29 | 53.03 | PASSED |
| 284504 | 1,145 | 90.21 | 47.56 | PASSED |
| 549831 | 785 | 78.58 | 41.08 | PASSED |

两次独立 `tvm.lower` 在正式补验的这 6 条上合计约 77.5–108.3 秒，随后仍需实际编译/导出。原整样本上限不能容纳这些累计成本；未观察到 token、解析、规范化 state、有效 state、格式化文本或 TIR 不一致。

**本次修正的是验证预算与诊断方式，没有消除 lowering 本身的长尾，也没有证明执行器提速。** 两阶段总允许时间增加，并明确记录；未跳过 lowering 或构建以换取通过。旧验证器仍保留原批处理逻辑；新执行方式目前用于这 18 条补验，尚未迁移为全量通用验证器，也没有同预算吞吐 A/B。

### 3.3 两轮凭证如何关联

- 原 manifest：`0413f414da23c0156affe890e87c4f2483640605863cee4e6fc37288c29fac94`。
- 原 results SHA256：`e73bf156fcb8104603933f14470cdec65ccbb123893c03ab013abd874e5bcd92`。
- 新 manifest：`6001f5b26018e77e67649146c98ac2158e88789d854bcd97d61ca88c580cfa9c`。
- 新 results SHA256：`d5297e46375dee97c35ea9ec30bba2ef9ee2245f4b00b6c5423082a9ca3d6eca`。

汇总脚本重新核验两份 manifest、自带摘要的每条结果、原样本与补验样本的全部字段、source_input_hash、冻结代码副本，以及正式 raw/Arrow/tokenizer 摘要。原 6,501 条复用原通过凭证，另外 6 条引用新通过凭证；train 6,183、validation 324，总计 6,507 条，覆盖 2,169 个 workload。

该验收是**已知后缀回放、解析、state/TIR 等价性和真实编译导出**，不等于全量 591,126 条回放，不等于模型自由生成成功率，也没有逐条 GPU 输出和延迟保证。

## 4. 代码改动与验证

| 文件 | 改动及目的 |
| --- | --- |
| [generation_roundtrip_worker.py](../../validation/generation_roundtrip_worker.py) | 阶段原子写盘；允许将重建和 LocalBuilder 编译导出分开执行 |
| [diagnose_generation_roundtrip.py](../../experiments/diagnose_generation_roundtrip.py) | 核验源身份、选取 unresolved 和兄弟对照、冻结代码、阶段硬超时、持续收集、结束身份复查 |
| [generation_protocol.py](../../generation/generation_protocol.py) | 公共 token 预算、prompt 编码、上下文限制、生成和 continuation 解码；Python/NumPy/PyTorch RNG 隔离 |
| [gen_state.py](../../generation/gen_state.py)、[eval_struct.py](../../validation/eval_struct.py) | 生产与评估调用共享生成实现；policy 每次请求决定阶段预算，显式 cap 默认不设；默认 minimum/scale 统一为 256/4 |
| [eval_struct.py](../../validation/eval_struct.py) | 候选去重后构建，确认导出非空；增加唯一可构建数宏平均、K 达标率、token 成本与回调数；空评估和异常显式失败 |
| [run_train_qwen3_clm.py](../../train/run_train_qwen3_clm.py)、[train_qwen3_clm.py](../../train/train_qwen3_clm.py) | 配置结构评估后默认启用构建并按 `eval_struct_unique_buildable_macro` 选模型；缺指标配置时失败；普通未配置路径仍用 accuracy |
| [eval_struct.py](../../validation/eval_struct.py) | rank 0 执行结构评估，结果或异常同步到所有 DDP rank；恢复模型 train/cache 状态 |
| [run_plan6_controlled_experiment.py](../../experiments/run_plan6_controlled_experiment.py) | 同基础模型和 workload 的原来源/过滤来源小规模训练与生成评估 |
| [measure_plan6_representatives.py](../../experiments/measure_plan6_representatives.py) | 独立 NumPy 输出参考、CUDA 执行和标准测量记录保存 |

验证合计 **101 项通过**：六个 `test_llm_*` 文件普通回归 98 项（13.32 秒）；双进程 CPU Gloo 2 项（27.94 秒）；追加的结构回调指标写入、异常恢复测试 1 项（4.10 秒）。生产/评估使用相同 prompt IDs、实际 generate 参数、上下文 cap 和解码的回归已通过。分布式验证覆盖指标与异常同步、LAPO padding 去重；未据此声称完成 NCCL 多卡生成、FSDP 或 DeepSpeed 验收。

格式使用 Black 100 列；仅处理本轮涉及的文件。没有提交、覆盖旧 checkpoint 或更改原正式数据。

尚未补全的评估项包括逐阶段解析损失、峰值显存及更细的短缺统计。`fallback_rate` 是历史字段名，当前表示零解析 workload 比例，评估没有随机补齐；批量生成的 token 计数包含矩形输出中的 padding，本试点 batch=1，不受此影响。

## 5. 实际训练对照：尚未证明改善

正式运行：[4090_plan6_controlled_20261109_02](../../audit_runs/4090_plan6_controlled_20261109_02)。首轮 `_01` 因当前 Accelerate 不支持显式 `data_seed`，在训练第一步之前失败；去掉该参数后由 Trainer 的 seed=42 控制普通 sampler。该尝试保留，未计入结果。

配置：

- 基础模型与 tokenizer：`/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1-v2-ext-b2`。
- GPU 0（独占）；两个分支分别从同一基线加载，各选相同 256 个训练 workload、每 workload 1 条。
- 旧来源为原始 raw，过滤来源为正式重建 raw；两组均完整重编码、max_length=2048、无截断。因此对照因素是来源过滤，不是旧 1024 截断与新长度的比较。
- seed=42，batch=2，梯度累积=4，32 optimizer steps，lr=5e-6、constant，BF16、gradient checkpointing，PPT 前缀不参与 loss。
- 4 个验证 workload：softmax、长卷积、matmul、batch matmul；每个 1 个固定 sketch、greedy、无 fallback、K=1、阶段 token cap=256、build timeout=30 秒。没有使用最终 hold-out。
- 本次为固定行数和步数试点；两组监督 token 分别 79,459 / 79,029，约差 0.54%，没有做到精确等 token 预算。

| 模型 | 可构建候选 workload | K=1 达标率 | 唯一可构建数宏平均 | 生成 token | 评估耗时（秒） | train loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练前基线 | 3/4 | 75% | 0.75 | 2,816 | 70.21 | — |
| 原来源训练 32 步 | 1/4 | 25% | 0.25 | 1,792 | 42.34 | 0.14145 |
| 过滤来源训练 32 步 | 2/4 | 50% | 0.50 | 2,048 | 49.37 | 0.12880 |

| 验证 workload 类别 | 基线 | 原来源 | 过滤来源 |
| --- | ---: | ---: | ---: |
| softmax | 1 | 0 | 1 |
| 长卷积 | 0 | 0 | 0 |
| matmul | 1 | 0 | 0 |
| batch matmul | 1 | 1 | 1 |

所有解析出的唯一候选均编译成功；本试点主要损失发生在生成/解析。过滤组比原来源多保留一个 softmax，但两组都丢失基线可生成的 matmul。loss 下降不能作为自由生成改善的证据。评估更快也伴随更少的 token 和更多失败，不能解释为效率提升。

**结论：这两个 checkpoint 仅为试点产物，不替换基线，不推荐据此扩大训练。** 单 seed、4 个 workload、256 条和一轮训练不能形成统计或因果结论；cap=256 的结果也不能直接当成无 cap 生产结果。下一轮应先扩展固定验证集合、精确控制监督 token，再做多 seed 和较温和的训练更新。

完整指标及逐 workload 结果：[evaluation.json](../../audit_runs/4090_plan6_controlled_20261109_02/evaluation.json)。训练 argv、token 统计、日志和 checkpoint 位于各 `original/`、`filtered/` 目录。该试点没有启动时的完整代码副本，不能宣称与补验相同级别的代码冻结；汇总报告保存的是执行后的代码和产物摘要，明确标注这一限制。

## 6. 独立正确性、真实测量与偏好产物

运行：[4090_plan6_measurement_20261109_01](../../audit_runs/4090_plan6_measurement_20261109_01)，11 月 9 日 16:31:26 完成，耗时 16.11 秒。

GPU 1 独占执行，`CUDA_VISIBLE_DEVICES=1` 后脚本内设备编号为 0。目标使用来源记录的 `sm_89`。选取 3 个训练 workload 的短/中/长代表，共 9 条**历史已知候选**，不是训练试点模型生成的结果。

独立 float32 NumPy 参考：matmul+bias 为 `A @ B.T + bias`；NN batch matmul 为 `np.matmul(A, B)`；NHWC 全局平均池化为输入第 1、2 维的均值。输入固定 seed=42、均匀分布 [-0.5, 0.5]，rtol=1e-3、atol=1e-4；预热 1 次，number=10、repeat=5、min_repeat_ms=100。

| 类别 | train row | 平均延迟（微秒） | 最大绝对误差 | 同前缀组 |
| --- | ---: | ---: | ---: | --- |
| matmul+bias | 456 | 9.433 | 2.38e-6 | A |
| matmul+bias | 741 | 22.552 | 4.29e-6 | B |
| matmul+bias | 806 | 107.086 | 4.29e-6 | B |
| 全局平均池化 | 306403 | 4.961 | 2.98e-8 | C |
| 全局平均池化 | 306653 | 4.925 | 2.98e-8 | C |
| 全局平均池化 | 306902 | 5.068 | 2.98e-8 | C |
| batch matmul | 359341 | 7.386 | 0 | D |
| batch matmul | 359559 | 10.157 | 0 | D |
| batch matmul | 359776 | 51.235 | 0 | D |

全部 9 条通过本次输入的正确性检查，保存标准 `MeasureResult` 和 5 次重复延迟。没有与默认 schedule、AutoScheduler 最优或 Relay 基线比较，因此上表不能称为模型加速比，也不能推算整网收益。

使用同 PPT 前缀、延迟比至少 1.5 的规则，9 条记录构成 4 个组，得到 2 对训练偏好：matmul row 741 对 806（约 4.75 倍差异）；batch matmul row 359341 对 359776（约 6.94 倍差异）。其余 3 个备选配对因差距不足过滤。matmul row 456 与另两条前缀不同，不能直接组成该训练对。

正式偏好产物：[preferences_sm89/train_pairs.jsonl](../../audit_runs/4090_plan6_measurement_20261109_01/preferences_sm89/train_pairs.jsonl)，统计：[build_stats.json](../../audit_runs/4090_plan6_measurement_20261109_01/preferences_sm89/build_stats.json)。首次 `preferences/` 构建未显式指定 arch，环境推断成 `sm_50`，仅保留排错；以显式 `-arch=sm_89` 的 `preferences_sm89/` 为准。已按真实训练 split 选取并排除 hold-out，每文件仅一个 workload，valid_percentage=0。两对样本不足以支持正式 LAPO，未启动该训练。

## 7. 关键执行与复核命令

从仓库根目录运行。下面是本轮参数；重跑必须使用新目录名，以免与已有身份冲突。训练详细 argv 另保存在分支 `command.json`。

```bash
PYTHONDONTWRITEBYTECODE=1 /home/qsy/anaconda3/envs/tlm/bin/python -u LLM/__main__.py diagnose-roundtrip \
  --source-run LLM/audit_runs/4090_plan5_roundtrip_full_20261102_01 \
  --run-dir LLM/audit_runs/4090_plan5_diagnosis_20261109_05 \
  --sample-timeout 120 --n-parallel 2 --separate-build --build-timeout 120

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 /home/qsy/anaconda3/envs/tlm/bin/python -u LLM/__main__.py controlled-experiment \
  --run-dir LLM/audit_runs/4090_plan6_controlled_20261109_02 \
  --rebuild-dir LLM/audit_runs/4090_plan5_rebuild_20261102_02 \
  --source-run LLM/audit_runs/4090_plan5_roundtrip_full_20261102_01

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 /home/qsy/anaconda3/envs/tlm/bin/python -u LLM/__main__.py measure-representatives \
  --run-dir LLM/audit_runs/4090_plan6_measurement_20261109_01 \
  --source-run LLM/audit_runs/4090_plan5_roundtrip_full_20261102_01

PYTHONDONTWRITEBYTECODE=1 /home/qsy/anaconda3/envs/tlm/bin/python LLM/__main__.py build-pairs \
  --target 'cuda -model=4090 -arch=sm_89' \
  --dataset_path LLM/audit_runs/4090_plan6_measurement_20261109_01/measure_records \
  --save_path LLM/audit_runs/4090_plan6_measurement_20261109_01/preferences_sm89 \
  --min_latency_gap_ratio 1.5 --max_prompt_tokens 2048 --max_suffix_tokens 2048 \
  --valid_percentage 0 --num_workers 1

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home/qsy/anaconda3/envs/tlm/bin/python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/plan6_final_regression_20261109 \
  tests/python/unittest/test_llm_generation_protocol.py \
  tests/python/unittest/test_llm_generation_roundtrip.py \
  tests/python/unittest/test_llm_audited_rebuild.py \
  tests/python/unittest/test_llm_schedule_audit.py \
  tests/python/unittest/test_llm_schedule_dataset.py \
  tests/python/unittest/test_llm_lapo_training.py \
  -k 'not distributed_evaluation_removes_padding_rows and not structural_metrics_and_errors_reach_all_training_ranks'
```

最后追加的回调恢复测试单独通过。两项 CPU 分布式测试需要本机通信端口，在已获批准的沙箱外执行；命令及结果保存于汇总目录 `validation.json`。CUDA 任务同样在能够访问实际 GPU 的已授权环境执行。

## 8. 距离总体完成还缺什么

1. **模型质量仍未过关。** 扩展固定 validation、记录每阶段首次解析失败和短缺；先定位长卷积与 matmul 的失败，再开展精确等监督 token、多 seed、小更新幅度的对照。EOS、workload 重平衡及模板覆盖应分别隔离变量，尚未执行这些正式对照。
2. **性能训练数据规模不足。** 扩展训练侧独立正确性检查与同前缀实测，形成足够规模的优质 SFT/偏好数据；当前 2 对只证明测量到偏好转换能运行。
3. **固定预算的最终验收尚缺。** 在不参与选模型的 hold-out 上比较唯一有效候选、K 达标率、GPU 正确性和最优延迟，再完成 `opt_level=3` 的 Relay 整网比较。
4. **验证吞吐优化尚缺量化。** 将诊断执行方式推广到生产验证器前，补齐有界构建队列、重试策略、恢复语义和同条件吞吐/资源 A/B；本次没有承诺加速倍数。

计划 5 的数据基础和代表性回放已取得可关联的通过证据；计划 6 完成 P0 关键实现及试点，P1 质量与性能提升仍需实验。整体优化方案因此仍是部分完成，当前没有足够证据把新训练模型作为最终成果发布。
