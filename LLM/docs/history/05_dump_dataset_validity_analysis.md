# 原始 dump 数据集有效性与可构建性核查

**2026-11-09 状态更新：**源记录审计与正式 raw/Arrow 重建已完成，正式集为 591,126 条、2,169 个 workload，无截断。11 月 2 日回放已结束，原结果为 6,501 通过、6 超时，状态 `FAILED`。本次保留原记录，在独立冻结运行中补验 6 条超时和 12 条对照，18/18 通过；关联两次凭证后，6,507 条代表样本均有真实解析、state/TIR 等价性和编译导出证据。该结论只覆盖代表样本，不等于全量 GPU 正确性或性能验收。

完整过程、阶段耗时、训练对照及实测见 [11 月 9 日执行记录](07_execution_record_20261109.md)；机器可核验的关联报告见 [report.json](../../audit_runs/4090_plan56_execution_20261109_01/report.json)。下文日期和进度快照保留为历史记录，其中“仍在运行”不代表当前状态。

核查日期：2026-10-21。

核查目录：`/home/qsy/workspace/dataset`，实际指向 `/data3/qsy/dataset`。

## 1. 结论

**最初 dump 的记录不是全部经过编译验证的数据，也不能认为全部可构建。已经找到原始文件中的同一 workload、同一调度 state 后来发生编译失败的实例。**

需要区分五层条件：

1. JSON 可以解析、字段可以读取。
2. 调度记录能够恢复为 TVM state，变换步骤能够应用。
3. 在指定 TVM、target、CUDA/LLVM 环境中能够构建并导出模块。
4. 在目标设备上能够运行，并经过独立的参考输出正确性检查。
5. 有可信的实测延迟，且相对基线或其他候选有收益。

这次对候选和测量记录的全量 JSON 扫描没有发现解析错误，但这不代表其余条件成立。没有逐条重新编译全部候选，不能给出全量当前可构建率。

## 2. 当前目录的逐行统计

下表按每个 workload 的 JSONL 文件扫描。`network_info` 的 pickle 不计入调度记录数；`lapo_dataset_v1` 的偏好 JSONL 也不与原始调度相加。

| 目录（相对 dataset） | JSON 文件数 | 调度记录数 | 已核实的含义 |
| --- | ---: | ---: | --- |
| `to_measure_programs/4090` | 2,458 | 2,581,346 | 全部为零延迟占位记录，不代表已通过构建 |
| `to_measure_programs/v100` | 2,374 | 2,498,119 | 全部为零延迟占位记录，target 为 sm_70 |
| `to_measure_programs/qwen3_0_6b_seq128` | 11 | 10,760 | 全部为零延迟占位记录，不能单凭记录证明可构建 |
| `to_measure_programs/llama3_2_1b_seq128` | 9 | 9,773 | 同上 |
| `to_measure_programs/llama3_2_1b_seq128_last_token` | 9 | 9,569 | 同上 |
| `to_measure_programs_build_valid/4090` | 246 | 172,935 | 日志有构建过滤证据，但仅为部分 workload，且没有实测延迟 |
| `measure_records/4090` | 550 | 8,958 | 混合保存成功与失败记录，需要按错误码和耗时共同过滤 |
| `measure_records/qwen_4090` | 0 | 0 | 当前为空，没有可用测量记录 |

五个原始 `to_measure_programs` 子目录合计 **5,109,567** 条，均为 `costs=[0]`、`error_no=0` 的占位格式。除 v100 为 sm_70 外，上表候选与测量记录中的 target 均为 sm_89；不能把两种 target 的验证结果互相替代。

`network_info/4090/all_tasks.pkl` 当前含 2,458 个任务，v100 为 2,374 个，三个独立 Qwen/Llama registry 分别为 11、9、9 个。`gen/dump_network_info.py` 提取并保存 Relay 和任务信息，不等于已经证明每个随机调度或者完整模型能够成功构建。

## 3. 为什么旧 dump 的 error_no=0 不能表示构建成功

### 3.1 历史 dump 没有调用构建器

核对历史版本 `git show 5bf5f3e:gen/dump_programs.py`，其流程为：

```text
sample_initial_population
  -> 按 state 字符串去重
  -> MeasureResult([0.0], 0, "", 0, timestamp)
  -> save_records
```

其中 `error_no=0` 是脚本填入的默认值，不是构建器返回的结果。

`sample_initial_population` 内确实执行初始化规则、InferBound、PruneInvalidState 和特征/代价模型筛选，见 [sketch_policy.cc](../../../src/auto_scheduler/search_policy/sketch_policy.cc)。这些检查不能替代完整的 `tvm.build` 和模块导出。因此不能把这些记录称为完全随机的垃圾数据，也不能把它们称为已验证可构建的正样本。

### 3.2 当前 dump 的改进不会追溯清洗旧文件

当前 [gen/dump_programs.py](../../../gen/dump_programs.py) 默认启用 `LocalBuilder`，只接受 `BuildResult.error_no == 0` 的 state。这能提高新生成数据的可靠性，但有三个边界：

- 目标文件存在且没有 `overwrite` 时直接返回 `skipped_existing`，不会逐条复核旧文件。
- CLI 允许 `--no-validate-build`，关闭构建过滤。
- 保存时仍使用零延迟占位记录，且关闭构建过滤时也调用相同保存函数。当前 `build_valid_unmeasured` 错误消息不会写入标准记录：见 [measure_record.cc](../../../src/auto_scheduler/measure_record.cc)，`r` 仅序列化 costs、error_no、all_cost、timestamp。

因此，**旧未验证记录、新构建通过记录、显式关闭验证后保存的记录，可能具有相同的 `r` 外观。** 后续不能只依靠文件夹名称、`error_no=0` 或消息字符串判断构建 provenance，需要独立审计结果和运行配置。

## 4. 实际测量记录中的失败证据

`measure_records/4090` 的 8,958 条记录分布如下：

| error_no | 代码定义 | 条数 | 解释 |
| ---: | --- | ---: | --- |
| 0 | NO_ERROR | 2,718 | 本次扫描中均有有限、正值且小于失败占位阈值的延迟 |
| 2 | COMPILE_HOST | 6,028 | `tvm.build` 或模块导出阶段失败；名称含 HOST，但也包括 CUDA 目标构建失败 |
| 6 | BUILD_TIMEOUT | 151 | 构建超时，不能一律等同于调度本身不可构建 |
| 4 | RUNTIME_DEVICE | 61 | 已到设备执行阶段但运行失败 |
| 合计失败 | | 6,240 | 历史测量子集的失败记录 |

成功记录占此测量子集的 **30.34%**。这些记录不是从全部原始数据中均匀随机抽样，且可能包含后续生成的不同 state，**不能声称原始 258 万或 510 万条数据有 69.66% 不可构建。**

### 4.1 原始占位记录与后续失败的直接对应

以规范化 workload key 加完整序列化 state 为键，已经找到以下原始记录和历史 `error_no=2` 的对应：

```text
原始目录：to_measure_programs/4090
测量目录：measure_records/4090
同名文件：
([f5b447dcba2b43cc270d7c2062043946,[1,1,1,672],[1,1,672,168],[1,1,1,168],[1,1,1,168]],cuda).json

原始第 2 行：   r = [[0], 0, 0, 1758609502]
对应测量记录： r = [[10000000000.0], 2, 0.782843, 1762852945]

原始第 109 行： r = [[0], 0, 0, 1758609502]
对应测量记录： r = [[10000000000.0], 2, 0.500974, 1762852951]
```

这不是仅凭相同文件名作推断，而是比较了 workload 和 state；对上述两条实例进一步核对，整个 `i` 输入字段也完全相同，包括完整 target。它证明旧 dump 中的零错误码没有保证后续构建成功。历史错误消息没有保存在标准记录中，仅凭这些 JSON 无法还原当时具体的编译异常。

对 550 个测量文件所对应的原始文件逐行匹配，找到 845 条原始记录与测量记录对应，其中 843 条对应成功、2 条对应上述构建失败；另有 8,113 个测量 state 在同名原始文件中未匹配。该匹配结果进一步说明不能把测量目录的全部 6,240 条失败都视作旧 dump 中的失败，也不能以匹配子集推算原始全集的通过率。

没有在现有 train/validation raw 文本中检索到上述示例 workload；不能把这两个具体例子说成已经进入了当前正式训练集。训练入口没有构建门禁所带来的风险，需要对训练集的实际 source records 继续验证。

### 4.2 本轮实际复编译结果

本轮读取原始记录，注册对应 workload，经 `recover_measure_input(rebuild_state=True)`、`apply_steps_from_state` 后执行 `tvm.build`，没有在 GPU 上运行或计时，也没有为成功样本执行模块导出测试。

| 原始样本 | 历史测量 | state 应用 | 本轮 tvm.build |
| --- | --- | --- | --- |
| 上述 `f5b447...` 文件第 2 行 | error_no=2 | 成功 | 失败，LowerThreadAllreduce 断言 |
| 上述 `f5b447...` 文件第 109 行 | error_no=2 | 成功 | 失败，LowerThreadAllreduce 断言 |
| `00a059b856ac30ac172b6252254479a6` workload 文件第 5 行 | error_no=0 | 成功 | 成功 |
| 同一 `00a059...` 文件第 9 行 | error_no=0 | 成功 | 成功 |

两个失败的关键异常分别来自 `src/tir/transforms/lower_thread_allreduce.cc:481` 和 `:442`：

```text
Check failed: (!load_remap_.count(buffers[idx]->data.get())) is false
Check failed: (!load_remap_.count(buffers[i]->data.get())) is false
```

它们说明当前环境的线程归约 lowering 无法处理这两个 state，不是因为所有 CUDA 构建都不可用。仅凭这个断言还不能断定是调度语义本身错误，还是底层编译器对该调度形式的实现限制；对于当前工具链下的可构建训练源，两种情况都需要过滤或隔离。

复编译环境为：

```text
Python: /home/qsy/anaconda3/envs/tlm/bin/python
tvm Python: /data3/qsy/complier/tlm/python/tvm/__init__.py
libtvm: /data3/qsy/complier/tlm/build/libtvm.so
CUDA target: sm_89，原记录的完整 target
```

特别注意：该环境实际加载的是相邻 `tlm` checkout 的 TVM Python 与共享库，并非证明本仓库当前源码已经完整重新构建。正式全量验证前必须固定实际使用的 Python 包、共享库与编译器指纹，并使其与训练后生成/测量环境一致。

## 5. build_valid 目录能否直接替代全部训练源

目前不能。

- 该目录仅有 246 个 workload 文件，约为 4090 registry 的 10.01%，远未覆盖全部 2,458 个任务。
- 日志 `dump_programs_0_all.log` 记录了构建拒绝，包含 72,204 个 `.E` 和 391 个 `.T` 标记。它支持“运行过实际构建过滤”的判断；不能把日志标记数直接当作全部原始数据的失败率。
- 日志的最后进度为 `246/2458`，第 247 个 workload 的最后可见轮次仅接受到 `680/1000`；没有发现最终 `dump_programs_stats_*.json`。当前快照没有全量完成证据。
- 日志中至少一个任务在 20 轮后只得到 759/1000 条，说明“有文件”也不保证样本数达标。
- 此流程重新采样新候选，不能直接把这 172,935 条解释成旧数据逐条过滤后的全部通过集合。
- 数据依然没有实测延迟，构建通过也不等于运行正确或性能优秀。

可作为有日志支撑的局部候选源，后续仍需补充逐记录验证身份、环境、覆盖统计和抽样复核；不能直接替换全量源并宣称数据集已修好。

## 6. 对当前预训练问题的影响

本计划实施前，[LLM/data/make_dataset.py](../../data/make_dataset.py) 的 `for_gen` 主要检查 PPT、suffix 长度和重复项，没有逐条调用构建器，也没有以 `r[1]` 过滤失败记录。没有有效延迟的记录仍会保留，并获得 `labels=1.0`。第 9 节记录了本轮新增的过滤和审计入口。

其中 `labels=1.0` 是 raw 中的相对延迟字段，后续 CLM 实际使用 token labels；这里没有基于可构建性降低无效调度的监督强度。更明确的风险是：**未验证的候选可以作为正常调度文本进入结构学习。**

原 `_get_latency` 只选取正数，没有检查 `error_no`、有限值或失败哨兵值。若直接向这个入口提供测量目录，`1e10` 等失败占位值可能被当作有效延迟。本轮已统一严格延迟校验；未测量候选仍需独立通过构建审计。

[LLM/data/rebuild_clm_dataset.py](../../data/rebuild_clm_dataset.py) 解决完整编码和 PPT 边界，不能修复源 state 的编译合法性。此前 2,281,247 条 raw 的数量、hold-out 隔离和 token 完整性审计，也不能证明这些调度全部可构建。

这构成模型难以生成合理调度的另一个有代码证据支持的风险因素，但尚不能从现有统计量化它的因果贡献；仍需用经过门禁的数据与原始候选数据做控制实验。即使全部候选可构建，随机结构监督也不会自动教会模型选择最快的优化组合。

## 7. 建议的处理顺序

### 7.1 先审计原始记录，不覆盖原始目录

第一轮优先处理正式 4090 训练来源，保留原 train/validation workload 划分及 hold-out 排除规则。4090、v100 和自定义模型使用各自 registry 与完整 target。

为每条原始记录生成验证键，至少包括规范化 workload、完整 target/host、完整 state、相关任务输入与布局配置，以及 TVM 构建和工具链指纹。只用 workload 或文件名做键不够。

输出按 workload/shard 持久化的 sidecar，包含：

```text
source_file / source_line / input_hash
workload_key / target / compiler_fingerprint
parse_status / state_restore_status / build_status
error_no / error_message / duration / retry_count
```

使用同一环境执行 `recover_measure_input`、`apply_steps_from_state`、`LocalBuilder` 构建和导出；记录失败原因。超时、OOM、缺工具链等环境失败独立归类并按策略复试，不能全部归咎于调度语义。

### 7.2 构建正式结构训练源

- 仅保留在目标环境通过构建门禁的原始候选作为正式正样本，并保留可追溯验证键。
- 单独保存失败和超时记录供诊断，不能把它们作为正常可行调度监督；未来可以另做带标签的合法性或修复任务。
- 统计每个 workload 的通过数、缺失数与失败类型。缺候选时只对缺口重新采样并验证，不能用零值或未经验证记录补齐。
- 应验证训练文本的序列化/反序列化和生成重建过程；原始 state 的验证不能替代文本往返验证。
- 对过滤后的样本重新完整编码。过滤版的样本数允许减少，验收应使用过滤清单中的通过数，不能继续要求保留全部 2,281,247 条。

### 7.3 再补运行正确性与性能监督

在独占目标 GPU 上对候选做测量，并独立加入与参考实现的输出比较。`LocalRunner` 当前计时代码随机填充输入并执行 time_evaluator，没有在该路径做参考输出比较，见 [measure.py](../../../python/tvm/auto_scheduler/measure.py)。因此成功计时不能自动视为数值正确性的证明。

从可信延迟记录中选择优秀调度做后缀 SFT，再构建同 prompt 的 DPO/LAPO 偏好对。进入性能监督的数据应同时满足状态/构建门禁、运行与正确性要求、`error_no == 0`，以及有限、正值、非占位的耗时检查。

### 7.4 后台执行要求

全量审计和数据重建都应按原要求使用独立后台进程、文件锁、日志、PID 和退出码。审计还需要按 shard 保存结果并支持从已完成清单续跑，避免异常后重做全部构建。

`nohup + setsid` 可以隔离终端断开；重启、断电或 OOM 仍需依靠落盘结果和续跑机制恢复。当前完整重编码器没有跨重启续建能力，不能把方案写成“绝不会中断”。核查阶段尚未实现审计工具；后续实施入口、验证结果和运行状态见第 9 节。

## 8. 本轮证据与限制

- 已全量扫描上述原始候选、build_valid 候选与测量记录的 JSON 字段，汇总数量、target、错误码与耗时类型。
- 已核对历史 dump 与当前 dump 的差异、TVM 的结果序列化、构建/计时路径和 CLM 数据入口。
- 已找到原始 state 与历史编译失败对应的直接证据；这并非全量当前编译实验。
- 已实际复编译 2 条历史失败原始记录和 2 条成功对照；失败复现于 LowerThreadAllreduce，实际环境与结论边界见第 4.2 节。
- 已修正 [全量重建方案](04_full_clm_dataset_rebuild_plan.md) 的使用边界：原样重编码只能作为序列修复对照，正式可构建训练集需要先进行源记录验证。
- 核查阶段没有覆盖或清理 `/home/qsy/workspace/dataset`，也没有启动全量后台任务；之后的实施与后台任务状态见第 9 节。

## 9. 按本计划实施：源记录门禁与后台审计

实施日期：2026-10-21。本节是第 8 节核查之后的进展，核查阶段的统计不是本工具的新全量构建结果。

### 9.1 已实现的入口

- [audit_schedule_dataset.py](../../audit/audit_schedule_dataset.py)：只读原始调度；冻结来源 SHA-256、registry、原有 train/validation workload 划分、hold-out、完整 target 与实际工具链；逐 workload/shard 审计并导出通过记录。
- [audit_tvm_backend.py](../../audit/audit_tvm_backend.py)：恢复 state、应用步骤，再执行默认 `LocalBuilder` 的构建和导出操作。恢复/应用在有超时限制的独立 worker 中运行。训练规范化后的 state 也必须恢复、应用并通过构建；有效调度相同时复用同一次构建。
- [audit_persistent_builder.py](../../audit/audit_persistent_builder.py)：复用可超时回收的编译进程，调用 TVM 同一个 `_local_build_worker`，保留实际编译和模块导出。`streaming` 引擎交错执行不同记录的恢复、构建和重试，独立的重试池保留加倍超时预算。
- [audit_reuse.py](../../audit/audit_reuse.py)：调整执行并发时，从已停止的兼容任务迁移持久化凭证；核对源数据、划分、工具链和格式化协议，逐条保留原审计来源。
- [schedule_record_utils.py](../../core/schedule_record_utils.py)：完整输入指纹、严格延迟检查、落盘后原子替换 manifest、批次 journal 和不完整尾行恢复。
- [prepare_audited_clm.py](../../data/prepare_audited_clm.py)：核验完成状态、manifest、每个 shard 的文件摘要和逐输入通过凭证；保持冻结的 workload 划分，生成新的 raw，不重新随机切分。
- [rebuild_clm_dataset.py](../../data/rebuild_clm_dataset.py)：新增 `--build-audit-manifest`；正式重编码时核对 raw 摘要，向 Arrow 输出目录保留构建审计来源报告。

[make_dataset.py](../../data/make_dataset.py) 的 `for_gen` 会过滤显式失败记录。其旧 CLI 默认拒绝直接处理未经审计的候选；历史对照必须显式传 `--allow_unverified_records True`。正式流程使用上面的审计、raw 导出和重编码三个入口。`for_gen_best` 和 [build_preference_pairs.py](../../train/build_preference_pairs.py) 统一使用严格延迟检查：错误码必须为 0，全部计时值必须有限、为正且低于失败哨兵阈值；混有零值或失败占位值的计时整条拒绝。

源记录原有零延迟保持不变，不伪造测量成功或性能结果。构建凭证独立保存，绑定全部任务输入字段、state、记录版本和实际编译器指纹。

### 9.2 输出和恢复规则

```text
LLM/audit_runs/<run>/
  audit.log
  launch.json                 # 提交命令和 PID
  worker.json                 # 实际 worker PID
  run.lock                    # 同一目录只能有一个 worker
  status.json                 # INITIALIZING / RUNNING / SUCCEEDED / FAILED
  manifest.json               # 固定来源、划分、代码和工具链
  progress.json               # 当前 workload 的批次进度
  summary.json                # 已完成 workload 的累计结果
  shards/<id>/events.jsonl     # 按唯一完整输入记录凭证和重试历史
  shards/<id>/progress.json    # 各 workload 的独立批次进度
  shards/<id>/done.json        # workload 提交标记及文件摘要
  records/<原文件名>.json       # 仅本轮通过的原始记录
  build_tmp/                  # 独立临时构建区，批次结束后清理
```

每批 journal 都执行 flush/fsync；先写记录文件，再原子提交 `done.json`。续跑复用已提交批次和已完成 workload，不重复构建已验证的相同输入。仅允许修复 journal 最后一次未完成写入，完整行损坏会报错。来源、划分、代码、工具链或设置变化时拒绝复用旧目录。

超时、环境错误和普通构建拒绝分开计数。默认对前两者重试一次，重试超时预算加倍，保留前次错误。重试后仍不确定的记录不会进入通过集合，后续可用新的 run 单独复核。`SUCCEEDED` 表示审计任务完成，不表示所有调度通过，必须同时看 `summary.json` 的分类统计与覆盖缺口。

### 9.3 后台启动

`--background` 使用 `start_new_session=True` 调用 `setsid`，关闭标准输入、输出写入文件，worker 忽略 SIGHUP；等效于本方案的终端隔离要求。已观察到本次工具沙箱会回收命令会话的子进程，因此长任务应从用户的正常终端启动，或使用已获准的沙箱外执行环境。只在沙箱内看到 `Submitted PID` 不能证明后台任务持续运行。

第一轮正式范围仅为现有 4090 raw 对应的训练/验证来源，排除 hold-out；不会同时启动 v100 和独立 Qwen/Llama registry 的全量验证。

```bash
PYTHONDONTWRITEBYTECODE=1 /home/qsy/anaconda3/envs/tlm/bin/python -u \
  LLM/audit/audit_schedule_dataset.py \
  --source-dir /data3/qsy/dataset/to_measure_programs/4090 \
  --registry /data3/qsy/dataset/network_info/4090/all_tasks.pkl \
  --split-from-raw /home/qsy/workspace/gen_data/4090_gen_qwen_v2_ext \
  --expected-arch sm_89 \
  --run-dir /data3/qsy/complier/llm_compiler/LLM/audit_runs/4090_plan5_full_fast_20261022_01 \
  --n-parallel 8 --workload-parallel 6 --batch-size 512 \
  --build-engine streaming --build-timeout 30 --max-retries 1 \
  --reuse-audit-dir LLM/audit_runs/4090_plan5_full_20261021_01 \
  --background
```

异常结束后，在同样环境用完全相同命令续跑。不要在已有 worker 运行时再次提交，也不要修改冻结的源数据和审计/格式化代码。重启或 SIGKILL 后状态文件可能停留在 `RUNNING`，需用 PID 与命令行、文件锁共同确认进程是否存活。

小规模试跑另用 run 目录，加 `--max-files 4 --max-records-per-file 8`，同时覆盖训练和验证来源。此类运行明确标记为 `partial_scope`，正式导出默认拒绝。诊断导出须显式 `--allow-partial`，诊断重编码须显式 `--allow-partial-audit`。

### 9.4 审计完成后的 raw 与 Arrow 重建

以下命令以正式全量审计成功结束为前提；不会自动恢复或启动训练。

```bash
PYTHONDONTWRITEBYTECODE=1 /home/qsy/anaconda3/envs/tlm/bin/python \
  LLM/data/prepare_audited_clm.py \
  --audit-dir LLM/audit_runs/4090_plan5_full_fast_20261022_01 \
  --output-dir LLM/audit_runs/4090_plan5_raw_20261022_01

PYTHONDONTWRITEBYTECODE=1 /home/qsy/anaconda3/envs/tlm/bin/python \
  LLM/data/rebuild_clm_dataset.py \
  --train-file LLM/audit_runs/4090_plan5_raw_20261022_01/train_raw.json \
  --validation-file LLM/audit_runs/4090_plan5_raw_20261022_01/validation_raw.json \
  --build-audit-manifest LLM/audit_runs/4090_plan5_raw_20261022_01/build_audit_manifest.json \
  --tokenizer-path /home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1-v2-ext-b2 \
  --output-dir LLM/audit_runs/4090_plan5_arrow_20261022_01 \
  --max-length 2048 --overlength-policy error
```

全量 raw 导出和 Arrow 重建同样应使用 [第 4 份计划](04_full_clm_dataset_rebuild_plan.md) 的独立后台 worker、锁和日志方式提交；上方两条展示实际参数，不应在需要断开 SSH 的前台终端直接执行。Arrow 重编码器尚不支持跨重启续建；异常时从已完成的审计/raw 另建新 Arrow 输出目录。

### 9.5 当前验证与尚需完成的工作

- 新审计器实际验证了 2 条成功和 2 条已知失败记录：成功对照通过构建和导出，失败样本再次被拒绝，未写入通过集合。
- 已完成真实后台试点 `LLM/audit_runs/4090_plan5_pilot_20261021_01`：覆盖训练和验证来源各 2 个 workload，每个取 8 条，共 32 条；25 条通过、7 条构建失败，任务状态为 `SUCCEEDED`。这是有选择的流程试点，不能用其通过率推算全量数据。
- 试点已完成审计记录到 raw、再到 Arrow 的实际转换。raw 目录为 `LLM/audit_runs/4090_plan5_pilot_raw_20261021_01`，Arrow 目录为 `LLM/audit_runs/4090_plan5_pilot_arrow_20261021_01`；训练 9 条、验证 16 条，没有超长或截断，也没有 workload 丢失全部样本。重新加载 Arrow 后逐条核对完整原文编码、labels、PPT 边界及审计报告一致性，检查结果保存在 Arrow 目录的 `verification.json`。两份试点数据均明确标记为局部范围，不作为正式全量数据。
- 三份相关测试文件 `test_llm_schedule_audit.py`、`test_llm_schedule_dataset.py`、`test_llm_lapo_training.py` 共 63 项通过：沙箱内通过 62 项，需本地 socket 的分布式评估测试在沙箱外单独通过。回归覆盖完整输入身份、失败计时过滤、批次续跑、重复状态复用、超时重试、原始文件/通过文件被修改后的拒绝、workload 划分冲突、未验证样本过滤及重编码前后的 raw 摘要检查。
- 实际加载的 TVM 仍为 `/data3/qsy/complier/tlm` 的 Python 包和共享库，工具会准确记录该环境；这不表示本 checkout 已完成重编译。部署生成/测量时必须使用兼容环境，换库后应重新审计。
- 此门禁验证原始记录和训练规范化记录的恢复/应用/构建。完整的 token 解码到 `SketchPolicy.gen_states` 生成重建链路仍须独立做对照，导出报告明确记录 `generation_parser_roundtrip_checked=false`。
- 运行正确性、真实性能测量、缺样 workload 补采、优秀后缀 SFT 与 DPO/LAPO 仍是后续步骤，不能把本轮构建通过率解释为模型性能提升。

### 9.6 原正式全量后台任务与迁移记录

2026-10-21 23:41:57（北京时间）已按第 9.3 节参数在沙箱外提交正式后台任务：

- 运行目录：`LLM/audit_runs/4090_plan5_full_20261021_01`。
- worker PID：`2509199`，记录于 `launch.json` 和 `worker.json`。
- 范围：原 raw 对应的 2,169 个来源 workload，保留 2,061 个训练、108 个验证 workload 的划分；`partial_scope=false`，目标架构 `sm_89`，8 路编译、每批 64 条。
- manifest ID：`e91b8262f6c9b448a16eea3c3d4cbf0ae23b09465943ad32d54aca662e54da49`。
- 工具链指纹：`914230252efaea3942e0f65592c926d3d9d937b74a489f662e8060afe15677eb`，与已完成试点一致。

截至 23:46:35 的进度快照，状态为 `RUNNING`；第一个 workload 已提交 576/1,012 个唯一输入的审计结果，其中通过 180、构建失败 395、超时 1。随后只读检查确认运行锁仍被持有且进度持续更新。这里是**当前 workload** 的进度，不是全量累计通过率；此时尚无已完成 workload 的 `summary.json`。完整结果应以最终清单为准。

从仓库根目录查看：

```bash
cat LLM/audit_runs/4090_plan5_full_20261021_01/status.json
cat LLM/audit_runs/4090_plan5_full_20261021_01/progress.json
tail -f LLM/audit_runs/4090_plan5_full_20261021_01/audit.log
```

完成 workload 后会更新 `summary.json`；全量结束且 `status.json` 为 `SUCCEEDED` 后，才能执行第 9.4 节正式 raw 与 Arrow 重建。本轮已启动全量源记录审计，尚未启动全量 Arrow 重建或训练。

**2026-10-22 00:11:16（北京时间）该旧 worker 已收到 SIGTERM 并停止，用于迁移到加速版本。** 其 `FAILED` 状态中的 `Received signal 15` 来自这次主动停止，不表示数据损坏。旧目录、原始代码快照和已提交的 journal 均保留。停止前已完成 3 个 workload 的 3,104 条，其中 807 条通过；另外有当前 workload 的批次结果可复用。当前实际任务见下一节。

### 9.7 加速实施与后台入口（历史记录）

机器有 80 个逻辑 CPU，原任务只开启 8 路，且每批需等待最慢的编译及其重试。加速版保留逐条恢复、应用、编译、导出、失败隔离和超时重试，调整执行方式：

- 同时处理 6 个 workload，每个最多 8 路主编译，共 48 路；恢复池和重试池独立受限，不是只有 48 个进程。
- 可复用的编译进程池减少重复启动；恢复、构建和重试交错执行，减少整批等待。
- 每个 workload 的批次扩大为 512 条，降低批次间空闲；各 worker 的临时构建目录隔离。异常中断时仍按已提交批次续跑，未提交批次需要重做。
- `--reuse-audit-dir` 只在原 worker 已停止且环境、输入和验证协议兼容时迁移；新的凭证含 `reused_from`，不修改旧数据和旧审计目录。新的 `reuse_report.json` 汇总迁移数。

完整流程对照使用相同的 192 条记录，原方式耗时 23.61 秒，加速方式耗时 11.18 秒，约 **2.11 倍**，逐条筛选结果一致。这一对照为 3 个 workload 并发的短样本实验，不能直接当作 228 万条全量吞吐或完成时间承诺。另有 256 条纯构建操作对照，原构建器和可复用构建器的逐条错误码一致。

新增并发、重试池预算、构建产物检查、旧凭证迁移与不兼容来源拒绝的回归覆盖。审计和数据完整性两份测试文件合计 **56 项通过**。

当前后台任务：

```text
run_dir = LLM/audit_runs/4090_plan5_full_fast_20261022_01
worker PID = 2600068
submitted = 2026-10-22 00:11:16 +08:00
```

00:12:34 已进入 `RUNNING`，成功导入旧任务的 **3,488 条**持久化凭证；3 个已完成 workload 的 807 条通过记录已在新目录重新提交。新 manifest ID 为 `767a648545b14ac71fb96309241654497f6e17d6f3744b3040a89de98d9b56cc`，工具链指纹与旧任务一致。代码快照保存在新目录的 `frozen_code/`。

当前应查看新目录的 `status.json`、`summary.json` 和 `audit.log`。多 workload 并发时，全局 `progress.json` 是最近一次提交批次的 workload 快照；各 workload 的进度保存在 `shards/<id>/progress.json`，不能只用全局进度计算全部记录数。`summary.json` 只累计完成的 workload，含 `reused_unique_inputs`；重新估算吞吐时应排除复用记录及初始化扫描耗时。

审计是在生成经过构建检查的训练来源：每个完成 workload 的通过记录落入 `records/`。它不生成新的随机调度，也不会在后台自动启动全量 raw/Arrow 转换或训练。全量审计成功后，仍需执行第 9.4 节；运行正确性和速度收益仍是独立验收项。

### 9.8 全量审计完成与下一阶段顺序（2026-11-02 更新）

全量审计已于 **2026-10-24 23:42:29（北京时间）**成功结束，`status.json` 为 `SUCCEEDED`、退出码为 0，运行锁已释放。加速任务耗时约 71.52 小时，2,169 个 workload 全部处理完成。

| 项目 | 数量 |
| --- | ---: |
| 总记录 | 2,281,247 |
| 构建通过 | 591,126（25.91%） |
| 构建失败 | 1,670,170 |
| 重试后仍超时 | 19,951 |
| 训练来源的通过记录 | 556,948 |
| 验证来源的通过记录 | 34,178 |

2026-11-02 已逐文件核对全部 2,169 个通过记录文件的 SHA-256、行数与完成清单，合计一致；每个 workload 均至少保留一条通过记录。随后已完成正式全量 raw/Arrow 后台重建，格式化、编码和独立验收后仍保留全部 591,126 条记录及 2,169 个 workload，执行记录见第 9.9 节。

后续按以下顺序推进：

1. **导出正式 raw。** 执行第 9.4 节 `prepare_audited_clm.py`，核验逐输入凭证后格式化通过记录，保留原有训练/验证 workload 划分、hold-out 排除及审计来源。最终数量以 `build_audit_manifest.json` 为准：短后缀过滤和去重可能使样本数少于 591,126，不能要求强行补回。检查过滤原因和 `workloads_without_samples`，按 workload 汇总保留数量。
2. **完整重编码为 Arrow 并验收。** 执行第 9.4 节 `rebuild_clm_dataset.py`，绑定 raw 审计报告、固定 tokenizer，使用 `--max-length 2048 --overlength-policy error`。超长时报错后检查分布，不静默截断。验收两份划分、样本数、完整 input/labels 长度、PPT 边界、无截断标记、来源报告及 workload 覆盖。raw 导出和编码均按第 4 份计划在独立后台会话中执行，记录 PID、日志、状态和退出码。
3. **验证生成重建链路。** 从两份 raw 按算子类型、workload、序列长度分层取样，把已知正确的后缀经过 token 编解码，再交给真实生成解析器和 `SketchPolicy.gen_states` 重建。检查恢复的调度与对应规范化记录是否一致并重新构建，区分格式化、PPT 切分、解析和重建失败。原始及规范化 state 构建通过不能替代这一步；此项通过前不进入长时间全量训练。
4. **小规模结构训练对照，再决定全量训练。** 固定基础模型、tokenizer、生成设置和评估 workload，用相同训练预算比较原始来源数据与审计过滤数据。记录解析成功率、可构建率、候选去重率、生成数量达标率及按 workload 的覆盖，不能只以 token accuracy/loss 判断改进。先确认数据到生成的链路有改善，再投入完整训练。
5. **补性能监督。** 从训练 workload 中按类型和候选多样性选小批次，在独占目标 GPU 上做参考输出正确性检查和实测延迟，先验证测量流程与收益。挑选优秀后缀进行 SFT，再构建同 prompt 的 DPO/LAPO 偏好对。hold-out workload 只用于评估；其测量结果不进入正式训练。构建失败和超时记录独立保留供诊断及后续补采，当前可以先推进通过集合，无需先重跑全部失败样本。

第 1、2 步已按用户授权在后台完成并通过数据验收，产物与验收结果见第 9.9 节；第 3 步生成重建链路验证已提交，执行记录见第 9.10 节。当前 591,126 条是通过构建检查的结构学习来源，不包含由本次审计新增的运行正确性或延迟证明。

### 9.9 正式 raw/Arrow 后台重建（2026-11-02）

新增 [rebuild_audited_clm.py](../../data/rebuild_audited_clm.py) 串联正式 raw 导出、完整 Arrow 编码和独立验收；新增 [verify_audited_clm.py](../../data/verify_audited_clm.py) 校验落盘结果。直接复用原有导出和编码器，没有修改本次审计冻结的格式化代码。

**第 1、2 步已于 2026-11-02 15:36:39（北京时间）完成。** 最终 run 为 `LLM/audit_runs/4090_plan5_rebuild_20261102_02/`，`status.json` 为 `SUCCEEDED`、退出码为 0；正式 `dataset/` 已发布，临时 `dataset.pending/` 已不存在，运行锁已释放。

| 划分 | raw 条数 | Arrow 条数 | workload 数 | 完整长度范围 | 超过 1024 token 的条数 | 完整原文编码抽查数 |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| train | 556,948 | 556,948 | 2,061 | 202–1,189 | 15,604 | 4,806 |
| validation | 34,178 | 34,178 | 108 | 206–1,177 | 747 | 1,207 |
| 合计 | 591,126 | 591,126 | 2,169 | — | 16,351 | 6,013 |

没有超长、截断或丢样；raw 的短后缀、重复、缺少 PPT、错误和无凭证过滤损失均为 0，`workloads_without_samples=[]`。训练与验证 workload 无交集，hold-out 已排除。训练完整 token 数为 330,588,281，验证为 18,387,046；后缀最短分别为 45 和 46 token。最终 raw 约 1.051 GiB，Arrow 与验收报告约 4.256 GiB。

正式产物：

- [训练 raw](../../audit_runs/4090_plan5_rebuild_20261102_02/raw/train_raw.json)、[验证 raw](../../audit_runs/4090_plan5_rebuild_20261102_02/raw/validation_raw.json) 和 [raw 来源报告](../../audit_runs/4090_plan5_rebuild_20261102_02/raw/build_audit_manifest.json)。
- [Arrow DatasetDict](../../audit_runs/4090_plan5_rebuild_20261102_02/dataset)、[长度与覆盖统计](../../audit_runs/4090_plan5_rebuild_20261102_02/dataset/tokenization_stats.json)、[独立验收报告](../../audit_runs/4090_plan5_rebuild_20261102_02/dataset/verification.json)。
- [最终状态](../../audit_runs/4090_plan5_rebuild_20261102_02/status.json)、[验收日志](../../audit_runs/4090_plan5_rebuild_20261102_02/rebuild.log) 和 [复用来源记录](../../audit_runs/4090_plan5_rebuild_20261102_02/reuse.json)。

原任务于 **15:14:55** 启动，后台 worker PID 为 `4128770`。已在启动命令结束后从沙箱外确认进程持续运行，worker 父进程为 PID 1、会话 ID 等于自身 PID，日志写入文件。原始命令如下，历史输出保留在 `_01`，不再重复提交：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/qsy/anaconda3/envs/tlm/bin/python -u \
  LLM/data/rebuild_audited_clm.py \
  --audit-dir /data3/qsy/complier/llm_compiler/LLM/audit_runs/4090_plan5_full_fast_20261022_01 \
  --run-dir /data3/qsy/complier/llm_compiler/LLM/audit_runs/4090_plan5_rebuild_20261102_01 \
  --tokenizer-path /home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1-v2-ext-b2 \
  --max-length 2048 --background
```

原 run 完成 raw 导出（645.45 秒）与 Arrow 编码（319.01 秒），但首次独立验收在 workload 覆盖比较时报错。原因是新增校验器只对 raw workload key 规范化，未对长度统计中的同一 JSON key 去除序列化空格，造成比较失败；并非发现了缺失 workload。已修复为两侧统一规范化，同时保持数量、逐 workload 条数和规范化身份唯一性检查。

为保留失败现场，新增 `--reuse-run-dir`：仅接受已结束、raw/Arrow 两阶段均完成而验收失败的 run；加锁后核对审计、工具链、tokenizer、长度设置、冻结代码与文件摘要。只允许编排器或验收器修复，格式化或编码代码改变时拒绝复用。复制原产物到新 run，不复制编码缓存、不修改原目录，并重新执行全部验收后才发布。

恢复任务 worker PID 为 `4162663`，于 **15:35:59** 提交并开始运行；复制约 4.42 秒、独立验收约 33.79 秒，约 39.37 秒后成功结束。命令如下：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/qsy/anaconda3/envs/tlm/bin/python -u \
  LLM/data/rebuild_audited_clm.py \
  --audit-dir /data3/qsy/complier/llm_compiler/LLM/audit_runs/4090_plan5_full_fast_20261022_01 \
  --run-dir /data3/qsy/complier/llm_compiler/LLM/audit_runs/4090_plan5_rebuild_20261102_02 \
  --tokenizer-path /home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1-v2-ext-b2 \
  --max-length 2048 \
  --reuse-run-dir /data3/qsy/complier/llm_compiler/LLM/audit_runs/4090_plan5_rebuild_20261102_01 \
  --background
```

长任务通过获准的沙箱外环境启动；在本工具沙箱内无法用 `ps` 观察到该外部 PID，不能据此认定进程已退出。查看最终结果：

```bash
cat LLM/audit_runs/4090_plan5_rebuild_20261102_02/status.json
cat LLM/audit_runs/4090_plan5_rebuild_20261102_02/dataset/verification.json
tail -n 40 LLM/audit_runs/4090_plan5_rebuild_20261102_02/rebuild.log
```

两份 run 均保留独立记录，目录约定如下：

- `manifest.json`、`frozen_code/`：冻结重建代码、tokenizer 文件摘要、审计身份与工具链身份。正式运行前已确认实际编译环境与全量审计完全一致。
- `launch.json`、`worker.json`、`run.lock`：启动信息与互斥运行锁；实际启动导出或编码子进程的 run 还保存 `child.json`。
- `status.json`、`rebuild.log`：阶段、状态、耗时、退出码和日志。状态按阶段更新；导出器长时间处理单个阶段时，状态时间戳不等于进程心跳。
- `raw/train_raw.json`、`raw/validation_raw.json`、`raw/build_audit_manifest.json`：正式 raw 和来源凭证。生成中的 `.pending` 文件尚未完成，不能用作训练输入。
- `dataset.pending/`：编码和验收期间的临时输出；只有验收成功后才原子改名为 `dataset/`。
- `dataset/verification.json`：完整验收报告。仅 `status.json` 为 `SUCCEEDED`、退出码为 0 且正式 `dataset/` 存在时，才表示两步已完成。

验收覆盖两份划分的记录数、来源摘要和 workload 隔离、逐条完整序列长度与 PPT 边界、所有保存 token 的词表范围和 labels、无截断及覆盖缺口。另对每个 workload 的首条和最长样本，加每份划分固定随机种子的 1,024 条样本，重新对完整原文编码并比对 Arrow。长度上限为 2048，超长直接报错，不能静默截断或丢弃。

修复后相关回归测试 **64 项通过**（`test_llm_audited_rebuild.py`、`test_llm_schedule_audit.py`、`test_llm_schedule_dataset.py`），覆盖非规范化 JSON key、完整长样本、篡改 raw/labels 拒绝、验收成功前禁止发布，以及完成阶段复用的来源检查与失败现场保留。新文件按 100 列 Black 格式化。发布后另起 Python 进程重新加载正式 DatasetDict，确认两份划分数量和最长样本的完整 input/labels、长度及无截断标记，运行锁已释放。

本轮只执行第 1、2 步。产物适用于结构学习的数据准备；全部记录尚无有效实测延迟，不能直接视为性能监督或 DPO/LAPO 偏好数据。下一步按第 9.8 节第 3 项验证真实生成解析器的 token 编解码与 state 重建链路；本次不包含该验证、运行正确性检查、GPU 测量或模型训练。

`--background` 隔离终端断开并忽略 SIGHUP；机器重启、断电、OOM 或 SIGKILL 仍可能终止任务。当前不支持任意不完整阶段的自动续建；raw/Arrow 已完成且仅验收失败时可通过上述受检查的复制入口恢复，其他失败须另建新 run。所有 run 与产物均置于已忽略的 `LLM/audit_runs/`，不覆盖原始数据或已发布输出。

### 9.10 第三步：真实生成解析器重建验证（2026-11-02）

新增 [validate_generation_roundtrip.py](../../validation/validate_generation_roundtrip.py) 后台检查入口，以及 [generation_roundtrip_worker.py](../../validation/generation_roundtrip_worker.py) 的真实 TVM 检查逻辑。输入为第 9.9 节正式发布的 raw/Arrow，不修改这些数据和原全量审计结果。

抽样方法：两份划分均覆盖全部 workload，每个 workload 根据完整 token 长度选最短、中位、最长三条，重复行号合并；本次恰好得到训练 **6,183 条 / 2,061 个 workload**、验证 **324 条 / 108 个 workload**，合计 **6,507 条 / 2,169 个 workload**。报告另按计算图名称识别的算子类别和 `<=512`、`513–1024`、`>1024` 三个长度区间分层统计。算子类别是基于名称的诊断分组，完整 workload 身份才是隔离和覆盖的依据。

每条样本执行以下检查：

1. 重新核对所选来源文件、逐输入构建凭证、审计身份与训练/验证归属。从审计通过记录恢复对应 state，执行原有 SP 规范化，确认实际格式化结果等于保存的 raw 文本。
2. 用固定 tokenizer 完整编码并与已发布 Arrow 的 `input_ids`、长度、PPT 边界比较；完整文本解码保留特殊 token。按真实生成入口的 `decode_decision_tokens` 解码 PPT 后的 continuation，核对完整 suffix 字段。
3. 截取记录中 PPT 之前的 sketch，调用真正的 `SketchPolicy.gen_states`。每次阶段回调使用生产入口的 `input_to_tokens` 和 `encode_generation_prompts`，检查动态 prompt 是原完整训练文本的精确前缀，再回放原文中对应的已知正确后缀。后缀通过同一个生产解码函数转成 C++ 解析器所需字段；不使用模型或 fallback。
4. 要求返回恰好一个 state，比较 SP 规范化后的完整变换步骤、有效 state 表示和重新格式化的完整文本。分别应用生成与参考 state，比较降低后的 TIR 结构等价性。
5. 调用 `LocalBuilder` 使用的同一个 `_local_build_worker`，实际编译并导出模块，要求错误码为 0 且导出文件非空。临时模块检查后清理；不会启动 GPU 运行或测量延迟。

检查器先验证正式重建状态、raw/Arrow 摘要、tokenizer、registry 和完整实际工具链与原审计一致，冻结检查代码、生成入口和样本身份。并发为 8；单样本恢复、解析、等价性检查与构建合计使用 60 秒的可回收 worker 预算。worker 超时、环境失败或构建阶段失败最多重试一次，预算加倍并保留前次失败。

已完成两轮真实试点：

- `4090_plan5_roundtrip_pilot_20261102_01`：24 条均在新增检查器的整段解码断言处失败。已确认原因是 `PPT` 本来就是 tokenizer 的特殊 token，检查器使用 `skip_special_tokens=True` 对完整 prompt 解码时将它删除；该轮尚未走到后缀解析，不表示原训练数据失效。已修复完整文本检查为保留特殊 token；生产 continuation 解码仍使用原函数和设置。
- `4090_plan5_roundtrip_pilot_20261102_02`：**2026-11-02 16:09:05（北京时间）**完成，`SUCCEEDED`、退出码 0。训练/验证各 4 个 workload，共 24 条样本全部通过，执行 72 次真实阶段回调，包含超过 1024 token 的卷积样本。耗时约 91.29 秒。该试点明确标记 `partial_scope=true`，不能替代下面的正式覆盖检查。

相关回归测试 **73 项通过**：`test_llm_generation_roundtrip.py`、`test_llm_audited_rebuild.py`、`test_llm_schedule_audit.py`、`test_llm_schedule_dataset.py`。覆盖分层选样、单样本角色合并、错误阶段 prompt、训练 token 篡改、生产解码字段损坏、完整 prompt 的特殊 PPT token 保留，以及报告范围边界。首次真实探针还确认原始 SP 字段可能不同，但规范化步骤、有效 state 和降低后的 TIR 均可一致，因此不能仅比较未规范化的原始步骤字段。

正式后台任务于 **2026-11-02 16:14:55（北京时间）**提交，worker PID 为 `23052`，目录为 `LLM/audit_runs/4090_plan5_roundtrip_full_20261102_01/`。命令如下，已提交，不应在 worker 存活时重复执行：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/qsy/anaconda3/envs/tlm/bin/python -u \
  LLM/validation/validate_generation_roundtrip.py \
  --rebuild-dir /data3/qsy/complier/llm_compiler/LLM/audit_runs/4090_plan5_rebuild_20261102_02 \
  --run-dir /data3/qsy/complier/llm_compiler/LLM/audit_runs/4090_plan5_roundtrip_full_20261102_01 \
  --n-parallel 8 --sample-timeout 60 --background
```

和前两步相同，通过获准的沙箱外环境启动独立会话，标准输入关闭、日志写入文件，忽略 SIGHUP。终端退出和 SSH 断开不会终止任务。每批结果写入 `results.jsonl` 并 fsync，随后原子写入 `summary.json` 和 `status.json`。若机器重启或任务被强制终止，确认原 worker 不存在且锁已释放后，以同样环境和完全相同命令续跑；仅复用 manifest 身份、冻结代码、输入摘要和样本身份一致的已提交结果，不重复构建已完成样本。

```bash
cat LLM/audit_runs/4090_plan5_roundtrip_full_20261102_01/status.json
cat LLM/audit_runs/4090_plan5_roundtrip_full_20261102_01/summary.json
tail -n 40 LLM/audit_runs/4090_plan5_roundtrip_full_20261102_01/roundtrip.log
```

`SAMPLING` 阶段会核对逐来源凭证并冻结 `samples.jsonl`，随后进入 `ROUNDTRIP_AND_BUILD`。状态按批更新，不能把更新时间当成进程心跳。截至 **2026-11-02 16:22:39（北京时间）**，正式任务已完成全部来源核对，冻结 6,507 条样本，进入解析与构建阶段；首批 **64 条全部通过、0 条失败**，已覆盖 22 个 workload。该进度已在启动命令返回后持续推进，证明后台任务没有随工具会话结束而被回收。最终完成后应复核 `status.json=SUCCEEDED`、退出码为 0、6,507 条均通过、全部 2,169 个 workload 覆盖，以及每个算子/长度分组无失败。

本步骤是已知正确后缀的分层回放检查。即使全部通过，也仅证明所选样本能够走通实际生成解析和构建链路，不能声称 591,126 条都经过该重建检查，更不能解释为模型自由生成质量、GPU 运行正确性或性能收益已通过验收。这些边界显式记录在报告中；原 raw/Arrow 的历史验收报告保持原样，由第三步独立报告绑定其身份。此项通过后才能推进第 9.8 节第 4 项小规模训练对照。

### 9.11 第三步初步评估：进度、速度、验收机制与质量（2026-11-02）

以下为 **2026-11-02 16:26:21（北京时间）**已提交结果的快照，不是最终验收。后台 PID `23052` 已从沙箱外再次确认存活，父进程为 PID 1，独立会话仍在运行；本次评估不改变正在执行的冻结代码、原数据或样本清单。

| 项目 | 当前证据 | 初步判断 |
| --- | --- | --- |
| 进度 | 224 / 6,507 条，约 3.44%；覆盖 75 / 2,169 个 workload | 正在执行，远未完成全部覆盖 |
| 最终样本结果 | 224 条通过、0 条最终失败；672 次生产解析阶段回调 | 已完成样本的格式、解析、调度等价性与构建链路一致 |
| 重试 | 1 条初次 60 秒 worker 超时，加倍预算后通过 | 需要保留首轮失败，不能把最终通过描述成全部首轮通过 |
| 完整性与等价性 | 224 条的规范化步骤、有效 state、格式化文本和降低后 TIR 比较全部通过 | 所检样本未发现 token 边界或 state 重建偏差 |
| 长样本 | 已通过样本中 21 条超过 1,024 token | 长序列链路已有正面证据，尚不能替代全部长 workload 覆盖 |
| 已提交结果分布 | 全部来自 train：卷积 197、matmul/dense 9、softmax 18 | 执行顺序带来类别和划分偏差，当前通过率不能外推到全集；验证划分已有前述小试点证据 |

**速度初评。** `ROUNDTRIP_AND_BUILD` 于 16:20:47 开始，到上述快照历时约 334.74 秒，提交 224 条，平均约 **0.67 条/秒**。按该阶段平均速度线性外推，剩余约 **2.6 小时**；这不是完成时间承诺，算子差异、编译长尾和重试会改变速度。样本最终成功尝试耗时中位数约 1.38 秒，最大约 62.85 秒；重试前的 60 秒不包含在该最大成功尝试耗时内。每批需等待全部样本完成，遇到超时还要等待重试，因此仅用近期最快批次或单样本中位数推算会明显偏乐观。若后续需要加速，可在保留原凭证和失败记录的前提下设计持续补位队列与独立重试队列；本次未修改正在运行的任务。

**验收机制初评。** 当前机制能够区分来源与工具链身份、文本/编码/PPT 边界、实际生产解析、规范化 state/TIR 等价性和编译导出，原始数据只读，逐条结果带身份摘要，按批 fsync 并支持同身份续跑。它比只检查 JSON、loss 或错误码更接近实际数据使用链路。仍有明确边界：每 workload 的三条是确定性的长度代表样本，不是随机独立抽样；当前执行顺序并不均衡；TIR 与参考 state 等价只证明重建一致，参考本身还没有独立输出正确性验证。因此不基于“暂时 100% 通过”计算或宣称总体可靠率。

**训练来源分布初评。** 将冻结样本中的 workload 类别映射回正式数据集逐 workload 的记录数，得到训练来源的以下分布。类别由计算图名称作诊断归类，存在融合算子分类的局限，不能当作精确算子普查。

| 训练类别 | 记录数 | 占训练样本比例 |
| --- | ---: | ---: |
| 卷积 | 310,931 | 55.83% |
| 池化 | 138,402 | 24.85% |
| softmax | 46,983 | 8.44% |
| elementwise/其他 | 34,776 | 6.24% |
| batch matmul | 17,333 | 3.11% |
| matmul/dense | 8,523 | 1.53% |

训练各 workload 的保留记录数最少 7、最多 1,844，中位数 185，有 1 个 workload 少于 10 条；验证 workload 最少 28、最多 1,631，中位数 201.5。覆盖未丢失不等于每个 workload 的候选多样性充分。当前没有进一步统计同 prompt 的不同调度数量、后缀结构多样性或相对性能分布，不能凭记录数量推断这些质量。

**质量结论。** 数据完整性和已检查样本的结构可用性有正面证据，适合继续完成第三步；目前不能宣布模型生成质量已改善。若重点是 Qwen/Transformer，matmul/dense 与 batch matmul 合计仅约 4.64%，需要在后续受控训练对照中评估按算子/workload 重平衡，并同时报告宏平均和分算子指标，避免总体指标掩盖薄弱类别。全部正式来源仍没有有效实测延迟，因此尚不能证明优化组合优于基线，也不能直接构成 DPO/LAPO 偏好监督。

下一阶段的质量判断应至少包括：固定评估 workload 与生成预算、关闭 fallback、分别统计格式/解析成功率、可构建率、唯一有效候选数量和按 workload 的达标率；GPU 输出正确性与实测延迟提升另行验收。先完成第三步，再按第 9.8 节进入小规模训练对照，本次初评不自动启动后续训练或测量。

### 9.12 进一步优化分析（2026-11-02）

新增 [重建数据后的进一步优化分析](06_next_optimization_analysis.md)，结合当前源码、正式 raw 的全量文本/前缀多样性统计和 Arrow 的全量终止 token 检查，给出后续优先级、修改边界和验收指标。

较第 9.11 节的新快照：**2026-11-02 17:17:55（北京时间）**已提交 **1,520 / 6,507** 条，**1,518 通过、2 条重试后仍 worker 超时**，涉及 507 个 workload；后台任务继续运行。这两条缺少可确定卡点的阶段信息，属于未获结论，不能推断为不可构建数据，也不能忽略后宣布第三步通过。

本次主要发现：结构评估没有使用 policy 的逐阶段预算，且默认 checkpoint 仍按 token accuracy 选择；训练集无完全重复文本，但每 workload 的 PPT 前缀数中位数为 1；591,126 条正式序列均未以 EOS 结尾；数据仍缺少性能监督。文档分别给出已确认事实和待实验验证的影响，并提出持续补位验证、阶段诊断、共享生成流程、目标分布重平衡、终止监督对照和代表性实测方案。本次仅更新文档，没有修改活动验证任务的冻结代码或启动训练。
