# 代码架构

本说明依据 2026-11-17 的实际源码。仓库的 Python/TVM 与 C++ AutoScheduler 提供底层调度器，LLM 子项目负责数据、模型和实验编排。gen/ 保留原 TLM 工作流；Qwen 主线使用 LLM 包。

## 主链路与数据边界

~~~mermaid
flowchart TD
    records[TVM 原始调度 JSONL] --> audit[audit 原始与归一化 state 构建]
    audit --> data[data 导出 raw / 编码 Arrow / 验证]
    data --> clm[train CLM]
    clm --> generation[generation 按阶段生成 SPC / TBS / PRS]
    sketches[workload registry 与 sketch] --> generation
    generation --> validation[validation 解析 / 资源 / 构建 / 正确性]
    validation --> measurement[benchmark 真实延迟测量]
    measurement --> pairs[train 同 prompt 偏好对]
    pairs --> lapo[train DPO / LAPO]
    lapo --> generation
    measurement --> selection[generation task_scheduler 选择下一轮 workload]
    selection --> generation
    measurement --> relay[benchmark Relay 整网正确性与性能]
~~~

这条图表示代码具备的处理链；不同环节由各自命令显式启动，目前没有一个命令自动完成全部闭环。

| 产物 | 主要字段/约定 | 下游使用条件 |
| --- | --- | --- |
| TVM 调度记录 JSONL | i=task/state，r=cost/error，v=版本 | 输入身份由 core.schedule_record_utils 统一计算 |
| 审计目录 | manifest、events、done、build_valid、文件和编译器指纹 | passed 表示原始及训练归一化 state 的构建凭证 |
| raw CLM JSONL | text、workload_key、source_input_hash、审计来源 | workload 划分与 hold-out 按规范化 key 核验 |
| Arrow CLM | input_ids、labels、length、ppt_end 和完整性字段 | 已截断的数据必须重新编码 |
| 生成候选 JSONL | 可恢复的 TVM state，可能还没有实测延迟 | 编译通过不能代替语义正确性与延迟 |
| 偏好 JSONL | 同 PPT prompt 的 chosen/rejected、latency_gap | 使用有效实测延迟；占位零值或失败 cost 不合格 |
| Relay 报告 | history 覆盖、输出误差、baseline/optimized 时间 | 同 target/workload、正确输出、完整 history 才能比较 |

## 模块分层

| 层 | 主要实现 | 职责与边界 |
| --- | --- | --- |
| core | common、schedule_record_utils、tokenization_protocol、io、runtime、layout | 共享身份、文件、边界与运行设施；common 的任务注册需要 TVM，其余纯函数可独立使用 |
| data | make_dataset、make_dataset_utils、prepare/rebuild/verify_audited_clm、sampling | 文本与编码、split、完整性和审计后数据发布 |
| generation | gen_state、generation_protocol、constrained_generation、resource_feasibility、task_scheduler | 生产候选生成、共享解码协议、state_v1/v2、资源预览、预算任务选择 |
| audit | audit_schedule_dataset、audit_tvm_backend、audit_persistent_builder、audit_reuse | 编译器身份与原始/归一化 state 门禁；不承诺运行正确性 |
| validation | eval_struct、generation_roundtrip_worker、validate_*、candidate_metrics、multisketch_* | 训练回调、解析重放和固定预算验收；统计函数不依赖实验 CLI |
| train | train_qwen3_clm、train_qwen3_dpo、build_preference_pairs、run_train_* | 模型训练与偏好数据；启动器显式运行后才创建进程 |
| benchmark | measure_programs、benchmark_relay、benchmark_relay_batch、history 分析 | 实测、整网输出对比与结果组织 |
| experiments | diagnose_*、run_plan6_*、measure_plan6_*、evaluate | 针对既有实验的编排；evaluate 只是带固定历史值的示例图 |

attention/ 和 export_onnx/ 是独立专项。它们复用基准或生成入口，不替代主流程。

## 关键函数的唯一归属

| 逻辑 | 归属及接口 | 调用者 |
| --- | --- | --- |
| 文件指纹与原子写入 | core/io.py：file_sha256、atomic_json、atomic_write_text | 审计、重建、验收、批量 benchmark、history 报告 |
| 审计身份与测量有效性 | core/schedule_record_utils.py：workload_key、input_hash、normalize_record、measured_latency | 数据、偏好构建、审计与验收 |
| PPT token 边界 | core/tokenization_protocol.py：find_ppt_marker_end、encode_prompt_suffix、encode_generation_prompts | CLM、DPO 和生成协议 |
| worker 清理与时间戳 | core/runtime.py：bounded_pool、now | roundtrip、审计、重建和诊断 |
| 样本分层与代表选择 | data/sampling.py：operator_family、length_bucket、representative_indices | roundtrip 选样和实验取样 |
| 同一生成协议 | generation/generation_protocol.py：generate_decisions、resolve_max_new_tokens、decode_decision_tokens | gen_state、eval_struct、诊断与验证 |
| 资源约束 | generation/resource_feasibility.py 与 constrained_generation.py | state_v2 completion 预览、受限解码与最终门禁 |
| 多候选统计 | validation/candidate_metrics.py：summarize_workload、aggregate | 单 sketch 与多 sketch 共同使用 |
| 多 sketch 身份与聚合 | validation/multisketch_selection.py、multisketch_metrics.py | 固定预算多 sketch 验收 |

core/io.py 的 canonical_workload_key 保留历史报告对非 JSON 字符串的容错。正式审计的 workload_key 使用严格规范化规则；这两种语义没有强行合并。

## 几条关键调用关系

- make_dataset.for_gen 负责训练文本；make_dataset_utils 负责完整编码及 PPT 定位；train_qwen3_clm 负责 mask、collator 与训练。
- gen_state 与 eval_struct 共同调用 generation_protocol.generate_decisions；state_v1/v2 再转到 constrained_generation，而资源预览归 resource_feasibility。
- prepare_audited_clm 校验凭证并导出 raw；rebuild_clm_dataset 编码；rebuild_audited_clm 编排子进程、校验后发布；verify_audited_clm 复核内容与来源。
- validate_multicandidate_generation 编排采样与候选检查；candidate_metrics 只计算结果；multisketch_metrics 复用计算函数，消除了统计库反向导入实验脚本的依赖。
- build_preference_pairs 按相同 PPT 分组并用延迟选 chosen/rejected；train_qwen3_dpo 执行 SFT、DPO 或 LAPO。
- task_scheduler 消费历史实测结果；benchmark_relay 检查 history 覆盖和整网输出，然后比较时间。

## 启动、路径与审计指纹

内部使用 LLM 子包的绝对导入，避免 common、tokenizer 等通用名字碰到 gen/ 或第三方模块。包初始化不加载模型、注册任务或启动训练。

根目录只有 __init__.py 和 __main__.py。命令注册与分发合并到 __main__；逐文件兼容脚本和 _legacy 层已删除。CLI 转发保留参数及退出状态，业务代码与测试使用完整包名导入，避免同一文件加载成两份模块。

layout.source_path 按唯一文件名定位真实实现；module_command 构造 python -m LLM.<子包>.<模块> 子进程命令；project_pythonpath 为新 worker 提供仓库根路径。后台审计、数据重建、嵌套验证、attention 编排和批量 benchmark 均使用这套规则。子进程不再依赖根目录转发文件。

code_dependencies 将统一入口纳入指纹，并从 AST 收集包内实现依赖，把抽取出的 io、runtime、sampling、metrics 等也纳入新 manifest 的指纹。旧运行的 frozen_code 与 manifest 保持原样；当前源码变化会阻止把旧凭证当作新源码下的可复用凭证。

## 继续开发时的约束

新增公共逻辑先选择所属模块，不增加一个通用 utils.py。生产路径不依赖 experiments；纯统计模块不依赖 CLI。训练与实验产物放外部配置目录或 audit_runs，测试放 tests/python/unittest。

make_dataset 和训练器仍保留各自内部实现，它们有较多模式和参数；后续细拆时应按数据表示、编码与 CLI 继续分离，并以真实数据的 roundtrip 结果约束行为，而不是仅按文件大小拆分。
