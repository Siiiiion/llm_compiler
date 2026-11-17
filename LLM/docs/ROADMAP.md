# 当前路线图

更新：2026-11-17。当前代码结构与函数归属见 [架构说明](ARCHITECTURE.md)。本文件是当前执行顺序的唯一入口；历史方案的“下一步”保留其当时语境。

## 已有实现和证据

| 工作 | 当前实现 | 已有验收与边界 |
| --- | --- | --- |
| 完整 CLM 样本与 token 边界 | data/make_dataset_utils、core/tokenization_protocol | [01](history/01_complete_clm_sequences.md)、[02](history/02_schedule_token_boundary.md)；旧截断 Arrow 需要重建 |
| LAPO 权重与评估汇总 | train/train_qwen3_dpo | [03](history/03_lapo_weights_and_metrics.md)；代码修复与性能改善分开判断 |
| 源调度审计与 raw/Arrow 重建 | audit/、data/ 的 prepare/rebuild/verify | [05](history/05_dump_dataset_validity_analysis.md)、[07](history/07_execution_record_20261109.md)；原失败记录及补验来源分别保留 |
| 共享生成与结构评估 | generation/generation_protocol、validation/eval_struct | [07](history/07_execution_record_20261109.md)；训练 token accuracy 不能代替有效候选指标 |
| state 约束、资源预览 | generation/constrained_generation、resource_feasibility | [09](history/09_state_constrained_generation_20261111.md)、[11](history/11_resource_feasibility_20261117.md) |
| 固定预算多候选与多 sketch | validation/validate_multicandidate_generation、validate_multisketch_generation | [10](history/10_multicandidate_validation_20261111.md)、[12](history/12_multisketch_validation_20261117.md) |
| 目录与函数整合 | core/data/generation/audit/validation/train/benchmark/experiments | [整理记录](REFACTORING.md)；统一命令入口，保留实验证据 |

最近的多 sketch 数字来自第 12 份历史验收记录：12 个 workload、19 个实际结构、304 次尝试；state_v1/state_v2 的不同正确候选为 112/144，case K=4 达标为 16/19 与 19/19。只有 3 个 workload 具有多个结构，新增 7 个结构两模式表现相同。

这些是历史记录中的结果，本次架构整理没有重跑 GPU 实验。资源预览有额外计算成本，现有候选通过率不能说明等算力收益或 kernel/整网加速。

## 下一步执行顺序

| 顺序 | 工作 | 完成条件 |
| --- | --- | --- |
| 1 | 对训练侧候选补独立语义参考和真实延迟 | 来源可追溯、hold-out 不混入、结果正确、同硬件/target/计时设置 |
| 2 | 构建可训练偏好对 | 相同 PPT prompt，chosen/rejected 都有有效实测延迟，报告有效对数量与 workload 覆盖 |
| 3 | 决定小规模 CLM / DPO / LAPO 对照 | 固定数据和生成预算，报告有效候选、K 达标、延迟与额外预览开销 |
| 4 | 扩大 workload 与整网验证 | 完整 history、正确输出、与 opt_level=3 baseline 的可比计时 |
| 5 | 再决定 8B 或跨硬件方案 | 前述链路已有可复现收益后选择成本与规模 |

## 模型规模选择

两份 2026 年 8–10 月的 8B 草案已合并到本节。先完成当前 0.6B 的正确候选、实测延迟与偏好训练对照，再决定扩大生成模型。

- 当前训练启动器使用 DDP；它在各卡复制模型与训练状态，不能通过增加卡数解决单卡显存不足。
- 若选择 8B 全参数训练，需要先实现并验证参数/优化器分片、必要的 CPU offload、梯度检查点及保存恢复链路。QLoRA 属于另一条实验路线，需单独实现和验证，不能视为现有能力。
- 扩词表模型、tokenizer 与完整序列数据必须匹配，不能使用旧版截断 Arrow 作为新训练基线。LAPO 还必须核验 reference 模型开销及有效实测偏好数量。
- 正式配置由实时硬件与模型检查、真实长度分布的短跑决定；验收覆盖保存、恢复、导出、生成、构建、正确性和实测。旧草案中的机器快照与时长估算不再作为配置依据。

扩大调度生成模型本身不要求更换 benchmark workload 或重新导出全部 ONNX。

## 后续架构演进

后续按真实使用需求逐项推进：

- 把 make_dataset 内的表示转换、模式分派和命令参数进一步分离，维持当前训练文本与 parser 的严格 roundtrip。
- 给训练启动器统一配置表达方式，逐步减少机器默认路径；每个正式 run 仍记录实际 model/tokenizer/target。
- 后续 CLI 新增在 LLM/__main__.py 注册，公共函数放对应模块，专项实验留 experiments。
- 新命令统一注册到包入口，子进程使用 module_command；避免重新增加根目录转发脚本。

## 文档维护

当前任务只改本路线图；完成一次实验后新增 history 记录，给出运行目录、完整命令、成功/失败状态与未证明的结论，再回链本表。research 中的推导和实验设想不自动成为“已完成”。
