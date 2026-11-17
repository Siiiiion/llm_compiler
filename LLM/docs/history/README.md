# 历史执行记录

这些文档按当时的源码、数据与环境记录问题、计划、尝试和验收。原始结果及失败状态保留；其中“尚未执行”“下一步”等表述属于记录当时的状态。当前顺序以 [路线图](../ROADMAP.md) 为准。

原 update_plan 中的 01–06 主要是问题分析与实施设计，07–12 主要是执行与验收记录。编号代表历史顺序。

| 记录 | 内容 |
| --- | --- |
| [01_complete_clm_sequences.md](01_complete_clm_sequences.md) | 第一个问题：修复 CLM 调度样本截断 |
| [02_schedule_token_boundary.md](02_schedule_token_boundary.md) | 第二个问题：统一调度序列的 token 边界 |
| [03_lapo_weights_and_metrics.md](03_lapo_weights_and_metrics.md) | 第三个问题：修复 LAPO 权重与偏好评估统计 |
| [04_full_clm_dataset_rebuild_plan.md](04_full_clm_dataset_rebuild_plan.md) | 全量 CLM 训练数据重建方案 |
| [05_dump_dataset_validity_analysis.md](05_dump_dataset_validity_analysis.md) | 原始 dump 数据集有效性与可构建性核查 |
| [06_next_optimization_analysis.md](06_next_optimization_analysis.md) | 重建数据后的进一步优化分析 |
| [07_execution_record_20261109.md](07_execution_record_20261109.md) | 计划 5、6 与总体优化方案执行记录 |
| [08_generation_failure_diagnosis_20261109.md](08_generation_failure_diagnosis_20261109.md) | 模型自由生成失败定位记录 |
| [09_state_constrained_generation_20261111.md](09_state_constrained_generation_20261111.md) | 状态约束生成实施与验证 |
| [10_multicandidate_validation_20261111.md](10_multicandidate_validation_20261111.md) | 固定预算多候选验证 |
| [11_resource_feasibility_20261117.md](11_resource_feasibility_20261117.md) | 资源可行性约束与固定预算复验 |
| [12_multisketch_validation_20261117.md](12_multisketch_validation_20261117.md) | 多 sketch 固定预算验收 |
| [LAPO_REPORT.md](LAPO_REPORT.md) | LAPO：基于真实硬件延迟的编译调度偏好优化 |

## 已合并的早期说明

旧版 Qwen tokenizer 迁移说明已收拢到当前训练指南：raw 数据必须用训练时匹配的 tokenizer 重新编码，不能沿用旧词表产生的 input_ids；依赖安装与数据构建以[训练指南](../guides/qwen3_train_guide.md)为准。

旧的生成优化总计划包含已完成的截断、token 边界、权重修复及后续设想，分别由上表 01–12 的实施和验收记录承接；当前执行顺序只维护[路线图](../ROADMAP.md)。2026 年 8–10 月两份 8B 草案中的环境快照、旧 1024-token 数据和单机内存数字已过期，规模选择要点已合并到路线图。

早期散落日志已清理。此处保留的历史实验结论仍需结合其日期与 audit_runs 的状态、统计和凭证阅读；本轮没有重新测量或将历史数值升级为当前结论。
