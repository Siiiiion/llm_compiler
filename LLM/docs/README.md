# LLM 文档入口

| 需要了解什么 | 阅读位置 |
| --- | --- |
| 项目模块、主链路、数据契约和依赖 | [架构说明](ARCHITECTURE.md) |
| 每个实现文件有哪些函数、谁调用谁 | [模块与函数索引](MODULE_INDEX.md) |
| 当前完成什么、下一步做什么 | [统一路线图](ROADMAP.md) |
| 重构迁移、入口迁移、验证和限制 | [本次整理记录](REFACTORING.md) |
| CLM 与 DPO/LAPO 操作方法 | [训练指南](guides/qwen3_train_guide.md) |
| 留出 workload 与整网测试 | [评估指南](guides/eval_workload_test_guide.md) |
| 历次实施与训练执行证据 | [历史记录](history/README.md) |
| 8B 与模型规模选择 | [路线图中的规模选择](ROADMAP.md#模型规模选择) |
| 原 plan 的论文和研究设想 | [研究规划](research/README.md) |

维护规则：当前执行顺序只维护 ROADMAP；实验结束后追加 history 记录并链接产物；模型、数据和机器路径写入运行配置或实验 manifest。源码链接指向实际子包；命令示例使用统一入口。audit_runs 中的历史 manifest、冻结代码和日志保留原样。
