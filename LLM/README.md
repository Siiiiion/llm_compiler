# LLM 引导的 TVM 自动调度

代码主线：调度记录 → 审计与数据集 → CLM 训练 → 候选生成 → 编译与正确性验证 → 延迟测量 → 偏好训练 → 下一轮 workload 选择。

首次阅读从 [架构说明](docs/ARCHITECTURE.md) 开始；具体函数查 [模块与函数索引](docs/MODULE_INDEX.md)，当前工作查 [统一路线图](docs/ROADMAP.md)。

| 目录 | 职责 |
| --- | --- |
| [core](core) | workload 注册、调度记录身份、token 边界、文件与 worker 公共设施 |
| [data](data) | 调度文本、raw/Arrow 数据、审计后重建和 tokenizer |
| [generation](generation) | 模型生成协议、state 约束、资源可行域和任务选择 |
| [audit](audit) | 原始/归一化 state 构建门禁、持久 worker、审计凭证复用 |
| [validation](validation) | 结构评估、roundtrip、多候选与多 sketch 验收和统计 |
| [train](train) | CLM、SFT/DPO/LAPO、偏好数据和启动器 |
| [benchmark](benchmark) | 候选测量、Relay 正确性与性能、history 分析 |
| [experiments](experiments) | 特定日期/计划的诊断、训练对照与历史示例绘图 |
| [attention](attention)、[export_onnx](export_onnx) | attention 专项与模型导出工具 |
| [docs](docs) | 当前架构与路线图、使用指南、历史执行记录和研究设想 |
| audit_runs/ | 审计与实验产物，不作为可复用源码目录 |

在仓库根目录使用统一入口：

~~~bash
python -m LLM --help
python -m LLM generate --help
python -m LLM rebuild-audited --help
python -m LLM train-clm --help
python -m LLM validate-sketches --help
~~~

根目录只保留 `__init__.py` 和 `__main__.py`；命令注册和分发合并在后者，业务代码归对应子包。

已删除原来的逐文件转发入口，命令改为：

| 原脚本 | 统一命令（仓库根目录） |
| --- | --- |
| make_dataset.py | python -m LLM dataset |
| gen_state.py | python -m LLM generate |
| measure_programs.py | python -m LLM measure |
| task_scheduler.py | python -m LLM schedule |
| benchmark_relay.py | python -m LLM benchmark |
| benchmark_relay_batch.py | python -m LLM benchmark-batch |
| rebuild_audited_clm.py | python -m LLM rebuild-audited |
| validate_multisketch_generation.py | python -m LLM validate-sketches |

如果已进入 `LLM/`，使用 `python __main__.py generate --help`；从任意目录也可执行 `python /绝对路径/LLM/__main__.py generate --help`。参数继续传给原实现。代码中使用完整包导入，例如 `from LLM.generation.gen_state import gen_func`。

生产命令需使用安装了项目 TVM、PyTorch、Transformers、Datasets 的环境；ONNX 导出环境见 [export_onnx](export_onnx/README.md)。统一入口的总帮助仅依赖 Python 标准库。
