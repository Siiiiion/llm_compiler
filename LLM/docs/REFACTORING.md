# 2026-11-17 架构整理记录

本次在现有工作区基础上整理，保留此前未提交的代码内容。用户给出的 /home/qsy/workspace/complier/llm_compiler/LLM 与当前 /data3/qsy/complier/llm_compiler/LLM 经 realpath 核对为同一目录。

## 迁移

原根目录 42 个实现文件归入 core、data、generation、audit、validation、benchmark、experiments，train 保留训练器。历史 evaluate 绘图归 experiments；attention 和 export_onnx 保留专项边界。

第二轮按“主目录文件太多”的要求继续收拢：根目录 Python 文件从 46 个减至 2 个，只保留 __init__.py 与 __main__.py。42 个逐文件转发脚本、_legacy.py 和 cli.py 已删除；命令注册和分发合并到 __main__.py，提供 31 个子命令。各项业务功能仍由对应子包实现。

原 update_plan、根 plan 和散落训练文档共 23 份正文迁入 docs 的 guides、plans、history、research。相对源码/产物链接已按新位置调整；后续清理已删除旧位置的跳转页，文档只从 docs 导航。

## 公共函数与清理

- 文件指纹、原子 JSON/文本写入归 core/io；history 分析重复的 workload key 容错规范化函数也归该文件。
- UTC 时间和 bounded_pool 归 core/runtime；代表样本选择与算子/长度分层归 data/sampling。
- 多候选汇总归 validation/candidate_metrics；multisketch_metrics 直接依赖统计模块。
- 删除空 tlm_tile 占位类及无引用导入；清掉示例绘图中已注释的旧实现。
- 第二轮删除根目录转发层，并清理弃用根模块的字节码；LLM/.gitignore 忽略后续解释器缓存。
- 绘图、baseline/tune_relay 参数解析和训练启动器移入 main，导入模块不会直接绘图、解析训练参数或启动进程。
- Qwen 生成只需 measured-history 查询，抽成 generation/measured_history，消除对 gen/postprocess CLI 及其模糊 common 导入的依赖。

文件名中的 plan6 和历史日期保留，方便运行证据对应；相关脚本已归 experiments，不再充当生产模块。

## 审计与兼容

source_path 按唯一文件名查找真实实现；module_command 统一构造 python -B -u -m LLM.<子包>.<模块>。后台审计、数据重建、多候选验证、attention 和批量 benchmark 已迁移，测试使用完整包名导入。project_pythonpath 为新 worker 提供仓库根；code_dependencies 收集包内源码依赖并包括 __main__.py。

旧的 python LLM/gen_state.py 改为在仓库根目录执行 python -m LLM generate；若已进入 LLM，则使用 python __main__.py generate。训练指南、评测指南和根 README 已同步迁移命令和源码链接。verify-data 也补齐真正调用 verify_dataset 的命令实现。

历史 audit_runs、frozen_code、数据集、模型与已有日志均保留。重构改变源码指纹，新代码会拒绝把旧指纹误当成新代码下的复用凭证。继续旧实验应使用其冻结源码；新实验应使用新目录并重新建立相应凭证，不修改历史 manifest 绕过检查。

## 验证

第二轮验证以本节的命令与结果为准；第一轮保留旧模块的兼容性测试已经改为包入口、子进程路径和退出状态测试。

- 普通回归：181 passed，2 deselected；独立 verify-data 命令已用微型真实 raw/Arrow 数据走通。另加下述 2 项分布式回归，共 183 项通过。
- CPU 分布式回归：2 passed，使用此前已批准的沙箱外本机 socket 权限。
- 31 个统一命令的 --help 从 /tmp 经绝对路径入口执行，全部通过。
- 真实 TVM PopenPool worker 能在其他工作目录导入新包并执行函数。
- 346 个 Markdown 本地链接通过；模块索引按当前 AST 重新生成。
- Python AST 与 Black 100 列检查通过；本轮不运行 GPU 训练或性能实验。

普通回归命令（仓库根目录，使用安装项目依赖的 tlm Python）：

~~~bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
HF_DATASETS_CACHE=/tmp/llm_architecture_hf_cache OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/qsy/anaconda3/envs/tlm/bin/python -m pytest -q -p no:cacheprovider \
--basetemp=/tmp/llm_consolidation_final tests/python/unittest/test_llm_*.py \
-k 'not structural_metrics_and_errors_reach_all_training_ranks and not distributed_evaluation_removes_padding_rows'
~~~

分布式回归命令：

~~~bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
HF_DATASETS_CACHE=/tmp/llm_architecture_hf_cache OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/qsy/anaconda3/envs/tlm/bin/python -m pytest -q -p no:cacheprovider \
--basetemp=/tmp/llm_architecture_distributed \
tests/python/unittest/test_llm_generation_protocol.py::test_structural_metrics_and_errors_reach_all_training_ranks \
tests/python/unittest/test_llm_lapo_training.py::test_distributed_evaluation_removes_padding_rows
~~~

本次不运行全量数据重建、GPU 训练、生成搜索或性能实验。架构整理的通过不能替代对应硬件上的运行和性能验收。


## 日志与旧文档清理

按后续清理要求，删除 13 份 2026 年 4–7 月的散落训练、生成和导出日志，共 45.71 MiB；删除前检查，未发现进程持有这些日志。audit_runs 下 705 份近期实验日志及原始凭证保留。

删除 20 份迁移跳转页，以及旧 tokenizer 迁移说明、旧生成优化总计划、两份 8B 草案。LLM 子项目的 Markdown 从 53 份减到 29 份；早期数据兼容要点合并到 history 导航，模型规模选择合并到 ROADMAP。现行架构、操作指南、逐次实施与验收记录及论文材料保留；所有入站文档链接指向当前维护位置。

本轮删除文件和修改前文档的压缩备份：`/tmp/llm_cleanup_20261117_173906.tar.gz`，含 cleanup_manifest.json 的逐文件 SHA-256。LLM/.gitignore 已忽略后续生成的日志。

清理验证：268 个本地 Markdown 链接无断链，模块索引校验、仅标准库的统一 CLI 帮助和 git diff --check 通过。本轮只修改文档与忽略规则、删除旧产物，没有重跑模型训练或性能实验。
