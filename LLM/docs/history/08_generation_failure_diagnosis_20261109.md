# 模型自由生成失败定位记录

**2026-11-11 后续实施：**已实现 opt-in 状态约束生成并完成原 4 个及扩展 12 个 validation workload 的实际编译和 GPU 输出检查。原始自由生成诊断结论保留，约束效果与覆盖边界见 [第 9 份执行记录](09_state_constrained_generation_20261111.md)。

开始日期：2026-11-09。时间统一使用北京时间，机器产物保留原始 UTC/Unix 时间。

承接 [上一轮执行记录](07_execution_record_20261109.md)。本轮聚焦基线长卷积失败，以及原来源/过滤来源短训练后 matmul 退化。

## 执行范围与方法

1. 固定上一轮 4 个 validation workload、相同 median sketch，分别检查基线、原来源 32 步模型和过滤来源 32 步模型。
2. 每个模型比较阶段 token cap=256 与生产默认预算（minimum=256、scale=4，无额外 cap），均 greedy、seed=42、关闭 fallback。
3. 先回放 4 条已知正确后缀，确认相同 sketch 经当前解析器可以得到可构建 state。
4. 保存每个回调的 prompt、原始 token IDs、原始生成文本、预算、EOS、预算用尽标记、state 变换步骤；按实际 C++ 解析规则定位首个固定字段错误或输出不足。
5. 每条得到的 state 真实编译导出，构建 timeout=60 秒。该 timeout 高于前一轮的 30 秒，分析时分开报告；本轮不进行延迟比较。
6. 运行前冻结代码和模型文件摘要，运行后再次核验。原实验、正式数据和 checkpoint 只读。

新增诊断脚本：[diagnose_model_generation.py](../../experiments/diagnose_model_generation.py)。GPU 0 在启动前空闲，RTX 4090 / sm_89。实际 TVM 仍为 `/data3/qsy/complier/tlm` 的 Python 和动态库，运行 manifest 记录完整身份。

目标目录：`LLM/audit_runs/4090_plan6_generation_diagnosis_20261109_01`。

## 源码证据

CUDA 的 `SketchPolicyNode::GenStates` 依次调用 `InitFillTileSize`、`InitThreadBind`、`InitUnroll`，最后进行 InferBound 和 PruneInvalidState。`InitFillTileSize` 在 PPT 之后检查模型输出的 SPC opcode、stage_id、iter_id、extent、因子数量和方向；`InitThreadBind` 先根据已生成分块做线程约束检查，再在 TBS 请求 SPC 决策；`InitUnroll` 在 PRS 请求 PR 展开参数。

因此，“没有最终 state”至少可能来自字段不匹配、输出不足、线程约束检查失败或后续 state 剪枝，不能统一称为编译失败。只增加最终 build 预算也不能修复解析阶段失败。

## 过程

- 诊断实现和字段定位回归已建立。正式运行结果待补充。
- 已知后缀回放 4/4 成功，均实际编译导出通过，排除了这 4 个 sketch 本身无法经过解析链路的解释。
- 基线长卷积在 cap=256 与生产默认预算下都失败，均到达 TBS；PPT 必需的 67 个字段通过检查。TBS 第 4 个 SPC 的 extent 应为 128，模型输出为 12；后面 stage 4 的 extent 应为 28800，模型输出为 225。扩大 PPT 预算没有修复该错误。
- 已观察到基线输出在有效决策后继续产生自然语言、代码或重复字段；已完成的生成调用均用尽预算、没有 EOS。这是已取得的终止行为证据，尚不能说明补 EOS 训练一定能修复所有解析失败。
- 原来源模型 matmul 与长卷积的 PPT 固定字段正确，但在 TBS 生成前已被 InitThreadBind 丢弃。将其 PPT 因子恢复到 state 并执行 InferBound 后，matmul 根 stage 9 的第三级空间 tile 为 1×16=16 threads；长卷积 bgemm 根 stage 9 为 1×1×1×16=16 threads。当前 warp_size=32，实际 C++ 代码对这些非小 workload 要求 thread extent 至少为 32，因此失败是资源约束冲突，不能靠增加 token 修复。
- 原来源 softmax 则在 TBS 插入错误的 stage=3 决策，当前下一项应为 stage=1，属于另一种字段顺序错误。

## 最终诊断结果

未修改生产生成行为、未重训模型。全部 37 个诊断/控制案例完成，运行均无基础设施异常：第一轮 28 个（4 个已知后缀控制 + 24 个模型生成），两轮人工干预分别 5 个、4 个。第一轮耗时 351.84 秒；后两轮分目录记录真实开始/完成时间，不将等待时间解释为生成耗时。

| 模型 | cap=256 的可构建 workload | 生产默认预算的可构建 workload | cap=256 / 默认预算的生成 token |
| --- | ---: | ---: | ---: |
| 基线 | 3/4 | 3/4 | 2,816 / 2,828 |
| 原来源短训练 | 1/4 | 1/4 | 1,792 / 1,804 |
| 过滤来源短训练 | 2/4 | 2/4 | 2,048 / 2,060 |

结果复现上一轮结论。只有长卷积的 PPT 预算由 256 增加到 268；该阶段必需字段均已生成，后续固定字段和线程约束错误不受这 12 个额外 token 修复。**对这 4 个 workload，取消额外 cap 没有提高成功率。** 该结论不是所有 workload 的预算结论。

### 1. 两个短训练模型的 matmul：线程数量不满足约束

基线生成的空间第三级 tile 为 `1 × 32 = 32`；原来源和过滤来源模型均生成 `1 × 16 = 16`。实际当前 state 的 InferBound 结果保存在第一轮干预的 `geometry.json`。CUDA `InitThreadBind` 对这个 workload 执行 minimum thread extent 检查，要求至少一个 warp（32），因此两个模型都在第一次回调后、TBS 请求前丢弃 state。

单因素控制：仅把 PPT 的 stage=2、iter=5 的第二个 factor 从 16 改为 32，其他已输出字段不变，后续按实际变化后的 prompt 继续调用同一个模型。

| 干预 | 原来源模型 | 过滤来源模型 |
| --- | --- | --- |
| 不干预 | 无 state | 无 state |
| 一个 factor：16 → 32 | 1 个 state，真实编译导出通过 | 1 个 state，真实编译导出通过 |

这说明该候选的直接阻塞来自合法资源配置约束，不是固定字段词法错误、编译器失效或 token 不够。它没有证明为何 32 步训练会改变该决策的统计原因；仍需多 seed、更大验证集合以及训练更新强度对照。

### 2. 长卷积：多阶段错误叠加

| 模型 | 原始首次失败 | 只修首个问题 | 完整诊断控制 |
| --- | --- | --- | --- |
| 基线 | TBS 复制 extent 错：128→12，28800→225 | 保留因子，只修两个 extent 后到达 PRS；PRS 却以 `5 2 1 SP ...` 开头，应为 `PR 4 0 auto_unroll_max_step$...` | 再提供合法 PRS 控制后，解析和编译导出通过 |
| 原来源短训练 | PPT 决策导出的 thread extent=16<32，TBS 前失败 | 线程 factor 16→32 后到达 PRS；输出以 `PR 5 O ...` 开始，当前要求的 stage 列表为 4、8、11 | 保留模型因子并使用 state 固定字段及合法 PRS 控制，解析和编译通过 |
| 过滤来源短训练 | 同上，thread extent=16<32 | 调整因子后 TBS 又复制错误 extent；修正 TBS 后到达 PRS，仍以 `PR 5 SPC ...` 开头 | 再补合法 PRS 控制后，解析和编译通过 |

完整控制的 PRS 统一使用当前 prompt 中的 stage 列表 `[4,8,11]`、iter=0、合法选项 `auto_unroll_max_step$0`。这是明确使用状态信息的 oracle 对照，**不计入模型质量，也不是已部署的自动修复策略**。固定字段控制不改变对应 SPC 的自由 factor；资源控制仅改变日志中列出的一个 factor。

所有长卷积在这些控制后都可编译，说明已经定位到足以解释当前失败的机制；真实模型生成仍保持原始失败结果。控制案例没有做独立 GPU 输出正确性和延迟检查，不能称为更快调度。

### 3. 原来源 softmax：重复/错序的固定 stage

TBS 期望 stage 顺序为 4、3、1。原来源模型在第三项插入 stage=3、iter=10，下一项本应 stage=1，因此解析器立即退出。两个预算设置下同样出现。基线和过滤模型没有该错误。尚未对这个 softmax 做修复对照，不外推到其余 workload。

### 4. 终止行为已得到直接证据

24 个模型案例一共 **52 次非空生成**，**52 次均用尽预算，0 次生成 EOS**。有效候选的输出也会在必需决策后继续自然语言说明、Python 代码或重复/畸形调度字段。解析器通常只消费当前阶段所需前缀，因此有些候选最终仍能通过；多余 token 不能当作有效调度决策数。

这与计划 6 已确认的“正式数据未追加 EOS”一致，但本次没有开展 EOS 重训，因此不能把追加 EOS 的改善称为已验证。更直接的工程方向是阶段结束即停止生成，PRS 限定合法选项；不能只增加预算。

## 本轮代码、验证和产物

新增两个独立诊断脚本及 [字段检查回归](../../../tests/python/unittest/test_llm_generation_diagnosis.py)。没有修改本轮启动前的生产/评估生成逻辑。格式和 Python 语法检查通过；诊断与共享生成相关回归 **14 passed、1 个分布式测试未重跑**（本轮未修改分布式代码），耗时 4.15 秒。`git diff --check` 通过。

- [第一轮原始模型 trace](../../audit_runs/4090_plan6_generation_diagnosis_20261109_01/results.json)：28 个案例，每回调保留完整 prompt、IDs 和原始输出。
- [第一轮人工干预](../../audit_runs/4090_plan6_generation_interventions_20261109_01/results.json)：matmul 单因子控制及长卷积首层控制。
- [线程几何证据](../../audit_runs/4090_plan6_generation_interventions_20261109_01/geometry.json)：从真实 state 推导的根 stage/iterator extent。
- [第二轮长卷积控制](../../audit_runs/4090_plan6_generation_interventions_20261109_02/results.json)：多阶段分离与最终编译证据。
- [关联汇总报告](../../audit_runs/4090_plan6_failure_localization_20261109_01/report.json)：三个运行的 manifest、冻结代码及模型输入摘要、统计和干预边界。

每个运行均保存启动时的 `frozen_code/` 和 manifest，结束时检查自身代码摘要。干预脚本在第一轮完成后扩展了第二轮 oracle 控制，各运行保留执行版本。汇总以冻结副本为准，不要求历史执行版本与未来源码始终相同。模型 safetensors/config/tokenizer 等关键文件在第一轮前后及汇总时都重新核对。

复现参数（重跑须改用新目录）：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/qsy/anaconda3/envs/tlm/bin/python -u LLM/__main__.py diagnose-generation \
  --run-dir LLM/audit_runs/4090_plan6_generation_diagnosis_20261109_01

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/qsy/anaconda3/envs/tlm/bin/python -u LLM/__main__.py diagnose-interventions \
  --run-dir LLM/audit_runs/4090_plan6_generation_interventions_20261109_01

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/qsy/anaconda3/envs/tlm/bin/python -u LLM/__main__.py diagnose-interventions \
  --run-dir LLM/audit_runs/4090_plan6_generation_interventions_20261109_02 --conv-oracle-controls
```

第一轮人工干预的准确代码版本见其冻结副本；当前版本仅增加可选 oracle 参数，默认案例一致。

## 下一步实施依据

1. **固定字段由当前 state 提供，自由决策由模型预测。** 优先用小规模约束生成方案，保留原始输出和修复来源，不能把人工补齐当作模型成功。需要处理字段缺失/多余决策，不能只按固定偏移盲改。
2. **在分块因子生成时考虑线程约束。** 先覆盖 minimum warp、vthread、线程上限和共享内存约束；本次的单因子控制只是两个具体 sketch 的机制证据，不是通用 repair 算法。
3. **PRS 改成合法状态与选项的有限选择，并增加阶段结束条件。** 选择完成后停止，分别报告未约束输出与约束输出的成功率、token 和耗时。EOS 监督另作隔离对照。
4. **训练暂不扩量。** 扩展固定验证集、等监督 token、多 seed，并比较较温和更新和基线保持情况；当前诊断解释具体失败，不足以决定新的学习率或训练规模。

本轮仍只使用 validation 诊断集，没有使用最终 hold-out。未取得新模型提升、GPU 正确性或整网加速结果。
