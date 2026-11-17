# 状态约束生成实施与验证

执行日期：2026-11-11。承接 [生成失败定位](08_generation_failure_diagnosis_20261109.md)。

## 实施范围

新增 [constrained_generation.py](../../generation/constrained_generation.py)，通过现有共享生成协议接入生产和结构评估。显式设置 `constraint_mode=state_v1` 才启用；默认仍为 `none`。

1. 当前 state 的 transform steps 决定 SPC 的 stage、iterator、extent、因子数量与方向，PRS 的 stage 列表也来自当前 state。固定字段在解码时强制提供，不从模型输出事后猜测或修补。
2. 模型选择自由因子；当前搜索子集限定为正整数且因子乘积整除 extent。这是保守的合法子集，CUDA 还允许部分带 padding 的组合，不能宣称覆盖所有合法调度。
3. 对可识别的标准 GPU FSP 根 stage，联合约束 vthread、thread 上限和最小 warp；小 workload 按当前 C++ 规则允许不足一个 warp。未支持的 tiled root 几何明确拒绝，不伪装成已覆盖。
4. PRS 限定到当前后端的 `0/16/64/512/1024`；每阶段所需字段完整后立即停止。预算耗尽但字段不完整时拒绝该候选。
5. 最终 state 单独 lowering，并使用实际硬件参数检查共享/局部内存、线程、vthread 和向量宽度。共享内存目前是生成后的资源门禁，尚未实现解码时的共享内存可行性搜索。

当前实际使用的 TVM 仍为 `/data3/qsy/complier/tlm/python/tvm` 及其动态库。本轮没有重编译或替换 C++ 后端。该构建注册了 `tir.transform.VerifyGPUCode`，但缺少同名 Python wrapper；实现通过注册函数取得该 pass。

## 验证设计与边界

[validate_constrained_generation.py](../../validation/validate_constrained_generation.py) 固定 validation sketch、seed=42、greedy、每 workload 一次候选尝试、关闭 fallback。阶段预算保持 minimum=256、scale=4、不设额外 cap，两种模式相同。

- 原 4 个 workload：基线、原来源 32 步模型、过滤来源 32 步模型，各自比较 `none` 与 `state_v1`，共 24 个案例，GPU 0。
- 扩展到 12 个 validation workload：基线比较两种模式，共 24 个案例，GPU 1。其中包含原 4 个；新增 8 个按算子类别和长度分组确定性选取，不是 12 个新增样本，也不是随机泛化评估。
- 每个最终候选在独立子进程中检查资源、实际编译导出，并运行 GPU 输出检查；每候选硬超时 180 秒。使用两组随机输入（42、43），`rtol=1e-3, atol=1e-4`。
- 参考结果来自相同 TE 的未调度 LLVM 实现，可检查调度前后输出一致性；它不是独立的算子语义参考。当前没有 GPU 延迟或整网速度比较。
- 每次运行冻结源文件、模型/输入摘要和编译器身份，结束复查身份。`status=SUCCEEDED` 表示流程完成，候选质量必须查看 `summary` 的 parsed、built、correct 及逐候选错误。

扩展集合覆盖 batch matmul 3 个、卷积 3 个、matmul/dense 2 个、softmax 2 个、pooling 1 个、其他类别 1 个。始终只用 validation，不使用最终 hold-out。

## 运行与结果

正式原 4 个 workload 对照：[4090_constraints_pilot_20261111_03](../../audit_runs/4090_constraints_pilot_20261111_03)。

扩展基线对照：[4090_constraints_expanded_20261111_01](../../audit_runs/4090_constraints_expanded_20261111_01)。

原 4 个 workload 的 24 个案例已经全部完成，运行身份复查通过，无基础设施异常。成功条目都通过了资源、编译和两组输入的 GPU 输出检查。

| 模型 | 自由生成通过 | 约束生成通过 | 自由生成 / 约束生成 token | 自由生成 / 约束生成耗时（秒） |
| --- | ---: | ---: | ---: | ---: |
| 基线 | 3/4 | 4/4 | 2,828 / 510 | 67.79 / 15.38 |
| 原来源 32 步模型 | 1/4 | 4/4 | 1,804 / 510 | 43.05 / 15.40 |
| 过滤来源 32 步模型 | 2/4 | 4/4 | 2,060 / 510 | 50.38 / 15.19 |

耗时只含生成阶段，不含编译、CPU 参考和 GPU 输出检查。约束解码已包含固定字段 token 与掩码开销；不是把这些字段当成模型自由预测正确。原 4 个案例的改善不能作为新训练模型质量提高的结论。

扩展运行的 24 个案例也已全部完成，身份复查通过，无基础设施异常：

| 基线模式 | 解析 / 编译 / GPU 输出通过 | 生成 token | 生成耗时（秒） |
| --- | ---: | ---: | ---: |
| 自由生成 | 6/12 | 8,472 | 203.11 |
| 状态约束 | 12/12 | 1,354 | 40.68 |

自由生成通过的 6 个均保留，另外 6 个恢复；本集合没有正确性退化。token 减少约 84.0%，生成时间减少约 80.0%。这是一轮固定顺序、单 seed、小验证集合的工程结果，不是调度 kernel 的加速比，也不是统计泛化保证。完整 48 个模型案例中，共 36 个最终候选，全部通过资源、真实编译和 GPU 输出检查；两个运行包含重复 workload，合计仍仅覆盖 12 个不同 workload。

[汇总凭证](../../audit_runs/4090_constraints_summary_20261111_01/report.json) 重新核对两个运行的结束状态、冻结/当前源码摘要、样本身份、案例完整性与汇总统计，并保存 manifest、results、summary、samples、status 的 SHA256。模型文件摘要已由各运行结束检查复核。相邻 `collect_report.py` 保留汇总脚本。

前两次试运行仅保留排错：`pilot_01` 的日志后缀参数缺少点号导致检查结果收集失败；`pilot_02` 在资源检查 Python API 不兼容处失败且随后中断。`pilot_02/base_none_000` 的检查文件曾用于修复后的单条构建探针，不能将该目录作为统一冻结的验收结果。正式结论只引用 `_03` 及扩展运行。

## 回归与后续门禁

新增测试覆盖动态数字前缀、因子依赖、warp/vthread/thread 联合约束、小 workload、PRS 有限选项、阶段停止、token 预算不足、生产/评估路由一致性，以及真实 lowering 后线程和共享内存超限拒绝。约束与共享协议回归 22 项通过（1 项未涉及的分布式测试未在该命令运行）；约束、诊断和 roundtrip 回归 25 项通过，两组有重叠，不能直接相加。

使用 Black 的纯格式化接口检查，三个新增文件与 Black 100 列输出一致；Python 语法和 `git diff --check` 通过。没有修改旧 checkpoint 或正式数据。

本次扩展集没有资源/grammar 拒绝，但约束覆盖仍限定于上述已识别模板和 divisor 因子子集，未验收完整 validation 集或所有 sketch。共享内存仍在 lowering 后检查。当前为 opt-in，不据此更改生产默认值。

后续按以下门禁推进：

1. 在更多 validation sketch 与固定 K>1/尝试预算下检查唯一候选数、重复率、资源拒绝及时间，区分能够构建与能够稳定提供足够候选。
2. 在训练侧候选上核对独立算子参考并做真实延迟比较，确认约束后的有限选择保留有价值的性能多样性；最终 hold-out 继续隔离。
3. 再根据上述指标设计较温和更新、等监督 token、多 seed 的训练对照。当前三种模型在原 4 个 workload 的约束有效性均为 4/4，单候选通过率已经无法区分模型，且没有 kernel 性能结果，因此本轮不以该指标饱和作为扩大训练的依据。上一轮 32 步训练的 token 总量差异仍需在新实验设计中消除。

## 复现参数

重跑须使用新目录，以下为本次实际运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/qsy/anaconda3/envs/tlm/bin/python -u LLM/__main__.py validate-constrained \
  --run-dir LLM/audit_runs/4090_constraints_pilot_20261111_03 --check-runtime

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/qsy/anaconda3/envs/tlm/bin/python -u LLM/__main__.py validate-constrained \
  --run-dir LLM/audit_runs/4090_constraints_expanded_20261111_01 \
  --max-workloads 12 --models base --check-runtime
```
