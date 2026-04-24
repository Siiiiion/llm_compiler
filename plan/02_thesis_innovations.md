# 毕业论文创新点设计（基于 TLM × Qwen3-0.6B × TVM auto_scheduler）

> 配套项目：`qusiyuan/projects/llm_compiler/LLM/`
> 生成日期：2026-04-24
> 定位：本文档给出 4 个**互不冲突、可组合**的论文创新点。每个创新点都给出：动机 → 方法 → 可落地实现路径 → 评估指标 → 与现有工作的差异 → 工作量/风险评估。
> 建议组合：**创新点 1 + 创新点 2** 作为论文主干（理论与性能双轮），**创新点 3** 作为应用侧验证（工业价值），**创新点 4** 作为 ablation / generalization 章节。

---

## 总体定位与选题背景

**问题域**：张量程序自动调优（tensor program autotuning）。TVM 的 Ansor/auto_scheduler 通过 SketchPolicy + 演化搜索在候选 schedule 空间中寻找低延迟实现；搜索成本高、迁移性差。

**已有工作**：

- **TenSet / TLP / MetaSchedule** — 用 cost model 做 latency 预测，替代真实测量。
- **BaCO / TpuGraphs / Moses** — 学习调度选择器。
- **Meta / OpenAI 做过用 LLM 写 CUDA kernel**（2024–2025 若干 arXiv 预印）但面向 kernel 源码而非 TVM schedule 决策。
- 本项目：**用继续预训练的 Qwen3-0.6B 作为 SketchPolicy.gen_states 的采样后端**。

**当前项目的短板 → 创新机会**：

1. 只学"合法结构"，没有"性能偏好"（数据侧没有 latency）。
2. 生成阶段无约束，非法率不低，依赖 LocalBuilder 过滤。
3. 没有跨硬件迁移实验。
4. 没有与传统搜索算法的深度融合实验。

这 4 点正好对应下面 4 个论文创新点。

---

## 创新点 1：语法约束下的张量程序生成（Grammar-Constrained Tensor Program Generation）

### 1.1 动机

- Qwen3 BPE tokenizer 对 TVM schedule token（`SPC` / `CI` / `CR` / 整数 tile factor 等）切分效率低，序列长度被放大 1.5–2×，训练与推理成本高。
- 生成阶段若不约束，模型可能输出「语法不合法」的 suffix（例如 SP step 参数个数错误、loop id 越界），需要 LocalBuilder 才能过滤，浪费算力且隐藏了模型真实决策能力。
- 现有「grammar-constrained decoding」工作主要在 JSON / SQL / Python 领域，**尚无针对 TVM auto_scheduler step grammar 的系统工作**。

### 1.2 方法

**两件事合并：领域词表 + 解码语法约束。**

#### (a) 领域词表扩展（Domain-Specialized Tokenizer Extension）

- 枚举 TVM auto_scheduler 所有 step 类型 token（`SP`、`PR`、`FU`、`CA`、`CI`、`CR`、`CHR`、`RF`、`PPT` 等）。
- 枚举高频整数常量集合（tile factor、循环数）。
- 通过 `tokenizer.add_tokens(...)` 一次性扩进词表，`model.resize_token_embeddings(...)`，用**子词均值**初始化新 embedding（warm start）。
- 论文贡献：给出「张量调度 DSL 的最小充分 token 集合」定义与构造流程。

#### (b) TVM Schedule Step Grammar + Logits Processor

把 auto_scheduler step 形式化成 BNF / CFG：

```bnf
suffix   ::= step+
step     ::= sp_step | ca_step | ci_step | cr_step | fu_step | rf_step | pa_step | pt_step
sp_step  ::= "SP" loop_id ":" factor_list
factor_list ::= integer ("," integer)*
ca_step  ::= "CA" stage_id ":" target_stage_id ":" level
...
```

实现方式：

- **方案 A（最轻）**：自写 `LogitsProcessor`，在每一步根据当前生成状态（FSM / 栈）计算 `allowed_token_ids`，对其它 logits 置 `-inf`。
- **方案 B（更通用）**：用 `outlines` 或 `lm-format-enforcer` 库，把 CFG 编译成 FSM 跑。
- **方案 C（最硬核，论文亮点）**：基于 auto_scheduler 的 C++ 合法性检查抽象出一个**在线合法性检查器**，每生成一步调用检查器过滤合法 next-token 集合 —— 真正做到 **100% 语法合法 + 局部语义合法**。

#### (c) 与现有项目的集成点

- `extend_tokenizer.py`（新增）：做 (a)。
- `make_dataset.py`：按新 tokenizer 重建 `4090_gen_qwen_v2`。
- `gen_state.py::gen_func`：在 `model.generate` 的 `logits_processor=` 里挂入 `TVMGrammarLogitsProcessor`。

### 1.3 实验与评估

| 指标 | 目标（对比 baseline 当前 stage1） |
| --- | --- |
| `parse_valid_rate` | 100%（hard constraint） |
| 平均 input_ids 长度 | 缩短 ≥ 30% |
| token 生成速度（tokens/s） | ≥ baseline 1.5× |
| `build_valid_rate` | ≥ baseline + 10pp |
| `eval_loss` | 不劣于 baseline |

Ablation：

- (a) only vs (a)+(b) vs (a)+(c)
- Grammar 粒度：只管 step 关键字 vs 管 step 内部参数个数 vs 管到 loop_id 合法性
- 小模型（0.6B）vs 大模型（7B）上的收益差

### 1.4 与现有工作差异

- Outlines / lm-format-enforcer：都在通用 JSON / regex 领域，**没有针对 TVM auto_scheduler 的工作**。
- TenSet / TLP：cost model 方向，不涉及生成。
- 本创新点可同时讲「DSL tokenizer 设计」+「grammar-constrained decoding 在张量编译器的首次系统应用」两条线。

### 1.5 工作量与风险

- 工作量：**2–3 周**（扩词表 + 重建数据 + 单元测试 + grammar processor + ablation）
- 风险：方案 C 要对 auto_scheduler 合法性规则做剥离，阅读 TVM C++ 代码有一定成本。若时间紧张可只做到方案 B。

---

## 创新点 2：基于真实延迟测量的偏好优化（Latency-Aware Preference Optimization, LAPO）

### 2.1 动机

- 当前 Stage1 只能学「合法结构」；Stage2 只是低学习率继续训练，**没有真实性能信号**。
- 这恰恰是「用 LLM 做 compiler autotuning」最核心的问题：**如何把 latency 测量变成可微训练信号？**
- 近两年 DPO（Direct Preference Optimization）在 RLHF 领域已经替代 PPO 成为事实标准，但在编译器领域尚无系统工作。

### 2.2 方法

分成**偏好对构造** + **DPO 训练** + **奖励模型辅助**三步。

#### (a) 偏好对构造（Preference Pair Construction）

对每个 workload / 每组相同 prompt（即 `steps[:ppt_idx]` 相同）：

- 在其所有 measured record 中，选 latency 最低的作为 `chosen`。
- 从其余记录中随机选一条 latency 显著更高（>1.5× `chosen` latency）的作为 `rejected`。
- 若数据侧 latency 不全，先用 cost model（见 (c)）给无 latency 样本打伪 latency。

数据：`(prompt, chosen_suffix, rejected_suffix, latency_gap)`。

#### (b) 直接偏好优化（DPO）

用 `trl.DPOTrainer`（HuggingFace TRL 生态现成支持）：

```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.1,
    loss_type="sigmoid",
    max_prompt_length=512,
    max_length=1024,
    ...
)
trainer = DPOTrainer(
    model=policy_model,            # = stage1 checkpoint
    ref_model=ref_model,            # = frozen stage1
    args=dpo_config,
    train_dataset=preference_ds,
    tokenizer=tokenizer,
)
```

论文亮点：提出 **Latency-Weighted DPO**（LW-DPO）——把 DPO loss 按 `log(latency_gap)` 或 `(lat_rej / lat_chosen - 1)` 加权：

```text
L = -E[ w(Δlat) · log σ(β · (π(chosen) - π(rejected)) - β · (π_ref(chosen) - π_ref(rejected))) ]
```

直觉：延迟差距越大，这对偏好越重要，loss 权重越高。可以证明等价于对 `implicit reward` 做幅度缩放。

#### (c) 可选：Reward Model 辅助（Cost Model as Reward）

- 训练一个轻量 latency 预测头（TLP-like 结构或直接 Qwen3 + regression head）做伪 latency 标注器。
- 用它给无真实 latency 的样本打分，构造大规模合成偏好对。
- 论文故事：**"真实 latency 稀缺 → cost model 放大偏好信号 → DPO 训练"**。

### 2.3 数据侧工程准备

**硬要求**：需要搭建 4090（或你目标硬件）真实测量流水线。

- 基于已有 `gen_state.py` 的 `LocalBuilder`，再追加 `LocalRunner` / `RPCRunner` 得到 ms 级 latency。
- 目标采集规模：每个 workload 至少 5 个 suffix（从 Ansor search history + LLM 生成 + 随机扰动 三路拼出）。
- 预计 1–2 周 GPU 时间。

### 2.4 实验与评估

| 指标 | 基线 | 目标 |
| --- | --- | --- |
| Geo-mean speedup on hold-out set（相对 Ansor） | 1.0× | ≥ 1.10× |
| `build_valid_rate` | stage1 值 | 不降 |
| Search time to reach `0.9 × optimal` latency | Ansor baseline | ≤ 0.5× |
| DPO vs PPO vs SFT baseline 对比 | — | 必做 |

Ablation：

- 偏好对数量（10K / 100K / 1M）
- β（DPO 温度）
- LW-DPO vs vanilla DPO vs reward model PPO
- Ref model 冷冻 vs slow update

### 2.5 与现有工作差异

- DPO 原始论文：通用语言偏好。
- 本工作：首次在**张量编译器调度生成**上应用 DPO，且提出 Latency-Weighted DPO 来融合连续 reward 与 pair-wise preference。
- 相比 cost model（TenSet / TLP）：cost model 只能给分数，不能生成；本工作**生成 + 偏好统一在同一个 LLM 里**。

### 2.6 工作量与风险

- 工作量：**4–6 周**（数据采集 2 周 + DPO 训练 1 周 + LW-DPO 公式推导与消融 1 周 + 写作）。
- 风险：
  - 真实 latency 采集耗 GPU；若无法采集，只能用合成 reward，说服力会打折。
  - Qwen3-0.6B 容量是否足够体现偏好信号：可以 fallback 到 1.5B / 3B。

---

## 创新点 3：LLM 引导的混合自动调优（LLM-Guided Hybrid Autotuning）

### 3.1 动机

- 纯 LLM 生成 schedule 存在"尾部长"问题：前几次命中率高，但长时间搜索下 evolutionary search 仍然更强。
- **实际工业场景关心的是「给定时间预算下找到的最低延迟」**，而不是单次命中率。
- 所以最实用的方式是：**LLM 做种群先验 / 变异建议，Ansor 的 evolutionary search 做精修**。这个组合在文献里几乎没有系统研究。

### 3.2 方法

把 LLM 以三种方式注入 Ansor 的 SketchPolicy：

#### (a) Warm-start Population

- 在 evolutionary search 第 0 代，用 LLM 采样生成 `K` 个高质量 state，作为初始种群。
- 替换 Ansor 的随机 `init_population` 步骤。

#### (b) LLM-as-Mutation Operator

- 在每一代，以概率 `p` 用 LLM 做 mutation（给 LLM 看当前 state 的 prompt + `"mutate"` 指令，让它生成一个变体）。
- 剩余 `1-p` 用 Ansor 原有 mutation rules。
- 论文贡献：给出 **p 随搜索 iteration 衰减** 的 schedule（早期 LLM 主导，后期传统 mutation 主导），并分析其收敛行为。

#### (c) LLM-as-Surrogate Reranker

- LLM（加一个 regression head）同时输出 latency 预测，用作 evolutionary search 的 ranking 函数。
- 与 Ansor 自带的 cost model 融合：`score = α · LLM_reward + (1-α) · cost_model_score`。

### 3.3 集成路径

- TVM 的 `SketchPolicy` 在 Python 侧开放了 `gen_states` 回调（项目已用到）、`mutation_callback`、`population_init_callback` 等插桩点（部分可能需要二次开发）。
- 不要 fork TVM C++ 侧，全部用 Python 回调实现，保证兼容性。
- 关键类：`tvm.auto_scheduler.SketchPolicy`、`PopulationGenerationRule`。

### 3.4 实验与评估

| 场景 | 指标 | 基线 | 目标 |
| --- | --- | --- | --- |
| Fixed time budget (2h) | Final latency geomean | Ansor | ≤ 0.90× |
| Fixed target (match Ansor best) | Time-to-target | Ansor | ≤ 0.50× |
| Warm-start only vs +Mutation vs +Reranker | ablation table | — | — |
| 不同 p(iter) 衰减曲线 | convergence curve | — | — |

### 3.5 与现有工作差异

- Ansor 自己的 XGB cost model：只做 ranking，不做生成。
- MetaSchedule：主要替换 SketchPolicy，但仍是传统 search，没有 LLM。
- 2024–2025 一些「LLM 生成 CUDA」的工作：端到端生成 kernel 源码，**没有与传统 autotuner 混合**。
- 本工作填补「LLM + evolutionary search」在张量编译领域的空白，强调**工业落地友好**（不替换现有 stack，只增强）。

### 3.6 工作量与风险

- 工作量：**3–4 周**（SketchPolicy 插桩 + 三种注入方式实现 + 多 workload 实验）。
- 风险：
  - `SketchPolicy` 的回调接口对 mutation 支持可能不完整，需要少量 C++ 改动，提前评估。
  - 评估需要多个 workload + 真实硬件，时间成本高。

---

## 创新点 4：跨硬件迁移与小样本适配（Cross-Architecture Transfer with Per-Device Adapters）

### 4.1 动机

- 当前项目仅在 4090 数据上训。
- 现实中硬件种类多（V100 / A100 / 4090 / H100 / 国产卡），每换一张卡就重跑 autotune 成本极高。
- 学术问题：**LLM 学到的调度知识有多少是硬件无关的？**

### 4.2 方法

#### (a) 把训练集合并成多硬件混合

- 输入 prompt 里显式注入 `<device=4090>` 等 special token。
- 训练一个 device-conditioned model。

#### (b) Device Adapter（LoRA-per-Device）

- 主干 Qwen3-0.6B 冻结；每张硬件训练一个独立的 LoRA 适配器（每个 ~10MB）。
- 推理时按 target device 动态挂载对应 adapter（`peft` 原生支持 `load_adapter/set_adapter`）。

#### (c) Few-shot adaptation

- 对新硬件 X，只给 100–1000 条 measurement，用 LoRA 几十步训练即可得到可用 adapter。
- 对比：
  - 0-shot（device token 但无 adapter）
  - Few-shot（100 / 500 / 1000 measurement）
  - Full retrain（作为上界）

### 4.3 数据侧工程准备

- 依赖 `TenSet` 或你已有的 `v100 / a100 / 4090 / i7` 多硬件测量数据。
- `common.py` 已经区分了 `model_list = ['i7', 'v100', 'a100', '2080', 'None', '4090']`，数据目录结构现成。

### 4.4 实验与评估

| 设置 | 指标 |
| --- | --- |
| 0-shot on A100 (trained on 4090) | parse_valid_rate / build_valid_rate / speedup vs Ansor |
| 100-shot LoRA on A100 | 同上 |
| Full retrain on A100 (upper bound) | 同上 |
| Device adapter ablation | LoRA r=4/8/16/32 对效果的影响 |

论文贡献：定义 **「Device-Invariant Scheduling Knowledge」指标**：在 0-shot 到 full-retrain 区间内，能保持上界多少百分比。

### 4.5 与现有工作差异

- TenSet 讨论过 cost model 的跨硬件迁移，但**没有讨论生成模型**。
- MoCoG / XGBoost 模型：硬编码 feature，很难迁移；本工作用 LLM 天然可共享 token 语义。
- 与大模型 LoRA-per-language / LoRA-per-domain 的工作遥相呼应，但**在编译器领域是首次**。

### 4.6 工作量与风险

- 工作量：**2–3 周**（多硬件数据归集 + 训 adapter + 实验）。
- 风险：
  - 如果 A100 / V100 测量数据质量差（latency 缺失），实验说服力下降。
  - 可以退化成只做 `4090 → V100` 单迁移路径。

---

## 组合策略与论文骨架建议

### 推荐论文骨架

```text
标题（建议）：
    LLM4TVM: 张量程序自动调度的大模型方法
    —— 从结构合法性到延迟偏好的统一建模

Chapter 1  绪论 & 相关工作
Chapter 2  基础：TVM auto_scheduler、Qwen3、DPO、Grammar-Constrained Decoding

Chapter 3  【创新点 1】语法约束下的张量程序生成
            3.1  TVM Schedule DSL 的 tokenization 困境
            3.2  领域词表扩展方法
            3.3  TVM Step Grammar 与 Logits Processor
            3.4  实验：结构合法率 / 序列长度 / 训练吞吐

Chapter 4  【创新点 2】基于真实延迟的偏好优化（LAPO / LW-DPO）
            4.1  偏好对构造
            4.2  Latency-Weighted DPO 公式
            4.3  Reward Model 辅助（可选）
            4.4  实验：speedup / time-to-target

Chapter 5  【创新点 3】LLM 引导的混合自动调优
            5.1  Warm-start / LLM-mutation / LLM-reranker
            5.2  衰减型注入策略
            5.3  实验：与 Ansor / MetaSchedule 对比

Chapter 6  【创新点 4】跨硬件迁移与适配
            6.1  Device-conditioned 训练
            6.2  LoRA-per-Device
            6.3  实验：0-shot / few-shot / full retrain

Chapter 7  工程实现与系统集成（`gen_state.py`, vLLM, TrainerCallback 等）
Chapter 8  总结与未来工作
```

### 选题取舍建议

若时间只有**3–4 个月**（单人硕士），建议至少保留：

- **必做**：创新点 1（语法约束 + 词表扩展）→ 直接对接现有代码，必出实验图。
- **必做**：创新点 2（至少 vanilla DPO 版本）→ 论文"性能"主线。
- **二选一**：创新点 3（工业价值高）或创新点 4（学术 novelty 高）。

若有 **6 个月**，可以全做，且互相能共享实验基础设施（都依赖你升级后的 `eval_struct.py` 与 vLLM 推理栈）。

---

## 工程先置条件 Checklist（开始论文实验前必须完成）

这些项目来自 `01_optimization_plan.md`，是所有创新点的地基：

- [ ] 动态 padding + `ppt_end` 预计算 → 训练时间压到 6–8h
- [ ] 扩 TVM 词表 → 序列长度 -30%（也是创新点 1 的基础实验）
- [ ] `eval_struct.py` + `TrainerCallback` → 有 `parse_valid_rate / build_valid_rate` 指标
- [ ] `gen_state.py` 切 vLLM + `is_build=True` 默认 → 推理吞吐 3–5×
- [ ] 真实 latency 测量流水线（`LocalRunner / RPCRunner`）→ 创新点 2 的数据源
- [ ] 多硬件数据归集（至少 4090 + V100）→ 创新点 4 的数据源

完成以上，再动笔做论文实验，整体节奏会顺利很多。

---

## 一句话总结

- **创新点 1** 负责"做得对"（合法性 + 效率）；
- **创新点 2** 负责"做得好"（性能偏好）；
- **创新点 3** 负责"做得快"（工业落地）；
- **创新点 4** 负责"做得广"（跨硬件迁移）。

四者在同一个 Qwen3-0.6B + TVM 代码仓上递进展开，实验环境共享，论文逻辑自洽，且每一个都有明确可度量的 KPI。
