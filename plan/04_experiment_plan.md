# LLM 辅助 TVM 自动调度的实验计划

## 1. 实验目标

本文实验旨在系统评估大语言模型辅助张量程序自动调度的有效性。围绕 TVM AutoScheduler 与 Qwen3-0.6B 的结合，实验需要回答以下问题：

1. 基于 TVM 原生测量记录的结构化序列表示，是否能够被大语言模型有效学习。
2. 以 `PPT` 为边界的调度草图补全任务，是否能够生成 TVM 可解析、可构建的调度状态。
3. 与传统 Ansor/AutoScheduler 搜索相比，LLM 生成候选 schedule 是否能降低搜索成本，并在真实硬件上获得有竞争力的推理性能。
4. 基于真实 latency 的偏好优化是否能进一步提升生成 schedule 的性能倾向。
5. 所提出方法是否能够在不同类型 workload 之间保持泛化能力。
6. 在 Qwen3、Llama3、Qwen-VL 等真实大模型算子形状上，LLM 生成 schedule 与 FlashAttention、CUTLASS、cuBLAS 等高性能算子库相比处于什么性能区间。

因此，实验设计分为三条主线：第一条验证结构化调度表示与模型训练有效性；第二条验证生成 schedule 的结构合法性与端到端性能；第三条验证真实硬件反馈驱动的偏好优化效果。

## 2. 当前实验基础

仓库中已经具备完整的实验流水线：

| 环节 | 脚本 | 作用 |
| --- | --- | --- |
| 数据构建 | `LLM/make_dataset.py` | 将 TVM measure records 转换为 Qwen3 训练数据，支持 `PPT` 切分与 hold-out workload |
| 领域词表扩展 | `LLM/extend_tokenizer.py` | 将 TVM step token 与常用整数加入 tokenizer，并 warm-start embedding |
| 结构监督训练 | `LLM/train_qwen3_clm.py` | 对 `PPT` 后缀进行 causal LM 训练 |
| 训练启动封装 | `LLM/run_train_qwen3_clm.py` | 启动 Stage1/Stage2 训练 |
| 结构合法性评估 | `LLM/eval_struct.py` | 评估 parse/build/fallback/distinct 等结构指标 |
| schedule 生成 | `LLM/gen_state.py` | 将模型接入 `SketchPolicy.gen_states` 生成 TVM State |
| latency 测量 | `LLM/measure_programs.py` | 使用 LocalRunner 补测候选 schedule 的真实 latency |
| 端到端编译评估 | `LLM/tune_relay.py`, `LLM/baseline.py` | 在 Relay 模型上应用 history best 并测量推理延迟 |
| 偏好对构建 | `LLM/build_preference_pairs.py` | 从真实 latency records 构造 `(prompt, chosen, rejected)` 偏好对 |
| DPO/LAPO 训练 | `LLM/train_qwen3_dpo.py`, `LLM/run_train_qwen3_dpo.py` | 进行 SFT、DPO 或 Latency-Aware DPO 训练 |

截至当前梳理，`qwen_4090_gen` 训练集约包含 2,156,881 条样本，验证集约包含 111,627 条样本。Stage1 模型已经完成一轮训练，验证集 token accuracy 约为 0.978，perplexity 约为 1.055。需要注意的是，当前大规模训练数据中的 `r` 字段大多仍为占位 latency，因此 Stage1 主要学习结构合法性，而不是性能偏好。

现有 hold-out 评估目录已经覆盖多个 workload：

| Workload | `0_merge.json` 样本数 | `gen_eval.json` 样本数 | 说明 |
| --- | ---: | ---: | --- |
| `bert_base` | 576 | 278 | Transformer 类 workload |
| `bert_large` | 576 | 288 | 更大规模 Transformer |
| `resnet_50` | 1728 | 864 | CNN 基准网络 |
| `mobilenet_v2` | 2048 | 1024 | 轻量 CNN |
| `mobilenet_v3` | 3328 | 1664 | 轻量 CNN |
| `resnext_50` | 1728 | 864 | 分组卷积网络 |
| `vgg_16` | 1152 | 576 | 经典 CNN |
| `densenet_121` | 4608 | 2304 | 密集连接 CNN |
| `wide_resnet_50` | 1728 | 864 | 宽残差网络 |

这些文件为结构合法性评估、生成消融和真实 latency 测量提供了直接的数据基础。

## 3. 实验环境与统一设置

所有主要实验建议统一在 RTX 4090 平台上完成，目标字符串使用 `nvidia/geforce-rtx-4090` 或等价 CUDA target。为保证结果可复现，所有实验应记录以下信息：

| 项目 | 建议记录内容 |
| --- | --- |
| 硬件 | GPU 型号、显存容量、CUDA 驱动版本 |
| 编译器 | TVM commit 或仓库版本 |
| 模型 | Qwen3-0.6B 原始模型、Stage1/Stage2/DPO checkpoint 路径 |
| tokenizer | 是否使用 TVM 扩词表，扩展 token 数量 |
| 数据集 | 训练集路径、验证集路径、hold-out workload 列表 |
| 生成参数 | `keep_cnt`, `temperature`, `top_p`, `top_k`, `min_gen_tokens`, `gen_token_scale` |
| 测量参数 | `repeat`, `number`, `min_repeat_ms`, `batch_size`, GPU 独占情况 |

所有 latency 测量应尽量在独占 GPU 环境下执行，避免后台进程造成测量抖动。每个 workload 的最终结果建议报告均值、中位数以及 geometric mean speedup。

## 4. 数据划分与评估 Workload

训练数据使用 `make_dataset.py` 中的 hold-out 排除机制，避免评估 workload 泄漏到训练集中。评估 workload 按模型结构划分如下：

| 类别 | Workload |
| --- | --- |
| Transformer | `bert_base`, `bert_large` |
| Decoder-only LLM | `qwen3`, `llama3` 的真实 GEMM、attention、MLP、norm 算子形状 |
| Vision-Language LLM | `qwen_vl`/`qwen2_5_vl` 的 vision encoder、multimodal projector、text decoder |
| 常规 CNN | `resnet_50`, `vgg_16` |
| 轻量 CNN | `mobilenet_v2`, `mobilenet_v3` |
| 复杂连接 CNN | `resnext_50`, `densenet_121`, `wide_resnet_50` |

论文主表建议至少覆盖 `bert_base`、`bert_large`、`resnet_50`、`mobilenet_v2`、`densenet_121` 五类代表性 workload。若时间允许，应报告全部已生成的 hold-out workload。

## 5. 实验一：结构化调度表示与训练有效性

### 5.1 实验目的

该实验验证 TVM 原生 measure record 序列是否适合作为大语言模型的训练输入，并评估 `PPT` 后缀监督、动态 padding 和领域 tokenizer 对训练质量与效率的影响。

### 5.2 对照组

| 组别 | 描述 |
| --- | --- |
| Full-CLM | 不使用 `PPT` mask，整条调度序列均参与 loss |
| PPT-Suffix CLM | 使用 `PPT` 之前 mask，只监督调度后缀 |
| PPT + Dynamic Padding | 当前新版训练方式，训练时动态 padding |
| PPT + Dynamic Padding + Extended Tokenizer | 使用 `extend_tokenizer.py` 后重建数据集并训练 |

### 5.3 评价指标

| 指标 | 含义 |
| --- | --- |
| `eval_loss` | 验证集语言建模损失 |
| `eval_accuracy` | 后缀 token 预测准确率 |
| `perplexity` | 语言建模困惑度 |
| 平均 token 长度 | tokenizer 对 TVM DSL 的编码效率 |
| `train_samples_per_second` | 训练吞吐 |
| 总训练时长 | 训练成本 |
| `parse_valid_rate` | 生成后缀能否被 TVM 解析 |

### 5.4 论文中预期结论

若 PPT 后缀监督在结构合法性指标上优于 Full-CLM，则说明调度草图补全比普通序列复现更贴合自动调度任务。若扩词表组显著降低平均 token 长度并提升训练吞吐，则可证明领域 tokenizer 对编译器 DSL 建模具有必要性。

## 6. 实验二：结构合法性与生成质量评估

### 6.1 实验目的

该实验评估模型生成的 schedule suffix 是否能被 TVM `SketchPolicy.gen_states` 成功解析，并进一步通过 LocalBuilder 构建。

### 6.2 对照组

| 方法 | 描述 |
| --- | --- |
| Base Qwen3-0.6B | 未经过 TVM 数据训练的基础模型 |
| Stage1 | 经过 `PPT` 后缀监督训练的模型 |
| Stage2 | 在 Stage1 基础上继续训练的模型 |
| Stage1 + Extended Tokenizer | 使用 TVM 扩词表训练的模型 |
| Stage1 + 不同采样策略 | 改变 temperature/top-p/keep_cnt 的生成设置 |

### 6.3 指标

| 指标 | 说明 |
| --- | --- |
| `parse_valid_rate` | 生成后缀被 `SketchPolicy.gen_states` 成功解析的比例 |
| `build_valid_rate` | LocalBuilder 构建成功的比例 |
| `fallback_rate` | 全部生成失败而回退到原 sketch 的 workload 比例 |
| `distinct_rate` | 去重后候选 state 数量占比 |
| `avg_parse_per_sketch` | 每个 sketch 平均可解析候选数 |
| `samples_per_sec` | 结构评估吞吐 |

### 6.4 执行建议

先运行不带 build 的快速评估，再对代表性 workload 运行 `--do_build=True`：

```bash
cd /home/qsy/workspace/complier/llm_compiler/LLM

/home/qsy/anaconda3/envs/tlm/bin/python eval_struct.py \
  --model_name_or_path /home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1 \
  --sketch_path /home/qsy/workspace/gen_data/4090_gen_eval_only_bert/0_merge.json \
  --target nvidia/geforce-rtx-4090 \
  --max_workloads 64 \
  --max_states_per_workload 2 \
  --max_new_tokens 512 \
  --output_json /home/qsy/workspace/gen_data/4090_gen_eval_only_bert/eval_struct_stage1.json
```

### 6.5 论文中预期结论

该实验用于证明模型生成的并非普通文本，而是能够重新进入 TVM 编译流程的结构化调度状态。相比 token accuracy，`parse_valid_rate` 和 `build_valid_rate` 更能反映模型在编译器任务中的真实有效性。

## 7. 实验三：采样策略与候选规模消融

### 7.1 实验目的

该实验研究生成候选数量和采样参数对 schedule 合法性、多样性和最佳性能的影响。

### 7.2 消融变量

| 变量 | 建议取值 |
| --- | --- |
| `keep_cnt` | 8, 16, 32, 64 |
| `temperature` | 0.2, 0.6, 1.0 |
| `top_p` | 0.8, 0.95, 1.0 |
| `top_k` | 0, 50 |
| `min_gen_tokens` | 128, 256, 512 |
| `gen_token_scale` | 2.0, 4.0 |
| `allow_repeat` | True, False |

### 7.3 指标

| 指标 | 说明 |
| --- | --- |
| 有效候选数量 | parse/build 成功的 schedule 数 |
| 去重候选数量 | 衡量多样性 |
| invalid/fallback 次数 | 生成稳定性 |
| best-of-K latency | 候选集合中真实测量最优 latency |
| 生成时间 | 生成成本 |

### 7.4 论文中预期结论

该实验用于解释主实验中参数选择的合理性。例如，若 `keep_cnt=32` 在生成成本和 best-of-K latency 之间达到较好平衡，则可将其作为后续端到端性能实验的默认设置。

## 8. 实验四：真实硬件 latency 与端到端推理性能

### 8.1 实验目的

该实验验证 LLM 生成的 schedule 在真实 RTX 4090 硬件上的性能收益，是论文最核心的性能实验。

### 8.2 对照组

| 方法 | 描述 |
| --- | --- |
| Relay Default | 不使用 AutoScheduler history best |
| Ansor-64 | Ansor 搜索 64 trials |
| Ansor-256 | Ansor 搜索 256 trials |
| Ansor-1024 | Ansor 搜索 1024 trials |
| LLM-only Top-K | 仅测量 LLM 生成的 K 个候选 |
| LLM Best-of-K | 从 LLM 生成候选中选择真实 latency 最小者 |
| LLM + Ansor Same Budget | LLM 生成加少量 Ansor 搜索，总时间与 Ansor 对齐 |

### 8.3 指标

| 指标 | 说明 |
| --- | --- |
| kernel best latency | `measure_programs.py` 输出的最优算子 latency |
| end-to-end inference latency | `tune_relay.py` 输出的模型推理延迟 |
| speedup over Relay | 相对默认 Relay 编译的加速比 |
| speedup over Ansor | 相对 Ansor baseline 的加速比 |
| tuning/search time | 搜索或生成候选所需时间 |
| trials-to-target | 达到目标 latency 所需测量次数 |

### 8.4 执行流程

1. 使用 `gen_state.py` 为每个 hold-out workload 生成候选 schedule。
2. 使用 `measure_programs.py` 对 `gen_eval.json` 进行真实 latency 测量，输出 `measured.json`。
3. 使用 `tune_relay.py` 读取 `measured.json`，编译 Relay 模型并测量端到端延迟。
4. 使用 `baseline.py` 跑 Ansor 不同 trials 的对照实验。
5. 汇总每个 workload 的 latency 与 speedup，并计算 geometric mean。

示例：

```bash
CUDA_VISIBLE_DEVICES=0 /home/qsy/anaconda3/envs/tlm/bin/python measure_programs.py \
  --target nvidia/geforce-rtx-4090 \
  --to-measure-path /home/qsy/workspace/gen_data/4090_gen_eval_only_bert_large/gen_eval.json \
  --measured-path /home/qsy/workspace/gen_data/4090_gen_eval_only_bert_large/measured.json \
  --batch-size 64 \
  --resume
```

### 8.5 论文中预期结论

若 LLM Best-of-K 能在更少候选或更短搜索时间下达到接近 Ansor-1024 的性能，或者在部分 workload 上超过 Ansor，则可证明 LLM 作为调度候选生成器具有实际价值。

## 9. 实验五：大模型 Transformer/VLM latency 与算子库对比

### 9.1 实验目的

该实验面向 Qwen3、Llama3、Qwen-VL 等真实大模型推理场景，验证 LLM 辅助 TVM 自动调度在 decoder-only LLM 和 vision-language model 上的性能边界。与 BERT 类 Relay workload 不同，大模型实验需要同时报告算子级、单层 block 级和端到端 prefill/decode 级结果，并将 FlashAttention、CUTLASS、cuBLAS 作为高性能算子库上界。

需要注意的是，FlashAttention、CUTLASS、cuBLAS 并不是同一层次的基线。FlashAttention 主要对应 attention kernel，CUTLASS/cuBLAS 主要对应 GEMM、batch matmul、dense 及部分 fused dense pattern。因此论文中应按算子类别进行公平对比，而不是简单把所有库放在同一列中比较整网 latency。

### 9.2 模型与输入设置

主实验建议优先选择单张 RTX 4090 可落地的模型规模：

| 类别 | 主实验模型 | 调试或补充模型 | 输入设置 |
| --- | --- | --- | --- |
| Decoder-only LLM | `Qwen3-8B`, `Llama3-8B` | `Qwen3-0.6B` | batch `{1, 4}`，seq `{128, 512, 2048}` |
| Vision-Language LLM | `Qwen2.5-VL-7B-Instruct` 或本地已有 `Qwen-VL` | 3B 级模型 | 1 张图像 + text seq `{128, 512}` |
| 单层压力测试 | Qwen3/Llama3 单个 decoder block | 长上下文配置 | seq `{4096, 8192}`，显存允许时启用 |

70B/72B 级别模型建议仅作为多 GPU 附录，不作为主表默认配置。所有端到端实验使用 greedy decoding，固定 prompt、固定 image 输入，默认排除 tokenizer、图片读取和图像预处理时间；若需要端到端应用视角，可单独报告包含 preprocessing 的 latency。

### 9.2.1 本地 Qwen/Qwen-VL Workload 落地方案

本仓库当前的 `dump_network_info.py` 与 `dump_programs.py` 不能直接把 HuggingFace 权重目录作为 TVM workload。对于本地已有模型，应先导出固定 shape 的 ONNX 或 Relay 子图，再通过 `dump_network_info.py --onnx-model` 抽取 AutoScheduler tasks，最后用 `dump_programs.py` 生成待测 schedule programs。

本地优先纳入如下两个模型：

| 模型 | 本地路径 | 建议实验定位 | 直接 dump 可行性 |
| --- | --- | --- | --- |
| Qwen3-0.6B | `/home/qsy/huggingface/model/Qwen3-0.6B` | text-only decoder-only LLM 主力调试 workload | 不能直接 dump HF 目录；先导出 ONNX |
| Qwen3-VL-2B-Instruct | `/home/qsy/huggingface/model/Qwen3-VL-2B-Instruct` | VLM 子图实验，重点拆 vision/text/projector | 不建议整模型直接导出；优先拆子模块 |

#### Qwen3-0.6B Workload 设计

Qwen3-0.6B 作为主实验中的本地 LLM workload，建议按从易到难分三层执行：

| 层级 | Workload | 固定输入 | 目的 |
| --- | --- | --- | --- |
| L0 | ONNX 全 forward，小 seq 调试 | batch `1`，seq `16/32` | 验证 HF -> ONNX -> Relay -> tasks 链路 |
| L1 | ONNX 全 forward，prefill shape | batch `1`，seq `128/512` | 生成真实 Qwen3 算子 workload |
| L2 | 单层 decoder block 或算子级子图 | hidden `1024`，heads `16`，kv heads `8`，head dim `128` | 与 Ansor/cuBLAS/CUTLASS/FlashAttention 做分层对比 |

推荐 shape 矩阵：

| 场景 | batch | seq length | dtype | 说明 |
| --- | ---: | ---: | --- | --- |
| smoke test | 1 | 16 | fp32 或 fp16 | 最先跑通导出和 TVM import |
| short prefill | 1 | 128 | fp16/bf16 优先，必要时 fp32 | 主调试配置 |
| medium prefill | 1 | 512 | fp16/bf16 | 论文补充配置 |
| small batch | 4 | 128 | fp16/bf16 | 验证 batch 泛化 |
| decode-like | 1 | 1，带 KV cache 子图 | fp16/bf16 | 需要单独导出 decode 子图，不作为首批必做 |

首批实验可以只导出 `use_cache=False` 的 prefill forward，输入保留 `input_ids` 和 `attention_mask`。若整网 ONNX 导入 TVM 失败，应退回到单层 decoder block 或算子级子图，优先覆盖 QKV projection、attention score/value、gate/up/down MLP、RMSNorm、RoPE 等真实 shape。

Qwen3-0.6B 的建议执行命令如下。先导出 ONNX，推荐后续新增一个 `tools/export_qwen3_to_onnx.py` 封装以下逻辑：

```bash
python tools/export_qwen3_to_onnx.py \
  --model-dir /home/qsy/huggingface/model/Qwen3-0.6B \
  --seq-len 128 \
  --output /home/qsy/workspace/dataset/onnx/qwen3_0_6b_seq128.onnx \
  --dtype fp32 \
  --disable-cache
```

在脚本尚未补齐前，可直接用 `torch.onnx.export` 包一层只返回 `logits` 的 wrapper；导出时固定 `input_ids:[1,128]` 和 `attention_mask:[1,128]`，并设置 `model.config.use_cache=False`。当前环境还需要确保已安装 `onnx`，否则导出和 `dump_network_info.py --onnx-model` 都会失败。

再抽取任务：

```bash
python gen/dump_network_info.py \
  --target "nvidia/geforce-rtx-4090" \
  --hardware-name qwen3_0_6b \
  --network-info-dir /home/qsy/workspace/dataset/network_info/qwen3_0_6b_seq128 \
  --only-workloads \
  --onnx-model /home/qsy/workspace/dataset/onnx/qwen3_0_6b_seq128.onnx \
  --model-name qwen3_0_6b_seq128 \
  --input-shapes 'input_ids:[1,128] attention_mask:[1,128]' \
  --input-dtype int64
```

最后生成待测 programs：

```bash
python gen/dump_programs.py \
  --target "nvidia/geforce-rtx-4090" \
  --hardware-name qwen3_0_6b \
  --network-info-dir /home/qsy/workspace/dataset/network_info/qwen3_0_6b_seq128 \
  --output-dir /home/qsy/workspace/dataset/to_measure_programs/qwen3_0_6b_seq128 \
  --size 1000
```

#### Qwen3-VL-2B-Instruct Workload 设计

Qwen3-VL-2B-Instruct 不建议首轮直接导出整模型。VLM 同时包含 vision encoder、multimodal projector、text decoder、图像 token/grid 元信息和多模态 RoPE，整图 ONNX/Relay 导入失败概率高，且难以解释性能来源。建议拆成三个 workload 族：

| Workload 族 | 子图范围 | 输入设置 | 主要指标 |
| --- | --- | --- | --- |
| Vision encoder | patch embedding、vision transformer blocks、vision MLP | 1 张图，固定分辨率或固定 patch token 数 | vision latency、GEMM/MLP latency |
| Multimodal projector | vision hidden -> text hidden 投影 | vision token `{256, 576, 1024}`，hidden `1024 -> 2048` | projector latency、GEMM 性能 |
| Text decoder | Qwen3-VL text decoder block | batch `1`，text seq `{128, 512}`，hidden `2048` | decoder block/prefill latency |

建议固定图像配置从小规模开始：

| 场景 | 图像/patch 设置 | text seq | 说明 |
| --- | --- | ---: | --- |
| VL smoke test | 固定 224x224 或最小 patch token | 32 | 验证导出链路 |
| single image short text | 1 张图，patch token 约 256/576 | 128 | 主调试配置 |
| single image medium text | 1 张图，patch token 约 576/1024 | 512 | 论文补充配置 |

VLM 的首批可交付结果应优先选择子图级而不是整网端到端：

1. vision encoder 中的 `conv/patch embedding`、vision MLP、attention projection GEMM；
2. projector 的 dense 或 MLP；
3. text decoder 的 QKV projection、attention、MLP、RMSNorm/RoPE。

若需要端到端 VLM latency，建议在子图抽取和算子测量稳定后再做，并将 preprocessing、image resize/tokenizer 时间与 TVM/TLM 编译后推理时间分开报告。

#### 结果记录与失败兜底

Qwen/Qwen-VL workload 需要额外记录导出和导入成功率：

| 指标 | 说明 |
| --- | --- |
| ONNX export status | HF/PyTorch 导出是否成功，opset 版本 |
| Relay import status | `relay.frontend.from_onnx` 是否成功 |
| extracted task count | AutoScheduler 抽取出的 task 数量 |
| dump program count | 每个 task 实际生成的 state 数量 |
| unsupported op list | ONNX/Relay 不支持的算子 |
| fallback granularity | 整网失败后退回 block/subgraph/operator 的粒度 |

如果整网导出或 Relay import 失败，按以下顺序降级：

1. 缩小 `seq_len`，先跑 `16/32`；
2. 关闭 `use_cache`，只导出 prefill forward；
3. 将 dtype 从 bf16/fp16 临时改为 fp32，先验证结构链路；
4. 从整网退回单层 decoder block；
5. 从 block 退回真实 shape 的 GEMM/attention/MLP/norm 算子级 workload。

### 9.3 对照组

| 方法 | 对比范围 | 说明 |
| --- | --- | --- |
| TVM Default / TOPI | 算子级、block 级、端到端 | 不使用自动调度 |
| Ansor-256 / Ansor-1024 | 算子级、block 级 | 传统搜索基线 |
| LLM Best-of-K | 算子级、block 级、端到端 | `K` 建议取 16、32、64 |
| LLM + Ansor Warm-start | 算子级、block 级 | 验证降低搜索成本 |
| TVM + cuBLAS BYOC | GEMM、batch matmul、dense | 仓库已有 `relay.op.contrib.cublas` pattern |
| TVM + CUTLASS BYOC | dense、dense+bias+activation、batch matmul | 仓库已有 `relay.op.contrib.cutlass` pattern |
| PyTorch / Transformers SDPA | attention 与端到端 | 框架基线 |
| PyTorch + FlashAttention | attention 与端到端 | attention kernel 上界 |

### 9.4 实验 A：Operator-level Microbenchmark

从 Qwen3、Llama3、Qwen-VL 中抽取真实 shape，构造独立算子 microbenchmark：

| 算子类别 | 典型来源 | 主要对比对象 |
| --- | --- | --- |
| QKV projection GEMM | attention 输入投影 | LLM/Ansor/cuBLAS/CUTLASS |
| Attention score + value | causal attention | TVM 分解实现/FlashAttention |
| MLP GEMM | gate/up/down projection | LLM/Ansor/cuBLAS/CUTLASS |
| dense+bias+GELU/SwiGLU | FFN 激活融合 | LLM/CUTLASS |
| RMSNorm/LayerNorm/RoPE | decoder block 辅助算子 | TVM/LLM/PyTorch kernel |
| vision patch embedding/vision MLP | Qwen-VL vision encoder | LLM/CUTLASS/cuBLAS |

核心指标包括 kernel latency、achieved TFLOPS、显存带宽、build 成功率、compile/tuning time、数值误差。对于同一算子 shape，应记录 dtype、layout、transpose、batch、seq length、hidden size、head 数和 head dim。

### 9.5 实验 B：单层 Transformer Block latency

单层 block 能避免 tokenizer、runtime scheduler、KV cache 管理等端到端因素干扰，适合解释性能来源：

| 配置 | Attention | GEMM/MLP | 目的 |
| --- | --- | --- | --- |
| TVM/TLM only | TVM 分解 attention | LLM 生成 schedule | 验证纯 TVM 路径 |
| TLM + cuBLAS | TVM 分解 attention | cuBLAS GEMM | 分析 GEMM 库收益 |
| TLM + CUTLASS | TVM 分解 attention | CUTLASS fused GEMM | 分析融合 GEMM 收益 |
| FlashAttention + cuBLAS/CUTLASS | FlashAttention | cuBLAS/CUTLASS | 工业库性能上界 |

该实验应报告 block 总 latency、attention 占比、MLP 占比、norm/RoPE/elementwise 占比，以及各配置相对 TVM Default 和 Ansor 的 speedup。

### 9.6 实验 C：端到端 prefill 与 decode latency

大模型端到端实验需要拆分 prefill 和 decode：

| 阶段 | 指标 |
| --- | --- |
| Prefill | total latency、tokens/s、attention latency 占比、GEMM latency 占比 |
| Decode | ms/token、tokens/s、KV cache 显存、p50/p90 latency |
| Qwen-VL vision path | vision encoder latency、projector latency、text decoder latency |

默认生成长度设为 128 tokens。若比较 PyTorch + FlashAttention，需要保证 dtype、batch、seq length、KV cache 设置、是否使用 causal mask、是否启用 torch.compile 等条件保持一致。

### 9.7 实验 D：搜索成本与性能边界

对每个模型选择 20 到 50 个高权重任务，比较 LLM 生成和传统搜索的成本：

| 变量 | 建议取值 |
| --- | --- |
| `keep_cnt` / Best-of-K | 16, 32, 64 |
| Ansor trials | 256, 1024 |
| same-time budget | 10min, 30min, 1h |
| 长度泛化 | train seq `{128, 512}`，test seq `{2048}` |

指标包括 time-to-best、trials-to-target、fixed-budget best latency、compile failure rate、parse/build valid rate。论文结论应避免宣称全面超过 FlashAttention/cuBLAS，而应说明 LLM 生成 schedule 在 shape-specific、非标准融合或库覆盖不足场景中的价值。

### 9.8 前置工程条件

1. 启用或补全 LLaMA workload 构建逻辑；当前 `meta/dataset_collect_models.py` 和 `meta/meta_common.py` 中的 `llama` 相关配置仍处于注释状态。
2. 新增 Qwen3/Qwen-VL 的模型导出或子图抽取脚本，优先支持单层 decoder block 和真实 GEMM/attention shape 抽取。
3. 确认 TVM 编译时启用 `USE_CUBLAS` 和 `USE_CUTLASS`，并记录 CUTLASS、CUDA、cuBLAS 版本。
4. FlashAttention 建议通过独立 PyTorch/Transformers benchmark 脚本测量，作为 attention 专用库上界，不强行接入 TVM Relay 主流程。
5. 所有库对比必须记录算子覆盖率，即每个模型中有多少算子实际由 TVM/TLM、cuBLAS、CUTLASS 或 FlashAttention 执行。

### 9.9 论文中预期结论

该实验用于给出方法的性能边界：若 LLM 生成 schedule 在部分 GEMM、MLP 或 elementwise-heavy 的真实 LLM shape 上接近 CUTLASS/cuBLAS，同时搜索成本低于 Ansor，则可证明其在大模型编译调优中具有实用价值；若 attention 部分仍明显落后于 FlashAttention，则应将其作为专用 kernel 的必要性分析，而不是方法失败。

## 10. 实验六：跨 workload 泛化能力

### 10.1 实验目的

该实验验证模型是否学习到通用调度规律，而不是记忆单一模型结构的 schedule 模板。

### 10.2 实验设置

将评估 workload 按结构类别分组，并分别报告结构指标和性能指标：

| 类别 | 指标 |
| --- | --- |
| Transformer | `bert_base`, `bert_large` 的 parse/build/latency |
| Decoder-only LLM | `qwen3`, `llama3` 算子级和 block 级 latency |
| Vision-Language LLM | `qwen_vl`/`qwen2_5_vl` vision/text 子图 latency |
| 常规 CNN | `resnet_50`, `vgg_16` 的 parse/build/latency |
| 轻量 CNN | `mobilenet_v2`, `mobilenet_v3` 的 parse/build/latency |
| 复杂连接 CNN | `resnext_50`, `densenet_121`, `wide_resnet_50` 的 parse/build/latency |

### 10.3 指标

| 指标 | 说明 |
| --- | --- |
| 类别内平均 `parse_valid_rate` | 是否能跨结构生成合法 schedule |
| 类别内平均 `build_valid_rate` | 是否能构建 |
| 类别内 geomean speedup | 是否有性能收益 |
| workload 间方差 | 方法稳定性 |

### 10.4 论文中预期结论

若模型在 Transformer、真实 LLM/VLM 与 CNN workload 上均保持较高结构合法性，则说明该方法学习到的是 TVM schedule DSL 与硬件映射之间的通用关系，而非单一 workload 的模板。

## 11. 实验七：Latency-Aware Preference Optimization

### 11.1 实验目的

Stage1 主要学习结构合法性，缺少真实性能信号。本实验通过真实 latency 构造偏好对，验证 DPO/LAPO 是否能够提升生成 schedule 的性能倾向。

### 11.2 前置条件

需要使用 `measure_programs.py` 补齐 `measure_records/4090` 中更多真实 latency。当前已有有效 latency 的记录数量有限，可以先进行小规模 proof-of-concept；若作为论文主实验，建议每个训练 workload 至少补测几十到上百条候选记录。

### 11.3 对照组

| 方法 | 描述 |
| --- | --- |
| Stage1 | 只学习结构合法性 |
| Stage1 + SFT-chosen | 只在低 latency chosen suffix 上继续监督 |
| Stage1 + DPO | 标准 Direct Preference Optimization |
| Stage1 + LAPO | 使用 `latency_weight` 加权的 DPO |
| LAPO weight ablation | 比较 `uniform`, `log_gap`, `linear_gap`, `clipped_linear` |

### 11.4 指标

| 指标 | 说明 |
| --- | --- |
| `reward/accuracy` | chosen 隐式 reward 是否高于 rejected |
| `reward/margin_mean` | 偏好区分度 |
| `parse_valid_rate` | 偏好训练后结构合法性是否保持 |
| `build_valid_rate` | 构建合法性是否保持 |
| best-of-K latency | 真实硬件上候选集合最优 latency |
| geomean speedup | across workloads 的整体性能收益 |

### 11.5 执行流程

1. 补测 latency：

```bash
CUDA_VISIBLE_DEVICES=0 /home/qsy/anaconda3/envs/tlm/bin/python measure_programs.py \
  --target "cuda -model=4090" \
  --input-dir /home/qsy/workspace/dataset/measure_records/4090 \
  --batch-size 128 \
  --resume \
  --drop-hold-out
```

2. 构造偏好对：

```bash
/home/qsy/anaconda3/envs/tlm/bin/python build_preference_pairs.py \
  --target "cuda -model=4090" \
  --dataset_path /home/qsy/workspace/dataset/measure_records/4090 \
  --save_path /home/qsy/workspace/gen_data/4090_prefs \
  --min_latency_gap_ratio 1.5 \
  --max_pairs_per_group 4 \
  --max_pairs_per_workload 512 \
  --weight_strategy log_gap \
  --drop_hold_out_workloads True
```

3. 训练 DPO/LAPO：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
PREFERENCE_DATASET=/home/qsy/workspace/gen_data/4090_prefs \
POLICY_MODEL_PATH=/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1 \
REF_MODEL_PATH=/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1 \
TRAINING_MODE=lapo \
/home/qsy/anaconda3/envs/tlm/bin/python run_train_qwen3_dpo.py
```

### 11.6 论文中预期结论

若 LAPO 在不降低结构合法性的前提下降低 best-of-K latency，则说明真实硬件反馈能够通过偏好学习有效注入调度生成模型。该实验可作为论文性能优化主创新点。

## 12. 实验八：LLM 与 Ansor 的混合自动调优

### 12.1 实验目的

该实验验证 LLM 生成候选能否作为 Ansor 的 warm-start，从而减少传统搜索的冷启动成本。

### 12.2 对照组

| 方法 | 描述 |
| --- | --- |
| Ansor 原始搜索 | 使用默认初始种群与搜索策略 |
| LLM-only | 只测量 LLM 生成候选 |
| LLM Warm-start Ansor | 将 LLM 生成候选作为初始 history 或候选池 |
| LLM + Ansor Same Budget | LLM 生成加 Ansor 搜索，总时间与 Ansor 对齐 |

### 12.3 指标

| 指标 | 说明 |
| --- | --- |
| fixed-budget best latency | 固定 10min/30min/1h 下的最优 latency |
| trials-to-target | 达到 Ansor 最优 90% 所需测量次数 |
| convergence curve | 随 trials/time 增加的 best latency |
| final end-to-end latency | 最终 Relay 推理性能 |

### 12.4 论文中预期结论

若 LLM warm-start 能在早期快速达到较优 latency，则可说明 LLM 适合作为传统 evolutionary search 的候选先验，而不是必须完全替代 Ansor。

## 13. 实验结果组织方式

论文中建议按如下表格组织结果。

### 13.1 训练效率表

| 方法 | 平均 token 长度 | train samples/s | train runtime | eval loss | eval accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full-CLM | | | | | |
| PPT-Suffix | | | | | |
| PPT + Dynamic Padding | | | | | |
| PPT + Extended Tokenizer | | | | | |

### 13.2 结构合法性表

| 方法 | parse valid | build valid | fallback | distinct | avg parse/sketch |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base Qwen3 | | | | | |
| Stage1 | | | | | |
| Stage2 | | | | | |
| DPO/LAPO | | | | | |

### 13.3 端到端性能表

| Workload | Relay default | Ansor-256 | Ansor-1024 | LLM Best-of-K | Speedup vs Relay | Speedup vs Ansor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bert_base` | | | | | | |
| `bert_large` | | | | | | |
| `resnet_50` | | | | | | |
| `mobilenet_v2` | | | | | | |
| `densenet_121` | | | | | | |
| Geomean | | | | | | |

### 13.4 偏好优化表

| 方法 | reward acc | parse valid | build valid | best latency | geomean speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage1 | | | | | |
| SFT-chosen | | | | | |
| DPO | | | | | |
| LAPO | | | | | |

### 13.5 大模型算子库对比表

| Model | Kernel group | TVM Default | Ansor-1024 | LLM Best-of-K | cuBLAS | CUTLASS | FlashAttention |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3 | QKV GEMM | | | | | | N/A |
| Qwen3 | Attention | | | | N/A | N/A | |
| Llama3 | MLP GEMM | | | | | | N/A |
| Qwen-VL | Vision MLP | | | | | | N/A |
| Qwen3-0.6B | Prefill GEMM/MLP, seq128 | | | | | | N/A |
| Qwen3-0.6B | Attention subgraph, seq128 | | | | N/A | N/A | |
| Qwen3-VL-2B-Instruct | Vision encoder MLP/projection | | | | | | N/A |
| Qwen3-VL-2B-Instruct | Projector dense | | | | | | N/A |

### 13.6 大模型端到端 latency 表

| Model | Stage | TVM/TLM | TLM + cuBLAS | TLM + CUTLASS | PyTorch SDPA | PyTorch + FlashAttention | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3 | Prefill | | | | | | |
| Qwen3 | Decode ms/token | | | | | | |
| Llama3 | Prefill | | | | | | |
| Qwen-VL | Vision encoder | | | | | | |
| Qwen-VL | Text decode | | | | | | |
| Qwen3-0.6B | Prefill seq128 | | | | | | |
| Qwen3-VL-2B-Instruct | Vision subgraph | | | | | | |
| Qwen3-VL-2B-Instruct | Projector + text decoder block | | | | | | |

## 14. 执行优先级

建议按以下顺序执行：

1. **优先级 P0：结构合法性实验**。先跑 `eval_struct.py`，快速获得 parse/build/fallback/distinct 指标。
2. **优先级 P0：生成策略消融**。对 `keep_cnt`、`temperature`、`top_p` 做小规模消融，确定主实验默认参数。
3. **优先级 P0：真实 latency 与端到端性能实验**。补测 `gen_eval.json`，使用 `tune_relay.py` 输出最终推理延迟。
4. **优先级 P0：大模型 operator-level microbenchmark**。先抽取 Qwen3/Llama3 的真实 GEMM、attention、MLP shape，对比 LLM/Ansor/cuBLAS/CUTLASS/FlashAttention。
5. **优先级 P1：大模型单层 block 与 prefill/decode 实验**。在算子级结果稳定后，扩展到 Qwen3、Llama3、Qwen-VL 的 block 级和端到端 latency。
6. **优先级 P1：训练效率与 tokenizer 消融**。若时间允许，完成扩词表重建数据与训练，对比训练成本和结构合法性。
7. **优先级 P1：DPO/LAPO 偏好优化**。在补齐更多真实 latency 后执行，作为论文性能增强实验。
8. **优先级 P2：LLM + Ansor 混合调优**。若论文时间充足，将其作为系统扩展实验。

## 15. 建议的论文实验叙事

实验章节可按如下逻辑展开：

首先，通过训练效率与结构合法性实验说明 TVM 原生调度状态序列能够被大语言模型有效学习，且 `PPT` 后缀监督使模型聚焦于真正的调度决策。其次，通过多 workload 的 parse/build 结果证明模型生成结果能够重新进入 TVM 编译流程，具备可回放、可构建的系统属性。然后，通过真实 latency 和端到端推理实验验证 LLM 生成候选在较低搜索成本下能够获得有竞争力的性能。进一步，通过 Qwen3、Llama3、Qwen-VL 与 FlashAttention、CUTLASS、cuBLAS 的分层对比，说明方法在真实大模型算子形状上的性能边界和适用范围。最后，通过 DPO/LAPO 实验证明真实硬件反馈可以进一步转化为偏好学习信号，使模型从“生成合法 schedule”升级为“生成高性能 schedule”。

该实验设计能够支撑硕士论文或学术论文中的方法有效性、系统可用性和性能收益三类核心论点。
