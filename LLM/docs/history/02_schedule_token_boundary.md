# 第二个问题：统一调度序列的 token 边界

修改日期：2026-10-08。

本文对应优化计划第 4 节的训练条件与生成 prompt 不一致问题。修复目标是让模型在训练时看到的第一个监督 token，和生成时接收的 prompt 处于同一个 token 前缀之后。

## 1. 已确认的问题

实际扩词表 Qwen tokenizer 的 `PPT` id 为 151669，空格可能是独立 id 220。旧实现有三处协议不一致：

- CLM 构建器从 `PPT`、`PPT ` 和带空格的组合中选择最长匹配，把 `PPT` 后的空格也写入 `ppt_end`；训练 mask 因而把该空格当成条件的一部分。
- 生成入口只编码以 `PPT` 结束的文本，训练条件和生成条件的 input ids 不相同。
- DPO/LAPO 分别编码 prompt 和 chosen/rejected，再直接拼接 id；实际联合文本中的分隔空格没有进入拼接结果。对现有 884 条训练 chosen 分支，旧的独立编码与联合文本编码全部存在差异。

这不是简单的词面空格问题。对 BPE tokenizer，独立编码还可能改变空格相关 token，导致模型学习到一个生成时不会出现的条件边界。

## 2. 采用的统一协议

协议版本为 `ppt_boundary_v2`：

1. 所有入口显式使用 `add_special_tokens=False`。
2. `PPT` marker 的结束位置只包含 marker 本身；前导空格可作为匹配上下文，尾随空格不能进入 `ppt_end`。
3. prompt 和 suffix 按完整文本 `prompt + " " + suffix` 编码。分隔空格属于 suffix，因此 CLM 和 DPO 都会在该位置开始计算监督损失。
4. 对联合编码结果做前缀断言：联合文本的前 `len(encode(prompt))` 个 id 必须等于生成时的 prompt id。断言失败时整条偏好对被过滤，禁止静默截断或拼接。
5. EOS 仍由 DPO 的 `append_eos_to_suffix` 显式控制；它追加到 suffix，不改变 prompt 边界。生成和结构评估继续使用现有 `trim_last_input_token` 兼容选项，默认值为 false。

## 3. 代码修改

- [tokenization_protocol.py](../../core/tokenization_protocol.py)：新增共享 marker 定位、联合 prompt/suffix 编码和生成 prompt 编码函数。
- [make_dataset_utils.py](../../data/make_dataset_utils.py)：CLM 使用共享 marker 定位；每条样本写入 `ppt_boundary_version`。训练完整性检查拒绝缺少或不支持该版本的数据。
- [train_qwen3_clm.py](../../train/train_qwen3_clm.py)：预计算 mask 和 legacy runtime mask 都使用同一 marker 定位规则。旧的、没有版本字段的 Arrow 不能在默认模式下绕过检查。
- [train_qwen3_dpo.py](../../train/train_qwen3_dpo.py)：chosen/rejected 分别按完整 prompt+suffix 文本编码；不再左截断 prompt 或右截断 suffix。超出上限、无法保持前缀或两分支 prompt 不一致时，整条 pair 被过滤，并记录过滤数量。
- [gen_state.py](../../generation/gen_state.py)、[eval_struct.py](../../validation/eval_struct.py)：共用显式无 special token 的生成 prompt 编码，保证结构评估和正式生成使用同一输入协议。

## 4. 数据重建和训练影响

第二个修复改变了 `ppt_end` 的语义，现有第一轮重建的 Arrow 也需要重新从 raw JSON 构建；不能只更新 Trainer，因为旧列已经包含了尾随空格边界。建议使用新的目录，例如：

```bash
/home/qsy/anaconda3/envs/tlm/bin/python LLM/__main__.py rebuild-dataset \
  --train-file /home/qsy/workspace/gen_data/4090_gen_qwen_v2_ext/train_raw.json \
  --validation-file /home/qsy/workspace/gen_data/4090_gen_qwen_v2_ext/validation_raw.json \
  --tokenizer-path /home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1-v2-ext-b2 \
  --output-dir /home/qsy/workspace/gen_data/4090_gen_qwen_v3_boundary \
  --max-length 2048 \
  --overlength-policy error
```

偏好数据不需要改写文本；重新运行 DPO/LAPO 数据加载即可按新协议编码。当前默认 `max_length=1024` 下，现有 884 条训练 pair 中有 33 条会因完整序列超过上限被整对过滤。这些 pair 不能通过右截断继续使用；应在确认显存后提高 DPO `max_length`，再重新统计保留数量。

## 5. 验证结果

- `tests/python/unittest/test_llm_schedule_dataset.py`：17 项通过，包含尾随空格不进入 `ppt_end`、联合编码前缀、DPO 整对过滤和完整序列回归。
- 使用真实 Qwen 扩词表 tokenizer 检查 884 条偏好训练 pair：chosen/rejected 的 prompt id 一致，联合编码的 prompt 前缀与生成 prompt 完全一致；33 条只因既有长度上限被整对过滤。
- 使用真实 CLM raw 样本检查：`PPT` 的 `ppt_end` 从旧的 193 改为 marker 的 192，id 220 的空格成为首个监督 token。
- 生成和结构评估入口通过 Python 语法检查；没有启动正式 GPU 训练或宣称性能提升。

## 6. 使用边界

该协议只负责 token 条件和监督边界一致，不保证生成的 suffix 能通过 TVM 解析、构建或测量。后续仍需分别处理生成约束、状态恢复、编译失败分类和 LAPO 的性能监督质量。
